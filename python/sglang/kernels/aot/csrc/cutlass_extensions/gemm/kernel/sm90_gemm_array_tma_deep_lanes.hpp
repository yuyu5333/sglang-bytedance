#pragma once

#include "cutlass_extensions/gemm/kernel/sm90_gemm_array_tma_warpspecialized_pingpong_precomputed.hpp"

namespace cutlass::gemm::kernel {

template <class Problem, class Mainloop, class Epilogue, class Scheduler>
class DeepLanePersistentGemm
    : public GemmUniversalPrecomputedScheduler<Problem, Mainloop, Epilogue, Scheduler> {
  using Base = GemmUniversalPrecomputedScheduler<Problem, Mainloop, Epilogue, Scheduler>;

 public:
  using Params = typename Base::Params;
  using TiledMma = typename Base::TiledMma;
  using TileShape = typename Base::TileShape;
  using ClusterShape = typename Base::ClusterShape;
  using MainPipeline = typename Mainloop::MainloopPipeline;
  using EpiLoadPipeline = typename Epilogue::LoadPipeline;
  using EpiStorePipeline = typename Epilogue::StorePipeline;
  using EpiOrder = cutlass::OrderedSequenceBarrier<1, 2>;
  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, cute::_1> {
      typename Mainloop::TensorStorage mainloop[2];
      typename Epilogue::TensorStorage epilogue;
    } tensors;
    struct PipelineStorage : cute::aligned_struct<16, cute::_1> {
      typename Mainloop::PipelineStorage mainloop[2];
      typename Epilogue::PipelineStorage epi_load;
      typename EpiOrder::SharedStorage epi_order;
    } pipelines;
    struct TensorMaps : cute::aligned_struct<128, cute::_1> {
      typename Mainloop::TensorMapStorage mainloop[2];
      typename Epilogue::TensorMapStorage epilogue;
    } tensormaps;
  };
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    using namespace cute;
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    static_assert(size(TiledMma{}) == 128 && size(ClusterShape{}) == 1);
    static_assert(Mainloop::DispatchPolicy::Stages >= 3);
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    int const tid = threadIdx.x;
    int const warp = canonical_warp_idx_sync();
    int const wg = tid / 128;
    int const lane = wg == 0 ? warp % 2 : wg - 1;
    bool const producer = wg == 0;
    typename MainPipeline::Params mp;
    mp.role = producer ? MainPipeline::ThreadCategory::Producer : MainPipeline::ThreadCategory::Consumer;
    mp.is_leader = producer && tid % 32 == 0;
    mp.num_consumers = 128;
    mp.num_producers = Mainloop::NumProducerThreadEvents;
    mp.initializing_warp = lane;
    mp.transaction_bytes = params.mainloop.tma_transaction_bytes;
    MainPipeline pipeline(storage.pipelines.mainloop[lane], mp, ClusterShape{});
    typename EpiLoadPipeline::Params ep;
    ep.role = EpiLoadPipeline::ThreadCategory::Consumer;
    ep.dst_blockid = 0;
    ep.producer_arv_count = 32;
    ep.consumer_arv_count = 128;
    if constexpr (Epilogue::RequiresTransactionBytes) {
      ep.transaction_bytes = params.epilogue.tma_transaction_bytes;
    }
    EpiLoadPipeline epi_load(storage.pipelines.epi_load, ep);
    typename EpiStorePipeline::Params sp;
    sp.always_wait = true;
    EpiStorePipeline epi_store(sp);
    typename EpiOrder::Params op;
    op.group_id = lane;
    op.group_size = 128;
    EpiOrder epi_order(storage.pipelines.epi_order, op);
    __syncthreads();
    if (producer) {
      cutlass::arch::warpgroup_reg_dealloc<40>();
      if (warp >= 2) {
        return;
      }
    } else {
      cutlass::arch::warpgroup_reg_alloc<232>();
    }
    Mainloop mainloop;
    Epilogue epilogue(params.epilogue, storage.tensors.epilogue);
    if (epilogue.is_producer_load_needed()) {
      return;
    }
    Scheduler scheduler{params.scheduler};
    auto work = scheduler.initial_work_tile_info(ClusterShape{});
    if (lane == 1 && work.is_valid()) {
      work = get<0>(scheduler.fetch_next_work(work));
    }
    if (!work.is_valid()) {
      return;
    }
    auto problem_shape = append<4>(params.problem_shape.get_problem_shape(work.L_idx), 1);
    auto inputs = mainloop.load_init(problem_shape, params.mainloop);
    auto gA = get<0>(inputs);
    auto gB = get<1>(inputs);
    TiledMma mma;
    typename Mainloop::PipelineState consumer_state;
    auto producer_state = cutlass::make_producer_start_state<MainPipeline>();
    typename Epilogue::LoadPipelineState epi_read;
    auto epi_write = cutlass::make_producer_start_state<EpiStorePipeline>();
    constexpr int c_steps = Epilogue::get_load_pipe_increment(TileShape{});
    constexpr int d_steps = Epilogue::get_store_pipe_increment(TileShape{});
    if (lane == 1) {
      epi_read.advance(c_steps);
      epi_write.advance(d_steps);
    }
    int group = -1;
    auto tmas = mainloop.tensormaps_init(
        params.mainloop, storage.tensormaps.mainloop[lane], params.hw_info.sm_count, blockIdx.x);
    cute::TmaDescriptor const* epi_tma = nullptr;
    if (!producer) {
      epi_tma = get<0>(epilogue.store_init(
          params.epilogue, storage.tensormaps.epilogue, params.hw_info.sm_count, blockIdx.x, lane));
    }
    while (work.is_valid()) {
      problem_shape = append<4>(params.problem_shape.get_problem_shape(work.L_idx), 1);
      bool const changed = group != work.L_idx;
      int const k_tiles = Scheduler::get_work_k_tile_count(work, problem_shape, TileShape{});
      auto m = idx2crd(work.M_idx, shape<2>(gA));
      auto n = idx2crd(work.N_idx, shape<2>(gB));
      if (producer) {
        if (changed) {
          inputs = mainloop.tensors_perform_update(inputs, params.mainloop, problem_shape, work.L_idx);
          mainloop.tensormaps_fence_acquire(tmas);
        }
        auto iter = make_coord_iterator(idx2crd(0, shape<3>(gA)), shape<3>(gA));
        mainloop.load(
            params.mainloop, pipeline, producer_state, inputs, tmas,
            make_coord(m, n, _, Int<0>{}), iter, k_tiles, tid % 32, 0, storage.tensors.mainloop[lane]);
        producer_state.advance(k_tiles);
      } else {
        if (changed && warp % 4 == 0) {
          epilogue.template tensormaps_perform_update<false>(
              storage.tensormaps.epilogue, params.epilogue, epi_tma, problem_shape, work.L_idx, lane);
          __syncwarp();
          epilogue.template tensormaps_cp_fence_release<false>(storage.tensormaps.epilogue, epi_tma, lane);
        }
        auto accum = partition_fragment_C(mma, take<0, 2>(TileShape{}));
        mainloop.mma(pipeline, consumer_state, accum, k_tiles, tid % 128,
                     storage.tensors.mainloop[lane], params.mainloop);
        mainloop.mma_tail(pipeline, consumer_state, k_tiles);
        consumer_state.advance(k_tiles);
        // Mainloops run independently; the epilogue scratch still has one owner.
        epi_order.wait();
        if (changed && warp % 4 == 0) {
          epilogue.template tensormaps_fence_acquire<false>(epi_tma);
        }
        auto states = epilogue.store(
            epi_load, epi_read, epi_store, epi_write, problem_shape, TileShape{},
            make_coord(m, n, _, idx2crd(work.L_idx, shape<4>(gB))), accum, mma, tid % 128,
            storage.tensors.epilogue, epi_tma, work.reduction_subtile_idx());
        epi_read = get<0>(states);
        epi_write = get<1>(states);
        auto tail = epilogue.store_tail(epi_load, epi_read, epi_store, epi_write);
        epi_read = get<0>(tail);
        epi_write = get<1>(tail);
        epi_read.advance(c_steps);
        epi_write.advance(d_steps);
        epi_order.arrive();
      }
      group = work.L_idx;
      work = get<0>(scheduler.fetch_next_work(work));
      if (work.is_valid()) {
        work = get<0>(scheduler.fetch_next_work(work));
      }
    }
    if (producer) {
      mainloop.load_tail(pipeline, producer_state);
    }
#endif
  }
};

}  // namespace cutlass::gemm::kernel
