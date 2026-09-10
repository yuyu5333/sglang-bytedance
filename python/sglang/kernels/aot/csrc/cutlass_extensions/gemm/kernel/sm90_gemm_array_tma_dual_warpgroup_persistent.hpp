#pragma once

#include "cutlass_extensions/gemm/kernel/sm90_gemm_array_tma_warpspecialized_pingpong_precomputed.hpp"

namespace cutlass::gemm::kernel {

template <class ProblemShape, class Mainloop, class Epilogue, class Scheduler>
class DualWarpgroupPersistentGemm
    : public GemmUniversalPrecomputedScheduler<ProblemShape, Mainloop, Epilogue, Scheduler> {
  using Base = GemmUniversalPrecomputedScheduler<ProblemShape, Mainloop, Epilogue, Scheduler>;

 public:
  using Params = typename Base::Params;
  using SharedStorage = typename Base::SharedStorage;
  using TileShape = typename Base::TileShape;
  using TiledMma = typename Base::TiledMma;
  using ClusterShape = typename Base::ClusterShape;
  static constexpr uint32_t MaxThreadsPerBlock = 256;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 2;
  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  static_assert(SharedStorageSize <= 232448 / 2, "Two resident CTAs must fit in SM90 shared memory");

  static dim3 get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    using namespace cute;
    static_assert(size(TiledMma{}) == 128 && size(ClusterShape{}) == 1);
    SharedStorage& shared = *reinterpret_cast<SharedStorage*>(smem_buf);
    int const thread = int(threadIdx.x);
    int const local_thread = thread % 128;
    int const warp = canonical_warp_idx_sync();
    bool const producer = thread < 128;

    using Pipeline = typename Mainloop::MainloopPipeline;
    typename Pipeline::Params pipe_params;
    if (producer && warp == 0) {
      pipe_params.role = Pipeline::ThreadCategory::Producer;
    } else if (!producer) {
      pipe_params.role = Pipeline::ThreadCategory::Consumer;
    }
    pipe_params.is_leader = local_thread == 0;
    pipe_params.num_consumers = 128;
    pipe_params.num_producers = Mainloop::NumProducerThreadEvents;
    pipe_params.transaction_bytes = params.mainloop.tma_transaction_bytes;
    Pipeline pipeline(shared.pipelines.mainloop, pipe_params, ClusterShape{});

    using LoadPipeline = typename Epilogue::LoadPipeline;
    typename LoadPipeline::Params load_params;
    if (!producer) {
      load_params.role = LoadPipeline::ThreadCategory::Consumer;
    }
    load_params.dst_blockid = 0;
    load_params.producer_arv_count = 32;
    load_params.consumer_arv_count = 128;
    if constexpr (Epilogue::RequiresTransactionBytes) {
      load_params.transaction_bytes = params.epilogue.tma_transaction_bytes;
    }
    LoadPipeline epi_load(shared.pipelines.epi_load, load_params);
    __syncthreads();

    // 32 + 224 registers per producer/consumer lane pair is 32768 words/CTA.
    // The producer releases its share before the consumer requests extra RF.
    if (producer) {
      cutlass::arch::warpgroup_reg_dealloc<32>();
      if (warp != 0) {
        return;
      }
    } else {
      cutlass::arch::warpgroup_reg_alloc<224>();
    }

    Scheduler scheduler(params.scheduler);
    auto work = scheduler.initial_work_tile_info(ClusterShape{});
    if (!work.is_valid()) {
      return;
    }
    Mainloop mainloop;
    auto problem = append<4>(params.problem_shape.get_problem_shape(work.L_idx), 1);
    auto inputs = mainloop.load_init(problem, params.mainloop);
    auto gA = get<0>(inputs);
    auto gB = get<1>(inputs);
    auto const tile = TileShape{};
    int const worker = int(blockIdx.x + blockIdx.y * gridDim.x);

    if (producer) {
      auto producer_state = cutlass::make_producer_start_state<Pipeline>();
      auto maps = mainloop.tensormaps_init(
          params.mainloop, shared.tensormaps.mainloop, params.hw_info.sm_count, worker);
      int previous_group = -1;
      while (work.is_valid()) {
        problem = append<4>(params.problem_shape.get_problem_shape(work.L_idx), 1);
        if (work.L_idx != previous_group) {
          inputs = mainloop.tensors_perform_update(inputs, params.mainloop, problem, work.L_idx);
          mainloop.tensormaps_fence_acquire(maps);
          previous_group = work.L_idx;
        }
        auto coord = make_coord(
            idx2crd(work.M_idx, shape<2>(gA)), idx2crd(work.N_idx, shape<2>(gB)), _, Int<0>{});
        int const count = Scheduler::get_work_k_tile_count(work, problem, tile);
        auto k_iter = make_coord_iterator(
            idx2crd(Scheduler::get_work_k_tile_start(work), shape<3>(gA)), shape<3>(gA));
        mainloop.load(
            params.mainloop,
            pipeline,
            producer_state,
            inputs,
            maps,
            coord,
            k_iter,
            count,
            canonical_lane_idx(),
            0,
            shared.tensors.mainloop);
        producer_state.advance(count);
        work = get<0>(scheduler.fetch_next_work(work));
      }
      mainloop.load_tail(pipeline, producer_state);
    } else {
      Epilogue epilogue(params.epilogue, shared.tensors.epilogue);
      if (epilogue.is_producer_load_needed()) {
        asm volatile("trap;");
      }
      using StorePipeline = typename Epilogue::StorePipeline;
      typename StorePipeline::Params store_params;
      store_params.always_wait = true;
      StorePipeline epi_store(store_params);
      typename Mainloop::PipelineState consumer_state;
      typename Epilogue::LoadPipelineState load_state;
      auto store_state = cutlass::make_producer_start_state<StorePipeline>();
      auto output_map = get<0>(epilogue.store_init(
          params.epilogue, shared.tensormaps.epilogue, params.hw_info.sm_count, worker, 0));
      TiledMma mma;
      int previous_group = -1;
      while (work.is_valid()) {
        problem = append<4>(params.problem_shape.get_problem_shape(work.L_idx), 1);
        bool const changed = work.L_idx != previous_group;
        if (changed && local_thread < 32) {
          epilogue.template tensormaps_perform_update<false>(
              shared.tensormaps.epilogue, params.epilogue, output_map, problem, work.L_idx, 0);
          __syncwarp();
          epilogue.template tensormaps_cp_fence_release<false>(shared.tensormaps.epilogue, output_map, 0);
        }
        int const count = Scheduler::get_work_k_tile_count(work, problem, tile);
        auto accum = partition_fragment_C(mma, take<0, 2>(tile));
        mainloop.mma(pipeline, consumer_state, accum, count, local_thread, shared.tensors.mainloop, params.mainloop);
        mainloop.mma_tail(pipeline, consumer_state, count);
        consumer_state.advance(count);
        Scheduler::fixup(params.scheduler, work, accum, 1, 0);
        if (changed && local_thread < 32) {
          epilogue.template tensormaps_fence_acquire<false>(output_map);
        }
        auto coord = make_coord(
            idx2crd(work.M_idx, shape<2>(gA)),
            idx2crd(work.N_idx, shape<2>(gB)),
            _,
            idx2crd(work.L_idx, shape<4>(gB)));
        auto states = epilogue.store(
            epi_load,
            load_state,
            epi_store,
            store_state,
            problem,
            tile,
            coord,
            accum,
            mma,
            local_thread,
            shared.tensors.epilogue,
            output_map,
            work.reduction_subtile_idx());
        load_state = get<0>(states);
        store_state = get<1>(states);
        states = epilogue.store_tail(epi_load, load_state, epi_store, store_state);
        load_state = get<0>(states);
        store_state = get<1>(states);
        previous_group = work.L_idx;
        work = get<0>(scheduler.fetch_next_work(work));
      }
    }
#endif
  }
};

}  // namespace cutlass::gemm::kernel
