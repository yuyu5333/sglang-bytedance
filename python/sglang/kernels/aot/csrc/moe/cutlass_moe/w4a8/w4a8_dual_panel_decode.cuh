#pragma once

#include "cutlass_extensions/detail/collective/mixed_input_utils.hpp"
#include "cutlass_extensions/gemm/kernel/sm90_gemm_array_tma_warpspecialized_pingpong_precomputed.hpp"

namespace sgl_kernel::w4a8_detail {

template <int Stages>
class DualPanelReadyPipeline : public cutlass::PipelineAsync<Stages> {
  using Base = cutlass::PipelineAsync<Stages>;

 public:
  using ThreadCategory = typename Base::ThreadCategory;
  using SharedStorage = typename Base::SharedStorage;
  using PipelineState = typename Base::PipelineState;
  struct Params {
    ThreadCategory role = ThreadCategory::NonParticipant;
    uint32_t transaction_bytes = 0;
    uint32_t is_leader = 0;
    uint32_t num_consumers = 256;
    uint32_t num_producers = 128;
  };

  template <class ClusterShape>
  CUTLASS_DEVICE DualPanelReadyPipeline(SharedStorage& storage, Params params, ClusterShape)
      : Base(storage, make_params(params)) {
    static_assert(cute::size(ClusterShape{}) == 1);
  }

 private:
  CUTLASS_DEVICE static typename Base::Params make_params(Params params) {
    typename Base::Params result;
    result.role = params.role;
    result.producer_arv_count = params.num_producers;
    result.consumer_arv_count = params.num_consumers;
    return result;
  }
};

template <class RawMainloop>
struct DualPanelDecodeMainloop : RawMainloop {
  using Base = RawMainloop;
  using TileShape = typename Base::TileShape;
  using DispatchPolicy = typename Base::DispatchPolicy;
  using ElementB = typename Base::ElementB;
  using ElementAccumulator = typename Base::ElementAccumulator;
  using RawPipeline = typename Base::MainloopPipeline;
  using MainloopPipeline = DualPanelReadyPipeline<DispatchPolicy::Stages>;
  using PipelineState = typename MainloopPipeline::PipelineState;
  using PipelineStorage = typename MainloopPipeline::SharedStorage;
  using RawMma = typename Base::TiledMma;
  using Utils = cutlass::gemm::collective::detail::MixedGroupedGemmInputUtils<Base>;
  static constexpr int KBlocks = cute::size<2>(TileShape{}) / 32;
  using PanelShape = cute::Shape<
      decltype(cute::size<0>(TileShape{})), cute::Int<32>, decltype(cute::size<2>(TileShape{}))>;
  using TiledMma = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<
          ElementB, ElementB, ElementAccumulator, PanelShape, cute::GMMA::Major::K, cute::GMMA::Major::K>()));
  using DecodedLayout = decltype(cute::tile_to_shape(
      cute::GMMA::Layout_K_SW128_Atom<ElementB>{},
      cute::make_shape(cute::size<0>(TileShape{}), cute::size<2>(TileShape{}), cute::Int<DispatchPolicy::Stages>{})));

  static_assert(cute::size<1>(TileShape{}) == 64);
  static_assert(!Base::UseIndependentTmaProducers);

  struct TensorStorage : Base::TensorStorage {
    alignas(128) cute::ArrayEngine<ElementB, cute::cosize_v<DecodedLayout>> decoded;
    alignas(16) typename RawPipeline::SharedStorage raw_pipeline;
  };

  CUTLASS_DEVICE void decode_stage(TensorStorage& storage, int stage) {
    using namespace cute;
    RawMma mma;
    int const tid = int(threadIdx.x) % 128;
    auto thread_mma = mma.get_thread_slice(tid);
    auto sA = make_tensor(make_smem_ptr(storage.smem_A.begin()), typename Base::SmemLayoutA{});
    auto sAPos = as_position_independent_swizzle_tensor(sA);
    auto fragment = thread_mma.partition_fragment_A(sAPos(_, _, Int<0>{}));
    auto ldsm_copy = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, ElementB>{}, mma);
    auto thread_copy = ldsm_copy.get_thread_slice(tid);
    auto source = thread_copy.partition_S(recast<ElementB>(sAPos));
    auto raw_shape = replace<2>(fragment.shape(), size<2>(fragment) / Int<2>{});
    auto raw = make_fragment_like<ElementB>(raw_shape);
    auto raw_copy = thread_copy.retile_D(raw);
    auto fp4_layout = make_layout(
        make_shape(size<0>(raw), get<1>(raw.shape()), make_shape(Int<2>{}, size<2>(raw))),
        make_stride(
            Int<1>{},
            size<0>(raw) * Int<2>{},
            make_stride(size<0>(raw), size<0>(raw) * Int<2>{} * size<1>(raw))));
    auto fp4 = make_tensor(recast_ptr<typename Base::ElementA>(raw.data()), fp4_layout);
    auto scales = make_tensor(
        make_smem_ptr(reinterpret_cast<typename Base::WeightScaleRawElement*>(storage.smem_scale.begin())),
        typename Base::SmemLayoutWeightScaleExpanded{});
    auto scale_source = thread_mma.partition_A(scales);
    auto scale_fragment = make_fragment_like<typename Base::WeightScaleRawElement>(fp4);
    auto decoded = make_tensor(make_smem_ptr(storage.decoded.begin()), DecodedLayout{});
    auto destination = thread_mma.partition_A(decoded);

    cute::for_each(cute::make_seq<KBlocks / 2>{}, [&](auto k) {
      copy(ldsm_copy, source(_, _, k, stage), raw_copy(_, _, k));
    });
    cute::for_each(cute::make_seq<KBlocks>{}, [&](auto k) {
      copy(scale_source(_, _, k, stage), scale_fragment(_, _, k));
      auto output = fragment(_, _, k);
      Utils::convert_A_kblock_fused_e8m0_pre_mma_raw_scale_to_slot(fp4, output, scale_fragment, k);
      copy(output, destination(_, _, k, stage));
    });
    cutlass::arch::fence_view_async_shared();
  }

  template <class Inputs, class TensorMaps, class Coord, class Iterator>
  CUTLASS_DEVICE void produce(
      typename Base::Params const& params,
      MainloopPipeline ready,
      Inputs const& inputs,
      TensorMaps const& maps,
      Coord const& coord,
      Iterator iterator,
      int tiles,
      TensorStorage& storage) {
    typename RawPipeline::Params raw_params;
    raw_params.role = RawPipeline::ThreadCategory::ProducerConsumer;
    raw_params.is_leader = threadIdx.x == 0;
    raw_params.num_consumers = 128;
    raw_params.num_producers = 1;
    raw_params.transaction_bytes = Base::TmaTransactionBytes;
    RawPipeline raw(storage.raw_pipeline, raw_params, cute::Shape<cute::_1, cute::_1, cute::_1>{});
    cutlass::arch::NamedBarrier::sync(128, 0);
    PipelineState write = cutlass::make_producer_start_state<RawPipeline>();
    PipelineState read{};
    PipelineState publish = cutlass::make_producer_start_state<MainloopPipeline>();
    int issued = 0;
    auto issue = [&] {
      ready.producer_acquire(write);
      if (threadIdx.x < 32) {
        Base::load(params, raw, write, inputs, maps, coord, iterator, 1,
                   int(threadIdx.x), 0, storage);
      }
      ++write;
      ++iterator;
      ++issued;
    };
    for (int i = 0; i < cute::min(tiles, DispatchPolicy::Stages - 1); ++i) {
      issue();
    }
    for (int i = 0; i < tiles; ++i) {
      raw.consumer_wait(read);
      decode_stage(storage, read.index());
      cutlass::arch::NamedBarrier::sync(128, 0);
      ready.producer_commit(publish);
      raw.consumer_release(read);
      ++read;
      ++publish;
      if (issued < tiles) {
        issue();
      }
    }
    ready.producer_tail(publish);
  }

  template <class Accumulators>
  CUTLASS_DEVICE void consume(
      MainloopPipeline ready, int tiles, int panel, Accumulators& accum, TensorStorage& storage) {
    using namespace cute;
    TiledMma mma;
    auto slice = mma.get_thread_slice(int(threadIdx.x) % 128);
    auto sA = make_tensor(make_smem_ptr(storage.decoded.begin()), DecodedLayout{});
    auto sB = make_tensor(make_smem_ptr(storage.smem_B.begin()), typename Base::SmemLayoutB{});
    auto panel_b = local_tile(
        sB, make_shape(Int<32>{}, size<2>(TileShape{}), Int<DispatchPolicy::Stages>{}), make_coord(panel, 0, 0));
    auto a = slice.make_fragment_A(slice.partition_A(sA));
    auto b = slice.make_fragment_B(slice.partition_B(panel_b));
    PipelineState read{};
    PipelineState release{};
    mma.accumulate_ = GMMA::ScaleOut::Zero;
    warpgroup_arrive();
    for (int tile = 0; tile < tiles; ++tile) {
      ready.consumer_wait(read);
      int const stage = read.index();
      cute::for_each(cute::make_seq<KBlocks>{}, [&](auto k) {
        cute::gemm(mma, a(_, _, k, stage), b(_, _, k, stage), accum);
        mma.accumulate_ = GMMA::ScaleOut::One;
      });
      warpgroup_commit_batch();
      warpgroup_wait<1>();
      if (tile > 0) {
        ready.consumer_release(release);
        ++release;
      }
      ++read;
    }
    warpgroup_wait<0>();
    ready.consumer_release(release);
  }
};

template <class ProblemShape, class Mainloop, class Epilogue, class Scheduler, int Ctas = 1>
struct DualPanelDecodeKernel
    : cutlass::gemm::kernel::GemmUniversalPrecomputedScheduler<ProblemShape, Mainloop, Epilogue, Scheduler> {
  using Base = cutlass::gemm::kernel::GemmUniversalPrecomputedScheduler<ProblemShape, Mainloop, Epilogue, Scheduler>;
  using Params = typename Base::Params;
  using SharedStorage = typename Base::SharedStorage;
  using TileShape = typename Mainloop::TileShape;
  using PanelShape = typename Mainloop::PanelShape;
  using TiledMma = typename Mainloop::TiledMma;
  static constexpr uint32_t MinBlocksPerMultiprocessor = Ctas;
  static_assert(Ctas == 1 || Ctas == 2);
  static_assert(sizeof(SharedStorage) <= (Ctas == 1 ? 227 : 114) * 1024);

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    using namespace cute;
    auto& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    int const tid = int(threadIdx.x);
    int const group = tid / 128;
    int const local_tid = tid % 128;
    if constexpr (Ctas == 1) {
      if (group == 0) {
        cutlass::arch::warpgroup_reg_dealloc<104>();
      } else {
        cutlass::arch::warpgroup_reg_alloc<192>();
      }
    } else {
      if (group == 0) {
        cutlass::arch::warpgroup_reg_alloc<96>();
      } else {
        cutlass::arch::warpgroup_reg_dealloc<72>();
      }
    }

    using EpiLoad = typename Epilogue::LoadPipeline;
    typename EpiLoad::Params epi_params;
    epi_params.role = EpiLoad::ThreadCategory::Consumer;
    epi_params.dst_blockid = 0;
    epi_params.producer_arv_count = 32;
    epi_params.consumer_arv_count = 128;
    if constexpr (Epilogue::RequiresTransactionBytes) {
      epi_params.transaction_bytes = params.epilogue.tma_transaction_bytes;
    }
    EpiLoad epi_load(storage.pipelines.epi_load, epi_params);
    using EpiStore = typename Epilogue::StorePipeline;
    typename EpiStore::Params store_params;
    store_params.always_wait = true;
    EpiStore epi_store(store_params);
    typename Epilogue::LoadPipelineState epi_read{};
    auto epi_write = cutlass::make_producer_start_state<EpiStore>();
    Epilogue epilogue(params.epilogue, storage.tensors.epilogue);
    Mainloop mainloop;
    TiledMma mma;
    Scheduler scheduler{params.scheduler};
    __syncthreads();
    cudaGridDependencySynchronize();
    auto work = scheduler.initial_work_tile_info(typename Base::ClusterShape{});
    int previous_expert = -1;
    while (work.is_valid()) {
      auto problem = append<4>(params.problem_shape.get_problem_shape(work.L_idx), 1);
      auto accum = partition_fragment_C(mma, take<0, 2>(PanelShape{}));
      using Pipeline = typename Mainloop::MainloopPipeline;
      typename Pipeline::Params ready_params;
      ready_params.role = group == 0 ? Pipeline::ThreadCategory::Producer : Pipeline::ThreadCategory::Consumer;
      Pipeline ready(storage.pipelines.mainloop, ready_params, typename Base::ClusterShape{});
      __syncthreads();
      int const tiles = cute::ceil_div(get<2>(problem), size<2>(TileShape{}));

      if (group == 0) {
        auto inputs = mainloop.load_init(problem, params.mainloop);
        inputs = mainloop.tensors_perform_update(inputs, params.mainloop, problem, work.L_idx);
        auto maps = mainloop.tensormaps_init(
            params.mainloop, storage.tensormaps.mainloop, params.hw_info.sm_count, int(blockIdx.x));
        if (tid < 32) {
          mainloop.tensormaps_fence_acquire(maps);
        }
        auto gA = get<0>(inputs);
        auto gB = get<1>(inputs);
        auto coord = make_coord(idx2crd(work.M_idx, shape<2>(gA)),
                                idx2crd(work.N_idx, shape<2>(gB)), _, 0);
        auto iterator = make_coord_iterator(idx2crd(0, shape<3>(gA)), shape<3>(gA));
        mainloop.produce(params.mainloop, ready, inputs, maps, coord, iterator, tiles, storage.tensors.mainloop);
      } else {
        mainloop.consume(ready, tiles, group - 1, accum, storage.tensors.mainloop);
      }
      __syncthreads();

      // The existing epilogue shares staging storage; serialize only its two panels.
      for (int panel = 0; panel < 2; ++panel) {
        if (group == panel + 1) {
          auto map = get<0>(epilogue.store_init(
              params.epilogue, storage.tensormaps.epilogue, params.hw_info.sm_count, int(blockIdx.x), panel));
          if (previous_expert != work.L_idx && local_tid < 32) {
            epilogue.template tensormaps_perform_update<false>(
                storage.tensormaps.epilogue, params.epilogue, map, problem, work.L_idx, panel);
            __syncwarp();
            epilogue.template tensormaps_cp_fence_release<false>(storage.tensormaps.epilogue, map, panel);
            epilogue.template tensormaps_fence_acquire<false>(map);
          }
          auto coord = make_coord(work.M_idx, work.N_idx * 2 + panel, _, work.L_idx);
          auto states = epilogue.store(
              epi_load, epi_read, epi_store, epi_write, problem, PanelShape{}, coord, accum, mma, local_tid,
              storage.tensors.epilogue, map, work.reduction_subtile_idx());
          epi_read = get<0>(states);
          epi_write = get<1>(states);
          auto tail = epilogue.store_tail(epi_load, epi_read, epi_store, epi_write);
          epi_read = get<0>(tail);
          epi_write = get<1>(tail);
        }
        __syncthreads();
      }
      previous_expert = work.L_idx;
      auto next = scheduler.fetch_next_work(work);
      work = get<0>(next);
    }
#else
    printf("DualPanelDecodeKernel requires sm90a\n");
#endif
  }
};

}  // namespace sgl_kernel::w4a8_detail
