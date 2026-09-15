#pragma once

#include "cutlass_extensions/gemm/kernel/sm90_gemm_array_tma_warpspecialized_pingpong_precomputed.hpp"

namespace sgl_kernel::w4a8_detail {

template <class Problem, class Mainloop, class Epilogue, class Scheduler, int Ctas = 2>
struct ChannelCooperativeKernel
    : cutlass::gemm::kernel::GemmUniversalPrecomputedScheduler<Problem, Mainloop, Epilogue, Scheduler> {
  using Base = cutlass::gemm::kernel::GemmUniversalPrecomputedScheduler<Problem, Mainloop, Epilogue, Scheduler>;
  using Params = typename Base::Params;
  using Pipeline = typename Mainloop::MainloopPipeline;
  using State = typename Mainloop::PipelineState;
  using ElementB = typename Mainloop::ElementB;
  using ElementD = typename Epilogue::ElementD;
  using Mma = typename Mainloop::TiledMma;
  using Utils = cutlass::gemm::collective::detail::MixedGroupedGemmInputUtils<Mainloop>;
  static constexpr int K = cute::size<2>(typename Mainloop::TileShape{});
  static constexpr int Stages = Mainloop::DispatchPolicy::Stages;
  static constexpr uint32_t MinBlocksPerMultiprocessor = Ctas;
  using Panel = cute::Shape<cute::_64, cute::_32, cute::Int<K>>;

  struct SharedStorage : Base::SharedStorage {
    alignas(16) ElementD output[128 * 32];
    float token_scale[32];
  };
  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  static_assert(Ctas == 2 || Ctas == 3);
  static_assert(SharedStorageSize + 1024 <= 228 * 1024 / Ctas);
  static_assert(cute::size<0>(typename Mainloop::TileShape{}) == 128);
  static_assert(cute::size<1>(typename Mainloop::TileShape{}) == 32);
  static_assert(!Mainloop::UseIndependentTmaProducers);

  template <class Accumulators>
  CUTLASS_DEVICE void consume(Pipeline pipeline, int tiles, int panel, Accumulators& accum, SharedStorage& storage) {
    using namespace cute;
    int const tid = int(threadIdx.x) % 128;
    Mma mma;
    auto slice = mma.get_thread_slice(tid);
    auto full_a = make_tensor(make_smem_ptr(storage.tensors.mainloop.smem_A.begin()), typename Mainloop::SmemLayoutA{});
    auto panel_a = local_tile(full_a, make_shape(Int<64>{}, Int<K>{}, Int<Stages>{}), make_coord(panel, 0, 0));
    auto sA = as_position_independent_swizzle_tensor(panel_a);
    auto fp8 = slice.partition_fragment_A(sA(_, _, Int<0>{}));
    auto copy_atom = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, ElementB>{}, mma);
    auto copy_thread = copy_atom.get_thread_slice(tid);
    auto source = copy_thread.partition_S(recast<ElementB>(sA));
    auto raw = make_fragment_like<ElementB>(replace<2>(fp8.shape(), size<2>(fp8) / Int<2>{}));
    auto raw_copy = copy_thread.retile_D(raw);
    auto packed_layout = make_layout(
        make_shape(size<0>(raw), get<1>(raw.shape()), make_shape(Int<2>{}, size<2>(raw))),
        make_stride(Int<1>{}, size<0>(raw) * Int<2>{},
                    make_stride(size<0>(raw), size<0>(raw) * Int<2>{} * size<1>(raw))));
    auto packed = make_tensor(recast_ptr<typename Mainloop::ElementA>(raw.data()), packed_layout);
    auto full_scales = make_tensor(
        make_smem_ptr(reinterpret_cast<typename Mainloop::WeightScaleRawElement*>(
            storage.tensors.mainloop.smem_scale.begin())),
        typename Mainloop::SmemLayoutWeightScaleExpanded{});
    auto panel_scales = local_tile(
        full_scales, make_shape(Int<64>{}, Int<K>{}, Int<Stages>{}), make_coord(panel, 0, 0));
    auto scale_source = slice.partition_A(panel_scales);
    auto scale_rf = make_fragment_like<typename Mainloop::WeightScaleRawElement>(packed);
    auto sB = make_tensor(make_smem_ptr(storage.tensors.mainloop.smem_B.begin()), typename Mainloop::SmemLayoutB{});
    auto b = slice.make_fragment_B(slice.partition_B(sB));
    State read{};
    mma.accumulate_ = GMMA::ScaleOut::Zero;
    for (int tile = 0; tile < tiles; ++tile) {
      pipeline.consumer_wait(read);
      int const stage = read.index();
      cute::for_each(cute::make_seq<K / 64>{}, [&](auto k) {
        copy(copy_atom, source(_, _, k, stage), raw_copy(_, _, k));
      });
      cute::for_each(cute::make_seq<K / 32>{}, [&](auto k) {
        copy(scale_source(_, _, k, stage), scale_rf(_, _, k));
        auto slot = fp8(_, _, k);
        Utils::convert_A_kblock_fused_e8m0_pre_mma_raw_scale_to_slot(packed, slot, scale_rf, k);
        warpgroup_arrive();
        cute::gemm(mma, slot, b(_, _, k, stage), accum);
        mma.accumulate_ = GMMA::ScaleOut::One;
        if constexpr ((int(k) + 1) % 4 == 0) {
          warpgroup_commit_batch();
          warpgroup_wait<1>();
        }
      });
      warpgroup_wait<0>();
      pipeline.consumer_release(read);
      ++read;
    }
  }

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    using namespace cute;
    auto& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    int const tid = int(threadIdx.x);
    int const group = tid / 128;
    int const local_tid = tid % 128;
    if (group == 0) {
      cutlass::arch::warpgroup_reg_dealloc<24>();
    } else {
      cutlass::arch::warpgroup_reg_alloc<Ctas == 2 ? 104 : 72>();
    }
    Mainloop mainloop;
    Scheduler scheduler{params.scheduler};
    cudaGridDependencySynchronize();
    auto work = scheduler.initial_work_tile_info(typename Base::ClusterShape{});
    while (work.is_valid()) {
      auto problem = append<4>(params.problem_shape.get_problem_shape(work.L_idx), 1);
      typename Pipeline::Params pipe_params;
      pipe_params.role = group == 0 ? Pipeline::ThreadCategory::Producer : Pipeline::ThreadCategory::Consumer;
      pipe_params.is_leader = tid == 0;
      pipe_params.num_consumers = 256;
      pipe_params.num_producers = Mainloop::NumProducerThreadEvents;
      pipe_params.transaction_bytes = Mainloop::TmaTransactionBytes;
      Pipeline pipeline(storage.pipelines.mainloop, pipe_params, typename Base::ClusterShape{});
      if (tid < 32) {
        auto const* scales = params.epilogue.thread.token_scale_ptr_array
                                 ? params.epilogue.thread.token_scale_ptr_array[work.L_idx]
                                 : nullptr;
        int row = work.N_idx * 32 + tid;
        storage.token_scale[tid] = scales && row < get<1>(problem)
                                      ? scales[row]
                                      : params.epilogue.thread.token_scale_default;
      }
      __syncthreads();
      int const tiles = int(get<2>(problem)) / K;
      if (group == 0) {
        if (tid < 32) {
          auto inputs = mainloop.load_init(problem, params.mainloop);
          inputs = mainloop.tensors_perform_update(inputs, params.mainloop, problem, work.L_idx);
          auto maps = mainloop.tensormaps_init(
              params.mainloop, storage.tensormaps.mainloop, params.hw_info.sm_count, int(blockIdx.x));
          mainloop.tensormaps_fence_acquire(maps);
          auto a = get<0>(inputs);
          auto b = get<1>(inputs);
          auto coord = make_coord(idx2crd(work.M_idx, shape<2>(a)), idx2crd(work.N_idx, shape<2>(b)), _, 0);
          auto iterator = make_coord_iterator(idx2crd(0, shape<3>(a)), shape<3>(a));
          State write = cutlass::make_producer_start_state<Pipeline>();
          mainloop.load(params.mainloop, pipeline, write, inputs, maps, coord, iterator, tiles, tid, 0,
                        storage.tensors.mainloop);
          write.advance(tiles);
          mainloop.load_tail(pipeline, write);
        }
      } else {
        Mma mma;
        auto accum = partition_fragment_C(mma, take<0, 2>(Panel{}));
        consume(pipeline, tiles, group - 1, accum, storage);
        auto coords = mma.get_thread_slice(local_tid).partition_C(make_identity_tensor(make_shape(Int<64>{}, Int<32>{})));
        cutlass::NumericConverter<ElementD, float> convert;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accum); ++i) {
          auto coord = coords(i);
          int const channel = (group - 1) * 64 + int(get<0>(coord));
          int const row = int(get<1>(coord));
          storage.output[row * 128 + channel] = convert(accum(i) * storage.token_scale[row]);
        }
      }
      __syncthreads();
      using Vector = cutlass::Array<ElementD, 8>;
      auto* output = params.epilogue.ptr_D[work.L_idx];
      auto const stride = get<1>(params.epilogue.dD[work.L_idx]);
      if (tid < 256) {
        for (int v = tid; v < 512; v += 256) {
          int const row = work.N_idx * 32 + v / 16;
          int const channel = work.M_idx * 128 + (v % 16) * 8;
          if (row < get<1>(problem) && channel < get<0>(problem)) {
            *reinterpret_cast<Vector*>(output + int64_t(row) * stride + channel) =
                reinterpret_cast<Vector const*>(storage.output)[v];
          }
        }
      }
      __syncthreads();
      work = get<0>(scheduler.fetch_next_work(work));
    }
#endif
  }
};

}  // namespace sgl_kernel::w4a8_detail
