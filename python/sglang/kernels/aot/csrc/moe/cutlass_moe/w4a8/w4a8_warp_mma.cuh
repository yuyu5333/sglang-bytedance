#pragma once

#include <cute/arch/mma_sm89.hpp>

namespace sgl_kernel::w4a8_detail {

template <class RawMainloop>
struct WarpMmaMainloop : RawMainloop {
  using Base = RawMainloop;
  using TileShape = typename Base::TileShape;
  using MainloopPipeline = typename Base::MainloopPipeline;
  using PipelineState = typename Base::PipelineState;
  using TensorStorage = typename Base::TensorStorage;
  using ElementB = typename Base::ElementB;
  using TiledMma = typename Base::TiledMma;
  using Utils = cutlass::gemm::collective::detail::MixedGroupedGemmInputUtils<Base>;
  static constexpr int KBlocks = cute::size<2>(TileShape{}) / 32;
  static constexpr int ChannelBlocks = cute::size<0>(TileShape{}) / 64;
  static constexpr int TokenBlocks = cute::size<1>(TileShape{}) / 8;
  static constexpr bool UseTailMmaHandoff = true;
  static constexpr int ConsumerRegisters = 104;
  static constexpr int CtasPerSm = 2;

  static_assert(!Base::UseIndependentTmaProducers);
  static_assert(cute::size<1>(TileShape{}) == 32);

  template <class Accumulators, class Refill, class Handoff>
  CUTLASS_DEVICE void mma_with_released_stage_producer(
      MainloopPipeline pipeline,
      PipelineState read,
      Accumulators& accum,
      int tiles,
      int thread_idx,
      TensorStorage& storage,
      typename Base::Params const&,
      Refill& refill,
      Handoff handoff) {
    using namespace cute;
    TiledMma mma;
    auto slice = mma.get_thread_slice(thread_idx);
    auto sA = make_tensor(make_smem_ptr(storage.smem_A.begin()), typename Base::SmemLayoutA{});
    auto position_a = as_position_independent_swizzle_tensor(sA);
    auto probe = slice.partition_fragment_A(position_a(_, _, Int<0>{}));
    auto copy_atom = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, ElementB>{}, mma);
    auto copy_thread = copy_atom.get_thread_slice(thread_idx);
    auto source = copy_thread.partition_S(recast<ElementB>(position_a));
    auto raw = make_fragment_like<ElementB>(replace<2>(probe.shape(), size<2>(probe) / Int<2>{}));
    auto raw_copy = copy_thread.retile_D(raw);
    auto fp4_layout = make_layout(
        make_shape(size<0>(raw), get<1>(raw.shape()), make_shape(Int<2>{}, size<2>(raw))),
        make_stride(Int<1>{}, size<0>(raw) * Int<2>{},
                    make_stride(size<0>(raw), size<0>(raw) * Int<2>{} * size<1>(raw))));
    auto fp4 = make_tensor(recast_ptr<typename Base::ElementA>(raw.data()), fp4_layout);
    auto decoded = make_fragment_like<ElementB>(take<0, 2>(probe.shape()));
    auto scales = make_tensor(
        make_smem_ptr(reinterpret_cast<typename Base::WeightScaleRawElement*>(storage.smem_scale.begin())),
        typename Base::SmemLayoutWeightScaleExpanded{});
    auto scale_source = slice.partition_A(scales);
    auto scale_fragment = make_fragment_like<typename Base::WeightScaleRawElement>(fp4);
    auto sB = make_tensor(make_smem_ptr(storage.smem_B.begin()), typename Base::SmemLayoutB{});
    clear(accum);
    int const lane = thread_idx % 32;

    for (int tile = 0; tile < tiles; ++tile) {
      pipeline.consumer_wait(read);
      int const stage = read.index();
      if (tile == tiles - 1) {
        handoff();
      }
      cute::for_each(cute::make_seq<KBlocks / 2>{}, [&](auto k) {
        copy(copy_atom, source(_, _, k, stage), raw_copy(_, _, k));
      });
      cute::for_each(cute::make_seq<KBlocks>{}, [&](auto k) {
        copy(scale_source(_, _, k, stage), scale_fragment(_, _, k));
        Utils::convert_A_kblock_fused_e8m0_pre_mma_raw_scale_to_slot(fp4, decoded, scale_fragment, k);
        auto a = recast<uint32_t>(decoded);
        cute::for_each(cute::make_seq<TokenBlocks>{}, [&](auto n) {
          uint32_t b[2] = {};
          CUTLASS_PRAGMA_UNROLL
          for (int half = 0; half < 2; ++half) {
            CUTLASS_PRAGMA_UNROLL
            for (int byte = 0; byte < 4; ++byte) {
              auto value = sB(int(n) * 8 + lane / 4, int(k) * 32 + half * 16 + lane % 4 * 4 + byte, stage);
              b[half] |= uint32_t(value.storage) << (byte * 8);
            }
          }
          cute::for_each(cute::make_seq<ChannelBlocks>{}, [&](auto m) {
            // Each physical warp owns the same 16 channel rows in both MMA layouts.
            auto& c0 = accum(int(n) * 4 + 0, m, 0);
            auto& c1 = accum(int(n) * 4 + 1, m, 0);
            auto& c2 = accum(int(n) * 4 + 2, m, 0);
            auto& c3 = accum(int(n) * 4 + 3, m, 0);
            cute::SM89_16x8x32_F32E4M3E4M3F32_TN::fma(
                c0, c1, c2, c3, a(0, m), a(1, m), a(2, m), a(3, m), b[0], b[1], c0, c1, c2, c3);
          });
        });
      });
      if (tile + 1 < tiles) {
        pipeline.consumer_release(read);
        refill();
      }
      ++read;
    }
  }
};

}  // namespace sgl_kernel::w4a8_detail
