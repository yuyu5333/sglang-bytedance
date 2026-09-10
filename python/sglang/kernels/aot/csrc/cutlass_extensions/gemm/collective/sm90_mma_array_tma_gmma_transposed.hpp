#pragma once

#include "cutlass_extensions/detail/collective/mixed_input_utils.hpp"

namespace cutlass::gemm::collective {

template <class RawMainloop>
struct TransposedMxfp4Mainloop : RawMainloop {
  using Base = RawMainloop;
  using TileShape = typename Base::TileShape;
  using ElementB = typename Base::ElementB;
  using ElementAccumulator = typename Base::ElementAccumulator;
  using Params = typename Base::Params;
  using MainloopPipeline = typename Base::MainloopPipeline;
  using PipelineState = typename Base::PipelineState;
  using TiledMma = typename Base::TiledMma;
  using Utils = detail::MixedGroupedGemmInputUtils<Base>;
  static constexpr int Channels = cute::size<0>(TileShape{});
  static constexpr int Tokens = cute::size<1>(TileShape{});
  static constexpr int TileK = cute::size<2>(TileShape{});
  static constexpr int KBlocks = TileK / 32;
  static_assert(Tokens == 64);
  using TransposedShape = cute::Shape<cute::Int<Tokens>, cute::Int<Channels>, cute::Int<TileK>>;
  using TransposedMma = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<
      ElementB, ElementB, ElementAccumulator, TransposedShape, cute::GMMA::Major::K, cute::GMMA::Major::K>()));
  using DecodedLayout = decltype(cute::tile_to_shape(
      cute::GMMA::Layout_K_SW128_Atom<ElementB>{}, cute::Shape<cute::Int<Channels>, cute::Int<TileK>, cute::_2>{}));

  struct TensorStorage : Base::TensorStorage {
    union alignas(128) {
      cute::ArrayEngine<ElementB, cute::cosize_v<DecodedLayout>> decoded;
      ElementAccumulator transpose[Tokens * (Channels + 4)];
    };
  };

  template <class FrgTensorC>
  CUTLASS_DEVICE void mma(
      MainloopPipeline pipeline,
      PipelineState read,
      FrgTensorC& accum,
      int tiles,
      int thread_idx,
      TensorStorage& storage,
      Params const&) {
    using namespace cute;
    TiledMma original_mma;
    auto original_thread = original_mma.get_thread_slice(thread_idx);
    auto packed = make_tensor(make_smem_ptr(storage.smem_A.begin()), typename Base::SmemLayoutA{});
    auto packed_pos = as_position_independent_swizzle_tensor(packed);
    auto converted = original_thread.partition_fragment_A(packed_pos(_, _, Int<0>{}));
    auto ldsm_copy = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, ElementB>{}, original_mma);
    auto thread_copy = ldsm_copy.get_thread_slice(thread_idx);
    auto source = thread_copy.partition_S(recast<ElementB>(packed_pos));
    auto raw_shape = replace<2>(converted.shape(), size<2>(converted) / Int<2>{});
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
    auto scale_source = original_thread.partition_A(scales);
    auto scale_fragment = make_fragment_like<typename Base::WeightScaleRawElement>(fp4);
    auto decoded = make_tensor(make_smem_ptr(storage.decoded.begin()), DecodedLayout{});
    auto decoded_destination = original_thread.partition_A(decoded);
    auto activation = make_tensor(make_smem_ptr(storage.smem_B.begin()), typename Base::SmemLayoutB{});

    TransposedMma mma;
    auto thread = mma.get_thread_slice(thread_idx);
    auto a = thread.make_fragment_A(thread.partition_A(activation));
    auto b = thread.make_fragment_B(thread.partition_B(decoded));
    auto transposed_accum = partition_fragment_C(mma, take<0, 2>(TransposedShape{}));
    PipelineState release = read;
    mma.accumulate_ = GMMA::ScaleOut::Zero;
    warpgroup_arrive();
    for (int tile = 0; tile < tiles; ++tile) {
      pipeline.consumer_wait(read);
      int const stage = read.index();
      int const decoded_stage = tile % 2;
      cute::for_each(cute::make_seq<KBlocks / 2>{}, [&](auto k) {
        copy(ldsm_copy, source(_, _, k, stage), raw_copy(_, _, k));
      });
      cute::for_each(cute::make_seq<KBlocks>{}, [&](auto k) {
        copy(scale_source(_, _, k, stage), scale_fragment(_, _, k));
        auto output = converted(_, _, k);
        Utils::convert_A_kblock_fused_e8m0_pre_mma_raw_scale_to_slot(fp4, output, scale_fragment, k);
        copy(output, decoded_destination(_, _, k, decoded_stage));
      });
      cutlass::arch::fence_view_async_shared();
      cutlass::arch::NamedBarrier::sync(128, 0);
      cute::for_each(cute::make_seq<KBlocks>{}, [&](auto k) {
        cute::gemm(mma, a(_, _, k, stage), b(_, _, k, decoded_stage), transposed_accum);
        mma.accumulate_ = GMMA::ScaleOut::One;
      });
      warpgroup_commit_batch();
      warpgroup_wait<1>();
      if (tile > 0) {
        pipeline.consumer_release(release);
        ++release;
      }
      ++read;
    }
    warpgroup_wait<0>();

    auto transposed_coordinates = thread.partition_C(make_identity_tensor(make_shape(Int<Tokens>{}, Int<Channels>{})));
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(transposed_accum); ++i) {
      auto coord = transposed_coordinates(i);
      storage.transpose[int(get<0>(coord)) * (Channels + 4) + int(get<1>(coord))] = transposed_accum(i);
    }
    cutlass::arch::NamedBarrier::sync(128, 0);
    auto coordinates = original_thread.partition_C(make_identity_tensor(make_shape(Int<Channels>{}, Int<Tokens>{})));
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accum); ++i) {
      auto coord = coordinates(i);
      accum(i) = storage.transpose[int(get<1>(coord)) * (Channels + 4) + int(get<0>(coord))];
    }
  }
};

}  // namespace cutlass::gemm::collective
