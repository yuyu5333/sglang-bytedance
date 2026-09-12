#pragma once

#include "cute/tensor.hpp"

namespace cutlass::gemm::collective {

template <bool Enabled, int Bytes, int Alignment>
struct PackedWeightScaleStorage {};

template <int Bytes, int Alignment>
struct PackedWeightScaleStorage<true, Bytes, Alignment> {
  CUTE_ALIGNAS(Alignment)
  cute::ArrayEngine<uint8_t, Bytes> smem_packed_weight;
};

template <int StageStride, class Layout>
CUTE_HOST_DEVICE constexpr auto packed_weight_stage_layout(Layout layout) {
  if constexpr (cute::is_composed_layout<Layout>::value) {
    auto base = layout.layout_b();
    return cute::make_composed_layout(
        layout.layout_a(),
        layout.offset(),
        cute::make_layout(
            cute::shape(base),
            cute::replace<2>(
                cute::stride(base), cute::compact_col_major(cute::shape<2>(base), cute::Int<StageStride>{}))));
  } else {
    return cute::make_layout(
        cute::shape(layout),
        cute::replace<2>(
            cute::stride(layout), cute::compact_col_major(cute::shape<2>(layout), cute::Int<StageStride>{})));
  }
}

}  // namespace cutlass::gemm::collective
