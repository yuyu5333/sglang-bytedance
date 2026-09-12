#pragma once

#include "cute/tensor.hpp"

namespace cutlass::gemm::collective {

template <bool Enabled, int Bytes>
struct PackedWeightScaleStorage {};

template <int Bytes>
struct PackedWeightScaleStorage<true, Bytes> {
  CUTE_ALIGNAS(128)
  cute::ArrayEngine<uint8_t, Bytes> smem_packed_weight;
};

template <int StageStride, class Layout>
CUTE_HOST_DEVICE constexpr auto packed_weight_stage_layout(Layout layout) {
  if constexpr (cute::is_composed_layout<Layout>::value) {
    auto base = layout.layout_b();
    return cute::make_composed_layout(
        layout.layout_a(),
        layout.offset(),
        cute::make_layout(cute::shape(base), cute::replace<2>(cute::stride(base), cute::Int<StageStride>{})));
  } else {
    return cute::make_layout(cute::shape(layout), cute::replace<2>(cute::stride(layout), cute::Int<StageStride>{}));
  }
}

}  // namespace cutlass::gemm::collective
