#pragma once

#include <cuda/atomic>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"

namespace cutlass::gemm::collective {

struct Gemm1FinalizeParams {
  __nv_bfloat16 const* gate_up = nullptr;
  __nv_fp8_e4m3* output_q = nullptr;
  float* output_scale = nullptr;
  float const* residual = nullptr;
  int32_t* tile_arrivals = nullptr;
  int32_t const* expert_offsets = nullptr;
  float swiglu_limit = 0.0f;
  bool has_swiglu_limit = false;
};

template <int TileM, int TileN, class ProblemShape, class BlockCoord, class ReleaseEpilogue>
CUTLASS_DEVICE void finalize_gemm1_tile(
    Gemm1FinalizeParams const& params,
    ProblemShape const& problem,
    BlockCoord const& block_coord,
    int thread_idx,
    int* shared_last,
    ReleaseEpilogue release_epilogue) {
  int const expert = int(cute::get<3>(block_coord));
  int const local_row = int(cute::get<1>(block_coord)) * TileN;
  int const expert_rows = int(cute::get<1>(problem));
  int const hidden = int(cute::get<0>(problem)) / 2;
  int const first_row = params.expert_offsets[expert] + local_row;
  int const tiles = (2 * hidden + TileM - 1) / TileM;
  auto sync_warpgroup = [] {
    cutlass::arch::NamedBarrier::sync(128, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
  };

  // The caller has waited for all TMA stores. Release/acquire orders completed
  // gate/up stores across CTAs; only the last CTA reads the complete rows.
  sync_warpgroup();
  if (thread_idx == 0) {
    cuda::atomic_ref<int32_t, cuda::thread_scope_device> arrivals(params.tile_arrivals[first_row]);
    *shared_last = int(arrivals.fetch_add(1, cuda::memory_order_acq_rel) == tiles - 1);
  }
  sync_warpgroup();
  bool const last = *shared_last != 0;
  sync_warpgroup();
  // Quantization below only touches RF and global memory.
  release_epilogue();
  if (!last) {
    return;
  }

  int const warp = thread_idx / 32;
  int const lane = thread_idx % 32;
  constexpr int Hidden = 2048;
  constexpr int Values = Hidden / 32;
  for (int row_in_tile = warp; row_in_tile < TileN; row_in_tile += 4) {
    if (local_row + row_in_tile >= expert_rows) {
      continue;
    }
    int64_t const row = int64_t(first_row) + row_in_tile;
    __nv_bfloat16 values[Values];
    float maximum = 0.0f;
    CUTLASS_PRAGMA_NO_UNROLL
    for (int j = 0; j < Values; ++j) {
      int const col = j * 32 + lane;
      __nv_bfloat16 gate = params.gate_up[row * (2 * Hidden) + col];
      __nv_bfloat16 up = params.gate_up[row * (2 * Hidden) + Hidden + col];
      if (params.has_swiglu_limit) {
        gate = __float2bfloat16_rn(fminf(__bfloat162float(gate), params.swiglu_limit));
        up = __float2bfloat16_rn(
            fmaxf(fminf(__bfloat162float(up), params.swiglu_limit), -params.swiglu_limit));
      }
      float const x = __bfloat162float(gate);
      values[j] = __float2bfloat16_rn((x / (1.0f + expf(-x))) * __bfloat162float(up));
      maximum = fmaxf(maximum, fabsf(__bfloat162float(values[j])));
    }
    CUTLASS_PRAGMA_UNROLL
    for (int delta = 16; delta > 0; delta /= 2) {
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, delta));
    }
    float const scale = maximum / 448.0f;
    float const scale_inv = scale == 0.0f ? 0.0f : 1.0f / scale;
    if (lane == 0) {
      params.output_scale[row] = scale * params.residual[expert];
    }
    CUTLASS_PRAGMA_NO_UNROLL
    for (int j = 0; j < Values; ++j) {
      float value = __bfloat162float(values[j]) * scale_inv;
      value = fmaxf(fminf(value, 448.0f), -448.0f);
      params.output_q[row * Hidden + j * 32 + lane] = static_cast<__nv_fp8_e4m3>(value);
    }
  }
}

}  // namespace cutlass::gemm::collective
