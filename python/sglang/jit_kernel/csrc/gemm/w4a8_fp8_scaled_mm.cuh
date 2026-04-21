#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/scalar_type.hpp>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <type_traits>

namespace {

constexpr int kW4A8FP8BlockSizeX = 16;
constexpr int kW4A8FP8BlockSizeY = 16;

template <typename T>
SGL_DEVICE float to_float(T value) {
  return static_cast<float>(value);
}

template <>
SGL_DEVICE float to_float<bf16_t>(bf16_t value) {
  return __bfloat162float(value);
}

template <>
SGL_DEVICE float to_float<fp16_t>(fp16_t value) {
  return __half2float(value);
}

template <typename T>
SGL_DEVICE T from_float(float value) {
  return static_cast<T>(value);
}

template <>
SGL_DEVICE bf16_t from_float<bf16_t>(float value) {
  return __float2bfloat16(value);
}

template <>
SGL_DEVICE fp16_t from_float<fp16_t>(float value) {
  return __float2half(value);
}

SGL_DEVICE int8_t unpack_signed_int4(const int8_t packed, const int64_t k_idx) {
  const uint8_t raw = static_cast<uint8_t>(packed);
  const uint8_t nibble = ((k_idx & 1) == 0) ? (raw & 0x0F) : ((raw >> 4) & 0x0F);
  return nibble >= 8 ? static_cast<int8_t>(nibble) - 16 : static_cast<int8_t>(nibble);
}

struct W4A8FP8ScaledMMParams {
  const fp8_e4m3_t* q_input;
  const int8_t* weight_packed;
  const float* x_scale;
  const float* weight_scale;
  const void* bias;
  void* output;
  int64_t m;
  int64_t n;
  int64_t k;
  int64_t packed_k;
  int64_t num_groups;
  int64_t group_size;
};

template <typename OutDType>
__global__ void w4a8_fp8_scaled_mm_naive_kernel(W4A8FP8ScaledMMParams params) {
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t row = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;

  if (row >= params.m || col >= params.n) {
    return;
  }

  const float x_scale = params.x_scale[row];
  float accum = 0.0f;

  for (int64_t k_idx = 0; k_idx < params.k; ++k_idx) {
    const float q_val = to_float(params.q_input[row * params.k + k_idx]);
    const int64_t packed_offset = col * params.packed_k + (k_idx >> 1);
    const int8_t w_int4 = unpack_signed_int4(params.weight_packed[packed_offset], k_idx);
    const float w_scale = params.weight_scale[col * params.num_groups + (k_idx / params.group_size)];
    accum += (q_val * x_scale) * (static_cast<float>(w_int4) * w_scale);
  }

  if (params.bias != nullptr) {
    const auto* bias = static_cast<const OutDType*>(params.bias);
    accum += to_float(bias[col]);
  }

  auto* output = static_cast<OutDType*>(params.output);
  output[row * params.n + col] = from_float<OutDType>(accum);
}

template <typename OutDType>
void launch_w4a8_fp8_scaled_mm(const W4A8FP8ScaledMMParams& params, DLDevice device) {
  using namespace host;

  if (params.m == 0 || params.n == 0) {
    return;
  }

  dim3 block_dim(kW4A8FP8BlockSizeX, kW4A8FP8BlockSizeY);
  dim3 grid_dim(div_ceil(params.n, int64_t(block_dim.x)), div_ceil(params.m, int64_t(block_dim.y)));
  LaunchKernel(grid_dim, block_dim, device)(w4a8_fp8_scaled_mm_naive_kernel<OutDType>, params);
}

template <typename OutDType>
void w4a8_fp8_scaled_mm(
    tvm::ffi::TensorView q_input,
    tvm::ffi::TensorView weight_packed,
    tvm::ffi::TensorView x_scale,
    tvm::ffi::TensorView weight_scale,
    tvm::ffi::TensorView output,
    int64_t group_size,
    tvm::ffi::Optional<tvm::ffi::TensorView> bias) {
  using namespace host;

  auto M = SymbolicSize{"M"};
  auto K = SymbolicSize{"K"};
  auto N = SymbolicSize{"N"};
  auto PackedK = SymbolicSize{"packed_K"};
  auto NumGroups = SymbolicSize{"num_groups"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  TensorMatcher({M, K})  //
      .with_dtype<fp8_e4m3_t>()
      .with_device(device)
      .verify(q_input);
  TensorMatcher({N, PackedK})  //
      .with_dtype<int8_t>()
      .with_device(device)
      .verify(weight_packed);
  TensorMatcher({N, NumGroups})  //
      .with_dtype<float>()
      .with_device(device)
      .verify(weight_scale);
  TensorMatcher({M, N})  //
      .with_dtype<OutDType>()
      .with_device(device)
      .verify(output);

  if (bias.has_value()) {
    TensorMatcher({N})  //
        .with_dtype<OutDType>()
        .with_device(device)
        .verify(bias.value());
  }

  RuntimeCheck(x_scale.ndim() == 1 || x_scale.ndim() == 2, "`x_scale` must be rank 1 or 2.");
  if (x_scale.ndim() == 1) {
    TensorMatcher({M})  //
        .with_dtype<float>()
        .with_device(device)
        .verify(x_scale);
  } else {
    TensorMatcher({M, 1})  //
        .with_dtype<float>()
        .with_device(device)
        .verify(x_scale);
  }

  RuntimeCheck(q_input.is_contiguous(), "`q_input` must be contiguous.");
  RuntimeCheck(weight_packed.is_contiguous(), "`weight_packed` must be contiguous.");
  RuntimeCheck(x_scale.is_contiguous(), "`x_scale` must be contiguous.");
  RuntimeCheck(weight_scale.is_contiguous(), "`weight_scale` must be contiguous.");
  RuntimeCheck(output.is_contiguous(), "`output` must be contiguous.");
  if (bias.has_value()) {
    RuntimeCheck(bias.value().is_contiguous(), "`bias` must be contiguous when provided.");
  }

  const int64_t m = M.unwrap();
  const int64_t n = N.unwrap();
  const int64_t k = K.unwrap();
  const int64_t packed_k = PackedK.unwrap();
  const int64_t num_groups = NumGroups.unwrap();

  RuntimeCheck(group_size > 0, "`group_size` must be positive, got ", group_size);
  RuntimeCheck(group_size == 128, "MVP only supports `group_size == 128`, got ", group_size);
  RuntimeCheck(k % group_size == 0, "`K` must be divisible by group_size. K = ", k, ", group_size = ", group_size);
  RuntimeCheck(
      packed_k * 2 == k,
      "`weight_packed` shape mismatch. Expected packed_K * 2 == K, got packed_K = ",
      packed_k,
      ", K = ",
      k);
  RuntimeCheck(
      num_groups == k / group_size,
      "`weight_scale` shape mismatch. Expected num_groups == K / group_size, got num_groups = ",
      num_groups,
      ", K / group_size = ",
      (k / group_size));

  auto params = W4A8FP8ScaledMMParams{
      .q_input = static_cast<const fp8_e4m3_t*>(q_input.data_ptr()),
      .weight_packed = static_cast<const int8_t*>(weight_packed.data_ptr()),
      .x_scale = static_cast<const float*>(x_scale.data_ptr()),
      .weight_scale = static_cast<const float*>(weight_scale.data_ptr()),
      .bias = bias.has_value() ? bias.value().data_ptr() : nullptr,
      .output = output.data_ptr(),
      .m = m,
      .n = n,
      .k = k,
      .packed_k = packed_k,
      .num_groups = num_groups,
      .group_size = group_size,
  };

  launch_w4a8_fp8_scaled_mm<OutDType>(params, device.unwrap());
}

}  // namespace
