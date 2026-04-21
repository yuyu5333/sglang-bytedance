#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/scalar_type.hpp>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

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
void launch_w4a8_fp8_scaled_mm(const W4A8FP8ScaledMMParams& params, DLDevice device) {
  using namespace host;
  (void)params;
  (void)device;

  RuntimeCheck(
      false,
      "w4a8_fp8_scaled_mm JIT kernel is not implemented yet. "
      "This file is currently a shape-checked skeleton only.");
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
  TensorMatcher({M, 1})  //
      .with_dtype<float>()
      .with_device(device)
      .verify(x_scale);
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

  RuntimeCheck(q_input.strides().back() == 1, "`q_input` last dim must be contiguous.");
  RuntimeCheck(weight_packed.strides().back() == 1, "`weight_packed` last dim must be contiguous.");
  RuntimeCheck(output.strides().back() == 1, "`output` last dim must be contiguous.");

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
