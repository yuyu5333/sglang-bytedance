#pragma once

#include "cutlass/gemm/collective/collective_builder.hpp"

namespace sgl_kernel::staged_fp8 {

// Decode the existing sign-preprocessed payload and folded scales into ordinary
// row-major FP8. Each thread writes four values in each of two logical rows.
__global__ void decode_weights(
    uint32_t const* packed,
    uint8_t const* scales,
    uint32_t* decoded,
    int32_t const* offsets,
    int channels,
    int k,
    int64_t pairs) {
  int64_t const idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= pairs) {
    return;
  }
  int const k4 = idx % (k / 4);
  int64_t const row_pair = idx / (k / 4);
  int const expert = row_pair / (channels / 2);
  if (offsets[expert + 1] == offsets[expert]) {
    return;
  }
  int const pair_in_expert = row_pair % (channels / 2);
  int const row = pair_in_expert / 8 * 16 + pair_in_expert % 8;
  int const lane = k4 % 16;
  int const dst_row = row + (lane % 8) / 4 * 8;
  int const dst_k4 = k4 / 16 * 16 + lane / 8 * 8 + lane % 4 * 2;
  int64_t const src16 = (int64_t(expert) * channels + dst_row) * (k / 4) + dst_k4;
  uint32_t const word = packed[src16 / 2];
  int64_t const fold = ((int64_t(expert) * (channels / 64) + row / 64) * (k / 128) + k4 / 32) * 256;
  int const scale_k = k4 / 8 % 4;
  uint32_t const s0 = scales[fold + row % 16 * 16 + row % 64 / 16 * 4 + scale_k];
  uint32_t const s1 = scales[fold + (row + 8) % 16 * 16 + (row + 8) % 64 / 16 * 4 + scale_k];
  uint32_t const selector = word & 0x77777777U;
  uint32_t a, b;
  uint32_t const lo0 = s0 * 0x08080800U + 0x0c080000U;
  uint32_t const hi0 = s0 * 0x08080808U + 0x1c181410U;
  uint32_t const lo1 = s1 * 0x08080800U + 0x0c080000U;
  uint32_t const hi1 = s1 * 0x08080808U + 0x1c181410U;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(a) : "r"(lo0), "r"(hi0), "r"(selector));
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(b) : "r"(lo1), "r"(hi1), "r"(selector >> 16));
  a |= word & 0x80808080U;
  b |= (word << 4) & 0x80808080U;
  decoded[(int64_t(expert) * channels + row) * (k / 4) + k4] = a;
  decoded[(int64_t(expert) * channels + row + 8) * (k / 4) + k4] = b;
}

__global__ void prepare_pointers(
    int64_t* pointers,
    int32_t const* offsets,
    cutlass::float_e4m3_t const* weight,
    cutlass::float_e4m3_t const* activation,
    cutlass::bfloat16_t* output,
    float const* row_scales,
    int experts,
    int channels,
    int k) {
  int const e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e < experts) {
    pointers[e] = reinterpret_cast<int64_t>(weight + int64_t(e) * channels * k);
    pointers[experts + e] = reinterpret_cast<int64_t>(activation + int64_t(offsets[e]) * k);
    pointers[2 * experts + e] = reinterpret_cast<int64_t>(output + int64_t(offsets[e]) * channels);
    pointers[3 * experts + e] = reinterpret_cast<int64_t>(row_scales + offsets[e]);
  }
}

template <int TileN>
struct Config {
  using Tile = cute::Shape<cute::_128, cute::Int<TileN>, cute::_128>;
  using Cluster = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using Epi = typename w4a8_detail::W4A8EpilogueSelector<
      false, true, false, Tile, Cluster,
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong>::Type;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      cutlass::float_e4m3_t,
      cutlass::layout::RowMajor*,
      16,
      cutlass::float_e4m3_t,
      cutlass::layout::ColumnMajor*,
      16,
      float,
      Tile,
      Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epi::SharedStorage)>,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<w4a8_detail::ProblemShape, Mainloop, Epi>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template <int TileN>
void run(
    torch::Tensor& output,
    torch::Tensor const& activation,
    torch::Tensor const& packed,
    torch::Tensor const& row_scales,
    torch::Tensor const& weight_scales,
    torch::Tensor const& offsets,
    torch::Tensor const& problems) {
  using C = Config<TileN>;
  using Gemm = typename C::Gemm;
  using Mainloop = typename C::Mainloop;
  using Epi = typename C::Epi;
  int const experts = packed.size(0);
  int const channels = packed.size(1);
  int const k = activation.size(1);
  TORCH_CHECK(channels % 64 == 0 && k % 128 == 0);
  auto stream = at::cuda::getCurrentCUDAStream(activation.device().index());
  auto expanded = torch::empty({experts, channels, k}, activation.options());
  int64_t const pairs = int64_t(experts) * channels * k / 8;
  decode_weights<<<(pairs + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<uint32_t const*>(packed.data_ptr()),
      weight_scales.data_ptr<uint8_t>(),
      reinterpret_cast<uint32_t*>(expanded.data_ptr()),
      offsets.data_ptr<int32_t>(),
      channels, k, pairs);
  auto ptrs = torch::empty({4, experts}, offsets.options().dtype(torch::kInt64));
  prepare_pointers<<<(experts + 255) / 256, 256, 0, stream>>>(
      ptrs.data_ptr<int64_t>(), offsets.data_ptr<int32_t>(),
      static_cast<cutlass::float_e4m3_t const*>(expanded.data_ptr()),
      static_cast<cutlass::float_e4m3_t const*>(activation.data_ptr()),
      static_cast<cutlass::bfloat16_t*>(output.data_ptr()),
      row_scales.data_ptr<float>(), experts, channels, k);
  auto stride_k = torch::full({experts}, k, ptrs.options());
  auto stride_n = torch::full({experts}, channels, ptrs.options());
  auto* ptr = ptrs.data_ptr<int64_t>();
  typename Gemm::Arguments args;
  decltype(args.epilogue.thread) fusion{};
  fusion.token_scale_default = 1.0f;
  fusion.token_scale_ptr_array = reinterpret_cast<float const* const*>(ptr + 3 * experts);
  cutlass::KernelHardwareInfo hw;
  hw.device_id = activation.device().index();
  hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw.device_id);
  args = typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {experts, static_cast<w4a8_detail::ProblemShape::UnderlyingProblemShape*>(problems.data_ptr()), nullptr},
      {reinterpret_cast<cutlass::float_e4m3_t const**>(ptr),
       static_cast<typename Mainloop::InternalStrideA*>(stride_k.data_ptr()),
       reinterpret_cast<cutlass::float_e4m3_t const**>(ptr + experts),
       static_cast<typename Mainloop::InternalStrideB*>(stride_k.data_ptr())},
      {fusion, nullptr, nullptr,
       reinterpret_cast<cutlass::bfloat16_t**>(ptr + 2 * experts),
       static_cast<typename Epi::InternalStrideD*>(stride_n.data_ptr())},
      hw};
  Gemm gemm;
  auto workspace = torch::empty(
      {static_cast<int64_t>(gemm.get_workspace_size(args))}, offsets.options().dtype(torch::kUInt8));
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "Staged FP8 shape unsupported");
  TORCH_CHECK(gemm.initialize(args, workspace.data_ptr(), stream) == cutlass::Status::kSuccess,
              "Staged FP8 initialization failed");
  TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess, "Staged FP8 launch failed");
}

}  // namespace sgl_kernel::staged_fp8
