#include <c10/cuda/CUDAGuard.h>
#include <cudaTypedefs.h>
#include <torch/all.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_set>

#include "cutlass/cutlass.h"
#include "w4a8_grouped_mm_c3x.cuh"

using namespace cute;

namespace {

enum class Sched { PP, CO };

template <int M, int N, int K, int A, int B, int C, Sched S>
struct SM90W4A8Config {
  using KernelSchedule = std::conditional_t<
      S == Sched::PP,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>;

  using EpilogueSchedule = std::conditional_t<
      S == Sched::PP,
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong,
      cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative>;

  using TileShape = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>;
  using ClusterShape = cute::Shape<cute::Int<A>, cute::Int<B>, cute::Int<C>>;
  using Cutlass3xW4A8Gemm = cutlass_3x_w4a8_group_gemm<TileShape, ClusterShape, KernelSchedule, EpilogueSchedule>;
};

template <int M, int N, int K, int A, int B, int C>
using SM90_PP = SM90W4A8Config<M, N, K, A, B, C, Sched::PP>;

template <int M, int N, int K, int A, int B, int C>
using SM90_CO = SM90W4A8Config<M, N, K, A, B, C, Sched::CO>;

template <typename Config>
inline void invoke_gemm(
    torch::Tensor& d_tensors,
    torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes,
    torch::Tensor const& a_strides,
    torch::Tensor const& b_strides,
    torch::Tensor const& d_strides,
    torch::Tensor const& s_strides,
    int64_t chunk_size) {
  using GemmT = typename Config::Cutlass3xW4A8Gemm;
  cutlass_w4a8_group_gemm_caller<GemmT>(
      d_tensors,
      a_tensors,
      b_tensors,
      a_scales,
      b_scales,
      expert_offsets,
      problem_sizes,
      a_strides,
      b_strides,
      d_strides,
      s_strides,
      chunk_size);
}

// Helper macro to reduce code duplication
// Note: Config must be wrapped in parentheses when it contains commas (e.g., template parameters)
// This uses a helper macro to strip the parentheses from the template parameter
#define INVOKE_GEMM_WITH_CONFIG_HELPER(...) \
  invoke_gemm<__VA_ARGS__>(                 \
      d_tensors,                            \
      a_tensors,                            \
      b_tensors,                            \
      a_scales,                             \
      b_scales,                             \
      expert_offsets,                       \
      problem_sizes,                        \
      a_strides,                            \
      b_strides,                            \
      d_strides,                            \
      s_strides,                            \
      chunk_size)
#define INVOKE_GEMM_WITH_CONFIG(Config) INVOKE_GEMM_WITH_CONFIG_HELPER Config

inline bool w4a8_branch_trace_enabled() {
  static bool enabled = [] {
    char const* env = std::getenv("SGLANG_W4A8_GEMM_TRACE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
  }();
  return enabled;
}

inline void trace_w4a8_branch_once(
    char const* branch_label,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    int64_t topk,
    int64_t chunk_size) {
  if (!w4a8_branch_trace_enabled()) {
    return;
  }

  static std::mutex trace_mutex;
  static std::unordered_set<std::string> seen_branch_labels;

  std::lock_guard<std::mutex> lock(trace_mutex);
  if (!seen_branch_labels.insert(branch_label).second) {
    return;
  }

  std::fprintf(
      stderr,
      "[SGLANG_W4A8_GEMM_TRACE] branch=%s m=%u n=%u k=%u topk=%lld chunk_size=%lld\n",
      branch_label,
      m,
      n,
      k,
      static_cast<long long>(topk),
      static_cast<long long>(chunk_size));
  std::fflush(stderr);
}

#define TRACE_AND_INVOKE_GEMM(label, Config) \
  do {                                       \
    trace_w4a8_branch_once(label, m, n, k, topk, chunk_size); \
    INVOKE_GEMM_WITH_CONFIG(Config);         \
  } while (0)

#define TRACE_AND_INVOKE_K512_CONFIG_FOR_TOKEN(token, shape_prefix)                         \
  do {                                                                                      \
    if (std::strcmp(token, "pp_64x16x512_111") == 0) {                                     \
      TRACE_AND_INVOKE_GEMM(shape_prefix "_SM90_PP_64x16x512_c111",                        \
                            (SM90_PP<64, 16, 512, 1, 1, 1>));                              \
    } else if (std::strcmp(token, "pp_64x32x512_111") == 0) {                              \
      TRACE_AND_INVOKE_GEMM(shape_prefix "_SM90_PP_64x32x512_c111",                        \
                            (SM90_PP<64, 32, 512, 1, 1, 1>));                              \
    } else if (std::strcmp(token, "pp_64x32x512_211") == 0) {                              \
      TRACE_AND_INVOKE_GEMM(shape_prefix "_SM90_PP_64x32x512_c211",                        \
                            (SM90_PP<64, 32, 512, 2, 1, 1>));                              \
    } else if (std::strcmp(token, "co_128x16x512_111") == 0) {                             \
      TRACE_AND_INVOKE_GEMM(shape_prefix "_SM90_CO_128x16x512_c111",                       \
                            (SM90_CO<128, 16, 512, 1, 1, 1>));                             \
    } else if (std::strcmp(token, "co_128x16x512_211") == 0) {                             \
      TRACE_AND_INVOKE_GEMM(shape_prefix "_SM90_CO_128x16x512_c211",                       \
                            (SM90_CO<128, 16, 512, 2, 1, 1>));                             \
    } else if (std::strcmp(token, "co_128x32x512_111") == 0) {                             \
      TRACE_AND_INVOKE_GEMM(shape_prefix "_SM90_CO_128x32x512_c111",                       \
                            (SM90_CO<128, 32, 512, 1, 1, 1>));                             \
    } else if (std::strcmp(token, "co_128x32x512_211") == 0) {                             \
      TRACE_AND_INVOKE_GEMM(shape_prefix "_SM90_CO_128x32x512_c211",                       \
                            (SM90_CO<128, 32, 512, 2, 1, 1>));                             \
    } else if (std::strcmp(token, "co_128x64x512_111") == 0) {                             \
      TRACE_AND_INVOKE_GEMM(shape_prefix "_SM90_CO_128x64x512_c111",                       \
                            (SM90_CO<128, 64, 512, 1, 1, 1>));                             \
    } else if (std::strcmp(token, "co_128x64x512_211") == 0) {                             \
      TRACE_AND_INVOKE_GEMM(shape_prefix "_SM90_CO_128x64x512_c211",                       \
                            (SM90_CO<128, 64, 512, 2, 1, 1>));                             \
    }                                                                                       \
  } while (0)

#define MAYBE_FORCE_K512_CONFIG(env_name, shape_prefix)                 \
  do {                                                                  \
    char const* forced_token = std::getenv(env_name);                   \
    if (forced_token != nullptr && forced_token[0] != '\0') {           \
      TRACE_AND_INVOKE_K512_CONFIG_FOR_TOKEN(forced_token, shape_prefix); \
      return;                                                           \
    }                                                                   \
  } while (0)

#define TRACE_AND_INVOKE_K128_CONFIG_FOR_TOKEN(token, shape_prefix)                         \
  do {                                                                                      \
    if (std::strcmp(token, "pp_64x16x128_111") == 0) {                                     \
      TRACE_AND_INVOKE_GEMM(shape_prefix "_SM90_PP_64x16x128_c111",                        \
                            (SM90_PP<64, 16, 128, 1, 1, 1>));                              \
    } else if (std::strcmp(token, "pp_128x32x128_111") == 0) {                             \
      TRACE_AND_INVOKE_GEMM(shape_prefix "_SM90_PP_128x32x128_c111",                       \
                            (SM90_PP<128, 32, 128, 1, 1, 1>));                             \
    } else if (std::strcmp(token, "pp_128x32x128_211") == 0) {                             \
      TRACE_AND_INVOKE_GEMM(shape_prefix "_SM90_PP_128x32x128_c211",                       \
                            (SM90_PP<128, 32, 128, 2, 1, 1>));                             \
    } else if (std::strcmp(token, "pp_128x64x128_111") == 0) {                             \
      TRACE_AND_INVOKE_GEMM(shape_prefix "_SM90_PP_128x64x128_c111",                       \
                            (SM90_PP<128, 64, 128, 1, 1, 1>));                             \
    } else if (std::strcmp(token, "co_128x16x128_111") == 0) {                             \
      TRACE_AND_INVOKE_GEMM(shape_prefix "_SM90_CO_128x16x128_c111",                       \
                            (SM90_CO<128, 16, 128, 1, 1, 1>));                             \
    } else if (std::strcmp(token, "co_128x16x128_211") == 0) {                             \
      TRACE_AND_INVOKE_GEMM(shape_prefix "_SM90_CO_128x16x128_c211",                       \
                            (SM90_CO<128, 16, 128, 2, 1, 1>));                             \
    } else if (std::strcmp(token, "co_128x32x128_111") == 0) {                             \
      TRACE_AND_INVOKE_GEMM(shape_prefix "_SM90_CO_128x32x128_c111",                       \
                            (SM90_CO<128, 32, 128, 1, 1, 1>));                             \
    } else if (std::strcmp(token, "co_128x32x128_211") == 0) {                             \
      TRACE_AND_INVOKE_GEMM(shape_prefix "_SM90_CO_128x32x128_c211",                       \
                            (SM90_CO<128, 32, 128, 2, 1, 1>));                             \
    } else if (std::strcmp(token, "co_128x64x128_111") == 0) {                             \
      TRACE_AND_INVOKE_GEMM(shape_prefix "_SM90_CO_128x64x128_c111",                       \
                            (SM90_CO<128, 64, 128, 1, 1, 1>));                             \
    } else if (std::strcmp(token, "co_128x64x128_211") == 0) {                             \
      TRACE_AND_INVOKE_GEMM(shape_prefix "_SM90_CO_128x64x128_c211",                       \
                            (SM90_CO<128, 64, 128, 2, 1, 1>));                             \
    }                                                                                       \
  } while (0)

#define MAYBE_FORCE_K128_CONFIG(env_name, shape_prefix)                 \
  do {                                                                  \
    char const* forced_token = std::getenv(env_name);                   \
    if (forced_token != nullptr && forced_token[0] != '\0') {           \
      TRACE_AND_INVOKE_K128_CONFIG_FOR_TOKEN(forced_token, shape_prefix); \
      return;                                                           \
    }                                                                   \
  } while (0)

void dispatch_w4a8_moe_mm_sm90(
    torch::Tensor& d_tensors,
    torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes,
    torch::Tensor const& a_strides,
    torch::Tensor const& b_strides,
    torch::Tensor const& d_strides,
    torch::Tensor const& s_strides,
    int64_t chunk_size,
    int64_t topk) {
  uint32_t const m = a_tensors.size(0) / topk;
  uint32_t const n = d_tensors.size(1);
  uint32_t const k = a_tensors.size(1);

  if (n == 4096 && k == 7168) {
    // group gemm 1
    if (m <= 4) {
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<64, 32, 512, 2, 1, 1>));
    } else if (m <= 32) {
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 16, 512, 2, 1, 1>));
    } else if (m <= 256) {
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 16, 512, 1, 1, 1>));
    } else if (m <= 1024) {
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 32, 512, 2, 1, 1>));
    } else if (m <= 4096) {
      // Optimized for prefill: seq_len up to 4096 (m=4096 with topk=1)
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 64, 512, 2, 1, 1>));
    } else {
      // Optimized for prefill: seq_len up to 8192 (m=8192 with topk=1)
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 64, 512, 1, 1, 1>));
    }
  } else if (n == 7168 && k == 2048) {
    // group gemm 2
    if (m <= 8) {
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<64, 16, 512, 1, 1, 1>));
    } else if (m <= 512) {
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 32, 512, 1, 1, 1>));
    } else if (m <= 4096) {
      // Optimized for prefill: larger cluster for better throughput
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 64, 512, 2, 1, 1>));
    } else {
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 64, 512, 1, 1, 1>));
    }
  } else if (n == 512 && k == 7168) {
    // group gemm 1 for tp
    if (m <= 4) {
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<64, 32, 512, 2, 1, 1>));
    } else if (m <= 32) {
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 16, 512, 2, 1, 1>));
    } else if (m <= 256) {
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 16, 512, 1, 1, 1>));
    } else if (m <= 1024) {
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 32, 512, 2, 1, 1>));
    } else {
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 64, 512, 1, 1, 1>));
    }
  } else if (n == 1024 && k == 4096) {
    if (m <= 32) {
      MAYBE_FORCE_K512_CONFIG("SGLANG_W4A8_FORCE_N1024_K4096_LE32", "n1024_k4096_forced");
    } else if (m <= 1024) {
      MAYBE_FORCE_K512_CONFIG("SGLANG_W4A8_FORCE_N1024_K4096_LE1024", "n1024_k4096_forced");
    } else {
      MAYBE_FORCE_K512_CONFIG("SGLANG_W4A8_FORCE_N1024_K4096_GT1024", "n1024_k4096_forced");
    }
    // TP4 PD prefill/decode gemm1 hot path
    if (m <= 8) {
      TRACE_AND_INVOKE_GEMM("n1024_k4096_m_le_8_SM90_CO_128x64x512_c111", (SM90_CO<128, 64, 512, 1, 1, 1>));
    } else if (m <= 24) {
      TRACE_AND_INVOKE_GEMM("n1024_k4096_m_le_24_SM90_PP_64x32x512_c111", (SM90_PP<64, 32, 512, 1, 1, 1>));
    } else if (m <= 32) {
      TRACE_AND_INVOKE_GEMM("n1024_k4096_m_le_32_SM90_CO_128x16x512_c111", (SM90_CO<128, 16, 512, 1, 1, 1>));
    } else if (m <= 64) {
      TRACE_AND_INVOKE_GEMM("n1024_k4096_m_le_64_SM90_CO_128x32x512_c211", (SM90_CO<128, 32, 512, 2, 1, 1>));
    } else if (m <= 96) {
      TRACE_AND_INVOKE_GEMM("n1024_k4096_m_le_96_SM90_CO_128x32x512_c111", (SM90_CO<128, 32, 512, 1, 1, 1>));
    } else if (m <= 128) {
      TRACE_AND_INVOKE_GEMM("n1024_k4096_m_le_128_SM90_CO_128x16x512_c111", (SM90_CO<128, 16, 512, 1, 1, 1>));
    } else if (m <= 1024) {
      TRACE_AND_INVOKE_GEMM("n1024_k4096_m_le_1024_SM90_PP_64x32x512_c111", (SM90_PP<64, 32, 512, 1, 1, 1>));
    } else {
      TRACE_AND_INVOKE_GEMM("n1024_k4096_m_gt_1024_SM90_PP_64x32x512_c111", (SM90_PP<64, 32, 512, 1, 1, 1>));
    }
  } else if (n == 512 && k == 4096) {
    if (m <= 32) {
      MAYBE_FORCE_K512_CONFIG("SGLANG_W4A8_FORCE_N512_K4096_LE32", "n512_k4096_forced");
    } else if (m <= 1024) {
      MAYBE_FORCE_K512_CONFIG("SGLANG_W4A8_FORCE_N512_K4096_LE1024", "n512_k4096_forced");
    } else {
      MAYBE_FORCE_K512_CONFIG("SGLANG_W4A8_FORCE_N512_K4096_GT1024", "n512_k4096_forced");
    }
    // TP8 colocated gemm1 hot path
    if (m <= 32) {
      TRACE_AND_INVOKE_GEMM("n512_k4096_m_le_32_SM90_CO_128x16x512_c211", (SM90_CO<128, 16, 512, 2, 1, 1>));
    } else if (m <= 64) {
      TRACE_AND_INVOKE_GEMM("n512_k4096_m_le_64_SM90_PP_64x16x512_c111", (SM90_PP<64, 16, 512, 1, 1, 1>));
    } else if (m <= 192) {
      TRACE_AND_INVOKE_GEMM("n512_k4096_m_le_192_SM90_PP_64x32x512_c111", (SM90_PP<64, 32, 512, 1, 1, 1>));
    } else if (m <= 1024) {
      TRACE_AND_INVOKE_GEMM("n512_k4096_m_le_1024_SM90_CO_128x64x512_c211", (SM90_CO<128, 64, 512, 2, 1, 1>));
    } else {
      TRACE_AND_INVOKE_GEMM("n512_k4096_m_gt_1024_SM90_PP_64x32x512_c111", (SM90_PP<64, 32, 512, 1, 1, 1>));
    }
  } else if (n == 4096 && k == 512) {
    if (m <= 32) {
      MAYBE_FORCE_K512_CONFIG("SGLANG_W4A8_FORCE_N4096_K512_LE32", "n4096_k512_forced");
    } else if (m <= 1024) {
      MAYBE_FORCE_K512_CONFIG("SGLANG_W4A8_FORCE_N4096_K512_LE1024", "n4096_k512_forced");
    } else {
      MAYBE_FORCE_K512_CONFIG("SGLANG_W4A8_FORCE_N4096_K512_GT1024", "n4096_k512_forced");
    }
    // TP4 PD prefill/decode gemm2 hot path
    if (m <= 16) {
      TRACE_AND_INVOKE_GEMM("n4096_k512_m_le_16_SM90_PP_64x32x512_c211", (SM90_PP<64, 32, 512, 2, 1, 1>));
    } else if (m <= 24) {
      TRACE_AND_INVOKE_GEMM("n4096_k512_m_le_24_SM90_CO_128x32x512_c211", (SM90_CO<128, 32, 512, 2, 1, 1>));
    } else if (m <= 64) {
      TRACE_AND_INVOKE_GEMM("n4096_k512_m_le_64_SM90_CO_128x64x512_c211", (SM90_CO<128, 64, 512, 2, 1, 1>));
    } else if (m <= 96) {
      TRACE_AND_INVOKE_GEMM("n4096_k512_m_le_96_SM90_CO_128x32x512_c111", (SM90_CO<128, 32, 512, 1, 1, 1>));
    } else if (m <= 512) {
      TRACE_AND_INVOKE_GEMM("n4096_k512_m_le_512_SM90_PP_64x32x512_c111", (SM90_PP<64, 32, 512, 1, 1, 1>));
    } else if (m <= 1024) {
      TRACE_AND_INVOKE_GEMM("n4096_k512_m_le_1024_SM90_CO_128x16x512_c111", (SM90_CO<128, 16, 512, 1, 1, 1>));
    } else {
      TRACE_AND_INVOKE_GEMM("n4096_k512_m_gt_1024_SM90_CO_128x64x512_c111", (SM90_CO<128, 64, 512, 1, 1, 1>));
    }
  } else if (n == 4096 && k == 256) {
    if (m <= 8) {
      MAYBE_FORCE_K128_CONFIG("SGLANG_W4A8_FORCE_N4096_K256_LE8", "n4096_k256_forced");
    } else if (m <= 32) {
      MAYBE_FORCE_K128_CONFIG("SGLANG_W4A8_FORCE_N4096_K256_LE32", "n4096_k256_forced");
    } else {
      MAYBE_FORCE_K128_CONFIG("SGLANG_W4A8_FORCE_N4096_K256_GT32", "n4096_k256_forced");
    }
    // TP8 colocated gemm2 hot path.
    if (m <= 8) {
      TRACE_AND_INVOKE_GEMM("n4096_k256_m_le_8_SM90_PP_64x16x128_c111", (SM90_PP<64, 16, 128, 1, 1, 1>));
    } else if (m <= 16) {
      TRACE_AND_INVOKE_GEMM("n4096_k256_m_le_16_SM90_PP_128x32x128_c211", (SM90_PP<128, 32, 128, 2, 1, 1>));
    } else if (m <= 32) {
      TRACE_AND_INVOKE_GEMM("n4096_k256_m_le_32_SM90_PP_128x32x128_c111", (SM90_PP<128, 32, 128, 1, 1, 1>));
    } else if (m <= 48) {
      TRACE_AND_INVOKE_GEMM("n4096_k256_m_le_48_SM90_PP_64x16x128_c111", (SM90_PP<64, 16, 128, 1, 1, 1>));
    } else if (m <= 64) {
      TRACE_AND_INVOKE_GEMM("n4096_k256_m_le_64_SM90_PP_128x64x128_c111", (SM90_PP<128, 64, 128, 1, 1, 1>));
    } else if (m <= 192) {
      TRACE_AND_INVOKE_GEMM("n4096_k256_m_le_192_SM90_PP_128x32x128_c211", (SM90_PP<128, 32, 128, 2, 1, 1>));
    } else if (m <= 4096) {
      TRACE_AND_INVOKE_GEMM("n4096_k256_m_le_4096_SM90_PP_128x64x128_c111", (SM90_PP<128, 64, 128, 1, 1, 1>));
    } else {
      TRACE_AND_INVOKE_GEMM("n4096_k256_m_gt_4096_SM90_PP_128x32x128_c211", (SM90_PP<128, 32, 128, 2, 1, 1>));
    }
  } else if (n == 7168 && k == 256) {
    // group gemm 2 for tp
    if (m <= 8) {
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<64, 16, 128, 1, 1, 1>));
    } else if (m <= 32) {
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<128, 32, 128, 1, 1, 1>));
    } else if (m <= 512) {
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<128, 32, 128, 2, 1, 1>));
    } else {
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<128, 64, 128, 1, 1, 1>));
    }
  } else {
    if (k % 512 == 0) {
      // For large m (prefill), prefer larger cluster
      if (m <= 32) {
        // Decode: target batch size (16-32) - use cluster size 1 for better latency
        INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 16, 512, 1, 1, 1>));
      } else if (m <= 1024) {
        // Decode: large batch or small prefill
        INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 32, 512, 1, 1, 1>));
      } else {
        // Prefill: large sequence length - prefer larger cluster
        INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 64, 512, 1, 1, 1>));
      }
    } else {
      if (m <= 32) {
        // Decode: target batch size (16-32) - use larger tile for better throughput
        INVOKE_GEMM_WITH_CONFIG((SM90_PP<128, 32, 128, 1, 1, 1>));
      } else {
        // Prefill: larger sequence length
        INVOKE_GEMM_WITH_CONFIG((SM90_PP<128, 64, 128, 1, 1, 1>));
      }
    }
  }
}

}  // namespace

void cutlass_w4a8_moe_mm_sm90(
    torch::Tensor& d_tensors,
    torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes,
    torch::Tensor const& a_strides,
    torch::Tensor const& b_strides,
    torch::Tensor const& d_strides,
    torch::Tensor const& s_strides,
    int64_t chunk_size,
    int64_t topk) {
  dispatch_w4a8_moe_mm_sm90(
      d_tensors,
      a_tensors,
      b_tensors,
      a_scales,
      b_scales,
      expert_offsets,
      problem_sizes,
      a_strides,
      b_strides,
      d_strides,
      s_strides,
      chunk_size,
      topk);
}
