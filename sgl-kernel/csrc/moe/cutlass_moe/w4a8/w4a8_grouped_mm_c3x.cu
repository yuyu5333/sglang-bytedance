#include <c10/cuda/CUDAGuard.h>
#include <cudaTypedefs.h>
#include <torch/all.h>

#include <cstdio>
#include <cstdlib>
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
      TRACE_AND_INVOKE_GEMM(
          "n4096_k7168_m_le_4_SM90_PP_64x32x512_c211",
          (SM90_PP<64, 32, 512, 2, 1, 1>));
    } else if (m <= 32) {
      TRACE_AND_INVOKE_GEMM(
          "n4096_k7168_m_le_32_SM90_CO_128x16x512_c211",
          (SM90_CO<128, 16, 512, 2, 1, 1>));
    } else if (m <= 256) {
      TRACE_AND_INVOKE_GEMM(
          "n4096_k7168_m_le_256_SM90_CO_128x16x512_c111",
          (SM90_CO<128, 16, 512, 1, 1, 1>));
    } else if (m <= 1024) {
      TRACE_AND_INVOKE_GEMM(
          "n4096_k7168_m_le_1024_SM90_CO_128x32x512_c211",
          (SM90_CO<128, 32, 512, 2, 1, 1>));
    } else if (m <= 4096) {
      // Optimized for prefill: seq_len up to 4096 (m=4096 with topk=1)
      TRACE_AND_INVOKE_GEMM(
          "n4096_k7168_m_le_4096_SM90_CO_128x64x512_c211",
          (SM90_CO<128, 64, 512, 2, 1, 1>));
    } else {
      // Optimized for prefill: seq_len up to 8192 (m=8192 with topk=1)
      TRACE_AND_INVOKE_GEMM(
          "n4096_k7168_m_gt_4096_SM90_CO_128x64x512_c111",
          (SM90_CO<128, 64, 512, 1, 1, 1>));
    }
  } else if (n == 7168 && k == 2048) {
    // group gemm 2
    if (m <= 8) {
      TRACE_AND_INVOKE_GEMM(
          "n7168_k2048_m_le_8_SM90_PP_64x16x512_c111",
          (SM90_PP<64, 16, 512, 1, 1, 1>));
    } else if (m <= 512) {
      TRACE_AND_INVOKE_GEMM(
          "n7168_k2048_m_le_512_SM90_CO_128x32x512_c111",
          (SM90_CO<128, 32, 512, 1, 1, 1>));
    } else if (m <= 4096) {
      // Optimized for prefill: larger cluster for better throughput
      TRACE_AND_INVOKE_GEMM(
          "n7168_k2048_m_le_4096_SM90_CO_128x64x512_c211",
          (SM90_CO<128, 64, 512, 2, 1, 1>));
    } else {
      TRACE_AND_INVOKE_GEMM(
          "n7168_k2048_m_gt_4096_SM90_CO_128x64x512_c111",
          (SM90_CO<128, 64, 512, 1, 1, 1>));
    }
  } else if (n == 512 && k == 7168) {
    // group gemm 1 for tp
    if (m <= 4) {
      TRACE_AND_INVOKE_GEMM(
          "n512_k7168_m_le_4_SM90_PP_64x32x512_c211",
          (SM90_PP<64, 32, 512, 2, 1, 1>));
    } else if (m <= 32) {
      TRACE_AND_INVOKE_GEMM(
          "n512_k7168_m_le_32_SM90_CO_128x16x512_c211",
          (SM90_CO<128, 16, 512, 2, 1, 1>));
    } else if (m <= 256) {
      TRACE_AND_INVOKE_GEMM(
          "n512_k7168_m_le_256_SM90_CO_128x16x512_c111",
          (SM90_CO<128, 16, 512, 1, 1, 1>));
    } else if (m <= 1024) {
      TRACE_AND_INVOKE_GEMM(
          "n512_k7168_m_le_1024_SM90_CO_128x32x512_c211",
          (SM90_CO<128, 32, 512, 2, 1, 1>));
    } else {
      TRACE_AND_INVOKE_GEMM(
          "n512_k7168_m_gt_1024_SM90_CO_128x64x512_c111",
          (SM90_CO<128, 64, 512, 1, 1, 1>));
    }
  } else if (n == 1024 && k == 4096) {
    // TP4 PD prefill/decode gemm1 hot path
    if (m <= 32) {
      TRACE_AND_INVOKE_GEMM(
          "n1024_k4096_m_le_32_SM90_CO_128x16x512_c111",
          (SM90_CO<128, 16, 512, 1, 1, 1>));
    } else if (m <= 1024) {
      TRACE_AND_INVOKE_GEMM(
          "n1024_k4096_m_le_1024_SM90_CO_128x32x512_c111",
          (SM90_CO<128, 32, 512, 1, 1, 1>));
    } else {
      TRACE_AND_INVOKE_GEMM(
          "n1024_k4096_m_gt_1024_SM90_CO_128x64x512_c111",
          (SM90_CO<128, 64, 512, 1, 1, 1>));
    }
  } else if (n == 512 && k == 4096) {
    // TP8 colocated gemm1 hot path
    if (m <= 32) {
      TRACE_AND_INVOKE_GEMM(
          "n512_k4096_m_le_32_SM90_CO_128x16x512_c111",
          (SM90_CO<128, 16, 512, 1, 1, 1>));
    } else if (m <= 1024) {
      TRACE_AND_INVOKE_GEMM(
          "n512_k4096_m_le_1024_SM90_CO_128x32x512_c111",
          (SM90_CO<128, 32, 512, 1, 1, 1>));
    } else {
      TRACE_AND_INVOKE_GEMM(
          "n512_k4096_m_gt_1024_SM90_CO_128x64x512_c111",
          (SM90_CO<128, 64, 512, 1, 1, 1>));
    }
  } else if (n == 4096 && k == 512) {
    // TP4 PD prefill/decode gemm2 hot path
    if (m <= 32) {
      TRACE_AND_INVOKE_GEMM(
          "n4096_k512_m_le_32_SM90_CO_128x16x512_c111",
          (SM90_CO<128, 16, 512, 1, 1, 1>));
    } else if (m <= 1024) {
      TRACE_AND_INVOKE_GEMM(
          "n4096_k512_m_le_1024_SM90_CO_128x32x512_c111",
          (SM90_CO<128, 32, 512, 1, 1, 1>));
    } else {
      TRACE_AND_INVOKE_GEMM(
          "n4096_k512_m_gt_1024_SM90_CO_128x64x512_c111",
          (SM90_CO<128, 64, 512, 1, 1, 1>));
    }
  } else if (n == 4096 && k == 256) {
    // TP8 colocated gemm2 hot path.
    if (m <= 8) {
      TRACE_AND_INVOKE_GEMM(
          "n4096_k256_m_le_8_SM90_PP_64x16x128_c111",
          (SM90_PP<64, 16, 128, 1, 1, 1>));
    } else if (m <= 32) {
      TRACE_AND_INVOKE_GEMM(
          "n4096_k256_m_le_32_SM90_PP_128x32x128_c111",
          (SM90_PP<128, 32, 128, 1, 1, 1>));
    } else {
      TRACE_AND_INVOKE_GEMM(
          "n4096_k256_m_gt_32_SM90_PP_128x64x128_c111",
          (SM90_PP<128, 64, 128, 1, 1, 1>));
    }
  } else if (n == 7168 && k == 256) {
    // group gemm 2 for tp
    if (m <= 8) {
      TRACE_AND_INVOKE_GEMM(
          "n7168_k256_m_le_8_SM90_PP_64x16x128_c111",
          (SM90_PP<64, 16, 128, 1, 1, 1>));
    } else if (m <= 32) {
      TRACE_AND_INVOKE_GEMM(
          "n7168_k256_m_le_32_SM90_PP_128x32x128_c111",
          (SM90_PP<128, 32, 128, 1, 1, 1>));
    } else if (m <= 512) {
      TRACE_AND_INVOKE_GEMM(
          "n7168_k256_m_le_512_SM90_PP_128x32x128_c211",
          (SM90_PP<128, 32, 128, 2, 1, 1>));
    } else {
      TRACE_AND_INVOKE_GEMM(
          "n7168_k256_m_gt_512_SM90_PP_128x64x128_c111",
          (SM90_PP<128, 64, 128, 1, 1, 1>));
    }
  } else {
    if (k % 512 == 0) {
      // For large m (prefill), prefer larger cluster
      if (m <= 32) {
        // Decode: target batch size (16-32) - use cluster size 1 for better latency
        TRACE_AND_INVOKE_GEMM(
            "fallback_k_mod_512_eq_0_m_le_32_SM90_CO_128x16x512_c111",
            (SM90_CO<128, 16, 512, 1, 1, 1>));
      } else if (m <= 1024) {
        // Decode: large batch or small prefill
        TRACE_AND_INVOKE_GEMM(
            "fallback_k_mod_512_eq_0_m_le_1024_SM90_CO_128x32x512_c111",
            (SM90_CO<128, 32, 512, 1, 1, 1>));
      } else {
        // Prefill: large sequence length - prefer larger cluster
        TRACE_AND_INVOKE_GEMM(
            "fallback_k_mod_512_eq_0_m_gt_1024_SM90_CO_128x64x512_c111",
            (SM90_CO<128, 64, 512, 1, 1, 1>));
      }
    } else {
      if (m <= 32) {
        // Decode: target batch size (16-32) - use larger tile for better throughput
        TRACE_AND_INVOKE_GEMM(
            "fallback_k_mod_512_ne_0_m_le_32_SM90_PP_128x32x128_c111",
            (SM90_PP<128, 32, 128, 1, 1, 1>));
      } else {
        // Prefill: larger sequence length
        TRACE_AND_INVOKE_GEMM(
            "fallback_k_mod_512_ne_0_m_gt_32_SM90_PP_128x64x128_c111",
            (SM90_PP<128, 64, 128, 1, 1, 1>));
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
