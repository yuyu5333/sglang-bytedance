// KVBit MLA decode attention — stage-1 (per-split) CUDA kernel.
//
// Replaces the Triton _mla_decode_kernel. Same ABI/contract: one program per
// (batch, head, split), produces a NORMALIZED per-split output (acc / e_sum)
// + logsum-exp in att_out/att_lse, which the host-side _reduce_splits merges
// across splits via log-sum-exp.
//
// Why CUDA (not Triton): the Triton kernel was 5.6x-29x slower than fa3 at
// bs=1 (13.7ms vs 2.8ms for 78 layers). bs=1 decode is a GEMV (M=1) so tensor
// cores (tl.dot/WGMMA) do NOT help — verified: the tl.dot variant regressed.
// fa3's speed comes from memory-path efficiency (vectorized loads, KV read
// once, online softmax in registers, software pipelining), NOT tensor cores.
// This kernel mirrors that: vectorized 128-bit loads of the 4bit packed KV,
// fused in-register dequant (no smem scratch for K), warp-parallel qk GEMV,
// online softmax + V accumulation in registers, split-K parallelism. The 4bit
// KV reads 288 B/token vs fa3's bf16 1152 B/token = 4x less HBM, so a
// bandwidth-bound decode should beat fa3 here.
//
// Layout (kvbit no_alloc, MLA absorbed, Q-FHT):
//   kvbit_packed  (num_slots, 288) uint8 — [256 code bytes | 32 header bytes]
//     per token: 8 groups, each 32 code bytes (64 4-bit codes) + 4 header
//     bytes (2 fp16: min, range). group g codes at bytes [g*32, g*32+32),
//     header (min,range) at bytes [256 + g*4, 256 + g*4 + 4).
//   rope_buffer   (num_slots, 1, 64) bf16 — raw RoPE tail (position-dependent,
//     not rotated).
//   q_nope        (batch, n_heads, 512) bf16 — Q-FHT folded (q@R).
//   q_pe          (batch, n_heads, 64) bf16.
//   page_table    (batch, max_len) int32 — slot indices per seq (page_size=1).
//   cache_seqlens (batch,) int32 — valid KV length per seq.
//   num_kv_splits (batch,) int32 — per-seq split count; split i owns
//     [i*ceil(seqlen/kv_splits), (i+1)*ceil(seqlen/kv_splits)).
//
// BLOCK_N = 64 KV tokens per CTA iteration. Each CTA has kWarps warps; the
// BLOCK_N KV tokens are split across warps for the qk GEMV, and the 512-dim V
// accumulator is sharded across threads.

#include <sgl_kernel/tensor.h>    // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>     // For RuntimeCheck, div_ceil
#include <sgl_kernel/utils.cuh>   // For LaunchKernel, SGL_DEVICE, bf16_t, fp32_t
#include <sgl_kernel/warp.cuh>    // For warp::reduce_sum, warp::reduce_max
#include <sgl_kernel/math.cuh>    // For device::math::exp/max

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cstdint>

namespace {

// 4bit packed row layout constants (kvbit no_alloc, rope stored separately).
constexpr int32_t kCodeBytesPerGroup = 32;   // 64 codes / 4 bits
constexpr int32_t kHeaderBytesPerGroup = 4;  // 2 x fp16 (min, range)
constexpr int32_t kNGroups = 8;              // 512 / 64
constexpr int32_t kCodeBytes = kNGroups * kCodeBytesPerGroup;      // 256
constexpr int32_t kHeaderBytes = kNGroups * kHeaderBytesPerGroup;  // 32
constexpr int32_t kRowBytes = kCodeBytes + kHeaderBytes;           // 288
constexpr int32_t kGroupSize = 64;
constexpr int32_t kLevels = 15;  // (1<<4)-1
constexpr int32_t kDnope = 512;  // kv_lora_rank
constexpr int32_t kDrope = 64;   // qk_rope_head_dim
constexpr int32_t kBlockN = 64;  // KV tokens per iteration

// Dequantize one 4-bit code pair from a byte: low nibble, high nibble.
SGL_DEVICE __forceinline__ void unpack_byte(uint8_t b, float& lo, float& hi) {
  lo = static_cast<float>(b & 0xF);
  hi = static_cast<float>((b >> 4) & 0xF);
}

// Load + dequantize group g for `count` KV tokens (count <= kBlockN), starting
// at slot indices kv_loc[0..count). Writes k_g[0..count*64) fp32.
// Each thread handles a stride of tokens; within a token, all 64 codes of the
// group are dequantized by the threads of the token's lane group.
template <int kBlockSize>
SGL_DEVICE void load_dequant_group(const uint8_t* __restrict__ kvbit_packed,
                                   const int32_t* __restrict__ kv_loc,
                                   int count, int g, float* k_g) {
  // k_g is (kBlockN, 64) in shared/register; thread t owns token t (one token
  // per thread when kBlockSize >= kBlockN). We use a 1-token-per-thread tile.
  const int tid = threadIdx.x;
  // 32 code bytes / 4 bytes vec = 8 vec loads per token; or do scalar nibble
  // unpack. For simplicity + correctness first, scalar unpack per thread.
  if (tid < count) {
    const int32_t slot = kv_loc[tid];
    const uint8_t* row = kvbit_packed + (int64_t)slot * kRowBytes;
    const uint8_t* codes = row + g * kCodeBytesPerGroup;
    // header (min fp16, range fp16) at row + 256 + g*4
    const uint16_t h0 = *reinterpret_cast<const uint16_t*>(row + kCodeBytes + g * 4);
    const uint16_t h1 = *reinterpret_cast<const uint16_t*>(row + kCodeBytes + g * 4 + 2);
    __nv_bfloat16 gmin_bf = *reinterpret_cast<const __nv_bfloat16*>(&h0);
    __nv_bfloat16 grange_bf = *reinterpret_cast<const __nv_bfloat16*>(&h1);
    float gmin = __bfloat162float(gmin_bf);
    float grange = __bfloat162float(grange_bf);
    grange = fmaxf(grange, 1e-8f);
    float step = grange / static_cast<float>(kLevels);
    float* out = k_g + (int64_t)tid * kGroupSize;
    #pragma unroll
    for (int j = 0; j < kCodeBytesPerGroup; ++j) {
      float lo, hi;
      unpack_byte(codes[j], lo, hi);
      out[2 * j] = lo * step + gmin;
      out[2 * j + 1] = hi * step + gmin;
    }
  }
}

template <bool kUsePDL>
__global__ void kvbit_mla_decode_stage1_kernel(
    const __nv_bfloat16* __restrict__ Q_NOPE,    // (B, H, 512)
    const __nv_bfloat16* __restrict__ Q_PE,      // (B, H, 64)
    const uint8_t* __restrict__ KVBIT_PACKED,    // (num_slots, 288)
    const __nv_bfloat16* __restrict__ ROPE_BUF,  // (num_slots, 1, 64)
    float sm_scale,
    const int32_t* __restrict__ PAGE_TABLE,      // (B, max_len)
    const int32_t* __restrict__ CACHE_SEQLENS,   // (B,)
    const int32_t* __restrict__ NUM_KV_SPLITS,   // (B,)
    float* __restrict__ ATT_OUT,                 // (B, H, max_splits, 512)
    float* __restrict__ ATT_LSE,                 // (B, H, max_splits)
    int32_t batch, int32_t n_heads, int32_t max_len, int32_t max_splits,
    int32_t stride_qn_b, int32_t stride_qn_h,
    int32_t stride_qp_b, int32_t stride_qp_h,
    int64_t stride_pk_b,
    int32_t stride_pt_b, int32_t stride_pt_n,
    int32_t stride_o_b, int32_t stride_o_h, int32_t stride_o_s) {
  const int cur_batch = blockIdx.x;
  const int cur_head = blockIdx.y;
  const int split_kv_id = blockIdx.z;

  device::PDLWaitPrimary<kUsePDL>();

  const int32_t seq_len = CACHE_SEQLENS[cur_batch];
  const int32_t kv_splits = NUM_KV_SPLITS[cur_batch];
  // empty / out-of-range split: leave att_out=0, att_lse=-inf (host pre-fills).
  if (split_kv_id >= kv_splits) return;

  const int32_t kv_len_per_split = (seq_len + kv_splits - 1) / kv_splits;
  const int32_t split_start = kv_len_per_split * split_kv_id;
  const int32_t split_end = min(split_start + kv_len_per_split, seq_len);
  if (split_end <= split_start) return;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;
  constexpr int kBlockSize = 256;  // 8 warps
  constexpr int kWarps = kBlockSize / 32;

  // Load query (Q-FHT folded nope + rope) into registers — shared across the
  // CTA. q_nope (512), q_pe (64). Shard across threads: each thread holds
  // ceil(512/kBlockSize) elements. With 256 threads, 2 elements/thread for
  // nope, 0-1 for rope. Simpler: load full q into shared memory.
  __shared__ float q_nope_sm[kDnope];   // 512 fp32 = 2KB
  __shared__ float q_pe_sm[kDrope];     // 64 fp32 = 256B
  __shared__ float k_g_sm[kBlockN * kGroupSize];  // 64*64 fp32 = 16KB (one group at a time)
  __shared__ float qk_sm[kBlockN];      // scores (64) = 256B
  __shared__ float p_sm[kBlockN];       // softmax weights (64) = 256B
  // V accumulator: 512 fp32, sharded across threads in shared.
  __shared__ float acc_sm[kDnope];      // 2KB

  // Load q_nope (512) and q_pe (64) into smem (bf16 -> fp32).
  for (int i = tid; i < kDnope; i += kBlockSize) {
    q_nope_sm[i] = __bfloat162float(
        Q_NOPE[cur_batch * stride_qn_b + cur_head * stride_qn_h + i]);
  }
  for (int i = tid; i < kDrope; i += kBlockSize) {
    q_pe_sm[i] = __bfloat162float(
        Q_PE[cur_batch * stride_qp_b + cur_head * stride_qp_h + i]);
  }
  // Init acc.
  for (int i = tid; i < kDnope; i += kBlockSize) acc_sm[i] = 0.0f;
  __syncthreads();

  float e_max = -INFINITY;
  float e_sum = 0.0f;

  // KV slot indices for the current chunk (kBlockN tokens).
  __shared__ int32_t kv_loc_sm[kBlockN];

  for (int start_n = split_start; start_n < split_end; start_n += kBlockN) {
    const int count = min(kBlockN, split_end - start_n);
    // Load kv_loc for this chunk.
    if (tid < count) {
      kv_loc_sm[tid] =
          PAGE_TABLE[cur_batch * stride_pt_b + (start_n + tid) * stride_pt_n];
    }
    __syncthreads();

    // --- qk_rope: q_pe @ k_rope^T  -> (kBlockN,) ---
    // Each thread computes one (or more) token's rope score.
    // token t: sum_{d} q_pe[d] * k_rope[t, d]. k_rope from ROPE_BUF[slot, 0, d].
    if (tid < count) {
      const int32_t slot = kv_loc_sm[tid];
      const __nv_bfloat16* k_rope = ROPE_BUF + (int64_t)slot * kDrope;
      float s = 0.0f;
      #pragma unroll
      for (int d = 0; d < kDrope; ++d) {
        s += q_pe_sm[d] * __bfloat162float(k_rope[d]);
      }
      qk_sm[tid] = s;
    } else {
      if (tid < kBlockN) qk_sm[tid] = 0.0f;
    }
    __syncthreads();

    // --- qk_nope: sum over 8 groups of (q_nope_g @ k_g^T) -> (kBlockN,) ---
    // For each group: load+dequant k_g (kBlockN, 64) into k_g_sm, then each
    // thread (one per token) computes dot(q_nope[g*64:(g+1)*64], k_g[t,:]).
    for (int g = 0; g < kNGroups; ++g) {
      load_dequant_group<kBlockSize>(KVBIT_PACKED, kv_loc_sm, count, g, k_g_sm);
      __syncthreads();
      if (tid < count) {
        const float* kg = k_g_sm + (int64_t)tid * kGroupSize;
        const float* qg = q_nope_sm + g * kGroupSize;
        float s = 0.0f;
        #pragma unroll
        for (int d = 0; d < kGroupSize; ++d) s += qg[d] * kg[d];
        qk_sm[tid] += s;
      }
      __syncthreads();
    }

    // scale + mask
    if (tid < count) {
      float qk = qk_sm[tid] * sm_scale;
      qk_sm[tid] = qk;
    } else if (tid < kBlockN) {
      qk_sm[tid] = -INFINITY;
    }
    __syncthreads();

    // --- online softmax (across kBlockN scores) ---
    // block-wide max via warp reduce + smem.
    float local_max = -INFINITY;
    if (tid < kBlockN) local_max = qk_sm[tid];
    float warp_max = device::warp::reduce_max(local_max);
    __shared__ float warp_max_sm[kWarps];
    if (lane == 0) warp_max_sm[wid] = warp_max;
    __syncthreads();
    float n_e_max = (tid < kWarps) ? warp_max_sm[tid] : -INFINITY;
    // reduce across kWarps by thread 0
    if (tid == 0) {
      for (int w = 1; w < kWarps; ++w) n_e_max = fmaxf(n_e_max, warp_max_sm[w]);
      warp_max_sm[0] = n_e_max;
    }
    __syncthreads();
    n_e_max = warp_max_sm[0];
    n_e_max = fmaxf(n_e_max, e_max);

    const float re_scale = expf(e_max - n_e_max);
    // compute p = exp(qk - n_e_max), accumulate e_sum, rescale acc.
    if (tid < kBlockN) {
      float p = (tid < count) ? expf(qk_sm[tid] - n_e_max) : 0.0f;
      p_sm[tid] = p;
    }
    // rescale acc
    for (int i = tid; i < kDnope; i += kBlockSize) acc_sm[i] *= re_scale;
    __syncthreads();

    // --- V accumulation: acc += p @ k_nope (V == k_nope_rot, absorbed) ---
    // For each group: reload k_g (same dequant), acc[d in group] += sum_t p[t]*k_g[t,d].
    // Shard the 64 group-dim across threads: 256 threads / 64 dims = 4 threads
    // per dim; reduce within the 4. Simpler: each thread handles one (token, dim)
    // partial and we reduce. Use: for each group, 64 dims, thread t handles dim
    // (t % 64) over a subset of tokens? Cleaner: one warp owns the whole group's
    // 64-dim reduce (32 threads do 2 dims each), loop tokens in the warp.
    for (int g = 0; g < kNGroups; ++g) {
      load_dequant_group<kBlockSize>(KVBIT_PACKED, kv_loc_sm, count, g, k_g_sm);
      __syncthreads();
      // acc_g[d] = sum_t p[t] * k_g[t, d], for d in [0,64).
      // 256 threads, 64 dims -> 4 threads per dim. Thread (wid, lane):
      //   dim = lane (0..31) for wid 0,1 ; but 64 dims need 2 warps to cover.
      // Use: 2 warps (64 threads) do the 64 dims, one dim per thread; the other
      // 6 warps idle for this group's V. (Inefficient but correct first.)
      if (wid < 2) {
        int d = wid * 32 + lane;  // 0..63
        if (d < kGroupSize) {
          float acc_g = 0.0f;
          for (int t = 0; t < count; ++t) {
            acc_g += p_sm[t] * k_g_sm[t * kGroupSize + d];
          }
          atomicAdd(&acc_sm[g * kGroupSize + d], acc_g);
        }
      }
      __syncthreads();
    }

    // update e_sum, e_max
    if (tid < kBlockN) {
      // e_sum = e_sum * re_scale + sum(p) ; but we already rescaled acc above.
    }
    // block-wide sum of p
    float local_p = (tid < kBlockN) ? p_sm[tid] : 0.0f;
    float warp_sum = device::warp::reduce_sum(local_p);
    __shared__ float warp_sum_sm[kWarps];
    if (lane == 0) warp_sum_sm[wid] = warp_sum;
    __syncthreads();
    if (tid == 0) {
      float ps = 0.0f;
      for (int w = 0; w < kWarps; ++w) ps += warp_sum_sm[w];
      e_sum = e_sum * re_scale + ps;
      warp_sum_sm[0] = e_sum;  // reuse as broadcast
    }
    __syncthreads();
    e_sum = warp_sum_sm[0];
    e_max = n_e_max;
    __syncthreads();
  }

  // store mid output (normalized) + lse.
  if (split_end > split_start) {
    for (int i = tid; i < kDnope; i += kBlockSize) {
      ATT_OUT[cur_batch * stride_o_b + cur_head * stride_o_h +
              split_kv_id * stride_o_s + i] = acc_sm[i] / e_sum;
    }
    if (tid == 0) {
      ATT_LSE[(cur_batch * n_heads + cur_head) * max_splits + split_kv_id] =
          e_max + logf(e_sum);
    }
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
struct KvbitMlaDecodeStage1Kernel {
  static constexpr auto kernel = kvbit_mla_decode_stage1_kernel<kUsePDL>;

  static void run(tvm::ffi::TensorView q_nope, tvm::ffi::TensorView q_pe,
                  tvm::ffi::TensorView kvbit_packed, tvm::ffi::TensorView rope_buf,
                  double sm_scale, tvm::ffi::TensorView page_table,
                  tvm::ffi::TensorView cache_seqlens,
                  tvm::ffi::TensorView num_kv_splits,
                  tvm::ffi::TensorView att_out, tvm::ffi::TensorView att_lse,
                  int32_t max_splits) {
    using namespace host;

    auto B = SymbolicSize{"batch"};
    auto H = SymbolicSize{"n_heads"};
    auto L = SymbolicSize{"max_len"};
    auto S = SymbolicSize{"num_slots"};
    auto Dn = SymbolicSize{"kv_lora_rank"};
    auto Dr = SymbolicSize{"qk_rope_head_dim"};
    auto Ms = SymbolicSize{"max_splits"};
    // strides (in elements)
    auto Sqn_b = SymbolicSize{"stride_qn_b"};
    auto Sqn_h = SymbolicSize{"stride_qn_h"};
    auto Sqp_b = SymbolicSize{"stride_qp_b"};
    auto Sqp_h = SymbolicSize{"stride_qp_h"};
    auto Spk_b = SymbolicSize{"stride_pk_b"};
    auto Spt_b = SymbolicSize{"stride_pt_b"};
    auto Spt_n = SymbolicSize{"stride_pt_n"};
    auto So_b = SymbolicSize{"stride_o_b"};
    auto So_h = SymbolicSize{"stride_o_h"};
    auto So_s = SymbolicSize{"stride_o_s"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B, H, Dn})
        .with_strides({Sqn_b, Sqn_h, 1})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device_)
        .verify(q_nope);
    TensorMatcher({B, H, Dr})
        .with_strides({Sqp_b, Sqp_h, 1})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device_)
        .verify(q_pe);
    TensorMatcher({S, -1})
        .with_strides({Spk_b, 1})
        .with_dtype<uint8_t>()
        .with_device<kDLCUDA>(device_)
        .verify(kvbit_packed);
    TensorMatcher({S, 1, Dr})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device_)
        .verify(rope_buf);
    TensorMatcher({B, L})
        .with_strides({Spt_b, Spt_n})
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device_)
        .verify(page_table);
    TensorMatcher({B}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(cache_seqlens);
    TensorMatcher({B}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(num_kv_splits);
    TensorMatcher({B, H, Ms, Dn})
        .with_strides({So_b, So_h, So_s, 1})
        .with_dtype<fp32_t>()
        .with_device<kDLCUDA>(device_)
        .verify(att_out);
    TensorMatcher({B, H, Ms})
        .with_dtype<fp32_t>()
        .with_device<kDLCUDA>(device_)
        .verify(att_lse);

    const auto batch = static_cast<int32_t>(B.unwrap());
    const auto n_heads = static_cast<int32_t>(H.unwrap());
    const auto max_len = static_cast<int32_t>(L.unwrap());
    const DLDevice dev = device_.unwrap();

    constexpr int32_t kBlockSize = 256;
    dim3 grid(batch, n_heads, max_splits);
    LaunchKernel(grid, kBlockSize, dev).enable_pdl(kUsePDL)(
        kernel, static_cast<const bf16_t*>(q_nope.data_ptr()),
        static_cast<const bf16_t*>(q_pe.data_ptr()),
        static_cast<const uint8_t*>(kvbit_packed.data_ptr()),
        static_cast<const bf16_t*>(rope_buf.data_ptr()),
        static_cast<float>(sm_scale),
        static_cast<const int32_t*>(page_table.data_ptr()),
        static_cast<const int32_t*>(cache_seqlens.data_ptr()),
        static_cast<const int32_t*>(num_kv_splits.data_ptr()),
        static_cast<float*>(att_out.data_ptr()),
        static_cast<float*>(att_lse.data_ptr()),
        batch, n_heads, max_len, max_splits,
        static_cast<int32_t>(Sqn_b.unwrap()), static_cast<int32_t>(Sqn_h.unwrap()),
        static_cast<int32_t>(Sqp_b.unwrap()), static_cast<int32_t>(Sqp_h.unwrap()),
        static_cast<int64_t>(Spk_b.unwrap()),
        static_cast<int32_t>(Spt_b.unwrap()), static_cast<int32_t>(Spt_n.unwrap()),
        static_cast<int32_t>(So_b.unwrap()), static_cast<int32_t>(So_h.unwrap()),
        static_cast<int32_t>(So_s.unwrap()));
  }
};

}  // namespace
