#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>

#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "defines.h"
#include "params.h"

using namespace cute;

namespace sm90::decode::sparse_fp8 {

template <ModelType MODEL_TYPE, int NUM_HEADS>
class KernelTemplate {
 public:
  static_assert(NUM_HEADS == 64 || NUM_HEADS == 128);
  static constexpr bool PACKED_INT4 = true;
  static constexpr int NUM_M_BLOCKS = NUM_HEADS / 64;
  static constexpr int CLUSTER_SIZE = NUM_M_BLOCKS;

  static constexpr int HEAD_DIM_K = MODEL_TYPE == ModelType::V32 ? 576 : 512;
  static constexpr int HEAD_DIM_V = 512;
  static constexpr int HEAD_DIM_ROPE = 64;
  static constexpr int HEAD_DIM_NOPE = HEAD_DIM_K - HEAD_DIM_ROPE;

  static constexpr int QUANT_TILE_SIZE = MODEL_TYPE == ModelType::V32 ? 128 : 64;
  static constexpr int NUM_SCALES = MODEL_TYPE == ModelType::V32 ? 4 : 8;  // For MODEL1: 7 fp8_e4m3 + 1 padding

  static constexpr int NUM_THREADS = 128 * 3;
  static constexpr int BLOCK_M = 64;
  static constexpr int TOPK_BLOCK_SIZE = 64;
  static constexpr int NUM_K_BUFS = 2;

  using SmemLayoutQTile =
      decltype(tile_to_shape(GMMA::Layout_SW128_Atom<bf16, GMMA::Major::K>{}, Shape<Int<BLOCK_M>, Int<64>>{}));

  template <int NUM_TILES>
  using SmemLayoutQTiles =
      decltype(tile_to_shape(SmemLayoutQTile{}, Shape<Int<BLOCK_M>, Int<64 * NUM_TILES>>{}, Step<_1, _2>{}));

  using SmemLayoutQ = SmemLayoutQTiles<HEAD_DIM_K / 64>;

  using SmemLayoutKTile = decltype(tile_to_shape(
      GMMA::Layout_INTER_Atom<bf16, GMMA::Major::K>{}, Shape<Int<TOPK_BLOCK_SIZE>, _64>{}, Step<_1, _2>{}));

  using SmemLayoutXTile =
      decltype(tile_to_shape(GMMA::Layout_SW128_Atom<bf16, GMMA::Major::K>{}, Shape<Int<TOPK_BLOCK_SIZE>, _64>{}));

  template <int NUM_TILES>
  using SmemLayoutKTiles =
      decltype(tile_to_shape(SmemLayoutKTile{}, Shape<Int<TOPK_BLOCK_SIZE>, Int<64 * NUM_TILES>>{}, Step<_1, _2>{}));

  template <int NUM_TILES>
  using SmemLayoutKTilesTransposed = decltype(composition(
      SmemLayoutKTiles<NUM_TILES>{},
      Layout<Shape<Int<64 * NUM_TILES>, Int<TOPK_BLOCK_SIZE>>, Stride<Int<TOPK_BLOCK_SIZE>, _1>>{}));

  static constexpr int OBUF_SW = 64;
  using SmemLayoutOBufAtom = GMMA::Layout_K_SW128_Atom<bf16>;
  using SmemLayoutOBuf =
      decltype(tile_to_shape(SmemLayoutOBufAtom{}, Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{}, Step<_1, _2>{}));

  using SmemLayoutOAccumBuf = Layout<
      Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>,
      Stride<Int<520>, _1>  // We use stride = 520 here to avoid bank conflict
      >;

  using SmemLayoutK = SmemLayoutKTiles<HEAD_DIM_K / 64>;
  using SmemLayoutV = SmemLayoutKTilesTransposed<HEAD_DIM_V / 64>;
  using SmemLayoutHalfV = SmemLayoutKTilesTransposed<HEAD_DIM_V / 64 / 2>;

  using SmemLayoutS =
      decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{}));

  struct SharedMemoryPlan {
    array_aligned<bf16, cosize_v<SmemLayoutQ>> q;
    union {
      array_aligned<bf16, cosize_v<SmemLayoutK>> k[NUM_K_BUFS];
      array_aligned<bf16, cosize_v<SmemLayoutOBuf>> oBuf;
      array_aligned<float, cosize_v<SmemLayoutOAccumBuf>> oAccumBuf;
    } u;
    CUTE_ALIGNAS(1024) array_aligned<bf16, cosize_v<SmemLayoutS>> s;
    bool is_kv_valid[NUM_K_BUFS][TOPK_BLOCK_SIZE];

    float sM[BLOCK_M], sL[BLOCK_M], sScale[BLOCK_M], sOScale[BLOCK_M];
    transac_bar_t bar_q, bar_k_local_ready[NUM_K_BUFS], bar_k_remote_ready[NUM_K_BUFS], bar_k_avail[NUM_K_BUFS];
  };
  static_assert(sizeof(SharedMemoryPlan) <= 208896, "INT4 FlashMLA shared-memory plan regressed");

  template <typename Shape_Q, typename TMA_Q>
  struct TmaParams {
    Shape_Q shape_Q;
    TMA_Q tma_Q;
    CUtensorMap tensor_map_o;
  };

  using TiledMMA_QK = decltype(make_tiled_mma(
      GMMA::MMA_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{}, Layout<Shape<_1, _1, _1>>{}));

  using TiledMMA_QK_rQ = decltype(make_tiled_mma(
      GMMA::MMA_64x64x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::K>{}, Layout<Shape<_1, _1, _1>>{}));

  using TiledMMA_PV_LocalP = decltype(make_tiled_mma(
      GMMA::MMA_64x256x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}, Layout<Shape<_1, _1, _1>>{}));

  using TiledMMA_PV_RemoteP = decltype(make_tiled_mma(
      GMMA::MMA_64x256x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::MN>{}, Layout<Shape<_1, _1, _1>>{}));

  enum NamedBarriers : uint32_t {
    sScale_and_sS_ready = 0,
    sScale_and_sS_free = 1,
    oBuf_free_and_sL_ready = 2,
    epilogue_r2s_ready = 3,
    batch_loop_sync = 4,
    warpgroup0_sync = 5,
    packed_kv_producer_sync = 6
  };

  // Synchronize all threads within the cluster (which processes one q token)
  static __forceinline__ __device__ void sync_all_threads_in_cluster() {
    if constexpr (CLUSTER_SIZE == 1) {
      __syncthreads();
    } else {
      ku::barrier_cluster_arrive_relaxed();
      ku::barrier_cluster_wait_acquire();
    }
  }

  // Save rPb (64x64, bfloat16) to sP using the stmatrix instruction
  template <typename Tensor0, typename Tensor1>
  static __forceinline__ __device__ void save_rPb_to_sP(Tensor0 const& rPb, Tensor1 const& sP, int idx_in_warpgroup) {
    auto r2s_copy = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, bf16>{}, TiledMMA_QK{});
    ThrCopy thr_copy = r2s_copy.get_slice(idx_in_warpgroup);
    Tensor thr_copy_rPb = thr_copy.retile_S(rPb);
    Tensor thr_copy_sP = thr_copy.partition_D(sP);
    cute::copy(r2s_copy, thr_copy_rPb, thr_copy_sP);
  }

  template <
      bool IS_NO_SPLIT,
      typename TMAParams,
      typename Tensor0,
      typename Tensor1,
      typename Tensor2,
      typename Tensor3>
  static __forceinline__ __device__ void store_o(
      Tensor0& rO,         // ((2, 2, 32), 1, 1)
      Tensor1& gOorAccum,  // (BLOCK_SIZE_M, HEAD_DIM_V)
      Tensor2& sOutputBuf,
      Tensor3& sOutputAccumBuf,
      SharedMemoryPlan& plan,
      float o_scales[2],
      TMAParams& tma_params,
      int batch_idx,
      int s_q_idx,
      int head_block_idx,
      int num_valid_seq_q,
      int warpgroup_idx,
      int idx_in_warpgroup) {
    using cutlass::arch::NamedBarrier;
    // Both consumer warpgroups first materialize their 256-column FP32
    // fragments in the common accumulator buffer. This gives WG0 a complete
    // logical O row on which to move the H256 transform out of the K producer.
    CUTLASS_PRAGMA_UNROLL
    for (int idx = 0; idx < size(rO); idx += 2) {
      int row = (idx_in_warpgroup / 32) * 16 + (idx_in_warpgroup % 32 / 4) + (idx % 4 >= 2 ? 8 : 0);
      int col = warpgroup_idx * 256 + (idx_in_warpgroup % 4) * 2 + idx / 4 * 8;
      *(float2*)(&(sOutputAccumBuf(row, col))) = float2{
          rO(idx) * o_scales[idx % 4 >= 2],
          rO(idx + 1) * o_scales[idx % 4 >= 2],
      };
    }
    cutlass::arch::fence_view_async_shared();
    NamedBarrier::arrive_and_wait(256, NamedBarriers::epilogue_r2s_ready);

    if (warpgroup_idx == 0) {
      const int warp = idx_in_warpgroup >> 5;
      const int lane = idx_in_warpgroup & 31;
      CUTE_UNROLL
      for (int round = 0; round < 16; ++round) {
        const int row = warp * 16 + round;
        float values[8];
        CUTE_UNROLL
        for (int j = 0; j < 8; ++j) {
          values[j] = sOutputAccumBuf(row, lane * 8 + j);
        }
        CUTE_UNROLL
        for (int span = 1; span < 8; span <<= 1) {
          CUTE_UNROLL
          for (int base = 0; base < 8; base += span << 1) {
            CUTE_UNROLL
            for (int j = 0; j < span; ++j) {
              const float a = values[base + j];
              const float b = values[base + span + j];
              values[base + j] = a + b;
              values[base + span + j] = a - b;
            }
          }
        }
        CUTE_UNROLL
        for (int mask = 1; mask < 32; mask <<= 1) {
          CUTE_UNROLL
          for (int j = 0; j < 8; ++j) {
            const float other = __shfl_xor_sync(0xffffffffu, values[j], mask);
            values[j] = (lane & mask) ? other - values[j] : values[j] + other;
          }
        }
        CUTE_UNROLL
        for (int j = 0; j < 8; ++j) {
          sOutputAccumBuf(row, lane * 8 + j) = values[j] * 0.0625f;
        }
      }
    }

    cutlass::arch::fence_view_async_shared();
    NamedBarrier::arrive_and_wait(256, NamedBarriers::warpgroup0_sync);
    if (warpgroup_idx != 0) return;

    if constexpr (IS_NO_SPLIT) {
      // Convert the complete, transformed FP32 tile to global BF16 only
      // after H256. Do not reuse the aliased sOutputBuf here: its BF16 stores
      // would clobber unread FP32 values in sOutputAccumBuf.
      for (int linear_idx = idx_in_warpgroup; linear_idx < num_valid_seq_q * HEAD_DIM_V; linear_idx += 128) {
        const int row = linear_idx / HEAD_DIM_V;
        const int col = linear_idx % HEAD_DIM_V;
        gOorAccum(row, col) = bf16(sOutputAccumBuf(row, col));
      }
    } else if (elect_one_sync()) {
      // WG0's four warp leaders cover all rows; split output remains FP32.
      for (int row = idx_in_warpgroup / 32; row < num_valid_seq_q; row += 4) {
        SM90_BULK_COPY_S2G::copy(&sOutputAccumBuf(row, _0{}), &gOorAccum(row, _0{}), HEAD_DIM_V * sizeof(float));
      }
      cute::tma_store_arrive();
    }
  }

  template <typename TMAParams>
  static __device__ __forceinline__ void devfunc(const SparseAttnDecodeParams& params, const TMAParams& tma_params);

  static void run_impl(const SparseAttnDecodeParams& params);

  static void run_int4(const SparseAttnDecodeParams& params);
};

}  // namespace sm90::decode::sparse_fp8
