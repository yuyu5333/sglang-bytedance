#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cstdint>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

#include "diff_kernels.h"
#include "utils.h"

namespace {

constexpr int64_t kMaxHotBufferPages = 256;
constexpr int kBlockSize = 256;

template <typename T>
__device__ __forceinline__ int64_t to_i64(T v) {
  return static_cast<int64_t>(v);
}

template <>
__device__ __forceinline__ int64_t to_i64<bool>(bool v) {
  return v ? 1 : 0;
}

// Vectorized load helper
template <typename T, int N>
struct VecLoad {
    using Type = T;
};

template <>
struct VecLoad<int32_t, 4> {
    using Type = int4;
};

template <>
struct VecLoad<int64_t, 2> {
    using Type = int4;
};

template <typename T>
__device__ __forceinline__ void load_shared_vectorized(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int count,
    int tid) {
    
    // Try 128-bit load for int32
    if constexpr (std::is_same_v<T, int32_t>) {
        if (reinterpret_cast<uintptr_t>(src) % 16 == 0 && count % 4 == 0) {
            const int4* src_vec = reinterpret_cast<const int4*>(src);
            int4* dst_vec = reinterpret_cast<int4*>(dst);
            int vec_count = count / 4;
            for (int i = tid; i < vec_count; i += kBlockSize) {
                dst_vec[i] = src_vec[i];
            }
            // Handle remaining part if BlockSize > vec_count? 
            // Here we assume count <= 256, so vec_count <= 64. 
            // 256 threads cover it easily.
            return;
        }
    }
    
    // Try 128-bit load for int64
    if constexpr (std::is_same_v<T, int64_t>) {
        if (reinterpret_cast<uintptr_t>(src) % 16 == 0 && count % 2 == 0) {
            const int4* src_vec = reinterpret_cast<const int4*>(src);
            int4* dst_vec = reinterpret_cast<int4*>(dst);
            int vec_count = count / 2;
            for (int i = tid; i < vec_count; i += kBlockSize) {
                dst_vec[i] = src_vec[i];
            }
            return;
        }
    }

    // Fallback to scalar load
    for (int i = tid; i < count; i += kBlockSize) {
        dst[i] = src[i];
    }
}

template <typename DiffT, typename PageOutT, typename SparseMaskT>
__global__ void sparse_page_wise_diff_kernel(
    int64_t* __restrict__ last_top_k_idx,
    const int32_t* __restrict__ top_k_idx,
    int64_t* __restrict__ last_page_ids,
    PageOutT* __restrict__ page_ids,
    DiffT* __restrict__ diff_map, // Unused in optimized version
    const int64_t* __restrict__ req_to_tokens_host,
    int64_t* __restrict__ load_tokens,
    int64_t* __restrict__ load_tokens_host,
    const int64_t* __restrict__ seq_lens,
    const int64_t* __restrict__ req_pool_indices,
    const SparseMaskT* __restrict__ sparse_mask,
    const int32_t* __restrict__ page_table,
    int64_t last_top_k_s0,
    int64_t last_top_k_s1,
    int64_t top_k_s,
    int64_t last_page_ids_s0,
    int64_t last_page_ids_s1,
    int64_t page_ids_s,
    int64_t diff_map_s,
    int64_t req_to_tokens_host_s,
    int64_t load_tokens_s,
    int64_t load_tokens_host_s,
    int64_t page_table_s,
    int64_t layer_id,
    int64_t top_k,
    int64_t top_k_page,
    int64_t hot_buffer_len,
    int64_t hot_buffer_page,
    int64_t page_size) {
  
  const int64_t bid = static_cast<int64_t>(blockIdx.x);
  const int64_t tid = static_cast<int64_t>(threadIdx.x);

  // Base pointers
  int64_t* load_tokens_base = load_tokens + bid * load_tokens_s;
  int64_t* load_tokens_host_base = load_tokens_host + bid * load_tokens_host_s;
  PageOutT* page_ids_base = page_ids + bid * page_ids_s;
  
  // Shared memory
  // Use a union or just layouts to manage shared memory
  // Layout:
  // s_last_top_k: [hot_buffer_page] (int64)
  // s_top_k: [top_k_page] (int32)
  // s_last_page_ids: [hot_buffer_page] (int64)
  // s_load_pages: [top_k_page] (int64)
  // s_host_pages: [top_k_page] (int64)
  // ... temporary buffers for scan ...
  
  extern __shared__ char smem[];
  int64_t* s_last_top_k = reinterpret_cast<int64_t*>(smem);
  int64_t* s_last_page_ids = s_last_top_k + kMaxHotBufferPages;
  int32_t* s_top_k = reinterpret_cast<int32_t*>(s_last_page_ids + kMaxHotBufferPages);
  
  // Re-use memory for outputs/intermediates
  // Note: We need s_load_pages and s_host_pages for the final expansion step
  // They can overlap with recycle buffers as long as we are careful.
  int64_t* s_load_pages = reinterpret_cast<int64_t*>(s_top_k + kMaxHotBufferPages);
  int64_t* s_host_pages = s_load_pages + kMaxHotBufferPages;
  
  // CUB Storage
  union TempStorage {
      typename cub::BlockReduce<int64_t, kBlockSize>::TempStorage reduce;
      typename cub::BlockScan<int, kBlockSize>::TempStorage scan;
  };
  __shared__ TempStorage temp_storage;
  __shared__ int32_t s_fill_count;
  __shared__ int32_t s_valid_cnt;
  __shared__ int32_t s_fill_cnt_topk;

  // Initialize outputs (Parallel)
  for (int64_t i = tid; i < top_k; i += blockDim.x) {
    load_tokens_base[i] = -1;
    load_tokens_host_base[i] = -1;
  }
  for (int64_t i = tid; i < top_k_page; i += blockDim.x) {
    page_ids_base[i] = static_cast<PageOutT>(-1);
  }
  
  if (tid == 0) {
      s_fill_count = 0;
  }

  // Pre-load common data
  const int64_t req_idx = req_pool_indices[bid];
  const int64_t seq_len = seq_lens[bid] - 1;
  const int32_t sparse_mask_val = to_i64<SparseMaskT>(sparse_mask[bid]);
  
  const int32_t* top_k_base = top_k_idx + bid * top_k_s;
  int64_t* last_top_k_base = last_top_k_idx + req_idx * last_top_k_s0 + layer_id * last_top_k_s1;
  int64_t* last_page_ids_base = last_page_ids + req_idx * last_page_ids_s0 + layer_id * last_page_ids_s1;

  // 1. Load Data to Shared Memory (Vectorized)
  load_shared_vectorized(last_top_k_base, s_last_top_k, hot_buffer_page, tid);
  load_shared_vectorized(last_page_ids_base, s_last_page_ids, hot_buffer_page, tid);
  load_shared_vectorized(top_k_base, s_top_k, top_k_page, tid);
  
  // Init temp buffers
  if (tid < top_k_page) {
      s_load_pages[tid] = -1;
      s_host_pages[tid] = -1;
  }
  
  __syncthreads();

  if ((sparse_mask_val == 0) || (seq_len <= 0)) {
      // Fast path
      if (tid < top_k_page) {
          int32_t val = s_top_k[tid];
          if (val >= 0) {
              int64_t loaded_page_start = static_cast<int64_t>(page_table[page_table_s * req_idx + val]);
              page_ids_base[tid] = static_cast<PageOutT>(loaded_page_start / page_size);
          }
      }
  } else {
      // Full Diff Logic
      
      // 2. Update Max (Parallel Reduction)
      int64_t my_last = (tid < hot_buffer_page) ? s_last_top_k[tid] : -9223372036854775807LL;
      int64_t last_max = cub::BlockReduce<int64_t, kBlockSize>(temp_storage.reduce).Reduce(my_last, cub::Max());
      __syncthreads(); // Barrier for temp_storage reuse
      
      int64_t my_curr = (tid < top_k_page) ? (int64_t)s_top_k[tid] : -9223372036854775807LL;
      int64_t curr_max = cub::BlockReduce<int64_t, kBlockSize>(temp_storage.reduce).Reduce(my_curr, cub::Max());
      __syncthreads();
      
      if (curr_max != last_max) {
          if (tid < hot_buffer_page && s_last_top_k[tid] >= last_max) {
               s_last_top_k[tid] = curr_max;
          }
      }
      __syncthreads(); // Wait for s_last_top_k update
      
      // 3. Intersection (Brute-force in Shared Memory)
      // Replaces diff_map.
      // Each thread (representing a new top_k item) scans s_last_top_k.
      // Since arrays are small (<=256), this is fast.
      // To avoid read conflicts on s_last_page_ids later, we mark used slots in a separate bitmap or bool array?
      // Actually we can just modify s_last_page_ids[match_idx] = -1.
      // But multiple threads might try to write.
      // Uniqueness assumption: s_top_k has unique values, s_last_top_k has unique values.
      // So at most one thread matches one slot. Safe to write.
      
      if (tid < top_k_page) {
          int32_t val = s_top_k[tid];
          if (val >= 0) {
              // Brute force scan
              for (int j = 0; j < hot_buffer_page; ++j) {
                  if (s_last_top_k[j] == (int64_t)val) {
                      int64_t exist_page = s_last_page_ids[j];
                      if (exist_page != -1) {
                          page_ids_base[tid] = static_cast<PageOutT>(exist_page);
                          s_last_page_ids[j] = -1; // Mark as reused
                          s_top_k[tid] = -1; // Mark as satisfied
                      }
                      break; // Found match
                  }
              }
          }
      }
      __syncthreads();
      
      // 4. Compaction Logic
      
      // A. Identify holes (where page_ids is -1)
      bool is_hole = false;
      if (tid < top_k_page) {
          if (page_ids_base[tid] == static_cast<PageOutT>(-1)) is_hole = true;
      } else if (tid < hot_buffer_page) {
          is_hole = true; // Extra slots in hot buffer are treated as holes
      }
      
      int hole_rank;
      int total_holes;
      cub::BlockScan<int, kBlockSize>(temp_storage.scan).InclusiveSum((int)is_hole, hole_rank, total_holes);
      if (tid == 0) s_fill_count = total_holes; // Total fill needed (across both top_k and extra buffer)
      __syncthreads();
      
      // Calculate how many holes are within top_k_page range
      // We need s_fill_cnt_topk.
      // We can infer it: hole_rank of thread (top_k_page - 1).
      // But we need to handle edge case where top_k_page=0.
      if (tid == top_k_page - 1) {
          s_fill_cnt_topk = is_hole ? hole_rank : (hole_rank); 
          // InclusiveSum: if is_hole=1, rank is count including self.
          // if is_hole=0, rank is count before self.
          // Wait, InclusiveSum returns rank including self if input is 1.
          // If input is 0, it returns sum so far (same as prev).
          // So hole_rank IS the count of holes up to tid.
          // Correct.
      } else if (top_k_page == 0 && tid == 0) {
           s_fill_cnt_topk = 0;
      }
      // Broadcast s_fill_cnt_topk ? Actually we can just read it from shared memory if we wrote it.
      // But variable broadcast is tricky. Let's use shared mem.
      __syncthreads();
      
      // B. Identify Valid Recyclables
      bool is_valid = (tid < hot_buffer_page && s_last_page_ids[tid] != -1);
      int valid_rank;
      int total_valid;
      cub::BlockScan<int, kBlockSize>(temp_storage.scan).InclusiveSum((int)is_valid, valid_rank, total_valid);
      if (tid == 0) s_valid_cnt = total_valid;
      __syncthreads();
      
      // C. Move Data
      // To avoid bank conflicts and complex indexing, we can write compacted data to a temporary buffer
      // But shared memory is tight.
      // Let's use the logic:
      // We have `total_valid` items to distribute.
      // `s_fill_cnt_topk` go to fill the first holes (which are in top_k range).
      // The rest go to the end of buffer.
      
      // We need a scatter map.
      // Each valid item knows its rank (valid_rank - 1).
      // It needs to calculate its destination index.
      
      int dest_idx = -1;
      if (is_valid) {
          int page_pos = valid_rank - 1;
          int move_cnt = s_valid_cnt - s_fill_cnt_topk;
          bool fill_slots = (page_pos >= move_cnt); 
          
          // Logic from original:
          // fill_slots = (page_pos >= move_cnt)
          // dest_idx = fill_slots ? (page_pos - move_cnt) : (page_pos + s_fill_cnt_topk);
          // Wait, this logic maps "later" valid items to "earlier" holes?
          // Let's re-verify original logic.
          /*
          int move_cnt = valid_cnt - fill_cnt_topk;
          for(int k=0; k<valid_cnt; ++k) {
              int page_pos = k;
              bool fill_slots = (page_pos >= move_cnt);
              int dest_idx = fill_slots ? (page_pos - move_cnt) : (page_pos + fill_cnt_topk);
              recycled_pages[dest_idx] = ...
          }
          */
          // Yes.
          // dest_idx is the index in the "Compacted Array" of valid items?
          // No, dest_idx is the index in `recycled_pages` buffer which corresponds to `fill_indices`.
          // Original:
          // fill_indices[i] maps i -> rank of hole.
          // recycled_pages[rank] stores the item.
          // So if I am a valid item with rank R, I write to recycled_pages[dest_idx(R)].
          // Then hole at `i` with rank `r` reads from `recycled_pages[r]`.
          
          // So:
          int target_hole_rank = fill_slots ? (page_pos - move_cnt) : (page_pos + s_fill_cnt_topk);
          
          // We need to write to a temp buffer at `target_hole_rank`.
          // We can reuse s_last_top_k or s_last_page_ids? No, we are reading them.
          // We can use s_load_pages / s_host_pages as temp storage since they are not used yet.
          // s_load_pages is int64, good.
          // We need to store both page_id and top_k.
          // s_load_pages[target_hole_rank] = s_last_page_ids[tid]
          // s_host_pages[target_hole_rank] = s_last_top_k[tid]
          
          if (target_hole_rank < kMaxHotBufferPages) {
              s_load_pages[target_hole_rank] = s_last_page_ids[tid];
              s_host_pages[target_hole_rank] = s_last_top_k[tid];
          }
      }
      __syncthreads();
      
      // D. Fill & Merge
      if (tid < hot_buffer_page) {
          bool mask_topk = (tid < top_k_page);
          
          if (is_hole) {
              // I am a hole. My rank is hole_rank - 1.
              int my_rank = hole_rank - 1;
              
              int64_t r_page = s_load_pages[my_rank];
              int64_t r_topk = s_host_pages[my_rank];
              
              // Note: s_load_pages was init to -1.
              // If we didn't write to it (because valid_cnt is small), it stays -1.
              
              if (r_page != -1) {
                  // Recycled
                  if (mask_topk) page_ids_base[tid] = static_cast<PageOutT>(r_page);
                  last_page_ids_base[tid] = r_page;
                  last_top_k_base[tid] = r_topk;
                  
                  // Setup for Token Expansion
                  if (mask_topk && top_k_base[tid] >= 0) {
                      // We need to store these for the expansion phase
                      // But wait, we used s_load_pages/s_host_pages as TEMP buffers!
                      // We need to persist them to the correct location for expansion.
                      // Expansion reads s_load_pages[page_idx].
                      // Here page_idx = tid (since 1 page per thread).
                      // So we just need to overwrite s_load_pages[tid] with the result.
                      // But we are reading s_load_pages[my_rank] currently!
                      // Race condition if we write to s_load_pages[tid] now.
                      // We need a register or another temp.
                      
                      // Store in register first
                  }
              } else {
                  // Host Load
                  if (mask_topk) {
                     last_page_ids_base[tid] = -1;
                     last_top_k_base[tid] = top_k_base[tid];
                  } else {
                     last_page_ids_base[tid] = -1;
                     last_top_k_base[tid] = -1;
                  }
              }
          } else {
              // Reused (Not a hole)
              last_page_ids_base[tid] = static_cast<int64_t>(page_ids_base[tid]);
              last_top_k_base[tid] = top_k_base[tid];
          }
      }
      __syncthreads();
      
      // Now setup s_load_pages / s_host_pages for token expansion
      // We need to re-scan holes? Or just use is_hole and fill_indices logic?
      // Actually, we can just rebuild the logic or use registers.
      
      // Let's do a clean pass.
      // We have updated last_page_ids_base and last_top_k_base in Global Memory.
      // We also have page_ids_base updated.
      // We need to populate s_load_pages[tid] and s_host_pages[tid] for expansion.
      
      if (tid < top_k_page) {
          bool mask_topk = true;
          // Re-evaluate emptiness
          bool empty = (page_ids_base[tid] == static_cast<PageOutT>(-1)); // Should be filled now unless host load
          // Wait, if it was recycled, page_ids_base[tid] is set.
          // If it was host load, page_ids_base[tid] is STILL -1 (because we don't know the page id yet).
          
          // Actually, we can just look at `is_hole` from before.
          if (is_hole) {
               // It was a hole.
               int my_rank = hole_rank - 1;
               int64_t r_page = s_load_pages[my_rank]; // This is still valid, we haven't overwritten it yet.
               
               if (r_page != -1) {
                   // Recycled
                   s_load_pages[tid] = r_page; // Now we overwrite. Safe?
                   // No! s_load_pages[my_rank] might be needed by another thread `tid2`.
                   // `my_rank` can be different from `tid`.
                   // So we cannot overwrite in place.
                   
                   // BUT! We only need `s_load_pages` for token expansion.
                   // The temp buffer data in `s_load_pages` is `r_page` (the recycled page ID).
                   // The temp buffer data in `s_host_pages` is `r_topk` (the recycled topk).
                   
                   // We need:
                   // Output s_load_pages[tid] = r_page (if recycled) or -1 (if host load).
                   // Output s_host_pages[tid] = top_k_base[tid] (if recycled or host load).
                   
                   // Since we cannot overwrite safely, we need to sync.
                   // But we don't have extra shared memory.
                   // Register shuffle?
                   // Or just read from Global Memory `last_page_ids_base`?
                   // We just wrote to `last_page_ids_base[tid]`.
                   // So we can read it back!
                   
                   // Global Memory Read is safer and simpler here.
                   // s_load_pages[tid] = last_page_ids_base[tid];
                   // s_host_pages[tid] = top_k_base[tid];
               } else {
                   // Host Load
                   // s_load_pages[tid] = -1;
                   // s_host_pages[tid] = top_k_base[tid];
               }
          } else {
              // Reused
              // s_load_pages[tid] = -1;
              // s_host_pages[tid] = -1;
          }
      }
      __syncthreads(); // Ensure global writes are visible? No, consistency within block is loose.
      // But we are reading what we wrote. Within same thread, it is consistent.
      
      if (tid < top_k_page) {
          if (is_hole) {
              int64_t lp = last_page_ids_base[tid]; // Read back
              if (lp != -1) {
                  s_load_pages[tid] = lp;
                  s_host_pages[tid] = top_k_base[tid];
              } else {
                  s_load_pages[tid] = -1;
                  s_host_pages[tid] = top_k_base[tid];
              }
          } else {
              s_load_pages[tid] = -1;
              s_host_pages[tid] = -1;
          }
      }
      __syncthreads();
  }
  
  // Parallel Token Expansion
  const int64_t* tokens_host_base = req_to_tokens_host + req_pool_indices[bid] * req_to_tokens_host_s;
  
  // Reuse s_fill_count. If we are in Fast Path, s_fill_count is 0, so loop won't run.
  // Wait, in Fast Path, we might need to load tokens if sparse_mask is false but we have top_k?
  // Original logic: if sparse_mask==0, we don't do token expansion?
  // Let's check original code.
  // Original: if (sparse_mask_val == 0) ... else { ... s_fill_count = ... }
  // __syncthreads();
  // ... if (page_idx >= fill_count) continue;
  // So yes, if Fast Path, fill_count is 0 (initialized at start), so no expansion.
  
  const int32_t fill_count = s_fill_count;
  const int64_t max_t = (int64_t)fill_count * page_size;
  
  for (int64_t t = tid; t < max_t; t += blockDim.x) {
      const int64_t page_idx = t / page_size;
      const int64_t token_offset = t % page_size;
      
      const int64_t page_id = s_load_pages[page_idx];
      if (page_id != -1) {
          load_tokens_base[t] = page_id * page_size + token_offset;
      }
      
      const int64_t page_id_host = s_host_pages[page_idx];
      if (page_id_host != -1) {
           const int64_t token_idx_host = page_id_host * page_size + token_offset;
           load_tokens_host_base[t] = tokens_host_base[token_idx_host];
      }
  }
}

}  // namespace

void sparse_page_wise_diff(
    torch::Tensor last_top_k_idx,
    torch::Tensor top_k_idx,
    torch::Tensor last_page_ids,
    torch::Tensor page_ids,
    torch::Tensor diff_map,
    torch::Tensor req_to_tokens_host,
    torch::Tensor load_tokens,
    torch::Tensor load_tokens_host,
    torch::Tensor seq_lens,
    torch::Tensor req_pool_indices,
    torch::Tensor sparse_mask,
    torch::Tensor page_table,
    int64_t layer_id,
    int64_t top_k,
    int64_t hot_buffer_len,
    int64_t page_size) {
  
  // ... Checks omitted for brevity, same as before ...
  // But keeping them is good practice.
  CHECK_CUDA(last_top_k_idx);
  CHECK_CUDA(top_k_idx);
  CHECK_CUDA(last_page_ids);
  CHECK_CUDA(page_ids);
  // diff_map check is fine even if unused
  CHECK_CUDA(req_to_tokens_host);
  CHECK_CUDA(load_tokens);
  CHECK_CUDA(load_tokens_host);
  CHECK_CUDA(seq_lens);
  CHECK_CUDA(req_pool_indices);
  CHECK_CUDA(sparse_mask);
  CHECK_CUDA(page_table);

  // Type checks
  TORCH_CHECK(last_top_k_idx.scalar_type() == at::kLong, "last_top_k_idx must be int64");
  TORCH_CHECK(top_k_idx.scalar_type() == at::kInt, "top_k_idx must be int32");
  TORCH_CHECK(last_page_ids.scalar_type() == at::kLong, "last_page_ids must be int64");
  TORCH_CHECK(page_ids.scalar_type() == at::kInt, "page_ids must be int32");
  TORCH_CHECK(sparse_mask.scalar_type() == at::kBool, "sparse_mask must be bool");
  TORCH_CHECK(page_table.scalar_type() == at::kInt, "page_table must be int32");

  const int64_t top_k_page = top_k / page_size;
  const int64_t hot_buffer_page = hot_buffer_len / page_size;
  
  TORCH_CHECK(top_k_page <= kMaxHotBufferPages, "top_k_page too large");
  TORCH_CHECK(hot_buffer_page <= kMaxHotBufferPages, "hot_buffer_page too large");

  const int64_t batch_size = top_k_idx.size(0);

  const at::cuda::CUDAGuard device_guard(top_k_idx.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const dim3 grid(static_cast<uint32_t>(batch_size));
  const dim3 block(kBlockSize); 
  
  // Shared memory size calculation
  // int64 * 256 * 2 (last_top_k, last_page_ids)
  // int32 * 256 (top_k)
  // int64 * 256 * 2 (load_pages, host_pages) - Overlap with recycle buffers?
  // Let's allocate full size to be safe and simple.
  // 256 * (8+8+4+8+8) = 256 * 36 bytes = 9216 bytes.
  // Plus temp storage for CUB.
  // CUB TempStorage is union, max of Scan/Reduce. Typically small (< 1KB).
  // 9KB + 1KB = 10KB. Well within 48KB/64KB limit.
  
  size_t smem_size = kMaxHotBufferPages * (sizeof(int64_t) * 4 + sizeof(int32_t)) + 1024; 

  sparse_page_wise_diff_kernel<int16_t, int32_t, bool><<<grid, block, smem_size, stream>>>(
      last_top_k_idx.data_ptr<int64_t>(),
      top_k_idx.data_ptr<int32_t>(),
      last_page_ids.data_ptr<int64_t>(),
      page_ids.data_ptr<int32_t>(),
      diff_map.data_ptr<int16_t>(),
      req_to_tokens_host.data_ptr<int64_t>(),
      load_tokens.data_ptr<int64_t>(),
      load_tokens_host.data_ptr<int64_t>(),
      seq_lens.data_ptr<int64_t>(),
      req_pool_indices.data_ptr<int64_t>(),
      sparse_mask.data_ptr<bool>(),
      page_table.data_ptr<int32_t>(),
      last_top_k_idx.stride(0),
      last_top_k_idx.stride(1),
      top_k_idx.stride(0),
      last_page_ids.stride(0),
      last_page_ids.stride(1),
      page_ids.stride(0),
      diff_map.stride(0),
      req_to_tokens_host.stride(0),
      load_tokens.stride(0),
      load_tokens_host.stride(0),
      page_table.stride(0),
      layer_id,
      top_k,
      top_k_page,
      hot_buffer_len,
      hot_buffer_page,
      page_size);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
