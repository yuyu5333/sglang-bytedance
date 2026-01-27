#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cstdint>

#include "diff_kernels.h"
#include "utils.h"

namespace {

constexpr int64_t kMaxHotBufferPages = 256;

template <typename T>
__device__ __forceinline__ int64_t to_i64(T v) {
  return static_cast<int64_t>(v);
}

template <>
__device__ __forceinline__ int64_t to_i64<bool>(bool v) {
  return v ? 1 : 0;
}

// Parallel Primitive: Block Reduce Max
template <typename T>
__device__ T block_reduce_max(T val) {
    // Warp reduce
    for (int offset = 16; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    
    static __shared__ T shared[32]; // Max 32 warps (1024 threads)
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    if (lane == 0) shared[wid] = val;
    
    __syncthreads();
    
    // First warp reduces shared
    // Initialize val for the first warp to a safe minimum if it exceeds active warps
    // We assume T is signed for this minimum or use specific logic
    T w_val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : (T)-9223372036854775807LL; 
    
    if (wid == 0) {
         for (int offset = blockDim.x / 32 / 2; offset > 0; offset /= 2) {
            w_val = max(w_val, __shfl_down_sync(0xFFFFFFFF, w_val, offset));
         }
         if (lane == 0) shared[0] = w_val;
    }
    
    __syncthreads();
    return shared[0];
}

// Parallel Primitive: Block Scan Inclusive (Prefix Sum)
// Returns the inclusive sum for the calling thread.
// Optionally writes the total sum to *total_sum (only valid for last thread).
__device__ int block_scan_inclusive(int val, int* total_sum) {
    // Warp scan
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    for (int offset = 1; offset < 32; offset <<= 1) {
        int temp = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (lane >= offset) val += temp;
    }
    
    static __shared__ int shared[32];
    if (lane == 31) shared[wid] = val;
    
    __syncthreads();
    
    // Scan shared (prefix sums of warps)
    if (wid == 0) {
        int w_val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
        for (int offset = 1; offset < 32; offset <<= 1) { 
             int temp = __shfl_up_sync(0xFFFFFFFF, w_val, offset);
             if (lane >= offset) w_val += temp;
        }
        if (threadIdx.x < blockDim.x / 32) shared[lane] = w_val;
    }
    
    __syncthreads();
    
    // Add base
    if (wid > 0) {
        val += shared[wid - 1];
    }
    
    if (total_sum && threadIdx.x == blockDim.x - 1) *total_sum = val;
    
    return val;
}

template <typename DiffT, typename PageOutT, typename SparseMaskT>
__global__ void sparse_page_wise_diff_kernel(
    int64_t* __restrict__ last_top_k_idx,
    const int32_t* __restrict__ top_k_idx,
    int64_t* __restrict__ last_page_ids,
    PageOutT* __restrict__ page_ids,
    DiffT* __restrict__ diff_map,
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
  
  // Shared memory buffers
  __shared__ int64_t s_load_pages[kMaxHotBufferPages];
  __shared__ int64_t s_host_pages[kMaxHotBufferPages];
  __shared__ int32_t s_fill_count;
  __shared__ int32_t s_fill_cnt_topk;
  __shared__ int32_t s_valid_cnt;

  // Local buffers in shared memory
  __shared__ int64_t s_last_top_k[kMaxHotBufferPages];
  __shared__ int32_t s_top_k[kMaxHotBufferPages];
  __shared__ int64_t s_last_page_ids[kMaxHotBufferPages];
  __shared__ int64_t recycled_pages[kMaxHotBufferPages];
  __shared__ int64_t recycled_top_k[kMaxHotBufferPages];
  __shared__ int32_t fill_indices[kMaxHotBufferPages];

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
  DiffT* diff_map_base = diff_map + bid * diff_map_s;

  // Load Data to Shared Memory
  if (tid < hot_buffer_page) {
      s_last_top_k[tid] = last_top_k_base[tid];
      s_last_page_ids[tid] = last_page_ids_base[tid];
  }
  if (tid < top_k_page) {
      s_top_k[tid] = top_k_base[tid];
  }
  
  __syncthreads();

  if ((sparse_mask_val == 0) || (seq_len <= 0)) {
      // Fast path: just load pages for top_k
      if (tid < top_k_page) {
          int32_t val = s_top_k[tid];
          if (val >= 0) {
              int64_t loaded_page_start = static_cast<int64_t>(page_table[page_table_s * req_idx + val]);
              page_ids_base[tid] = static_cast<PageOutT>(loaded_page_start / page_size);
          }
      }
      // Note: In fast path, we don't need to populate s_fill_count for token expansion 
      // because token expansion loop checks `page_idx >= fill_count`.
      // If we don't set s_fill_count, it remains 0 (set by tid=0 above).
      // But we need to ensure s_load_pages/s_host_pages are handled?
      // The original code only uses s_fill_count for "Parallel Token Expansion".
      // In fast path, `sparse_mask` is false, so maybe token expansion is not needed?
      // Wait, original code:
      // if ((sparse_mask_val == 0) || (seq_len <= 0)) { ... } else { ... s_fill_count = ... }
      // __syncthreads();
      // Parallel Token Expansion ... if (page_idx >= fill_count) continue;
      // So if fill_count is 0, token expansion does nothing.
      // Correct.
  } else {
      // Full Diff Logic
      
      // 1. Update last_top_k with max values
      int64_t my_last = (tid < hot_buffer_page) ? s_last_top_k[tid] : -9223372036854775807LL;
      int64_t last_max = block_reduce_max(my_last);
      
      int64_t my_curr = (tid < top_k_page) ? (int64_t)s_top_k[tid] : -9223372036854775807LL;
      int64_t curr_max = block_reduce_max(my_curr);
      
      if (curr_max != last_max) {
          if (tid < hot_buffer_page && s_last_top_k[tid] >= last_max) {
               s_last_top_k[tid] = curr_max;
          }
      }
      __syncthreads();
      
      // 2. Build Diff Map
      // Clear diff map for NEW top_k values first
      if (tid < top_k_page) {
          int32_t val = s_top_k[tid];
          if (val >= 0) diff_map_base[val] = static_cast<DiffT>(-1);
      }
      __syncthreads();
      
      // Populate diff map with OLD top_k values
      if (tid < hot_buffer_page) {
          int64_t val = s_last_top_k[tid];
          if (val >= 0) diff_map_base[val] = static_cast<DiffT>(tid);
      }
      __syncthreads();
      
      // 3. Intersection & Reuse
      if (tid < top_k_page) {
          int32_t val = s_top_k[tid];
          if (val >= 0) {
              int32_t exist_idx = static_cast<int32_t>(diff_map_base[val]);
              if (exist_idx >= 0) {
                  // Atomic exchange to claim
                  unsigned long long* ptr = (unsigned long long*)&s_last_page_ids[exist_idx];
                  unsigned long long old_page = atomicExch(ptr, (unsigned long long)-1);
                  
                  if ((int64_t)old_page != -1) {
                      page_ids_base[tid] = static_cast<PageOutT>(old_page);
                      s_top_k[tid] = -1; // Mark as satisfied
                      // Clear diff map entry
                      diff_map_base[val] = static_cast<DiffT>(-1); 
                  }
              }
          }
      }
      __syncthreads();
      
      // 4. Compaction Logic
      // Initialize buffers
      if (tid < kMaxHotBufferPages) {
          recycled_pages[tid] = -1;
          recycled_top_k[tid] = -1;
      }
      
      // Identify holes
      bool is_hole = false;
      if (tid < top_k_page) {
          if (page_ids_base[tid] == static_cast<PageOutT>(-1)) is_hole = true;
      } else if (tid < hot_buffer_page) {
          is_hole = true;
      }
      
      // Prefix sum of holes
      int hole_rank = block_scan_inclusive((int)is_hole, &s_fill_count);
      
      if (is_hole) {
          fill_indices[tid] = hole_rank - 1;
      } else {
          fill_indices[tid] = -1;
      }
      
      // Compute s_fill_cnt_topk
      if (tid == top_k_page - 1) {
          s_fill_cnt_topk = hole_rank;
      } else if (top_k_page == 0 && tid == 0) {
          s_fill_cnt_topk = 0;
      }
      __syncthreads();
      
      // Collect valid recycled pages
      bool is_valid = (tid < hot_buffer_page && s_last_page_ids[tid] != -1);
      int valid_rank = block_scan_inclusive((int)is_valid, &s_valid_cnt);
      
      // Recycled Data Move
      if (is_valid) {
          int page_pos = valid_rank - 1;
          int move_cnt = s_valid_cnt - s_fill_cnt_topk;
          bool fill_slots = (page_pos >= move_cnt);
          int dest_idx = fill_slots ? (page_pos - move_cnt) : (page_pos + s_fill_cnt_topk);
          
          if (dest_idx < kMaxHotBufferPages) {
               recycled_pages[dest_idx] = s_last_page_ids[tid];
               recycled_top_k[dest_idx] = s_last_top_k[tid];
          }
      }
      __syncthreads();
      
      // Fill & Merge Logic
      if (tid < hot_buffer_page) {
          bool mask_topk = (tid < top_k_page);
          
          if (is_hole) {
              int fpos = fill_indices[tid];
              
              int64_t r_page = recycled_pages[fpos];
              int64_t r_topk = recycled_top_k[fpos];
              
              if (r_page != -1) {
                  // Recycled
                  if (mask_topk) page_ids_base[tid] = static_cast<PageOutT>(r_page);
                  last_page_ids_base[tid] = r_page;
                  last_top_k_base[tid] = r_topk;
                  
                  if (mask_topk && top_k_base[tid] >= 0) {
                      s_load_pages[fpos] = r_page;
                      s_host_pages[fpos] = top_k_base[tid];
                  } else {
                      s_load_pages[fpos] = -1;
                      s_host_pages[fpos] = -1;
                  }
              } else {
                  // Host Load
                  if (mask_topk) {
                     s_host_pages[fpos] = top_k_base[tid];
                     last_page_ids_base[tid] = -1;
                     last_top_k_base[tid] = top_k_base[tid];
                  } else {
                     last_page_ids_base[tid] = -1;
                     last_top_k_base[tid] = -1;
                     s_host_pages[fpos] = -1;
                  }
                  s_load_pages[fpos] = -1;
              }
          } else {
              // Reused (Not a hole)
              last_page_ids_base[tid] = static_cast<int64_t>(page_ids_base[tid]);
              last_top_k_base[tid] = top_k_base[tid];
          }
      }
      
      // Cleanup diff_map
      if (tid < hot_buffer_page) {
         if (s_last_top_k[tid] >= 0) diff_map_base[s_last_top_k[tid]] = static_cast<DiffT>(-1);
      }
      __syncthreads();
  }
  
  // Parallel Token Expansion
  const int32_t fill_count = s_fill_count;
  const int64_t* tokens_host_base = req_to_tokens_host + req_pool_indices[bid] * req_to_tokens_host_s;

  for (int64_t t = tid; t < top_k; t += blockDim.x) {
      const int64_t page_idx = t / page_size;
      if (page_idx >= fill_count) continue;
      
      const int64_t token_offset = t % page_size;
      
      // Device Load
      const int64_t page_id = s_load_pages[page_idx];
      if (page_id != -1) {
          load_tokens_base[t] = page_id * page_size + token_offset;
      }
      
      // Host Load
      const int64_t page_id_host = s_host_pages[page_idx];
      if (page_id_host != -1) { // -1 means no host load or handled by recycle
          // page_id_host is the logical page index (from top_k_idx)
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
  
  // Checks
  CHECK_CUDA(last_top_k_idx);
  CHECK_CUDA(top_k_idx);
  CHECK_CUDA(last_page_ids);
  CHECK_CUDA(page_ids);
  CHECK_CUDA(diff_map);
  CHECK_CUDA(req_to_tokens_host);
  CHECK_CUDA(load_tokens);
  CHECK_CUDA(load_tokens_host);
  CHECK_CUDA(seq_lens);
  CHECK_CUDA(req_pool_indices);
  CHECK_CUDA(sparse_mask);
  CHECK_CUDA(page_table);

  CHECK_LAST_DIM_CONTIGUOUS(last_top_k_idx);
  CHECK_LAST_DIM_CONTIGUOUS(top_k_idx);
  CHECK_LAST_DIM_CONTIGUOUS(last_page_ids);
  CHECK_LAST_DIM_CONTIGUOUS(page_ids);
  CHECK_LAST_DIM_CONTIGUOUS(diff_map);
  CHECK_LAST_DIM_CONTIGUOUS(req_to_tokens_host);
  CHECK_LAST_DIM_CONTIGUOUS(load_tokens);
  CHECK_LAST_DIM_CONTIGUOUS(load_tokens_host);
  CHECK_LAST_DIM_CONTIGUOUS(seq_lens);
  CHECK_LAST_DIM_CONTIGUOUS(req_pool_indices);
  CHECK_LAST_DIM_CONTIGUOUS(sparse_mask);
  CHECK_LAST_DIM_CONTIGUOUS(page_table);

  // Type checks
  TORCH_CHECK(last_top_k_idx.scalar_type() == at::kLong, "last_top_k_idx must be int64");
  TORCH_CHECK(top_k_idx.scalar_type() == at::kInt, "top_k_idx must be int32");
  TORCH_CHECK(last_page_ids.scalar_type() == at::kLong, "last_page_ids must be int64");
  TORCH_CHECK(page_ids.scalar_type() == at::kInt, "page_ids must be int32");
  TORCH_CHECK(diff_map.scalar_type() == at::kShort, "diff_map must be int16");
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
  const dim3 block(256); // 256 threads per block

  sparse_page_wise_diff_kernel<int16_t, int32_t, bool><<<grid, block, 0, stream>>>(
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
