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
  
  __syncthreads();

  // Single-threaded core logic (tid == 0) for correctness
  if (tid == 0) {
      const int64_t req_idx = req_pool_indices[bid];
      const int64_t seq_len = seq_lens[bid] - 1;
      const int32_t sparse_mask_val = to_i64<SparseMaskT>(sparse_mask[bid]);
      
      const int32_t* top_k_base = top_k_idx + bid * top_k_s;
      int64_t* last_top_k_base = last_top_k_idx + req_idx * last_top_k_s0 + layer_id * last_top_k_s1;
      int64_t* last_page_ids_base = last_page_ids + req_idx * last_page_ids_s0 + layer_id * last_page_ids_s1;
      DiffT* diff_map_base = diff_map + bid * diff_map_s;

      // Local buffers
      int64_t s_last_top_k[kMaxHotBufferPages];
      int32_t s_top_k[kMaxHotBufferPages];
      int64_t s_last_page_ids[kMaxHotBufferPages];
      
      // Load Data
      for (int i = 0; i < hot_buffer_page; ++i) {
          s_last_top_k[i] = last_top_k_base[i];
          s_last_page_ids[i] = last_page_ids_base[i];
      }
      for (int i = 0; i < top_k_page; ++i) {
          s_top_k[i] = top_k_base[i];
      }

      if ((sparse_mask_val == 0) || (seq_len <= 0)) {
          // Fast path: just load pages for top_k
          for (int i = 0; i < top_k_page; ++i) {
              int32_t val = s_top_k[i];
              if (val >= 0) {
                  int64_t loaded_page_start = static_cast<int64_t>(page_table[page_table_s * req_idx + val]);
                  page_ids_base[i] = static_cast<PageOutT>(loaded_page_start / page_size);
              }
          }
      } else {
          // Full Diff Logic
          
          // 1. Update last_top_k with max values
          int64_t last_max = -9223372036854775807LL - 1;
          for(int i=0; i<hot_buffer_page; ++i) {
              if (s_last_top_k[i] > last_max) last_max = s_last_top_k[i];
          }
          int64_t curr_max = -9223372036854775807LL - 1;
          for(int i=0; i<top_k_page; ++i) {
              if ((int64_t)s_top_k[i] > curr_max) curr_max = (int64_t)s_top_k[i];
          }
          
          for(int i=0; i<hot_buffer_page; ++i) {
              if (curr_max != last_max) {
                  if (s_last_top_k[i] < last_max) {
                      // Keep it
                  } else {
                      // Update to new max
                      s_last_top_k[i] = curr_max;
                  }
              }
          }
          
          // 2. Build Diff Map
          // Clear diff map for NEW top_k values first
          for(int i=0; i<top_k_page; ++i) {
              int32_t val = s_top_k[i];
              if (val >= 0) diff_map_base[val] = static_cast<DiffT>(-1);
          }
          // Populate diff map with OLD top_k values
          for(int i=0; i<hot_buffer_page; ++i) {
              int64_t val = s_last_top_k[i];
              if (val >= 0) diff_map_base[val] = static_cast<DiffT>(i);
          }
          
          // 3. Intersection & Reuse
          for(int i=0; i<top_k_page; ++i) {
              int32_t val = s_top_k[i];
              if (val >= 0) {
                  int32_t exist_idx = static_cast<int32_t>(diff_map_base[val]);
                  if (exist_idx >= 0) {
                      int64_t exist_page = s_last_page_ids[exist_idx];
                      if (exist_page >= 0) {
                          page_ids_base[i] = static_cast<PageOutT>(exist_page);
                          s_last_page_ids[exist_idx] = -1; // Mark as reused
                          s_top_k[i] = -1; // Mark as satisfied
                          
                          // Clear diff map entry
                          diff_map_base[val] = static_cast<DiffT>(-1); 
                      }
                  }
              }
          }
          
          // 4. Compaction Logic
          // Calculate fill indices (where page_ids is empty)
          int32_t fill_indices[kMaxHotBufferPages]; 
          int fill_cnt_total = 0;
          int fill_cnt_topk = 0;
          for(int i=0; i<hot_buffer_page; ++i) {
              bool mask_topk = (i < top_k_page);
              int64_t p = mask_topk ? static_cast<int64_t>(page_ids_base[i]) : -1;
              if (p == -1) {
                  fill_indices[i] = fill_cnt_total; 
                  fill_cnt_total++;
                  if (mask_topk) fill_cnt_topk++;
              } else {
                  fill_indices[i] = -1;
              }
          }
          s_fill_count = fill_cnt_topk;
          
          // Collect valid recycled pages
          int32_t valid_indices[kMaxHotBufferPages];
          int valid_cnt = 0;
          for(int i=0; i<hot_buffer_page; ++i) {
              if (s_last_page_ids[i] != -1) {
                  valid_indices[valid_cnt++] = i;
              }
          }
          
          int32_t move_cnt = valid_cnt - fill_cnt_topk;
          
          // Temp buffers for recycled data
          int64_t recycled_pages[kMaxHotBufferPages];
          int64_t recycled_top_k[kMaxHotBufferPages];
          for(int i=0; i<hot_buffer_page; ++i) { 
              recycled_pages[i] = -1; 
              recycled_top_k[i] = -1; 
          }
          
          for(int k=0; k<valid_cnt; ++k) {
              int idx_in_last = valid_indices[k];
              int page_pos = k;
              bool fill_slots = (page_pos >= move_cnt);
              int dest_idx = fill_slots ? (page_pos - move_cnt) : (page_pos + fill_cnt_topk);
              if (dest_idx < kMaxHotBufferPages) {
                  recycled_pages[dest_idx] = s_last_page_ids[idx_in_last];
                  recycled_top_k[dest_idx] = s_last_top_k[idx_in_last];
              }
          }
          
          // Fill & Merge Logic
          for(int i=0; i<hot_buffer_page; ++i) {
              bool mask_topk = (i < top_k_page);
              bool empty = mask_topk ? (page_ids_base[i] == static_cast<PageOutT>(-1)) : true;
              
              if (empty) {
                  int fpos = fill_indices[i]; // valid for empty slots
                  
                  int64_t r_page = recycled_pages[fpos];
                  int64_t r_topk = recycled_top_k[fpos];
                  
                  if (r_page != -1) {
                      // Recycled
                      if (mask_topk) page_ids_base[i] = static_cast<PageOutT>(r_page);
                      last_page_ids_base[i] = r_page;
                      last_top_k_base[i] = r_topk;
                      
                      // Only generate load token if not padding
                      if (mask_topk && top_k_base[i] >= 0) {
                          s_load_pages[fpos] = r_page;
                          s_host_pages[fpos] = top_k_base[i];
                      } else {
                          s_load_pages[fpos] = -1;
                          s_host_pages[fpos] = -1;
                      }
                  } else {
                      // Host Load but no Recycled Page (Should not happen if cache is sufficient)
                      if (mask_topk) {
                         // Load corresponding top_k_base[i]
                         s_host_pages[fpos] = top_k_base[i];
                         last_page_ids_base[i] = -1;
                         last_top_k_base[i] = top_k_base[i];
                      } else {
                         last_page_ids_base[i] = -1;
                         last_top_k_base[i] = -1;
                         s_host_pages[fpos] = -1;
                      }
                      s_load_pages[fpos] = -1;
                  }
              } else {
                  // Reused
                  last_page_ids_base[i] = static_cast<int64_t>(page_ids_base[i]);
                  last_top_k_base[i] = top_k_base[i];
              }
          }
          
          // Cleanup diff_map
          for(int i=0; i<hot_buffer_page; ++i) {
             if (s_last_top_k[i] >= 0) diff_map_base[s_last_top_k[i]] = static_cast<DiffT>(-1);
          }
      }
  }
  
  __syncthreads();
  
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
