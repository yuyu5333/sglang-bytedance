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

  int64_t* load_tokens_base = load_tokens + bid * load_tokens_s;
  int64_t* load_tokens_host_base = load_tokens_host + bid * load_tokens_host_s;
  PageOutT* page_ids_base = page_ids + bid * page_ids_s;

  for (int64_t i = tid; i < top_k; i += blockDim.x) {
    load_tokens_base[i] = -1;
    load_tokens_host_base[i] = -1;
  }
  for (int64_t i = tid; i < top_k_page; i += blockDim.x) {
    page_ids_base[i] = static_cast<PageOutT>(-1);
  }
  __syncthreads();

  __shared__ int32_t s_fill_pos[kMaxHotBufferPages];
  __shared__ int64_t s_load_pages[kMaxHotBufferPages];
  __shared__ int64_t s_host_pages[kMaxHotBufferPages];
  __shared__ int32_t s_fill_count;
  __shared__ int32_t s_req_idx;
  __shared__ int32_t s_sparse_mask_val;
  __shared__ int64_t s_seq_len;
  __shared__ int64_t s_last_max_top_k;
  __shared__ int64_t s_curr_max_top_k;

  if (tid == 0) {
    s_seq_len = seq_lens[bid] - 1;
    s_req_idx = static_cast<int32_t>(req_pool_indices[bid]);
    s_sparse_mask_val = static_cast<int32_t>(to_i64<SparseMaskT>(sparse_mask[bid]));
    s_fill_count = 0;
    s_last_max_top_k = 0;
    s_curr_max_top_k = 0;
  }
  __syncthreads();

  const int64_t req_idx = static_cast<int64_t>(s_req_idx);
  const int64_t seq_len = s_seq_len;
  const int32_t sparse_mask_val = s_sparse_mask_val;

  const int32_t* top_k_base = top_k_idx + bid * top_k_s;
  int64_t* last_top_k_base = last_top_k_idx + req_idx * last_top_k_s0 + layer_id * last_top_k_s1;
  int64_t* last_page_ids_base = last_page_ids + req_idx * last_page_ids_s0 + layer_id * last_page_ids_s1;
  DiffT* diff_map_base = diff_map + bid * diff_map_s;
  const int64_t* tokens_host_base = req_to_tokens_host + req_idx * req_to_tokens_host_s;

  if ((sparse_mask_val == 0) || (seq_len <= 0)) {
    for (int64_t i = tid; i < top_k_page; i += blockDim.x) {
      int64_t top_k_val = static_cast<int64_t>(top_k_base[i]);
      if (top_k_val >= 0) {
        int64_t loaded_page_start =
            static_cast<int64_t>(page_table[page_table_s * req_idx + top_k_val]);
        page_ids_base[i] = static_cast<PageOutT>(loaded_page_start / page_size);
      }
    }
    return;
  }

  if (tid == 0) {
    int64_t last_max = -9223372036854775807LL - 1;
    for (int64_t i = 0; i < hot_buffer_page; ++i) {
      int64_t v = last_top_k_base[i];
      if (v > last_max) last_max = v;
    }
    int64_t curr_max = -9223372036854775807LL - 1;
    for (int64_t i = 0; i < top_k_page; ++i) {
      int64_t v = static_cast<int64_t>(top_k_base[i]);
      if (v > curr_max) curr_max = v;
    }
    s_last_max_top_k = last_max;
    s_curr_max_top_k = curr_max;
  }
  __syncthreads();

  const int64_t last_max_top_k = s_last_max_top_k;
  const int64_t curr_max_top_k = s_curr_max_top_k;

  if (tid == 0) {
    for (int64_t i = 0; i < hot_buffer_page; ++i) {
      int64_t v = last_top_k_base[i];
      if (curr_max_top_k != last_max_top_k) {
        v = (v < last_max_top_k) ? v : curr_max_top_k;
      }
      diff_map_base[v] = static_cast<DiffT>(i);
    }
  }
  __syncthreads();

  if (tid == 0) {
    for (int64_t i = 0; i < top_k_page; ++i) {
      const int64_t top_k_origin = static_cast<int64_t>(top_k_base[i]);
      const int32_t exist_top_k_idx = static_cast<int32_t>(diff_map_base[top_k_origin]);
      if (exist_top_k_idx >= 0) {
        const int64_t exist_page = last_page_ids_base[exist_top_k_idx];
        page_ids_base[i] = static_cast<PageOutT>(exist_page);
        last_page_ids_base[exist_top_k_idx] = -1;
      } else {
        load_tokens_host_base[i] = top_k_origin;
      }
    }

    for (int64_t i = 0; i < hot_buffer_page; ++i) {
      int64_t v = last_top_k_base[i];
      if (curr_max_top_k != last_max_top_k) {
        v = (v < last_max_top_k) ? v : curr_max_top_k;
      }
      diff_map_base[v] = static_cast<DiffT>(-1);
    }

    int32_t empty_count = 0;
    for (int64_t i = 0; i < hot_buffer_page; ++i) {
      const bool mask_topk = i < top_k_page;
      const int64_t curr_page = mask_topk ? static_cast<int64_t>(page_ids_base[i]) : -1;
      const bool empty = (curr_page == -1);
      const int32_t empty_int = empty ? 1 : 0;
      s_fill_pos[i] = empty_count;
      empty_count += empty_int;
      if (mask_topk) {
        s_fill_count += empty_int;
      }
    }

    const int32_t fill_count = s_fill_count;

    int32_t page_valid_count = 0;
    for (int64_t i = 0; i < hot_buffer_page; ++i) {
      const int64_t last_page_val = last_page_ids_base[i];
      page_valid_count += (last_page_val != -1) ? 1 : 0;
    }
    const int32_t move_count = page_valid_count - fill_count;

    int32_t page_pos_prefix = 0;
    for (int64_t i = 0; i < hot_buffer_page; ++i) {
      const int64_t last_page_val = last_page_ids_base[i];
      const bool page_valid = (last_page_val != -1);
      if (!page_valid) continue;

      int32_t page_pos = page_pos_prefix;
      page_pos_prefix += 1;

      const bool fill_slots = page_pos >= move_count;
      page_pos = fill_slots ? (page_pos - move_count) : (page_pos + fill_count);

      load_tokens_base[page_pos] = last_page_val;

      int64_t last_top_k_val = last_top_k_base[i];
      if (curr_max_top_k != last_max_top_k) {
        last_top_k_val = (last_top_k_val < last_max_top_k) ? last_top_k_val : curr_max_top_k;
      }
      last_top_k_base[page_pos] = last_top_k_val;
    }

    for (int64_t i = 0; i < hot_buffer_page; ++i) {
      const bool mask_topk = i < top_k_page;
      const int64_t curr_page = mask_topk ? static_cast<int64_t>(page_ids_base[i]) : -1;
      const int64_t curr_top_k = mask_topk ? load_tokens_host_base[i] : -1;
      const bool empty = (curr_page == -1);
      const int32_t fill_pos = s_fill_pos[i];

      const int64_t fill_page = empty ? load_tokens_base[fill_pos] : -1;
      const int64_t fill_top_k = empty ? last_top_k_base[fill_pos] : -1;

      const int64_t final_page = empty ? fill_page : curr_page;
      const int64_t final_top_k = empty ? fill_top_k : curr_top_k;

      last_page_ids_base[i] = final_page;
      if (mask_topk) {
        page_ids_base[i] = static_cast<PageOutT>(final_page);
      }
      last_top_k_base[i] = final_top_k;
    }

    for (int64_t i = 0; i < top_k_page; ++i) {
      last_top_k_base[i] = static_cast<int64_t>(top_k_base[i]);
    }

    for (int64_t i = 0; i < hot_buffer_page; ++i) {
      if (i >= fill_count) {
        load_tokens_base[i] = -1;
      }
    }

    int64_t tmp_host_vals[kMaxHotBufferPages];
    for (int64_t i = 0; i < hot_buffer_page; ++i) {
      tmp_host_vals[i] = load_tokens_host_base[i];
    }
    for (int64_t i = 0; i < hot_buffer_page; ++i) {
      load_tokens_host_base[i] = -1;
    }
    for (int64_t i = 0; i < hot_buffer_page; ++i) {
      const int64_t curr_page = (i < top_k_page) ? static_cast<int64_t>(page_ids_base[i]) : -1;
      const bool empty = (curr_page == -1);
      if (!empty) continue;
      const int32_t fill_pos = s_fill_pos[i];
      load_tokens_host_base[fill_pos] = tmp_host_vals[i];
    }
    for (int64_t i = fill_count; i < hot_buffer_page; ++i) {
      load_tokens_host_base[i] = -1;
    }
  }
  __syncthreads();

  const int32_t fill_count = s_fill_count;

  for (int64_t i = tid; i < fill_count; i += blockDim.x) {
    s_load_pages[i] = load_tokens_base[i];
    s_host_pages[i] = load_tokens_host_base[i];
  }
  __syncthreads();

  for (int64_t t = tid; t < top_k; t += blockDim.x) {
    const int64_t page_idx = t / page_size;
    if (page_idx >= fill_count) continue;
    const int64_t token_offset = t - page_idx * page_size;

    const int64_t page_id = s_load_pages[page_idx];
    load_tokens_base[t] = page_id * page_size + token_offset;

    const int64_t page_id_host = s_host_pages[page_idx];
    const int64_t token_idx_host = page_id_host * page_size + token_offset;
    load_tokens_host_base[t] = tokens_host_base[token_idx_host];
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

  TORCH_CHECK(last_top_k_idx.scalar_type() == at::kLong, "last_top_k_idx must be int64");
  TORCH_CHECK(top_k_idx.scalar_type() == at::kInt, "top_k_idx must be int32");
  TORCH_CHECK(last_page_ids.scalar_type() == at::kLong, "last_page_ids must be int64");
  TORCH_CHECK(req_to_tokens_host.scalar_type() == at::kLong, "req_to_tokens_host must be int64");
  TORCH_CHECK(load_tokens.scalar_type() == at::kLong, "load_tokens must be int64");
  TORCH_CHECK(load_tokens_host.scalar_type() == at::kLong, "load_tokens_host must be int64");
  TORCH_CHECK(seq_lens.scalar_type() == at::kLong, "seq_lens must be int64");
  TORCH_CHECK(req_pool_indices.scalar_type() == at::kLong, "req_pool_indices must be int64");
  TORCH_CHECK(page_table.scalar_type() == at::kInt, "page_table must be int32");

  TORCH_CHECK(diff_map.scalar_type() == at::kShort || diff_map.scalar_type() == at::kInt, "diff_map must be int16 or int32");
  TORCH_CHECK(
      sparse_mask.scalar_type() == at::kBool || sparse_mask.scalar_type() == at::kInt || sparse_mask.scalar_type() == at::kLong,
      "sparse_mask must be bool/int32/int64");
  TORCH_CHECK(page_ids.scalar_type() == at::kInt || page_ids.scalar_type() == at::kLong, "page_ids must be int32 or int64");

  TORCH_CHECK(page_size > 0, "page_size must be > 0");
  TORCH_CHECK(top_k >= 0, "top_k must be >= 0");
  TORCH_CHECK(hot_buffer_len >= 0, "hot_buffer_len must be >= 0");
  TORCH_CHECK(top_k % page_size == 0, "top_k must be divisible by page_size");
  TORCH_CHECK(hot_buffer_len % page_size == 0, "hot_buffer_len must be divisible by page_size");

  const int64_t top_k_page = top_k / page_size;
  const int64_t hot_buffer_page = hot_buffer_len / page_size;

  TORCH_CHECK(top_k_page <= kMaxHotBufferPages, "top_k_page too large: ", top_k_page);
  TORCH_CHECK(hot_buffer_page <= kMaxHotBufferPages, "hot_buffer_page too large: ", hot_buffer_page);

  TORCH_CHECK(last_top_k_idx.dim() == 3, "last_top_k_idx must be 3D");
  TORCH_CHECK(last_page_ids.dim() == 3, "last_page_ids must be 3D");
  TORCH_CHECK(top_k_idx.dim() == 2, "top_k_idx must be 2D");
  TORCH_CHECK(page_ids.dim() == 2, "page_ids must be 2D");
  TORCH_CHECK(diff_map.dim() == 2, "diff_map must be 2D");
  TORCH_CHECK(req_to_tokens_host.dim() == 2, "req_to_tokens_host must be 2D");
  TORCH_CHECK(load_tokens.dim() == 2, "load_tokens must be 2D");
  TORCH_CHECK(load_tokens_host.dim() == 2, "load_tokens_host must be 2D");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be 1D");
  TORCH_CHECK(req_pool_indices.dim() == 1, "req_pool_indices must be 1D");
  TORCH_CHECK(sparse_mask.dim() == 1, "sparse_mask must be 1D");
  TORCH_CHECK(page_table.dim() == 2, "page_table must be 2D");

  const int64_t batch_size = top_k_idx.size(0);
  TORCH_CHECK(seq_lens.size(0) >= batch_size, "seq_lens too small");
  TORCH_CHECK(req_pool_indices.size(0) >= batch_size, "req_pool_indices too small");
  TORCH_CHECK(sparse_mask.size(0) >= batch_size, "sparse_mask too small");
  TORCH_CHECK(page_ids.size(0) >= batch_size, "page_ids batch too small");
  TORCH_CHECK(load_tokens.size(0) >= batch_size, "load_tokens batch too small");
  TORCH_CHECK(load_tokens_host.size(0) >= batch_size, "load_tokens_host batch too small");
  TORCH_CHECK(diff_map.size(0) >= batch_size, "diff_map batch too small");

  TORCH_CHECK(top_k_idx.size(1) >= top_k_page, "top_k_idx second dim too small");
  TORCH_CHECK(page_ids.size(1) >= top_k_page, "page_ids second dim too small");
  TORCH_CHECK(load_tokens.size(1) >= top_k, "load_tokens second dim too small");
  TORCH_CHECK(load_tokens_host.size(1) >= top_k, "load_tokens_host second dim too small");

  TORCH_CHECK(layer_id >= 0 && layer_id < last_top_k_idx.size(1), "layer_id out of range");

  const at::cuda::CUDAGuard device_guard(top_k_idx.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const dim3 grid(static_cast<uint32_t>(batch_size));
  const dim3 block(256);

  const int64_t last_top_k_s0 = last_top_k_idx.stride(0);
  const int64_t last_top_k_s1 = last_top_k_idx.stride(1);
  const int64_t top_k_s = top_k_idx.stride(0);
  const int64_t last_page_ids_s0 = last_page_ids.stride(0);
  const int64_t last_page_ids_s1 = last_page_ids.stride(1);
  const int64_t page_ids_s = page_ids.stride(0);
  const int64_t diff_map_s = diff_map.stride(0);
  const int64_t req_to_tokens_host_s = req_to_tokens_host.stride(0);
  const int64_t load_tokens_s = load_tokens.stride(0);
  const int64_t load_tokens_host_s = load_tokens_host.stride(0);
  const int64_t page_table_s = page_table.stride(0);

  if (diff_map.scalar_type() == at::kShort) {
    if (page_ids.scalar_type() == at::kInt) {
      if (sparse_mask.scalar_type() == at::kBool) {
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
            last_top_k_s0,
            last_top_k_s1,
            top_k_s,
            last_page_ids_s0,
            last_page_ids_s1,
            page_ids_s,
            diff_map_s,
            req_to_tokens_host_s,
            load_tokens_s,
            load_tokens_host_s,
            page_table_s,
            layer_id,
            top_k,
            top_k_page,
            hot_buffer_len,
            hot_buffer_page,
            page_size);
      } else if (sparse_mask.scalar_type() == at::kInt) {
        sparse_page_wise_diff_kernel<int16_t, int32_t, int32_t><<<grid, block, 0, stream>>>(
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
            sparse_mask.data_ptr<int32_t>(),
            page_table.data_ptr<int32_t>(),
            last_top_k_s0,
            last_top_k_s1,
            top_k_s,
            last_page_ids_s0,
            last_page_ids_s1,
            page_ids_s,
            diff_map_s,
            req_to_tokens_host_s,
            load_tokens_s,
            load_tokens_host_s,
            page_table_s,
            layer_id,
            top_k,
            top_k_page,
            hot_buffer_len,
            hot_buffer_page,
            page_size);
      } else {
        sparse_page_wise_diff_kernel<int16_t, int32_t, int64_t><<<grid, block, 0, stream>>>(
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
            sparse_mask.data_ptr<int64_t>(),
            page_table.data_ptr<int32_t>(),
            last_top_k_s0,
            last_top_k_s1,
            top_k_s,
            last_page_ids_s0,
            last_page_ids_s1,
            page_ids_s,
            diff_map_s,
            req_to_tokens_host_s,
            load_tokens_s,
            load_tokens_host_s,
            page_table_s,
            layer_id,
            top_k,
            top_k_page,
            hot_buffer_len,
            hot_buffer_page,
            page_size);
      }
    } else {
      if (sparse_mask.scalar_type() == at::kBool) {
        sparse_page_wise_diff_kernel<int16_t, int64_t, bool><<<grid, block, 0, stream>>>(
            last_top_k_idx.data_ptr<int64_t>(),
            top_k_idx.data_ptr<int32_t>(),
            last_page_ids.data_ptr<int64_t>(),
            page_ids.data_ptr<int64_t>(),
            diff_map.data_ptr<int16_t>(),
            req_to_tokens_host.data_ptr<int64_t>(),
            load_tokens.data_ptr<int64_t>(),
            load_tokens_host.data_ptr<int64_t>(),
            seq_lens.data_ptr<int64_t>(),
            req_pool_indices.data_ptr<int64_t>(),
            sparse_mask.data_ptr<bool>(),
            page_table.data_ptr<int32_t>(),
            last_top_k_s0,
            last_top_k_s1,
            top_k_s,
            last_page_ids_s0,
            last_page_ids_s1,
            page_ids_s,
            diff_map_s,
            req_to_tokens_host_s,
            load_tokens_s,
            load_tokens_host_s,
            page_table_s,
            layer_id,
            top_k,
            top_k_page,
            hot_buffer_len,
            hot_buffer_page,
            page_size);
      } else if (sparse_mask.scalar_type() == at::kInt) {
        sparse_page_wise_diff_kernel<int16_t, int64_t, int32_t><<<grid, block, 0, stream>>>(
            last_top_k_idx.data_ptr<int64_t>(),
            top_k_idx.data_ptr<int32_t>(),
            last_page_ids.data_ptr<int64_t>(),
            page_ids.data_ptr<int64_t>(),
            diff_map.data_ptr<int16_t>(),
            req_to_tokens_host.data_ptr<int64_t>(),
            load_tokens.data_ptr<int64_t>(),
            load_tokens_host.data_ptr<int64_t>(),
            seq_lens.data_ptr<int64_t>(),
            req_pool_indices.data_ptr<int64_t>(),
            sparse_mask.data_ptr<int32_t>(),
            page_table.data_ptr<int32_t>(),
            last_top_k_s0,
            last_top_k_s1,
            top_k_s,
            last_page_ids_s0,
            last_page_ids_s1,
            page_ids_s,
            diff_map_s,
            req_to_tokens_host_s,
            load_tokens_s,
            load_tokens_host_s,
            page_table_s,
            layer_id,
            top_k,
            top_k_page,
            hot_buffer_len,
            hot_buffer_page,
            page_size);
      } else {
        sparse_page_wise_diff_kernel<int16_t, int64_t, int64_t><<<grid, block, 0, stream>>>(
            last_top_k_idx.data_ptr<int64_t>(),
            top_k_idx.data_ptr<int32_t>(),
            last_page_ids.data_ptr<int64_t>(),
            page_ids.data_ptr<int64_t>(),
            diff_map.data_ptr<int16_t>(),
            req_to_tokens_host.data_ptr<int64_t>(),
            load_tokens.data_ptr<int64_t>(),
            load_tokens_host.data_ptr<int64_t>(),
            seq_lens.data_ptr<int64_t>(),
            req_pool_indices.data_ptr<int64_t>(),
            sparse_mask.data_ptr<int64_t>(),
            page_table.data_ptr<int32_t>(),
            last_top_k_s0,
            last_top_k_s1,
            top_k_s,
            last_page_ids_s0,
            last_page_ids_s1,
            page_ids_s,
            diff_map_s,
            req_to_tokens_host_s,
            load_tokens_s,
            load_tokens_host_s,
            page_table_s,
            layer_id,
            top_k,
            top_k_page,
            hot_buffer_len,
            hot_buffer_page,
            page_size);
      }
    }
  } else {
    if (page_ids.scalar_type() == at::kInt) {
      if (sparse_mask.scalar_type() == at::kBool) {
        sparse_page_wise_diff_kernel<int32_t, int32_t, bool><<<grid, block, 0, stream>>>(
            last_top_k_idx.data_ptr<int64_t>(),
            top_k_idx.data_ptr<int32_t>(),
            last_page_ids.data_ptr<int64_t>(),
            page_ids.data_ptr<int32_t>(),
            diff_map.data_ptr<int32_t>(),
            req_to_tokens_host.data_ptr<int64_t>(),
            load_tokens.data_ptr<int64_t>(),
            load_tokens_host.data_ptr<int64_t>(),
            seq_lens.data_ptr<int64_t>(),
            req_pool_indices.data_ptr<int64_t>(),
            sparse_mask.data_ptr<bool>(),
            page_table.data_ptr<int32_t>(),
            last_top_k_s0,
            last_top_k_s1,
            top_k_s,
            last_page_ids_s0,
            last_page_ids_s1,
            page_ids_s,
            diff_map_s,
            req_to_tokens_host_s,
            load_tokens_s,
            load_tokens_host_s,
            page_table_s,
            layer_id,
            top_k,
            top_k_page,
            hot_buffer_len,
            hot_buffer_page,
            page_size);
      } else if (sparse_mask.scalar_type() == at::kInt) {
        sparse_page_wise_diff_kernel<int32_t, int32_t, int32_t><<<grid, block, 0, stream>>>(
            last_top_k_idx.data_ptr<int64_t>(),
            top_k_idx.data_ptr<int32_t>(),
            last_page_ids.data_ptr<int64_t>(),
            page_ids.data_ptr<int32_t>(),
            diff_map.data_ptr<int32_t>(),
            req_to_tokens_host.data_ptr<int64_t>(),
            load_tokens.data_ptr<int64_t>(),
            load_tokens_host.data_ptr<int64_t>(),
            seq_lens.data_ptr<int64_t>(),
            req_pool_indices.data_ptr<int64_t>(),
            sparse_mask.data_ptr<int32_t>(),
            page_table.data_ptr<int32_t>(),
            last_top_k_s0,
            last_top_k_s1,
            top_k_s,
            last_page_ids_s0,
            last_page_ids_s1,
            page_ids_s,
            diff_map_s,
            req_to_tokens_host_s,
            load_tokens_s,
            load_tokens_host_s,
            page_table_s,
            layer_id,
            top_k,
            top_k_page,
            hot_buffer_len,
            hot_buffer_page,
            page_size);
      } else {
        sparse_page_wise_diff_kernel<int32_t, int32_t, int64_t><<<grid, block, 0, stream>>>(
            last_top_k_idx.data_ptr<int64_t>(),
            top_k_idx.data_ptr<int32_t>(),
            last_page_ids.data_ptr<int64_t>(),
            page_ids.data_ptr<int32_t>(),
            diff_map.data_ptr<int32_t>(),
            req_to_tokens_host.data_ptr<int64_t>(),
            load_tokens.data_ptr<int64_t>(),
            load_tokens_host.data_ptr<int64_t>(),
            seq_lens.data_ptr<int64_t>(),
            req_pool_indices.data_ptr<int64_t>(),
            sparse_mask.data_ptr<int64_t>(),
            page_table.data_ptr<int32_t>(),
            last_top_k_s0,
            last_top_k_s1,
            top_k_s,
            last_page_ids_s0,
            last_page_ids_s1,
            page_ids_s,
            diff_map_s,
            req_to_tokens_host_s,
            load_tokens_s,
            load_tokens_host_s,
            page_table_s,
            layer_id,
            top_k,
            top_k_page,
            hot_buffer_len,
            hot_buffer_page,
            page_size);
      }
    } else {
      if (sparse_mask.scalar_type() == at::kBool) {
        sparse_page_wise_diff_kernel<int32_t, int64_t, bool><<<grid, block, 0, stream>>>(
            last_top_k_idx.data_ptr<int64_t>(),
            top_k_idx.data_ptr<int32_t>(),
            last_page_ids.data_ptr<int64_t>(),
            page_ids.data_ptr<int64_t>(),
            diff_map.data_ptr<int32_t>(),
            req_to_tokens_host.data_ptr<int64_t>(),
            load_tokens.data_ptr<int64_t>(),
            load_tokens_host.data_ptr<int64_t>(),
            seq_lens.data_ptr<int64_t>(),
            req_pool_indices.data_ptr<int64_t>(),
            sparse_mask.data_ptr<bool>(),
            page_table.data_ptr<int32_t>(),
            last_top_k_s0,
            last_top_k_s1,
            top_k_s,
            last_page_ids_s0,
            last_page_ids_s1,
            page_ids_s,
            diff_map_s,
            req_to_tokens_host_s,
            load_tokens_s,
            load_tokens_host_s,
            page_table_s,
            layer_id,
            top_k,
            top_k_page,
            hot_buffer_len,
            hot_buffer_page,
            page_size);
      } else if (sparse_mask.scalar_type() == at::kInt) {
        sparse_page_wise_diff_kernel<int32_t, int64_t, int32_t><<<grid, block, 0, stream>>>(
            last_top_k_idx.data_ptr<int64_t>(),
            top_k_idx.data_ptr<int32_t>(),
            last_page_ids.data_ptr<int64_t>(),
            page_ids.data_ptr<int64_t>(),
            diff_map.data_ptr<int32_t>(),
            req_to_tokens_host.data_ptr<int64_t>(),
            load_tokens.data_ptr<int64_t>(),
            load_tokens_host.data_ptr<int64_t>(),
            seq_lens.data_ptr<int64_t>(),
            req_pool_indices.data_ptr<int64_t>(),
            sparse_mask.data_ptr<int32_t>(),
            page_table.data_ptr<int32_t>(),
            last_top_k_s0,
            last_top_k_s1,
            top_k_s,
            last_page_ids_s0,
            last_page_ids_s1,
            page_ids_s,
            diff_map_s,
            req_to_tokens_host_s,
            load_tokens_s,
            load_tokens_host_s,
            page_table_s,
            layer_id,
            top_k,
            top_k_page,
            hot_buffer_len,
            hot_buffer_page,
            page_size);
      } else {
        sparse_page_wise_diff_kernel<int32_t, int64_t, int64_t><<<grid, block, 0, stream>>>(
            last_top_k_idx.data_ptr<int64_t>(),
            top_k_idx.data_ptr<int32_t>(),
            last_page_ids.data_ptr<int64_t>(),
            page_ids.data_ptr<int64_t>(),
            diff_map.data_ptr<int32_t>(),
            req_to_tokens_host.data_ptr<int64_t>(),
            load_tokens.data_ptr<int64_t>(),
            load_tokens_host.data_ptr<int64_t>(),
            seq_lens.data_ptr<int64_t>(),
            req_pool_indices.data_ptr<int64_t>(),
            sparse_mask.data_ptr<int64_t>(),
            page_table.data_ptr<int32_t>(),
            last_top_k_s0,
            last_top_k_s1,
            top_k_s,
            last_page_ids_s0,
            last_page_ids_s1,
            page_ids_s,
            diff_map_s,
            req_to_tokens_host_s,
            load_tokens_s,
            load_tokens_host_s,
            page_table_s,
            layer_id,
            top_k,
            top_k_page,
            hot_buffer_len,
            hot_buffer_page,
            page_size);
      }
    }
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
