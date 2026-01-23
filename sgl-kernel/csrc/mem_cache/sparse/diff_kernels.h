#pragma once

#include <torch/all.h>

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
    int64_t page_size);

