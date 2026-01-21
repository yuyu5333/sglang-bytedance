#pragma once

#include <torch/extension.h>
#include <optional>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor> quest_retrieval_score_and_combine_indices(
    int64_t bs,
    torch::Tensor seq_lens,
    int64_t page_size,
    torch::Tensor req_to_token,
    torch::Tensor page_k_min,
    torch::Tensor page_k_max,
    torch::Tensor queries,
    torch::Tensor req_pool_indices,
    int64_t num_recent_pages,
    std::optional<int64_t> fixed_topk_page_cnt,
    double sparsity_ratio,
    torch::Tensor sparse_mask);
