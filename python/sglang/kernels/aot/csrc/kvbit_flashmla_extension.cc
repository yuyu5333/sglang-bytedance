/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <torch/all.h>
#include <torch/library.h>

#include "api/sparse_decode.h"
#include "sgl_kernel_ops.h"

static std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>> kvbit_sparse_decode_fwd(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    const std::optional<at::Tensor>& topk_length,
    const std::optional<at::Tensor>& attn_sink,
    std::optional<at::Tensor> tile_scheduler_metadata,
    std::optional<at::Tensor> num_splits,
    const std::optional<at::Tensor>& extra_kv,
    const std::optional<at::Tensor>& extra_indices,
    const std::optional<at::Tensor>& extra_topk_length,
    int64_t d_v,
    double sm_scale,
    const at::Tensor& packed_kcache,
    const at::Tensor& scale_kcache,
    const at::Tensor& r_matrix,
    const at::Tensor& zero_point,
    const at::Tensor& dim_of_bit,
    const at::Tensor& bitpos_in_dim,
    int64_t bit_uniform,
    bool q_nope_is_folded,
    bool identity_tail_bypass,
    bool debug_u32_packed_load,
    const std::optional<at::Tensor>& extra_packed_kcache) {
  return sparse_attn_decode_interface(
      q,
      kv,
      indices,
      topk_length,
      attn_sink,
      tile_scheduler_metadata,
      num_splits,
      extra_kv,
      extra_indices,
      extra_topk_length,
      static_cast<int>(d_v),
      static_cast<float>(sm_scale),
      packed_kcache,
      scale_kcache,
      r_matrix,
      zero_point,
      dim_of_bit,
      bitpos_in_dim,
      bit_uniform,
      std::nullopt,
      q_nope_is_folded,
      identity_tail_bypass,
      debug_u32_packed_load,
      extra_packed_kcache);
}

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def(
      "kvbit_sparse_decode_fwd(Tensor q, Tensor kv, Tensor indices, Tensor? topk_length, Tensor? attn_sink, "
      "Tensor? tile_scheduler_metadata, Tensor? num_splits, Tensor? extra_kv, Tensor? extra_indices, "
      "Tensor? extra_topk_length, int d_v, float sm_scale, Tensor packed_kcache, Tensor scale_kcache, "
      "Tensor r_matrix, Tensor zero_point, Tensor dim_of_bit, Tensor bitpos_in_dim, int bit_uniform, "
      "bool q_nope_is_folded, bool identity_tail_bypass, bool debug_u32_packed_load, "
      "Tensor? extra_packed_kcache) -> (Tensor, Tensor, Tensor?, Tensor?)");
  m.impl("kvbit_sparse_decode_fwd", torch::kCUDA, &kvbit_sparse_decode_fwd);
}

REGISTER_EXTENSION(kvbit_flashmla_ops)
