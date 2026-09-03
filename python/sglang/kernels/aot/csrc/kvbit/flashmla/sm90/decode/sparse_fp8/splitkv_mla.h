#pragma once

#include "params.h"

namespace sm90::decode::sparse_fp8 {

template <ModelType MODEL_TYPE, int NUM_HEADS>
void run_flash_splitkv_mla_fp8_sparse_kernel(const SparseAttnDecodeParams& params);

template <ModelType MODEL_TYPE, int NUM_HEADS>
void run_flash_splitkv_mla_mxint4_sparse_kernel(const SparseAttnDecodeParams& params);

template <ModelType MODEL_TYPE, int NUM_HEADS>
void run_flash_splitkv_mla_sint4_fp16step_sparse_kernel(const SparseAttnDecodeParams& params);

}  // namespace sm90::decode::sparse_fp8
