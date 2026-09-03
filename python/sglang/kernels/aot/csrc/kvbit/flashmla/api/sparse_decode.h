#pragma once

#include "common.h"
#include "params.h"
#include "sm90/decode/sparse_fp8/splitkv_mla.h"
#include "smxx/decode/combine/combine.h"
#include "smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.h"

// Feature set of sparse decoding kernels
enum class DecodeFeatures : int {
  HEAD_64,
  HEAD_DIM_512,
  MODEL1_KVCACHE_FORMAT,

  ATTN_SINK,
  TOPK_LENGTH,
  EXTRA_KVCACHE,
  EXTRA_TOPK_LENGTH
};

struct DecodeImplMeta {
  int num_sm_parts;
  int fixed_overhead_num_blocks;
  int block_size_topk;
};

class DecodeImplBase : public ImplBase<SparseAttnDecodeParams, DecodeFeatures> {
 public:
  virtual DecodeImplMeta get_meta(int h_q, int s_q) = 0;
};

class Decode_Sm90_Impl : public DecodeImplBase {
  DECLARE_SUPPORTED_FEATURES(
      DecodeFeatures::HEAD_64,
      DecodeFeatures::HEAD_DIM_512,
      DecodeFeatures::MODEL1_KVCACHE_FORMAT,
      DecodeFeatures::ATTN_SINK,
      DecodeFeatures::TOPK_LENGTH,
      DecodeFeatures::EXTRA_KVCACHE,
      DecodeFeatures::EXTRA_TOPK_LENGTH)

 public:
  DecodeImplMeta get_meta(int h_q, int s_q) override {
    Arch arch = Arch();
    return {std::max(arch.num_sms / s_q / (h_q / 64), 1), 5, 64};
  }

 protected:
  void run_(const SparseAttnDecodeParams& params, const std::vector<FeatureT>& required_features) override {
    sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::MODEL1, 64>(params);
  }
};

class Decode_MxInt4_Sm90_Impl : public DecodeImplBase {
  DECLARE_SUPPORTED_FEATURES(
      DecodeFeatures::HEAD_64,
      DecodeFeatures::HEAD_DIM_512,
      DecodeFeatures::MODEL1_KVCACHE_FORMAT,
      DecodeFeatures::ATTN_SINK,
      DecodeFeatures::TOPK_LENGTH,
      DecodeFeatures::EXTRA_KVCACHE,
      DecodeFeatures::EXTRA_TOPK_LENGTH)

 public:
  DecodeImplMeta get_meta(int h_q, int s_q) override {
    Arch arch = Arch();
    return {std::max(arch.num_sms / s_q / (h_q / 64), 1), 5, 64};
  }

 protected:
  void run_(const SparseAttnDecodeParams& params, const std::vector<FeatureT>& required_features) override {
    sm90::decode::sparse_fp8::run_flash_splitkv_mla_mxint4_sparse_kernel<ModelType::MODEL1, 64>(params);
  }
};

class Decode_SInt4Fp16Step_Sm90_Impl : public DecodeImplBase {
  DECLARE_SUPPORTED_FEATURES(
      DecodeFeatures::HEAD_64,
      DecodeFeatures::HEAD_DIM_512,
      DecodeFeatures::MODEL1_KVCACHE_FORMAT,
      DecodeFeatures::ATTN_SINK,
      DecodeFeatures::TOPK_LENGTH,
      DecodeFeatures::EXTRA_KVCACHE,
      DecodeFeatures::EXTRA_TOPK_LENGTH)

 public:
  DecodeImplMeta get_meta(int h_q, int s_q) override {
    Arch arch = Arch();
    return {std::max(arch.num_sms / s_q / (h_q / 64), 1), 5, 64};
  }

 protected:
  void run_(const SparseAttnDecodeParams& params, const std::vector<FeatureT>& required_features) override {
    sm90::decode::sparse_fp8::run_flash_splitkv_mla_sint4_fp16step_sparse_kernel<ModelType::MODEL1, 64>(params);
  }
};

// ---------------------------------------------------------------------------
// [M3.c.4 Stage-1a] sparse-path packed buffer validator.
//
// Mirrors validate_packed_buffers() in csrc/extension/sm90/dense_fp8/
// dense_fp8_packed_entry.cpp but adapted to sparse path's kv layout
// (`kv` is [num_blocks, page_block_size, h_kv=1, bytes_per_token]).
// Stage-1a: kernel does NOT yet read these fields. Buffer wiring here
// only ensures call-site ABI is stable and `params.*_ptr` slots are
// populated for the next-stage S2-S2 fused-dequant kernel.
// ---------------------------------------------------------------------------
inline void sparse_validate_packed_buffers(
    const at::Tensor& packed_kcache,
    const at::Tensor& scale_kcache,
    const at::Tensor& R_matrix,
    const at::Tensor& zero_point,
    int kv_num_rows,
    int* out_qk_nope_head_dim,
    int* out_packed_row_bytes) {
  KU_CHECK_DEVICE(packed_kcache);
  KU_CHECK_DEVICE(scale_kcache);
  KU_CHECK_DEVICE(R_matrix);
  KU_CHECK_DEVICE(zero_point);

  TORCH_CHECK(packed_kcache.dtype() == at::kByte, "packed_kcache must be uint8");
  TORCH_CHECK(scale_kcache.dtype() == at::kFloat, "scale_kcache must be float32");
  TORCH_CHECK(R_matrix.dtype() == at::kBFloat16, "fixed-BU4 R_matrix must be bfloat16");
  TORCH_CHECK(zero_point.dtype() == at::kFloat, "zero_point must be float32");

  KU_CHECK_CONTIGUOUS(packed_kcache);
  KU_CHECK_CONTIGUOUS(scale_kcache);
  KU_CHECK_CONTIGUOUS(R_matrix);
  KU_CHECK_CONTIGUOUS(zero_point);

  TORCH_CHECK(R_matrix.dim() == 2, "R_matrix must be rank-2, got ", R_matrix.dim());
  const auto R0 = R_matrix.size(0);
  const auto R1 = R_matrix.size(1);
  TORCH_CHECK(R0 == R1, "R_matrix must be square, got [", R0, ", ", R1, "]");
  const int qk_nope = static_cast<int>(R0);
  TORCH_CHECK(qk_nope == 448, "fixed-BU4 qk_nope_head_dim must be 448, got ", qk_nope);

  TORCH_CHECK(
      zero_point.dim() == 1 && zero_point.size(0) == qk_nope,
      "zero_point must be [qk_nope_head_dim], got [",
      zero_point.sizes(),
      "]");

  TORCH_CHECK(
      packed_kcache.dim() == 2, "packed_kcache must be rank-2 [num_rows, row_bytes], got ", packed_kcache.dim());
  const auto pk_rows = packed_kcache.size(0);
  const auto pk_cols = packed_kcache.size(1);
  TORCH_CHECK(
      pk_rows == kv_num_rows,
      "packed_kcache row count ",
      pk_rows,
      " must equal kv num_rows ",
      kv_num_rows,
      " (= num_blocks * page_block_size)");
  TORCH_CHECK(
      pk_cols == 380,
      "fixed-BU4 packed_kcache row_bytes must be 380 "
      "(224-byte NoPE + 28-byte header + 128-byte RoPE), got ",
      pk_cols);

  TORCH_CHECK(
      scale_kcache.dim() == 1 && scale_kcache.size(0) == qk_nope,
      "fixed-BU4 scale_kcache must be [448], got ",
      scale_kcache.sizes());

  *out_qk_nope_head_dim = qk_nope;
  *out_packed_row_bytes = static_cast<int>(pk_cols);
}

inline void sparse_validate_mxint4_buffer(const at::Tensor& packed_kcache, int kv_num_rows, const char* name) {
  KU_CHECK_DEVICE(packed_kcache);
  TORCH_CHECK(packed_kcache.dtype() == at::kByte, name, " must be uint8");
  TORCH_CHECK(packed_kcache.dim() == 2, name, " must be rank-2 [num_rows, 360], got ", packed_kcache.dim());
  TORCH_CHECK(packed_kcache.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(
      packed_kcache.size(0) == kv_num_rows,
      name,
      " row count ",
      packed_kcache.size(0),
      " must equal KV num_rows ",
      kv_num_rows);
  TORCH_CHECK(
      packed_kcache.size(1) == 360,
      name,
      " row_bytes must be 360 (224-byte signed nibbles + 7-byte UE8M0 + "
      "1-byte padding + 128-byte BF16 RoPE), got ",
      packed_kcache.size(1));
}

inline void sparse_validate_sint4_fp16step_buffer(const at::Tensor& packed_kcache, int kv_num_rows, const char* name) {
  KU_CHECK_DEVICE(packed_kcache);
  TORCH_CHECK(packed_kcache.dtype() == at::kByte, name, " must be uint8");
  TORCH_CHECK(packed_kcache.dim() == 2, name, " must be rank-2 [num_rows, 368], got ", packed_kcache.dim());
  TORCH_CHECK(packed_kcache.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(
      packed_kcache.size(0) == kv_num_rows,
      name,
      " row count ",
      packed_kcache.size(0),
      " must equal KV num_rows ",
      kv_num_rows);
  TORCH_CHECK(
      packed_kcache.size(1) == 368,
      name,
      " row_bytes must be 368 (224-byte signed nibbles + 14-byte FP16 steps + "
      "2-byte padding + 128-byte BF16 RoPE), got ",
      packed_kcache.size(1));
}

static std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>>
sparse_attn_decode_interface(
    const at::Tensor& q,                           // [b, s_q, h_q, d_qk]
    const at::Tensor& kv,                          // [num_blocks, page_block_size, h_k, d_qk]
    const at::Tensor& indices,                     // [b, s_q, topk]
    const std::optional<at::Tensor>& topk_length,  // [b, s_q]
    const std::optional<at::Tensor>& attn_sink,    // [h_q]
    // [Stage-1a fix] non-const ref had broken pybind11 std::optional caster
    // for Python None on torch 2.9.1 build; switch to const-ref + local
    // mutable copy below to restore None-acceptance (matches 71c7379 behavior).
    const std::optional<at::Tensor>& tile_scheduler_metadata_in,  // num_sm_parts x (DecodingSchedMetaSize/4)
    const std::optional<at::Tensor>& num_splits_in,               // batch_size + 1
    const std::optional<at::Tensor>& extra_kv,
    const std::optional<at::Tensor>& extra_indices,
    const std::optional<at::Tensor>& extra_topk_length,
    int d_v,
    float sm_scale,
    // ---- [M3.c.4 Stage-1a] packed-FP8 device-side wiring (default None). ----
    // All six optional. Mode selection:
    //   * all-6 None     -> bit-exact pre-stage-1a behavior (kernel ignores
    //                       packed fields; SparseAttnDecodeParams defaults
    //                       leave them nullptr/0).
    //   * all-6 non-None -> Stage-1a wiring path: shape/dtype validate +
    //                       write pointers into params.* slots; the
    //                       current sparse_fp8 kernel still doesn't read
    //                       them, so output is still bit-exact vs the
    //                       all-None path. Stage-2 will swap the K-tile
    //                       cp.async with fused unpack+R@x+FP8 convert
    //                       and start consuming these fields.
    //   * mixed          -> hard fail (TORCH_CHECK).
    const std::optional<at::Tensor>& packed_kcache = std::nullopt,
    const std::optional<at::Tensor>& scale_kcache = std::nullopt,
    const std::optional<at::Tensor>& R_matrix = std::nullopt,
    const std::optional<at::Tensor>& zero_point = std::nullopt,
    const std::optional<at::Tensor>& dim_of_bit = std::nullopt,
    const std::optional<at::Tensor>& bitpos_in_dim = std::nullopt,
    // [M3.c.4 Stage-5 Route G step 5] uniform-bit layout switch.
    // bit_uniform == 0 -> legacy variable-bit layout (default;
    // backward-compat, byte-identical to pre-step-5). bit_uniform > 0
    // -> every nope dim uses bit_uniform contiguous bits; per-token
    // affine (min, range) lives in fp16 header at end of packed row.
    int64_t bit_uniform = 0,
    const std::optional<at::Tensor>& q_for_extra = std::nullopt,
    bool q_nope_is_folded = false,
    bool identity_tail_bypass = false,
    bool debug_u32_packed_load = false,
    // [c4c128-packed] Extra (c4/c128 sink) packed KV cache byte buffer.
    // When provided (alongside the six SWA packed tensors), the extra
    // blocks read packed rows + fused dequant using the SHARED calib
    // (scale/R/zero/dim_of_bit/bitpos/row_bytes/bit_uniform). Only the
    // byte buffer + its per-page stride differ from SWA. None -> extra
    // blocks stay on the dense FP8 (shadow/native) path (byte-identical).
    const std::optional<at::Tensor>& extra_packed_kcache = std::nullopt,
    bool is_mxint4 = false,
    bool is_sint4_fp16step = false) {
  using bf16 = cutlass::bfloat16_t;

  TORCH_CHECK(!(is_mxint4 && is_sint4_fp16step), "packed KV format selectors are mutually exclusive");

  // [Stage-1a fix] Re-introduce mutable local copies so the rest of this
  // function (which used to take non-const refs) keeps working unchanged.
  // The function may emplace freshly-allocated tensors below when callers
  // pass None, then return them out via the std::tuple<...> at the end.
  std::optional<at::Tensor> tile_scheduler_metadata = tile_scheduler_metadata_in;
  std::optional<at::Tensor> num_splits = num_splits_in;

  KU_CHECK_NDIM(q, 4);
  KU_CHECK_NDIM(kv, 4);
  KU_CHECK_NDIM(indices, 3);

  int b = q.size(0);
  int s_q = q.size(1);
  int h_q = q.size(2);
  int d_qk = q.size(3);
  int num_blocks = kv.size(0);
  int page_block_size = kv.size(1);
  int h_kv = kv.size(2);
  int topk = indices.size(2);

  bool have_topk_length = topk_length.has_value();
  bool have_extra_kcache = extra_kv.has_value();
  bool have_extra_topk_length = extra_topk_length.has_value();
  bool have_attn_sink = attn_sink.has_value();

  int extra_num_blocks = 0, extra_page_block_size = 0, extra_topk = 0;
  if (have_extra_kcache) {
    extra_num_blocks = extra_kv->size(0);
    extra_page_block_size = extra_kv->size(1);
  }
  if (extra_indices.has_value()) {
    extra_topk = extra_indices->size(-1);
  }

  // metadata sanity check
  TORCH_CHECK(b > 0);
  TORCH_CHECK(s_q > 0);
  TORCH_CHECK(h_q == 64, "KVBit sparse decode supports exactly 64 query heads, got ", h_q);
  TORCH_CHECK(h_kv == 1, "Currently only MQA (i.e. h_kv == 1) is supported for sparse decoding");
  TORCH_CHECK(
      d_qk == 512,
      "KVBit sparse decode supports only MODEL1 head_size_k=512 "
      "(448 NoPE + 64 RoPE), got ",
      d_qk);
  TORCH_CHECK(d_v == 512, "KVBit sparse decode supports only head_size_v=512");
  TORCH_CHECK(topk > 0);
  TORCH_CHECK(bit_uniform == 4, "KVBit sparse decode supports only bit_uniform=4");
  TORCH_CHECK(
      identity_tail_bypass, "KVBit sparse decode requires identity_tail_bypass=true for the fixed 448+64 layout");
  TORCH_CHECK(
      !q_nope_is_folded, "KVBit sparse decode does not support folded Q because it changes the BF16 rounding path");
  TORCH_CHECK(!q_for_extra.has_value(), "KVBit sparse decode does not support q_for_extra");

  if (have_extra_kcache) {
    TORCH_CHECK(
        extra_indices.has_value(),
        "extra_indices_in_kvcache must be provided when extra_kcache is provided for sparse attention");
  } else {
    TORCH_CHECK(
        !extra_indices.has_value(), "extra_indices_in_kvcache must not be provided when extra_k_cache is not provided");
    TORCH_CHECK(
        !extra_topk_length.has_value(), "extra_topk_length must not be provided when extra_k_cache is not provided");
  }

  // Check device
  KU_CHECK_DEVICE(q);
  KU_CHECK_DEVICE(kv);
  KU_CHECK_DEVICE(indices);
  KU_CHECK_DEVICE(topk_length);
  KU_CHECK_DEVICE(attn_sink);
  KU_CHECK_DEVICE(tile_scheduler_metadata);
  KU_CHECK_DEVICE(num_splits);
  KU_CHECK_DEVICE(extra_kv);
  KU_CHECK_DEVICE(extra_indices);
  KU_CHECK_DEVICE(extra_topk_length);
  KU_CHECK_DEVICE(q_for_extra);
  KU_CHECK_DEVICE(packed_kcache);
  KU_CHECK_DEVICE(scale_kcache);
  KU_CHECK_DEVICE(R_matrix);
  KU_CHECK_DEVICE(zero_point);
  KU_CHECK_DEVICE(dim_of_bit);
  KU_CHECK_DEVICE(bitpos_in_dim);
  KU_CHECK_DEVICE(extra_packed_kcache);

  const auto q_device = q.device();
  auto check_same_device = [&](const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(
        tensor.device() == q_device, name, " must be on the same device as q (", q_device, "), got ", tensor.device());
  };
  check_same_device(kv, "kv");
  check_same_device(indices, "indices");
  if (topk_length.has_value()) check_same_device(*topk_length, "topk_length");
  if (attn_sink.has_value()) check_same_device(*attn_sink, "attn_sink");
  if (tile_scheduler_metadata.has_value()) check_same_device(*tile_scheduler_metadata, "tile_scheduler_metadata");
  if (num_splits.has_value()) check_same_device(*num_splits, "num_splits");
  if (extra_kv.has_value()) check_same_device(*extra_kv, "extra_kv");
  if (extra_indices.has_value()) check_same_device(*extra_indices, "extra_indices");
  if (extra_topk_length.has_value()) check_same_device(*extra_topk_length, "extra_topk_length");
  if (packed_kcache.has_value()) check_same_device(*packed_kcache, "packed_kcache");
  if (scale_kcache.has_value()) check_same_device(*scale_kcache, "scale_kcache");
  if (R_matrix.has_value()) check_same_device(*R_matrix, "R_matrix");
  if (zero_point.has_value()) check_same_device(*zero_point, "zero_point");
  if (dim_of_bit.has_value()) check_same_device(*dim_of_bit, "dim_of_bit");
  if (bitpos_in_dim.has_value()) check_same_device(*bitpos_in_dim, "bitpos_in_dim");
  if (extra_packed_kcache.has_value()) check_same_device(*extra_packed_kcache, "extra_packed_kcache");

  at::cuda::CUDAGuard device_guard{(char)q.get_device()};
  Arch arch = Arch();
  TORCH_CHECK(arch.is_sm90a(), "KVBit sparse decode supports only SM90");

  // Check data type
  KU_CHECK_DTYPE(q, torch::kBFloat16);
  KU_CHECK_DTYPE(q_for_extra, torch::kBFloat16);
  TORCH_CHECK(
      kv.dtype() == torch::kFloat8_e4m3fn || kv.dtype() == torch::kInt8 || kv.dtype() == torch::kUInt8,
      "key must have dtype fp8_e4m3fn, int8 or uint8");
  if (extra_kv.has_value()) {
    TORCH_CHECK(
        extra_kv->dtype() == torch::kFloat8_e4m3fn || extra_kv->dtype() == torch::kInt8 ||
            extra_kv->dtype() == torch::kUInt8,
        "extra k cache must have dtype fp8_e4m3fn, int8 or uint8");
  }
  KU_CHECK_DTYPE(indices, torch::kInt32);
  KU_CHECK_DTYPE(topk_length, torch::kInt32);
  KU_CHECK_DTYPE(attn_sink, torch::kFloat32);
  KU_CHECK_DTYPE(tile_scheduler_metadata, torch::kInt32);
  KU_CHECK_DTYPE(num_splits, torch::kInt32);
  KU_CHECK_DTYPE(extra_indices, torch::kInt32);
  KU_CHECK_DTYPE(extra_topk_length, torch::kInt32);

  // Check layout
  KU_CHECK_LAST_DIM_CONTIGUOUS(q);
  KU_CHECK_LAST_DIM_CONTIGUOUS(q_for_extra);
  KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
  KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
  KU_CHECK_CONTIGUOUS(topk_length);
  KU_CHECK_CONTIGUOUS(attn_sink);

  KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
  KU_CHECK_CONTIGUOUS(num_splits);

  KU_CHECK_LAST_DIM_CONTIGUOUS(extra_kv);
  KU_CHECK_LAST_DIM_CONTIGUOUS(extra_indices);
  KU_CHECK_CONTIGUOUS(extra_topk_length);

  // Check shape
  KU_CHECK_SHAPE(q, b, s_q, h_q, d_qk);
  {
    // The shape-carrier aliases the fixed packed row ABI.
    const int bytes_per_token = is_mxint4 ? 360 : (is_sint4_fp16step ? 368 : 380);
    KU_CHECK_SHAPE(kv, num_blocks, page_block_size, h_kv, bytes_per_token);
    if (extra_kv.has_value()) {
      const int extra_bpt = static_cast<int>(extra_kv->size(3));
      TORCH_CHECK(
          extra_bpt == bytes_per_token,
          is_mxint4           ? "MXINT4 extra_kv bytes_per_token must be 360, got "
          : is_sint4_fp16step ? "SINT4 FP16-step extra_kv bytes_per_token must be 368, got "
                              : "fixed-BU4 extra_kv bytes_per_token must be 380, got ",
          extra_bpt);
      KU_CHECK_SHAPE(extra_kv, extra_num_blocks, extra_page_block_size, h_kv, extra_bpt);
      TORCH_CHECK(
          extra_kv->stride(1) == extra_bpt,
          "The whole block must be contiguous when is_fp8_cache is True for extra kv cache");
    }
    TORCH_CHECK(
        kv.stride(1) == bytes_per_token, "The whole block must be contiguous when is_fp8_cache is True for kv cache");
  }
  KU_CHECK_SHAPE(indices, b, s_q, topk);
  KU_CHECK_SHAPE(topk_length, b);
  KU_CHECK_SHAPE(attn_sink, h_q);
  KU_CHECK_SHAPE(extra_indices, b, s_q, extra_topk);
  KU_CHECK_SHAPE(extra_topk_length, b);

  auto opts = q.options();

  at::Tensor out = torch::empty({b, s_q, h_q, d_v}, opts);
  at::Tensor lse = torch::empty({b, s_q, h_q}, opts.dtype(at::kFloat));

  constexpr ModelType model_type = ModelType::MODEL1;

  std::vector<DecodeFeatures> features;
  features.push_back(DecodeFeatures::HEAD_64);
  features.push_back(DecodeFeatures::HEAD_DIM_512);
  features.push_back(DecodeFeatures::MODEL1_KVCACHE_FORMAT);
  if (have_attn_sink) {
    features.push_back(DecodeFeatures::ATTN_SINK);
  }
  if (have_topk_length) {
    features.push_back(DecodeFeatures::TOPK_LENGTH);
  }
  if (have_extra_kcache) {
    features.push_back(DecodeFeatures::EXTRA_KVCACHE);
  }
  if (have_extra_topk_length) {
    features.push_back(DecodeFeatures::EXTRA_TOPK_LENGTH);
  }

  DecodeImplBase* impl = is_mxint4           ? static_cast<DecodeImplBase*>(new Decode_MxInt4_Sm90_Impl())
                         : is_sint4_fp16step ? static_cast<DecodeImplBase*>(new Decode_SInt4Fp16Step_Sm90_Impl())
                                             : static_cast<DecodeImplBase*>(new Decode_Sm90_Impl());

  DecodeImplMeta impl_meta = impl->get_meta(h_q, s_q);

  SparseAttnDecodeParams params = {
      b,
      s_q,
      h_q,
      h_kv,
      d_qk,
      d_v,
      sm_scale,
      sm_scale * LOG_2_E,
      num_blocks,
      page_block_size,
      topk,
      model_type,

      (bf16*)q.data_ptr(),
      ku::get_optional_tensor_ptr<bf16>(q_for_extra),
      (bf16*)kv.data_ptr(),
      (int*)indices.data_ptr(),
      ku::get_optional_tensor_ptr<int>(topk_length),
      ku::get_optional_tensor_ptr<float>(attn_sink),
      (float*)lse.data_ptr(),
      (bf16*)out.data_ptr(),

      extra_num_blocks,
      extra_page_block_size,
      extra_topk,
      ku::get_optional_tensor_ptr<bf16>(extra_kv),
      ku::get_optional_tensor_ptr<int>(extra_indices),
      ku::get_optional_tensor_ptr<int>(extra_topk_length),

      int64_stride_to_int(q.stride(0)),
      int64_stride_to_int(q.stride(1)),
      int64_stride_to_int(q.stride(2)),
      q_for_extra.has_value() ? int64_stride_to_int(q_for_extra->stride(0)) : 0,
      q_for_extra.has_value() ? int64_stride_to_int(q_for_extra->stride(1)) : 0,
      q_for_extra.has_value() ? int64_stride_to_int(q_for_extra->stride(2)) : 0,
      int64_stride_to_int(kv.stride(0)),
      int64_stride_to_int(kv.stride(1)),
      int64_stride_to_int(indices.stride(0)),
      int64_stride_to_int(indices.stride(1)),
      int64_stride_to_int(lse.stride(0)),
      int64_stride_to_int(lse.stride(1)),
      int64_stride_to_int(out.stride(0)),
      int64_stride_to_int(out.stride(1)),
      int64_stride_to_int(out.stride(2)),

      have_extra_kcache ? int64_stride_to_int(extra_kv->stride(0)) : 0,
      have_extra_kcache ? int64_stride_to_int(extra_kv->stride(1)) : 0,
      have_extra_kcache ? int64_stride_to_int(extra_indices->stride(0)) : 0,
      have_extra_kcache ? int64_stride_to_int(extra_indices->stride(1)) : 0,
      at::cuda::getCurrentCUDAStream().stream()};

  // Get MLA metadata if necessary
  at::Tensor o_accum, lse_accum;
  if (!tile_scheduler_metadata.has_value()) {
    tile_scheduler_metadata =
        torch::empty({impl_meta.num_sm_parts, sizeof(DecodingSchedMeta) / 4}, opts.dtype(torch::kInt32));
    num_splits = torch::empty({b + 1}, opts.dtype(torch::kInt32));
    KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
    KU_CHECK_CONTIGUOUS(num_splits);

    GetDecodeSchedMetaParams get_sched_meta_params = {
        b,
        s_q,
        impl_meta.block_size_topk,
        impl_meta.fixed_overhead_num_blocks,
        topk,
        extra_topk,
        ku::get_optional_tensor_ptr<int>(topk_length),
        ku::get_optional_tensor_ptr<int>(extra_topk_length),
        nullptr,
        (DecodingSchedMeta*)tile_scheduler_metadata->data_ptr(),
        num_splits->data_ptr<int>(),
        impl_meta.num_sm_parts,
        at::cuda::getCurrentCUDAStream().stream()};
    smxx::decode::run_get_decoding_sched_meta_kernel(get_sched_meta_params);
  }
  // Stick the metadata pointers to `params`
  KU_CHECK_DEVICE(tile_scheduler_metadata);
  KU_CHECK_DEVICE(num_splits);
  KU_CHECK_DTYPE(tile_scheduler_metadata, torch::kInt32);
  KU_CHECK_DTYPE(num_splits, torch::kInt32);
  KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
  KU_CHECK_CONTIGUOUS(num_splits);
  KU_CHECK_SHAPE(tile_scheduler_metadata, impl_meta.num_sm_parts, sizeof(DecodingSchedMeta) / sizeof(int));
  KU_CHECK_SHAPE(num_splits, b + 1);
  params.tile_scheduler_metadata_ptr = (DecodingSchedMeta*)tile_scheduler_metadata->data_ptr();
  params.num_splits_ptr = num_splits->data_ptr<int>();
  params.num_sm_parts = impl_meta.num_sm_parts;

  // Allocate intermediate buffers for split-KV
  const int total_num_splits = b + impl_meta.num_sm_parts;
  lse_accum = torch::empty({total_num_splits, s_q, h_q}, opts.dtype(at::kFloat));
  o_accum = torch::empty({total_num_splits, s_q, h_q, d_v}, opts.dtype(at::kFloat));
  KU_CHECK_CONTIGUOUS(lse_accum);
  KU_CHECK_CONTIGUOUS(o_accum);
  params.lse_accum = lse_accum.data_ptr<float>();
  params.o_accum = o_accum.data_ptr<float>();
  params.stride_lse_accum_split = int64_stride_to_int(lse_accum.stride(0));
  params.stride_lse_accum_s_q = int64_stride_to_int(lse_accum.stride(1));
  params.stride_o_accum_split = int64_stride_to_int(o_accum.stride(0));
  params.stride_o_accum_s_q = int64_stride_to_int(o_accum.stride(1));
  params.stride_o_accum_h_q = int64_stride_to_int(o_accum.stride(2));

  // ---- [M3.c.4 Stage-1a] packed-FP8 wiring (mode selection). ----
  // Mirror dense_fp8_packed_entry.cpp fork pattern. Stage-1a: kernel
  // does not yet consume these fields, so byte-identical regression
  // vs pre-Stage-1a is asserted by the all-None branch (which is the
  // path covered by tests/test_flash_mla_dense_decoding.py sparse
  // smokes today). Stage-2 will swap the K-tile cp.async to read
  // these fields and emit FP8 on the fly.
  {
    const int num_packed_present = (packed_kcache.has_value() ? 1 : 0) + (scale_kcache.has_value() ? 1 : 0) +
                                   (R_matrix.has_value() ? 1 : 0) + (zero_point.has_value() ? 1 : 0) +
                                   (dim_of_bit.has_value() ? 1 : 0) + (bitpos_in_dim.has_value() ? 1 : 0);
    TORCH_CHECK(
        (is_mxint4 || is_sint4_fp16step) ? num_packed_present == 1 : num_packed_present == 6,
        is_mxint4           ? "KVBit MXINT4 sparse decode requires only packed_kcache"
        : is_sint4_fp16step ? "KVBit SINT4 FP16-step sparse decode requires only packed_kcache"
                            : "KVBit sparse decode requires all six packed tensors "
                              "(packed_kcache, scale_kcache, R_matrix, zero_point, dim_of_bit, bitpos_in_dim)",
        "; got non-None count=",
        num_packed_present);
    TORCH_CHECK(packed_kcache.has_value(), "KVBit sparse decode requires packed_kcache");
    TORCH_CHECK(
        !have_extra_kcache || extra_packed_kcache.has_value(),
        "KVBit sparse decode requires extra_packed_kcache when extra_kv is present");

    if (is_mxint4) {
      const at::Tensor& pk = packed_kcache.value();
      sparse_validate_mxint4_buffer(pk, num_blocks * page_block_size, "packed_kcache");
      params.packed_kcache_ptr = pk.data_ptr();
      params.packed_row_bytes = 360;
      params.packed_kv_block_stride = static_cast<int64_t>(page_block_size) * 360;
      params.qk_nope_head_dim = 448;
      params.row_bits = 1792;
      params.bit_uniform = 4;
      params.identity_tail_bypass = 1;
      params.uniform_group_size = 64;
      params.uniform_num_groups = 7;
      params.uniform_header_bytes = 8;

      if (extra_packed_kcache.has_value()) {
        TORCH_CHECK(extra_kv.has_value(), "extra_packed_kcache requires extra_kv");
        const at::Tensor& epk = extra_packed_kcache.value();
        sparse_validate_mxint4_buffer(epk, extra_num_blocks * extra_page_block_size, "extra_packed_kcache");
        params.extra_packed_kcache_ptr = epk.data_ptr();
        params.extra_packed_kv_block_stride = static_cast<int64_t>(extra_page_block_size) * 360;
      }
    } else if (is_sint4_fp16step) {
      const at::Tensor& pk = packed_kcache.value();
      sparse_validate_sint4_fp16step_buffer(pk, num_blocks * page_block_size, "packed_kcache");
      params.packed_kcache_ptr = pk.data_ptr();
      params.packed_row_bytes = 368;
      params.packed_kv_block_stride = static_cast<int64_t>(page_block_size) * 368;
      params.qk_nope_head_dim = 448;
      params.row_bits = 1792;
      params.bit_uniform = 4;
      params.identity_tail_bypass = 1;
      params.uniform_group_size = 64;
      params.uniform_num_groups = 7;
      params.uniform_header_bytes = 16;

      if (extra_packed_kcache.has_value()) {
        TORCH_CHECK(extra_kv.has_value(), "extra_packed_kcache requires extra_kv");
        const at::Tensor& epk = extra_packed_kcache.value();
        sparse_validate_sint4_fp16step_buffer(epk, extra_num_blocks * extra_page_block_size, "extra_packed_kcache");
        params.extra_packed_kcache_ptr = epk.data_ptr();
        params.extra_packed_kv_block_stride = static_cast<int64_t>(extra_page_block_size) * 368;
      }
    } else if (num_packed_present == 6) {
      const at::Tensor& pk = packed_kcache.value();
      const at::Tensor& sk = scale_kcache.value();
      const at::Tensor& Rm = R_matrix.value();
      const at::Tensor& zp = zero_point.value();
      const at::Tensor& dob = dim_of_bit.value();
      const at::Tensor& bpd = bitpos_in_dim.value();

      int qk_nope_head_dim_val = 0;
      int packed_row_bytes_val = 0;
      const int kv_num_rows = num_blocks * page_block_size;
      sparse_validate_packed_buffers(pk, sk, Rm, zp, kv_num_rows, &qk_nope_head_dim_val, &packed_row_bytes_val);

      KU_CHECK_DEVICE(dob);
      KU_CHECK_DEVICE(bpd);
      KU_CHECK_CONTIGUOUS(dob);
      KU_CHECK_CONTIGUOUS(bpd);
      TORCH_CHECK(dob.dtype() == at::kInt, "dim_of_bit must be int32");
      TORCH_CHECK(bpd.dtype() == at::kInt, "bitpos_in_dim must be int32");
      TORCH_CHECK(dob.dim() == 1 && bpd.dim() == 1, "dim_of_bit and bitpos_in_dim must be rank-1");
      TORCH_CHECK(dob.size(0) == bpd.size(0), "dim_of_bit and bitpos_in_dim must have same length");
      const int row_bits_val = static_cast<int>(dob.size(0));
      TORCH_CHECK(
          row_bits_val == 1792, "fixed-BU4 metadata must contain exactly 1792 bits (448 * 4), got ", row_bits_val);

      params.packed_kcache_ptr = pk.data_ptr();
      params.scale_kcache_ptr = reinterpret_cast<float*>(sk.data_ptr());
      params.R_matrix_bf16_ptr = Rm.data_ptr();
      params.zero_point_ptr = reinterpret_cast<float*>(zp.data_ptr());
      params.dim_of_bit_ptr = reinterpret_cast<int*>(dob.data_ptr());
      params.bitpos_in_dim_ptr = reinterpret_cast<int*>(bpd.data_ptr());
      params.packed_row_bytes = packed_row_bytes_val;
      params.packed_kv_block_stride =
          static_cast<int64_t>(page_block_size) * static_cast<int64_t>(packed_row_bytes_val);
      params.qk_nope_head_dim = qk_nope_head_dim_val;
      params.row_bits = row_bits_val;

      params.bit_uniform = 4;
      params.q_nope_is_folded = 0;
      params.identity_tail_bypass = 1;
      params.debug_u32_packed_load = debug_u32_packed_load ? 1 : 0;
      params.uniform_group_size = 64;
      params.uniform_num_groups = 7;
      params.uniform_header_bytes = 28;

      // [c4c128-packed] Wire the extra (c4/c128) packed byte buffer.
      // It shares the SWA calib validated above; only the byte
      // buffer + per-page stride differ. Requires extra_kv present
      // (extra pool active) and same packed_row_bytes as SWA.
      if (extra_packed_kcache.has_value()) {
        const at::Tensor& epk = extra_packed_kcache.value();
        KU_CHECK_DEVICE(epk);
        TORCH_CHECK(epk.dtype() == at::kByte, "extra_packed_kcache must be uint8");
        KU_CHECK_CONTIGUOUS(epk);
        TORCH_CHECK(epk.dim() == 2, "extra_packed_kcache must be rank-2 [num_rows, row_bytes], got ", epk.dim());
        TORCH_CHECK(extra_kv.has_value(), "extra_packed_kcache requires extra_kv to be present");
        const auto epk_cols = epk.size(1);
        TORCH_CHECK(
            static_cast<int>(epk_cols) == packed_row_bytes_val,
            "extra_packed_kcache row_bytes ",
            epk_cols,
            " must equal SWA packed_row_bytes ",
            packed_row_bytes_val);
        const auto epk_rows = epk.size(0);
        const int extra_kv_num_rows = extra_num_blocks * extra_page_block_size;
        TORCH_CHECK(
            static_cast<int64_t>(epk_rows) == extra_kv_num_rows,
            "extra_packed_kcache row count ",
            epk_rows,
            " must equal extra_kv num_rows ",
            extra_kv_num_rows);
        params.extra_packed_kcache_ptr = epk.data_ptr();
        params.extra_packed_kv_block_stride =
            static_cast<int64_t>(extra_page_block_size) * static_cast<int64_t>(packed_row_bytes_val);
      }
    }
  }

  impl->run(params, features);

  CombineParams combine_params = {
      b,
      s_q,
      h_q,
      d_v,

      params.lse,
      params.out,
      params.stride_lse_b,
      params.stride_lse_s_q,
      params.stride_o_b,
      params.stride_o_s_q,
      params.stride_o_h_q,

      params.lse_accum,
      params.o_accum,
      params.stride_lse_accum_split,
      params.stride_lse_accum_s_q,
      params.stride_o_accum_split,
      params.stride_o_accum_s_q,
      params.stride_o_accum_h_q,

      params.tile_scheduler_metadata_ptr,
      params.num_splits_ptr,
      params.num_sm_parts,

      ku::get_optional_tensor_ptr<float>(attn_sink),
      at::cuda::getCurrentCUDAStream().stream()};
  smxx::decode::run_flash_mla_combine_kernel<bf16>(combine_params);

  delete impl;

  return {out, lse.transpose(1, 2), tile_scheduler_metadata, num_splits};
}
