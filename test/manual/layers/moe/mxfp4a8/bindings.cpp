#include "sgl_kernel_ops.h"

torch::Tensor pack_mxfp4a8_stage_weights_sm90(torch::Tensor const&, torch::Tensor const&);

TORCH_LIBRARY_EXPAND(EXPERIMENT_NAMESPACE, m) {
  m.def("core", &cutlass_mxfp4a8_fused_moe_core);
  m.def("metadata", &get_cutlass_w4a8_moe_mm_data_with_permutation);
  m.def("int4_mm", &cutlass_w4a8_moe_mm);
  m.def("mxfp4_mm", &cutlass_mxfp4a8_moe_mm);
  m.def("pack_stage_weights", &pack_mxfp4a8_stage_weights_sm90);
}
