cmake_minimum_required(VERSION 3.26)
project(mxfp4a8_experiment LANGUAGES CXX CUDA)

# An isolated SM90 build, without replacing the installed sgl_kernel package.
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(CUDAToolkit REQUIRED)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_PREFIX}")
find_package(Torch REQUIRED)

set(EXPERIMENT_NAMESPACE mxfp4a8_baseline CACHE STRING "Unique torch.ops namespace")
set(CUTLASS_ROOT "" CACHE PATH "Existing CUTLASS source checkout")
set(FLASHINFER_ROOT "" CACHE PATH "Existing FlashInfer source checkout")
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)
set(AOT "${REPO_ROOT}/python/sglang/kernels/aot")
set(MOE "${AOT}/csrc/moe/cutlass_moe/w4a8")
if(NOT EXISTS "${CUTLASS_ROOT}/include/cutlass/cutlass.h")
  message(FATAL_ERROR "Set CUTLASS_ROOT to the production CUTLASS dependency")
endif()
if(NOT EXISTS "${FLASHINFER_ROOT}/include/flashinfer/activation.cuh")
  message(FATAL_ERROR "Set FLASHINFER_ROOT to the production FlashInfer dependency")
endif()

add_library(${EXPERIMENT_NAMESPACE} SHARED
  bindings.cpp
  "${MOE}/w4a8_grouped_mm_c3x.cu"
  "${MOE}/w4a8_moe_data.cu"
  "${MOE}/scaled_mm_entry.cu"
  "${AOT}/csrc/gemm/per_token_quant_fp8.cu"
  "${AOT}/csrc/elementwise/activation.cu")
set_target_properties(${EXPERIMENT_NAMESPACE} PROPERTIES
  CXX_STANDARD 17 CUDA_STANDARD 17 CUDA_ARCHITECTURES OFF
  CXX_VISIBILITY_PRESET hidden POSITION_INDEPENDENT_CODE ON)
target_include_directories(${EXPERIMENT_NAMESPACE} PRIVATE
  "${AOT}/include" "${AOT}/csrc" "${Python3_INCLUDE_DIRS}"
  "${CUTLASS_ROOT}/include" "${CUTLASS_ROOT}/tools/util/include"
  "${FLASHINFER_ROOT}/include" "${FLASHINFER_ROOT}/csrc")
target_link_libraries(${EXPERIMENT_NAMESPACE} PRIVATE "${TORCH_LIBRARIES}" CUDA::cuda_driver)
target_compile_definitions(${EXPERIMENT_NAMESPACE} PRIVATE
  EXPERIMENT_NAMESPACE=${EXPERIMENT_NAMESPACE}
  CUTLASS_MIXED_GEMM_FP4_FP8_PREPROCESSED_SIGNS=1
  CUTE_USE_PACKED_TUPLE=1 CUTLASS_ENABLE_TENSOR_CORE_MMA=1
  FLASHINFER_ENABLE_F16 FLASHINFER_ENABLE_BF16 FLASHINFER_ENABLE_FP8
  FLASHINFER_ENABLE_FP8_E4M3 FLASHINFER_ENABLE_FP8_E5M2 ENABLE_NVFP4=1)
target_compile_options(${EXPERIMENT_NAMESPACE} PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-gencode=arch=compute_90a,code=sm_90a;--expt-relaxed-constexpr;--expt-extended-lambda;--use_fast_math;--threads=8;-Xcompiler=-fvisibility=hidden;-Xptxas=-v;-lineinfo>)
