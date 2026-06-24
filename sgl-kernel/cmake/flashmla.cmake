include(FetchContent)

# flash_mla
# 指向项目自有 fork (`yuyu5333/FlashMLA`)，工作分支 `kv2bit-dev`。
# 当前 SHA 2956de8 = upstream abb54777 + 7 commits:
#   c2693a0 [flashmla-kv2bit] add fork-link probe header _fork_banner.h
#   b8a02f0 [flashmla-kv2bit] add dense_fp8_fork_probe.cpp host TU exporting flashmla_fork_probe()
#   a121479 [flashmla-kv2bit] add dense_fp8_packed_entry.cpp scaffold (nullptr fallback bit-exact)
#   6003f52 [flashmla-kv2bit] M3.c.4 stage-1: wire 4 packed tensors into DecodingParams_fp8
#   d5e7b73 [flashmla-kv2bit] M3.c.4 stage-2 contract: kernel-side insertion-point doc + banner bump
#   d3bd701 [flashmla-kv2bit] fix TypeMeta build error in packed entry + extend params to 6 args
#   2956de8 [flashmla-kv2bit] S2-S2: real fused dequant in warp group 1
#   e98e0da [flashmla-kv2bit] fix: bpd.data() -> bpd.data_ptr() in packed entry
# S2-S2 delta:
#   * flash_fwd_mla_kernel.h: warp group 1 KV-load path replaced with real
#     fused dequant — bit-unpack (atomicOr) + affine + R@x + FP8 convert
#     + rope BF16→FP8, written to sK via dense smem staging buffer
#   * flash_mla.h: DecodingParams_fp8 extended with dim_of_bit_ptr /
#     bitpos_in_dim_ptr / row_bits
#   * dense_fp8_packed_entry.cpp: host entry extended to 6 packed args
#   * SharedStorageMLA: added smem_k_dense_nope (64 x 512 FP8 staging)
#   * _fork_banner.h: kForkBanner 20260624 -> 20260625
# Dense path unchanged (bit-exact).
FetchContent_Declare(
    repo-flashmla
    GIT_REPOSITORY https://github.com/yuyu5333/FlashMLA
    GIT_TAG f1c8e0d77ee2d22bf357f9cfb27c2503401a7473
    GIT_SHALLOW OFF
)
FetchContent_Populate(repo-flashmla)

set(FLASHMLA_CUDA_FLAGS
    "--expt-relaxed-constexpr"
    "--expt-extended-lambda"
    "--use_fast_math"

    "-Xcudafe=--diag_suppress=177"   # variable was declared but never referenced
)

# The FlashMLA kernels only work on hopper and require CUDA 12.4 or later.
# Only build FlashMLA kernels if we are building for something compatible with
# sm90a
if(${CUDA_VERSION} VERSION_GREATER 12.4)
    list(APPEND FLASHMLA_CUDA_FLAGS
        "-gencode=arch=compute_90a,code=sm_90a"
    )
endif()
if(${CUDA_VERSION} VERSION_GREATER 12.8)
    list(APPEND FLASHMLA_CUDA_FLAGS
        "-gencode=arch=compute_100a,code=sm_100a"
    )
endif()
if(${CUDA_VERSION} VERSION_GREATER_EQUAL "13.0")
    # Patch FlashMLA sources for SM103a support.
    # These patches are only needed (and only valid) with CUDA 13+.

    # Patch utils.h: widen IS_SM100 to cover the full SM100 family.
    # Newer FlashMLA versions use csrc/utils.h.
    set(FLASHMLA_UTILS_FILE "${repo-flashmla_SOURCE_DIR}/csrc/utils.h")
    file(READ "${FLASHMLA_UTILS_FILE}" FLASHMLA_UTILS_CONTENT)
    string(REPLACE
        "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
#define IS_SM100 1"
        "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && (__CUDA_ARCH__ < 1100)
#define IS_SM100 1"
        FLASHMLA_UTILS_CONTENT "${FLASHMLA_UTILS_CONTENT}")
    file(WRITE "${FLASHMLA_UTILS_FILE}" "${FLASHMLA_UTILS_CONTENT}")
    message(STATUS "Patched utils.h for SM103a support")

    # Patch cutlass/arch/config.h: add SM103 architecture defines.
    # The new block is inserted right before the existing "// SM101 and SM101a"
    # anchor in the upstream header.
    set(CUTLASS_CONFIG_FILE "${repo-flashmla_SOURCE_DIR}/csrc/cutlass/include/cutlass/arch/config.h")
    file(READ "${CUTLASS_CONFIG_FILE}" CUTLASS_CONFIG_CONTENT)
    string(FIND "${CUTLASS_CONFIG_CONTENT}" "SM103" SM103_FOUND)
    if(SM103_FOUND EQUAL -1)
        string(REPLACE
"// SM101 and SM101a"
"// SM103 and SM103a
#if !CUTLASS_CLANG_CUDA && (__CUDACC_VER_MAJOR__ >= 13)
  #define CUTLASS_ARCH_MMA_SM103_SUPPORTED 1
  #if (!defined(CUTLASS_ARCH_MMA_SM103_ENABLED) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030)
    #define CUTLASS_ARCH_MMA_SM103_ENABLED 1
    #if !defined(CUTLASS_ARCH_MMA_SM100A_ENABLED)
      #define CUTLASS_ARCH_MMA_SM100A_ENABLED 1
    #endif
    #if !defined(CUTLASS_ARCH_MMA_SM100F_ENABLED)
      #define CUTLASS_ARCH_MMA_SM100F_ENABLED 1
    #endif
  #endif
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

// SM101 and SM101a"
            CUTLASS_CONFIG_CONTENT "${CUTLASS_CONFIG_CONTENT}")
        file(WRITE "${CUTLASS_CONFIG_FILE}" "${CUTLASS_CONFIG_CONTENT}")
        message(STATUS "Patched cutlass/arch/config.h for SM103a support")
    else()
        message(STATUS "cutlass/arch/config.h already patched for SM103a")
    endif()

    list(APPEND FLASHMLA_CUDA_FLAGS
        "-gencode=arch=compute_103a,code=sm_103a"
    )
endif()


set(FlashMLA_SOURCES
    "csrc/flashmla_extension.cc"

    # Compatibility shim for sgl-kernel torch.ops API.
    ${repo-flashmla_SOURCE_DIR}/csrc/python_api.cpp

    # Decode metadata/combine kernels.
    ${repo-flashmla_SOURCE_DIR}/csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/smxx/decode/combine/combine.cu

    # sm90 dense decode.
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90/decode/dense/instantiations/fp16.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90/decode/dense/instantiations/bf16.cu

    # sm90 sparse decode.
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90/decode/sparse_fp8/instantiations/model1_persistent_h64.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90/decode/sparse_fp8/instantiations/model1_persistent_h128.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90/decode/sparse_fp8/instantiations/v32_persistent_h64.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90/decode/sparse_fp8/instantiations/v32_persistent_h128.cu

    # sm90 sparse prefill.
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90/prefill/sparse/fwd.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90/prefill/sparse/instantiations/phase1_k512.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90/prefill/sparse/instantiations/phase1_k512_topklen.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90/prefill/sparse/instantiations/phase1_k576.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90/prefill/sparse/instantiations/phase1_k576_topklen.cu

    # sm100 dense prefill/bwd.
    ${repo-flashmla_SOURCE_DIR}/csrc/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm100/prefill/dense/fmha_cutlass_bwd_sm100.cu

    # sm100 sparse prefill.
    ${repo-flashmla_SOURCE_DIR}/csrc/sm100/prefill/sparse/fwd/head64/instantiations/phase1_k512.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm100/prefill/sparse/fwd/head64/instantiations/phase1_k576.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm100/prefill/sparse/fwd/head128/instantiations/phase1_k512.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm100/prefill/sparse/fwd/head128/instantiations/phase1_k576.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_prefill_k512.cu

    # sm100 sparse decode.
    ${repo-flashmla_SOURCE_DIR}/csrc/sm100/decode/head64/instantiations/v32.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm100/decode/head64/instantiations/model1.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512.cu

    ${repo-flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/dense_fp8_python_api.cpp
    ${repo-flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/flash_fwd_mla_fp8_sm90.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/flash_fwd_mla_metadata.cu

    # Fork dev loop probe TU (kv2bit-dev): exports flashmla_fork_probe() -> int64.
    ${repo-flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/dense_fp8_fork_probe.cpp

    # Packed_fp8 entry scaffold (kv2bit-dev): exports fwd_kvcache_mla_packed_fp8.
    # nullptr fallback (4 占位 tensor 全 None) 直通 dense_fp8 kernel (bit-exact).
    ${repo-flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/dense_fp8_packed_entry.cpp
)

Python_add_library(flashmla_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${FlashMLA_SOURCES})
target_compile_options(flashmla_ops PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++20>
    $<$<COMPILE_LANGUAGE:CUDA>:-std=c++20>
    $<$<COMPILE_LANGUAGE:CUDA>:${FLASHMLA_CUDA_FLAGS}>
)
target_include_directories(flashmla_ops PRIVATE
    ${repo-flashmla_SOURCE_DIR}/csrc
    ${repo-flashmla_SOURCE_DIR}/csrc/kerutils/include
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90
    ${repo-flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/
    ${repo-flashmla_SOURCE_DIR}/csrc/cutlass/include
    ${repo-flashmla_SOURCE_DIR}/csrc/cutlass/tools/util/include
)

target_link_libraries(flashmla_ops PRIVATE ${TORCH_LIBRARIES} c10 cuda)

install(TARGETS flashmla_ops LIBRARY DESTINATION "sgl_kernel")

target_compile_definitions(flashmla_ops PRIVATE)
