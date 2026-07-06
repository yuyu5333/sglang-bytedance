include(FetchContent)

# flash_mla
# 指向项目自有 fork (`yuyu5333/FlashMLA`)，工作分支 `kv2bit-dev`。
# 当前 SHA 1db1482 = upstream abb54777 + 13 commits:
#   c2693a0 [flashmla-kv2bit] add fork-link probe header _fork_banner.h
#   b8a02f0 [flashmla-kv2bit] add dense_fp8_fork_probe.cpp host TU exporting flashmla_fork_probe()
#   a121479 [flashmla-kv2bit] add dense_fp8_packed_entry.cpp scaffold (nullptr fallback bit-exact)
#   6003f52 [flashmla-kv2bit] M3.c.4 stage-1: wire 4 packed tensors into DecodingParams_fp8
#   d5e7b73 [flashmla-kv2bit] M3.c.4 stage-2 contract: kernel-side insertion-point doc + banner bump
#   d3bd701 [flashmla-kv2bit] fix TypeMeta build error in packed entry + extend params to 6 args
#   2956de8 [flashmla-kv2bit] S2-S2: real fused dequant in warp group 1
#   e98e0da [flashmla-kv2bit] fix: bpd.data() -> bpd.data_ptr() in packed entry
#   f1c8e0d [flashmla-kv2bit] (prior baseline)
#   e207047 [flashmla-kv2bit] sparse_attn_decode_interface Stage-1a: 6 packed-FP8 optional args wiring
#   3d1126d [flashmla-kv2bit] (rolled back)
#   996c900 [flashmla-kv2bit] (rolled back)
#   8f33b14 [flashmla-kv2bit] sparse_decode_fwd: const-ref + py::arg/py::none() defaults
#   01a9a93 [flashmla-kv2bit] sparse_decode_fwd: stl.h + py::arg names + py::none() on trailing 6 packed args
#   1db1482 [flashmla-kv2bit] api.cpp: include <torch/extension.h> for torch's specialized optional<at::Tensor> caster
# Stage-1a status: smoke verified (12-arg / 18-arg all-None / kwargs all-None all PASS, byte-identical).
# Stage-1a delta (sparse-path counterpart of dense_fp8 path):
#   * csrc/params.h: SparseAttnDecodeParams extended with 6 packed
#     pointer fields (default nullptr) + packed_kv_block_stride /
#     packed_row_bytes / qk_nope_head_dim / row_bits meta.
#   * csrc/api/sparse_decode.h: sparse_attn_decode_interface adds 6
#     std::optional<at::Tensor> formals (default nullopt). All-None vs
#     all-non-None mode select; shape/dtype validation; params field
#     write-back before impl->run(). sparse sm90/sm100 kernels still
#     ignore packed fields → all-None is byte-identical to before.
#     8f33b14 fix: tile_scheduler_metadata/num_splits switched to
#     const-ref + local mutable copy so pybind11 std::optional caster
#     correctly accepts Python None on torch 2.9.1+cu129.
#   * csrc/api/api.cpp: m.def(sparse_decode_fwd) annotated with
#     py::arg + py::none() defaults for all 13 optional kwargs to
#     make None-acceptance contract explicit at registration time.
#   * flash_mla/flash_mla_interface.py: flash_mla_with_kvcache adds 6
#     kwargs (default None), sparse branch passes them through to
#     flash_mla_cuda.sparse_decode_fwd.
# Stage-2 (DONE for identity calib): sparse_fp8 K-tile fused dequant (bit-unpack + affine).
#   * c8499aa: splitkv_mla.cuh use_packed branch fills staging with bit-unpack +
#     affine dequant per dim-block, writes GMMA-layout sK; identity-calib
#     numerical sanity tests pass (zero-K, code=1/const scale both finite).
#   * 7b4cdf3: generalize bit-unpack to row_bits/dob/bpd table-driven. Inner
#     code accumulates code = sum_{i:dob[i]==d} bit_i << bpd[i] for variable
#     bit-width layouts; identity-calib case (dob[i]=i, bpd[i]=0) reduces to
#     the previous hardcoded 1-bit/dim path (byte-identical, still PASS).
#   * d21761c: switch kernel to per-dim scale (sk_base[d_global]) to match
#     M3.c.* calib layout; relax C++ validator to accept rank-1 scale_kcache
#     [qk_nope] and packed_row_bytes up to 2*qk_nope (rope tail).
#   fc7a138: [M3.c.4 Stage-5] relax KU_CHECK_SHAPE on `kv` last dim so the
#     packed-FP8 path accepts kv tensors whose bytes_per_token is the packed
#     row layout (e.g. 268 B/tok for DSv4 b=2.5) instead of the native 584.
#     Required for drop_shadow=1 wall-storage (true 2.18x KV compression).
#   3374633: [M3.c.4 Stage-5b] relax MODEL1 stride_kv_row kernel-level assert
#     when packed_kcache_ptr is non-null. Required to actually launch the
#     sm90 sparse_fp8 kernel with packed 268 B/tok kv tensors.
#   e1f6037: [flashmla-kv2bit] relax extra_kv bpt check (accept any positive).
#   dc6e30b: [flashmla-kv2bit] sparse_fp8 use_packed: implement R @ x dequant.
#     Bug 2 root-cause fix for M3.c.4 Stage-5 wall+drop_shadow token salad.
#     Per-token full unpack + affine + R @ x in producer warpgroup, matching
#     calibration math nope = (codes*scale + zero) @ R.t().
#   71e17cb: [flashmla-kv2bit] sparse_fp8 use_packed: unified barrier seq for
#     invalid tok. Bug-3 root-cause for Stage-5 token salad after dc6e30b:
#     the invalid-token branch was skipping all 4 NamedBarriers in the
#     per-token loop while valid path went through them. Mixed-topology
#     iterations are fragile w.r.t. consumer wait on bar_k_local_ready.
#     Fix unifies both branches through the same 4-barrier sequence
#     (init/atomicOr/affine/R@x); invalid loads s_x=0 so R@x naturally
#     collapses to staging[t]=0. Mirrors dense_fp8 fork single-branch
#     prologue+prefetch pattern.
#   2ac14f3: [flashmla-kv2bit] sparse_fp8 use_packed: KDUMP diagnostic printf.
#     Temporary instrumentation to localize Stage-5 swa packed token salad:
#     one-shot dump of codes/s_x/staging/sk/zp/R first 4 entries plus
#     pk_block_stride at producer thread (batch_idx=begin, block=start,
#     buf=0, t=0, dim_block=0, idx_in_wg=0, warpgroup_idx=2, blockIdx=0).
#     NamedBarrier::sync added before printf so neighbor staging writes are
#     visible. Will revert after root cause is fixed.
#   460dbd0: [flashmla-kv2bit] KDUMP2 per-band R@x partial sums + R samples.
#     Adds sum_lo/sum_hi (R[0,*]*s_x[*] over d<256 / d>=256), R[0,d] and
#     s_x[d] samples at d in {0,64,128,255,256,300,447}, R[1..3,0..3] for
#     row-major stride verification, and staging[4..15]. Used to localize
#     whether R@x diverges in a specific dim band (orientation/stride bug)
#     vs the staging->sK GMMA-swizzle write being incorrect.
#   f8bfb84: [flashmla-kv2bit] KDUMP3 per-slot staging vs s_sum_dbg + split
#     KDUMP2. Prior KDUMP2 single 41-arg printf produced bogus 1e+143 /
#     1e+218 readbacks for staging[4..15] which were CUDA device-printf
#     varargs overrun artifacts, not real data. KDUMP2 split into <=4-arg
#     chunked printfs (KDUMP2a-h). KDUMP3 adds s_sum_dbg[64] smem capture:
#     every producer thread iwg=0..63 writes its R@x partial sum, then
#     [KDUMP] thread sweeps all 64 (sum, staging) pairs in groups of 4
#     after producer-sync. Discriminates coverage bug (NaN in sum) vs
#     staging clobber (sum != bf16(staging)) vs R@x math bug.
#     KDUMP3 result: R@x is fully correct (full 64-slot coverage, all
#     staging[d] == bf16(s_sum_dbg[d]) within 1 ULP, sum_hi=0 confirms R
#     identity tail). Bug relocated to staging->sK transport.
#   67b3cd7: [flashmla-kv2bit] KDUMP4 staging->sK transport instrumentation.
#     KDUMP4a (single thread abs_token=0/dim_in_block=0/wg=2): dumps
#     staging readback (stg_lo[0..7], stg_hi[0..7]), register readback
#     (val_lo[0..7], val_hi[0..7]), sK readback after store
#     (sK_lo[0..7], sK_hi[0..7]), and absolute smem offsets. Triangulates
#     read-side (staging->reg) vs write-side (reg->sK) vs base offset
#     bugs. KDUMP4b (first 16 producer-wg threads): dumps (idx_in_wg,
#     warp, lane, my_token, abs_token, dim_in_block, idx_in_cluster) to
#     verify the producer tiling bijectively covers expected sK layout.
FetchContent_Declare(
    repo-flashmla
    GIT_REPOSITORY https://github.com/yuyu5333/FlashMLA
    GIT_TAG 9adac11
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
