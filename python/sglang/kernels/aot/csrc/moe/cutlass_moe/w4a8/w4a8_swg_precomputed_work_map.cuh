#pragma once

#include <cstdint>
#include <type_traits>

#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass_extensions/gemm/kernel/sm90_tile_scheduler_group_precomputed.hpp"

namespace sgl_kernel::swg_detail {

using SwgWorkTile = cutlass::gemm::kernel::detail::PrecomputedGroupWorkTile;

constexpr int kSwgWorkMapMaxSwizzle = 8;

CUTLASS_HOST_DEVICE uint64_t swg_div_up(uint64_t value, uint64_t divisor) {
  return (value + divisor - 1) / divisor;
}

template <int TileM, int TileN>
uint64_t swg_max_work_tiles(int groups, uint64_t total_tokens, uint64_t channels) {
  if (groups <= 0 || total_tokens == 0 || channels == 0) {
    return 0;
  }

  uint64_t const channel_tiles = swg_div_up(channels, uint64_t(TileM));
  uint64_t const nonempty_groups =
      uint64_t(groups) < total_tokens ? uint64_t(groups) : total_tokens;
  uint64_t const extra_tokens = total_tokens - nonempty_groups;
  // One token opens a tile row; every further tile row in that group costs TileN tokens.
  uint64_t const max_token_tiles = nonempty_groups + extra_tokens / uint64_t(TileN);
  return channel_tiles * max_token_tiles;
}

struct SwgPrecomputedWorkMap {
  torch::Tensor storage;
  torch::Tensor prebuilt_tma_desc_a;
  torch::Tensor prebuilt_tma_desc_b;
  uint32_t worker_count = 0;
  uint32_t tiles_per_worker = 0;
};

constexpr int kSwgPrebuiltTmaDescriptorCount = 2;
constexpr size_t kSwgPrebuiltTmaDescriptorScratchBytes =
    kSwgPrebuiltTmaDescriptorCount * sizeof(cute::TmaDescriptor);

CUTE_DEVICE void swg_publish_prebuilt_tma_descriptor(
    cute::TmaDescriptor const* gmem_desc,
    cute::TmaDescriptor& smem_desc,
    int publisher_warp) {
  if ((threadIdx.x >> 5) == publisher_warp) {
    __syncwarp();
    if (cute::elect_one_sync()) {
      cute::tma_desc_commit_group();
      cute::tma_desc_wait_group();
    }
    cute::tma_descriptor_cp_fence_release(gmem_desc, smem_desc);
    __syncwarp();
  }
}

template <class MainloopParams, class Problem>
__device__ __forceinline__ void swg_build_prebuilt_tma_descriptors(
    MainloopParams const& mainloop_params,
    Problem const& problem,
    int group,
    cute::TmaDescriptor* smem_descs,
    cute::TmaDescriptor* prebuilt_tma_desc_a,
    cute::TmaDescriptor* prebuilt_tma_desc_b) {
  if (group == 0) {
    cute::TmaDescriptor& smem_desc = smem_descs[0];
    if (threadIdx.x == 0) {
      constexpr int MaxTensorRank = 5;
      cute::array<uint32_t, MaxTensorRank> shape_a = {1, 1, 1, 1, 1};
      cute::array<uint64_t, MaxTensorRank> stride_a = {0, 0, 0, 0, 0};
      using PtrA = std::remove_reference_t<decltype(mainloop_params.ptr_A[group])>;
      PtrA ptr_a = nullptr;
      uint32_t const M = static_cast<uint32_t>(cute::get<0>(problem));
      uint32_t const K = static_cast<uint32_t>(cute::get<2>(problem));
      auto d_a = mainloop_params.ptr_dA[group];
      auto stride_m = cute::get<0>(d_a);
      auto stride_k = cute::get<1>(d_a);
      int64_t const term_m = static_cast<int64_t>(M) * static_cast<int64_t>(stride_m);
      int64_t const term_k = static_cast<int64_t>(K) * static_cast<int64_t>(stride_k);
      int64_t const stride_l = term_m > term_k ? term_m : term_k;
      auto tensor_a = cute::make_tensor(
          ptr_a,
          cute::make_layout(
              cute::make_shape(M, K, static_cast<uint32_t>(mainloop_params.num_groups)),
              cute::make_stride(stride_m, stride_k, stride_l)));

      smem_desc = *mainloop_params.tma_load_a.get_tma_descriptor();
      cute::tma_descriptor_replace_addr_in_shared_mem(smem_desc, mainloop_params.ptr_A[0]);
      cute::detail::fill_tma_gmem_shape_stride(
          mainloop_params.tma_load_a, tensor_a, shape_a, stride_a);
      using ElementA = std::remove_cv_t<std::remove_pointer_t<PtrA>>;
      for (uint64_t& stride : stride_a) {
        stride = (stride * cutlass::sizeof_bits<ElementA>::value) / 8;
      }
      cute::tma_descriptor_replace_dims_strides_in_shared_mem(smem_desc, shape_a, stride_a);
    }
    swg_publish_prebuilt_tma_descriptor(prebuilt_tma_desc_a, smem_desc, 0);
  }

  if (cute::get<1>(problem) == 0) {
    return;
  }

  cute::TmaDescriptor& smem_desc = smem_descs[1];
  if (threadIdx.x == 32) {
    constexpr int MaxTensorRank = 5;
    cute::array<uint32_t, MaxTensorRank> shape_b = {1, 1, 1, 1, 1};
    cute::array<uint64_t, MaxTensorRank> stride_b = {0, 0, 0, 0, 0};
    using PtrB = std::remove_reference_t<decltype(mainloop_params.ptr_B[group])>;
    PtrB ptr_b = nullptr;
    uint32_t const N = static_cast<uint32_t>(cute::get<1>(problem));
    uint32_t const K = static_cast<uint32_t>(cute::get<2>(problem));
    auto d_b = mainloop_params.ptr_dB[group];
    auto stride_n = cute::get<0>(d_b);
    auto stride_k = cute::get<1>(d_b);
    auto tensor_b = cute::make_tensor(
        ptr_b,
        cute::make_layout(
            cute::make_shape(N, K, uint32_t(1)),
            cute::make_stride(stride_n, stride_k, int64_t(0))));

    smem_desc = *mainloop_params.tma_load_b.get_tma_descriptor();
    cute::tma_descriptor_replace_addr_in_shared_mem(smem_desc, mainloop_params.ptr_B[group]);
    cute::detail::fill_tma_gmem_shape_stride(
        mainloop_params.tma_load_b, tensor_b, shape_b, stride_b);
    using ElementB = std::remove_cv_t<std::remove_pointer_t<PtrB>>;
    for (uint64_t& stride : stride_b) {
      stride = (stride * cutlass::sizeof_bits<ElementB>::value) / 8;
    }
    cute::tma_descriptor_replace_dims_strides_in_shared_mem(smem_desc, shape_b, stride_b);
  }
  swg_publish_prebuilt_tma_descriptor(
      prebuilt_tma_desc_b + group, smem_desc, 1);
}

template <int TileM, int TileN, class Problem>
__device__ __forceinline__ void swg_group_info(
    Problem const& problem, uint64_t& channel_tiles, uint64_t& token_tiles) {
  channel_tiles =
      swg_div_up(uint64_t(cute::get<0>(problem)), uint64_t(TileM));
  token_tiles = swg_div_up(uint64_t(cute::get<1>(problem)), uint64_t(TileN));
}

template <int TileM, int TileN, class Problem, class MainloopParams>
__global__ void build_swg_precomputed_work_map_kernel(
    Problem const* problem_shapes,
    int groups,
    uint32_t worker_count,
    uint32_t work_tiles_per_worker,
    uint64_t* work_tiles,
    MainloopParams mainloop_params,
    cute::TmaDescriptor* prebuilt_tma_desc_a,
    cute::TmaDescriptor* prebuilt_tma_desc_b) {
  extern __shared__ __align__(64) unsigned char shared_storage[];
  auto* smem_descs = reinterpret_cast<cute::TmaDescriptor*>(shared_storage);
  __shared__ uint64_t total_tiles;
  __shared__ uint64_t tiles_per_worker;
  __shared__ uint64_t group_start;
  __shared__ uint64_t group_channel_tiles;
  __shared__ uint64_t group_token_tiles;

  if (threadIdx.x == 0) {
    total_tiles = 0;
    for (int group = 0; group < groups; ++group) {
      uint64_t channel_tiles = 0;
      uint64_t token_tiles = 0;
      swg_group_info<TileM, TileN>(
          problem_shapes[group], channel_tiles, token_tiles);
      total_tiles += channel_tiles * token_tiles;
    }
    tiles_per_worker =
        total_tiles == 0 ? 1 : swg_div_up(total_tiles, uint64_t(worker_count));
    group_start = 0;
  }
  __syncthreads();

  for (int group = 0; group < groups; ++group) {
    if (threadIdx.x == 0) {
      swg_group_info<TileM, TileN>(
          problem_shapes[group], group_channel_tiles, group_token_tiles);
    }
    __syncthreads();

    swg_build_prebuilt_tma_descriptors(
        mainloop_params,
        problem_shapes[group],
        group,
        smem_descs,
        prebuilt_tma_desc_a,
        prebuilt_tma_desc_b);
    __syncthreads();

    uint64_t const group_tiles = group_channel_tiles * group_token_tiles;
    for (uint64_t local_tile = uint64_t(threadIdx.x); local_tile < group_tiles;
         local_tile += uint64_t(blockDim.x)) {
      uint64_t const global_tile = group_start + local_tile;
      uint64_t const worker = global_tile / tiles_per_worker;
      uint64_t const worker_tile = global_tile % tiles_per_worker;
      uint64_t const channel_tile = local_tile % group_channel_tiles;
      uint64_t const token_tile = local_tile / group_channel_tiles;
      work_tiles[worker * uint64_t(work_tiles_per_worker) + worker_tile] =
          SwgWorkTile::pack(channel_tile, token_tile, uint64_t(group));
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      group_start += group_tiles;
    }
    __syncthreads();
  }

  for (uint64_t worker = uint64_t(threadIdx.x); worker < uint64_t(worker_count);
       worker += uint64_t(blockDim.x)) {
    uint64_t const worker_start = worker * tiles_per_worker;
    uint64_t const worker_tile_count =
        worker_start < total_tiles
            ? ((total_tiles - worker_start) < tiles_per_worker
                   ? total_tiles - worker_start
                   : tiles_per_worker)
            : 0;
    work_tiles[worker * uint64_t(work_tiles_per_worker) + worker_tile_count] =
        SwgWorkTile::Invalid;
  }
}

template <class Gemm, class Problem>
SwgPrecomputedWorkMap build_swg_precomputed_work_map(
    Problem const* problem_shapes,
    int groups,
    uint64_t total_tokens,
    uint64_t channels,
    cutlass::KernelHardwareInfo const& hw_info,
    torch::Device device) {
  using Scheduler = typename Gemm::GemmKernelScaleOnly::TileScheduler;
  using SchedulerParams = typename Scheduler::Params;
  using RasterOrderOptions = typename SchedulerParams::RasterOrderOptions;

  cutlass::gemm::GemmCoord const cluster_shape(1, 1, 1);
  dim3 const problem_blocks =
      SchedulerParams::get_tiled_cta_shape_mnl(
          cluster_shape, static_cast<uint32_t>(hw_info.sm_count), 1);
  dim3 const grid_shape = SchedulerParams::get_grid_shape(
      problem_blocks,
      cluster_shape,
      hw_info,
      kSwgWorkMapMaxSwizzle,
      RasterOrderOptions::AlongM,
      true);
  uint64_t const worker_count_u64 =
      uint64_t(grid_shape.x) * uint64_t(grid_shape.y) * uint64_t(grid_shape.z);
  TORCH_CHECK(worker_count_u64 > 0, "SWG precomputed work map requires workers");
  TORCH_CHECK(
      worker_count_u64 <= uint64_t(UINT32_MAX),
      "SWG precomputed work map worker count exceeds uint32");
  TORCH_CHECK(
      groups <= int(SwgWorkTile::ExpertMask + 1),
      "SWG precomputed work map expert index exceeds packed limit");

  constexpr int TileM = Gemm::SingleWarpgroupTileM;
  constexpr int TileN = Gemm::SingleWarpgroupTileN;
  uint64_t const max_tiles =
      swg_max_work_tiles<TileM, TileN>(groups, total_tokens, channels);
  TORCH_CHECK(
      swg_div_up(channels, uint64_t(TileM)) <= SwgWorkTile::ChannelMask + 1,
      "SWG precomputed work map channel index exceeds packed limit");
  TORCH_CHECK(
      swg_div_up(total_tokens, uint64_t(TileN)) <= SwgWorkTile::TokenMask + 1,
      "SWG precomputed work map token index exceeds packed limit");

  uint64_t const work_tiles_per_worker_u64 =
      swg_div_up(max_tiles, worker_count_u64) + 1;
  uint64_t const capacity = worker_count_u64 * work_tiles_per_worker_u64;
  TORCH_CHECK(
      work_tiles_per_worker_u64 <= uint64_t(UINT32_MAX),
      "SWG precomputed work map worker chunk exceeds uint32");
  TORCH_CHECK(
      capacity <= uint64_t(INT64_MAX) / sizeof(uint64_t),
      "SWG precomputed work map capacity exceeds tensor size");

  auto options = torch::TensorOptions().dtype(torch::kUInt8).device(device);
  SwgPrecomputedWorkMap result;
  result.storage =
      torch::empty(int64_t(capacity * sizeof(uint64_t)), options);
  result.prebuilt_tma_desc_a =
      torch::empty(int64_t(sizeof(cute::TmaDescriptor)), options);
  result.prebuilt_tma_desc_b = torch::empty(
      int64_t((groups > 0 ? groups : 1) * sizeof(cute::TmaDescriptor)), options);
  result.worker_count = static_cast<uint32_t>(worker_count_u64);
  result.tiles_per_worker = static_cast<uint32_t>(work_tiles_per_worker_u64);
  return result;
}

template <class Gemm, class Problem, class MainloopParams>
void launch_swg_precomputed_work_map(
    SwgPrecomputedWorkMap const& work_map,
    Problem const* problem_shapes,
    int groups,
    MainloopParams const& mainloop_params,
    cudaStream_t stream) {
  constexpr int TileM = Gemm::SingleWarpgroupTileM;
  constexpr int TileN = Gemm::SingleWarpgroupTileN;
  // A single CTA owns the map build; its threads write a one-to-one partition of
  // global_tile, publish one sentinel per worker, and prebuild the grouped A/B
  // TMA descriptors required by the prescale collective.
  build_swg_precomputed_work_map_kernel<TileM, TileN>
      <<<1, 128, kSwgPrebuiltTmaDescriptorScratchBytes, stream>>>(
          problem_shapes,
          groups,
          work_map.worker_count,
          work_map.tiles_per_worker,
          static_cast<uint64_t*>(work_map.storage.data_ptr()),
          mainloop_params,
          static_cast<cute::TmaDescriptor*>(work_map.prebuilt_tma_desc_a.data_ptr()),
          static_cast<cute::TmaDescriptor*>(work_map.prebuilt_tma_desc_b.data_ptr()));
  TORCH_CHECK(
      cudaPeekAtLastError() == cudaSuccess,
      "Failed to launch SWG precomputed work-map kernel");
}

}  // namespace sgl_kernel::swg_detail
