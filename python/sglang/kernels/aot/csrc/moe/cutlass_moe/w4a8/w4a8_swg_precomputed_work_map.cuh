#pragma once

#include <cstdint>

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
  uint32_t tiles_per_worker = 0;
};

template <int TileM, int TileN, class Problem>
__device__ __forceinline__ void swg_group_info(
    Problem const& problem, uint64_t& channel_tiles, uint64_t& token_tiles) {
  channel_tiles =
      swg_div_up(uint64_t(cute::get<0>(problem)), uint64_t(TileM));
  token_tiles = swg_div_up(uint64_t(cute::get<1>(problem)), uint64_t(TileN));
}

template <int TileM, int TileN, class Problem>
__global__ void build_swg_precomputed_work_map_kernel(
    Problem const* problem_shapes,
    int groups,
    uint32_t worker_count,
    uint32_t work_tiles_per_worker,
    uint64_t* work_tiles) {
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
    torch::Device device,
    cudaStream_t stream) {
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
  result.tiles_per_worker = static_cast<uint32_t>(work_tiles_per_worker_u64);
  // A single CTA owns the map build; its threads write a one-to-one partition of
  // global_tile, then publish one sentinel in the next slot of each worker chunk.
  build_swg_precomputed_work_map_kernel<TileM, TileN>
      <<<1, 128, 0, stream>>>(
          problem_shapes,
          groups,
          static_cast<uint32_t>(worker_count_u64),
          result.tiles_per_worker,
          static_cast<uint64_t*>(result.storage.data_ptr()));
  TORCH_CHECK(
      cudaPeekAtLastError() == cudaSuccess,
      "Failed to launch SWG precomputed work-map kernel");
  return result;
}

}  // namespace sgl_kernel::swg_detail
