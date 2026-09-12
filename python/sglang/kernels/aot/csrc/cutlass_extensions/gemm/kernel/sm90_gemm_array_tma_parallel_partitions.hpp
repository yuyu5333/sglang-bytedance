#pragma once

#include "cute/layout.hpp"
#include "cutlass/kernel_launch.h"

namespace cutlass::gemm::kernel {

template <class First, class Second>
struct ParallelPartitionGemm {
  using ArchTag = typename First::ArchTag;
  static constexpr int MaxThreadsPerBlock = First::MaxThreadsPerBlock;
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr bool IsGdcEnabled = true;
  static constexpr int SharedStorageSize =
      First::SharedStorageSize > Second::SharedStorageSize ? First::SharedStorageSize : Second::SharedStorageSize;
  static_assert(MaxThreadsPerBlock == Second::MaxThreadsPerBlock);
  static_assert(cute::size(typename First::ClusterShape{}) == 1);
  static_assert(cute::size(typename Second::ClusterShape{}) == 1);
  static_assert(First::LoadRegisterRequirement == Second::LoadRegisterRequirement);
  static_assert(First::MmaRegisterRequirement == Second::MmaRegisterRequirement);

  struct Params {
    typename First::Params first;
    typename Second::Params second;
  };

  CUTLASS_DEVICE void operator()(Params const& params, char* smem) {
    // The branch is CTA-uniform; each CTA runs exactly one existing protocol.
    if (blockIdx.z == 0) {
      First{}(params.first, smem);
    } else {
      Second{}(params.second, smem);
    }
  }
};

template <class First, class Second>
cutlass::Status launch_parallel_partitions(
    typename First::Params const& first,
    typename Second::Params const& second,
    dim3 grid,
    cudaStream_t stream) {
  using Kernel = ParallelPartitionGemm<First, Second>;
  auto status = cudaFuncSetAttribute(
      cutlass::device_kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel::SharedStorageSize);
  if (status != cudaSuccess) {
    return cutlass::Status::kErrorInternal;
  }
  grid.z = 2;
  return cutlass::kernel_launch<Kernel>(
      grid,
      dim3(Kernel::MaxThreadsPerBlock, 1, 1),
      Kernel::SharedStorageSize,
      stream,
      typename Kernel::Params{first, second},
      true);
}

}  // namespace cutlass::gemm::kernel
