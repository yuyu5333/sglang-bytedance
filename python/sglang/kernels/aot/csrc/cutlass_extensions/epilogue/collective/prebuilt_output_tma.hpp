#pragma once

#include <cstdint>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"

namespace sgl_kernel::w4a8_detail {

template <class Base>
struct PrebuiltOutputTmaEpilogue : Base {
  using BaseArguments = typename Base::Arguments;
  using BaseParams = typename Base::Params;
  using TensorStorage = typename Base::TensorStorage;
  using TensorMapStorage = typename Base::TensorMapStorage;
  using ElementC = typename Base::ElementC;
  using ElementD = typename Base::ElementD;
  using StrideC = typename Base::StrideC;
  using StrideD = typename Base::StrideD;
  using ThreadArguments = decltype(BaseArguments{}.thread);

  struct Arguments : BaseArguments {
    cute::TmaDescriptor* prebuilt_output_tma = nullptr;

    Arguments() = default;
    Arguments(ThreadArguments thread, ElementC const** c, StrideC dc, ElementD** d, StrideD dd)
        : BaseArguments{thread, c, dc, d, dd} {}
  };

  struct Params : BaseParams {
    cute::TmaDescriptor* prebuilt_output_tma = nullptr;
  };

  template <class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const& problem, Arguments const& args, void* workspace) {
    Params result;
    static_cast<BaseParams&>(result) = Base::to_underlying_arguments(problem, args, workspace);
    result.prebuilt_output_tma = args.prebuilt_output_tma;
    return result;
  }

  CUTLASS_DEVICE PrebuiltOutputTmaEpilogue(Params const& params, TensorStorage& storage)
      : Base(params, storage), descriptors_(params.prebuilt_output_tma) {}

  CUTLASS_DEVICE auto store_init(Params const&, TensorMapStorage&, int32_t, int32_t, int32_t) {
    return cute::make_tuple(static_cast<cute::TmaDescriptor const*>(nullptr));
  }

  template <bool IsLoad, class ProblemShape, class TensorMap>
  CUTLASS_DEVICE void tensormaps_perform_update(
      TensorMapStorage& storage,
      Params const& params,
      TensorMap& descriptor,
      ProblemShape problem,
      int32_t group,
      int32_t warp_group) {
    if constexpr (IsLoad) {
      Base::template tensormaps_perform_update<IsLoad>(storage, params, descriptor, problem, group, warp_group);
    } else {
      descriptor = descriptors_ + group;
    }
  }

  template <bool IsLoad>
  CUTLASS_DEVICE void tensormaps_cp_fence_release(
      TensorMapStorage& storage, cute::TmaDescriptor const* descriptor, int32_t warp_group = 0) {
    if constexpr (IsLoad) {
      Base::template tensormaps_cp_fence_release<IsLoad>(storage, descriptor, warp_group);
    }
  }

 private:
  cute::TmaDescriptor const* descriptors_;
};

}  // namespace sgl_kernel::w4a8_detail
