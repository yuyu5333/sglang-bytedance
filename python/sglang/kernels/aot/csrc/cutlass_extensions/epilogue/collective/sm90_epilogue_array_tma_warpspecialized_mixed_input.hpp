/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/sm90_epilogue_array_tma_warpspecialized.hpp"

namespace tensorrt_llm::cutlass_extensions::epilogue::collective {

namespace detail = cutlass::epilogue::collective::detail;
namespace fusion = cutlass::epilogue::fusion;
using namespace cute;

// Mixed-input pre-MMA kernels need a regular ptr-array TMA epilogue. Keep the
// construction local instead of routing through the single-warpgroup adapter:
// the latter synchronizes a different participant count and can deadlock when
// used by a regular warp-specialized kernel.
template <class ArchTag, class OpClass, class TileShape_MNK, class ClusterShape_MNK,
          class EpilogueTileType, class ElementAccumulator, class ElementCompute, class ElementC_,
          class GmemLayoutTagC_, int AlignmentC, class ElementD_, class GmemLayoutTagD,
          int AlignmentD, class Schedule,
          class FusionOpOrCallbacks = cutlass::epilogue::fusion::LinearCombination<
              ElementD_, ElementCompute, ElementC_, ElementCompute>>
struct MixedInputSm90TmaEpilogueBuilder {
  static_assert(cute::is_same_v<ArchTag, cutlass::arch::Sm90>,
                "Mixed-input TMA epilogue builder is SM90-only.");
  static_assert(detail::sm90_is_ptr_array_tma_v<Schedule>,
                "Mixed-input TMA epilogue builder expects a ptr-array TMA schedule.");

 private:
  static_assert(detail::is_aligned<ElementC_, AlignmentC, ElementD_, AlignmentD>(),
                "C/D should meet TMA alignment requirement.");

  using ElementD = cute::conditional_t<cute::is_void_v<ElementD_>,
                                       fusion::get_element_aux_t<FusionOpOrCallbacks>, ElementD_>;
  using ElementC = cute::conditional_t<cute::is_void_v<ElementC_>, ElementD, ElementC_>;
  using GmemLayoutTagC =
      cute::conditional_t<cute::is_void_v<ElementC_>, GmemLayoutTagD, GmemLayoutTagC_>;

  using EpilogueTile_MN =
      decltype(detail::sm90_compute_tile_shape_or_override<ElementD, EpilogueTileType, Schedule,
                                                           TileShape_MNK>());
  using DispatchPolicy =
      decltype(detail::sm90_get_tma_dispatch_policy<TileShape_MNK, EpilogueTile_MN, ElementC,
                                                    ElementD, Schedule>());

  using GmemStrideTypeC = cutlass::detail::TagToStrideC_t<GmemLayoutTagC>;
  using GmemStrideTypeD = cutlass::detail::TagToStrideC_t<GmemLayoutTagD>;
  using UnderlyingGmemStrideTypeC = cute::remove_pointer_t<GmemStrideTypeC>;
  using UnderlyingGmemStrideTypeD = cute::remove_pointer_t<GmemStrideTypeD>;

  using CopyOpS2G = cute::conditional_t<detail::is_im2col_mode<GmemLayoutTagD>,
                                        SM90_TMA_STORE_IM2COL, SM90_TMA_STORE>;
  using CopyOpG2S = cute::conditional_t<detail::is_im2col_mode<GmemLayoutTagC>,
                                        SM90_TMA_LOAD_IM2COL, SM90_TMA_LOAD>;

  using CopyAtomC =
      cute::conditional_t<size<1>(EpilogueTile_MN{}) % 16 == 0,
                          Copy_Atom<SM90_U32x4_STSM_N, cutlass::half_t>,
                          cute::conditional_t<size<1>(EpilogueTile_MN{}) % 8 == 0,
                                              Copy_Atom<SM90_U32x2_STSM_N, cutlass::half_t>, void>>;
  static_assert(!cute::is_same_v<CopyAtomC, void>, "CopyAtomC cannot be void.");

  using FusionCallbacks = typename detail::CallbacksBuilder<
      DispatchPolicy, FusionOpOrCallbacks, TileShape_MNK, EpilogueTile_MN,
      ElementAccumulator>::Callbacks;

 public:
  using CollectiveOp = cutlass::epilogue::collective::CollectiveEpilogue<
      DispatchPolicy, TileShape_MNK, EpilogueTile_MN, ElementC_, GmemStrideTypeC, ElementD_,
      GmemStrideTypeD, FusionCallbacks, CopyOpG2S,
      decltype(detail::sm90_get_epilogue_smem_swizzle_layout_atom<UnderlyingGmemStrideTypeC,
                                                                  ElementC, EpilogueTile_MN>()),
      decltype(detail::sm90_get_smem_load_op_for_source<UnderlyingGmemStrideTypeC, ElementC,
                                                        EpilogueTile_MN>()),
      CopyOpS2G,
      decltype(detail::sm90_get_epilogue_smem_swizzle_layout_atom<UnderlyingGmemStrideTypeD,
                                                                  ElementD, EpilogueTile_MN>()),
      decltype(detail::sm90_get_smem_store_op_for_accumulator<UnderlyingGmemStrideTypeD, ElementD,
                                                              EpilogueTile_MN>()),
      CopyAtomC, void>;
};

}  // namespace tensorrt_llm::cutlass_extensions::epilogue::collective
