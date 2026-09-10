#pragma once

#include "cutlass_extensions/detail/collective/mixed_input_utils.hpp"

namespace cutlass::gemm::collective {

template <int Stages>
class DecodedOperandPipeline : public cutlass::PipelineAsync<Stages> {
  using Base = cutlass::PipelineAsync<Stages>;

 public:
  using ThreadCategory = typename Base::ThreadCategory;
  using SharedStorage = typename Base::SharedStorage;
  struct Params {
    ThreadCategory role = ThreadCategory::NonParticipant;
    uint32_t transaction_bytes = 0;
    uint32_t is_leader = 0;
    uint32_t num_consumers = 128;
    uint32_t num_producers = 128;
  };

  template <class ClusterShape>
  CUTLASS_DEVICE DecodedOperandPipeline(SharedStorage& storage, Params params, ClusterShape)
      : Base(storage, make_params(params)) {
    static_assert(cute::size(ClusterShape{}) == 1);
  }

 private:
  CUTLASS_DEVICE static typename Base::Params make_params(Params params) {
    typename Base::Params result;
    result.role = params.role;
    result.producer_arv_count = params.num_producers;
    result.consumer_arv_count = params.num_consumers;
    return result;
  }
};

template <class RawMainloop>
struct ProducerDecodeMainloop : RawMainloop {
  using Base = RawMainloop;
  using TileShape = typename Base::TileShape;
  using DispatchPolicy = typename Base::DispatchPolicy;
  using ElementB = typename Base::ElementB;
  using ElementAccumulator = typename Base::ElementAccumulator;
  using Params = typename Base::Params;
  using RawPipeline = typename Base::MainloopPipeline;
  using MainloopPipeline = DecodedOperandPipeline<DispatchPolicy::Stages>;
  using PipelineState = typename MainloopPipeline::PipelineState;
  using PipelineStorage = typename MainloopPipeline::SharedStorage;
  using TensorMapStorage = typename Base::TensorMapStorage;
  using RawMma = typename Base::TiledMma;
  using Utils = detail::MixedGroupedGemmInputUtils<Base>;
  static constexpr bool ProducerDecodesA = true;
  static constexpr int NumProducerThreadEvents = 128;
  static constexpr int KBlocks = cute::size<2>(TileShape{}) / 32;
  using TiledMma = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<
          ElementB, ElementB, ElementAccumulator, TileShape, cute::GMMA::Major::K, cute::GMMA::Major::K>()));
  using DecodedLayout = decltype(cute::tile_to_shape(
      cute::GMMA::Layout_K_SW128_Atom<ElementB>{},
      cute::make_shape(cute::size<0>(TileShape{}), cute::size<2>(TileShape{}), cute::Int<DispatchPolicy::Stages>{})));

  struct TensorStorage : Base::TensorStorage {
    alignas(128) cute::ArrayEngine<ElementB, cute::cosize_v<DecodedLayout>> decoded;
    alignas(16) typename RawPipeline::SharedStorage raw_pipeline;
  };

  CUTLASS_DEVICE static void producer_sync() {
    cutlass::arch::NamedBarrier::sync(128, 0);
  }

  CUTLASS_DEVICE void decode_stage(TensorStorage& storage, int stage) {
    using namespace cute;
    int const tid = int(threadIdx.x) % 128;
    RawMma mma;
    auto thread_mma = mma.get_thread_slice(tid);
    auto sA = make_tensor(make_smem_ptr(storage.smem_A.begin()), typename Base::SmemLayoutA{});
    auto sAPos = as_position_independent_swizzle_tensor(sA);
    auto mma_fragment = thread_mma.partition_fragment_A(sAPos(_, _, Int<0>{}));
    auto ldsm_copy = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, ElementB>{}, mma);
    auto thread_copy = ldsm_copy.get_thread_slice(tid);
    auto source = thread_copy.partition_S(recast<ElementB>(sAPos));
    auto raw_shape = replace<2>(mma_fragment.shape(), size<2>(mma_fragment) / Int<2>{});
    auto raw = make_fragment_like<ElementB>(raw_shape);
    auto raw_copy = thread_copy.retile_D(raw);
    auto fp4_layout = make_layout(
        make_shape(size<0>(raw), get<1>(raw.shape()), make_shape(Int<2>{}, size<2>(raw))),
        make_stride(
            Int<1>{},
            size<0>(raw) * Int<2>{},
            make_stride(size<0>(raw), size<0>(raw) * Int<2>{} * size<1>(raw))));
    auto fp4 = make_tensor(recast_ptr<typename Base::ElementA>(raw.data()), fp4_layout);
    auto scales = make_tensor(
        make_smem_ptr(reinterpret_cast<typename Base::WeightScaleRawElement*>(storage.smem_scale.begin())),
        typename Base::SmemLayoutWeightScaleExpanded{});
    auto scale_source = thread_mma.partition_A(scales);
    auto scale_fragment = make_fragment_like<typename Base::WeightScaleRawElement>(fp4);
    auto decoded = make_tensor(make_smem_ptr(storage.decoded.begin()), DecodedLayout{});
    auto destination = thread_mma.partition_A(decoded);

    cute::for_each(cute::make_seq<KBlocks / 2>{}, [&](auto k) {
      copy(ldsm_copy, source(_, _, k, stage), raw_copy(_, _, k));
    });
    cute::for_each(cute::make_seq<KBlocks>{}, [&](auto k) {
      copy(scale_source(_, _, k, stage), scale_fragment(_, _, k));
      auto output = mma_fragment(_, _, k);
      Utils::convert_A_kblock_fused_e8m0_pre_mma_raw_scale_to_slot(fp4, output, scale_fragment, k);
      copy(output, destination(_, _, k, stage));
    });
    cute::fence_view_async_shared();
  }

  template <class... Ts, class... TMs, class KTileIterator, class BlockCoord>
  CUTLASS_DEVICE void load(
      Params const& params,
      MainloopPipeline pipeline,
      PipelineState write,
      cute::tuple<Ts...> const& inputs,
      cute::tuple<TMs...> const& tensormaps,
      BlockCoord const& coord,
      KTileIterator iterator,
      int tiles,
      int thread_idx,
      uint32_t cluster_rank,
      TensorStorage& storage) {
    typename RawPipeline::Params raw_params;
    raw_params.role = RawPipeline::ThreadCategory::ProducerConsumer;
    raw_params.is_leader = threadIdx.x == 0;
    raw_params.num_consumers = 128;
    raw_params.num_producers = 1;
    raw_params.transaction_bytes = Base::TmaTransactionBytes;
    if (write.count() == 0) {
      RawPipeline::init_barriers(storage.raw_pipeline, raw_params, typename DispatchPolicy::ClusterShape{});
      producer_sync();
    }
    RawPipeline raw_pipeline(
        storage.raw_pipeline, raw_params, typename DispatchPolicy::ClusterShape{}, cute::false_type{}, cute::true_type{});
    PipelineState read{write.index(), !write.phase(), write.count()};
    PipelineState ready = write;
    int issued = 0;
    auto issue = [&] {
      pipeline.producer_acquire(write);
      if (threadIdx.x < 32) {
        Base::load(params, raw_pipeline, write, inputs, tensormaps, coord, iterator, 1, thread_idx, cluster_rank, storage);
      }
      ++write;
      ++iterator;
      ++issued;
    };
    // Leave one slot free so refill cannot block the second ready stage.
    for (int i = 0; i < cute::min(tiles, DispatchPolicy::Stages - 1); ++i) {
      issue();
    }
    for (int i = 0; i < tiles; ++i) {
      raw_pipeline.consumer_wait(read);
      decode_stage(storage, read.index());
      producer_sync();
      pipeline.producer_commit(ready);
      raw_pipeline.consumer_release(read);
      ++read;
      ++ready;
      if (issued < tiles) {
        issue();
      }
    }
  }

  template <class FrgTensorC>
  CUTLASS_DEVICE void mma(
      MainloopPipeline pipeline,
      PipelineState read,
      FrgTensorC& accum,
      int tiles,
      int thread_idx,
      TensorStorage& storage,
      Params const&) {
    using namespace cute;
    TiledMma mma;
    auto slice = mma.get_thread_slice(thread_idx);
    auto sA = make_tensor(make_smem_ptr(storage.decoded.begin()), DecodedLayout{});
    auto sB = make_tensor(make_smem_ptr(storage.smem_B.begin()), typename Base::SmemLayoutB{});
    auto a = slice.make_fragment_A(slice.partition_A(sA));
    auto b = slice.make_fragment_B(slice.partition_B(sB));
    PipelineState release = read;
    mma.accumulate_ = GMMA::ScaleOut::Zero;
    warpgroup_arrive();
    for (int tile = 0; tile < tiles; ++tile) {
      pipeline.consumer_wait(read);
      int const stage = read.index();
      cute::for_each(cute::make_seq<KBlocks>{}, [&](auto k) {
        cute::gemm(mma, a(_, _, k, stage), b(_, _, k, stage), accum);
        mma.accumulate_ = GMMA::ScaleOut::One;
      });
      warpgroup_commit_batch();
      warpgroup_wait<1>();
      if (tile > 0) {
        pipeline.consumer_release(release);
        ++release;
      }
      ++read;
    }
    warpgroup_wait<0>();
  }

  CUTLASS_DEVICE void mma_tail(MainloopPipeline pipeline, PipelineState release, int tiles) {
    release.advance(tiles - 1);
    pipeline.consumer_release(release);
  }

  CUTLASS_DEVICE void load_tail(MainloopPipeline pipeline, PipelineState write) {
    pipeline.producer_tail(write);
  }
};

}  // namespace cutlass::gemm::collective
