#pragma once

#include "cutlass/pipeline/pipeline.hpp"

namespace cutlass::gemm::collective {

// RF-sourced weights stop using SMEM before the asynchronous B operand does.
template <int Stages, uint32_t WeightTransactionBytes>
class SplitWeightTmaPipeline : public cutlass::PipelineTmaAsync<Stages> {
 public:
  using Base = cutlass::PipelineTmaAsync<Stages>;
  using Params = typename Base::Params;
  using PipelineState = typename Base::PipelineState;

  struct SharedStorage {
    typename Base::SharedStorage activation;
    typename Base::SharedStorage weight;
  };

  Base weight;

  template <class ClusterShape>
  CUTLASS_DEVICE SplitWeightTmaPipeline(SharedStorage& storage, Params params, ClusterShape cluster_shape)
      : Base(storage.activation, activation_params(params), cluster_shape),
        weight(storage.weight, weight_params(params), cluster_shape) {
    static_assert(cute::size(ClusterShape{}) == 1, "Split weight lifetime currently supports cluster-1 only.");
  }

  CUTLASS_DEVICE void consumer_wait(
      PipelineState state, cutlass::ConsumerToken token = {cutlass::BarrierStatus::WaitAgain}) {
    weight.consumer_wait(state);
    Base::consumer_wait(state, token);
  }

  CUTLASS_DEVICE void producer_tail(PipelineState state) {
    weight.producer_tail(state);
    Base::producer_tail(state);
  }

 private:
  CUTLASS_DEVICE static Params activation_params(Params params) {
    params.transaction_bytes -= WeightTransactionBytes;
    return params;
  }

  CUTLASS_DEVICE static Params weight_params(Params params) {
    params.transaction_bytes = WeightTransactionBytes;
    return params;
  }
};

}  // namespace cutlass::gemm::collective
