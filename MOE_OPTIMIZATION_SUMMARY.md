# MoE Optimization Summary

Date: 2026-09-15

## 结论

当前这一组 MoE 优化探索不应合并成一个通用生产选择器。可保留的结果是若干低层、默认关闭的实验路径；真正能进入生产候选的范围仍然很窄。

最明确的正结果是 M1 decode 方向：旧 SIMT direct 在两个 M1 形状上分别降低 CUDA Graph 延迟 6.52% 和 30.31%，因此只做 M1-only opt-in 是合理的。后续 streamed SIMT 在主尺寸 BF16 M1 上进一步把原 Triton 的约 39 us 降到约 32 us，但它没有接入 runner，因为窄形状、FP16 和热点路由都有明确负结果。

bounded workspace 是显存工作集优化，不是性能优化。它能显著降低调用峰值显存，但所有测到的 chunked prefill 都比 unchunked 慢。它适合保留为 default-off 的容量/内存压力工具，不适合宣传为加速。

grouped tensor-core decode 只在非常集中的 M8 hot 路由上有 6.23% 到 6.60% 的窄收益；同一 tile 在 M8 uniform 上慢 25.40%。adaptive reuse 试图在 GPU 内按 expert 复用率选择 SIMT 或 tensor-core，但默认配置在 M4/M8 uniform、hot、mixed 全部慢于原 Triton，不能接入。

总体判断：当前 SGLang 原 Triton MoE 路径已经在多 token 场景中利用了 expert grouping 和调度配置。简单把 unsorted direct、streamed SIMT、expert-local tensor-core 和动态分支组合起来，不能稳定超过原路径。形状、路由分布、dtype 和 tile 对结果影响很大，shape-only 或单一阈值选择都不够。

下一步若继续做 MoE decode，应避免把轻量 SIMT 和重 tensor-core 分支塞进同一个 kernel。更有希望的方向是先用一个极轻量 GPU classification 得到 compact routing metadata，再调用资源专门化的 SIMT 或 grouped kernel；但这个方向必须先证明分类和额外 launch 开销不会抵消收益。

## 决策矩阵

| 方向 | 当前结论 | 是否建议接入 runner | 主要依据 |
|---|---|---:|---|
| bounded MoE workspace | 保留 default-off 实验 | 否 | 显存峰值下降，但 chunked prefill 全部变慢 |
| two-launch SIMT direct | 保留 M1-only opt-in | 仅 M1 opt-in | M1 有稳定收益，M4 已测慢 11.24% |
| grouped tensor-core decode | 保留低层实验 | 否 | 仅 M8 hot 有窄收益，uniform 明显退化 |
| streamed SIMT down | 保留低层实验 | 否 | 主尺寸 BF16 M1/M4/M8 uniform 有收益，hot/FP16/narrow 退化 |
| adaptive expert reuse | 保留负结果 | 否 | 动态混合分支全局成本过高，fresh A/B 不支持 |

## 关键数据

### bounded workspace

分支：`feat/marlin-moe-bounded-workspace`

结果：

| 项目 | 结果 |
|---|---|
| Marlin/Triton 公共预算 | 已实现，默认 `SGLANG_MOE_WORKSPACE_BUDGET_BYTES=0` |
| CPU 回归 | 326 passed |
| GPU 数值 | 多组 Triton/Marlin smoke 通过 |
| 显存 | chunking 明显降低调用峰值 |
| 性能 | 所有实测 chunked prefill 都更慢 |
| 遗留风险 | AOT alignment synccheck divergence、Marlin shared-memory race 未解决 |

结论：这是内存工作集工具，不是吞吐优化。报告见 [MOE_WORKSPACE_CUDA_VALIDATION.md](MOE_WORKSPACE_CUDA_VALIDATION.md)。

### two-launch SIMT direct

分支：`feat/moe-direct-decode`

| 形状 | 原 Triton | Direct | 变化 |
|---|---:|---:|---:|
| M1/H4096/I512/E256/top-k6 | 40.270 us | 37.646 us | -6.52% |
| M1/H2048/I1024/E64/top-k4 | 29.093 us | 20.276 us | -30.31% |
| M4/H4096/I512/E256/top-k6 | 98.329 us | 109.385 us | +11.24% |

结论：decode batch 为 1 时，去掉 align、activation、reduce 等额外 launch 和 padding 有实际收益；M4 开始失去跨 token/expert 复用，因此 runner 只允许 M1。报告见 [MOE_DIRECT_DECODE.md](MOE_DIRECT_DECODE.md)。

### grouped tensor-core decode

分支：`feat/moe-direct-decode-grouped`

| 场景 | 原 Triton | Grouped | 变化 |
|---|---:|---:|---:|
| M8 hot, N32/K256, seed42 | 55.2544 us | 51.6062 us | -6.60% |
| M8 hot, N32/K256, seed7 | 55.1360 us | 51.7005 us | -6.23% |
| M8 hot, N32/K256, seed123 | 55.1832 us | 51.7298 us | -6.26% |
| M8 uniform, N32/K256 | 197.2824 us | 247.3861 us | +25.40% |

结论：热点路由有复用收益，但 uniform 路由基本没有可用复用，leader scan、无效行、route output 和额外 sum 成本会超过收益。shape-only selector 无法区分真实路由分布。报告见 [MOE_GROUPED_DECODE.md](MOE_GROUPED_DECODE.md)。

### streamed SIMT down

分支：`feat/moe-direct-decode-streamed`

主尺寸 BF16，H4096/I512/E256/top-k6，selected 参数 `up_n=16, down_n=16, unroll=2`：

| 场景 | 原 Triton | Streamed | 变化 |
|---|---:|---:|---:|
| M1 uniform | 39.1555 us | 31.9512 us | -18.40% |
| M1 hot | 39.2123 us | 31.9560 us | -18.51% |
| M4 uniform | 97.7042 us | 93.9024 us | -3.89% |
| M4 hot | 42.9869 us | 62.5960 us | +45.62% |
| M8 uniform | 195.9251 us | 176.2362 us | -10.05% |
| M8 hot | 53.9226 us | 115.2803 us | +113.79% |

补充负结果：

| 形状 / dtype | 原 Triton | Old direct | Streamed |
|---|---:|---:|---:|
| M1/H2048/I1024/E64/top-k4, BF16 | 28.7552 us | 20.3038 us | 31.6187 us |
| M1/H4096/I512/E256/top-k6, FP16 | 39.9355 us | 37.8710 us | 40.0427 us |

同进程同 tile 消融：

| M | same-tile vectorized | streamed | 变化 |
|---:|---:|---:|---:|
| 1 | 34.1904 us | 31.7765 us | -7.06% |
| 4 | 102.8760 us | 94.6622 us | -7.98% |

结论：streamed down 本身有额外收益，但 selected tile 和路由分布都强相关。它不能替代原 Triton，也不应作为 runner 扩展。报告见 [MOE_STREAMED_DECODE.md](MOE_STREAMED_DECODE.md)。

### adaptive expert reuse

分支：`feat/moe-adaptive-reuse-decode`

默认 adaptive：`min_reuse=4, simt_n=16, group_n=32, group_k=256`。

| 场景 | 原 Triton | Adaptive | 变化 |
|---|---:|---:|---:|
| M1 uniform | 39.4088 us | 32.7690 us | -16.85% |
| M4 uniform | 97.1566 us | 216.5658 us | +122.90% |
| M4 hot | 42.9534 us | 48.4179 us | +12.72% |
| M4 mixed | 77.0242 us | 209.8354 us | +172.43% |
| M8 uniform | 194.3034 us | 423.3429 us | +117.88% |
| M8 hot | 53.7429 us | 61.6157 us | +14.65% |
| M8 mixed | 118.8134 us | 253.9061 us | +113.70% |

关键诊断：

| 配置 | M4 uniform adaptive / original | M8 hot adaptive / original |
|---|---:|---:|
| default N32/K256 | 216.5658 / 97.1566 us | 61.6157 / 53.7429 us |
| small tile N16/K32 confirm | 103.4104 / 98.9024 us | 100.9338 / 55.1283 us |
| threshold9 force SIMT | 98.4010 / 98.3475 us | 121.5731 / 55.1034 us |

结论：动态分支在数值上可行，但性能上失败。默认 kernel 即使 runtime 全走 SIMT，编译资源仍受 grouped 分支影响；小 tile 缓解 uniform，却严重损害 hot。报告见 [MOE_ADAPTIVE_REUSE_DECODE.md](MOE_ADAPTIVE_REUSE_DECODE.md)。

## 根因分析

### 1. 小 batch decode 的收益来自“删工作”，不是换更复杂的 kernel

M1 direct 能赢，是因为它直接绕过了原路径中对小 batch 不划算的组件：专家排序、padding、activation launch、top-k reduce launch 和较大的中间张量。这个方向符合 M1 的结构特征，因为几乎没有跨 token 的 expert 复用可利用。

一旦 M 增大，原 Triton 的 sorted grouped GEMM 开始回收跨 token 的权重复用。此时 direct/streamed 的 route-local 读权重方式会重复读取同一 expert 权重，热点路由尤其明显。因此 M4/M8 hot 下 direct 和 streamed 都明显退化。

### 2. 路由分布比 batch size 更关键

M8 hot 和 M8 uniform 在同一 H/I/E/top-k 下表现相反。Grouped tensor-core 在 M8 hot 能赢 6% 左右，但同 tile 的 M8 uniform 慢 25.40%。这说明 batch size 不是充分 selector，必须知道 selected experts 的实际复用分布。

但当前约束是不把 routing values 读回 host，也不做 runtime host decision。因此 shape-only runner 不能安全扩展到 M4/M8。

### 3. GPU 内动态选择没有自动保留两个路径的优点

Adaptive 的初衷是让 GPU 自己按 expert 复用率选择 SIMT 或 grouped，避免 host 读回 routing。实测失败的核心原因是同一个 JIT kernel 必须同时容纳轻分支和重分支的资源形态。

default M4 uniform 中，虽然 initial routing 的 24 个 route 全走 SIMT，但 `_adaptive_gate_up` trace 仍显示 81920 bytes shared memory；同一场景 small tile 降到 6144 bytes 后，延迟从 216 us 降到约 103 us。threshold9 编译期消除 grouped 分支后，延迟接近原 Triton。但 threshold9 放弃了 hot-route tensor-core 复用，M8 hot 变慢到 121.57 us。

结论是：把两个资源形态完全不同的算法塞进一个 CTA-level branch，不等于得到了两个算法的最小成本。

### 4. 显存峰值下降不能直接转化为吞吐结论

bounded workspace、direct、streamed、grouped/adaptive 都降低了某些单调用显式 scratch 或峰值分配，但这些不是 KV capacity、不是 serving QPS，也不代表端到端吞吐。尤其 bounded workspace 的实测清楚表明：显存下降可以伴随显著 latency 增加。

所有性能数据必须仍按单个 workload 的配置解释：模型未加载、无 HTTP、无请求并发、无 TTFT/TPOT、无 speculative decoding、无真实 KV cache 竞争。

## 验证状态

| 阶段 | CPU | GPU 数值 / graph / 双流 | Sanitizer |
|---|---:|---|---|
| bounded workspace | 326 passed | 多组通过 | 存在旧 AOT/Marlin 风险 |
| direct SIMT | 415 passed 累计基线 | M1/M4/边界通过 | 新 kernel scoped checks 通过；旧问题不修 |
| grouped tensor-core | 435 passed 累计 | 矩阵通过 | filtered checks 通过；unfiltered API errors 保留 |
| streamed SIMT | 470 passed 累计 | 50 进程中 49 成功，1 个 unfiltered exit86 | 8 组 filtered device checks 通过 |
| adaptive reuse | 494 passed 累计 | 43 进程中 42 成功，1 个 unfiltered exit86 | 9 组 filtered device checks 通过 |

unfiltered memcheck 的 34 个 `cuGetProcAddress_v2` import-time API errors 在 grouped、streamed、adaptive 中均复现。数值通过不覆盖该进程失败；报告中均按失败保留。

## 当前分支和产物

当前分支：

```text
feat/moe-adaptive-reuse-decode
```

当前 HEAD：

```text
a54663590550cd43964f7a68ab79623cd8ca0860
```

关键提交：

| Commit | 内容 |
|---|---|
| `7e885c8446` | adaptive GPU-only expert reuse prototype |
| `27c1794a56` | adaptive wrapper 单测 |
| `a546635905` | adaptive 实测报告 |
| `447c642d87` | streamed decode kernel prototype |
| `109a00cca5` | streamed 单测和同进程消融 |
| `d91ead9db6` | streamed 实测报告 |
| `366a6d331c` | grouped 实测报告 |

证据目录：

```text
validation/moe-adaptive-decode-20260915/
validation/moe-streamed-decode-20260914/
validation/moe-grouped-decode-20260912/
```

## 建议

1. 不要把 grouped、streamed 或 adaptive 扩入现有 runner selector。
2. 保留 M1 direct opt-in 的边界，不扩大到 M4/M8。
3. 若要形成一个可交付 PR，优先选择 default-off、低风险、结论明确的方向：bounded workspace 或 M1 direct，而不是 adaptive。
4. 若继续性能探索，下一轮应把“路由分类”和“计算 kernel”分离：先生成 compact GPU metadata，再调度专门化 SIMT/grouped kernel。这个方向必须 fresh A/B 验证分类开销、额外 launch、真实路由分布和 sanitizer。
5. 上模型前必须补真实 checkpoint routing 分布、模型精度、真实 serving TTFT/TPOT/throughput，以及与 KV cache 并发压力下的端到端结果。
