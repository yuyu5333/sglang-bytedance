# W4A8-FP8 Dense Linear 当前实现记录与问题清单

本文档盘点 W4A8-FP8 dense `Linear` 路径（非 MoE）已经落地的改动，并基于静态阅读逐个指出发现的逻辑漏洞 / 潜在错误。

---

## 一、当前实现涉及的文件

### 1. Python 层：高层 Linear 封装
- `python/sglang/srt/layers/quantization/w4afp8_linear.py`
  - 新增 `cutlass_w4a8_fp8_linear(...)`：dense W4A8-FP8 线性层入口。
  - 新增 `quantize_input_to_fp8_per_token(...)`：`sglang_per_token_quant_fp8` 的 2D 包装。
  - 新增 `quantize_input_to_fp8(...)`：统一激活量化入口（动态 per-token / scalar-static）。
  - 新增内部工具 `_unpack_int4_from_int8_packed`、`_dequantize_w4_groupwise`，供临时 fallback 使用。
  - 运行时支持判定通过则走 `w4a8_fp8_scaled_mm`，否则走 `dequantize + matmul` 的 reference fallback。

### 2. Python 层：kernel 薄封装 + runtime 探针
- `python/sglang/srt/layers/quantization/w4afp8_kernel.py`
  - 新增 `w4a8_fp8_scaled_mm(...)`：对 JIT 自定义算子的 Python 薄封装。
  - 新增 `is_w4a8_fp8_linear_supported()`（`lru_cache`）：CUDA + SM90+ + JIT 源文件存在即视为支持。
  - 新增 `_get_w4a8_fp8_jit_kernel_api()`（`lru_cache`）：延迟导入 JIT 入口，避免硬依赖。
  - 新增 `_validate_w4a8_fp8_inputs(...)`：全面的 shape / dtype / device / group_size 校验。

### 3. Python 层：JIT 绑定
- `python/sglang/jit_kernel/w4a8_fp8_scaled_mm.py`
  - 新增 `w4a8_fp8_scaled_mm(...)` custom op，内含 fake impl `_fake_w4a8_fp8_scaled_mm`。
  - 新增 `has_w4a8_fp8_scaled_mm_jit_kernel()`（仅判断 `.cuh` 文件是否存在）。
  - 新增 `_jit_w4a8_fp8_scaled_mm_module(out_dtype)`（`cache_once`）：基于 out_dtype 模板实例化 JIT module。

### 4. CUDA kernel
- `python/sglang/jit_kernel/csrc/gemm/w4a8_fp8_scaled_mm.cuh`
  - 新增 naive CUDA kernel `w4a8_fp8_scaled_mm_naive_kernel<OutDType>`（16×16 block，每线程一个 (row, col) 输出）。
  - 新增 `to_float<T>`/`from_float<T>` 工具特化，覆盖 `bf16_t`、`fp16_t`、`fp8_e4m3_t`。
  - 新增 `unpack_signed_int4(packed, k_idx)`：按奇偶索引取低 / 高 nibble 并做 sign-extension。
  - 新增 FFI wrapper `w4a8_fp8_scaled_mm<OutDType>`（**位于全局命名空间**以便 TVM FFI 链接）。
  - 做了 contiguous / dtype / device / group_size=128 / PackedK*2==K 的校验。

### 5. compressed-tensors 接入（dense）
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py`（新增文件）
  - 新增 class `CompressedTensorsW4AFP8(CompressedTensorsLinearScheme)`。
  - 实现：
    - `__init__`：断言 INT4 / symmetric / 非 dynamic 权重 + FP8 dynamic activation + `pack-quantized` 格式。
    - `get_min_capability() -> 90`。
    - `_validate_shapes(...)`：K 对齐到 `pack_factor`(8) 和 `group_size`。
    - `create_weights(...)`：注册 `weight_packed`(int32, `[N, K/8]`)、`weight_scale`(fp32, `[N, K/group]`)、`weight_shape`。
    - `process_weights_after_loading(...)`：int32 → int8 repack，置 `is_w4afp8_converted=True`。
    - `apply_weights(...)`：若仍是 int32 则即时 repack，然后 dispatch 到 `cutlass_w4a8_fp8_linear(...)`。
  - 新增本地 `_unpack_repack_int32_to_cutlass_int8(...)`（与 MoE 同名实现重复）。
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/__init__.py`
  - 追加 `from .compressed_tensors_w4a8_fp8 import CompressedTensorsW4AFP8` 及 `__all__`。
- `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`
  - 新增方法 `_is_w4afp8(self, weight_quant, input_quant)`。
  - 在 `_get_scheme_from_parts(...)` 中增加 dense W4AFP8 分支。
  - 在 MoE 分支也通过 `_is_w4afp8` 复用判定。

### 6. 测试
- `test/registered/quant/test_w4a8_fp8_scaled_mm.py`：CUDA-only，JIT 烟雾测试 + 与 PyTorch reference 的数值一致性。
- `test/registered/quant/test_w4afp8_linear_wrapper.py`：CPU 可运行，仅用 mock 验证 dispatch / fallback / 量化助手选择。
- `test/registered/quant/test_compressed_tensors_w4a8_fp8_linear.py`：CPU 可运行，覆盖 `create_weights` / `process_weights_after_loading` / `apply_weights` / `_unpack_repack_int32_to_cutlass_int8` 以及 `_is_w4afp8` 分支判定。

---

## 二、发现的逻辑漏洞与错误

> 严重程度：**Blocker**=阻断功能 / 测试必然失败；**High**=会导致错误结果或静默 bug；**Medium**=鲁棒性 / 易用性问题；**Low**=风格 / 冗余。

### B1. `_is_w4afp8` 调用签名与测试不匹配（Blocker）

- 源码定义为实例方法：
  ```python
  def _is_w4afp8(self, weight_quant, input_quant) -> bool:
  ```
  位于 [compressed_tensors.py#L404-L416](file:///Users/bytedance/Desktop/WYZ/Code/20260305-Sglang-bytedance/sglang-bytedance/python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py#L404-L416)。
- 但测试 [test_compressed_tensors_w4a8_fp8_linear.py#L364-L440](file:///Users/bytedance/Desktop/WYZ/Code/20260305-Sglang-bytedance/sglang-bytedance/test/registered/quant/test_compressed_tensors_w4a8_fp8_linear.py#L364-L440) 传入了 **三个**位置参数：
  ```python
  self.ConfigCls._is_w4afp8(config, weight_quant, input_quant)
  ```
  以类方法方式调用时，`self` 槽位被 `config` 占用，`weight_quant` 被当作 `self`，`input_quant` 被当作 `weight_quant`，完全错位。该测试在任何环境下都会失败，除非把 `_is_w4afp8` 改为 `@staticmethod` 或让测试改用实例调用。

### B2. `test_roundtrip_with_known_values` 与 repack 的编码约定不一致（Blocker）

- MoE 注释明确 `pack_to_int32` 存放的是 **unsigned-offset**（zero-point=8）编码，`_unpack_repack_int32_to_cutlass_int8` 因此执行 `- offset`（即 `-8`）来还原 signed int4。见 [compressed_tensors_w4a8_fp8_moe.py#L59-L88](file:///Users/bytedance/Desktop/WYZ/Code/20260305-Sglang-bytedance/sglang-bytedance/python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8_moe.py#L59-L88)。
- dense 测试 [test_compressed_tensors_w4a8_fp8_linear.py#L288-L317](file:///Users/bytedance/Desktop/WYZ/Code/20260305-Sglang-bytedance/sglang-bytedance/test/registered/quant/test_compressed_tensors_w4a8_fp8_linear.py#L288-L317) 却用 **signed 值按二进制补码直接填入 int32**：
  ```python
  val = weight_int4[:, col].to(torch.int32) & 0x0F
  packed_int32[:, group] |= val << shift
  ```
  例：`weight=-1` → 存入 `0xF`；解码时 `((0xF) & 0xF) - 8 = 7`，与期望 `-1` 相差 8。断言 `nibble == weight_int4[row, col]` 会失败。
- 结论：dense 测试对编码方式做了**相反**的假设。需要二选一：
  - 若 ckpt 是 unsigned-offset（与 MoE 注释一致），测试打包阶段应是 `val = ((weight_int4[:, col] + 8) & 0x0F)`；
  - 若 ckpt 是 two's-complement signed int4，则 `_unpack_repack_int32_to_cutlass_int8` 不应执行 `- offset`，需要调整算子而非测试。

**同时意味着**：dense 与 MoE 共享同一个 `_unpack_repack_int32_to_cutlass_int8` 实现，真正正确的编码约定至今仍未澄清 —— 这会直接影响真实 ckpt 运行时 naive kernel 的数值结果。

### B3. `apply_weights` 里 int32 兜底路径不调用 `.data`（High）

[compressed_tensors_w4a8_fp8.py#L184-L188](file:///Users/bytedance/Desktop/WYZ/Code/20260305-Sglang-bytedance/sglang-bytedance/python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py#L184-L188)：
```python
weight_packed = layer.weight_packed
if weight_packed.dtype == torch.int32:
    weight_packed = _unpack_repack_int32_to_cutlass_int8(weight_packed, ...)
```
这里把 `nn.Parameter` 整个传给 repack 函数，`weight_packed >> low_shift` 在 `Parameter` 上触发的是 `__torch_function__`，通常仍可工作但可能被 autograd 追踪（因为 Parameter 默认 `requires_grad=True`，且 int 类型下 `requires_grad` 会被 Torch 强制拒绝，但把 Parameter 当张量做 bitwise 运算在部分 Torch 版本里会 warn）。更重要的是：**正常流程下 `process_weights_after_loading` 已经保证运行前是 int8**，`apply_weights` 再放一份 int32→int8 兜底意义不大，反而每次前向都重建一次 repack，若真命中这条分支会非常慢且不缓存。建议：要么移除这条兜底并在 `apply_weights` 入口断言 `is_w4afp8_converted`；要么在兜底分支里把结果 **写回 layer** 并置 `is_w4afp8_converted=True`。

### B4. JIT wrapper 签名与实际 CUDA wrapper 签名的参数顺序不一致（High）

- Python 调用见 [w4a8_fp8_scaled_mm.py#L74-L84](file:///Users/bytedance/Desktop/WYZ/Code/20260305-Sglang-bytedance/sglang-bytedance/python/sglang/jit_kernel/w4a8_fp8_scaled_mm.py#L74-L84)：
  ```python
  module.w4a8_fp8_scaled_mm(
      q_input, weight_packed, x_scale, weight_scale, output, group_size, bias,
  )
  ```
  按位置传参，顺序为 `(q_input, weight_packed, x_scale, weight_scale, output, group_size, bias)`。
- C++ 侧 [w4a8_fp8_scaled_mm.cuh#L119-L127](file:///Users/bytedance/Desktop/WYZ/Code/20260305-Sglang-bytedance/sglang-bytedance/python/sglang/jit_kernel/csrc/gemm/w4a8_fp8_scaled_mm.cuh#L119-L127)：
  ```cpp
  void w4a8_fp8_scaled_mm(
      q_input, weight_packed, x_scale, weight_scale, output,
      int64_t group_size,
      Optional<TensorView> bias);
  ```
  **该顺序匹配**。注意 roadmap 文档里宣称的 "launcher signature" 中 `output` 是 hidden field，实际 Python / FFI 都把 `output` 当作一个入参（作为输出缓冲），这和代码实现一致。✅ 这一项虽然容易出错，但现在是对齐的。请在后续修改中保持同步。

### B5. FFI `weight_scale` dtype 约束与 Python 侧不完全一致（Medium）

- CUDA wrapper 要求 `weight_scale` 是 `float` / fp32（见 [w4a8_fp8_scaled_mm.cuh#L146-L149](file:///Users/bytedance/Desktop/WYZ/Code/20260305-Sglang-bytedance/sglang-bytedance/python/sglang/jit_kernel/csrc/gemm/w4a8_fp8_scaled_mm.cuh#L146-L149)）。
- 但 Python 层 `_validate_w4a8_fp8_inputs` 只要求 `weight_scale.dtype.is_floating_point`，roadmap 也说 "fp32/bf16"，且 kernel 内部把指针直接 cast 为 `const float*`。若上层误传 `bfloat16` 的 weight_scale，Python 层放行而 C++ `TensorMatcher<float>` 失败，错误信息将是 FFI 级别不便排查，且与 wrapper docstring 口径不符。
- 建议：Python 层显式 `weight_scale.dtype == torch.float32`，或在 `process_weights_after_loading` 里统一 cast 到 fp32。

### B6. `x_scale` 在 FFI 侧是 `float*`，但 Python 未严格要求 fp32（High）

[w4afp8_kernel.py#L121-L126](file:///Users/bytedance/Desktop/WYZ/Code/20260305-Sglang-bytedance/sglang-bytedance/python/sglang/srt/layers/quantization/w4afp8_kernel.py#L121-L126) 只校验 `x_scale.dtype.is_floating_point`。而 kernel 读 `const float*`。若上层传 bf16/fp16 scale，结果静默错误（指针 reinterpretation）。虽然 `quantize_input_to_fp8_per_token` 返回 fp32，但 `quantize_input_to_fp8(..., input_scale=scalar)` 最后才 `.to(torch.float32)`，`input_scale` 自身若不是 fp32 也会被强制转回。仍建议在 Python 端显式要求 `x_scale.dtype == torch.float32`，与 C++ `TensorMatcher<float>` 对齐，避免依赖 FFI 报错。

### B7. `process_weights_after_loading` 不处理 bf16/fp16 scale（Medium）

`layer.weight_scale` 注册时是 `fp32`，但 compressed-tensors 真实 ckpt 有时会是 `bf16` / `float16`。当前代码只是 `data.contiguous()` 透传。如果 ckpt 下发 bf16 scale，加上上面 B5 提到的 kernel 只接受 fp32，会运行时失败。建议在 `process_weights_after_loading` 里无条件 `weight_scale.to(torch.float32)`。

### B8. `cutlass_w4a8_fp8_linear` 对 bias dtype 未做保证（Medium）

[w4afp8_linear.py#L132](file:///Users/bytedance/Desktop/WYZ/Code/20260305-Sglang-bytedance/sglang-bytedance/python/sglang/srt/layers/quantization/w4afp8_linear.py#L132)：直接 `bias.contiguous()` 下发；而 FFI 侧要求 bias dtype 等于 `OutDType`（[w4a8_fp8_scaled_mm.cuh#L155-L160](file:///Users/bytedance/Desktop/WYZ/Code/20260305-Sglang-bytedance/sglang-bytedance/python/sglang/jit_kernel/csrc/gemm/w4a8_fp8_scaled_mm.cuh#L155-L160)），例如 `bf16`。若上游传 fp32 bias 或与 `output_dtype` 不一致的 bias，将被 FFI 拒绝。建议在 wrapper 处自动 `bias.to(output_dtype)`。

### B9. `is_w4a8_fp8_linear_supported()` 过于乐观（Medium）

[w4afp8_kernel.py#L32-L50](file:///Users/bytedance/Desktop/WYZ/Code/20260305-Sglang-bytedance/sglang-bytedance/python/sglang/srt/layers/quantization/w4afp8_kernel.py#L32-L50) 的检查仅包括：
1. `is_cuda()`
2. `major >= 9`
3. `has_w4a8_fp8_scaled_mm_jit_kernel()` 返回 True（**只看 `.cuh` 是否存在**）
4. `jit_op is callable`

**不包括** JIT 编译是否成功 / kernel 是否能加载。因此若 `.cuh` 存在但编译失败，`is_w4a8_fp8_linear_supported()` 仍会返回 True，`cutlass_w4a8_fp8_linear` 会直接尝试 dispatch 而非 fallback → 抛异常。这和 roadmap 的 "Step 1 tighten probe" 任务一致，需要实际执行一次 `_jit_w4a8_fp8_scaled_mm_module(torch.bfloat16)` 之类的试编译，并捕获异常后降级。

### B10. `lru_cache` 导致多 GPU / dtype 组合下重复计算的遗漏（Low）

`is_w4a8_fp8_linear_supported()` 被 `@lru_cache(maxsize=1)` 无参缓存，意味着：
- 在 multi-process / multi-GPU 场景中，`get_device_capability()` 可能在不同 rank 上返回不同结果，但首次调用结果被全进程生命周期缓存。当前 SGLang 大多是每个 rank 一个 Python 进程，所以问题不大，但需要注意。

### B11. Naive CUDA kernel 性能非常差，但准确性依赖 `x_scale` 的 layout（Medium）

- Kernel 中 `params.x_scale[row]` 假设 `x_scale` 是连续的 float 数组，长度至少 `M`。Python 接受 `[M]` 或 `[M, 1]`，都是连续 fp32 共 `M` 个元素，OK。
- 但如果上游传 `[M, 1]` 但非 contiguous（例如来自 `.view(M,1)` 经 slice 过），FFI 侧 `is_contiguous` 检查会拦截，尚可。

### B12. `_fake_w4a8_fp8_scaled_mm` 的 FakeTensor 未考虑 `out_dtype` 与 `q_input.new_empty` 兼容性（Low）

[w4a8_fp8_scaled_mm.py#L17-L27](file:///Users/bytedance/Desktop/WYZ/Code/20260305-Sglang-bytedance/sglang-bytedance/python/sglang/jit_kernel/w4a8_fp8_scaled_mm.py#L17-L27) 中 `q_input.new_empty(..., dtype=out_dtype)`，但 `q_input` 是 FP8 dtype；部分 Torch 版本的 `new_empty` 对"FP8 tensor 作为 source 创建 bf16 FakeTensor"有限制。一般在 export path 下可能抛错。若此 fake 仅是 compile 路径的 meta 占位，可以考虑直接 `torch.empty((M, N), dtype=out_dtype, device=q_input.device)`。

### B13. `_unpack_repack_int32_to_cutlass_int8` 的潜在溢出（Low）

[compressed_tensors_w4a8_fp8.py#L47-L51](file:///Users/bytedance/Desktop/WYZ/Code/20260305-Sglang-bytedance/sglang-bytedance/python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py#L47-L51)：
```python
low_nibbles  = ((weight_packed >> low_shift) & mask) - offset
high_nibbles = ((weight_packed >> high_shift) & mask) - offset
out[..., pair_idx] = ((high_nibbles << 4) | (low_nibbles & 0x0F)).to(torch.int8)
```
`high_nibbles` 在 `- offset` 之后范围 `[-8, 7]`。`(-8) << 4 = -128`，`(7) << 4 = 112`。然后 `| (low_nibbles & 0x0F)`（`low_nibbles & 0x0F` ∈ `[0, 15]`）。在 int32 上该位运算仍是 well-defined，但**当 `high_nibbles` 为负时，`high_nibbles << 4` 在 PyTorch 上对带符号 int 做左移存在平台差异**（C++ UB，PyTorch 对 int32 的 `<<` 依赖 ATen 约定）。可靠写法：`((high_nibbles & 0x0F) << 4) | (low_nibbles & 0x0F)`。这样才保证和 kernel 侧 `unpack_signed_int4` 互逆。

### B14. `_dequantize_w4_groupwise` 与 naive kernel 的算术对不齐（Medium）

- Python fallback 的 reference：`dequant_weight = int4 * weight_scale`, 然后 `input.to(fp) @ dequant_weight.T`（**不做**输入 FP8 量化）。
- Naive kernel：`(q_fp8 * x_scale) * (int4 * weight_scale)`（做 FP8 量化）。
- `cutlass_w4a8_fp8_linear` 的 fallback 路径使用的是 "原始 input（bf16）× dequant 权重"，**不经过 FP8 量化**。因此 `test_fallback_path_uses_reference_matmul` 的"输入→输出"关系和真实 kernel 不一致。作为 **临时 correctness fallback** 虽然可以，但它不能用于做 `test_apply_weights_matches_reference_dequant_matmul` 之类的端到端数值比对，否则会出现 kernel 与 fallback 的系统性偏差。需要在 roadmap 中显式写明：fallback 只是行为占位而非精度基准。

### B15. `create_weights` 没有把 `output_partition_sizes` 传给 Parameter 用于 merged linear 切分（High）

[compressed_tensors_w4a8_fp8.py#L127-L138](file:///Users/bytedance/Desktop/WYZ/Code/20260305-Sglang-bytedance/sglang-bytedance/python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py#L127-L138) 只注册了 `input_dim=1, output_dim=0, packed_dim=1`，但没有设置 `marlin_tile_size` 或 `output_partition_sizes`。**`MergedColumnParallelLinear` 和 `QKVParallelLinear` 在加载时需要正确识别每个逻辑列块的边界**（通常通过 `layer.logical_widths` + loader 决定切片）。目前 `layer.logical_widths = output_partition_sizes` 有设置，所以 weight_loader 能切对列，但 `PackedvLLMParameter` 的 load 行为在 `packed_dim=1` 且 packed 列维度是 `K/8` 而非 `N`，切分维度正好是 row (output_dim=0)，因此不受 packed 影响 —— 但需要测试 merged linear 确认，roadmap Step 3 提到。

### B16. `lru_cache` 无参缓存 + 全局 `_is_cuda` 捕获导致测试难隔离（Low）

`_is_cuda = is_cuda()` 在 module import 时求值并被 `lru_cache` 的 `is_w4a8_fp8_linear_supported` 捕获。单元测试要 mock `is_w4a8_fp8_linear_supported` 的返回值时，已经在 `w4afp8_linear.py` 侧通过 `patch("...w4afp8_linear.is_w4a8_fp8_linear_supported")` 覆盖，没问题；但若别的测试直接 `w4afp8_kernel.is_w4a8_fp8_linear_supported()`，缓存值无法清空，需要显式 `.cache_clear()`。

### B17. `apply_weights` 内 int32 兜底 repack 没有加 `.contiguous()`（Low）

相比 `process_weights_after_loading` 中 `.contiguous()` 的规范做法，`apply_weights` 的兜底路径直接把 `_unpack_repack_int32_to_cutlass_int8(...)` 的输出（函数末尾有 `.contiguous()`）喂给 kernel，但 `layer.weight_scale` 在该路径下**未 `.contiguous()`**。若 scheme init 时 scale 非 contiguous，会被 FFI 拦截。

### B18. 可能出现的 `x_scale` shape 扩展不必要的 `.expand`（Low）

[w4afp8_linear.py#L92-L95](file:///Users/bytedance/Desktop/WYZ/Code/20260305-Sglang-bytedance/sglang-bytedance/python/sglang/srt/layers/quantization/w4afp8_linear.py#L92-L95)：
```python
q_input, x_scale = scaled_fp8_quant(input_2d, input_scale)
if x_scale.ndim == 1:
    x_scale = x_scale.view(1, 1).expand(input_2d.shape[0], 1)
return q_input.contiguous(), x_scale.contiguous().to(torch.float32)
```
`scaled_fp8_quant` 在静态 scalar 情形下可能返回 shape `[1]` 的 scalar scale，这里 `.view(1,1).expand(M, 1)` 确实会被 `.contiguous()` materialize 为 `[M, 1]`。但注意：`expand` 之后的 tensor 非 contiguous，`.contiguous()` 会物化 M 个 float —— 语义没问题，但每次前向都会分配。对 static-scale 路径影响不大（该路径本来就是边界情形），保留。

### B19. FFI `TensorMatcher` 与 Python validator 双重校验带来的错误定位混乱（Low）

Python validator 已经比较全面，但每项规则在 FFI 侧又重做一次。错误消息不一致时很难排查。不影响正确性。

---

## 三、建议的修复顺序（最小代价）

1. **B1（Blocker）**：把 `_is_w4afp8` 改为 `@staticmethod`，或修正测试用 `config._is_w4afp8(w, i)` 调用方式。
2. **B2（Blocker）**：确认 `pack_to_int32` 的实际编码（查 compressed-tensors 上游源或 MoE 的真实 ckpt），统一 repack 与 dense 测试。若 MoE 线上已跑通 ckpt，应相信 MoE 注释，**修测试**。
3. **B6 / B7**：强制 `x_scale` / `weight_scale` 为 fp32；在 `process_weights_after_loading` 统一 cast。
4. **B8**：在 wrapper 内 `bias = bias.to(output_dtype)` 自动对齐。
5. **B9**：`is_w4a8_fp8_linear_supported` 加一次小 shape 的试运行或至少试编译。
6. **B3 / B17**：去掉 `apply_weights` 里的兜底 repack，或补 `.contiguous()` 并回写 layer。
7. **B14**：在 fallback 说明中清楚注明它不是精度基准，集成测试只用真实 kernel 对比。
8. **B13**：用 `(high_nibbles & 0x0F) << 4` 以消除带符号左移歧义。
9. 其它 Medium/Low 项按 roadmap Step 1-3 推进。

---

## 四、静态分析无法覆盖的开放项

- naive CUDA kernel 尚未在真实 GPU 上做过一次端到端编译运行；无法验证 `to_float<fp8_e4m3_t>` 特化是否在当前 `sgl_kernel/utils.cuh` 导出的 `fp8_e4m3_t` 类型上语法通过。
- `fp8_e4m3_t` 是否为 `__nv_fp8_storage_t` 的 alias 决定了 `__nv_cvt_fp8_to_fp16(value, __NV_E4M3)` 的调用是否成立。需真机校验。
- TVM FFI 对 `Optional<TensorView>` 从 Python `None` 入参的自动转换需在 JIT 编译通过后再烟雾测试一次。
