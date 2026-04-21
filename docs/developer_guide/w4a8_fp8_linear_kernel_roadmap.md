# W4A8-FP8 Linear Kernel Development Roadmap

## Goal

This document breaks down the implementation plan for a true `W4A8-FP8` dense `Linear` path in SGLang, with tasks organized by file and precise function-level interfaces.

The target is not a debug-only fallback such as:

- dequantize `INT4` weights to `bf16/fp16`
- run `torch.matmul`

Instead, the target path is:

- keep weights in `INT4 packed` runtime format
- quantize activations to dynamic `FP8` at runtime
- run a dedicated dense GEMM kernel that consumes quantized operands directly

## MVP Scope

The first implementation should intentionally keep the scope small:

- CUDA only
- SM90+ only
- `bf16` input and output
- `group_size = 128`
- dynamic per-token FP8 activation quantization
- compressed-tensors checkpoint format remains `pack_to_int32`
- runtime weight layout uses CUTLASS-friendly `int8-packed int4`

## Planned Call Chain

The expected end-to-end call chain is:

1. `CompressedTensorsConfig._get_scheme_from_parts()` recognizes `W4AFP8`
2. `CompressedTensorsW4AFP8.create_weights()` registers `weight_packed` and `weight_scale`
3. `CompressedTensorsW4AFP8.process_weights_after_loading()` converts checkpoint layout into runtime kernel layout
4. `CompressedTensorsW4AFP8.apply_weights()` calls `cutlass_w4a8_fp8_linear(...)`
5. `cutlass_w4a8_fp8_linear(...)` quantizes the input to FP8 and calls the dense kernel

## Current Status

As of the current prototype:

- `CompressedTensorsW4AFP8` has been added and wired into dense `CompressedTensors` scheme dispatch.
- `create_weights()` and `process_weights_after_loading()` are implemented, including checkpoint `int32` -> runtime `int8` repacking.
- `apply_weights()` has been split to call `python/sglang/srt/layers/quantization/w4afp8_linear.py`.
- `python/sglang/srt/layers/quantization/w4afp8_kernel.py` exists and now lazily imports the JIT entry instead of hard-importing it at module import time.
- `python/sglang/jit_kernel/w4a8_fp8_scaled_mm.py` exists as the JIT wrapper entry.
- `python/sglang/jit_kernel/csrc/gemm/w4a8_fp8_scaled_mm.cuh` now exists as a shape-checked kernel skeleton.

The main remaining gaps are:

- `w4afp8_linear.py` still uses a temporary reference path (`dequantize + matmul`) instead of `FP8 quant + kernel dispatch`.
- `w4a8_fp8_scaled_mm.cuh` is only a skeleton and intentionally raises "not implemented yet" at runtime.
- `is_w4a8_fp8_linear_supported()` is more robust than before, but it still does not prove that the JIT kernel can successfully compile and execute end-to-end.
- Dense utility tests and integration tests are still missing.

## Runtime Tensor Contract

### Checkpoint Layout

- `weight_packed`: `[N, K // 8]`, `int32`
- `weight_scale`: `[N, K // group_size]`, `float32`

This matches compressed-tensors `pack_to_int32` serialization for 4-bit weights.

### Runtime Kernel Layout

- `q_input`: `[M, K]`, `float8_e4m3fn`
- `x_scale`: `[M, 1]`, `float32`
- `weight_packed`: `[N, K // 2]`, `int8`
- `weight_scale`: `[N, K // group_size]`, `float32`
- `output`: `[M, N]`, `bf16`

Each `int8` weight byte stores two signed `int4` values.

## File-by-File Development Tasks

### 1. Add Dense Linear Wrapper

- File: `python/sglang/srt/layers/quantization/w4afp8_linear.py`
- Purpose: provide the Python entry point for dense `W4A8-FP8` linear inference and hide input quantization plus kernel invocation details.
- Status: partially complete. The file exists, but it still routes through a temporary reference implementation.

#### Functions to Add

```python
def cutlass_w4a8_fp8_linear(
    input: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Args:
        input: [*, K], bf16/fp16. Internally reshaped to [M, K].
        weight_packed: [N, K // 2], int8, two signed int4 values per byte.
        weight_scale: [N, K // group_size], fp32/bf16.
        group_size: quantization group size. MVP fixes this to 128.
        input_scale: optional externally provided activation scale.
        bias: [N] or None.
        output_dtype: defaults to input.dtype, MVP should use bf16.
    Returns:
        output: [*, N]
    """
```

```python
def quantize_input_to_fp8_per_token(
    input_2d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        input_2d: [M, K], bf16/fp16.
    Returns:
        q_input: [M, K], fp8.
        x_scale: [M, 1], fp32.
    """
```

```python
def quantize_input_to_fp8(
    input_2d: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Unified input quantization entry.
    MVP should default to dynamic per-token FP8.
    """
```

#### Notes

- Reuse existing helpers from `fp8_kernel.py`, preferably `scaled_fp8_quant()` or `sglang_per_token_quant_fp8()`.
- The current prototype still contains a debug/reference fallback in this file. Replacing it with `FP8 quant + kernel dispatch` is now the highest-priority Python-side task.

### 2. Add CompressedTensors Dense Linear Scheme

- File: `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py`
- Purpose: implement the compressed-tensors dense `Linear` scheme for `W4AFP8`.
- Status: complete for the prototype stage.

#### Class to Add

```python
class CompressedTensorsW4AFP8(CompressedTensorsLinearScheme):
    def __init__(
        self,
        quant_config,
        weight_quant,
        input_quant,
    ) -> None:
        ...
```

#### Functions to Add

```python
@classmethod
def get_min_capability(cls) -> int:
    """
    Returns:
        90
    """
```

```python
def create_weights(
    self,
    layer: torch.nn.Module,
    input_size_per_partition: int,
    output_partition_sizes: list[int],
    input_size: int,
    output_size: int,
    params_dtype: torch.dtype,
    weight_loader: Callable,
    **kwargs,
) -> None:
    """
    Register checkpoint-aligned parameters.

    Args:
        input_size_per_partition: local K for the current partition.
        output_partition_sizes: output partitions for the current layer.
        input_size: global K.
        output_size: global N.

    Side Effects:
        Register the following parameters on `layer`:
        - weight_packed
        - weight_scale
        - weight_shape (recommended)
    """
```

```python
def process_weights_after_loading(
    self,
    layer: torch.nn.Module,
) -> None:
    """
    Convert compressed-tensors checkpoint layout into runtime kernel layout.

    Input params on layer:
        weight_packed: [N_local, K_local // 8], int32
        weight_scale: [N_local, K_local // group_size], fp32

    Output params on layer:
        weight_packed: [N_local, K_local // 2], int8
        weight_scale: keep original layout or reorder if kernel requires it
    """
```

```python
def apply_weights(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Args:
        x: [*, K_local]
        bias: [N_local] or None
    Returns:
        y: [*, N_local]
    """
```

```python
def _validate_shapes(
    self,
    input_size_per_partition: int,
    output_size_per_partition: int,
) -> None:
    """
    Validate K alignment, group size support, and MVP constraints.
    """
```

```python
def _runtime_weight_shape(
    self,
    output_size_per_partition: int,
    input_size_per_partition: int,
) -> tuple[int, int]:
    """
    Returns:
        (N_local, K_local // 2) for int8-packed int4 runtime layout
    """
```

```python
def _checkpoint_weight_shape(
    self,
    output_size_per_partition: int,
    input_size_per_partition: int,
) -> tuple[int, int]:
    """
    Returns:
        (N_local, K_local // 8) for int32 pack_to_int32 checkpoint layout
    """
```

#### Notes

- Reuse the repack logic already present in `compressed_tensors_w4a8_fp8_moe.py`.
- Avoid adding complex weight-scale interleave in the first dense version unless the kernel strictly requires it.
- The current dense prototype keeps helper logic local to the scheme file. Moving shared helpers into `w4afp8_utils.py` remains cleanup work, not a blocker.

### 3. Export the New Scheme

- File: `python/sglang/srt/layers/quantization/compressed_tensors/schemes/__init__.py`
- Purpose: make the new dense scheme importable from the compressed-tensors scheme package.
- Status: complete.

#### Changes

Add:

```python
from .compressed_tensors_w4a8_fp8 import CompressedTensorsW4AFP8
```

Update `__all__` with:

```python
"CompressedTensorsW4AFP8",
```

### 4. Connect Scheme Dispatch

- File: `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`
- Purpose: enable `Linear` layers to select the new `W4AFP8` dense scheme.
- Status: complete.

#### Changes

- Import `CompressedTensorsW4AFP8`
- Add a `W4AFP8` branch inside `_get_scheme_from_parts(...)`

#### Helper Function to Add

```python
def _build_w4afp8_linear_scheme(
    self,
    weight_quant: BaseModel,
    input_quant: BaseModel,
) -> "CompressedTensorsW4AFP8":
    """
    Returns:
        A CompressedTensorsW4AFP8 instance.
    """
```

#### Target Dispatch Branch

```python
if self._is_w4afp8(weight_quant, input_quant):
    return CompressedTensorsW4AFP8(
        quant_config=self,
        weight_quant=weight_quant,
        input_quant=input_quant,
    )
```

### 5. Extract Shared W4AFP8 Utilities

- File: `python/sglang/srt/layers/quantization/w4afp8_utils.py`
- Purpose: hold shared weight repack and debug/reference dequant helpers for both dense and MoE paths.
- Status: not started.

#### Functions to Add

```python
def unpack_repack_int32_to_cutlass_int8(
    weight_packed: torch.Tensor,
    num_bits: int,
) -> torch.Tensor:
    """
    Args:
        weight_packed: [..., K // 8], int32
    Returns:
        repacked: [..., K // 2], int8
    """
```

```python
def dequantize_w4_groupwise(
    weight_packed_int8: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Debug/reference-only helper, not the main inference path.

    Args:
        weight_packed_int8: [N, K // 2], int8
        weight_scale: [N, K // group_size]
    Returns:
        weight_dequant: [N, K], output_dtype
    """
```

```python
def unpack_int4_int8_packed(
    weight_packed_int8: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        weight_packed_int8: [N, K // 2], int8
    Returns:
        weight_int4_expanded: [N, K], int8 in the range [-8, 7]
    """
```

#### Notes

- This file is also the best place for unit tests.
- Once this file exists, `compressed_tensors_w4a8_fp8_moe.py` should ideally reuse it.
- This refactor is lower priority than getting the dense kernel path working end-to-end.

### 6. Add Dense Kernel Python Binding Layer

- File: `python/sglang/srt/layers/quantization/w4afp8_kernel.py`
- Purpose: thin Python wrapper around the new `jit_kernel` dense operator entry.
- Status: partially complete. The wrapper exists, validates inputs, and lazily imports the JIT entry.

#### Functions to Add

```python
def w4a8_fp8_scaled_mm(
    q_input: torch.Tensor,
    weight_packed: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Args:
        q_input: [M, K], fp8
        weight_packed: [N, K // 2], int8
        x_scale: [M, 1], fp32
        weight_scale: [N, K // group_size], fp32/bf16
        out_dtype: bf16/fp16
    Returns:
        output: [M, N]
    """
```

```python
def is_w4a8_fp8_linear_supported() -> bool:
    """
    Returns:
        Whether the current hardware/runtime supports the kernel.
    """
```

If fake-op registration is needed in a Torch custom-op path:

```python
@torch.library.register_fake("sglang::w4a8_fp8_scaled_mm")
def _w4a8_fp8_scaled_mm_abstract(
    q_input,
    weight_packed,
    x_scale,
    weight_scale,
    group_size,
    out_dtype,
    bias=None,
):
    ...
```

### 7. Add CUDA/CUTLASS Dense Kernel

- Files:
  - `python/sglang/jit_kernel/w4a8_fp8_scaled_mm.py`
  - `python/sglang/jit_kernel/csrc/gemm/w4a8_fp8_scaled_mm.cuh`
- Purpose: implement the actual dense `W4A8-FP8` GEMM kernel behind the JIT wrapper.
- Status: partially complete. The Python wrapper and `.cuh` skeleton exist, but the launcher/kernel body is not implemented yet.

#### JIT Export Signature

```cpp
template <typename OutDType>
void w4a8_fp8_scaled_mm(
    tvm::ffi::TensorView q_input,
    tvm::ffi::TensorView weight_packed,
    tvm::ffi::TensorView x_scale,
    tvm::ffi::TensorView weight_scale,
    tvm::ffi::TensorView output,
    int64_t group_size,
    tvm::ffi::Optional<tvm::ffi::TensorView> bias
);
```

#### Launcher Signature

```cpp
template <typename OutDType>
void launch_w4a8_fp8_scaled_mm(const W4A8FP8ScaledMMParams& params, DLDevice device);
```

#### Kernel Input Contract

- `q_input`: `[M, K]`, `float8_e4m3fn`
- `weight_packed`: `[N, K / 2]`, `int8`
- `x_scale`: `[M, 1]`, `float`
- `weight_scale`: `[N, K / group_size]`, `float`
- `output`: `[M, N]`, `bf16`

### 8. Add Build and Registration Wiring

- Files:
  - `python/sglang/jit_kernel/w4a8_fp8_scaled_mm.py`
  - `python/sglang/srt/layers/quantization/w4afp8_kernel.py`
- Purpose: ensure the JIT entry is discoverable from Python and the runtime support probe does not over-claim availability.

#### Required Outcome

The following import should work:

```python
from sglang.jit_kernel.w4a8_fp8_scaled_mm import w4a8_fp8_scaled_mm
```

And the higher-level wrapper should route through:

```python
from sglang.srt.layers.quantization.w4afp8_kernel import w4a8_fp8_scaled_mm
```

The next improvement here is to make runtime support probing more conservative so that a placeholder `.cuh` file is not treated as "fully supported".

### 9. Refactor MoE Path to Reuse Shared Utility

- File: `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8_moe.py`
- Purpose: reduce duplicated repack logic between dense and MoE.

#### Suggested Refactor

- Remove the file-local `_unpack_repack_int32_to_cutlass_int8`
- Import `unpack_repack_int32_to_cutlass_int8` from `w4afp8_utils.py`

This is not mandatory for the first dense prototype, but it should be part of the cleanup phase.

### 10. Add Utility Unit Tests

- File: `tests/python/quantization/test_w4afp8_utils.py`
- Purpose: verify repack and reference dequant logic.

#### Tests to Add

```python
def test_unpack_repack_int32_to_cutlass_int8_matches_reference() -> None:
    ...
```

```python
def test_unpack_int4_int8_packed_roundtrip() -> None:
    ...
```

```python
def test_dequantize_w4_groupwise_matches_manual_formula() -> None:
    ...
```

### 11. Add Dense Scheme Unit Tests

- File: `tests/python/quantization/test_compressed_tensors_w4a8_fp8_linear.py`
- Purpose: verify dense compressed-tensors scheme behavior.

#### Tests to Add

```python
def test_create_weights_registers_expected_params() -> None:
    ...
```

```python
def test_process_weights_after_loading_repacks_weight_layout() -> None:
    ...
```

```python
def test_apply_weights_matches_reference_dequant_matmul() -> None:
    ...
```

```python
def test_compressed_tensors_config_selects_w4afp8_linear_scheme() -> None:
    ...
```

### 12. Add Integration Tests

- File: `tests/python/models/test_w4afp8_linear_integration.py`
- Purpose: verify the new path on real linear module variants.

#### Tests to Add

```python
def test_column_parallel_linear_w4afp8_forward() -> None:
    ...
```

```python
def test_row_parallel_linear_w4afp8_forward() -> None:
    ...
```

```python
def test_merged_column_parallel_linear_w4afp8_forward() -> None:
    ...
```

`MergedColumnParallelLinear` must be covered because fused MLP paths such as `gate_up_proj` are one of the highest-risk integration points.

## Remaining Implementation Order

### Step 1

- Replace the temporary reference path in `w4afp8_linear.py`
- Add `quantize_input_to_fp8_per_token()`
- Add `quantize_input_to_fp8()`
- Route `cutlass_w4a8_fp8_linear()` to `w4a8_fp8_scaled_mm(...)`

### Step 2

- Make `is_w4a8_fp8_linear_supported()` more conservative
- Ensure the probe reflects "can really execute" rather than just "wrapper exists"

### Step 3

- Fill in `python/sglang/jit_kernel/csrc/gemm/w4a8_fp8_scaled_mm.cuh`
- Start with a minimal correct kernel path before tuning
- Keep `group_size = 128` and `bf16` output fixed for MVP

### Step 4

- Add dense utility tests and dense scheme tests
- Validate checkpoint repack, shape contracts, and reference numerical parity

### Step 5

- Add integration tests for standard and merged linear modules
- Validate tensor-parallel partition handling

### Step 6

- Refactor MoE path to reuse shared repack utilities
- Clean up any temporary debug-only code

## Suggested Remaining Commit Split

### Commit 1

- `w4afp8_linear.py`
- `w4afp8_kernel.py`
- support-probe hardening
- focused unit tests for wrapper behavior

### Commit 2

- `jit_kernel/w4a8_fp8_scaled_mm.py`
- `jit_kernel/csrc/gemm/w4a8_fp8_scaled_mm.cuh`
- first runnable dense kernel path
- kernel path tests

### Commit 3

- integration tests
- merged linear and tensor-parallel support
- shared utility refactor for MoE
- cleanup

## Interface Freeze Checklist

Before implementation begins, freeze the following interfaces to avoid churn:

- runtime `weight_packed` shape is `[N, K // 2]`
- `weight_scale` shape is `[N, K // group_size]`
- `x_scale` shape is `[M, 1]` for MVP
- kernel output dtype is `bf16`
- activation quantization is dynamic per-token FP8 for MVP

## Non-Goals for the First Version

The first version should not attempt to solve everything:

- ROCm support
- static activation scale support
- arbitrary group sizes
- all backend variants
- fused activation or residual epilogues
- block-sparse or mixed sparse kernels

Keeping the first version narrow will make it much easier to validate correctness and performance before broadening support.
