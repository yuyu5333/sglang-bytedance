
# 一、先把类比对齐
图像水印（频域隐写）做的事：

``` bash
image  →  FFT  →  在「鲁棒频段」嵌入信息  →  iFFT  →  传输/攻击  →  FFT  →  解码
                          ↑                       ↑
                  选能量集中、对扰动稳定的系数      旋转/裁剪/摩尔纹 = 信道噪声
```

它能抗攻击的关键有三点：
变换到一个 "扰动更可预测" 的域（频域里，几何攻击 ≈ 低频系数稳定 + 高频系数被破坏）；
能量在变换后稀疏 / 集中（少数频段承载主要信息）；
把 "信号" 放到鲁棒子空间，把 "噪声" 放到牺牲子空间—— 这是一种带先验的资源分配。
KV Cache 低 bit 量化做的事，本质上是同一个问题：

``` bash
K/V tensor  →  R 旋转  →  量化到 INT2/INT3  →  解码  →  attention 计算
                ↑              ↑                      ↑
        把 outlier "摊平"     有损压缩 = 信道噪声      注意力 = 解码器
```

两者的数学结构几乎一一对应：旋转 R ↔ FFT 基；量化噪声 ↔ 几何攻击；attention output ↔ 解码后的水印。

# 启发一："鲁棒频段" 概念 → 非均匀比特分配
图像水印不会把水印均匀撒到所有 DCT 系数里 —— 而是挑中频：低频改了人眼能看见，高频改了会被 JPEG 压缩掉，中频才是 "扰动小 + 不易被破坏" 的甜区。
KV 量化里完全可以照搬：
- 把 K（或 V）经过旋转 R 之后，不同坐标对 attention output 的 "重要度" 是高度不均的—— 这是 OSCAR 用协方差谱推导旋转的根本动机。
- 但 OSCAR 还是给所有坐标分配相同的 bit 数（INT2 一刀切）。
- 可以更进一步：用谱能量 / Fisher information / 对 softmax (QKᵀ) 的 Jacobian 范数，给每个坐标算一个 "重要度分数"，然后做非均匀 bit allocation—— 重要坐标 INT3/INT4，次要坐标 INT1/INT2，平均比特数仍是 2。

这就是经典的 rate-distortion 比特分配（Shannon-Lloyd 的 reverse water-filling），图像编码用了 30 年，KV 量化目前几乎没人用。
OCTOPUS 的 "三元组联合量化 + 解析最优 bit 分配" 已经触及这个思路，但它的分配只依赖维度数 d，不依赖具体坐标的能量，还有进一步空间。

# 启发二（激进路线 / 已落地）：真正"频域 KV cache"
把每个 KV page 当成沿 token 维的长度为 P 的"信号"，做正交 DCT-II，只保留前 r 个低频系数作为该 page 的表示，所有的稀疏检索（top-k page selection）完全在频域里完成、永不需要重建原始 K 矩阵。这是图像水印"先 FFT 再嵌入"那套完整搬到 KV cache 上的最干净版本。

## 数学结构
- 设一个 page 的 keys 为 K ∈ R^{P×H×D}，正交 DCT-II 系数为
  Y[k, h, d] = Σ_n C[k, n] · K[n, h, d],   C 是 [r, P] 的截断 DCT-II 基（orthonormal scaling）
- 仅保留 r 个低频行，存储量从 P·H·D 降到 r·H·D（r=4, P=64 时 16× 压缩，r=1 时退化为 mean-pooling）
- 给定 query q ∈ R^{H×D}：
  · DC 项 ⟨q, Y[0]⟩ 等价于 q · 页均值（严格泛化 mean pooling 的 page selection）
  · L1 envelope Σ_k |⟨q, Y[k]⟩| 是该页内任一 token 真实点积分数 max_n ⟨q, K[n]⟩ 的紧上界
  · L2 / Parseval 模式 √Σ_k ⟨q, Y[k]⟩² 给的是页能量重要性

## 鲁棒性的来源（与图像水印一致）
1. token 级噪声（驱逐、量化误差）会被 DCT 摊到所有低频系数上，单点扰动不会摧毁 page 表示——这就是"频域水印抗裁剪"的同构版本
2. 高频抖动（page 内边界 token 突变 ≈ 摩尔纹）天然落到被丢弃的高频带，不进入 score
3. r 越小越鲁棒、越省内存，但也越接近 mean-pooling；r=4 在压缩与判别度之间是甜区

## 当前仓库的落地点
- 新增算法实现：[freq_domain_algorithm.py](python/sglang/srt/mem_cache/sparsity/algorithms/freq_domain_algorithm.py)
  · `_build_dct_basis(P, r)` 构造正交 DCT-II 基（按 a_0=√(1/P), a_k=√(2/P) 的标准 scaling）
  · `_compute_page_representations`：在 prefill / decode 时一次 einsum 把每个 page 的 K 投影到低频空间，padding token 先掩零再变换
  · `_retrieve_page_scores`：批量算 ⟨q, Y[k]⟩，按 score_mode ∈ {l1, l2, dc} 三选一聚合
- 注册：[algorithms/__init__.py](python/sglang/srt/mem_cache/sparsity/algorithms/__init__.py)、[sparsity/__init__.py](python/sglang/srt/mem_cache/sparsity/__init__.py) 导出 `FreqDomainAlgorithm`；[factory.py](python/sglang/srt/mem_cache/sparsity/factory.py) 在 `_ALGORITHM_REGISTRY` 中加入 `"freq_domain"`
- 启动方式：在 `--hisparse-config` 里传 JSON
  ```json
  {
    "algorithm": "freq_domain",
    "backend": "fa3",
    "top_k": 2048,
    "page_size": 64,
    "num_freq_keep": 4,
    "score_mode": "l1",
    "sparsity_ratio": 0.7,
    "num_recent_pages": 4
  }
  ```

## 后续可继续推进的子方向
1. 基础换成 Walsh-Hadamard：纯 ±1 矩阵，无 cos 计算，Triton kernel 友好，本质做的是同样的"分散 outliers"
2. r 自适应：按层 / 按 head 让 r 不同——浅层 r 大、深层 r 小，沿用启发一中"非均匀 bit 分配"的思路（这里是非均匀 freq 分配）
3. 在频域系数 Y 上再叠 INT4/INT2 量化：先变换 → 再量化，是 QuaRot/SpinQuant 的频域版本
4. 把 r=1 + L1 envelope 和 Quest 的 min/max 做组合：DC 给主项、Quest 给次项，可作为打分融合


# 启发三（主线 / 直接改 KV cache 存储）：旋转 + 非均匀 bit 分配（QuaRot/SpinQuant + Lloyd reverse water-filling）

跟启发二的"频域稀疏检索"完全是两回事：这条路**直接改 KV cache 的存储位宽**，目标是把默认的 16 bit BF16 KV cache 真正换成平均 ~2 bit 的 packed INT 存储。

## 总体方案
```
write 路径:  K_raw (bs, H, D)
              │
              ├─ 1) 左乘 Hadamard 旋转 R   →  K_rot = K_raw @ R
              │       (R 是 D×D 正交、±1/√D ⇒ outliers 摊平到 D 维)
              │
              ├─ 2) 按 per-coordinate bit 表 b[d] 做非均匀 affine 量化:
              │       q[d] = round((K_rot[..,d] - z[d]) / s[d]) ∈ [0, 2^b[d] - 1]
              │       (s[d], z[d] 来自校准期统计，b[d] 来自 reverse water-filling)
              │
              └─ 3) bit-pack 到 uint8 buffer（每 head 行长 = ⌈Σ_d b[d] / 8⌉ 字节）

read 路径:   uint8 row → unpack → 反 affine → 右乘 Rᵀ → K̂
              (K̂ 接回正常 attention，Q 同步左乘 R 后等价计算)
```

R 选 Walsh-Hadamard：D=128 时只有 ±1/√128，无三角函数，Triton 里就是 sign + 加法，前向几乎零 overhead；正交性保证 `<Q,K> = <QR, KR>`，attention 数学上等价。

## 数学结构（精度的来源）

### a) 旋转分散 outliers
对一个 head 的 K ∈ ℝ^{N×D}，列协方差 Σ = Kᵀ K / N 经过正交旋转 R 后变成 RᵀΣR。Hadamard R 在期望意义上把 Σ 的对角能量推向均匀分布：原本某几列 σ_i² 极大（outlier 通道），旋转后每列方差近似 tr(Σ)/D。这是 QuaRot 的核心引理。

### b) Lloyd reverse water-filling 给出最优 b[d]
量化噪声平均功率 D[d] ≈ σ_rot²[d] · 2^{-2 b[d]} · c（c 与分布形状有关，均匀分布下 c=1/3）。在「平均比特数 b̄ 固定」的约束下最小化 Σ D[d]，KKT 解出
```
b[d] = b̄ + (1/2) · log₂( σ_rot²[d] / G ),     G = (∏_d σ_rot²[d])^(1/D)
```
也就是 σ² 大的坐标多 bit、小的坐标少 bit；最后投影到整数集合 {1,2,3,4} 并保持 Σ b[d] = b̄·D（用最大权值法做整数化修正）。

### c) per-coordinate affine quantizer
- 范围用 robust quantile：`s[d] = (q_hi - q_lo) / (2^b[d] - 1)`、`z[d] = q_lo`
- `q_lo = quantile(K_rot[:,d], 0.001)`，`q_hi = quantile(0.999)`，避免单点极值毁了整列 scale

## 落地点（M0 / M1 / M2 三阶段）

### M0：算法 + 校准器（CPU/GPU 都能跑，不接 attention）—— **✅ 已完成**
新文件 [layers/quantization/rotated_kv_quant.py](python/sglang/srt/layers/quantization/rotated_kv_quant.py)：

| 组件 | 函数 | 说明 |
|---|---|---|
| 正交旋转 | `build_hadamard(D)` | 递归 Sylvester 构造，仅当 D 是 2 的幂；非 2 幂用块对角 Hadamard + 余项 identity |
| 校准统计 | `KVCalibrator.observe(K_rot)` | 在线累计每维 mean/var/quantile（Welford + reservoir） |
| 比特分配 | `allocate_bits(var, b_mean, b_min=1, b_max=4)` | reverse water-filling + 整数化修正，返回 b[d] ∈ {1..4} |
| 量化器 | `RotatedQuantizer(R, b, s, z)` | `quantize(K)` 返回 `(packed: uint8[…, row_bytes], )`；`dequantize(packed)` 还原 |
| 打包 | `bitpack_rowwise(codes, bits)` / `bitunpack_rowwise(...)` | 纯 PyTorch 版本（M0 用），逐位拼接 |

校准产物保存到一个 `.pt` 文件，结构 `{layer_id: {"R": [D,D], "b": [D], "s": [D], "z": [D], "row_bits": int, "row_bytes": int}}`。

**验收**：随机 K (N=4096, D=128, fp32) 经过 quantize → dequantize → 反旋转后，相对 L2 误差在 b̄=2 时 < 5%、b̄=3 时 < 2%。

### M1：MHATokenToKVPool 的 packed 存储 —— **接进 KV cache**
新文件 [mem_cache/rotated_quant_memory_pool.py](python/sglang/srt/mem_cache/rotated_quant_memory_pool.py)：

- class `RotatedQuantMHATokenToKVPool(MHATokenToKVPool)`：
  - `_create_buffers()`：按 `row_bytes_k`、`row_bytes_v`（来自校准产物）分配 `uint8` buffer，每个 token 一行
  - 重写 `set_kv_buffer()`：进来的 `cache_k/cache_v` 走 `RotatedQuantizer.quantize` → 写到 packed buffer
  - 重写 `get_key_buffer()/get_value_buffer()`：默认走"懒解包"——返回一个轻量 view 对象，attention 调用时再触发 `dequantize()` 把当前 batch 用到的 token 解到 BF16 workspace
  - 提供 `get_dequant_workspace(loc)`：把 `loc` 指向的 token 解到一块预分配的 BF16 buffer 上，attention backend 拿 BF16 buffer 跑

- `dequantize` 用 Triton kernel：每个 thread 取一个 token 的 packed row，按 `b[d]` 解码 → 反 affine → 反 Hadamard。M1 阶段 Hadamard 反变换可以先用 `torch.matmul` 做（D=128 的 mat-vec 几乎没成本），M2 再换成纯 sign+加法的 fast WHT kernel。

**接入点**：
- 在 [memory_pool_factory](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/python/sglang/srt/mem_cache/memory_pool.py)（即 KVCache 子类的工厂处）按 `--kv-cache-dtype rotated_quant` 路由到 `RotatedQuantMHATokenToKVPool`
- 校准产物路径用 `--rotated-kv-quant-config /path/to/calib.pt` 传入

**M1 自检结果（已通过）**：
- 校准 .pt 加载、`_DeviceBoundQuantizer.encode/decode` shape 校验通过
- 端到端 `set_kv_buffer`（quantize→packed uint8 buffer→`index_put`）+ `get_dequant_workspace`（loc 切片→bitunpack→反 affine→反 Hadamard）数值正确
- D=128, b̄=2.5 时 row_bytes=40，per-token 字节 = 160 vs BF16 1024 字节，**ratio=0.156（≈ b̄/16，理论值）**
- `--rotated-kv-quant-config` 已在 [server_args.py](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/python/sglang/srt/server_args.py) 注册，dataclass 字段 `rotated_kv_quant_config: Optional[str]`

**M1 已知限制 / 下一步**：
- 工厂处尚未路由：实例化 `RotatedQuantMHATokenToKVPool` 的代码点（`ScheduleBatch` 创建 KV pool 处）需要后续在 [model_runner / mem_cache 工厂](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/python/sglang/srt/mem_cache/memory_pool.py) 接 `if server_args.rotated_kv_quant_config: ...`
- bitpack/bitunpack 当前在 CPU 上做（往返一次），M2 替成 Triton 后可省掉 H2D/D2H
- 全 pool dequant（`get_key_buffer`）用作正确性 fallback，attention backend 应改走 `get_dequant_workspace(loc)`

### M2：性能优化 —— **真正加速**
- pack/unpack Triton kernel：一个 token row 一个 thread block，每 thread 处理 4 个坐标的 bit-extract（用 `tl.bfloat16` 反 affine）
- Fast Walsh-Hadamard：D=128 的 fast WHT 是 7 层蝶形，纯加减，比 mat-vec 快 ~10×
-融合 dequant + attention：把 dequant kernel 的输出直接喂给 FA3 的 query workspace，省一次 GMEM 往返
- 评估指标：
  - 显存：以 BF16 为基线，b̄=2 时 KV 占用 ≤ 16/2 = **8× 压缩**（含 row_bytes 对齐 padding 后实际 ~6.5×）
  - 延迟：长上下文 decode（L=128k）relative latency vs BF16 ≤ 0.55（即 ≥1.8× 加速）

## 风险与回退
1. **校准漂移**：长尾 prompt 可能让 σ² 估计偏移，b̄=2 时易出现尾部精度下降。回退路线：每层独立保留一个"逃逸通道" raw FP8 区，承接 q[d]=2^b[d]-1 饱和的 token；M0 阶段先不做，记录在 `dequantize` 的统计里
2. **GQA / MLA 兼容**：DSA / MLA 的 KV layout 不是 [N, H, D]，是 [N, H, nope+rope]。M1 先只挂在标准 MHA 上，MLA 走 [deepseek_v4_memory_pool.py](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py) 的现有 FP8 路径，留作 M2 之后的扩展
3. **Hadamard 与 RoPE 的交换**：RoPE 是逐 head_dim 做 (cos, sin) 旋转，跟 Hadamard 不交换；正确做法是 Hadamard 只乘到 nope 部分，rope 部分保留原样。M1 实现要在 quantizer 里区分 nope/rope 切片
4. **CUDA Graph**：dequant workspace 用静态预分配 + ring buffer，避免 kernel 形状变化打断 graph capture

## M0 即将实现的代码骨架
- `python/sglang/srt/layers/quantization/rotated_kv_quant.py`
  - `build_hadamard(D) -> torch.Tensor`
  - `class KVCalibrator`（observe / finalize）
  - `def allocate_bits(var, b_mean, b_min, b_max) -> Tensor[D] int`
  - `class RotatedQuantizer`（quantize / dequantize / save / load）
  - `def bitpack_rowwise(codes, bits)`、`def bitunpack_rowwise(packed, bits, D)`
- 配套自检（独立脚本，不依赖 sglang 运行时）：
  - Hadamard 正交性 `H Hᵀ = I`
  - bit 分配的 reverse water-filling 守恒：`Σ b[d] == b̄ · D`
  - quantize-dequantize roundtrip 在 b̄∈{2,3,4} 上的相对 L2 误差曲线



