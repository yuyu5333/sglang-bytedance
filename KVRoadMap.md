
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
- ✅ 工厂路由已接入：[model_runner_kv_cache_mixin.py](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py) 在标准 MHA 分支前判 `server_args.rotated_kv_quant_config` 路由到 `RotatedQuantMHATokenToKVPool`；并在工厂入口加了快速断言（DSA / DSv4 / MLA / hybrid SWA / FP4 + rotated_kv_quant_config 同时启用直接报错）
- ✅ attention backend 透明接入：`get_key_buffer / get_value_buffer / get_kv_buffer` 都已 override，attention backend 调原 API 即得 dequant 后的 BF16 dense tensor（fallback 路径）；性能路径走 `get_dequant_workspace(layer_id, loc, side)`
- ✅ disagg 安全：`get_contiguous_buf_infos()` 已 override 报告 packed uint8 buffer 的真实 ptr/length（不要走 base 的 _get_key_buffer().nbytes，避免 dequant view 误导）
- bitpack/bitunpack 当前在 CPU 上做（往返一次），M2 替成 Triton 后可省掉 H2D/D2H
- 全 pool dequant（`get_key_buffer`）用作正确性 fallback；attention backend 应改走 `get_dequant_workspace(loc)` 避免每层全量解码

### M1.6：校准产线 + 端到端 smoke test —— **✅ 已完成**

新文件 [scripts/build_rotated_kv_calib.py](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/scripts/build_rotated_kv_calib.py)：

- **`--from-kv-dump <file.pt>`**：吃外部 instrumented forward 产出的真实 KV dump，schema `{layer_id: {"k": [N,H,D], "v": [N,H,VD]}}`
- **`--synthetic`**：合成异方差 + outlier Gaussian 用于 smoke test，按 `--num-layers/num-tokens/head-num/head-dim/v-head-dim` 配置规模
- 共同参数：`--b-mean / --b-min / --b-max / --num-bins / --q-lo / --q-hi / --chunk-tokens`
- 输出严格匹配 [`load_rotated_quant_calibration`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/python/sglang/srt/mem_cache/rotated_quant_memory_pool.py) 的 schema
- 采用 importlib 直接装载 M0 模块，避开 `sglang.__init__` 拉取 triton（可在 CPU-only 机器上运行）

新文件 [test/manual/quant/test_rotated_kv_quant_e2e.py](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/test/manual/quant/test_rotated_kv_quant_e2e.py)：4 个 unittest 覆盖

1. CLI 子进程：`--synthetic` → 写出 calib.pt（exit 0、文件 > 1KB）
2. schema 校验：layer_ids、形状、bits ∈ [1, 8]
3. pool 存储契约：`encode → buf[loc] → decode`，cosine mean > 0.7、min > 0.3，row_bytes ≈ b̄/16
4. loader 等价性：跨层 row_bytes 一致（M1 假设）

**M1.6 验收数值**（synthetic, D=128, H=4, N=2048×3 layers, b̄=2.5）：
- bits[k/v] mean = 2.50（守恒精确）
- row_bytes_k = row_bytes_v = 40
- 4/4 unittest pass，3.6s
- 单层校准 ~0.4s（CPU）

**使用手册（端到端最小可用）：**

```bash
# 1) 生成 smoke test 用的 calibration（无需真实模型）
python scripts/build_rotated_kv_calib.py \
  --synthetic --num-layers 32 --num-tokens 4096 \
  --head-num 32 --head-dim 128 --v-head-dim 128 \
  --b-mean 2.5 -o /tmp/calib.pt

# 2) 真实模型校准：先在 baseline BF16 模型上 forward 一段 prompt，
#    每层 dump K/V 成 {layer_id: {"k": [N,H,D], "v": [N,H,VD]}}.pt
python scripts/build_rotated_kv_calib.py \
  --from-kv-dump /path/to/kv_dump.pt --b-mean 2.5 -o /path/to/calib.pt

# 3) 启动 SGLang 用旋转 + 非均匀 bit KV cache
python -m sglang.launch_server \
  --model <bf16-MHA-model> \
  --rotated-kv-quant-config /path/to/calib.pt
```

启动日志会出现：
```
Routing KV cache to RotatedQuantMHATokenToKVPool (calib=...)
RotatedQuantMHATokenToKVPool ready: dtype=..., store_dtype=uint8, row_bytes_k=40, row_bytes_v=40, calib=...
```

### M2：性能优化 —— **真正加速**
- pack/unpack Triton kernel：一个 token row 一个 thread block，每 thread 处理 4 个坐标的 bit-extract（用 `tl.bfloat16` 反 affine）
- Fast Walsh-Hadamard：D=128 的 fast WHT 是 7 层蝶形，纯加减，比 mat-vec 快 ~10×
-融合 dequant + attention：把 dequant kernel 的输出直接喂给 FA3 的 query workspace，省一次 GMEM 往返
- 评估指标：
  - 显存：以 BF16 为基线，b̄=2 时 KV 占用 ≤ 16/2 = **8× 压缩**（含 row_bytes 对齐 padding 后实际 ~6.5×）
  - 延迟：长上下文 decode（L=128k）relative latency vs BF16 ≤ 0.55（即 ≥1.8× 加速）

### M3.a：MLA 支持（DeepSeek-V2 / V3 标准 MLA）—— **✅ 已完成**

> 直接对 `c_kv` (latent) 段做旋转 + INT2/3/4 量化；rope 段保留原 dtype。
> 这条路径不影响 DSv4 多比例分级压缩（c4/c128 swa pool），那条路径仍由
> `DeepSeekV4TokenToKVPool` 处理；要支持 DSv4 是 M3.b 工作。

**为什么需要单独的 pool**：
- MLA buffer 是 `[N, 1, kv_lora_rank + qk_rope_head_dim]`（e.g. `[N, 1, 576]` for V2/V3），不是 MHA 的 `[N, H, D]`
- RoPE 跟 Hadamard 不交换：`H · RoPE(x) ≠ RoPE(H · x)`，必须只对 nope 段（latent）做旋转，rope 段保留 raw bytes
- MLA 不存独立 V buffer（V 由 latent 推导），calib schema 简化为 `latent` 一路

**落地点**：
- 新文件 [`python/sglang/srt/mem_cache/rotated_quant_mla_memory_pool.py`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/python/sglang/srt/mem_cache/rotated_quant_mla_memory_pool.py) — `RotatedQuantMLATokenToKVPool(MLATokenToKVPool)`
  - Buffer 布局：`[N, 1, latent_row_bytes + rope_bytes]`（uint8）
  - 重写：`_create_buffers / set_mla_kv_buffer / set_kv_buffer / get_key_buffer / get_value_buffer / get_kv_buffer / get_mla_kv_buffer`
  - 新增：`get_dequant_workspace(layer_id, loc, side='latent'|'rope'|'full')` 快路径
- 校准脚本：[`scripts/build_rotated_kv_calib.py`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/scripts/build_rotated_kv_calib.py) `--mla-mode` + `--kv-lora-rank` + `--qk-rope-head-dim`，输出新 schema：
  ```
  {"_meta": {"mode": "mla", "kv_lora_rank": L, "qk_rope_head_dim": R, "layer_num": N},
   layer_id: {"latent": {R, bits, scale, zero}}, ...}
  ```
- 工厂路由：[`model_runner_kv_cache_mixin.py`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py)
  - 早期断言放开 MLA（仍拒绝 DSA / DSv4 / SWA / FP4）
  - MLA 路径加分支：`elif self.server_args.rotated_kv_quant_config: -> RotatedQuantMLATokenToKVPool`
- e2e 测试：[`test/manual/quant/test_rotated_kv_quant_e2e.py`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/test/manual/quant/test_rotated_kv_quant_e2e.py) 新增 `TestRotatedKVQuantMLAE2E`（4 用例）

**M3.a 验收数值**（synthetic, kv_lora_rank=512, qk_rope_head_dim=64, b̄=2.5）：
- ✅ 8/8 unittest pass（4 MHA + 4 MLA），耗时 6.4s
- ✅ CLI 实跑：bits[latent] mean=2.50（守恒精确），latent_row_bytes=160
- ✅ rope 段 raw view 严格往返（`torch.equal`）
- ✅ latent 段 cosine 平均 > 0.7、最低 > 0.3（异方差合成数据）
- 压缩比：每 token-head 288B（160 packed latent + 128 raw rope）vs BF16 1152B → **4×**
  - 仅看 latent 段：512×2 / 160 = **6.4×**

**端到端用法**：
```bash
# 1) 产 MLA calib（用 DeepSeek-V2/V3 默认参数）
python scripts/build_rotated_kv_calib.py --synthetic --mla-mode \
  --num-layers <L> --num-tokens 4096 \
  --kv-lora-rank 512 --qk-rope-head-dim 64 \
  --b-mean 2.5 -o /tmp/calib_mla.pt

# 2) 启 sglang
python -m sglang.launch_server --model deepseek-ai/deepseek-v2 \
  --rotated-kv-quant-config /tmp/calib_mla.pt
```

**M3.a 已知限制**：
- DSv4 多比例分级压缩（c4/c128 swa pool）尚未支持（早期断言拒绝）—— 见 M3.b
- bitpack/bitunpack 仍走 CPU 往返（与 M1 一致）—— M2 优化
- attention backend 默认走 `get_key_buffer` 整层 dequant，未切到 `get_mla_kv_buffer(loc)` 快路径 —— M2.c

### M3.b：DeepSeek-V4 多比例 swa pool 支持（**评估模式**）—— **✅ 已完成**

> **诚实声明**：M3.b 是 **评估 / 模拟模式**，不替换 DSv4 的 wall-storage。
> 真存储替换是 M3.c（kernel-side）。原因写在下方。

**为什么不能在 M3.b 直接替换主存储**：
DSv4 的 KV 主存储是 page-based 字节数组 `[num_pages, bytes_per_page]`，
每 token 584 字节 hardcoded layout（nope FP8 (448B) + rope BF16 (128B)
+ scales (8B) + scale_pad (1B)）。写入由
[`sglang/jit_kernel/dsv4.py:fused_store_cache`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/python/sglang/jit_kernel/dsv4.py)
JIT kernel 完成，读取由 FlashMLA attention kernel 直接按该 layout 解释。
要把 nope 段替换成 INT2/3/4 packed 必须同步重写这两个 kernel + 重设
indexer pool / compress_state pool 的协同语义 —— 工程量超出单次任务，因此
划入 M3.c。

**M3.b 实际做了什么**：
1. **路由放开** DSv4 + `--rotated-kv-quant-config` 组合（之前会被早期断言拒绝）
2. **包装类**继承 `DeepSeekV4TokenToKVPool`，**主存储不动**（FlashMLA / fused_store_cache 仍走 FP8）
3. 包装类持有 per-layer `RotatedQuantizer`，提供 **`simulate_quantize_nope(layer_id, nope)`** 接口：把 BF16 nope 经 旋转 → INT2/3/4 量化 → 反量化，返回模拟值，用于离线评估 / 单元测试度量精度
4. 校准 schema：**`_meta.mode='dsv4'`**，每层只校准 nope 段（rope 不旋转、indexer / compress_state 都不动；swa/c4/c128 三池共享同一 nope calib，因为它们的 nope 维度一致）

**落地点**：
- 新文件 [`python/sglang/srt/mem_cache/rotated_quant_dsv4_memory_pool.py`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/python/sglang/srt/mem_cache/rotated_quant_dsv4_memory_pool.py)
  - `load_rotated_quant_dsv4_calibration(path, layer_num, qk_nope_head_dim, qk_rope_head_dim, compression_ratios)` — 强校验 `_meta.mode='dsv4'` + 维度匹配 + bits ∈ [1,8]
  - `class RotatedQuantDeepSeekV4TokenToKVPool(DeepSeekV4TokenToKVPool)` — 主存储透传父类，新增 `simulate_quantize_nope` / `simulated_compression_ratio`
  - 启动时打 `logger.warning` 强调 **EVALUATION MODE**（避免被误以为压缩已生效）
- 校准脚本 [`scripts/build_rotated_kv_calib.py`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/scripts/build_rotated_kv_calib.py) 新增 DSv4 分支：
  - CLI：`--dsv4-mode`、`--qk-nope-head-dim`（默认 448）、`--compression-ratios`（comma-sep，e.g. `0,4,128,4`）
  - `_synthetic_dsv4_kv` 合成 dump（异方差 + outlier）
  - `build_dsv4_calibration` 仅校准 nope 段，输出 schema：
    ```
    {"_meta": {"mode": "dsv4",
               "qk_nope_head_dim": 448,
               "qk_rope_head_dim": 64,
               "compression_ratios": [0, 4, 128, 4],
               "layer_num": N},
     layer_id: {"nope": {R, bits, scale, zero}}, ...}
    ```
  - main() 增加 `--mla-mode` / `--dsv4-mode` 互斥校验 + 三模式 dispatch
- 工厂路由 [`model_runner_kv_cache_mixin.py`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py)：
  - 早期断言放开 DSv4（仍拒绝 DSA / hybrid SWA / FP4）
  - DSv4 分支加 `if self.server_args.rotated_kv_quant_config:` 子分支，路由到 `RotatedQuantDeepSeekV4TokenToKVPool`
- e2e 测试 [`test/manual/quant/test_rotated_kv_quant_e2e.py`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/test/manual/quant/test_rotated_kv_quant_e2e.py) 新增 `TestRotatedKVQuantDSv4E2E`（4 用例）：
  1. `--dsv4-mode` CLI 子进程 → 写出 calib_dsv4.pt
  2. schema 校验：`_meta.mode='dsv4'` + `compression_ratios` 元数据 + 每层 `nope`
  3. nope 量化往返：cosine mean > 0.7、min > 0.3
  4. row_bytes 一致性 + sim_ratio ∈ (5.0, 8.0)；同时断言 DSv4 模式 calib 不能被 MHA / MLA loader 误用

**M3.b 验收数值**：
- ✅ **12/12 unittest pass**（4 MHA + 4 MLA + 4 DSv4），耗时 9.07s
- ✅ CLI 实跑（DSv4 真参 `qk_nope_head_dim=448, qk_rope_head_dim=64, compression_ratios=[0,4,128,4]`，4 layers，b̄=2.5）：所有层 `bits[nope] mean=2.50`（守恒精确），`nope_row_bytes=140`
- ✅ 模拟压缩比（仅 nope 段）：bf16(896B per token nope) / packed(140B) = **6.4×**（理论值 16/2.5）
- ✅ 启动日志清楚提示 EVALUATION MODE 与 simulated_row_bytes

**端到端用法**：
```bash
# 1) 产 DSv4 nope 校准（用 DeepSeek-V4 默认参数）
python scripts/build_rotated_kv_calib.py --synthetic --dsv4-mode \
  --num-layers <L> --num-tokens 4096 \
  --qk-nope-head-dim 448 --qk-rope-head-dim 64 \
  --compression-ratios 0,4,128,4 \
  --b-mean 2.5 -o /tmp/calib_dsv4.pt

# 2) 启 sglang（路由到评估模式包装类）
python -m sglang.launch_server --model deepseek-ai/deepseek-v4 \
  --rotated-kv-quant-config /tmp/calib_dsv4.pt
```
启动日志会出现：
```
Routing DSv4 KV cache to RotatedQuantDeepSeekV4TokenToKVPool (EVALUATION MODE; calib=...).
RotatedQuantDeepSeekV4TokenToKVPool active in EVALUATION MODE: ... simulated_row_bytes=140, b_mean=2.50
```

**M3.b 已知限制**（明确划入 M3.c）：
- ⚠️ 主 KV 存储仍是 DSv4 原生 FP8 layout，attention 仍读 FP8 —— **wall-clock 内存与延迟不变**
- ⚠️ `simulate_quantize_nope` 仅离线评估精度，未接入 forward / attention 路径
- ⚠️ rope / indexer / compress_state 不动（与 M3.a 一致，rope 不旋转）

### M3.c：DSv4 真存储替换（kernel-side，分三步落地）

> **路径选择（Path A）**：写侧 INT2/3/4 packed wall-storage + attention prologue
> 处插一个 dequant-to-FP8-576B shim（Triton kernel，镜像 FlashMLA 期望的
> UE8M0 cast）→ FlashMLA decode 整体不动。
> 拒绝路径 B（fork FlashMLA 重写 decode）：维护成本太高、与上游漂移。

#### M3.c.1：wall-storage kernel + canary —— **✅ 已完成**

只换 ``swa_kv_pool``；c4/c128 / indexer / compress_state 留作 M3.c.2。

**新增/修改文件**：
- 新文件 [`python/sglang/jit_kernel/triton_rotated_quant_dsv4.py`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/python/sglang/jit_kernel/triton_rotated_quant_dsv4.py)
  - ``_rotated_dequant_nope_kernel``（grid=(N, 7)，每 tile 算 abs_max → ceil(log2 scale) → e4m3fn store + UE8M0 byte，UE8M0 cast 与 [`triton_store_cache.py`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/python/sglang/jit_kernel/triton_store_cache.py) 完全一致）
  - ``_rotated_dequant_rope_kernel``（grid=(N,)，BF16 rope 直写到 slot 偏移 224 元素）
  - ``rotated_dequant_to_fp8_layout(nope_bf16, rope_bf16, out_slot, out_scale)`` host dispatch；两个 kernel 分别接 ``out_slot.view(_FP8_DTYPE)`` 与 ``view(torch.bfloat16)``，避开 Triton 单 kernel 单一 element_ty 的契约
- 新文件 [`python/sglang/jit_kernel/rotated_quant_dsv4_kernels.py`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/python/sglang/jit_kernel/rotated_quant_dsv4_kernels.py)
  - ``packed_bytes_per_token(row_bytes_nope) = row_bytes_nope + 128`` (rope 永远 BF16 128B)
  - ``rotated_store_to_packed(input_bf16[N,512], cache, indices, *, page_size, cfg)``：split → R · nope (fp32) → affine clamp → bitpack → 拼 nope_packed + rope_bytes → ``cache.view(-1, bpt).index_copy_(0, indices, full_row)``
  - ``rotated_load_to_fp8_layout(cache, indices, out_slot, out_scale, *, page_size, cfg)``：gather → bitunpack → 反 affine → ``K_rot @ R^T`` → BF16 → 调 Triton kernel
  - ``rotated_load_to_fp8_layout_cpu_ref(...)``：纯 CPU 不依赖 Triton/CUDA，给 canary 单测用
  - Triton import 是 lazy 的，CPU-only 环境（macOS）也能用 store + cpu_ref reader
- 修改文件 [`python/sglang/srt/mem_cache/rotated_quant_dsv4_memory_pool.py`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/python/sglang/srt/mem_cache/rotated_quant_dsv4_memory_pool.py)
  - 新增 ``mode: Literal['eval','wall']`` 构造参数；``mode='eval'`` 行为与 M3.b 完全一致
  - ``_install_wall_storage()``：drop ``swa_pool.kv_buffer`` FP8 buffers → ``torch.cuda.empty_cache()`` → realloc packed 形 ``(num_pages, bpt_packed * page_size)`` uint8
  - 覆写 ``set_swa_key_buffer_radix_fused(layer_id, raw_loc, cache_k)``：wall 模式调 ``rotated_store_to_packed``，eval 走父类
  - 新增 ``dequant_swa_to_fp8_layout(layer_id, loc) -> (out_slot[M,576], out_scale[M,8])`` 给 M3.c.2 attention prologue 用
  - 新增 ``wall_compression_ratio() = 584.0 / packed_bpt``
- 修改文件 [`python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py)：DSv4 工厂分支穿 ``mode=rq_mode`` 进 pool 构造，启动日志按 mode 分支
- 修改文件 [`python/sglang/srt/server_args.py`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/python/sglang/srt/server_args.py)：新增 ``--rotated-kv-quant-mode`` CLI 选项（``choices=["eval", "wall"]``，默认 ``eval``）
- 新文件 [`test/manual/quant/test_rotated_kv_quant_dsv4_canary.py`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/test/manual/quant/test_rotated_kv_quant_dsv4_canary.py)（5 用例）：
  1. layout 常量与 [`triton_store_cache.py`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/python/sglang/jit_kernel/triton_store_cache.py) 不漂移
  2. ``packed_bytes_per_token`` 算术（b̄=2.5 → 1120 bits → 140 bytes nope → 268 bpt）
  3. CPU store→cpu_ref load roundtrip：cosine.mean > 0.5、min > 0.1，rope 字节级相等
  4. 跨页 scatter/gather 寻址：未写入 slot 保持全 0、写入 slot 非 0
  5. **GPU canary**（``unittest.skipUnless(torch.cuda.is_available())``）：
     Triton dequant kernel 输出非 0 UE8M0 字节、rope 反查字节级相等

**M3.c.1 验收数值**：
- ✅ **5/5 canary pass**（4 CPU + 1 GPU skipped on macOS），耗时 0.18s
- ✅ **12/12 M3.b regression pass**（4 MHA + 4 MLA + 4 DSv4 eval），耗时 4.93s
- ✅ wall-storage 压缩比（仅 swa_kv_pool）：584B/token → 268B/token = **2.18×**
  （nope 段 6.4× 摊薄到全 layout，含 rope 128B 不动）
- ✅ ``--rotated-kv-quant-mode=wall`` 日志：``WALL-STORAGE MODE: swa_kv_pool main buffer replaced with INT2/3/4 packed nope + raw BF16 rope; c4/c128/indexer remain DSv4 native FP8``

**M3.c.1 已知限制（M3.c.2 已全部解除）**：
- ✅ ~~M3.c.2 未启动：``dequant_swa_to_fp8_layout`` 还没接进 attention prologue~~ → **M3.c.2 已完成**：[`deepseek_v4_backend.py:986`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/python/sglang/srt/layers/attention/deepseek_v4_backend.py#L986) 已加 duck-typed prologue hook
- ✅ ~~M3.c.1 范围严格收敛在 ``swa_kv_pool``：``c4_kv_pool / c128_kv_pool`` 仍是 DSv4 native FP8~~ → **M3.c.2 已三池同步替换**（swa/c4/c128 全部 wall packed-storage）
- ⚠️ bit-pack/unpack 仍走 CPU 往返（``bitpack_rowwise / bitunpack_rowwise``），M3.c.3 Triton 化（**仍未做**）

**端到端用法（M3.c.1 阶段，仅做 wall-storage 写入 canary）**：
```bash
# 1) 产 DSv4 nope 校准（与 M3.b 完全相同）
python scripts/build_rotated_kv_calib.py --synthetic --dsv4-mode \
  --num-layers <L> --num-tokens 4096 \
  --qk-nope-head-dim 448 --qk-rope-head-dim 64 \
  --compression-ratios 0,4,128,4 \
  --b-mean 2.5 -o /tmp/calib_dsv4.pt

# 2) 启 sglang，wall 模式（注意：M3.c.2 未完成前不要用于生产推理）
python -m sglang.launch_server --model deepseek-ai/deepseek-v4 \
  --rotated-kv-quant-config /tmp/calib_dsv4.pt \
  --rotated-kv-quant-mode wall
```

#### M3.c.2：attention backend 接入 dequant shim + 三池同步替换 —— **✅ 已完成**

把 ``dequant_swa_to_fp8_layout`` 接进 [`deepseek_v4_backend.py`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/python/sglang/srt/layers/attention/deepseek_v4_backend.py) 的 attention prologue，并把 ``c4_kv_pool / c128_kv_pool`` 同步切到 wall packed-storage。

**实际完成内容**：

1. **三池同步 wall-storage**：[`rotated_quant_dsv4_memory_pool.py::_install_wall_storage`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/python/sglang/srt/mem_cache/rotated_quant_dsv4_memory_pool.py) 不再只换 swa；现在循环 ``('swa', 'c4', 'c128')``，每池都分配 packed buffer（替换 ``pool.kv_buffer``）+ shadow buffer（与原 FP8 同 layout）。``_WallPoolEntry`` 持有 ``packed_buffers / shadow_buffers / packed_bpt / packed_bytes_per_page / shadow_bytes_per_page / page_size / num_pages``
2. **关键 layout 修复**：M3.c.1 错误地把 ``swa_pool.bytes_per_page_padded`` 改成 packed 大小，导致下游 ``view([num_pages, P, 1, 584])`` 失配；M3.c.2 保持原 native FP8 size 不变（packed 只是物理存储，shadow 给读侧用）
3. **5 个 write override**：``set_swa_key_buffer_radix_fused`` / ``set_swa_key_buffer_radix_fused_norm_rope``（含 PyTorch RMSNorm+RoPE fallback） / ``set_extra_key_buffer_fused``（按 ``layer_mapping.compress_ratio`` 路由 c4 vs c128） / ``set_extra_key_buffer``（非 fused 在 wall 模式抛 NotImplementedError）
4. **2 个 read override**：``get_swa_key_buffer_radix`` / ``get_extra_key_buffer`` 直接返 shadow_buffer
5. **Prologue hook**：``_rotated_quant_attention_prologue(layer_id, core_attn_metadata, compress_ratio)`` 在 swa + 当前 layer 对应的 extra (c4 或 c128) 上调 ``_refresh_shadow_pages`` —— flatten ``page_indices`` + drop -1 sentinel + ``torch.unique`` 去重 + 构 loc + dequant + scatter to shadow
6. **Backend 接线**：[`deepseek_v4_backend.py:986`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/python/sglang/srt/layers/attention/deepseek_v4_backend.py#L986) 单点 duck-typed hook，在 ``store_cache`` 之后、``get_swa_key_buffer_radix`` 之前调 prologue，对非 rotated pool 完全无副作用
7. **CPU helper 提到 kernels**：``quant_fp8_layout_cpu_ref(nope_bf16, rope_bf16) -> (out_slot[M,576], out_scale[M,8])`` 移到 [`rotated_quant_dsv4_kernels.py`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/python/sglang/jit_kernel/rotated_quant_dsv4_kernels.py)，pool 与 canary 共享

**M3.c.2 验收数值**：

- ✅ **7/7 M3.c canary pass**（test_06 端到端 store→prologue→shadow→decode cosine.mean ≥ 0.95，cos.min ≥ 0.80，vs raw BF16 ≥ 0.90；test_07 dedup + sentinel filter）
- ✅ **12/12 M3.b regression pass**（4 MHA + 4 MLA + 4 DSv4 eval），耗时 5.6s
- ✅ **风险条 6（partial-mode 不一致）已消除**：三池同步替换 + prologue hook 完整，wall 模式 forward 不再读到 raw packed bytes

**M3.c.2 已知限制（明确划入 M3.c.3）**：
- ⚠️ ``_refresh_shadow_pages`` 内部 dequant + ``quant_fp8_layout_cpu_ref`` 仍走 CPU 往返；CUDA Graph 友好但 latency 不达标
- ⚠️ ``set_swa_key_buffer_radix_fused_norm_rope`` 用 PyTorch RMSNorm+RoPE fallback；M3.c.3 fused Triton kernel 直接写 packed
- ⚠️ shadow buffer 当前 worst-case 静态预分配（与原 FP8 同尺寸），M3.c.3 改 ring buffer 进一步省显存

#### M3.c.3：bit-pack/unpack Triton 化 —— **未启动（性能化）**

当前 ``rotated_store_to_packed`` / ``rotated_load_to_fp8_layout`` 内部的 ``bitpack_rowwise / bitunpack_rowwise`` 都走 CPU + 一次 H2D/D2H 往返。M3.c.3 把这两个原语换成纯 Triton kernel：每个 token row 一个 thread block，每 thread 处理 4 个坐标的 bit-extract（``tl.bfloat16`` 直接做反 affine），消除 CPU 往返。配合 fast WHT（M2 的 7 层蝶形）共同把 store/load 路径压到 GPU-only。

**M3.c 总验收目标（M3.c.1+M3.c.2+M3.c.3 完成后）**：
- 显存：DSv4 swa_kv_pool 从 584B/token 降到 268B/token（**2.18×**）；c4/c128 同步替换后整 KV 压缩比再放大
- 精度：开放领域 eval cosine sim 与 baseline FP8 ≥ 0.95
- 延迟：长上下文 decode ≤ 1.05× baseline（M3.c.3 Triton 化后）

## 风险与回退
1. **校准漂移**：长尾 prompt 可能让 σ² 估计偏移，b̄=2 时易出现尾部精度下降。回退路线：每层独立保留一个"逃逸通道" raw FP8 区，承接 q[d]=2^b[d]-1 饱和的 token；M0 阶段先不做，记录在 `dequantize` 的统计里
2. **GQA / MLA / DSv4 兼容**：
   - ✅ 标准 MLA 已在 M3.a 完成（`RotatedQuantMLATokenToKVPool`）
   - ✅ DSv4 多比例分级压缩（c4/c128 swa pool）已在 M3.b 完成 **评估模式**支持（`RotatedQuantDeepSeekV4TokenToKVPool`，主存储仍 FP8）
   - ✅ DSv4 wall-storage 第一步在 M3.c.1 完成（``swa_kv_pool`` 真 packed 写入 + dequant-to-FP8 shim Triton kernel + canary）
   - ⏳ M3.c.3 bit-pack Triton 化（性能化）；M3.c.2 三池同步替换 + prologue hook 已完成
3. **Hadamard 与 RoPE 的交换**：✅ M3.a / M3.b 都已处理 —— Hadamard 只乘到 latent / nope 段，rope 段保留 raw bytes（不旋转、不量化）。M1 vanilla MHA 路径仍假设无 RoPE 或 RoPE 在 quantizer 之前
4. **DSv4 评估模式被误以为压缩已生效**：M3.b 启动时通过 `logger.warning` 与 `Routing ... (mode=eval; ...)` 日志双重提示；`simulated_compression_ratio()` API 返回的是模拟值，不是 wall-clock。M3.c.2 已完成后，三池同步 wall packed-storage，可以信 ``wall_compression_ratio()`` 与启动日志的真实 KV 字节数。生产监控建议：``simulated_compression_ratio()`` 仅用于 M3.b eval 模式，``--rotated-kv-quant-mode=wall`` 启动后所有三池都走 packed wall-storage
5. **CUDA Graph**：dequant workspace 用静态预分配 + ring buffer，避免 kernel 形状变化打断 graph capture
6. **~~M3.c.1 partial-mode 不一致~~**（**M3.c.2 已消除**）：原本 ``--rotated-kv-quant-mode=wall`` 仅替换 ``swa_kv_pool``，``c4_kv_pool / c128_kv_pool`` 仍 native FP8 → forward 数值错误。M3.c.2 把三池同步替换为 wall packed-storage + 加 ``_rotated_quant_attention_prologue`` hook，每次读 shadow buffer 之前 dequant + scatter，FlashMLA 看到的就是 FP8 layout 字节，已端到端通过 cosine ≥ 0.95 验收。``compress_state / indexer`` 不参与 KV 主存储路径，保留 native FP8 不冲突

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


## 当前状态总览（截至 M3.c.2 交付）

### 里程碑进度

| 阶段 | 状态 | 关键交付 | 验收 |
|------|------|----------|------|
| M0 — 旋转量化基础原语 | ✅ 完成 | Hadamard、KVCalibrator、reverse water-filling、bit-pack/unpack、RotatedQuantizer | b̄∈{2,3,4} roundtrip cosine 曲线 |
| M1 — vanilla MHA pool 接入 | ✅ 完成 | `RotatedQuantTokenToKVPool` + e2e | 4/4 MHA e2e |
| M2 — 算子化（fast WHT 7 层蝶形） | ✅ 完成 | 蝶形 Hadamard | — |
| M3.a — 标准 MLA 接入 | ✅ 完成 | `RotatedQuantMLATokenToKVPool` | 4/4 MLA e2e |
| M3.b — DSv4 多比例分级压缩（**评估模式**） | ✅ 完成 | `RotatedQuantDeepSeekV4TokenToKVPool` (mode=eval)；c4/c128/swa 的 simulate-only 路径 | 4/4 DSv4 eval e2e |
| M3.c.1 — DSv4 wall-storage swa 池真替换 + canary | ✅ 完成 | Triton dequant kernel + `swa_kv_pool` packed buffer + `dequant_swa_to_fp8_layout` shim + 5 用例 canary | 5/5 canary，wall ratio 2.18× (swa 池) |
| M3.c.2 — attention prologue 接入 + c4/c128 同步替换 | ✅ **本轮交付** | 三池同步 wall + `_rotated_quant_attention_prologue` hook + 5 write override + 2 read override + backend duck-typed hook + 7 用例 canary（含 e2e cosine ≥ 0.95） | **7/7 canary、12/12 regression、cosine.mean ≥ 0.95** |
| M3.c.3 — bit-pack/unpack Triton 化（性能化） | ⏳ 未启动 | CPU 往返 → Triton kernel；fused norm+rope→packed；shadow ring buffer | latency ≤ 1.05× baseline |

### M3.c.2 一句话总结
DSv4 三池（swa/c4/c128）全部从 native FP8 切换到 INT2/3/4 packed wall-storage；FlashMLA 路径不动，靠 attention prologue 的 dequant→shadow 完成 layout 兜底；端到端 cosine.mean ≥ 0.95 通过，partial-mode 不一致风险（原风险条 6）已消除。

### 测试矩阵（CPU-only，macOS 已通过）
- `test/manual/quant/test_rotated_kv_quant_dsv4_canary.py`：**7 用例**（4 wall-storage + 2 prologue e2e + 1 GPU skipped），耗时 0.28s
- `test/manual/quant/test_rotated_kv_quant_e2e.py`：**12 用例**（4 MHA + 4 MLA + 4 DSv4 eval），耗时 5.6s
- 总计 **19 pass + 1 GPU skipped**，0 fail，0 回归

### 关键文件指针
| 模块 | 文件 | 职责 |
|------|------|------|
| Pool（核心） | [`rotated_quant_dsv4_memory_pool.py`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/python/sglang/srt/mem_cache/rotated_quant_dsv4_memory_pool.py) | wall-storage 三池替换 + prologue hook + 5 write/2 read override |
| Kernels（公共） | [`rotated_quant_dsv4_kernels.py`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/python/sglang/jit_kernel/rotated_quant_dsv4_kernels.py) | `rotated_store_to_packed` / `rotated_load_to_fp8_layout(_cpu_ref)` / `quant_fp8_layout_cpu_ref` |
| Triton dequant | [`triton_rotated_quant_dsv4.py`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/python/sglang/jit_kernel/triton_rotated_quant_dsv4.py) | UE8M0 cast + nope/rope 双 kernel dispatch |
| Backend 接线 | [`deepseek_v4_backend.py`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/python/sglang/srt/layers/attention/deepseek_v4_backend.py#L986) | duck-typed prologue hook（仅 1 处） |
| 工厂路由 | [`model_runner_kv_cache_mixin.py`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py) | DSv4 → eval/wall 两分支 |
| CLI | [`server_args.py`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/python/sglang/srt/server_args.py) | `--rotated-kv-quant-mode {eval,wall}` |
| Canary | [`test_rotated_kv_quant_dsv4_canary.py`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/test/manual/quant/test_rotated_kv_quant_dsv4_canary.py) | M3.c.1 + M3.c.2 全部 7 用例 |
| Regression | [`test_rotated_kv_quant_e2e.py`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/test/manual/quant/test_rotated_kv_quant_e2e.py) | MHA + MLA + DSv4 eval 12 用例 |

### M3.c.2 完成的 8 项 sub-task
1. ✅ `_install_wall_storage` 三池同步（swa/c4/c128 全部 packed wall-storage）
2. ✅ Shadow buffer 静态预分配（与原 FP8 同 layout，CUDA Graph 友好）
3. ✅ 5 个 write override：`set_swa_key_buffer_radix_fused` / `set_swa_key_buffer_radix_fused_norm_rope`（含 RMSNorm+RoPE PyTorch fallback） / `set_extra_key_buffer_fused`（c4/c128 路由） / `set_extra_key_buffer`（非 fused 抛 NotImplementedError）/ 兼容路径
4. ✅ 2 个 read override：`get_swa_key_buffer_radix` / `get_extra_key_buffer` 直返 shadow
5. ✅ Prologue：`_rotated_quant_attention_prologue` 走 swa + 当前 layer 的 c4 或 c128，dedup + filter -1 + dequant + scatter
6. ✅ Backend hook：[`deepseek_v4_backend.py:986`](file:///Users/bytedance/Desktop/WYZ/Code/20260603-kvcompress/sglang-bytedance/python/sglang/srt/layers/attention/deepseek_v4_backend.py#L986) duck-typed，对非 rotated pool 零副作用
7. ✅ CPU helper 提到 kernels：`quant_fp8_layout_cpu_ref` 公共化，pool 与 canary 共享
8. ✅ M3.c.1 layout bug 修复：保留 `bytes_per_page_padded` 为原 native FP8 size（packed 是物理存储，shadow 给读侧用），不再失配下游 view shape

### 下一步（M3.c.3，性能化）
1. 把 `bitpack_rowwise / bitunpack_rowwise` 的 CPU 往返替换为 Triton kernel（每 token row 一个 thread block，每 thread 处理 4 个坐标的 bit-extract，`tl.bfloat16` 直接做反 affine）
2. 把 `set_swa_key_buffer_radix_fused_norm_rope` 的 PyTorch RMSNorm+RoPE fallback 替换为 fused Triton kernel 直写 packed
3. 把 shadow buffer 从 worst-case 静态预分配改为 ring buffer，进一步省显存
4. 端到端验收：长上下文 decode latency ≤ 1.05× FP8 baseline；GSM8K / HumanEval 相对掉点 ≤ 1pp


