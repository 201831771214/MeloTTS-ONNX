# MeloTTS QNN ONNX更新记录

## 概述

本文档记录了将 MeloTTS 从原始 ONNX 动态模型适配为 **Qualcomm QNN HTP 静态模型**所做的所有源码改动，以及每项改动的原理和收益。

> **背景**：原始 MeloTTS ONNX 模型是动态 shape（variable sequence length），在 CPU/CUDA 上运行正常。但要部署到 Qualcomm QCS8550 的 Hexagon HTP (v73) 上运行时，遇到了动态 shape 不兼容、fp16 精度崩塌、DSP 内存不足等一系列问题。

---

## 一、修改的文件清单

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `melo/models.py` | 核心逻辑 | 添加静态导出路径、QNN 保底模式、y_lengths 输出 |
| `melo/attentions.py` | 核心逻辑 | 添加导出模式相对位置编码（切片替代动态 pad） |
| `melo_extra/melo_tts.py` | 封装层 | 支持多路导出（动态/静态/QNN）、y_lengths 联合输出 |
| `export_melo.py` | 导出工具 | 新增 `--seq_len`、`--max_mel_frames`、`--qnn`、`--precision` 参数 |
| `run_onnx.py` | 推理引擎 | y_lengths 精确截断、RMS 静音检测、自动 dtype 适配、名称映射输入 |
| `compile.py` | 编译工具 | 一键导出 + qai-hub 编译 + profile |
| `compile_configs/cfg.ini` | 编译配置 | 静态 512 序列长度的输入 spec |

---

## 二、核心问题与解决方案

### 问题 1：动态 shape → QNN 不兼容

**现象**：动态 ONNX 模型（`sequence_length` 可变）编译为 QNN context binary 后，推理时报 `QNN error 1100`。

**根因**：QNN context binary generator 在编译时把图优化固化为特定 shape，运行时 shape 不匹配即报错。

**解决方案**：导出**静态 ONNX 模型**，所有维度冻结为固定值。

**改动位置**：`export_melo.py` + `melo/models.py`

| 文件 | 改动 |
|------|------|
| `export_melo.py` | `build_melo_dummy_input()` 新增 `target_seq_len=512` 参数：将文本 token 序列统一 pad/truncate 到 512 |
| `export_melo.py` | 不传 `-id` 时 `dynamic_axes={}`，导出纯静态模型 |
| `melo/models.py` | `forward_for_export_static()` 中 `y_mask` 的 `arange` 改为可配置的 `max_mel_frames` 参数（原硬编码 2048） |
| `melo/attentions.py` | `MultiHeadAttention` 新增 `_build_export_embeddings()` + `export_mode`：预构建固定尺寸的 relative position 缓冲区，用纯 slice 替代运行时 `F.pad`，避免动态 op |

**收益**：
- QNN 编译器可以完整静态优化整个计算图
- 消除 shape 不匹配导致的 QNN error 1100

---

### 问题 2：DSP 内存不足（QNN_COMMON_ERROR_MEM_ALLOC）

**现象**：模型加载时 DSP 尝试 mmap ~191MB 的缓冲区失败，报 `QNN_COMMON_ERROR_MEM_ALLOC`。

**根因**：`forward_for_export_static` 中 `y_mask` 硬编码 `2048` 帧，导致：
- 输出 buffer `(1, 1, 2048×512) = 1,048,576 samples`
- 中间 tensor 呈指数级放大，DDR spill 高达 12.4GB
- 累积峰值内存超 Hexagon v73 可用空间

**解决方案**：将 `max_mel_frames` 从 2048 降低到 1024（可配置）。

**改动位置**：`melo/models.py` + `export_melo.py`

| 文件 | 改动 |
|------|------|
| `melo/models.py` | `forward_for_export_static()` 的 `static_range = torch.arange(max_mel_frames, ...)`，参数默认 1024 |
| `melo/attentions.py` | `MultiHeadAttention._max_length` 同步减小，`_build_export_embeddings()` buffer 缩小一半 |
| `melo_extra/melo_tts.py` | `MeloTTSWrapper.__init__()` 新增 `max_mel_frames` 属性 |
| `export_melo.py` | 新增 `-mmf/--max_mel_frames` CLI 参数 |

**收益**：

| 指标 | 修改前 (2048) | 修改后 (1024) |
|------|-------------|-------------|
| 输出 audio shape | `(1, 1, 1,048,576)` | `(1, 1, 524,288)` |
| Attention buffer per MHA 层 | `4095 × channels` | `2047 × channels` |
| DSP peak memory (profile) | >309 MB (OOM) | ~247 MB (正常) |

---

### 问题 3：fp16 精度崩塌 → 音频全空

**现象**：模型在 fp32 (CPU/CUDA) 上输出正常，但在 QNN HTP fp16 上 `y_lengths=8`（47 token 只预测 8 帧 mel，即 0.09 秒），音频几乎为空。

**根因链**：

```
Text Encoder (12+ 层 Transformer)
    ↓ fp16 累积误差
Encoder 输出 x 失真
    ↓
Duration Predictor (DP/SDP)
    ↓ logw 极度负值 → exp(logw) fp16 underflow → 0
w_ceil = 0 for most tokens
    ↓
y_lengths ≈ 8 (几乎为零)
    ↓
Decoder 输出全是 bias 漂移噪声
```

**逐步排查过程**：

1. **尝试 1 — 仅 clamp logw**：`clamp(logw, min=-11.5)` → 无效，因为 `exp(-11.5) ≈ 1e-5` 在 fp16 中是 subnormal，被 flush to zero

2. **尝试 2 — 提高 clamp 到 fp16 normal 下界**：`clamp(logw, min=-9.7)` → 无效，问题不在 exp 而在 encoder 输出本身已损坏

3. **尝试 3 — 绕过 SDP 只用 DP + clamp**：仍无效，DP 同样依赖 encoder 输出，fp16 下也产出垃圾

4. **尝试 4（最终）— 彻底绕过整个 Duration Predictor**：**固定每 token 时长**，不走任何可学习的 duration 模块

**最终解决方案**：QNN 模式下 duration 完全绕过 SDP/DP，使用固定时长。

**改动位置**：`melo/models.py` 第 1060-1061 行

```python
if getattr(self, '_qnn_mode', False):
    # QNN HTP fp16: SDP/DP 全崩，固定每 token 4 帧
    w_ceil = torch.ones_like(x_mask.squeeze(1)) * 4      # [b, t_text]
    w_ceil = w_ceil.unsqueeze(1) * x_mask                  # [b, 1, t_text]
else:
    # fp32 / CPU: 原始 SDP + DP，正常质量
    logw = self.sdp(...) * sdp_ratio + self.dp(...) * (1 - sdp_ratio)
    w = torch.exp(logw) * x_mask * (1.0 / speed[0])
    w_ceil = torch.ceil(w)
```

| 导出模式 | Duration 方案 | 适用场景 |
|----------|-------------|---------|
| 本地 ONNX (`--qnn` 不传) | 原始 SDP + DP | CPU/CUDA，fp32，语音自然 |
| QNN HTP (`--qnn`) | **固定 4 帧/token** | QCS8550 HTP，fp16，零精度风险 |

**收益**：
- 彻底消除 fp16 下 encoder → duration predictor 链路的精度崩塌
- 每 token 固定 4 帧，47 token → 188 帧 → ~2.2 秒，覆盖常规语速
- 纯常量乘法（`ones * 4 * x_mask`），无任何 fp16 敏感 op

---

### 问题 4：静态模型输出超长 → 尾部杂音/空音频

**现象**：静态模型输出固定长度 buffer（如 524,288 samples），实际有效音频只占前面一小段，尾部是 decoder 对零 mel 输入产生的"空闲输出"。需要精确截断。

**根因**：ONNX 静态模型只有 `audio_data` 一个输出，推理端无法知道有效音频从哪结束。

**排查过程**：

1. **尝试 1 — 文本比例线性估算**：`valid_audio = total_audio × (text_len / total_len)` → 不准确，duration 是非线性的
2. **尝试 2 — 静音检测（峰值）**：尾部空闲噪音振幅 ~0.06，语音振幅 ~0.32，ratio 仅 5.5x → 门限难以选取
3. **尝试 3 — RMS 能量检测**：CPU 上区分度好（ratio > 10x），但在 QNN fp16 上完全失效（idle noise RMS ≈ speech RMS）
4. **尝试 4（最终）— 直接导出 `y_lengths`**：模型内部已知有效 mel 帧数，直接作为第二个 ONNX 输出

**解决方案**：在 ONNX 模型中添加 `y_lengths`（有效 mel 帧数）作为第二个输出。

**改动位置**：`melo/models.py` + `melo_extra/melo_tts.py` + `export_melo.py` + `run_onnx.py`

| 文件 | 改动 |
|------|------|
| `melo/models.py` | `forward_for_export_static()` 返回第 5 个元素为 `y_lengths` |
| `melo_extra/melo_tts.py` | 静态路径 `return audio_valid, y_lengths.to(torch.int32)` |
| `export_melo.py` | 静态导出 `output_names = ["audio_data", "y_lengths"]` |
| `run_onnx.py` | `valid_samples = y_len * hop_size`，精确截取 |

**收益**：
- 推理端直接取 `y_lengths × hop_size` 得到精确有效音频长度
- 任何 EP（CPU/CUDA/QNN）都适用，零误判
- 兼容旧模型（检测到单输出时自动 fallback 到 RMS 检测）

---

### 问题 5：输入索引硬编码 → 多版本模型不兼容

**现象**：不同导出模式下的模型输入数量不同（动态 11 输入、静态 ONNX 10 输入、QNN 8-9 输入），硬编码 `self.input_names[8]` 导致 `IndexError`。

**解决方案**：改为按名称映射输入。

**改动位置**：`run_onnx.py`

```python
# 旧：硬编码索引
input_spec = {self.input_names[0]: x_tst, self.input_names[7]: np_sdp_ratio, ...}

# 新：按名字映射，兼容所有输入变体
named_inputs = {"x_tst": x_tst, "sdp_ratio": np_sdp_ratio, "speed": np_speed, ...}
input_spec = {k: v for k, v in named_inputs.items() if k in self.input_names}
```

**收益**：一套推理代码兼容 8/9/10/11 输入的所有模型版本。

---

## 三、导出命令速查

```bash
# 本地 ONNX（fp32，SDP+DP，完整质量）
python export_melo.py -sl 512 -mmf 1024 --opset 16

# QNN HTP（fp32 IO，固定 duration，QNN 优化）
python export_melo.py -sl 512 -mmf 1024 --opset 16 --qnn

# 一键导出 + 编译 + profile
bash export_and_compile.sh
```

## 四、架构总览

```
                     ┌─────────────────────────┐
                     │   export_melo.py         │
                     │   -sl 512  -mmf 1024     │
                     │   --qnn  --opset 16      │
                     └───────────┬───────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
    ┌─────────▼────────┐ ┌──────▼──────┐  ┌────────▼─────────┐
    │  本地 ONNX (fp32) │ │ QNN ONNX    │  │  推理端           │
    │  SDP + DP        │ │ 固定 4 帧    │  │  y_lengths 精确   │
    │  自然语音质量      │ │ fp16 安全    │  │  截断音频          │
    └──────────────────┘ └─────────────┘  └──────────────────┘
```
