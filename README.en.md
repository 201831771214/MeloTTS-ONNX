```markdown
---
license: Apache License 2.0
language:
  - zh
  - en
tasks:
  - text-to-speech
frameworks: other
base_model:
  - myshell-ai/MeloTTS-Chinese
base_model_relation: repackage
metrics:
  - bertscore
tags:
  - text-to-speech
  - multilingual
  - zh
  - en
  - onnx
  - QNN EP
  - Qualcomm
  - QCS8550
---

![MeloTTS](./rep_sources/melotts.png)

## MeloTTS-ONNX Project Details

![MIT License](./rep_sources/License-MIT-yellow.png)
![Numpy License](./rep_sources/NumPy-1.26.4-blue.png)
![Onnx Runtime License](./rep_sources/OnnxRuntime-1.23.2-blue.png)
![Onnx License](./rep_sources/ONNX-1.20.1-green.png)
![Python License](./rep_sources/Python-3.10+-blue.png)

### 1. Project Overview

**MeloTTS-ONNX** is an ONNX inference version of [MeloTTS](https://github.com/MoonshotAI/MeloTTS), optimized specifically for **real-time CPU inference**. The project supports:

- ✅ Chinese-English mixed TTS
- ✅ Multiple languages: Chinese, English, Japanese, Korean, Spanish, French, etc.
- ✅ ONNX Runtime inference with fast speed

---

#### Repository Addresses

- Gitee: https://gitee.com/jackroing/melo-tts-onnx.git
- GitHub: https://github.com/201831771214/MeloTTS-ONNX.git
- ModelScope Model Repository: https://www.modelscope.cn/models/KeanuX/MeloTTS-ZH-MIXED-EN-ONNX

```shell
# Clone from Gitee
git clone https://gitee.com/jackroing/melo-tts-onnx.git

# Clone from GitHub
git clone https://github.com/201831771214/MeloTTS-ONNX.git

# Download models
modelscope download --model KeanuX/MeloTTS-ZH-MIXED-EN-ONNX --local_dir ./
```

### 2. Project Architecture

```
melo-tts-onnx/
├── melo/                      # Original PyTorch training code
│   ├── api.py                 # TTS API interface
│   ├── models.py              # Model definition (SynthesizerTrn)
│   ├── modules.py             # Model modules
│   ├── text/                  # Text processing (tokenization, phoneme conversion)
│   │   ├── chinese.py         # Chinese text processing
│   │   ├── english.py         # English text processing
│   │   └── ...
│   ├── train.py / train.sh    # Training scripts
│   └── ...
│
├── melo_extra/                # Text processing modules used during inference
│   ├── melo_tts.py            # ONNX export wrapper
│   └── inference/
│       ├── text/              # Text processing for inference (corresponds to melo/text)
│       ├── commons.py         # Common utilities
│       └── utils.py           # Parameter configuration
│
├── models/melotts/            # ONNX model directory
│   ├── melotts_14.onnx        # Exported ONNX model
│   ├── config.json            # Model configuration
│   └── bert-base-multilingual-uncased/  # BERT model
│
├── run_onnx.py                # ⭐ Core inference script
├── export_melo.py             # ONNX export script
├── export_model_info.py       # Model information export utility
└── README.md
```

---

### 3. Core Script Usage

#### 3.1 Inference Script: `run_onnx.py`

This is the **most frequently used script** for converting text to speech:

```python
from run_onnx import MeloTTS

# Initialize the model
model_path = "./models/melotts/"
melo_tts = MeloTTS(model_path, device="cpu")

# Generate audio
audio, sr = melo_tts.generate_audio(
    text="你好，我是中英混合模型。Hello I am a mixed language model.",
    language="ZH_MIX_EN",      # Language: ZH_MIX_EN, EN, JP, KR, etc.
    sdp_ratio=0.2,            # SDP ratio
    noise_scale=0.667,        # Noise scale
    noise_scale_w=0.8,        # Noise weight
    speed=1.0                 # Speaking speed
)
```

**Main Parameter Descriptions:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `text` | Input text | Required |
| `language` | Language code | `"ZH_MIX_EN"` |
| `sdp_ratio` | SDP ratio (0-1) | 0.2 |
| `noise_scale` | Noise scale | 0.667 |
| `noise_scale_w` | Noise weight | 0.8 |
| `speed` | Speaking speed | 1.0 |

**Supported Language Codes:**

- `ZH_MIX_EN` - Chinese (supports Chinese-English mixing)
- `EN` - English
- etc.

---

#### 3.2 ONNX Export Script: `export_melo.py`

Used to export a PyTorch model to ONNX format:

```bash
python export_melo.py \
    -m /path/to/ckpt \
    -c /path/to/config.json \
    -o /path/to/save_dir \
    --opset 14
    ...
```

**Main Parameters:**

- `--ckpt_path` - Model checkpoint path
- `--cfg_path` - Configuration file path
- `--output_path` - Output ONNX file path
- `--opset` - ONNX opset version (default 14)

---

#### 3.3 Model Information Export: `export_model_info.py`

Used to export detailed information about an ONNX model (input/output shapes, parameter count, etc.):

```bash
python export_model_info.py -m ./models/melotts/melotts_14.onnx -o ./infos/melotts_14.info
```

Sample output:

```text
============================================================
ONNX Model Basic Information
============================================================
Model file path: ./models/melotts_onnx/melotts_14_dynamic.onnx
ONNX version: 7
Producer: pytorch 2.8.0
Model version: 0
Description: 

============================================================
Model Input Information (11 inputs in total)
============================================================
Input 1: x_tst
  Data type: int32
  Shape: [0, 0]

Input 2: x_tst_lengths
  Data type: int32
  Shape: [0]

Input 3: speakers
  Data type: int32
  Shape: [0]

Input 4: tones
  Data type: int32
  Shape: [0, 0]

Input 5: lang_ids
  Data type: int32
  Shape: [0, 0]

Input 6: bert
  Data type: float32
  Shape: [0, 1024, 0]

Input 7: ja_bert
  Data type: float32
  Shape: [0, 768, 0]

Input 8: sdp_ratio
  Data type: float32
  Shape: [0]

Input 9: noise_scale
  Data type: float32
  Shape: [0]

Input 10: noise_scale_w
  Data type: float32
  Shape: [0]

Input 11: speed
  Data type: float32
  Shape: [0]

============================================================
Model Output Information (1 output)
============================================================
Output 1: audio_data
  Data type: float32
  Shape: [1, 0]


============================================================
ONNX Model Basic Information
============================================================
Model file path: ./models/melotts_onnx/melotts_14_static.onnx
ONNX version: 7
Producer: pytorch 2.8.0
Model version: 0
Description: 

============================================================
Model Input Information (10 inputs in total)
============================================================
Input 1: x_tst
  Data type: int32
  Shape: [1, 239]

Input 2: x_tst_lengths
  Data type: int32
  Shape: [1]

Input 3: speakers
  Data type: int32
  Shape: [1]

Input 4: tones
  Data type: int32
  Shape: [1, 239]

Input 5: lang_ids
  Data type: int32
  Shape: [1, 239]

Input 6: bert
  Data type: float32
  Shape: [1, 1024, 239]

Input 7: ja_bert
  Data type: float32
  Shape: [1, 768, 239]

Input 8: sdp_ratio
  Data type: float32
  Shape: [1]

Input 9: noise_scale_w
  Data type: float32
  Shape: [1]

Input 10: speed
  Data type: float32
  Shape: [1]

============================================================
Model Output Information (1 output)
============================================================
Output 1: audio_data
  Data type: float32
  Shape: [1, 0]

============================================================
ONNX Model Basic Information
============================================================
Model file path: ./melo_tts_qnn/model.onnx
ONNX version: 11
Producer: Qualcomm AI Hub Workbench aihub-2026.02.26.0
Model version: 0
Description: 

============================================================
Model Input Information (10 inputs in total)
============================================================
Input 1: x_tst
  Data type: int32
  Shape: [1, 239]

Input 2: x_tst_lengths
  Data type: int32
  Shape: [1]

Input 3: speakers
  Data type: int32
  Shape: [1]

Input 4: tones
  Data type: int32
  Shape: [1, 239]

Input 5: lang_ids
  Data type: int32
  Shape: [1, 239]

Input 6: bert
  Data type: float32
  Shape: [1, 1024, 239]

Input 7: ja_bert
  Data type: float32
  Shape: [1, 768, 239]

Input 8: sdp_ratio
  Data type: float32
  Shape: [1]

Input 9: noise_scale_w
  Data type: float32
  Shape: [1]

Input 10: speed
  Data type: float32
  Shape: [1]

============================================================
Model Output Information (1 output)
============================================================
Output 1: audio_data
  Data type: float32
  Shape: [1, 429568]

============================================================
Model Weight Information (0 parameters in total)
============================================================
Total parameters: 0

============================================================
Model Operator Information (1 operator in total)
============================================================
Operator Type       | Count
------------------------------
EPContext       | 1

Detailed Layer Information:
------------------------------------------------------------
Layer 1: QNNContext (EPContext)
  Inputs: x_tst, x_tst_lengths, speakers, tones, lang_ids, bert, ja_bert, sdp_ratio, noise_scale_w, speed
  Outputs: audio_data
  Attributes:
    - embed_mode: 
    - ep_cache_context: ./model.bin
    - source: QNN


============================================================
Additional Information
============================================================
IR version: -13
Graph name: qnn-onnx-model
```

#### Update Information
Updated MeloTTS ONNX Static Model:
- Increased the sequence length of the melotts_14_static.onnx model to 512, reducing the intermediate parameter count by half.

#### Precompiled QNN ONNX Profile Info

![info0](./rep_sources/0.png)
![info1](./rep_sources/1.png)

#### Dear developers, when running the MeloTTS QNN ONNX EP, please disable the following inputs:

```python
# Inside generate_audio_chunked or generate_audio

named_inputs = {
    "x_tst": x_tst_part,
    "x_tst_lengths": x_tst_lengths_part,
    "speakers": speaker_id,
    "tones": tone_part,
    "lang_ids": lang_ids_part,
    "bert": bert_part,
    "ja_bert": ja_bert_part,
    "sdp_ratio": np_sdp_ratio,
    # "noise_scale": np_noise_scale,
    # "noise_scale_w": np_noise_scale_w,
    # "speed": np_speed,
}
```

- The original MeloTTS source code uses a DP + SDP scheme to ensure audio quality by dynamically computing frames. However, on QNN HTP, both modules produce incorrect output under fp16 precision (encoder → duration predictor chain collapse, `exp()` underflow). We bypass the entire duration predictor and use a fixed 4 frames/token instead, completely avoiding fp16 precision issues. For a detailed explanation of all QNN adaptations, see [UPDATE.md](./UPDATE.md).

---

### 4. Working Principle

```
Text Input
   ↓
┌─────────────────────────────────────────┐
│  Text Preprocessing (clean_text)        │
│  - Tokenization                         │
│  - Conversion to phoneme                │
│  - Tone extraction                      │
│  - BERT feature extraction              │
└─────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────┐
│  ONNX Model Inference                   │
│  - Glow-TTS (text → mel spectrogram)    │
│  - HiFi-GAN (mel spectrogram → audio)   │
└─────────────────────────────────────────┘
   ↓
Audio Output (44.1kHz)
```

**ONNX Dynamic Model Inputs (11):**

1. `x_tst` - Text token IDs
2. `x_tst_lengths` - Text length
3. `speakers` - Speaker ID
4. `tones` - Tone IDs
5. `lang_ids` - Language IDs
6. `bert` - BERT features (1024 dimensions)
7. `ja_bert` - Japanese BERT features (768 dimensions)
8. `sdp_ratio` - SDP ratio
9. `noise_scale` - Noise scale
10. `noise_scale_w` - Noise weight
11. `speed` - Speaking speed

**ONNX Model Output (1):**

- `audio_data` - Generated audio data

---

**ONNX Static Model Inputs (10):**

1. `x_tst` - Text token IDs
2. `x_tst_lengths` - Text length
3. `speakers` - Speaker ID
4. `tones` - Tone IDs
5. `lang_ids` - Language IDs
6. `bert` - BERT features (1024 dimensions)
7. `ja_bert` - Japanese BERT features (768 dimensions)
8. `sdp_ratio` - SDP ratio
9. `noise_scale_w` - Noise weight
10. `speed` - Speaking speed

**ONNX Model Output (1):**

- `audio_data` - Generated audio data

---

#### Note: The code example is only suitable for deploying the Dynamic model. To deploy the Static model, please adjust the `generate_audio` method in the code accordingly. Set the chunk size to 239; if the input exceeds this size, split it into multiple parts for sequential processing, and finally merge all audio segments.

### 5. Quick Start Example

```python
"""MeloTTS ONNX Runtime Inference

Static ONNX Model Inference:
    ============================================================
    Model Input Information (10 inputs in total)
    ============================================================
    Input 1: x_tst
     Data type: int32
     Shape: [1, 239]

    Input 2: x_tst_lengths
     Data type: int32
     Shape: [1]

    Input 3: speakers
     Data type: int32
     Shape: [1]

    Input 4: tones
     Data type: int32
     Shape: [1, 239]

    Input 5: lang_ids
     Data type: int32
     Shape: [1, 239]

    Input 6: bert
     Data type: float32
     Shape: [1, 1024, 239]

    Input 7: ja_bert
     Data type: float32
     Shape: [1, 768, 239]

    Input 8: sdp_ratio
     Data type: float32
     Shape: [1]

    Input 9: noise_scale_w
     Data type: float32
     Shape: [1]

    Input 10: speed
     Data type: float32
     Shape: [1]

    ============================================================
    Model Output Information (1 output)
    ============================================================
    Output 1: audio_data
     Data type: float32
     Shape: [1, 0]

Dynamic ONNX Model Inference:
    ============================================================
    ONNX Model Basic Information
    ============================================================
    Model file path: ./models/melotts_onnx/melotts_14_dynamic.onnx
    ONNX version: 7
    Producer: pytorch 2.8.0
    Model version: 0
    Description: 

    ============================================================
    Model Input Information (11 inputs in total)
    ============================================================
    Input 1: x_tst
     Data type: int32
     Shape: [0, 0]

    Input 2: x_tst_lengths
     Data type: int32
     Shape: [0]

    Input 3: speakers
     Data type: int32
     Shape: [0]

    Input 4: tones
     Data type: int32
     Shape: [0, 0]

    Input 5: lang_ids
     Data type: int32
     Shape: [0, 0]

    Input 6: bert
     Data type: float32
     Shape: [0, 1024, 0]

    Input 7: ja_bert
     Data type: float32
     Shape: [0, 768, 0]

    Input 8: sdp_ratio
     Data type: float32
     Shape: [0]

    Input 9: noise_scale
     Data type: float32
     Shape: [0]

    Input 10: noise_scale_w
     Data type: float32
     Shape: [0]

    Input 11: speed
     Data type: float32
     Shape: [0]

    ============================================================
    Model Output Information (1 output)
    ============================================================
    Output 1: audio_data
     Data type: float32
     Shape: [1, 0]

QNN ONNX Model Inference:
    ============================================================
    ONNX Model Basic Information
    ============================================================
    Model file path: ./models/melotts_qnn/melotts_14/model.onnx
    ONNX version: 11
    Producer: Qualcomm AI Hub Workbench aihub-2026.02.26.0
    Model version: 0
    Description: 

    ============================================================
    Model Input Information (10 inputs in total)
    ============================================================
    Input 1: x_tst
     Data type: int32
     Shape: [1, 239]

    Input 2: x_tst_lengths
     Data type: int32
     Shape: [1]

    Input 3: speakers
     Data type: int32
     Shape: [1]

    Input 4: tones
     Data type: int32
     Shape: [1, 239]

    Input 5: lang_ids
     Data type: int32
     Shape: [1, 239]

    Input 6: bert
     Data type: float32
     Shape: [1, 1024, 239]

    Input 7: ja_bert
     Data type: float32
     Shape: [1, 768, 239]

    Input 8: sdp_ratio
     Data type: float32
     Shape: [1]

    Input 9: noise_scale_w
     Data type: float32
     Shape: [1]

    Input 10: speed
     Data type: float32
     Shape: [1]

    ============================================================
    Model Output Information (1 output)
    ============================================================
    Output 1: audio_data
     Data type: float32
     Shape: [1, 429568]

    ============================================================
    Model Weight Information (0 parameters in total)
    ============================================================
    Total parameters: 0

    ============================================================
    Model Operator Information (1 operator in total)
    ============================================================
    Operator Type       | Count
    ------------------------------
    EPContext       | 1

    Detailed Layer Information:
    ------------------------------------------------------------
    Layer 1: QNNContext (EPContext)
     Inputs: x_tst, x_tst_lengths, speakers, tones, lang_ids, bert, ja_bert, sdp_ratio, noise_scale_w, speed
     Outputs: audio_data
     Attributes:
        - embed_mode: 
        - ep_cache_context: ./model.bin
        - source: QNN


    ============================================================
    Additional Information
    ============================================================
    IR version: -13
    Graph name: qnn-onnx-model
"""


import onnxruntime as ort
import numpy as np
import os
import sys
import soundfile as sf
from typing import Tuple
from melo_extra.inference.utils import HParams, get_hparams_from_file
from melo_extra.inference.text.cleaner import clean_text
from melo_extra.inference.text import cleaned_text_to_sequence, get_bert, get_zh_mix_en_bert
from melo_extra.inference import commons
import argparse
import logging

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler("./logs/run_melo_onnx.log", mode="w", encoding="utf-8")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

class MeloTTS:
    def __init__(self, model_root:str, device:str="cpu", provider_options:list[dict]=None, is_dynamic:bool=True) -> None:
        self.hop_size = 512  # Generator upsampling factor
        self.model_list = {}
        
        for f in os.listdir(model_root):
            if f.endswith(".onnx"):
                f_name = f[0:]
                logger.info(f"find model: {f_name}")
                self.model_list[f_name] = os.path.join(model_root, f)
        
        if is_dynamic:
            for m_k, m_v in self.model_list.items():
                if "dynamic" in m_k.lower():
                    self.model_name = m_v
                    break
                else:
                    self.model_name = None
                    continue
        else:
            if device == "qnn":
                self.model_name = os.path.join(model_root, "precompiled_qnn_onnx", "model.onnx")
            else:
                for m_k, m_v in self.model_list.items():
                    if "static" in m_k.lower():
                        self.model_name = m_v
                        break
                    else:
                        self.model_name = None
                        continue
        
        if self.model_name == None:
            logger.error(f"model name not found, please check model root: {model_root}")
            sys.exit(1)
        
        logger.info(f"Use model: {self.model_name}")
        self.model_path = os.path.join(self.model_name)
        self.cfg_path = os.path.join(model_root, "config.json")
        self.bert_model_path = os.path.join(model_root, "bert-base-multilingual-uncased")
        
        self.cfg = get_hparams_from_file(self.cfg_path)
        self.sample_rate = self.cfg.data.sampling_rate
        
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            self.providers = ["CUDAExecutionProvider"]
        elif device == "cpu":
            self.providers = ["CPUExecutionProvider"]
        elif device == "qnn" and "QNNExecutionProvider" in ort.get_available_providers() and provider_options != None:
            self.providers = ["QNNExecutionProvider"]
        else:
            logger.info(f"device {device} not supported, use cpu instead")
            self.providers = ["CPUExecutionProvider"]
            
        self.provider_options = provider_options if self.providers == ["QNNExecutionProvider"] else None
        
        self.session = ort.InferenceSession(self.model_path, providers=self.providers, provider_options=provider_options)
        logger.info(f"Using EP: {self.session.get_providers()}, model inputs: {len(self.session.get_inputs())}")
        
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Automatically detect floating-point dtype: matches the model's expected precision (fp32 or fp16)
        onnx_dtype_to_np = {1: np.float32, 10: np.float16}  # TensorProto.FLOAT=1, FLOAT16=10
        self.float_dtype = np.float32  # default
        for inp in self.session.get_inputs():
            if inp.name == "bert":  # bert feature must be a floating-point input
                self.float_dtype = onnx_dtype_to_np.get(inp.type, np.float32)
                break
        
        logger.info(f"model input names: {self.input_names}")
        logger.info(f"model output names: {self.output_names}")
        logger.info(f"auto-detected float dtype: {self.float_dtype}")
    
    @staticmethod
    def _trim_trailing_silence(audio: np.ndarray) -> np.ndarray:
        """Trim trailing silence.

        Uses RMS energy instead of peak – idle noise may have occasional spikes (~0.06), but RMS
        energy is much lower than real speech, providing better discrimination than peak.
        """
        if len(audio) < 2048:
            return audio

        # Diagnostics: output overall statistics
        logger.info(f"_trim: audio_len={len(audio)}, min={audio.min():.6f}, "
                    f"max={audio.max():.6f}, mean={np.abs(audio).mean():.6f}, "
                    f"nonzero_ratio={np.count_nonzero(audio)/len(audio):.4f}")

        win = 512
        # RMS energy per frame (more discriminative than peak)
        frame_rms = np.array([
            np.sqrt(np.mean(audio[i:i+win].astype(np.float64) ** 2))
            for i in range(0, len(audio) - win, win)
        ])

        # Last 10% frames = idle noise RMS
        tail_n = max(1, len(frame_rms) // 10)
        idle_rms = np.max(frame_rms[-tail_n:])

        # Threshold = max(idle × 3, absolute lower bound 5e-4)
        threshold = max(idle_rms * 3, 5e-4)

        logger.info(f"_trim: total_frames={len(frame_rms)}, tail={tail_n}, "
                    f"idle_rms={idle_rms:.6f}, threshold={threshold:.6f}, "
                    f"max_rms={frame_rms.max():.6f}")

        above = np.where(frame_rms > threshold)[0]
        if len(above) == 0:
            logger.warning("_trim: no frame above threshold, returning empty")
            return audio[:0]

        cut_frame = above[-1] + 1
        cut_sample = min(cut_frame * win, len(audio))
        logger.info(f"_trim: cut at frame {above[-1]}/{len(frame_rms)}, "
                    f"sample {cut_sample}/{len(audio)}")
        return audio[:cut_sample]

    def __preprocess(self, text:str, language:str):
        """Preprocess text input
        Args:
            text (str): Input text to be preprocessed.
            language (str): Language of the input text.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Preprocessed tensors ready for inference.
        """
        norm_text, phone, tone, word2ph = clean_text(text, language)
        symbol_to_id = {s: i for i, s in enumerate(self.cfg.symbols)}
        phone, tone, language = cleaned_text_to_sequence(phone, tone, language, symbol_to_id)
        
        if self.cfg.data.add_blank:
            phone = commons.intersperse(phone, 0)
            tone = commons.intersperse(tone, 0)
            language = commons.intersperse(language, 0)
            for i in range(len(word2ph)):
                word2ph[i] = word2ph[i] * 2
            word2ph[0] += 1
        
        if getattr(self.cfg.data, "disable_bert", True):
            bert = np.zeros((1024, len(phone)), dtype=self.float_dtype)
            ja_bert = np.zeros((768, len(phone)), dtype=self.float_dtype)
        else:
            bert = get_zh_mix_en_bert(self.bert_model_path, text, word2ph, "cpu")
            del word2ph
            assert bert.shape[-1] == len(phone), phone

            if language == "ZH":
                bert = bert.astype(self.float_dtype)
                ja_bert = np.zeros((768, len(phone)), dtype=self.float_dtype)
            elif language in ["JP", "EN", "ZH_MIX_EN", 'KR', 'SP', 'ES', 'FR', 'DE', 'RU']:
                ja_bert = bert.astype(self.float_dtype)
                bert = np.zeros((1024, len(phone)), dtype=self.float_dtype)
            else:
                raise NotImplementedError()
        
        assert bert.shape[-1] == len(
            phone
        ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

        phone = np.array(phone, dtype=np.int32)
        tone = np.array(tone, dtype=np.int32)
        language = np.array(language, dtype=np.int32)
        
        x_tst = np.expand_dims(phone, axis=0)
        x_tst_lengths = np.array([phone.size], dtype=np.int32)
        tones = np.expand_dims(tone, axis=0)
        lang_ids = np.expand_dims(language, axis=0)
        
        bert = np.expand_dims(bert, axis=0)
        ja_bert = np.expand_dims(ja_bert, axis=0)
        
        speaker_id = np.array([1], dtype=np.int32)
        
        return x_tst, x_tst_lengths, speaker_id, tones, lang_ids, bert, ja_bert
    
    def generate_audio_chunked(self,
                           text:str,
                           language:str="ZH_MIX_EN",
                           sdp_ratio:float=0.2,
                           noise_scale_w:float=0.8,
                           speed:float=1.0,
                           chunk_size:int=239):
        """Generate audio in chunks
        Args:
            text (str): Input text to be synthesized.
            language (str, optional): Language of the input text. Defaults to "ZH_MIX_EN".
            sdp_ratio (float, optional): Ratio for SDP (Speech Decoder Proportion). Defaults to 0.2.
            noise_scale_w (float, optional): Scale factor for noise. Defaults to 0.8.
            speed (float, optional): Speed factor for audio playback. Defaults to 1.0.
            chunk_size (int, optional): Size of audio chunks for processing. Defaults to 239.
        
        Returns:
            Tuple[np.ndarray, int]: Audio data and sample rate
        """
        
        if language.lower() == "zh_mix_en" or language.lower() == "zh":
            language = "ZH_MIX_EN"
        elif language.lower() == "en":
            language = "EN"
        else:
            raise ValueError(f"language {language} not supported")
        
        x_tst, x_tst_lengths, speaker_id, tones, lang_ids, bert, ja_bert = self.__preprocess(text, language)
        
        np_sdp_ratio = np.array([sdp_ratio], dtype=self.float_dtype)
        np_noise_scale = np.array([0.667], dtype=self.float_dtype)
        np_noise_scale_w = np.array([noise_scale_w], dtype=self.float_dtype)
        np_speed = np.array([speed], dtype=self.float_dtype)
        
        total_len = x_tst_lengths[0]
        num_part = total_len // chunk_size + (1 if total_len % chunk_size != 0 else 0)
        logger.info(f"total_len: {total_len}, num_part: {num_part}")
        
        audio_seg = []
        
        for part in range(num_part):
            start = part * chunk_size
            end = min((part + 1) * chunk_size, total_len)
            actual_len = end - start  # ← save actual length, don't overwrite
            pad_len = chunk_size - actual_len  # ← length to pad
            
            x_tst_part    = x_tst[:, start:end]
            tone_part      = tones[:, start:end]
            lang_ids_part  = lang_ids[:, start:end]
            bert_part      = bert[:, :, start:end]
            ja_bert_part   = ja_bert[:, :, start:end]
            
            # Use actual_len for judgment, pad with pad_len
            if pad_len > 0:
                x_tst_part   = np.pad(x_tst_part,   ((0, 0), (0, pad_len)),      constant_values=0)
                tone_part     = np.pad(tone_part,     ((0, 0), (0, pad_len)),      constant_values=0)
                lang_ids_part = np.pad(lang_ids_part, ((0, 0), (0, pad_len)),      constant_values=0)
                bert_part     = np.pad(bert_part,     ((0, 0), (0, 0), (0, pad_len)), constant_values=0)
                ja_bert_part  = np.pad(ja_bert_part,  ((0, 0), (0, 0), (0, pad_len)), constant_values=0)
            
            # x_tst_lengths pass actual length (model uses mask internally), shape fixed to chunk_size
            x_tst_lengths_part = np.array([actual_len], dtype=np.int32)
            
            logger.info(f"part {part}: start={start}, end={end}, actual_len={actual_len}, pad_len={pad_len}")
            logger.info(f"  x_tst_part shape: {x_tst_part.shape}")
            logger.info(f"  tone_part shape: {tone_part.shape}")
            
            # Map inputs by name, compatible with 9 inputs (QNN) and 10 inputs (local ONNX)
            input_spec = {}
            named_inputs = {
                "x_tst": x_tst_part,
                "x_tst_lengths": x_tst_lengths_part,
                "speakers": speaker_id,
                "tones": tone_part,
                "lang_ids": lang_ids_part,
                "bert": bert_part,
                "ja_bert": ja_bert_part,
                "sdp_ratio": np_sdp_ratio,
                "noise_scale": np_noise_scale,
                "noise_scale_w": np_noise_scale_w,
                "speed": np_speed,
            }
            for k, v in named_inputs.items():
                if k in self.input_names:
                    input_spec[k] = v
            
            output_spec = self.session.run(self.output_names, input_spec)
            
            # Diagnostics: QNN raw output statistics
            audio_out = output_spec[0]
            logger.info(f"part {part}: raw_output shape={audio_out.shape}, "
                        f"min={audio_out.min():.6f}, max={audio_out.max():.6f}, "
                        f"nonzero={np.count_nonzero(audio_out)}")
            
            # If there is a y_lengths output, use it for precise trimming (most reliable)
            if len(output_spec) > 1:
                y_len = int(output_spec[1][0])  # mel frames
                valid_samples = y_len * self.hop_size
                audio_full = np.squeeze(audio_out, axis=0)[:valid_samples]
                logger.info(f"part {part}: y_lengths={y_len}, valid_samples={valid_samples}, "
                            f"trimmed_len={len(audio_full)}")
            else:
                # Compatibility with old models: use silence detection to trim
                audio_full = np.squeeze(audio_out, axis=0)
                audio_full = self._trim_trailing_silence(audio_full)
                logger.info(f"part {part}: full_audio_len={audio_out.shape[-1]}, "
                            f"trimmed_len={audio_full.shape[0]}")
            
            audio_seg.append(audio_full)
        
        combined_audio = np.concatenate(audio_seg, axis=0)
        return combined_audio, self.sample_rate
        
    def generate_audio(self,
                       text:str,
                       language:str="ZH_MIX_EN",
                       sdp_ratio:float=0.2,
                       noise_scale:float=0.667,
                       noise_scale_w:float=0.8,
                       speed:float=1.0) -> Tuple[np.ndarray, int]:
        """_summary_

        Args:
            text (str): User input text
            language (str, optional): Language of the text. Defaults to "ZH_MIX_EN".
            sdp_ratio (float, optional): Ratio of SDP. Defaults to 0.2.
            noise_scale (float, optional): Scale of noise. Defaults to 0.667.
            noise_scale_w (float, optional): Weight of noise scale. Defaults to 0.8.
            speed (float, optional): Speed of the audio. Defaults to 1.0.

        Returns:
            Tuple[np.ndarray, int]: Audio data and sample rate
        """
        if language.lower() == "zh_mix_en" or language.lower() == "zh":
            language = "ZH_MIX_EN"
        elif language.lower() == "en":
            language = "EN"
        else:
            raise ValueError(f"language {language} not supported")
        
        x_tst, x_tst_lengths, speaker_id, tones, lang_ids, bert, ja_bert = self.__preprocess(text, language)
        
        np_sdp_ratio = np.array([sdp_ratio], dtype=self.float_dtype)
        np_noise_scale = np.array([noise_scale], dtype=self.float_dtype)
        np_noise_scale_w = np.array([noise_scale_w], dtype=self.float_dtype)
        np_speed = np.array([speed], dtype=self.float_dtype)
        
        input_spec = {
            self.input_names[0]: x_tst,
            self.input_names[1]: x_tst_lengths,
            self.input_names[2]: speaker_id,
            self.input_names[3]: tones,
            self.input_names[4]: lang_ids,
            self.input_names[5]: bert,
            self.input_names[6]: ja_bert,
            self.input_names[7]: np_sdp_ratio,
            self.input_names[8]: np_noise_scale,
            self.input_names[9]: np_noise_scale_w,
            self.input_names[10]: np_speed,
        }

        output_spec = self.session.run(self.output_names, input_spec)[0]
        
        audio_data = np.squeeze(output_spec, axis=0)
        
        return audio_data, self.sample_rate

default_model_path = "./models/MeloTTS-ZH-MIXED-EN-ONNX/"
default_output_path = "./"
default_test_txt = "We officially launch a large language model aimed at tackling complex systems engineering and long-cycle intelligent agent tasks. Scaling up remains one of the most important ways to enhance the efficiency of artificial general intelligence. I also support numbers like 123."


if __name__ == "__main__":
    msg_info = "Run MeloTTS ONNX Inference. Both Support Dynamic and Static Model."
    
    usg_info = """
    Dynamic Model:
        python run_onnx.py -m ./models/melotts_onnx/ -o ./ -d qnn -i -t "你好，我是中英混合模型。Hello I am a mixed language model.我支持数字123" -l ZH_MIX_EN -sdp 0.2 -ns 0.667 -nsw 0.8 -s 1.0
    
    Static Model:
        python run_onnx.py -m ./models/melotts_onnx/ -o ./ -s -i -t "你好，我是中英混合模型。Hello I am a mixed language model.我支持数字123" -l ZH_MIX_EN -sdp 0.2 -nsw 0.8 -s 1.0
    """
    parser = argparse.ArgumentParser(description=msg_info, usage=usg_info)
    parser.add_argument("-m", "--model_path", type=str, default=default_model_path, help="Path to the ONNX model")
    parser.add_argument("-o", "--output_path", type=str, default=default_output_path, help="Path to save the output audio")
    parser.add_argument("-d", "--device", type=str, default="qnn", help="Device to run the model on")
    parser.add_argument("-i", "--is_dynamic", action="store_true", help="Whether to use dynamic model")
    parser.add_argument("-t", "--text", type=str, default=default_test_txt, help="User input text")
    parser.add_argument("-l", "--language", type=str, default="ZH_MIX_EN", help="Language of the text")
    parser.add_argument("-sdp", "--sdp_ratio", type=float, default=0.2, help="Ratio of SDP")
    parser.add_argument("-ns", "--noise_scale", type=float, default=0.667, help="Scale of noise")
    parser.add_argument("-nsw", "--noise_scale_w", type=float, default=0.8, help="Weight of noise scale")
    parser.add_argument("-s", "--speed", type=float, default=1.0, help="Speed of the audio")
    
    args = parser.parse_args()
    
    if args.device == "qnn":
        provider_options = [
            {
                'backend_path':f'{os.environ["QNN_SDK_ROOT"]}/lib/aarch64-oe-linux-gcc11.2/libQnnHtp.so',
                # 'htp_performance_mode': 'burst'
            }
        ]
    else:
        provider_options = None
    
    melo_tts = MeloTTS(model_root=args.model_path, 
                       device=args.device, 
                       provider_options=provider_options, 
                       is_dynamic=args.is_dynamic)
    
    if args.is_dynamic:
        audio, sr = melo_tts.generate_audio(
            text=args.text,
            language=args.language,
            sdp_ratio=args.sdp_ratio,
            noise_scale=args.noise_scale,
            noise_scale_w=args.noise_scale_w,
            speed=args.speed,
        )
    else:
        audio, sr = melo_tts.generate_audio_chunked(
            text=args.text,
            language=args.language,
            sdp_ratio=args.sdp_ratio,
            noise_scale_w=args.noise_scale_w,
            speed=args.speed,
            chunk_size=512 # same as the model input sequence length
        )
    
    audio_save_path = os.path.join(args.output_path, "dynamic_output.wav" if args.is_dynamic else "static_output.wav")
    os.makedirs(args.output_path, exist_ok=True)
    sf.write(audio_save_path, audio, sr)
```