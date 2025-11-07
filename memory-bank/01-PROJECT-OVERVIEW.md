# 01 - Project Overview

## 🎯 Dự án là gì?

**F5-TTS-Vietnamese** là một pipeline fine-tuning để training model Text-to-Speech (TTS) và Voice Cloning cho tiếng Việt, dựa trên kiến trúc F5-TTS (Flow Matching).

### Nguồn gốc
- **Base Project:** [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS)
- **Vietnamese Adaptation:** [lehieu29/TTS](https://github.com/lehieu29/TTS)
- **Demo:** https://huggingface.co/spaces/hynt/F5-TTS-Vietnamese-100h

---

## 🎨 Tính năng chính

### ✅ Đã có (Production Ready)

1. **Fine-tuning Pipeline**
   - Training model TTS cho giọng tiếng Việt cụ thể
   - Tự động mở rộng vocabulary
   - Checkpoint management
   - Multi-GPU support

2. **Inference System**
   - CLI inference tool
   - Gradio web UI
   - Custom model loading
   - Speed control (0.3x - 2.0x)

3. **Voice Cloning**
   - Zero-shot voice cloning với reference audio
   - Multi-speaker support
   - Voice chat integration

### 🚧 Kế hoạch mở rộng (từ YEUCAU.md)

1. **Audio Preprocessing**
   - Tách giọng nói khỏi nhạc nền (music separation)
   - Voice Activity Detection
   - Audio enhancement

2. **Automated Dataset Preparation**
   - Auto transcription với Whisper
   - Smart audio segmentation
   - Quality filtering

3. **Multi-Speaker Training System**
   - Upload và quản lý nhiều giọng
   - Training progress tracking
   - Speaker management UI

4. **Production Interface**
   - Google Colab integration
   - Gradio UI với 2 tabs (Training + Inference)
   - Real-time progress monitoring

---

## 🏗️ Kiến trúc Model

### F5-TTS Architecture
```
Text Input
    ↓
Text Encoder (Transformer)
    ↓
Duration Predictor
    ↓
Flow Matching (CFM)
    ↓
Mel-Spectrogram
    ↓
Vocoder (Vocos)
    ↓
Audio Output
```

### Đặc điểm kỹ thuật
- **Model Type:** DiT (Diffusion Transformer)
- **Tokenizer:** Character-based (tiếng Việt)
- **Sample Rate:** 24kHz
- **Mel Channels:** 100
- **Vocoder:** Vocos (default)

---

## 📦 Cấu trúc thư mục quan trọng

```
F5-TTS-Vietnamese/
├── src/f5_tts/              # Core library
│   ├── model/               # Model architecture
│   ├── train/               # Training scripts
│   └── infer/               # Inference scripts
│
├── data/                    # Data directory
│   ├── your_dataset/        # Raw audio + text
│   └── your_training_dataset/  # Processed data
│
├── ckpts/                   # Model checkpoints
│   └── your_training_dataset/
│       ├── pretrained_model_*.pt
│       └── model_*.pt       # Trained models
│
├── fine_tuning.sh           # Main training script
├── infer.sh                 # Inference script
├── prepare_metadata.py      # Data preparation
├── check_vocab_pretrained.py  # Vocab checking
└── extend_embedding_pretrained.py  # Embedding expansion
```

---

## 🎯 Use Cases

### 1. Voice Cloning cho người nổi tiếng
- Input: 100+ giờ audio podcast của người đó
- Output: Model TTS có thể nói bất kỳ text nào bằng giọng của họ

### 2. Audiobook Generation
- Input: Text sách tiếng Việt
- Output: Audiobook với giọng đọc tự nhiên

### 3. Multi-Speaker TTS System
- Input: Dữ liệu nhiều giọng nói khác nhau
- Output: System có thể chuyển đổi giữa các giọng

### 4. Voice Assistant tiếng Việt
- Input: Text response từ AI
- Output: Voice response tự nhiên

---

## 📊 Performance Metrics

### Training Results (từ tác giả)
- **100h data:** Đủ cho single voice với quality tốt
- **1000h data:** Excellent voice cloning cho multiple speakers
- **WER:** Thấp khi training với transcription chính xác

### Inference Speed
- **T4 GPU:** ~2-4s cho câu 10 giây
- **CPU:** ~10-20s cho câu 10 giây

---

## 🔗 Liên quan

### Papers
- **F5-TTS:** [A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching](https://arxiv.org/abs/2410.06885)
- **E2-TTS:** [Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS](https://arxiv.org/abs/2406.18009)

### Resources
- Original Repo: https://github.com/SWivid/F5-TTS
- Vietnamese Repo: https://github.com/lehieu29/TTS
- HuggingFace Model: https://huggingface.co/SWivid/F5-TTS

---

## 🎓 Yêu cầu kiến thức

### Cơ bản (để sử dụng)
- Python basics
- Command line
- Audio file formats

### Nâng cao (để customize)
- PyTorch
- Transformer architecture
- Audio signal processing
- Flow Matching / Diffusion Models

---

## 📈 Project Status

| Component | Status | Notes |
|-----------|--------|-------|
| Fine-tuning Pipeline | ✅ Production | Stable và tested |
| Inference CLI | ✅ Production | Command-line tool |
| Inference Gradio | ✅ Production | Web UI |
| Audio Preprocessing | 🚧 Planned | Trong YEUCAU.md |
| Multi-Speaker UI | 🚧 Planned | Trong YEUCAU.md |
| Google Colab | 🚧 Planned | Trong YEUCAU.md |

---

**Next:** [`02-QUICK-START.md`](02-QUICK-START.md) - Hướng dẫn bắt đầu nhanh



