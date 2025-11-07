# 02 - Quick Start Guide

## 🚀 Bắt đầu trong 15 phút

Hướng dẫn này giúp bạn setup và chạy inference nhanh nhất.

---

## 📋 Prerequisites

### Hệ thống
- **OS:** Linux/Windows/Mac
- **Python:** 3.10
- **GPU:** NVIDIA GPU với CUDA (khuyến nghị)
- **RAM:** 16GB+ (cho training)

### Tools
- Git
- Conda hoặc virtualenv
- sox, ffmpeg (cho audio processing)

---

## ⚙️ Installation

### Step 1: Clone repository

```bash
git clone https://github.com/lehieu29/TTS.git
cd F5-TTS-Vietnamese
```

### Step 2: Setup environment

```bash
# Tạo conda environment
conda create -n f5-tts python=3.10
conda activate f5-tts
```

### Step 3: Install PyTorch

```bash
# Với CUDA 12.4
pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

# Hoặc CPU only
pip install torch==2.4.0 torchaudio==2.4.0
```

### Step 4: Install F5-TTS

```bash
cd F5-TTS-Vietnamese
pip install -e .
```

### Step 5: Install audio tools

**Linux:**
```bash
sudo apt-get update
sudo apt-get install sox ffmpeg
```

**Windows:**
```bash
# Download sox và ffmpeg từ official websites
# Thêm vào PATH
```

**Mac:**
```bash
brew install sox ffmpeg
```

---

## 🎤 Quick Inference Test

### Option 1: CLI (Nhanh nhất)

```bash
f5-tts_infer-cli \
--model "F5TTS_Base" \
--ref_audio ref.wav \
--ref_text "cả hai bên hãy cố gắng hiểu cho nhau" \
--gen_text "xin chào, tôi là trợ lý ảo tiếng Việt" \
--speed 1.0
```

**Parameters:**
- `--model`: Model name (F5TTS_Base, E2TTS_Base, hoặc custom path)
- `--ref_audio`: File audio mẫu (giọng bạn muốn clone)
- `--ref_text`: Text của audio mẫu
- `--gen_text`: Text bạn muốn tạo giọng nói
- `--speed`: Tốc độ (0.3 - 2.0)

### Option 2: Gradio Web UI (Dễ dùng)

```bash
f5-tts_infer-gradio
```

Mở browser tại: `http://localhost:7860`

**Cách sử dụng UI:**
1. Upload file audio mẫu (~10s)
2. Nhập text của audio mẫu (hoặc để trống để auto-transcribe)
3. Nhập text muốn tạo giọng
4. Click "Synthesize"

---

## 🎓 Quick Training Test

### Step 1: Chuẩn bị dữ liệu mẫu

```bash
mkdir -p data/your_dataset
```

Đặt các file vào `data/your_dataset/`:
```
data/your_dataset/
├── audio_001.wav    # Audio file
├── audio_001.txt    # "xin chào các bạn"
├── audio_002.wav
├── audio_002.txt
└── ...
```

**Yêu cầu:**
- Format: WAV, 24kHz, mono
- Duration: 3-10 giây/file
- Tối thiểu: 50-100 files (~5-10 phút audio)
- Khuyến nghị: 100+ giờ cho quality tốt

### Step 2: Chỉnh sửa config

Mở `fine_tuning.sh` và thay đổi:

```bash
# Line 11: Tên dataset
DATASET_DIR="data/your_training_dataset"

# Line 18: Tên thí nghiệm
EXP_NAME="F5TTS_Base"
DATASET_NAME="your_training_dataset"

# Line 27: Stage muốn chạy (0-5)
stage=0      # Bắt đầu từ stage 0
stop_stage=5 # Chạy đến stage 5
```

### Step 3: Chạy training

```bash
bash fine_tuning.sh
```

**Stages sẽ chạy:**
1. Stage 0: Convert sample rate → 24kHz
2. Stage 1: Prepare metadata
3. Stage 2: Check vocabulary
4. Stage 3: Extend embedding
5. Stage 4: Feature extraction
6. Stage 5: Fine-tuning

**Thời gian ước tính:**
- 10 phút audio: ~30-60 phút training (50 epochs)
- 1 giờ audio: ~2-4 giờ training
- 100 giờ audio: ~2-3 ngày training

### Step 4: Test model đã train

Chỉnh sửa `infer.sh`:

```bash
f5-tts_infer-cli \
--model "F5TTS_Base" \
--ref_audio ref.wav \
--ref_text "cả hai bên hãy cố gắng hiểu cho nhau" \
--gen_text "đây là giọng nói được tạo bởi model của tôi" \
--speed 1.0 \
--vocab_file data/your_training_dataset/vocab.txt \
--ckpt_file ckpts/your_training_dataset/model_last.pt
```

Chạy:
```bash
bash infer.sh
```

---

## 🔍 Verify Installation

### Test 1: Check packages

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchaudio; print(f'TorchAudio: {torchaudio.__version__}')"
python -c "import f5_tts; print('F5-TTS: OK')"
```

### Test 2: Check CUDA (nếu có GPU)

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

### Test 3: Check audio tools

```bash
sox --version
ffmpeg -version
```

---

## 🐛 Common Issues

### Issue 1: "No module named 'f5_tts'"

**Solution:**
```bash
cd F5-TTS-Vietnamese
pip install -e .
```

### Issue 2: "CUDA out of memory"

**Solution:**
- Giảm `batch_size` trong `fine_tuning.sh`
- Hoặc train trên CPU (chậm hơn)

### Issue 3: "sox command not found"

**Solution:**
- Linux: `sudo apt-get install sox`
- Windows: Download từ https://sourceforge.net/projects/sox/
- Mac: `brew install sox`

### Issue 4: Audio không phát được

**Solution:**
- Check sample rate: phải là 24kHz
- Convert: `sox input.wav -r 24000 output.wav`

---

## 📁 File Structure After Setup

```
F5-TTS-Vietnamese/
├── data/
│   ├── your_dataset/              # Dữ liệu gốc
│   │   ├── *.wav
│   │   └── *.txt
│   │
│   └── your_training_dataset/     # Dữ liệu đã xử lý
│       ├── wavs/
│       ├── metadata.csv
│       ├── vocab.txt
│       ├── raw.arrow
│       └── duration.json
│
├── ckpts/
│   └── your_training_dataset/
│       ├── pretrained_model_1200000.pt  # Base model
│       ├── model_10000.pt               # Checkpoints
│       ├── model_20000.pt
│       └── model_last.pt                # Latest checkpoint
│
└── src/f5_tts/
    └── (source code)
```

---

## 🎯 Next Steps

### Để training model thật:
→ Đọc [`06-DATA-REQUIREMENTS.md`](06-DATA-REQUIREMENTS.md) - Chi tiết về dữ liệu

### Để hiểu pipeline training:
→ Đọc [`04-TRAINING-PIPELINE.md`](04-TRAINING-PIPELINE.md) - Chi tiết từng stage

### Để customize inference:
→ Đọc [`05-INFERENCE-PIPELINE.md`](05-INFERENCE-PIPELINE.md) - Advanced usage

### Để implement tính năng mới:
→ Đọc [`08-EXPANSION-ROADMAP.md`](08-EXPANSION-ROADMAP.md) - Kế hoạch mở rộng

---

## 💡 Tips

1. **Test với dữ liệu nhỏ trước** (~10 phút audio) để verify pipeline
2. **Luôn backup checkpoints** quan trọng
3. **Monitor GPU usage** với `nvidia-smi`
4. **Reference audio ngắn** (<15s) cho inference tốt hơn
5. **Quality > Quantity** - Audio rõ ràng quan trọng hơn số lượng

---

**Prev:** [`01-PROJECT-OVERVIEW.md`](01-PROJECT-OVERVIEW.md)  
**Next:** [`03-ARCHITECTURE.md`](03-ARCHITECTURE.md)



