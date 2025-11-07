# 10 - Troubleshooting

## 🔧 Common Issues & Solutions

---

## 📦 Installation Issues

### Issue: "No module named 'f5_tts'"

**Symptoms:**
```bash
ModuleNotFoundError: No module named 'f5_tts'
```

**Solutions:**
```bash
# 1. Install in editable mode
cd F5-TTS-Vietnamese
pip install -e .

# 2. Verify installation
python -c "import f5_tts; print('OK')"

# 3. Check Python path
python -c "import sys; print(sys.path)"
```

### Issue: "CUDA not available" hoặc GPU không được detect

**Symptoms:**
```python
torch.cuda.is_available() returns False
```

**Solutions:**
```bash
# 1. Check CUDA installation
nvidia-smi

# 2. Reinstall PyTorch với CUDA
pip uninstall torch torchaudio
pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

# 3. Verify
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.cuda)"
```

### Issue: "sox: command not found"

**Solutions:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install sox libsox-fmt-all

# Mac
brew install sox

# Windows
# Download from https://sourceforge.net/projects/sox/
# Add to PATH
```

---

## 🎓 Training Issues

### Issue: "CUDA out of memory"

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Solutions:**

**Option 1: Giảm batch size**
```bash
# Trong fine_tuning.sh
BATCH_SIZE=3200  # Từ 7000 → 3200

# Hoặc trong command
--batch_size_per_gpu 3200
```

**Option 2: Gradient accumulation**
```bash
--batch_size_per_gpu 3200 \
--grad_accumulation_steps 2  # Effectively 6400
```

**Option 3: Mixed precision training**
```python
# Đã enable mặc định
# Kiểm tra xem có dùng fp16 không
--mixed_precision fp16
```

**Option 4: Clear cache**
```python
import torch
torch.cuda.empty_cache()
```

### Issue: Loss không giảm / NaN loss

**Symptoms:**
```
Epoch 1 | Loss: 2.345
Epoch 2 | Loss: 2.341
Epoch 3 | Loss: 2.338
... (không giảm đáng kể)

# Hoặc
Epoch 5 | Loss: nan
```

**Causes & Solutions:**

**1. Learning rate quá cao**
```bash
# Giảm learning rate
--learning_rate 5e-6  # Thay vì 1e-5
```

**2. Gradient exploding**
```bash
# Check gradient norm
# Nếu > 10.0 → có vấn đề
--max_grad_norm 0.5  # Giảm từ 1.0
```

**3. Data quality kém**
```python
# Check transcription accuracy
# Check audio quality (SNR > 20dB)
# Check duration distribution
```

**4. Batch size quá nhỏ**
```bash
# Tăng batch size hoặc gradient accumulation
--batch_size_per_gpu 7000 \
--grad_accumulation_steps 2
```

### Issue: "vocab.txt not found"

**Symptoms:**
```
FileNotFoundError: data/your_training_dataset/vocab.txt
```

**Solutions:**
```bash
# Phải chạy Stage 2 trước
stage=2
stop_stage=2
bash fine_tuning.sh

# Verify
ls data/your_training_dataset/vocab.txt
```

### Issue: Training quá chậm

**Symptoms:**
- Steps per second < 0.2
- ETA > 1 tuần

**Solutions:**

**1. Tăng batch size**
```bash
--batch_size_per_gpu 10000  # Nếu có GPU memory
```

**2. Tăng num_workers**
```bash
# Trong prepare_csv_wavs.py
--workers 8  # Thay vì 4
```

**3. Use faster storage**
```bash
# Copy data to local SSD
cp -r data/ /tmp/data/
# Update paths trong training script
```

**4. Profile bottlenecks**
```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA]
) as prof:
    # Training code
    ...

print(prof.key_averages().table())
```

---

## 🎤 Inference Issues

### Issue: Output audio có nhiều silence

**Solutions:**

**Option 1: Enable remove_silence**
```bash
f5-tts_infer-cli \
--ref_audio ref.wav \
--ref_text "..." \
--gen_text "..." \
--remove_silence
```

**Option 2: Post-process manually**
```python
from f5_tts.infer.utils_infer import remove_silence_for_generated_wav

remove_silence_for_generated_wav("output.wav")
```

### Issue: Giọng không giống reference

**Causes & Solutions:**

**1. Reference audio quá ngắn**
```bash
# Solution: Use 5-10s reference
# Check duration
ffprobe -i ref.wav -show_entries format=duration
```

**2. Reference text không chính xác**
```bash
# Solution: Provide accurate ref_text
--ref_text "chính xác text của ref_audio"
# Không để trống
```

**3. Reference audio quality kém**
```python
# Check SNR
import soundfile as sf
import numpy as np

audio, sr = sf.read("ref.wav")
# Nên có SNR > 20dB, ít noise
```

### Issue: Phát âm sai (tiếng Việt)

**Causes & Solutions:**

**1. Model chưa train với tiếng Việt**
```bash
# Solution: Fine-tune với dữ liệu tiếng Việt
# Minimum 1-10 giờ data
```

**2. Vocab thiếu ký tự**
```bash
# Check vocab.txt có đầy đủ ă â đ ê ô ơ ư không
cat data/your_training_dataset/vocab.txt | grep 'ă\|â\|đ\|ê\|ô\|ơ\|ư'
```

**3. Text có ký tự lạ**
```python
# Normalize text trước khi inference
def normalize(text):
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    return text
```

### Issue: Output có artifacts/glitches

**Solutions:**

**1. Tăng NFE steps**
```bash
--nfe_step 64  # Thay vì 32
```

**2. Giảm speed**
```bash
--speed 0.9  # Chậm hơn một chút
```

**3. Check model checkpoint**
```bash
# Thử checkpoint khác
--ckpt_file ckpts/your_training_dataset/model_20000.pt
# Thay vì model_last.pt
```

---

## 📁 Data Issues

### Issue: "metadata.csv is empty" hoặc ít samples

**Causes & Solutions:**

**1. Duration filter quá strict**
```python
# Trong prepare_metadata.py
# Kiểm tra filter conditions
if duration < 1 or duration > 30:  # Có thể nới lỏng
    continue
```

**2. Text filter**
```python
if len(text.split()) < 3:  # Có thể giảm xuống 2
    continue
```

**3. File paths sai**
```bash
# Check
ls data/your_dataset/*.wav
ls data/your_dataset/*.txt
# Phải có cả 2
```

### Issue: Audio quality kém sau voice separation

**Solutions:**

**1. Thử model Demucs khác**
```bash
# Thay vì htdemucs
demucs -n mdx_extra ...
```

**2. Skip separation nếu audio sẵn clean**
```bash
# Trong fine_tuning.sh
stage=1  # Bỏ qua stage 0 (convert_sr) và separation
```

**3. Manual cleanup**
```bash
# Dùng Audacity để manual clean
# Export clean audio
```

### Issue: Transcription không chính xác

**Solutions:**

**1. Dùng Whisper large-v3**
```python
model = whisper.load_model("large-v3")  # Thay vì medium
```

**2. Manual correction**
```python
# Review và sửa transcriptions
# Quan trọng nhất cho quality
```

**3. Dùng ASR tiếng Việt chuyên dụng**
```python
# FPT.AI ASR hoặc VAIS ASR
# Accuracy cao hơn cho tiếng Việt
```

---

## 🐛 System Issues

### Issue: "Disk space full"

**Solutions:**
```bash
# 1. Clean temp files
rm -rf /tmp/*

# 2. Remove old checkpoints
cd ckpts/your_training_dataset/
rm model_1000.pt model_2000.pt  # Giữ lại model_last.pt

# 3. Clean cache
pip cache purge
rm -rf ~/.cache/torch
rm -rf ~/.cache/huggingface
```

### Issue: Process killed / Out of memory (RAM)

**Solutions:**
```bash
# 1. Check RAM usage
free -h

# 2. Reduce num_workers
--workers 2  # Thay vì 8

# 3. Process in batches
# Chia dataset thành nhiều phần nhỏ
```

### Issue: "Connection timeout" khi download model

**Solutions:**
```bash
# 1. Manual download
wget https://huggingface.co/SWivid/F5-TTS/resolve/main/F5TTS_Base/model_1200000.pt

# 2. Use mirror
export HF_ENDPOINT=https://hf-mirror.com

# 3. Resume download
wget -c <URL>
```

---

## 🔍 Debugging Tips

### Enable Debug Logging

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Check GPU Usage

```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Or use gpustat
pip install gpustat
gpustat -i 1
```

### Profile Training

```python
# Add to training script
import time

start = time.time()
# ... training code ...
elapsed = time.time() - start

print(f"Time: {elapsed:.2f}s")
print(f"Steps/sec: {num_steps / elapsed:.2f}")
```

### Validate Data

```python
# Check data statistics
import pandas as pd
import matplotlib.pyplot as plt

# Load metadata
df = pd.read_csv("data/your_training_dataset/metadata.csv", sep="|")

# Duration distribution
plt.hist(df['duration'], bins=50)
plt.xlabel("Duration (s)")
plt.ylabel("Count")
plt.show()

# Text length distribution
df['text_length'] = df['text'].str.len()
plt.hist(df['text_length'], bins=50)
plt.show()
```

---

## 📞 Getting Help

### Before Asking for Help

**Collect Information:**
```bash
# 1. System info
uname -a
python --version
pip list | grep torch

# 2. Error messages
# Copy FULL error traceback

# 3. Configuration
cat fine_tuning.sh
ls -lh data/
ls -lh ckpts/

# 4. Logs
# Include relevant logs
```

### Where to Ask

1. **GitHub Issues**: https://github.com/lehieu29/TTS/issues
2. **Original F5-TTS Repo**: https://github.com/SWivid/F5-TTS
3. **Stack Overflow**: Tag với `tts`, `pytorch`

### Include in Bug Report

```markdown
## Environment
- OS: Ubuntu 22.04
- Python: 3.10.12
- PyTorch: 2.4.0+cu124
- GPU: NVIDIA RTX 3090

## Issue
[Clear description]

## Steps to Reproduce
1. ...
2. ...

## Error Message
```
[Full traceback]
```

## Expected Behavior
[What should happen]

## Actual Behavior
[What actually happens]

## Additional Context
[Config files, logs, etc.]
```

---

**Prev:** [`09-IMPLEMENTATION-GUIDE.md`](09-IMPLEMENTATION-GUIDE.md)  
**Next:** [`11-FAQ.md`](11-FAQ.md)



