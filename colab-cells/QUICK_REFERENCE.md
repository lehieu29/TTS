# 🚀 Quick Reference Guide

## ⚡ Fast Start (TL;DR)

### Minimum Steps to Train Model:

```
1. Create new Colab notebook
2. Enable GPU: Runtime → Change runtime type → GPU
3. Copy & Run cells in order: 01 → 02 → 03 → ... → 11
4. Wait for training (~3-5 hours total)
5. Done! Use Gradio UI (Cell 11)
```

---

## 📊 Cell Quick Reference

### Must Run (Sequential):
```
01 → Setup environment (venv + Drive)
02 → Install F5-TTS
03 → Install preprocessing tools
04 → Upload audio
08 → Prepare training data
09 → Train model
10 → Test inference
```

### Optional:
```
05 → Voice separation (skip if audio is clean)
06 → VAD segmentation (auto-included in 08)
07 → Transcription (auto-included in 08)
11 → Gradio UI (nice to have)
```

---

## 🎯 Critical Commands

### Check GPU:
```python
!nvidia-smi
```

### Activate venv (in each cell):
```python
venv_python = "/content/venv/bin/python"
venv_pip = "/content/venv/bin/pip"
```

### Check disk space:
```bash
!df -h /content
```

### Kill process if needed:
```python
!pkill -f python
```

---

## 🔥 Common Issues - Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| numpy conflict | Use venv (Cell 01) |
| CUDA OOM | Reduce batch_size in Cell 09 |
| Session timeout | Checkpoints auto-saved to Drive |
| No GPU | Runtime → GPU |
| Disk full | Clean `/content/tmp/` |
| Import error | Restart runtime, run from Cell 01 |

---

## 📦 File Locations (Quick Access)

```bash
# Models
/content/models/{speaker}/model.pt

# Drive Backup
/content/drive/MyDrive/F5TTS_Vietnamese/models/

# Outputs
/content/outputs/

# Config
/content/processing_config.json
```

---

## ⚙️ Config Quick Edit

### Reduce memory usage (Cell 09):
```python
TRAINING_CONFIG["batch_size"] = 3200  # Lower
```

### Faster training (less quality):
```python
TRAINING_CONFIG["epochs"] = 30  # Reduce
```

### Better quality (slower):
```python
TRAINING_CONFIG["epochs"] = 100  # Increase
TRAINING_CONFIG["batch_size"] = 10000  # If you have GPU memory
```

---

## 🎤 Inference Quick Test

```python
# Cell 10 - Fast test
venv_python /content/F5-TTS-Vietnamese/src/f5_tts/infer/infer_cli.py \
  --model F5TTS_Base \
  --ref_audio /content/data/speaker/wavs/sample.wav \
  --ref_text "xin chào" \
  --gen_text "hôm nay trời đẹp" \
  --vocab_file /content/models/speaker/vocab.txt \
  --ckpt_file /content/models/speaker/model.pt
```

---

## 💡 Pro Tips

### 1. Save Time:
- Skip Cell 05 if audio is clean
- Use smaller dataset for testing first
- Resume training from checkpoints

### 2. Save Resources:
- Clear outputs: Edit → Clear all outputs
- Remove old checkpoints: `!rm /content/ckpts/*/model_1*.pt`
- Compress Drive backups

### 3. Better Quality:
- More data > fancy techniques
- Clean audio > fancy processing
- Accurate transcription > auto-transcribe

### 4. Debugging:
- Check logs in `/content/drive/.../logs/`
- Test with short audio first (5 min)
- Verify each step before proceeding

---

## 🔄 Resume After Disconnect

```python
# 1. Run Cell 01 (setup)
# 2. Run Cell 02 (install)
# 3. Load config:
import json
with open('/content/drive/MyDrive/F5TTS_Vietnamese/processing_config.json') as f:
    config = json.load(f)

# 4. Resume from Cell 09 (training will continue from checkpoint)
```

---

## 📈 Monitor Progress

### During Training:
```bash
# Watch GPU
watch -n 1 nvidia-smi

# Check logs
tail -f /content/drive/MyDrive/F5TTS_Vietnamese/logs/*_training.log

# Check checkpoint size
du -h /content/ckpts/
```

### Check Quality:
```python
# After Cell 10
from IPython.display import Audio
Audio('/content/outputs/generated.wav', rate=24000)
```

---

## 🎯 Expected Results

### Training Loss:
```
Epoch 1: Loss ~2.0
Epoch 25: Loss ~0.8
Epoch 50: Loss ~0.5
Epoch 100: Loss ~0.3

✅ Good: Loss decreasing steadily
❌ Bad: Loss flat or increasing
```

### Audio Quality:
```
✅ Clear pronunciation
✅ Natural prosody
✅ Sounds like speaker
✅ No artifacts

❌ Robotic
❌ Mispronunciations
❌ Unnatural pauses
❌ Distortion
```

---

## 🆘 Emergency Commands

### Stop everything:
```bash
!pkill -9 python
!killall python
```

### Free memory:
```python
import gc
import torch
gc.collect()
torch.cuda.empty_cache()
```

### Reset completely:
```
Runtime → Factory reset runtime
# Then start from Cell 01
```

### Backup NOW:
```bash
!cp -r /content/ckpts/* /content/drive/MyDrive/F5TTS_Vietnamese/emergency_backup/
!cp -r /content/models/* /content/drive/MyDrive/F5TTS_Vietnamese/emergency_backup/
```

---

## 📞 Get Help

### Check These First:
1. ✅ GPU enabled?
2. ✅ venv activated?
3. ✅ Enough disk space?
4. ✅ Drive mounted?
5. ✅ Config exists?

### Still Stuck?
- Read error message carefully
- Check memory-bank docs
- Try with minimal data first
- Check Drive backups
- Restart from Cell 01

---

## 🎓 Learning Path

### Day 1: Setup & Test
```
- Run Cells 01-03 (setup)
- Test with 5-min audio sample
- Verify each step works
```

### Day 2: Small Training
```
- Use 30-min audio
- Complete full pipeline
- Check quality
```

### Day 3: Full Training
```
- Use full dataset (50-100h)
- Train for production
- Fine-tune parameters
```

---

## ✅ Checklist Before Training

```
□ GPU enabled and detected
□ Google Drive mounted
□ Virtual environment created
□ All dependencies installed
□ Audio files uploaded
□ Transcriptions accurate
□ Features extracted
□ Pretrained model downloaded
□ Enough disk space (>10GB)
□ Enough time (3-5 hours)
```

---

## 🎉 Success Metrics

### Setup Success:
```
✅ No import errors
✅ GPU detected
✅ Packages installed
```

### Training Success:
```
✅ Loss decreasing
✅ Checkpoints saved
✅ No crashes
```

### Inference Success:
```
✅ Audio generated
✅ Quality acceptable
✅ Gradio working
```

---

**💡 Remember: Quality = Data Quality × Training Time × Patience**

**🚀 Good luck with your Voice Cloning! 🎙️**

---

**Quick Links:**
- [00_README.md](00_README.md) - Full instructions
- [99_SUMMARY.md](99_SUMMARY.md) - Detailed summary
- Memory Bank: `../memory-bank/` - Complete documentation



