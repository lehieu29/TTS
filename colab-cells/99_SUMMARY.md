# 📊 Google Colab Cells Summary

## ✅ Complete! Đã tạo 12 cells

### 🎯 Cell Overview

| Cell | Name | Purpose | Time |
|------|------|---------|------|
| 00 | README.md | Hướng dẫn sử dụng | - |
| 01 | setup_environment.py | Mount Drive, tạo venv | 2 min |
| 02 | install_dependencies.py | Install F5-TTS + PyTorch | 15 min |
| 03 | install_preprocessing.py | Install Demucs, Whisper, VAD | 15 min |
| 04 | upload_and_prepare.py | Upload audio files | 5 min |
| 05 | voice_separation.py | Tách giọng/nhạc (Demucs) | 30-60 min |
| 06 | segment_audio.py | VAD segmentation | 5-10 min |
| 07 | transcribe.py | Whisper transcription | 10-15 min |
| 08 | prepare_training_data.py | Prepare features | 5-10 min |
| 09 | train_model.py | Train F5-TTS | 2-4 hours |
| 10 | test_inference.py | Test generated speech | 5 min |
| 11 | gradio_interface.py | Web UI demo | 2 min |

**Total Time: ~3-5 hours** (phụ thuộc vào training time)

---

## 🚀 Workflow Tổng Quát

```
01. Setup → 02. Install Core → 03. Install Tools
                     ↓
04. Upload Audio → 05. Voice Separation (optional)
                     ↓
06. VAD Segmentation → 07. Transcription
                     ↓
08. Prepare Training Data → 09. TRAIN (2-4h)
                     ↓
10. Test Inference → 11. Gradio UI
```

---

## 📋 Cách Sử Dụng

### Bước 1: Setup (Cells 01-03)
```
Run: 01 → 02 → 03
Time: ~30 minutes
Output: Environment ready, all tools installed
```

### Bước 2: Data Preparation (Cells 04-08)
```
Run: 04 → 05 → 06 → 07 → 08
Time: ~1-2 hours (tùy audio length)
Output: Training data ready
```

### Bước 3: Training (Cell 09)
```
Run: 09
Time: 2-4 hours
Output: Trained model
```

### Bước 4: Inference (Cells 10-11)
```
Run: 10 → 11
Time: ~10 minutes
Output: Working demo
```

---

## 🎯 Key Features

### ✅ Virtual Environment
- Tránh numpy conflict
- Isolated dependencies
- Stable environment

### ✅ Google Drive Integration
- Auto backup models
- Persistent storage
- Resume training

### ✅ Progress Monitoring
- Real-time logs
- Progress bars
- Status updates

### ✅ Error Handling
- Validation checks
- Clear error messages
- Recovery options

### ✅ User-Friendly
- Step-by-step instructions
- Examples included
- Interactive prompts

---

## 💡 Important Notes

### 🔴 MUST DO:
1. **Enable GPU:** Runtime → Change runtime type → GPU
2. **Use venv:** Bắt buộc để tránh numpy conflict
3. **Mount Drive:** Để save models và data
4. **Run sequentially:** Đúng thứ tự 01 → 11

### ⚠️ OPTIONAL:
1. **Voice Separation (Cell 05):** Skip nếu audio sạch
2. **Reference Text:** Có thể để trống (auto-transcribe)
3. **Custom Texts:** Hoặc dùng examples

### 📊 RESOURCES:
```yaml
GPU: T4 minimum (Free Colab OK)
RAM: 12GB recommended
Disk: 10-20GB
Runtime: Keep-alive (training takes hours)
```

---

## 🐛 Troubleshooting

### Issue: "numpy version conflict"
```python
# Solution: Check venv is activated
# Every cell should use venv_python and venv_pip
```

### Issue: "CUDA out of memory"
```python
# Solution: In Cell 09, reduce batch_size
TRAINING_CONFIG["batch_size"] = 3200  # From 7000
```

### Issue: "Session disconnected"
```python
# Solution: 
# 1. Checkpoints auto-saved to Drive
# 2. Resume from Cell 09
# 3. Model will continue from last checkpoint
```

### Issue: "Poor audio quality"
```python
# Causes:
# 1. Not enough training data (need 50-100 hours)
# 2. Bad transcriptions
# 3. Noisy audio

# Solutions:
# - Add more clean data
# - Verify transcriptions
# - Use voice separation (Cell 05)
```

---

## 📁 Output Structure

```
/content/
├── venv/                           # Virtual environment
├── F5-TTS-Vietnamese/              # Source code
├── uploads/                        # Uploaded audio
├── processed/
│   ├── vocals/                     # Separated vocals
│   └── segments/                   # VAD segments
├── data/
│   └── {speaker}_training/         # Training data
│       ├── wavs/
│       ├── metadata.csv
│       ├── vocab.txt
│       ├── raw.arrow
│       └── duration.json
├── ckpts/
│   └── {speaker}_training/         # Training checkpoints
│       └── model_*.pt
├── models/
│   └── {speaker}/                  # Final models
│       ├── model.pt
│       ├── vocab.txt
│       └── config.json
└── outputs/                        # Generated audio

/content/drive/MyDrive/F5TTS_Vietnamese/
├── models/                         # Backed up models
├── checkpoints/                    # Training checkpoints
├── outputs/                        # Generated samples
├── training_data/                  # Processed data
├── logs/                           # Training logs
└── processing_config.json          # Configuration
```

---

## 🎉 Success Criteria

### ✅ Setup Complete:
- GPU detected
- All packages installed
- No errors

### ✅ Data Ready:
- Audio segmented
- All transcribed
- Features extracted

### ✅ Training Complete:
- Model saved
- Checkpoints backed up
- No crashes

### ✅ Inference Working:
- Speech generated
- Quality acceptable
- Gradio UI running

---

## 📞 Support

### Need Help?
1. Check cell output for errors
2. Read memory-bank docs
3. Check Drive backups
4. Try with smaller data first

### Report Issues:
- Cell number
- Error message
- Configuration used
- System specs

---

## 🎯 Tips for Best Results

### 1. Data Quality
```
✅ DO:
- Use clean audio (no music/noise)
- Accurate transcriptions (100%)
- 50-100 hours of data
- Consistent speaker

❌ DON'T:
- Use noisy audio
- Skip transcription check
- Use too little data (<1 hour)
- Mix multiple speakers
```

### 2. Training
```
✅ DO:
- Monitor GPU usage
- Save checkpoints frequently
- Use T4/V100 GPU
- Train for 50-100 epochs

❌ DON'T:
- Use CPU
- Skip checkpoints
- Train too few epochs
- Interrupt training randomly
```

### 3. Inference
```
✅ DO:
- Use clear reference audio (5-10s)
- Provide reference text
- Use proper Vietnamese text
- Test with multiple texts

❌ DON'T:
- Use very long reference (>15s)
- Skip reference text
- Use text without diacritics
- Expect perfection immediately
```

---

## 🚀 Next Steps After Completion

### 1. Improve Quality
- Add more training data
- Clean up transcriptions
- Fine-tune hyperparameters

### 2. Experiment
- Try different speakers
- Test various texts
- Adjust inference parameters

### 3. Deploy
- Share Gradio link
- Export models
- Create API service

### 4. Scale Up
- Train on more data
- Multi-speaker model
- Production deployment

---

**🎊 Chúc bạn thành công với Voice Cloning! 🎙️✨**

---

**Last Updated:** 2025-11-06  
**Version:** 1.0  
**Status:** ✅ Complete & Ready to Use



