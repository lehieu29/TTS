# 🎙️ Google Colab Cells - F5-TTS Vietnamese Voice Cloning

## 📋 Hướng dẫn sử dụng

### Cấu trúc Cells

Các cells được đánh số theo thứ tự chạy:
- `01_*` - Setup môi trường
- `02_*` - Install dependencies  
- `03_*` - Upload và preprocessing
- `04_*` - Training
- `05_*` - Inference & Testing

### Quan trọng: Virtual Environment

Do F5-TTS yêu cầu `numpy<2` nhưng Colab mặc định dùng `numpy>=2`, **BẮT BUỘC phải dùng venv**.

### Cách chạy trên Google Colab

#### Bước 1: Tạo notebook mới
- Vào Google Colab: https://colab.research.google.com
- File → New Notebook

#### Bước 2: Copy từng cell
- Mở file trong thư mục `colab-cells/`
- Copy nội dung
- Paste vào cell trong Colab
- Run cell (Ctrl+Enter hoặc Shift+Enter)

#### Bước 3: Chạy theo thứ tự
```
01_setup_environment.py          # Setup venv + mount Drive
02_install_dependencies.py       # Install F5-TTS + tools
03_install_preprocessing.py      # Install Demucs, Whisper, VAD
04_upload_audio.py               # Upload podcast/audio files
05_separate_vocals.py            # Tách giọng khỏi nhạc nền
06_detect_segments.py            # Voice Activity Detection
07_transcribe_audio.py           # Auto transcription
08_prepare_dataset.py            # Prepare training data
09_train_model.py                # Training
10_inference_test.py             # Test model
11_gradio_interface.py           # Web UI
```

### Runtime Settings

**Khuyến nghị:**
```yaml
Runtime Type: Python 3
Hardware Accelerator: GPU (T4 minimum)
GPU Type: 
  - Free: T4
  - Pro: V100 hoặc A100
RAM: High RAM (nếu có)
```

**Setup:**
1. Runtime → Change runtime type
2. Chọn GPU
3. Save

### Storage Management

#### Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

#### Thư mục làm việc
```
/content/
├── drive/MyDrive/F5TTS/
│   ├── models/              # Saved models
│   ├── datasets/            # Processed datasets
│   └── outputs/             # Generated audio
├── venv/                    # Virtual environment
├── uploads/                 # Uploaded audio
└── F5-TTS-Vietnamese/       # Source code
```

### Tips

1. **Save checkpoints thường xuyên** - Colab có thể disconnect
2. **Backup vào Drive** - Tránh mất data
3. **Test với data nhỏ trước** - Verify pipeline
4. **Monitor GPU usage** - `!nvidia-smi`
5. **Clear output khi cần** - Tiết kiệm RAM

### Troubleshooting

#### Issue: "Session crashed"
→ Restart runtime và chạy lại từ cell 01

#### Issue: "Disk space full"
→ Clean up: `!rm -rf /content/tmp/*`

#### Issue: "numpy version conflict"
→ Đảm bảo đã activate venv trong mỗi cell

#### Issue: "GPU not available"
→ Runtime → Change runtime type → GPU

### Time Estimates

```yaml
Setup (cells 01-03): ~10 phút
Preprocessing (cells 04-07): ~30-60 phút cho podcast 30 phút
Training (cell 09): ~2-4 giờ cho 30 phút audio
Inference (cell 10): ~2-5 giây/sentence
```

### Notes

- ⚠️ Mỗi cell có thể mất vài phút chạy
- ⚠️ Colab Free có 12-hour limit
- ⚠️ Luôn save checkpoints vào Drive
- ✅ Test với audio ngắn trước (5 phút)
- ✅ Monitor progress với tqdm bars

---

**Bắt đầu từ cell 01 và chạy tuần tự! 🚀**



