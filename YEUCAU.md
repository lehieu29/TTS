# Kế hoạch chi tiết: Hệ thống xử lý âm thanh và Clone giọng tiếng Việt

## 🎯 Tổng quan hệ thống

Xây dựng pipeline hoàn chỉnh trên Google Colab để:
1. Xử lý file podcast (loại nhạc nền, tách giọng)
2. Chuẩn bị dữ liệu huấn luyện
3. Training model clone giọng với F5-TTS-Vietnamese
4. Tạo giao diện sử dụng đa giọng

---

## 📋 PHASE 1: Tiền xử lý Audio (Audio Preprocessing)

### 1.1. Upload và quản lý file
**Mục tiêu**: Cho phép người dùng upload nhiều file MP3/WAV

**Công nghệ**:
- Gradio FileUpload component
- Lưu trữ tạm trong `/content/uploads/`
- Hỗ trợ batch upload nhiều file cùng lúc

**Thông tin cần thu thập**:
- Tên file gốc
- Tên giọng (do user đặt) - text input
- Metadata: duration, sample rate

### 1.2. Tách giọng nói khỏi nhạc nền (Voice Separation)
**Vấn đề**: File podcast 30 phút có nhạc nền

**Giải pháp đề xuất**:

#### Option 1: **Demucs** (Facebook Research) - ĐỀ XUẤT
- **Lý do chọn**: 
  - SOTA trong voice separation
  - Pretrained model tốt với tiếng Việt
  - Xử lý nhanh trên GPU
  - Model: `htdemucs` hoặc `htdemucs_ft`

- **Quy trình**:
  ```
  Input MP3 (30 phút) 
  → Demucs separation 
  → Output: vocals.wav (giọng nói thuần)
  ```

#### Option 2: **Spleeter** (Deezer)
- Backup option nếu Demucs quá chậm
- Model 2stems (vocals/accompaniment)

**Tối ưu xử lý file 30 phút**:
- Chunk processing: chia file thành segments 5-10 phút
- Process parallel nếu có multi-GPU
- Áp dụng batch processing cho nhiều file

### 1.3. Voice Activity Detection (VAD)
**Mục tiêu**: Loại bỏ đoạn im lặng, chỉ giữ lại speech segments

**Công nghệ**:
- **Silero VAD** (đề xuất) - tốt với tiếng Việt
- Hoặc **WebRTC VAD**

**Quy trình**:
1. Detect speech segments
2. Loại bỏ silence > 0.5s
3. Extract clean speech segments
4. Lưu timestamps cho mỗi segment

### 1.4. Audio Quality Enhancement (Optional nhưng quan trọng)
**Mục tiêu**: Cải thiện chất lượng audio sau khi tách

**Công nghệ**:
- **DeepFilterNet**: Noise reduction
- **Resemble Enhance**: Audio super-resolution

**Áp dụng**:
- Noise reduction
- Normalize volume
- Resample về 24kHz (yêu cầu của F5-TTS)

---

## 📋 PHASE 2: Chuẩn bị Dataset cho Training

### 2.1. Audio Segmentation
**Mục tiêu**: Chia audio dài thành clips ngắn phù hợp training

**Yêu cầu F5-TTS**:
- Duration: 3-10 giây/clip (optimal: 5-7s)
- Format: WAV, 24kHz, mono
- Chất lượng: SNR > 20dB

**Chiến lược chia segments**:
1. **Smart Segmentation**:
   - Dùng VAD timestamps
   - Chia theo câu hoàn chỉnh (dùng pause detection)
   - Tránh cắt giữa từ

2. **Filtering**:
   - Loại clip < 2s hoặc > 12s
   - Loại clip có SNR thấp
   - Loại clip có music bleed-through còn sót

3. **Quality Check**:
   - Auto-detect clips có vấn đề
   - Manual review interface (nghe mẫu ngẫu nhiên)

### 2.2. Transcription (Chuyển âm thanh thành text)
**Vấn đề**: F5-TTS cần cặp (audio, text) để training

**Giải pháp**:

#### Option 1: **Whisper Large-v3** - ĐỀ XUẤT
- Accuracy cao nhất với tiếng Việt
- Model: `openai/whisper-large-v3`
- Có timestamp alignment

#### Option 2: **FPT.AI ASR** hoặc **VAIS ASR**
- Nếu cần accuracy cao hơn cho tiếng Việt
- API-based (cần internet)

**Quy trình**:
1. Transcribe từng segment
2. Lưu text file với cùng tên audio
3. Format: `segment_001.wav` → `segment_001.txt`

**Tối ưu cho file 30 phút**:
- Batch transcription
- Cache results
- Progress bar hiển thị

### 2.3. Text Normalization
**Mục tiêu**: Chuẩn hóa text cho training

**Xử lý**:
- Lowercase (nếu cần)
- Remove special characters không cần thiết
- Chuẩn hóa số → chữ (123 → một trăm hai mươi ba)
- Xử lý viết tắt
- Đảm bảo Unicode NFD normalization

### 2.4. Dataset Organization
**Cấu trúc thư mục**:
```
/content/datasets/
├── speaker_001/
│   ├── wavs/
│   │   ├── segment_001.wav
│   │   ├── segment_002.wav
│   │   └── ...
│   ├── metadata.csv  # path|text|speaker_id
│   └── sample.wav    # Audio demo 5-10s
├── speaker_002/
│   └── ...
└── config.json       # Lưu thông tin speakers
```

**Metadata Format**:
```csv
audio_path,text,speaker_name,duration
wavs/segment_001.wav,"xin chào các bạn",speaker_001,3.2
```

---

## 📋 PHASE 3: Training với F5-TTS-Vietnamese

### 3.1. Setup Environment
**Cài đặt**:
1. Install dependencies
2. Download pretrained base model (nếu có)
3. Setup GPU (T4/V100 trên Colab)

### 3.2. Training Configuration
**Hyperparameters cần điều chỉnh**:

```yaml
# Training config
batch_size: 4-8 (tùy GPU memory)
learning_rate: 1e-4
max_epochs: 50-100
gradient_accumulation: 2
mixed_precision: fp16

# Data config
sample_rate: 24000
hop_length: 256
max_audio_length: 10s

# Speaker embedding
speaker_embedding_dim: 256
```

**Chiến lược Training**:
1. **Quick Training** (cho podcast 30 phút):
   - Epochs: 50-100 (không cần quá nhiều)
   - Early stopping: monitor validation loss
   - Checkpoint mỗi 10 epochs

2. **Multi-speaker Training**:
   - Train riêng cho từng speaker → các checkpoints độc lập
   - HOẶC multi-speaker model với speaker embeddings

### 3.3. Training Pipeline cho nhiều giọng
**Workflow**:
1. User upload file MP3 mới
2. Click "Thêm giọng mới" → nhập tên
3. Tự động xử lý pipeline PHASE 1 + 2
4. Click "Bắt đầu Training"
5. Progress bar hiển thị:
   - Data preprocessing: X%
   - Training: Epoch Y/Z, Loss: W
   - ETA: M phút

**Quản lý checkpoints**:
```
/content/models/
├── speaker_001/
│   ├── best_model.pth
│   ├── config.json
│   └── sample_audio.wav
├── speaker_002/
│   └── ...
```

### 3.4. Tối ưu Training Speed
**Cho file podcast 30 phút**:
- Expected segments: 200-300 clips (5-7s/clip)
- Training time ước tính: 2-4 giờ trên T4 GPU
- Tricks:
  - Mixed precision training (fp16)
  - Gradient checkpointing
  - DataLoader num_workers=2
  - Batch size optimal

---

## 📋 PHASE 4: Inference Interface (Giao diện sử dụng)

### 4.1. Gradio UI Components

#### Layout tổng thể:
```
┌─────────────────────────────────────────┐
│  HỆ THỐNG CLONE GIỌNG TIẾNG VIỆT        │
├─────────────────────────────────────────┤
│  [TAB 1: TRAINING]                      │
│  - Upload Audio                         │
│  - Xử lý & Training                     │
│                                         │
│  [TAB 2: TEXT-TO-SPEECH]               │
│  - Chọn giọng                          │
│  - Nhập text                           │
│  - Generate                             │
└─────────────────────────────────────────┘
```

#### TAB 1: Training Interface
**Components**:
1. **File Upload Area**:
   - `gr.File(file_count="multiple")` - upload nhiều file
   - Accept: .mp3, .wav
   - Display: danh sách files đã upload

2. **Speaker Management**:
   - `gr.Textbox()` - Nhập tên giọng
   - `gr.Button("Thêm giọng mới")`
   - `gr.Dropdown()` - Chọn giọng đang xử lý

3. **Processing Pipeline**:
   - `gr.Button("1. Tách giọng khỏi nhạc nền")`
   - `gr.Button("2. Chuẩn bị Dataset")`
   - `gr.Button("3. Bắt đầu Training")`
   - Progress bars cho mỗi bước

4. **Status Display**:
   - `gr.Textbox()` - Hiển thị logs
   - `gr.Plot()` - Training curves (loss)

#### TAB 2: Text-to-Speech Interface
**Components**:
1. **Speaker Selection**:
   - `gr.Radio()` - Chọn giọng
   - Auto-load available speakers từ `/content/models/`
   - Khi click → auto play sample audio

2. **Demo Audio Player**:
   - `gr.Audio()` - Phát sample của giọng được chọn
   - Auto-trigger khi đổi giọng

3. **Text Input**:
   - `gr.Textbox(lines=5)` - Nhập text tiếng Việt
   - Placeholder: "Nhập văn bản tiếng Việt cần chuyển thành giọng nói..."
   - Character counter: hiển thị độ dài

4. **Generation Settings**:
   - `gr.Slider()` - Speed (0.8 - 1.5x)
   - `gr.Slider()` - Temperature (creativity)
   - `gr.Checkbox()` - Enable/disable post-processing

5. **Generate Button**:
   - `gr.Button("🎙️ Tạo giọng nói")`
   - Processing indicator

6. **Output**:
   - `gr.Audio()` - Phát và download audio sinh ra
   - `gr.Button("💾 Lưu audio")`

### 4.2. Backend Functions

#### Function 1: `process_upload(audio_files, speaker_name)`
**Input**: List audio files, tên giọng
**Output**: Processed data ready for training
**Steps**:
1. Save files to `/content/uploads/{speaker_name}/`
2. Run Demucs separation
3. VAD segmentation
4. Whisper transcription
5. Save to dataset folder

#### Function 2: `train_speaker(speaker_name, epochs, batch_size)`
**Input**: Config training
**Output**: Trained model checkpoint
**Steps**:
1. Load dataset
2. Initialize F5-TTS model
3. Training loop với progress updates
4. Save best checkpoint

#### Function 3: `list_available_speakers()`
**Output**: List speakers đã train
**Logic**: Scan `/content/models/` folder

#### Function 4: `generate_speech(text, speaker_name, speed, temperature)`
**Input**: Text + config
**Output**: Audio file
**Steps**:
1. Load model checkpoint
2. Text preprocessing
3. F5-TTS inference
4. Post-processing
5. Return audio

#### Function 5: `play_speaker_demo(speaker_name)`
**Input**: Tên giọng
**Output**: Sample audio
**Logic**: Load `sample_audio.wav` từ model folder

---

## 📋 PHASE 5: Tối ưu cho Podcast 30 phút

### 5.1. Processing Pipeline Optimization

**Strategy 1: Chunked Processing**
```
30 phút podcast
↓
Chia thành 6 chunks × 5 phút
↓
Process parallel (nếu có multi-CPU)
↓
Merge results
```

**Strategy 2: Smart Caching**
- Cache kết quả Demucs separation
- Cache transcription results
- Reuse nếu process lại

**Strategy 3: Progressive Processing**
- Hiển thị progress real-time
- Cho phép dừng/tiếp tục
- Save intermediate results

### 5.2. Memory Management
**Vấn đề**: File 30 phút → ~90MB RAM

**Giải pháp**:
- Stream processing thay vì load toàn bộ
- Clear cache sau mỗi bước
- Garbage collection
- Monitor GPU memory

### 5.3. Quality vs Speed Tradeoff
**Fast Mode** (10-15 phút processing):
- Demucs with lower quality setting
- Skip enhancement
- Basic VAD

**High Quality Mode** (30-45 phút processing):
- Best Demucs model
- DeepFilterNet enhancement
- Careful segmentation
- Manual review option

---

## 📋 PHASE 6: Storage & Persistence

### 6.1. Google Drive Integration
**Mục tiêu**: Lưu models, datasets lâu dài

**Setup**:
```python
from google.colab import drive
drive.mount('/content/drive')

# Symlinks
/content/models → /content/drive/MyDrive/voice_cloning/models
/content/datasets → /content/drive/MyDrive/voice_cloning/datasets
```

### 6.2. Auto-save Strategy
- Auto-save checkpoints mỗi N epochs
- Save best model based on validation loss
- Backup config files

### 6.3. Export/Import Speakers
**Features**:
- Export speaker package (model + config + sample)
- Import speaker từ .zip file
- Share speakers giữa sessions

---

## 🚀 PHASE 7: User Experience Enhancements

### 7.1. Validation & Error Handling
**Pre-processing checks**:
- File format validation
- Audio quality check (sample rate, channels)
- Duration limits
- Warning nếu có nhiều background noise

**Training checks**:
- Minimum data requirements (ít nhất 100 segments)
- GPU availability
- Disk space

### 7.2. Helpful Features
1. **Tutorial Mode**: 
   - Guided walkthrough cho lần đầu
   - Example files để test

2. **Quality Metrics**:
   - Hiển thị data quality score
   - SNR của từng segment
   - Transcription confidence

3. **Comparison Tool**:
   - So sánh giọng gốc vs generated
   - A/B testing interface

4. **Batch Generation**:
   - Input multiple texts
   - Generate all với cùng giọng
   - Download as ZIP

---

## 📊 Timeline & Resource Estimation

### Time Estimates (per 30-min podcast):
1. **Upload**: 1-2 phút (tùy bandwidth)
2. **Voice Separation**: 5-10 phút (GPU)
3. **VAD + Segmentation**: 2-3 phút
4. **Transcription**: 3-5 phút (Whisper)
5. **Dataset Prep**: 2 phút
6. **Training**: 2-4 giờ (50-100 epochs)

**Total**: ~3-4 giờ từ upload đến có model sử dụng được

### Resource Requirements:
- **RAM**: 12-16GB (Colab Pro recommended)
- **GPU**: T4 minimum, V100/A100 optimal
- **Disk**: 5-10GB per speaker (raw + processed)
- **Runtime**: GPU runtime, High RAM

---

## 🎯 Phụ lục: Technical Stack Summary

### Core Libraries:
```
# Audio Processing
- demucs (separation)
- silero-vad (voice detection)
- librosa (audio manipulation)
- pydub (format conversion)

# Speech Recognition
- openai-whisper (transcription)

# Voice Cloning
- F5-TTS-Vietnamese (main model)

# UI
- gradio (interface)

# Utilities
- torch, torchaudio
- numpy, scipy
- pandas (metadata)
```

### Installation Priority:
1. Core audio libs (demucs, whisper)
2. F5-TTS repo + dependencies
3. Gradio UI
4. Enhancement tools (optional)

Kế hoạch này đảm bảo:
✅ Xử lý tốt podcast 30 phút có nhạc nền  
✅ Training nhanh và hiệu quả  
✅ Multi-speaker support  
✅ UI/UX thân thiện  
✅ Tối ưu cho Google Colab  
✅ Tương thích tốt với tiếng Việt