# 08 - Expansion Roadmap

> **Source:** Tổng hợp từ YEUCAU.md - Kế hoạch chi tiết hệ thống xử lý âm thanh và Clone giọng tiếng Việt

## 🎯 Vision

Xây dựng pipeline hoàn chỉnh trên Google Colab để:
1. Xử lý file podcast (loại nhạc nền, tách giọng)
2. Chuẩn bị dữ liệu huấn luyện tự động
3. Training model clone giọng với F5-TTS-Vietnamese
4. Tạo giao diện sử dụng đa giọng

---

## 📋 PHASE 1: Audio Preprocessing Pipeline

### 1.1 File Upload & Management

**Goal:** Upload nhiều file MP3/WAV, quản lý tập trung

**Features:**
```python
- Gradio FileUpload component (multi-file)
- Storage: /content/uploads/
- Metadata collection:
  * Tên file gốc
  * Tên giọng (user input)
  * Duration, sample rate
```

**UI Components:**
```python
upload_area = gr.File(
    file_count="multiple",
    file_types=[".mp3", ".wav"],
    label="Upload Audio Files"
)

speaker_name = gr.Textbox(
    label="Tên giọng",
    placeholder="Nhập tên người nói..."
)
```

### 1.2 Voice Separation (Tách giọng/nhạc)

**Problem:** Podcast 30 phút có nhạc nền

**Solution: Demucs (RECOMMENDED)**

```yaml
Tool: Demucs (Facebook Research)
Model: htdemucs hoặc htdemucs_ft

Why Demucs:
  - SOTA trong voice separation
  - Pretrained tốt với tiếng Việt
  - Xử lý nhanh trên GPU
  - Quality cao

Process:
  Input: podcast.mp3 (30 phút)
  ↓
  Demucs separation
  ↓
  Output: vocals.wav (giọng nói thuần)
```

**Implementation:**
```python
import demucs.separate

def separate_vocals(audio_path, output_dir):
    """
    Tách giọng nói khỏi nhạc nền
    """
    # Demucs command
    cmd = [
        "python", "-m", "demucs.separate",
        "-n", "htdemucs",  # Model name
        "--two-stems", "vocals",  # Only vocals
        "-o", output_dir,
        audio_path
    ]
    
    subprocess.run(cmd)
    
    vocals_path = f"{output_dir}/htdemucs/{basename(audio_path)}/vocals.wav"
    return vocals_path
```

**Optimization cho file dài:**
```python
# Chunk processing
def process_long_audio(audio_path, chunk_duration=600):  # 10 phút/chunk
    """
    Chia file 30 phút thành 3 chunks × 10 phút
    Process parallel nếu có multi-GPU
    """
    chunks = split_audio(audio_path, chunk_duration)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(separate_vocals, chunks)
    
    # Merge results
    final_vocals = concatenate_audio(results)
    return final_vocals
```

**Alternative: Spleeter**
```python
# Backup nếu Demucs quá chậm
from spleeter.separator import Separator

separator = Separator('spleeter:2stems')  # vocals/accompaniment
separator.separate_to_file(audio_path, output_dir)
```

### 1.3 Voice Activity Detection (VAD)

**Goal:** Loại bỏ đoạn im lặng, chỉ giữ speech segments

**Solution: Silero VAD (RECOMMENDED)**

```yaml
Tool: Silero VAD
Why: Tốt với tiếng Việt, fast, accurate

Process:
  1. Detect speech segments
  2. Loại bỏ silence > 0.5s
  3. Extract clean speech segments
  4. Lưu timestamps
```

**Implementation:**
```python
import torch
import torchaudio

# Load Silero VAD
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad'
)

(get_speech_timestamps, _, _, _, _) = utils

def detect_speech(audio_path):
    """
    Detect speech segments
    """
    wav, sr = torchaudio.load(audio_path)
    
    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(
        wav, 
        model,
        sampling_rate=sr,
        threshold=0.5,
        min_speech_duration_ms=500,
        min_silence_duration_ms=500
    )
    
    return speech_timestamps
```

### 1.4 Audio Quality Enhancement

**Goal:** Cải thiện chất lượng audio sau khi tách

**Tools:**

1. **DeepFilterNet** - Noise reduction
```python
from deepfilternet import DeepFilterNet

model = DeepFilterNet()
clean_audio = model.enhance(noisy_audio)
```

2. **Resemble Enhance** - Audio super-resolution
```python
from resemble_enhance import enhance_audio

enhanced = enhance_audio(
    audio_path,
    output_sr=24000,
    denoise=True
)
```

**Processing Pipeline:**
```python
def enhance_audio(audio_path):
    """
    Complete enhancement pipeline
    """
    # 1. Load audio
    audio, sr = librosa.load(audio_path, sr=24000)
    
    # 2. Noise reduction
    audio = denoise(audio)
    
    # 3. Normalize volume
    audio = librosa.util.normalize(audio)
    
    # 4. Resample to 24kHz (F5-TTS requirement)
    audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
    
    return audio
```

---

## 📋 PHASE 2: Automated Dataset Preparation

### 2.1 Audio Segmentation

**Goal:** Chia audio dài thành clips ngắn 3-10s

**Smart Segmentation:**
```python
def smart_segment(audio_path, speech_timestamps):
    """
    Chia audio theo VAD timestamps + sentence boundaries
    """
    segments = []
    
    for ts in speech_timestamps:
        start, end = ts['start'], ts['end']
        duration = end - start
        
        # Filter by duration
        if duration < 2 or duration > 12:
            continue
        
        # Check SNR
        segment = extract_segment(audio_path, start, end)
        snr = calculate_snr(segment)
        if snr < 20:
            continue
        
        segments.append({
            'path': save_segment(segment),
            'start': start,
            'end': end,
            'duration': duration,
            'snr': snr
        })
    
    return segments
```

**Quality Filtering:**
```python
def filter_segments(segments):
    """
    Loại bỏ segments không đạt chất lượng
    """
    filtered = []
    
    for seg in segments:
        # Duration check
        if seg['duration'] < 3 or seg['duration'] > 10:
            continue
        
        # SNR check
        if seg['snr'] < 20:
            continue
        
        # Music bleed-through check
        if detect_music_leak(seg['path']):
            continue
        
        filtered.append(seg)
    
    return filtered
```

### 2.2 Automatic Transcription

**Solution: Whisper Large-v3 (RECOMMENDED)**

```python
import whisper

model = whisper.load_model("large-v3")

def transcribe_audio(audio_path):
    """
    Transcribe tiếng Việt với Whisper
    """
    result = model.transcribe(
        audio_path,
        language="vi",  # Vietnamese
        task="transcribe",
        word_timestamps=True
    )
    
    return result['text']
```

**Batch Transcription:**
```python
def batch_transcribe(segments, batch_size=8):
    """
    Transcribe nhiều segments cùng lúc
    """
    transcriptions = []
    
    for i in tqdm(range(0, len(segments), batch_size)):
        batch = segments[i:i+batch_size]
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(transcribe_audio, batch)
        
        transcriptions.extend(results)
    
    return transcriptions
```

**Alternative: FPT.AI ASR / VAIS ASR**
```python
# Nếu cần accuracy cao hơn cho tiếng Việt
# API-based, cần internet

import requests

def fpt_transcribe(audio_path):
    """
    FPT.AI Speech-to-Text API
    """
    with open(audio_path, 'rb') as f:
        response = requests.post(
            'https://api.fpt.ai/hmi/asr/general',
            headers={'api-key': FPT_API_KEY},
            files={'file': f}
        )
    
    return response.json()['hypotheses'][0]['utterance']
```

### 2.3 Text Normalization

**Goal:** Chuẩn hóa text cho training

```python
def normalize_text(text):
    """
    Chuẩn hóa text tiếng Việt
    """
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove special characters (giữ dấu câu quan trọng)
    text = re.sub(r'[^a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ\s,.\!\?]', '', text)
    
    # 3. Normalize numbers
    text = num2words(text, lang='vi')  # 123 → một trăm hai mươi ba
    
    # 4. Handle abbreviations
    abbreviations = {
        'tp.': 'thành phố',
        'ths.': 'thạc sĩ',
        # ... more
    }
    for abbr, full in abbreviations.items():
        text = text.replace(abbr, full)
    
    # 5. Unicode normalization (NFD)
    text = unicodedata.normalize('NFD', text)
    
    # 6. Clean whitespace
    text = ' '.join(text.split())
    
    return text
```

### 2.4 Dataset Organization

**Output Structure:**
```
/content/datasets/
├── speaker_001/
│   ├── wavs/
│   │   ├── segment_0001.wav
│   │   ├── segment_0002.wav
│   │   └── ...
│   ├── metadata.csv
│   └── sample.wav  # Demo audio 5-10s
├── speaker_002/
│   └── ...
└── config.json
```

**metadata.csv Format:**
```csv
audio_path,text,speaker_name,duration,snr
wavs/segment_0001.wav,"xin chào các bạn",speaker_001,3.2,28.5
wavs/segment_0002.wav,"hôm nay trời đẹp",speaker_001,4.1,31.2
```

**config.json:**
```json
{
  "speakers": [
    {
      "id": "speaker_001",
      "name": "Nguyen Van A",
      "total_duration": 1800.5,
      "num_segments": 350,
      "sample_audio": "sample.wav"
    }
  ]
}
```

---

## 📋 PHASE 3: Multi-Speaker Training System

### 3.1 Training Configuration

**Hyperparameters cho podcast 30 phút:**
```yaml
Training:
  batch_size: 4000-7000 (tùy GPU)
  learning_rate: 1e-5
  epochs: 50-100  # Không cần quá nhiều
  gradient_accumulation: 2
  mixed_precision: fp16

Data:
  sample_rate: 24000
  hop_length: 256
  max_audio_length: 10s

Early Stopping:
  monitor: validation_loss
  patience: 10
  
Checkpointing:
  save_every: 10 epochs
  keep_best: 3
```

### 3.2 Training Pipeline cho nhiều giọng

**Workflow:**
```python
def train_new_speaker(audio_file, speaker_name):
    """
    Complete pipeline cho 1 giọng mới
    """
    # 1. Upload & save
    save_path = f"/content/uploads/{speaker_name}/"
    save_file(audio_file, save_path)
    
    # 2. Preprocessing
    vocals = separate_vocals(audio_file)
    segments = detect_and_segment(vocals)
    
    # 3. Transcription
    transcriptions = batch_transcribe(segments)
    
    # 4. Dataset preparation
    dataset_dir = prepare_dataset(segments, transcriptions, speaker_name)
    
    # 5. Training
    model = train_model(
        dataset_dir=dataset_dir,
        speaker_name=speaker_name,
        epochs=50
    )
    
    # 6. Save checkpoint
    save_checkpoint(model, f"ckpts/{speaker_name}/model_best.pt")
    
    return model
```

**Progress Tracking:**
```python
# UI components
progress_bar = gr.Progress()

def update_progress(stage, percentage):
    """
    Update training progress
    """
    stages = [
        "1. Tách giọng nói...",
        "2. Phát hiện đoạn nói...",
        "3. Transcription...",
        "4. Chuẩn bị dataset...",
        "5. Training..."
    ]
    
    progress_bar(percentage, desc=stages[stage])
```

### 3.3 Checkpoint Management

**Structure:**
```
/content/models/
├── speaker_001/
│   ├── best_model.pt
│   ├── config.json
│   ├── vocab.txt
│   └── sample_audio.wav
├── speaker_002/
│   └── ...
```

**Google Drive Integration:**
```python
from google.colab import drive

# Mount Drive
drive.mount('/content/drive')

# Symlinks
models_dir = "/content/drive/MyDrive/voice_cloning/models"
!ln -s {models_dir} /content/models
```

---

## 📋 PHASE 4: Production Interface

### 4.1 Gradio UI Layout

```python
with gr.Blocks() as app:
    gr.Markdown("# HỆ THỐNG CLONE GIỌNG TIẾNG VIỆT")
    
    with gr.Tabs():
        # TAB 1: TRAINING
        with gr.Tab("Training"):
            with gr.Row():
                # Upload section
                upload_files = gr.File(
                    file_count="multiple",
                    label="Upload Audio (MP3/WAV)"
                )
                speaker_name = gr.Textbox(
                    label="Tên giọng"
                )
            
            # Processing buttons
            with gr.Row():
                btn_separate = gr.Button("1. Tách giọng khỏi nhạc nền")
                btn_prepare = gr.Button("2. Chuẩn bị Dataset")
                btn_train = gr.Button("3. Bắt đầu Training")
            
            # Progress display
            progress_bar = gr.Progress()
            status_text = gr.Textbox(
                label="Status",
                lines=10,
                interactive=False
            )
            loss_plot = gr.Plot(label="Training Loss")
        
        # TAB 2: TEXT-TO-SPEECH
        with gr.Tab("Text-to-Speech"):
            # Speaker selection
            speaker_radio = gr.Radio(
                choices=list_available_speakers(),
                label="Chọn giọng"
            )
            
            # Demo audio
            demo_audio = gr.Audio(
                label="Demo giọng đã chọn",
                autoplay=True
            )
            
            # Text input
            gen_text = gr.Textbox(
                label="Nhập văn bản",
                lines=5,
                placeholder="Nhập văn bản tiếng Việt cần chuyển thành giọng nói..."
            )
            
            # Settings
            with gr.Accordion("Advanced Settings", open=False):
                speed_slider = gr.Slider(0.8, 1.5, 1.0, label="Speed")
                temperature = gr.Slider(0.1, 1.0, 0.7, label="Temperature")
                remove_silence = gr.Checkbox(label="Remove Silence")
            
            # Generate
            generate_btn = gr.Button("🎙️ Tạo giọng nói", variant="primary")
            output_audio = gr.Audio(label="Audio Output")
            download_btn = gr.Button("💾 Lưu audio")
```

### 4.2 Backend Functions

```python
def process_upload(audio_files, speaker_name):
    """
    Function 1: Process uploaded files
    """
    # Save files
    save_dir = f"/content/uploads/{speaker_name}/"
    os.makedirs(save_dir, exist_ok=True)
    
    for audio in audio_files:
        shutil.copy(audio, save_dir)
    
    # Run Demucs
    vocals = separate_vocals(audio_files[0])
    
    # VAD segmentation
    segments = segment_audio(vocals)
    
    # Whisper transcription
    transcriptions = transcribe_batch(segments)
    
    # Save to dataset folder
    dataset_dir = organize_dataset(
        segments, 
        transcriptions,
        speaker_name
    )
    
    return f"✅ Processed {len(segments)} segments"

def train_speaker(speaker_name, epochs, batch_size):
    """
    Function 2: Train model
    """
    # Load dataset
    dataset_dir = f"/content/datasets/{speaker_name}"
    
    # Initialize model
    model = initialize_f5tts()
    
    # Training loop
    for epoch in range(epochs):
        loss = train_epoch(model, dataset_dir, batch_size)
        yield f"Epoch {epoch}/{epochs} | Loss: {loss:.4f}"
        
        # Save checkpoint
        if epoch % 10 == 0:
            save_checkpoint(model, f"model_epoch_{epoch}.pt")
    
    # Save final
    save_checkpoint(model, "model_best.pt")
    return "✅ Training completed!"

def list_available_speakers():
    """
    Function 3: List trained speakers
    """
    models_dir = "/content/models/"
    speakers = [d for d in os.listdir(models_dir) if os.path.isdir(f"{models_dir}/{d}")]
    return speakers

def generate_speech(text, speaker_name, speed, temperature):
    """
    Function 4: Generate speech
    """
    # Load model
    model_path = f"/content/models/{speaker_name}/model_best.pt"
    model = load_model(model_path)
    
    # Text preprocessing
    text = normalize_text(text)
    
    # Inference
    audio = model.infer(
        gen_text=text,
        speed=speed,
        temperature=temperature
    )
    
    # Post-processing
    audio = postprocess(audio)
    
    return audio

def play_speaker_demo(speaker_name):
    """
    Function 5: Play demo audio
    """
    demo_path = f"/content/models/{speaker_name}/sample_audio.wav"
    return demo_path
```

---

## 📋 PHASE 5-7: Optimization & Production

### Phase 5: Podcast Optimization

**Chunked Processing:**
```python
def process_long_podcast(audio_path, chunk_duration=300):
    """
    30 phút → 6 chunks × 5 phút
    """
    chunks = split_audio(audio_path, chunk_duration)
    
    # Process parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_chunk, chunks))
    
    # Merge
    final_result = merge_results(results)
    return final_result
```

**Smart Caching:**
```python
import joblib

@joblib.Memory(location='/tmp/cache').cache
def separate_vocals_cached(audio_path):
    """
    Cache Demucs results
    """
    return separate_vocals(audio_path)
```

### Phase 6: Storage & Persistence

**Auto-save Strategy:**
```python
def auto_save_checkpoint(model, epoch, loss):
    """
    Save best model based on validation loss
    """
    if loss < best_loss:
        save_checkpoint(model, "model_best.pt")
        
        # Backup to Drive
        shutil.copy(
            "model_best.pt",
            "/content/drive/MyDrive/voice_cloning/backups/"
        )
```

### Phase 7: UX Enhancements

**Validation & Error Handling:**
```python
def validate_upload(audio_file):
    """
    Pre-processing checks
    """
    # Format check
    if not audio_file.endswith(('.wav', '.mp3')):
        raise ValueError("Chỉ hỗ trợ WAV và MP3")
    
    # Duration check
    duration = get_duration(audio_file)
    if duration < 60:
        raise ValueError("Audio quá ngắn (< 1 phút)")
    if duration > 3600:
        raise ValueError("Audio quá dài (> 1 giờ). Vui lòng chia nhỏ.")
    
    # Quality check
    sr = get_sample_rate(audio_file)
    if sr < 16000:
        raise ValueError("Sample rate quá thấp (< 16kHz)")
```

---

## 📊 Timeline & Resource Estimation

### Time Estimates (per 30-min podcast):

```yaml
1. Upload: 1-2 phút
2. Voice Separation (Demucs): 5-10 phút (GPU)
3. VAD + Segmentation: 2-3 phút
4. Transcription (Whisper): 3-5 phút
5. Dataset Prep: 2 phút
6. Training (50 epochs): 2-4 giờ

Total: ~3-4 giờ từ upload đến model ready
```

### Resource Requirements:

```yaml
RAM: 12-16GB (Colab Pro recommended)
GPU: T4 minimum, V100/A100 optimal
Disk: 5-10GB per speaker
Runtime: GPU runtime, High RAM
```

---

**Prev:** [`07-TECHNICAL-SPECS.md`](07-TECHNICAL-SPECS.md)  
**Next:** [`09-IMPLEMENTATION-GUIDE.md`](09-IMPLEMENTATION-GUIDE.md)



