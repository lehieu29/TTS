# 05 - Inference Pipeline

## 🎤 Inference Overview

Inference là quá trình sử dụng model đã train để tạo giọng nói từ text.

---

## 🔄 Inference Flow

```
Reference Audio + Text
    ↓
Preprocessing
    ↓
Speaker Embedding Extraction
    ↓
Text Encoding
    ↓
Flow Matching Generation
    ↓
Mel-Spectrogram
    ↓
Vocoder (Vocos)
    ↓
Audio Output
```

---

## 🛠️ Inference Methods

### Method 1: CLI (Command Line)

#### Basic Usage
```bash
f5-tts_infer-cli \
--model "F5TTS_Base" \
--ref_audio ref.wav \
--ref_text "cả hai bên hãy cố gắng hiểu cho nhau" \
--gen_text "xin chào, tôi là trợ lý ảo tiếng Việt" \
--speed 1.0
```

#### With Custom Model
```bash
f5-tts_infer-cli \
--model "F5TTS_Base" \
--ref_audio ref.wav \
--ref_text "xin chào các bạn" \
--gen_text "hôm nay trời đẹp quá" \
--speed 1.0 \
--vocoder_name vocos \
--vocab_file data/your_training_dataset/vocab.txt \
--ckpt_file ckpts/your_training_dataset/model_last.pt
```

#### All Parameters
```bash
f5-tts_infer-cli \
--model "F5TTS_Base"              # Model architecture
--ref_audio ref.wav               # Reference audio (giọng mẫu)
--ref_text "text"                 # Text của ref_audio (optional)
--gen_text "text to generate"    # Text muốn tạo giọng
--gen_file output.wav             # Output file path (optional)
--remove_silence                  # Remove silence (optional)
--output_dir "outputs"            # Output directory
--output_format "wav"             # Output format (wav/mp3/flac)
--speed 1.0                       # Speed (0.3-2.0)
--cross_fade_duration 0.15        # Cross-fade (seconds)
--nfe_step 32                     # NFE steps (quality vs speed)
--sway_sampling_coef -1.0         # Sampling coefficient
--cfg_strength 2.0                # CFG strength
--fix_duration None               # Fix duration (seconds)
--vocoder_name vocos              # Vocoder (vocos/bigvgan)
--vocab_file path/to/vocab.txt    # Custom vocab
--ckpt_file path/to/model.pt      # Custom checkpoint
```

### Method 2: Gradio Web UI

#### Launch
```bash
f5-tts_infer-gradio
# Hoặc
python src/f5_tts/infer/infer_gradio.py
```

#### Access
```
http://localhost:7860
```

#### UI Features

**Tab 1: Basic-TTS**
- Upload reference audio
- Nhập reference text (optional - auto-transcribe với Whisper)
- Nhập text muốn tạo
- Advanced settings:
  - Speed slider (0.3-2.0)
  - NFE steps (4-64)
  - Cross-fade duration
  - Remove silence toggle
- Generate button
- Audio player + Spectrogram visualization

**Tab 2: Multi-Speech**
- Upload multiple speech types/speakers
- Format: `{Speaker1} text here {Speaker2} more text`
- Dynamic speech type addition
- Batch generation

**Tab 3: Voice-Chat**
- AI chat với voice output
- Reference audio cho voice
- Microphone input
- Real-time TTS response

### Method 3: Python API

```python
from f5_tts.api import F5TTS

# Initialize
f5tts = F5TTS(
    model_type="F5-TTS",  # or "E2-TTS"
    ckpt_file="path/to/model.pt",
    vocab_file="path/to/vocab.txt"
)

# Generate
audio, sample_rate, spectrogram = f5tts.infer(
    ref_file="ref.wav",
    ref_text="xin chào",
    gen_text="hôm nay trời đẹp",
    speed=1.0
)

# Save
import soundfile as sf
sf.write("output.wav", audio, sample_rate)
```

---

## 🎛️ Key Parameters Explained

### ref_audio (Reference Audio)
**Purpose:** Giọng mẫu để model clone

**Requirements:**
- **Duration:** 3-15 giây (optimal: 5-10s)
- **Quality:** Rõ ràng, ít noise
- **Content:** Giọng nói liên tục, không im lặng nhiều
- **Format:** WAV, MP3, FLAC

**Tips:**
```python
# Good reference:
✅ Clean speech, single speaker
✅ 5-10 seconds long
✅ Natural prosody
✅ Consistent volume

# Bad reference:
❌ Multiple speakers
❌ Background music/noise
❌ Too short (<3s) or too long (>15s)
❌ Lots of pauses
```

### ref_text (Reference Text)
**Purpose:** Text tương ứng với ref_audio

**Options:**
1. **Provide manually** (recommended)
   ```bash
   --ref_text "xin chào các bạn"
   ```

2. **Auto-transcribe** (nếu không cung cấp)
   ```bash
   # Không set ref_text
   # → Model tự động dùng Whisper để transcribe
   # → Có thể không chính xác 100%
   ```

**Why it matters:**
- Model cần biết ref_audio nói gì
- Sai ref_text → quality giảm
- Auto-transcribe OK cho tiếng Anh/Trung
- Tiếng Việt nên provide manually

### gen_text (Generation Text)
**Purpose:** Text bạn muốn model nói

**Formatting:**
```python
# Short text
gen_text = "xin chào"

# Long text - tự động chia chunks
gen_text = """
Hôm nay trời đẹp quá. 
Tôi muốn đi chơi. 
Bạn có rảnh không?
"""

# With punctuation
gen_text = "Xin chào! Bạn khỏe không?"
```

**Tips:**
- Dùng dấu câu đúng → prosody tốt hơn
- Text dài → tự động chia chunks
- Lowercase vs Uppercase: không ảnh hưởng nhiều

### speed
**Purpose:** Điều chỉnh tốc độ nói

```python
speed = 0.5   # Rất chậm
speed = 0.8   # Chậm
speed = 1.0   # Bình thường (default)
speed = 1.2   # Nhanh
speed = 1.5   # Rất nhanh
speed = 2.0   # Maximum
```

### nfe_step (Number of Function Evaluations)
**Purpose:** Số bước sampling trong flow matching

**Trade-off: Quality vs Speed**
```python
nfe_step = 8    # Nhanh nhất, quality thấp
nfe_step = 16   # Nhanh, quality OK
nfe_step = 32   # Default - balanced
nfe_step = 64   # Chậm, quality cao nhất
```

**Recommendations:**
- Development/testing: 16
- Production: 32
- High quality: 64

### cross_fade_duration
**Purpose:** Thời gian cross-fade giữa các chunks

```python
cross_fade_duration = 0.0    # No cross-fade
cross_fade_duration = 0.15   # Default
cross_fade_duration = 0.5    # Smooth transitions
```

**When to use:**
- Text dài được chia thành chunks
- Tránh "click" sound giữa chunks

### remove_silence
**Purpose:** Loại bỏ silence trong output

```python
remove_silence = False  # Default - giữ nguyên
remove_silence = True   # Remove silence
```

**Note:**
- Model có xu hướng tạo silence dài
- remove_silence giúp output ngắn gọn hơn
- Có thể gây artifacts

---

## 🔧 Advanced Inference

### Long Text Generation

**Problem:** Text dài (>100 từ) khó generate một lượt

**Solution:** Auto-chunking

```python
def chunk_text(text, max_chars=135):
    """
    Chia text thành chunks nhỏ
    """
    sentences = text.split('. ')
    chunks = []
    buffer = []
    
    for sentence in sentences:
        buffer.append(sentence)
        if len(' '.join(buffer)) > max_chars:
            chunks.append('. '.join(buffer))
            buffer = []
    
    if buffer:
        chunks.append('. '.join(buffer))
    
    return chunks

# Generate từng chunk
for chunk in chunk_text(long_text):
    audio_chunk = model.infer(ref_audio, ref_text, chunk)
    audio_segments.append(audio_chunk)

# Concatenate với cross-fade
final_audio = concatenate_with_crossfade(
    audio_segments, 
    cross_fade_duration=0.15
)
```

### Multi-Speaker Generation

**Use case:** Tạo audio với nhiều giọng khác nhau

```python
speakers = {
    "Alice": {
        "ref_audio": "alice_ref.wav",
        "ref_text": "Hello, I'm Alice"
    },
    "Bob": {
        "ref_audio": "bob_ref.wav", 
        "ref_text": "Hi, I'm Bob"
    }
}

script = [
    {"speaker": "Alice", "text": "How are you today?"},
    {"speaker": "Bob", "text": "I'm doing great, thanks!"},
]

audio_segments = []
for line in script:
    speaker_info = speakers[line["speaker"]]
    audio = model.infer(
        ref_file=speaker_info["ref_audio"],
        ref_text=speaker_info["ref_text"],
        gen_text=line["text"]
    )
    audio_segments.append(audio)

# Merge
final_audio = concatenate(audio_segments)
```

### Voice Conversion

**Use case:** Convert giọng A sang giọng B

```python
# Source audio
source_audio = "source.wav"
source_text = transcribe(source_audio)  # Whisper

# Target voice
target_ref = "target_ref.wav"
target_ref_text = "sample text"

# Convert
converted_audio = model.infer(
    ref_file=target_ref,
    ref_text=target_ref_text,
    gen_text=source_text
)
```

---

## 📊 Performance Optimization

### GPU Inference
```python
# Sử dụng float16 cho faster inference
model.to("cuda").half()

# Batch inference (nếu có nhiều texts)
batch_results = model.batch_infer(
    ref_files=[ref_audio] * N,
    ref_texts=[ref_text] * N,
    gen_texts=text_list
)
```

### CPU Inference
```python
# Chậm hơn nhưng vẫn work
model.to("cpu")

# Tối ưu:
import torch
torch.set_num_threads(8)  # Use multiple CPU cores
```

### Caching
```python
# Cache speaker embedding
speaker_embed = extract_speaker_embedding(ref_audio)

# Reuse cho multiple generations
for text in text_list:
    audio = model.infer_with_embed(
        speaker_embed=speaker_embed,
        gen_text=text
    )
```

---

## 🎨 Post-Processing

### Normalize Volume
```python
import soundfile as sf
import numpy as np

audio, sr = sf.read("output.wav")

# Normalize to -3dB
audio = audio / np.max(np.abs(audio)) * 0.7

sf.write("output_normalized.wav", audio, sr)
```

### Remove Silence
```python
from f5_tts.infer.utils_infer import remove_silence_for_generated_wav

remove_silence_for_generated_wav("output.wav")
```

### Format Conversion
```python
# WAV to MP3
from pydub import AudioSegment

audio = AudioSegment.from_wav("output.wav")
audio.export("output.mp3", format="mp3", bitrate="192k")
```

---

## 🐛 Common Issues

### Issue: Output có nhiều silence
**Solution:**
```bash
--remove_silence
# Hoặc post-process manually
```

### Issue: Giọng không giống reference
**Causes:**
1. Reference audio quality kém
2. Reference audio quá ngắn (<5s)
3. Reference text không chính xác

**Solutions:**
- Dùng reference 5-10s
- Provide ref_text manually
- Chọn reference rõ ràng, ít noise

### Issue: Output có artifacts/glitches
**Causes:**
1. NFE steps quá thấp
2. Model chưa train tốt
3. Text quá dài

**Solutions:**
```bash
--nfe_step 64  # Tăng quality
# Hoặc chia text thành chunks nhỏ hơn
```

### Issue: Tiếng Việt phát âm sai
**Causes:**
1. Model chưa train với dữ liệu tiếng Việt đủ
2. Vocab không đầy đủ
3. Text có ký tự lạ

**Solutions:**
- Fine-tune với dữ liệu tiếng Việt
- Check vocab.txt có đầy đủ ký tự không
- Normalize text (lowercase, remove special chars)

---

## 💡 Best Practices

### 1. Reference Audio Selection
```python
✅ DO:
- Chọn audio rõ ràng, giọng đơn
- Duration: 5-10 giây
- Natural prosody
- Consistent volume

❌ DON'T:
- Nhiều speaker
- Background noise/music
- Quá ngắn hoặc quá dài
- Im lặng nhiều
```

### 2. Text Formatting
```python
✅ DO:
gen_text = "Xin chào! Bạn khỏe không?"  # Có dấu câu

❌ DON'T:
gen_text = "xin chao ban khoe khong"   # Thiếu dấu
```

### 3. Quality vs Speed
```python
# Development
nfe_step = 16, speed = 1.2  # Fast iteration

# Production
nfe_step = 32, speed = 1.0  # Balanced

# High Quality
nfe_step = 64, speed = 1.0  # Best quality
```

---

**Prev:** [`04-TRAINING-PIPELINE.md`](04-TRAINING-PIPELINE.md)  
**Next:** [`06-DATA-REQUIREMENTS.md`](06-DATA-REQUIREMENTS.md)



