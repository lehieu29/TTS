# 06 - Data Requirements

## 📊 Dataset Specifications

Chất lượng và quy mô dữ liệu quyết định chất lượng model.

---

## 🎯 Quick Reference

| Use Case | Duration | Files | Quality | Transcription |
|----------|----------|-------|---------|---------------|
| Testing | 5-10 phút | 50-100 | OK | Acceptable |
| Single Voice | 1-10 giờ | 500-1000 | Good | Important |
| Good Voice Clone | 50-100 giờ | 5k-10k | High | Critical |
| Multi-Speaker | 1000+ giờ | 100k+ | High | Critical |

---

## 📁 Data Format Requirements

### Audio Files

#### Format
```python
Format: WAV (recommended), MP3, FLAC
Sample Rate: 24000 Hz (24kHz)
Channels: 1 (Mono)
Bit Depth: 16-bit hoặc 32-bit float
```

#### Duration per File
```python
Minimum: 1 giây
Maximum: 30 giây
Optimal: 5-7 giây

# Why?
- Quá ngắn (<2s): Không đủ context
- Quá dài (>15s): Khó học patterns
- 5-7s: Sweet spot cho TTS
```

#### Audio Quality
```python
Signal-to-Noise Ratio (SNR): >20dB
Background Noise: Minimal
Music/Sound Effects: None
Clipping/Distortion: None
Sample Rate: Consistent 24kHz
```

### Text Files

#### Format
```python
File: UTF-8 plain text (.txt)
Naming: Same as audio file
# audio_001.wav → audio_001.txt
```

#### Content
```python
Language: Tiếng Việt (Vietnamese)
Casing: Lowercase recommended
Diacritics: Full Vietnamese diacritics required
Punctuation: Include (helps prosody)
Numbers: Can be digits or words
```

#### Example
```txt
# audio_001.txt
xin chào các bạn, hôm nay tôi sẽ nói về trí tuệ nhân tạo.

# audio_002.txt  
việt nam là một đất nước xinh đẹp với 54 dân tộc anh em.
```

---

## 📏 Dataset Size Guidelines

### Minimum Viable Dataset
```python
Duration: 10 phút - 1 giờ
Files: 100-500 files
Purpose: Testing, proof of concept
Expected Quality: Basic, testing only
Training Time: 30 phút - 2 giờ

Limitations:
- Giọng có thể không ổn định
- Phát âm một số từ sai
- Prosody không tự nhiên
```

### Single Speaker Clone
```python
Duration: 5-10 giờ
Files: 500-2000 files
Purpose: Clone giọng cụ thể (e.g., podcast host)
Expected Quality: Good for that specific voice
Training Time: 4-12 giờ

Characteristics:
- Giọng ổn định với speaker đó
- Phát âm chính xác
- Natural prosody
- Có thể generalize cho text mới
```

### Production Quality (Single Voice)
```python
Duration: 50-100 giờ
Files: 5000-10000+ files
Purpose: High-quality single voice TTS
Expected Quality: Excellent
Training Time: 2-4 ngày

Characteristics:
- Rất giống giọng gốc
- Phát âm chuẩn
- Natural prosody và emotion
- Robust với text mới
```

### Multi-Speaker System
```python
Duration: 1000+ giờ
Files: 100000+ files
Speakers: 100+ speakers
Purpose: Universal Vietnamese TTS
Expected Quality: Excellent voice cloning
Training Time: 1-2 tuần

Characteristics:
- Zero-shot voice cloning
- Generalize tốt cho giọng mới
- Robust với diverse texts
- Professional quality
```

---

## 🎨 Data Quality Criteria

### Audio Quality Checklist

```python
✅ Clear Speech
- Single speaker per file
- Consistent volume
- Natural speaking pace
- No overlapping speech

✅ Clean Recording
- No background music
- No sound effects
- No noise (AC, fan, traffic)
- No echo/reverb
- No clipping/distortion

✅ Technical Specs
- 24kHz sample rate
- Mono channel
- 16-bit or 32-bit float
- Proper normalization

✅ Content Quality
- Complete sentences
- Natural prosody
- No heavy accent (unless desired)
- Consistent style
```

### Transcription Quality Checklist

```python
✅ Accuracy
- 100% accurate transcription
- Every word must match audio exactly
- Include all filler words if present

✅ Vietnamese Diacritics
- Full diacritics: á à ả ã ạ
- Đ (d with stroke)
- Special vowels: ă â ê ô ơ ư
# Wrong: xin chao
# Right: xin chào

✅ Punctuation
- Use proper punctuation
- Helps model learn prosody
- Comma, period, question mark, exclamation

✅ Numbers & Abbreviations
- Can use digits: 123
- Or spell out: một trăm hai mươi ba
- Abbreviations: expand or keep (consistent)
```

---

## 🗂️ Dataset Organization

### Recommended Structure

```
data/your_dataset/
├── audio_0001.wav
├── audio_0001.txt
├── audio_0002.wav
├── audio_0002.txt
├── audio_0003.wav
├── audio_0003.txt
└── ...

# After processing → becomes:
data/your_training_dataset/
├── wavs/
│   ├── audio_0001.wav
│   ├── audio_0002.wav
│   └── ...
├── metadata.csv
├── vocab.txt
├── raw.arrow
└── duration.json
```

### Multi-Speaker Structure

```
data/multi_speaker/
├── speaker_001/
│   ├── audio_001.wav
│   ├── audio_001.txt
│   └── ...
├── speaker_002/
│   ├── audio_001.wav
│   ├── audio_001.txt
│   └── ...
└── speaker_NNN/
    └── ...

# Metadata includes speaker_id
speaker_001|wavs/speaker_001_audio_001.wav|xin chào
speaker_002|wavs/speaker_002_audio_001.wav|hôm nay trời đẹp
```

---

## 🎤 Data Collection Methods

### Method 1: Professional Recording
```python
Pros:
- Highest quality
- Controlled environment
- Consistent

Cons:
- Expensive
- Time-consuming

Tools:
- Professional microphone
- Soundproof booth
- Audio interface
- DAW software
```

### Method 2: Podcast/YouTube Audio
```python
Pros:
- Large amount of data
- Natural speech
- Free/available

Cons:
- May have background music
- Need separation
- Need transcription

Pipeline:
1. Download audio
2. Music separation (Demucs)
3. Voice Activity Detection
4. Transcription (Whisper)
5. Quality filtering
```

### Method 3: Audiobook Data
```python
Pros:
- Clean audio
- Have text available
- Long duration

Cons:
- Copyright issues
- May be read-style (not natural)

Sources:
- LibriVox (public domain)
- Self-recorded
```

### Method 4: Crowdsourcing
```python
Pros:
- Scalable
- Multi-speaker data
- Cost-effective

Cons:
- Quality varies
- Need QA process

Platforms:
- Custom web interface
- Mobile app
- Recording instructions
```

---

## 🔍 Data Filtering Guidelines

### Automatic Filtering

```python
# Duration filter
if duration < 1.0 or duration > 30.0:
    reject()

# SNR filter (if available)
if SNR < 20:
    reject()

# Text length filter
if len(text.split()) < 3:
    reject()

# Sample rate check
if sample_rate != 24000:
    resample_or_reject()

# Silence ratio
silence_ratio = detect_silence(audio)
if silence_ratio > 0.5:  # >50% silence
    reject()
```

### Manual Quality Check

```python
# Sample random files
sample_size = min(100, len(dataset) * 0.01)  # 1% or 100 files
sample_files = random.sample(all_files, sample_size)

# Check for:
for audio_file, text_file in sample_files:
    # 1. Audio quality
    ✅ Clear voice?
    ✅ No background noise?
    ✅ Proper volume?
    
    # 2. Transcription accuracy
    ✅ Text matches audio?
    ✅ Full diacritics?
    ✅ Proper punctuation?
    
    # 3. Content quality
    ✅ Natural prosody?
    ✅ Complete sentences?
    ✅ Consistent style?
```

---

## 📊 Dataset Statistics

### Key Metrics to Track

```python
# Duration distribution
Total Duration: 100.5 hours
Min Duration: 1.2s
Max Duration: 28.5s
Mean Duration: 6.3s
Median Duration: 5.8s

# File count
Total Files: 57,345
Valid Files: 56,890 (99.2%)
Rejected: 455 (0.8%)

# Vocabulary
Unique Characters: 87
Unique Words: 12,450
OOV Rate: 0.3%

# Quality metrics
Mean SNR: 28.5 dB
Files with SNR > 20dB: 98.5%
Transcription Accuracy: 99.8%
```

### Distribution Plots

```python
import matplotlib.pyplot as plt

# Duration histogram
plt.hist(durations, bins=50)
plt.xlabel("Duration (seconds)")
plt.ylabel("Count")
plt.title("Audio Duration Distribution")

# Word frequency
top_words = Counter(all_words).most_common(50)
plt.bar(words, counts)
plt.title("Top 50 Words")
```

---

## 🚨 Common Data Issues

### Issue 1: Background Music
```python
Problem: Podcast có nhạc nền

Solution:
1. Use Demucs for source separation
2. Extract vocals only
3. Quality check separated audio

Tools:
- demucs (Facebook Research)
- spleeter (Deezer)
```

### Issue 2: Multiple Speakers
```python
Problem: Conversation/interview với nhiều người

Solution:
1. Speaker diarization
2. Segment by speaker
3. Label speaker IDs
4. Train multi-speaker model

Tools:
- pyannote.audio
- resemblyzer
```

### Issue 3: Transcription Errors
```python
Problem: ASR không chính xác 100%

Solution:
1. Use best ASR model (Whisper large-v3)
2. Manual correction for critical data
3. Quality check randomly
4. Use confidence scores

Priority:
- High confidence → auto accept
- Medium confidence → review
- Low confidence → manual transcribe
```

### Issue 4: Inconsistent Quality
```python
Problem: Audio quality khác nhau giữa các files

Solution:
1. Normalize volume across dataset
2. Apply same preprocessing
3. Filter low quality files
4. Consistent sample rate

Pipeline:
audio → normalize → resample → denoise → check_quality
```

---

## 💡 Best Practices

### 1. Start Small, Scale Up
```python
# Phase 1: Test (1 giờ)
- Verify pipeline works
- Check quality
- Iterate quickly

# Phase 2: Expand (10 giờ)
- Scale up collection
- Refine process
- Evaluate quality

# Phase 3: Production (100+ giờ)
- Full dataset
- Final training
- Deploy model
```

### 2. Quality > Quantity
```python
10 giờ clean data > 100 giờ noisy data

Priorities:
1. Accurate transcription
2. Clean audio (no music/noise)
3. Natural speech
4. Consistent quality
```

### 3. Diverse Content
```python
✅ DO collect:
- Different topics
- Different speaking styles
- Different sentence structures
- Various vocabulary

❌ DON'T:
- Only one topic
- Repetitive content
- Same sentences
- Limited vocabulary
```

### 4. Version Control for Data
```python
data/
├── v1.0/  # Initial dataset
├── v1.1/  # Fixed transcriptions
├── v2.0/  # Added more data
└── latest → v2.0

# Track changes
- CHANGELOG.md
- Data statistics
- Known issues
```

---

**Prev:** [`05-INFERENCE-PIPELINE.md`](05-INFERENCE-PIPELINE.md)  
**Next:** [`07-TECHNICAL-SPECS.md`](07-TECHNICAL-SPECS.md)



