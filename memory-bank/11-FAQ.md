# 11 - FAQ (Frequently Asked Questions)

## ❓ General Questions

### Q: F5-TTS là gì?

**A:** F5-TTS (Flow Matching Text-to-Speech) là một architecture tiên tiến cho Text-to-Speech, sử dụng Flow Matching thay vì traditional diffusion. Nó cho phép zero-shot voice cloning chất lượng cao.

### Q: Tại sao chọn F5-TTS thay vì các TTS models khác?

**A:** 
- ✅ **Zero-shot voice cloning:** Chỉ cần 5-10s reference audio
- ✅ **High quality:** Naturalness và similarity cao
- ✅ **Fast inference:** Flow matching nhanh hơn diffusion
- ✅ **Open source:** MIT license, có thể customize
- ✅ **Vietnamese support:** Dễ dàng fine-tune cho tiếng Việt

### Q: Dự án này khác gì với F5-TTS gốc?

**A:**
- ✅ **Optimized cho tiếng Việt:** Vocabulary, text processing
- ✅ **Complete pipeline:** Từ raw audio → trained model
- ✅ **Vietnamese documentation:** Hướng dẫn tiếng Việt
- ✅ **Expansion roadmap:** Audio preprocessing, multi-speaker UI

---

## 📊 Data Questions

### Q: Cần bao nhiêu dữ liệu để train model?

**A:**
```yaml
Testing: 10 phút - 1 giờ
  Purpose: Verify pipeline
  Quality: Basic

Single Voice: 5-10 giờ
  Purpose: Clone giọng cụ thể
  Quality: Good

Production: 50-100 giờ
  Purpose: High-quality single voice
  Quality: Excellent

Multi-Speaker: 1000+ giờ
  Purpose: Universal TTS system
  Quality: State-of-the-art
```

### Q: Dữ liệu cần format gì?

**A:**
```yaml
Audio:
  Format: WAV (preferred), MP3, FLAC
  Sample Rate: 24kHz
  Channels: Mono
  Duration: 3-10s per file (optimal)
  
Text:
  Format: UTF-8 .txt files
  Content: Exact transcription
  Diacritics: Full Vietnamese diacritics required
  Punctuation: Include for better prosody
```

### Q: Có thể dùng podcast/YouTube audio không?

**A:** **Có**, nhưng cần preprocessing:
1. **Voice separation** (Demucs) để tách giọng/nhạc nền
2. **Voice Activity Detection** để detect speech segments
3. **Transcription** (Whisper) để tạo text
4. **Quality filtering** để loại bỏ bad samples

→ Xem [08-EXPANSION-ROADMAP.md](08-EXPANSION-ROADMAP.md) để implement.

### Q: Làm sao transcribe audio nhanh?

**A:**
```python
# Option 1: Whisper large-v3 (recommended)
import whisper
model = whisper.load_model("large-v3")
result = model.transcribe(audio_path, language="vi")

# Option 2: FPT.AI ASR (higher accuracy, paid)
# API-based

# Option 3: Manual (highest accuracy, slow)
# Sử dụng Transcribe tool + human review
```

---

## 🎓 Training Questions

### Q: Training mất bao lâu?

**A:**
```yaml
GPU: T4 (Google Colab Free)
  10 phút data: ~30-60 phút
  1 giờ data: ~2-4 giờ
  10 giờ data: ~1-2 ngày
  
GPU: V100
  10 phút data: ~15-30 phút
  1 giờ data: ~1-2 giờ
  10 giờ data: ~12-24 giờ
  
GPU: A100
  10 phút data: ~10-20 phút
  1 giờ data: ~40-80 phút
  10 giờ data: ~8-16 giờ
```

### Q: GPU nào tốt nhất?

**A:**
```yaml
Budget:
  T4 (16GB): OK cho testing và small datasets
  RTX 3060 (12GB): Good cho home/small projects
  
Recommended:
  RTX 3090 (24GB): Best price/performance
  RTX 4090 (24GB): Fastest consumer GPU
  
Professional:
  V100 (32GB): Cloud standard
  A100 (40GB/80GB): Best for large-scale training
```

### Q: Có thể train trên CPU không?

**A:** **Technically có**, nhưng **KHÔNG khuyến nghị**:
- ⚠️ Rất chậm (10-100x chậm hơn GPU)
- ⚠️ Chỉ practical cho dataset < 10 phút
- ✅ OK cho testing code/debugging

### Q: Có thể train trên Google Colab Free không?

**A:** **Có**, nhưng có limitations:
- ✅ T4 GPU (16GB) - đủ để train
- ⚠️ 12-hour runtime limit → Phải save checkpoints thường xuyên
- ⚠️ Disk space limited → Clean up thường xuyên
- 💡 **Recommendation:** Colab Pro ($10/month) để có V100/A100 và longer runtime

### Q: Làm sao biết model đã train tốt?

**A:** Check các metrics:
```yaml
Loss:
  - Giảm đều qua epochs
  - Converge về < 0.5
  
Audio Quality:
  - Listen to generated samples
  - So sánh với giọng gốc
  - Check prosody, pronunciation
  
Metrics (nếu có validation set):
  - MOS (Mean Opinion Score): > 4.0
  - WER (Word Error Rate): < 5%
  - Speaker Similarity: > 0.8
```

### Q: Model overfitting, làm sao?

**A:**
```yaml
Symptoms:
  - Training loss giảm nhưng validation loss tăng
  - Generated audio giống training samples quá
  
Solutions:
  1. More data
  2. Data augmentation
  3. Early stopping
  4. Reduce epochs
  5. Add regularization
```

---

## 🎤 Inference Questions

### Q: Inference mất bao lâu?

**A:**
```yaml
GPU: T4
  10s audio: ~2-3s inference (real-time factor: 0.2-0.3x)
  
GPU: V100
  10s audio: ~1-1.5s inference (RTF: 0.1-0.15x)
  
GPU: A100
  10s audio: ~0.5-1s inference (RTF: 0.05-0.1x)
  
CPU: 8 cores
  10s audio: ~15-20s inference (RTF: 1.5-2.0x)
```

### Q: Reference audio cần như thế nào?

**A:**
```yaml
Duration: 5-10 giây (optimal)
Quality:
  ✅ Clear voice, single speaker
  ✅ Minimal background noise
  ✅ Natural prosody
  ✅ Consistent volume
  
Avoid:
  ❌ Multiple speakers
  ❌ Background music
  ❌ Very short (<3s) or long (>15s)
  ❌ Lots of pauses/silence
```

### Q: Có cần provide reference text không?

**A:**
```yaml
Recommended: Yes
  - Provide accurate transcription
  - Better quality results
  
Optional: No
  - Model tự động dùng Whisper để transcribe
  - May not be 100% accurate
  - OK cho English/Chinese
  - Vietnamese nên provide manually
```

### Q: Làm sao generate text dài (>100 từ)?

**A:** Model tự động chia thành chunks:
```python
# Automatic chunking
long_text = """
Đây là một đoạn text rất dài.
Nó sẽ được tự động chia thành nhiều chunks nhỏ.
Mỗi chunk được generate riêng rẽ.
Sau đó được concatenate lại với cross-fade.
"""

# Model handles automatically
audio = model.infer(ref_audio, ref_text, long_text)
```

### Q: Output có nhiều silence, làm sao?

**A:**
```bash
# Option 1: Enable remove_silence
f5-tts_infer-cli --remove_silence ...

# Option 2: Adjust NFE steps
--nfe_step 64  # Higher quality, less silence

# Option 3: Post-process
from f5_tts.infer.utils_infer import remove_silence_for_generated_wav
remove_silence_for_generated_wav("output.wav")
```

---

## 🔧 Technical Questions

### Q: F5-TTS architecture thế nào?

**A:** 
```
Text → Transformer Encoder → Text Features
                                    ↓
Reference Audio → Speaker Embedding ┘
                                    ↓
                        Flow Matching (DiT)
                                    ↓
                            Mel-Spectrogram
                                    ↓
                            Vocoder (Vocos)
                                    ↓
                                  Audio
```

### Q: Khác gì với traditional TTS?

**A:**
```yaml
Traditional (Tacotron, FastSpeech):
  - Autoregressive or non-autoregressive
  - Requires extensive training data
  - Limited voice cloning capability
  
F5-TTS (Flow Matching):
  - Non-autoregressive
  - Zero-shot voice cloning
  - Faster inference
  - Better quality with less data
```

### Q: Có thể customize model architecture không?

**A:** **Có**, các options:
```yaml
Model Size:
  - F5TTS_Small: ~100M params, faster, lower quality
  - F5TTS_Base: ~200M params, balanced (default)
  - F5TTS_Large: Custom, higher quality
  
Architecture:
  - DiT (Diffusion Transformer): Default
  - UNetT (U-Net Transformer): E2-TTS style
  - MMDiT (Multi-Modal DiT): Experimental
```

### Q: Có thể train từ scratch không?

**A:** **Có**, nhưng **không khuyến nghị**:
- ⚠️ Cần > 1000 giờ data
- ⚠️ Training time rất lâu (weeks-months)
- ⚠️ Compute cost cao
- 💡 **Better:** Fine-tune từ pretrained model

### Q: Pretrained model được train trên data gì?

**A:**
```yaml
Base Model (SWivid/F5-TTS):
  Languages: Chinese + English
  Dataset: Emilia (multi-lingual)
  Duration: ~1000+ hours
  
Vietnamese Model (hynt/F5-TTS-Vietnamese-100h):
  Language: Vietnamese
  Base: Fine-tuned from SWivid/F5-TTS
  Duration: ~100 hours
```

---

## 💡 Best Practices Questions

### Q: Tips để có model quality tốt nhất?

**A:**
```yaml
Data Quality (most important):
  1. Accurate transcription (99%+)
  2. Clean audio (SNR > 20dB)
  3. Consistent speaker
  4. Natural prosody
  
Data Quantity:
  5. Minimum 10 hours
  6. Recommended 50-100 hours
  7. Diverse content
  
Training:
  8. Fine-tune từ pretrained
  9. Monitor loss curves
  10. Save best checkpoints
  11. Test regularly
```

### Q: Làm sao tối ưu inference speed?

**A:**
```yaml
Hardware:
  1. Use GPU (T4 minimum)
  2. fp16 inference
  3. Batch inference (nếu có nhiều texts)
  
Parameters:
  4. Lower NFE steps (16 thay vì 32)
  5. Cache speaker embeddings
  6. Preload model
  
Code:
  7. Use TorchScript (if applicable)
  8. ONNX export (advanced)
```

### Q: Có thể commercial sử dụng không?

**A:**
```yaml
License: MIT (permissive)
  ✅ Can use commercially
  ✅ Can modify
  ✅ Can distribute
  
BUT:
  ⚠️ Voice rights: Cần consent từ voice owner
  ⚠️ Model rights: Check pretrained model license
  ⚠️ Data rights: Check dataset licenses
  
Recommendation:
  - Use own recorded data
  - Get explicit permission
  - Consult legal advisor
```

---

## 🚀 Advanced Questions

### Q: Multi-speaker training như thế nào?

**A:**
```python
# Option 1: Single model, speaker embeddings
# Metadata includes speaker_id
speaker_001|wavs/audio_001.wav|xin chào
speaker_002|wavs/audio_002.wav|hôm nay trời đẹp

# Option 2: Separate models per speaker
# Train riêng cho mỗi speaker
train_speaker("speaker_001", data_001)
train_speaker("speaker_002", data_002)
```

### Q: Có thể control emotion không?

**A:**
```yaml
Currently: Limited
  - Model learns prosody từ training data
  - Reference audio influence emotion
  
Future:
  - Emotion conditioning (planned)
  - Style transfer
  - Prosody control
  
Workaround:
  - Use reference audio với desired emotion
  - Fine-tune with emotion-labeled data
```

### Q: Làm sao implement real-time TTS?

**A:**
```python
# Streaming inference (experimental)
from f5_tts.infer import StreamingTTS

streaming_tts = StreamingTTS(model_path)

# Generate incrementally
for chunk in text_chunks:
    audio_chunk = streaming_tts.generate_chunk(chunk)
    play_audio(audio_chunk)  # Play while generating
```

### Q: Có API server sẵn không?

**A:** Chưa có official, nhưng có thể build:
```python
# Example với FastAPI
from fastapi import FastAPI, File
from f5_tts.api import F5TTS

app = FastAPI()
tts = F5TTS()

@app.post("/tts")
async def generate_speech(
    text: str,
    speaker: str = "default"
):
    audio = tts.infer(text=text, speaker=speaker)
    return {"audio": audio}

# Run: uvicorn server:app --host 0.0.0.0 --port 8000
```

---

## 🌐 Deployment Questions

### Q: Deploy lên production như thế nào?

**A:**
```yaml
Options:
  1. Docker Container
     - Package model + dependencies
     - Deploy on AWS/GCP/Azure
     
  2. API Service
     - FastAPI / Flask
     - Load balancer
     - GPU instances
     
  3. Edge Deployment
     - ONNX export
     - TensorRT optimization
     - Mobile/embedded devices
     
  4. Serverless
     - AWS Lambda (CPU inference)
     - Google Cloud Functions
     - Azure Functions
```

### Q: Làm sao scale cho nhiều users?

**A:**
```yaml
Architecture:
  Load Balancer
      ↓
  Multiple Inference Servers (GPU)
      ↓
  Model Cache (Redis)
      ↓
  Storage (S3/GCS)
  
Optimization:
  - Batch inference
  - Model caching
  - Request queuing
  - Autoscaling
```

---

## 📚 Learning Resources

### Q: Học thêm về F5-TTS ở đâu?

**A:**
- 📄 **Paper:** [F5-TTS ArXiv](https://arxiv.org/abs/2410.06885)
- 💻 **Code:** [GitHub Repo](https://github.com/SWivid/F5-TTS)
- 🎮 **Demo:** [HuggingFace Space](https://huggingface.co/spaces/hynt/F5-TTS-Vietnamese-100h)
- 📖 **Memory Bank:** Docs trong `memory-bank/`

### Q: Cộng đồng Vietnamese TTS ở đâu?

**A:**
- GitHub Issues: [F5-TTS-Vietnamese](https://github.com/nguyenthienhy/F5-TTS-Vietnamese)
- Discord/Telegram: (TBD)
- Facebook Groups: AI Vietnam communities

---

**Prev:** [`10-TROUBLESHOOTING.md`](10-TROUBLESHOOTING.md)  
**Back to Index:** [`00-INDEX.md`](00-INDEX.md)



