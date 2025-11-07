# 07 - Technical Specifications

## 🖥️ System Requirements

### Minimum Requirements

```yaml
Hardware:
  CPU: 4 cores, 2.5GHz+
  RAM: 16GB
  GPU: NVIDIA GTX 1060 (6GB VRAM) hoặc tương đương
  Storage: 50GB available space

Software:
  OS: Linux (Ubuntu 20.04+), Windows 10/11, macOS
  Python: 3.10
  CUDA: 11.8+ (cho GPU)
```

### Recommended Requirements

```yaml
Hardware:
  CPU: 8+ cores, 3.0GHz+
  RAM: 32GB+
  GPU: NVIDIA RTX 3090/4090 (24GB VRAM) hoặc V100
  Storage: 500GB SSD

Software:
  OS: Linux (Ubuntu 22.04)
  Python: 3.10
  CUDA: 12.1+
```

### Cloud Options

```yaml
Google Colab:
  Free Tier: T4 GPU (16GB), 12GB RAM
  Pro: V100/A100, 32GB RAM
  Pro+: A100, 51GB RAM
  
AWS:
  g4dn.xlarge: T4 GPU, 16GB RAM
  p3.2xlarge: V100 GPU, 61GB RAM
  p4d.24xlarge: 8x A100
  
Azure:
  NC6s_v3: V100, 112GB RAM
  ND40rs_v2: 8x V100
```

---

## 📦 Dependencies

### Core Dependencies

```python
# Deep Learning
torch>=2.0.0
torchaudio>=2.0.0
torchdiffeq
ema-pytorch>=0.5.2

# Model Architecture
x-transformers>=1.31.14

# Audio Processing
librosa
soundfile
pydub
vocos  # Vocoder

# UI & API
gradio>=3.45.2
click

# Utilities
numpy<=1.26.4
scipy
matplotlib
tqdm>=4.65.0
cached_path
datasets
safetensors

# Training
accelerate>=0.33.0
wandb  # Logging
hydra-core>=1.3.0

# Optional
bitsandbytes>0.37.0  # 8-bit optimization
transformers  # For Whisper transcription
faster-whisper==0.10.1  # Faster Whisper
```

### Installation via pip

```bash
pip install -e .
# Installs all dependencies from pyproject.toml
```

---

## 🏗️ Model Specifications

### F5TTS_Base (Default)

```yaml
Architecture: DiT (Diffusion Transformer)
Parameters: ~200M

Text Encoder:
  Type: Transformer
  Embedding Dim: 512
  Layers: 22
  Attention Heads: 16
  FFN Multiplier: 2
  
Backbone (DiT):
  Dimension: 1024
  Depth: 22 layers
  Attention Heads: 16
  FFN Multiplier: 2
  Conv Layers: 4
  Position Encoding: Learnable
  
Mel Spectrogram:
  Sample Rate: 24000 Hz
  N FFT: 1024
  Hop Length: 256
  Win Length: 1024
  Mel Channels: 100
  
Vocoder:
  Type: Vocos
  Architecture: Convolutional
```

### F5TTS_Small

```yaml
Architecture: DiT
Parameters: ~100M

Backbone:
  Dimension: 768
  Depth: 18 layers
  Attention Heads: 12
  FFN Multiplier: 2
  
# Faster inference, slightly lower quality
```

### E2TTS_Base

```yaml
Architecture: UNetT (U-Net Transformer)
Parameters: ~250M

Backbone:
  Dimension: 1024
  Depth: 24 layers
  Attention Heads: 16
  FFN Multiplier: 4
  
# Alternative architecture to DiT
```

---

## 🎛️ Training Hyperparameters

### Default Configuration

```yaml
# Optimizer
optimizer: AdamW
learning_rate: 1.0e-5
weight_decay: 0.0
betas: [0.9, 0.999]
eps: 1.0e-8

# Learning Rate Schedule
scheduler: WarmupScheduler
num_warmup_updates: 20000
warmup_type: linear

# Training
epochs: 1000
batch_size_per_gpu: 7000  # frames
batch_size_type: frame  # or 'sample'
max_samples: 64
gradient_accumulation_steps: 1
max_grad_norm: 1.0
mixed_precision: fp16  # or bf16

# EMA
ema_decay: 0.9999

# Checkpointing
save_per_updates: 10000
last_per_updates: 10000
keep_last_n_checkpoints: 3

# Logging
log_interval: 100
log_samples: true
```

### Batch Size Guidelines

```yaml
GPU Memory vs Batch Size (frames):
  8GB:  3200
  12GB: 5000
  16GB: 7000
  24GB: 10000
  32GB: 14000
  40GB: 18000

# Adjust based on your GPU
# Larger batch = faster training but more memory
```

### Learning Rate Tuning

```yaml
Fine-tuning (from pretrained):
  Initial LR: 1e-5
  Min LR: 1e-7
  
Pretraining (from scratch):
  Initial LR: 1e-4
  Min LR: 1e-6
  
# Lower LR for fine-tuning to preserve pretrained knowledge
```

---

## ⚡ Performance Metrics

### Training Speed

```yaml
GPU: NVIDIA T4
Batch Size: 7000 frames
Speed: ~0.5 steps/second
Memory: ~14GB VRAM

Estimates (50 epochs):
  10 phút data: ~30-60 phút training
  1 giờ data: ~2-4 giờ training
  10 giờ data: ~1-2 ngày training
  100 giờ data: ~1-2 tuần training
```

```yaml
GPU: NVIDIA V100
Batch Size: 10000 frames
Speed: ~1.0 steps/second
Memory: ~20GB VRAM

Estimates (50 epochs):
  10 phút data: ~15-30 phút training
  1 giờ data: ~1-2 giờ training
  10 giờ data: ~12-24 giờ training
  100 giờ data: ~5-7 ngày training
```

```yaml
GPU: NVIDIA A100
Batch Size: 14000 frames
Speed: ~1.5 steps/second
Memory: ~30GB VRAM

Estimates (50 epochs):
  10 phút data: ~10-20 phút training
  1 giờ data: ~40-80 phút training
  10 giờ data: ~8-16 giờ training
  100 giờ data: ~3-5 ngày training
```

### Inference Speed

```yaml
GPU: T4
Text Length: 10 giây audio
NFE Steps: 32
Speed: ~2-3 giây inference time
Real-time Factor: 0.2-0.3x

GPU: V100
Text Length: 10 giây audio
NFE Steps: 32
Speed: ~1-1.5 giây inference time
Real-time Factor: 0.1-0.15x

GPU: A100
Text Length: 10 giây audio
NFE Steps: 32
Speed: ~0.5-1 giây inference time
Real-time Factor: 0.05-0.1x

CPU: 8 cores
Text Length: 10 giây audio
NFE Steps: 32
Speed: ~15-20 giây inference time
Real-time Factor: 1.5-2.0x
```

---

## 💾 Storage Requirements

### Training Data

```yaml
Raw Audio (per hour):
  WAV 24kHz: ~170 MB
  MP3 192kbps: ~85 MB
  FLAC: ~120 MB

Processed Features (per hour):
  raw.arrow: ~200 MB
  Mel-spectrograms (cached): ~300 MB

Total per hour: ~500-700 MB

Examples:
  10 giờ dataset: ~5-7 GB
  100 giờ dataset: ~50-70 GB
  1000 giờ dataset: ~500-700 GB
```

### Model Checkpoints

```yaml
Single Checkpoint:
  F5TTS_Base: ~800 MB
  F5TTS_Small: ~400 MB
  E2TTS_Base: ~1 GB

Training Session (with history):
  Pretrained model: ~800 MB
  Checkpoints (10-20): ~8-16 GB
  Optimizer states: ~1.6 GB per checkpoint
  
Total: ~10-20 GB per training session
```

### Disk Space Planning

```yaml
Small Project (10h data):
  Data: 10 GB
  Training: 15 GB
  Total: 25 GB + margin = 50 GB

Medium Project (100h data):
  Data: 70 GB
  Training: 20 GB
  Total: 90 GB + margin = 150 GB

Large Project (1000h data):
  Data: 700 GB
  Training: 30 GB
  Total: 730 GB + margin = 1 TB
```

---

## 🌐 Network Requirements

### For Training

```yaml
Initial Setup:
  Pretrained Model Download: ~800 MB
  Dependencies: ~2 GB
  Total: ~3 GB

During Training:
  Logging to WandB: ~1-10 MB/hour (optional)
  Checkpoint backup: only if cloud storage
```

### For Inference

```yaml
Model Loading:
  First time: ~800 MB download
  Cached: no network needed

Gradio UI:
  Local: no network
  Shared: requires internet for ngrok tunnel
```

---

## 🔧 Optimization Settings

### Mixed Precision Training

```yaml
PyTorch AMP (Automatic Mixed Precision):
  Enabled: mixed_precision: fp16
  Benefits:
    - 2x faster training
    - 50% less GPU memory
    - Minimal quality loss
    
Usage:
  automatic via Trainer
  or manual with torch.cuda.amp
```

### Gradient Checkpointing

```yaml
# Trade computation for memory
gradient_checkpointing: true

Benefits:
  - 30-50% less GPU memory
  - Can use larger batch size

Drawbacks:
  - 10-20% slower training
```

### Multi-GPU Training

```yaml
# Using accelerate
accelerate config
# Follow prompts to setup

# Then run
accelerate launch src/f5_tts/train/finetune_cli.py ...

Scaling:
  2 GPUs: ~1.8x speedup
  4 GPUs: ~3.5x speedup
  8 GPUs: ~6.5x speedup
  
# Not perfect linear due to communication overhead
```

---

## 📊 Monitoring & Logging

### Metrics to Track

```yaml
Training Metrics:
  - Loss (primary)
  - Learning Rate
  - Gradient Norm
  - Steps per Second
  - GPU Memory Usage
  - GPU Utilization

Validation Metrics:
  - Validation Loss
  - MOS Score (if available)
  - WER (Word Error Rate)
  - Speaker Similarity

System Metrics:
  - GPU Temperature
  - Disk I/O
  - Network I/O (if distributed)
```

### Logging Tools

```yaml
WandB (Weights & Biases):
  Setup: wandb login
  Usage: --logger wandb
  Features:
    - Real-time metrics
    - Audio samples
    - Model checkpoints
    - Comparison

TensorBoard:
  Setup: automatic
  Usage: --logger tensorboard
  View: tensorboard --logdir ckpts/
  Features:
    - Scalar metrics
    - Spectrograms
    - Model graph
```

---

## 🔐 Security & Privacy

### Data Privacy

```yaml
Considerations:
  - Training data may contain sensitive info
  - Model can memorize training samples
  - Voice can be used for impersonation

Best Practices:
  - Use consent from voice owners
  - Secure storage for datasets
  - Access control for models
  - Watermarking (if applicable)
```

### Model Security

```yaml
Risks:
  - Model theft
  - Unauthorized use
  - Adversarial attacks

Mitigations:
  - Checkpoint encryption
  - API authentication
  - Rate limiting
  - Usage monitoring
```

---

## 🌍 Localization

### Vietnamese Support

```yaml
Character Set:
  - Lowercase: a-z
  - Vietnamese vowels: ă â đ ê ô ơ ư
  - Tone marks: ́ ̀ ̉ ̃ ̣
  - Combined: á à ả ã ạ (và variations)

Vocab Size:
  Base (Chinese+English): 770 tokens
  Vietnamese additions: ~42 tokens
  Total: ~812 tokens

Text Processing:
  - NFD normalization
  - Lowercase
  - Keep diacritics
  - Punctuation preserved
```

---

## 📈 Scalability

### Horizontal Scaling

```yaml
Multi-Node Training:
  Nodes: 2-8
  GPUs per Node: 4-8
  Total GPUs: 8-64
  
Communication:
  Backend: NCCL
  Network: InfiniBand (optimal)
  
Efficiency:
  8 GPUs: ~7x speedup
  16 GPUs: ~13x speedup
  32 GPUs: ~25x speedup
```

### Data Parallelism

```yaml
Strategy: DistributedDataParallel (DDP)
Batch Size: batch_size_per_gpu × num_gpus
Gradient Sync: After each backward pass
Recommended: GPU count = 2^n (2, 4, 8, 16, ...)
```

---

**Prev:** [`06-DATA-REQUIREMENTS.md`](06-DATA-REQUIREMENTS.md)  
**Next:** [`08-EXPANSION-ROADMAP.md`](08-EXPANSION-ROADMAP.md)



