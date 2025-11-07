# 03 - Architecture

## 🏗️ System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     F5-TTS SYSTEM                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────┐ │
│  │   Training   │      │  Inference   │      │   API    │ │
│  │   Pipeline   │      │   Pipeline   │      │  Server  │ │
│  └──────┬───────┘      └──────┬───────┘      └────┬─────┘ │
│         │                     │                    │       │
│         └─────────────────────┼────────────────────┘       │
│                               │                            │
│                    ┌──────────▼──────────┐                 │
│                    │   F5-TTS Model      │                 │
│                    │   (DiT/UNetT)       │                 │
│                    └──────────┬──────────┘                 │
│                               │                            │
│                    ┌──────────▼──────────┐                 │
│                    │   Vocoder (Vocos)   │                 │
│                    └─────────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧠 Model Architecture

### F5-TTS Core Components

```python
F5-TTS Model
├── Text Encoder
│   ├── Character Embedding
│   ├── Position Encoding
│   └── Transformer Blocks
│
├── Duration Predictor
│   └── Predicts phoneme durations
│
├── Flow Matching Module (CFM)
│   ├── DiT (Diffusion Transformer) hoặc
│   └── UNetT (U-Net Transformer)
│
└── Vocoder
    └── Vocos (Neural Vocoder)
```

### Detailed Architecture

#### 1. Text Encoder
```python
Input: "xin chào" (text)
    ↓
Character Tokenization: ['x','i','n',' ','c','h','à','o']
    ↓
Embedding Layer: [vocab_size × 512]
    ↓
Positional Encoding
    ↓
Transformer Blocks: 22 layers (F5TTS_Base)
    - Dim: 1024
    - Heads: 16
    - FF Mult: 2
    ↓
Text Features: [seq_len × 512]
```

#### 2. Flow Matching (CFM)
```python
Text Features + Reference Audio Embedding
    ↓
Conditional Flow Matching
    ↓
DiT Blocks (Diffusion Transformer)
    - Depth: 22 layers
    - Dim: 1024
    - Attention Heads: 16
    - Conv Layers: 4
    ↓
Mel-Spectrogram: [time × 100 mel-bins]
```

#### 3. Vocoder (Vocos)
```python
Mel-Spectrogram [time × 100]
    ↓
Vocos Neural Vocoder
    ↓
Waveform [sample_rate × duration]
    ↓
Output: 24kHz Audio
```

---

## 📊 Model Configurations

### F5TTS_Base (Default)
```python
{
    "dim": 1024,           # Model dimension
    "depth": 22,           # Number of transformer layers
    "heads": 16,           # Attention heads
    "ff_mult": 2,          # Feed-forward multiplier
    "text_dim": 512,       # Text embedding dimension
    "conv_layers": 4,      # Convolutional layers
    "pe_attn_head": 1      # Positional encoding attention heads
}
```

### F5TTS_Small (Faster, less quality)
```python
{
    "dim": 768,
    "depth": 18,
    "heads": 12,
    "ff_mult": 2,
    "text_dim": 512,
    "conv_layers": 4
}
```

### E2TTS_Base (Alternative architecture)
```python
{
    "model_type": "UNetT",  # U-Net instead of DiT
    "dim": 1024,
    "depth": 24,
    "heads": 16,
    "ff_mult": 4
}
```

---

## 🔄 Training Architecture

### Training Pipeline Flow

```
Data Loading
    ↓
┌─────────────────────────────────────┐
│  Dataset (metadata.csv + wavs/)    │
│  - Audio paths                      │
│  - Transcriptions                   │
│  - Speaker IDs (optional)           │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│  Data Preprocessing                 │
│  - Audio: Load + Resample (24kHz)  │
│  - Text: Tokenize                   │
│  - Mel-Spec: Extract features       │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│  Model Training Loop                │
│  1. Forward Pass                    │
│     - Text → Text Embedding         │
│     - Audio → Mel-Spectrogram       │
│     - CFM Flow Matching             │
│  2. Loss Calculation                │
│     - Flow Matching Loss            │
│     - Duration Loss                 │
│  3. Backward Pass                   │
│     - Gradient Computation          │
│     - Optimizer Step (AdamW)        │
│  4. EMA Update                      │
│     - Exponential Moving Average    │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│  Checkpoint Saving                  │
│  - model_state_dict                 │
│  - ema_model_state_dict             │
│  - optimizer_state_dict             │
│  - training_stats                   │
└─────────────────────────────────────┘
```

### Key Training Components

#### 1. CFM (Conditional Flow Matching)
```python
class CFM(nn.Module):
    """
    Conditional Flow Matching model
    """
    def __init__(self, transformer, mel_spec_kwargs, vocab_char_map):
        self.transformer = transformer  # DiT or UNetT
        self.mel_spec = MelSpec(**mel_spec_kwargs)
        self.vocab_char_map = vocab_char_map
        
    def forward(self, inp, text, duration, lens, noise_scheduler):
        # Text encoding
        text_embed = self.text_encoder(text)
        
        # Mel-spectrogram from audio
        mel = self.mel_spec(inp)
        
        # Flow matching
        z = self.transformer(mel, text_embed, duration)
        
        return loss
```

#### 2. Trainer
```python
class Trainer:
    """
    Training orchestration
    """
    def __init__(self, model, epochs, learning_rate, ...):
        self.model = model
        self.optimizer = AdamW(params, lr=learning_rate)
        self.scheduler = WarmupScheduler(...)
        
    def train(self, train_dataset):
        for epoch in epochs:
            for batch in train_loader:
                loss = self.model(batch)
                loss.backward()
                self.optimizer.step()
                self.ema_update()
```

---

## 🎯 Inference Architecture

### Inference Flow

```
User Input
    ↓
┌────────────────────────────────────┐
│  Reference Audio + Text            │
│  - ref_audio.wav (10s)             │
│  - ref_text: "xin chào"            │
└──────────────┬─────────────────────┘
               ↓
┌────────────────────────────────────┐
│  Preprocessing                     │
│  1. Audio → Mel-Spectrogram        │
│  2. Text → Token IDs               │
│  3. Extract Speaker Embedding      │
└──────────────┬─────────────────────┘
               ↓
┌────────────────────────────────────┐
│  Generation Text                   │
│  - gen_text: "tôi là AI"           │
│  - Tokenize → IDs                  │
└──────────────┬─────────────────────┘
               ↓
┌────────────────────────────────────┐
│  F5-TTS Model Inference            │
│  1. Encode gen_text                │
│  2. Condition on ref_audio         │
│  3. Flow Matching Sampling         │
│     - NFE steps: 32 (default)      │
│     - Speed control                │
│  4. Generate Mel-Spectrogram       │
└──────────────┬─────────────────────┘
               ↓
┌────────────────────────────────────┐
│  Vocoder (Vocos)                   │
│  - Mel → Waveform                  │
│  - Sample rate: 24kHz              │
└──────────────┬─────────────────────┘
               ↓
┌────────────────────────────────────┐
│  Post-Processing                   │
│  - Remove silence (optional)       │
│  - Normalize volume                │
│  - Save to file                    │
└──────────────┬─────────────────────┘
               ↓
          Output Audio
```

---

## 🗂️ Code Structure

### Main Modules

```
src/f5_tts/
├── model/
│   ├── __init__.py
│   ├── cfm.py              # Conditional Flow Matching
│   ├── dataset.py          # Dataset loading
│   ├── trainer.py          # Training loop
│   ├── modules.py          # Building blocks
│   ├── utils.py            # Utilities
│   └── backbones/
│       ├── dit.py          # DiT architecture
│       ├── mmdit.py        # MM-DiT architecture
│       └── unett.py        # UNetT architecture
│
├── train/
│   ├── finetune_cli.py     # Training CLI
│   ├── finetune_gradio.py  # Training UI
│   ├── train.py            # Core training
│   └── datasets/
│       └── prepare_csv_wavs.py  # Data preparation
│
└── infer/
    ├── infer_cli.py        # Inference CLI
    ├── infer_gradio.py     # Inference UI
    └── utils_infer.py      # Inference utilities
```

### Class Hierarchy

```python
# Model
CFM
├── transformer: DiT | UNetT | MMDiT
├── mel_spec: MelSpec
└── vocab_char_map: dict

# Trainer
Trainer
├── model: CFM
├── optimizer: AdamW
├── scheduler: WarmupScheduler
└── ema_model: ExponentialMovingAverage

# Dataset
load_dataset()
├── metadata.csv → audio_path, text pairs
├── wavs/ → audio files
└── vocab.txt → character vocabulary
```

---

## 🔐 Key Design Patterns

### 1. EMA (Exponential Moving Average)
```python
# Duy trì shadow weights cho stable inference
ema_model = EMA(model, beta=0.9999)

# Training
for batch in data:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    ema_model.update()  # Update shadow weights

# Inference - sử dụng EMA weights
with ema_model.average_parameters():
    output = model(input)
```

### 2. Flow Matching
```python
# Thay vì diffusion steps, dùng continuous flow
t = random.uniform(0, 1)  # Flow time
z_t = t * data + (1 - t) * noise
velocity = model(z_t, t, condition)
loss = ||velocity - (data - noise)||²
```

### 3. Character-based Tokenization
```python
# Không dùng phoneme, dùng trực tiếp characters
text = "xin chào"
tokens = [vocab[c] for c in text]
# → ['x','i','n',' ','c','h','à','o']
```

---

## 📈 Memory & Compute Requirements

### Training

| Batch Size (frames) | GPU Memory | Speed (steps/s) |
|---------------------|------------|-----------------|
| 3200                | 8GB        | 0.5             |
| 7000                | 16GB       | 0.8             |
| 10000               | 24GB       | 1.0             |

### Inference

| Duration | GPU Memory | Time (T4) | Time (CPU) |
|----------|------------|-----------|------------|
| 5s       | 2GB        | 2s        | 10s        |
| 10s      | 2GB        | 3s        | 15s        |
| 30s      | 3GB        | 8s        | 40s        |

---

**Prev:** [`02-QUICK-START.md`](02-QUICK-START.md)  
**Next:** [`04-TRAINING-PIPELINE.md`](04-TRAINING-PIPELINE.md)



