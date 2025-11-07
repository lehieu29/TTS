# 04 - Training Pipeline

## 🔄 Complete Training Pipeline

Training pipeline bao gồm 6 stages chính, được quản lý bởi `fine_tuning.sh`.

---

## 📊 Pipeline Overview

```
Stage 0: Convert Sample Rate
    ↓
Stage 1: Prepare Metadata
    ↓
Stage 2: Check Vocabulary
    ↓
Stage 3: Extend Embedding
    ↓
Stage 4: Feature Extraction
    ↓
Stage 5: Fine-tuning
    ↓
Trained Model
```

---

## 🎯 Stage 0: Convert Sample Rate

### Purpose
Chuyển đổi tất cả audio về 24kHz mono (yêu cầu của F5-TTS).

### Script
`convert_sr.py`

### Process
```python
# Input: data/your_dataset/*.wav (bất kỳ sample rate nào)
# Output: data/your_dataset/*.wav (24kHz mono)

for audio_file in dataset:
    sox audio_file -r 24000 -c 1 output_file
```

### Technical Details
```bash
# Tool: sox
# Command: sox input.wav -r 24000 -c 1 output.wav
# Parameters:
#   -r 24000: Resample to 24kHz
#   -c 1: Convert to mono
```

### Why 24kHz?
- F5-TTS model được train với 24kHz
- Balance giữa quality và compute
- Standard cho modern TTS

### Skip Condition
Nếu audio của bạn đã là 24kHz mono, set `stage=1` để bỏ qua.

---

## 📝 Stage 1: Prepare Metadata

### Purpose
Tạo file metadata.csv chứa mapping audio ↔ text và vocab.

### Script
`prepare_metadata.py`

### Input
```
data/your_dataset/
├── audio_001.wav
├── audio_001.txt  → "xin chào các bạn"
├── audio_002.wav
├── audio_002.txt  → "hôm nay trời đẹp"
└── ...
```

### Output
```
data/your_training_dataset/
├── wavs/                    # Copied audio files
│   ├── audio_001.wav
│   ├── audio_002.wav
│   └── ...
├── metadata.csv             # Audio-text pairs
└── vocab_your_dataset.txt   # Character vocabulary
```

### metadata.csv Format
```csv
wavs/audio_001.wav|xin chào các bạn
wavs/audio_002.wav|hôm nay trời đẹp
wavs/audio_003.wav|tôi là trợ lý ảo
```

### vocab_your_dataset.txt
```txt
 
a
à
á
ả
ã
ạ
ă
b
c
...
```

### Filtering Rules
```python
# Loại bỏ audio không hợp lệ
if duration < 1 or duration > 30:
    skip  # Too short or too long
    
if len(text.split()) < 3:
    skip  # Text too short
```

### Code Flow
```python
def process_dataset():
    wav_paths = glob("data/your_dataset/*.wav")
    tokens = set()
    
    with open("metadata.csv", "w") as fw:
        for wav_path in wav_paths:
            # Read text
            txt_path = wav_path.replace(".wav", ".txt")
            text = open(txt_path).read().strip().lower()
            
            # Check duration
            duration = get_audio_duration(wav_path)
            if duration < 1 or duration > 30:
                continue
                
            # Copy audio
            shutil.copy(wav_path, f"wavs/{basename(wav_path)}")
            
            # Write metadata
            fw.write(f"wavs/{basename(wav_path)}|{text}\n")
            
            # Collect vocab
            tokens.update(text)
    
    # Save vocab
    with open("vocab.txt", "w") as fv:
        fv.write("\n".join(sorted(tokens)))
```

---

## 🔍 Stage 2: Check Vocabulary

### Purpose
Tìm các token trong dataset mà chưa có trong pretrained model vocab.

### Script
`check_vocab_pretrained.py`

### Process
```python
# Load vocabs
pretrained_vocab = load("data/Emilia_ZH_EN_pinyin/vocab.txt")
dataset_vocab = load("data/your_training_dataset/vocab_your_dataset.txt")

# Find missing tokens
missing = []
for token in dataset_vocab:
    if token not in pretrained_vocab:
        missing.append(token)

# Create new vocab
new_vocab = pretrained_vocab + missing
save("data/your_training_dataset/vocab.txt", new_vocab)
```

### Why This Matters
- Pretrained model có vocab cho Chinese + English
- Tiếng Việt có các ký tự đặc biệt: ă, â, đ, ê, ô, ơ, ư và dấu thanh
- Cần thêm tokens này vào model

### Output
```
Số token thiếu trong vocab pretrained: 42
Vocab mới đã được lưu tại data/your_training_dataset/vocab.txt
Tổng số token: 812
```

### Common Missing Tokens (Vietnamese)
```txt
ă ắ ằ ẳ ẵ ặ
â ấ ầ ẩ ẫ ậ
đ
ê ế ề ể ễ ệ
ô ố ồ ổ ỗ ộ
ơ ớ ờ ở ỡ ợ
ư ứ ừ ử ữ ự
```

---

## 🔧 Stage 3: Extend Embedding

### Purpose
Mở rộng embedding layer của pretrained model để support tokens mới.

### Script
`extend_embedding_pretrained.py`

### Process
```python
# Load checkpoint
ckpt = torch.load("pretrained_model_1200000.pt")

# Get current embedding
old_embed = ckpt["ema_model.transformer.text_embed.weight"]
vocab_old, embed_dim = old_embed.shape  # e.g., [770, 512]

# Calculate new size
num_new_tokens = 42  # From Stage 2
vocab_new = vocab_old + num_new_tokens  # 770 + 42 = 812

# Create new embedding
new_embed = torch.zeros(vocab_new, embed_dim)
new_embed[:vocab_old] = old_embed  # Copy old weights
new_embed[vocab_old:] = torch.randn(num_new_tokens, embed_dim)  # Initialize new

# Save
ckpt["ema_model.transformer.text_embed.weight"] = new_embed
torch.save(ckpt, "pretrained_model_extended.pt")
```

### Key Points
- **Preserve old weights:** Giữ nguyên embedding của tokens đã học
- **Random init new weights:** Khởi tạo ngẫu nhiên cho tokens mới
- **Seed control:** Set seed=666 để reproducible

### File Paths
```python
# Input
ckpt_path = "hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.pt"

# Output
new_ckpt_path = "ckpts/your_training_dataset/pretrained_model_1200000.pt"
```

---

## 🎨 Stage 4: Feature Extraction

### Purpose
Preprocess tất cả audio + text thành features ready for training.

### Script
`src/f5_tts/train/datasets/prepare_csv_wavs.py`

### Process
```python
# Input: metadata.csv + wavs/
# Output: raw.arrow + duration.json + vocab.txt

for audio_path, text in metadata:
    # 1. Load audio
    audio, sr = torchaudio.load(audio_path)
    
    # 2. Get duration
    duration = len(audio) / sr
    
    # 3. Text processing (character tokenization)
    # Note: Không dùng pinyin cho tiếng Việt
    processed_text = text
    
    # 4. Save to Arrow format
    writer.write({
        "audio_path": audio_path,
        "text": processed_text,
        "duration": duration
    })
```

### Output Files

#### raw.arrow
Binary format chứa processed data, nhanh hơn CSV.

#### duration.json
```json
{
    "duration": [3.2, 5.1, 4.8, 6.3, ...]
}
```

#### vocab.txt (final)
```txt

a
à
á
...
```

### Performance Optimization
```python
# Multi-threading
MAX_WORKERS = cpu_count() - 1
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    results = executor.map(process_audio_file, audio_files)

# Chunk processing
CHUNK_SIZE = 100
for chunk in chunks(audio_files, CHUNK_SIZE):
    process_chunk(chunk)
```

### Duration Distribution
```python
# Ví dụ output
For your_training_dataset, sample count: 1247
For your_training_dataset, vocab size is: 87
For your_training_dataset, total 10.52 hours
```

---

## 🚀 Stage 5: Fine-tuning

### Purpose
Train model với dữ liệu của bạn.

### Script
`src/f5_tts/train/finetune_cli.py`

### Command
```bash
python src/f5_tts/train/finetune_cli.py \
    --exp_name "F5TTS_Base" \
    --dataset_name "your_training_dataset" \
    --batch_size_per_gpu 7000 \
    --num_warmup_updates 20000 \
    --save_per_updates 10000 \
    --last_per_updates 10000 \
    --finetune \
    --log_samples \
    --pretrain "ckpts/your_training_dataset/pretrained_model_1200000.pt"
```

### Key Parameters

#### Batch Size
```python
--batch_size_per_gpu 7000  # Frames per batch
# Larger = faster but more memory
# GPU Memory requirements:
#   3200: 8GB
#   7000: 16GB
#   10000: 24GB
```

#### Learning Rate
```python
--learning_rate 1e-5  # Default cho fine-tuning
# Không set quá cao → pretrained knowledge bị destroy
```

#### Warmup
```python
--num_warmup_updates 20000
# Gradually tăng learning rate từ 0 → target
# Giúp training ổn định
```

#### Checkpointing
```python
--save_per_updates 10000     # Save checkpoint mỗi 10k updates
--last_per_updates 10000     # Save model_last.pt mỗi 10k updates
--keep_last_n_checkpoints 3  # Chỉ giữ 3 checkpoints gần nhất
```

### Training Loop

```python
for epoch in range(epochs):
    for batch in dataloader:
        # 1. Forward pass
        loss = model(
            inp=batch["audio"],
            text=batch["text"],
            lens=batch["lens"]
        )
        
        # 2. Backward pass
        loss.backward()
        
        # 3. Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            max_grad_norm=1.0
        )
        
        # 4. Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # 5. EMA update
        ema_model.update()
        
        # 6. Logging
        if step % log_interval == 0:
            logger.log({
                "loss": loss.item(),
                "lr": optimizer.param_groups[0]["lr"],
                "step": step
            })
        
        # 7. Checkpoint saving
        if step % save_per_updates == 0:
            save_checkpoint(
                f"model_{step}.pt",
                model, optimizer, ema_model
            )
```

### Training Monitoring

#### Console Output
```
Epoch 1/100 | Step 100/50000 | Loss: 0.234 | LR: 1.2e-6 | Time: 0.5s/step
Epoch 1/100 | Step 200/50000 | Loss: 0.198 | LR: 2.4e-6 | Time: 0.5s/step
...
```

#### Checkpoint Files
```
ckpts/your_training_dataset/
├── pretrained_model_1200000.pt  # Extended base model
├── model_10000.pt               # Checkpoint at 10k steps
├── model_20000.pt               # Checkpoint at 20k steps
├── model_30000.pt               # Checkpoint at 30k steps
└── model_last.pt                # Latest checkpoint
```

### Multi-GPU Training

```bash
# Sử dụng accelerate
accelerate launch src/f5_tts/train/finetune_cli.py \
    --exp_name "F5TTS_Base" \
    --dataset_name "your_training_dataset" \
    --batch_size_per_gpu 7000 \
    ...
```

---

## 📊 Training Time Estimates

| Dataset Size | Epochs | GPU (T4) | GPU (V100) | CPU |
|--------------|--------|----------|------------|-----|
| 10 phút      | 50     | 30 phút  | 15 phút    | 4h  |
| 1 giờ        | 50     | 2 giờ    | 1 giờ      | 24h |
| 10 giờ       | 50     | 10 giờ   | 5 giờ      | 7d  |
| 100 giờ      | 50     | 3 ngày   | 1.5 ngày   | N/A |

---

## 🎯 Best Practices

### 1. Data Quality > Quantity
```python
# 10 giờ clean audio > 100 giờ noisy audio
- Rõ ràng, ít noise
- Transcription chính xác
- Consistent quality
```

### 2. Start Small
```python
# Test với small dataset trước
stage = 0
stop_stage = 5
# Chạy full pipeline với 10 phút data
# Verify everything works
# Sau đó scale lên
```

### 3. Monitor Training
```python
# Watch for:
- Loss giảm đều
- Không bị overfitting (nếu có validation set)
- Audio samples quality (--log_samples)
```

### 4. Checkpoint Management
```python
# Luôn backup:
- pretrained_model_*.pt
- model_last.pt
- Best checkpoint dựa trên validation

# Xóa intermediate checkpoints nếu thiếu disk space
```

---

## 🐛 Common Issues

### Issue: "CUDA out of memory"
```bash
# Solution: Giảm batch_size
--batch_size_per_gpu 3200  # Từ 7000 → 3200
```

### Issue: "vocab.txt not found"
```bash
# Solution: Check Stage 2 output
ls data/your_training_dataset/vocab.txt
# Phải tồn tại sau Stage 2
```

### Issue: Loss không giảm
```python
# Causes:
1. Learning rate quá cao → giảm xuống
2. Data quality kém → check audio + transcription
3. Batch size quá nhỏ → tăng lên nếu có GPU memory
```

---

**Prev:** [`03-ARCHITECTURE.md`](03-ARCHITECTURE.md)  
**Next:** [`05-INFERENCE-PIPELINE.md`](05-INFERENCE-PIPELINE.md)



