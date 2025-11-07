---

## 🚀 Google Colab Quick Start

For easy training on Google Colab, we provide ready-to-use cells in the `colab-cells/` directory.

### 📁 Colab Cells Structure

```
colab-cells/
├── 00_README.md                 # Complete instructions
├── INDEX.md                     # Overview
├── QUICK_REFERENCE.md           # Quick tips
│
├── Setup (3 cells)
│   ├── 01_setup_environment.py
│   ├── 02_install_dependencies.py
│   └── 03_install_preprocessing.py
│
├── Data Processing (5 cells)
│   ├── 04_upload_and_prepare.py
│   ├── 05_voice_separation.py
│   ├── 06_segment_audio.py
│   ├── 07_transcribe.py
│   └── 08_prepare_training_data.py
│
└── Training & Inference (3 cells)
    ├── 09_train_model.py
    ├── 10_test_inference.py
    └── 11_gradio_interface.py
```

### ⚡ Usage Scenarios

#### Scenario 1: First Time Training (Full Pipeline)

**Time:** 3-5 hours | **GPU:** T4+ Required

```
Steps: 01 → 02 → 03 → 04 → 08 → 09 → 10 → 11

1. Open Google Colab (colab.research.google.com)
2. Runtime → Change runtime type → GPU ✅
3. Copy & run cells in order
4. Upload your audio files (Cell 04)
5. Wait for training (Cell 09: ~2-4 hours)
6. Test with Gradio UI (Cell 11)
```

**What you get:**
- Trained models saved to Google Drive
- Auto-backup checkpoints
- Web UI for generating speech
- All data preserved for reuse

---

#### Scenario 2: Inference Only (Already Have Trained Models)

**Time:** 20-30 minutes | **GPU:** Optional

**If you already trained models before and they're saved in Drive:**

```
Steps: 01 → 02 → 10 → 11

1. Open Google Colab
2. Enable GPU (recommended)
3. Run Cell 01 (mount Drive)
4. Run Cell 02 (install F5-TTS)
5. Skip cells 03-09 ❌ (no need!)
6. Run Cell 10 (loads model from Drive)
7. Run Cell 11 (Gradio UI)
8. Generate speech! 🎉
```

**Time saved:** ~90% (30 min vs 3-5 hours)

**Why it works:**
- Models are saved in: `/content/drive/MyDrive/F5TTS_Vietnamese/models/`
- Cell 10 & 11 automatically load from Drive
- No preprocessing or training needed

---

#### Scenario 3: Train Additional Speaker

**Time:** 3-4 hours | **GPU:** T4+ Required

**To train a new speaker while keeping existing ones:**

```
Steps: 01 → 02 → 03 → 04 → 08 → 09 → 10 → 11

1. Run setup (Cells 01, 02)
2. Optional: Cell 03 (if new audio has music)
3. Upload NEW audio only (Cell 04)
4. Prepare new speaker data (Cell 08)
5. Train new speaker (Cell 09: ~2-4h)
6. Test new speaker (Cell 10)
7. Gradio UI shows ALL speakers (old + new)
```

**What happens:**
- Old models remain in Drive
- New speaker added to collection
- Cell 11 lists all available speakers
- Can switch between voices in UI

---

#### Scenario 4: Demo Only (Share with Others)

**Time:** 15-20 minutes | **GPU:** Optional

**Quick demo for presentations or sharing:**

```
Steps: 01 → 02 → 11

1. Run Cell 01 (mount Drive)
2. Run Cell 02 (install F5-TTS)
3. Run Cell 11 (Gradio UI with share=True)
4. Share the public link 🔗
5. Anyone can use your trained voices!
```

**Perfect for:**
- Demonstrations
- Sharing with team
- Public demos
- Quick testing

---

### 📊 Time Comparison

| Scenario | Cells Needed | Time | vs First Time |
|----------|--------------|------|---------------|
| **First Time (Full)** | 01→11 | 3-5h | 100% |
| **Inference Only** | 01,02,10,11 | 30min | **10%** ⚡ |
| **Train New Speaker** | 01-04,08-11 | 3-4h | 70% |
| **Demo Only** | 01,02,11 | 20min | **7%** ⚡⚡ |

---

### 💡 Key Features

- ✅ **Virtual Environment:** Solves numpy < 2.0 compatibility issue
- ✅ **Auto-backup:** All models saved to Google Drive
- ✅ **Resume Training:** Can continue if disconnected
- ✅ **Multi-Speaker:** Train and use multiple voices
- ✅ **User-Friendly:** Step-by-step with progress bars
- ✅ **Complete Pipeline:** Voice separation, transcription, training, inference

---

### 📖 Documentation

For detailed instructions, see:
- **`colab-cells/00_README.md`** - Complete guide
- **`colab-cells/INDEX.md`** - Quick overview
- **`colab-cells/QUICK_REFERENCE.md`** - Tips & troubleshooting
- **`memory-bank/`** - Full project documentation

---

### ⚠️ Important Notes

#### Must Do Every Time:
- ✅ Enable GPU in Colab settings
- ✅ Run Cell 01 (mount Drive)
- ✅ Run Cell 02 (install F5-TTS)
- ⚠️ Colab runtime resets each session

#### Can Skip (If Already Trained):
- ❌ Cell 03 (preprocessing tools)
- ❌ Cells 04-08 (data preparation)
- ❌ Cell 09 (training - use existing models)

#### One-Time Investment:
- Train once (3-5 hours)
- Reuse forever (20-30 min setup)
- Models persist in Google Drive
- No need to retrain!

---

## Fine-tuning pipline

Steps:

- Prepare `audio_name` and corresponding transcriptions  
- Add missing vocabulary from your dataset to the pretrained model  
- Expand the model's embedding to support the updated vocabulary  
- Perform feature extraction  
- Fine-tune the model

```bash
bash fine_tuning.sh
```

### Inference

```bash
bash infer.sh
```

### References

- Original F5-TTS repository: [https://github.com/SWivid/F5-TTS](https://github.com/SWivid/F5-TTS)