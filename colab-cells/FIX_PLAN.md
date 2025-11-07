# 🔧 Kế Hoạch Sửa Lỗi - F5-TTS Vietnamese Colab Cells

## 📊 Tóm Tắt Vấn Đề

| # | Vấn Đề | Mức Độ | Nguyên Nhân | Impact |
|---|---------|---------|-------------|--------|
| 1 | Vocab size = 34 (quá nhỏ) | 🔴 CRITICAL | NFD normalization tách dấu | Mất 93% vocab |
| 2 | Duration = 0.04h từ 30 phút | 🔴 CRITICAL | VAD filter 3-10s quá strict | Mất 93% data |
| 3 | Không break khi lỗi nghiêm trọng | 🟠 HIGH | Dùng `continue` thay vì `sys.exit()` | User mất thời gian |
| 4 | Thiếu validation checks | 🟠 HIGH | Không validate thresholds | Silent failures |
| 5 | UI upload không rõ ràng | 🟡 MEDIUM | Instructions thiếu | User confusion |

---

## 🎯 Kế Hoạch Chi Tiết

### **Phase 1: Critical Fixes (NGAY LẬP TỨC)** ⏱️ ~30 phút

#### ✅ Fix 1.1: Unicode Normalization (Cell 07)
**File:** `07_transcribe.py`
**Priority:** 🔴 CRITICAL
**Thời gian:** 2 phút

**Thay đổi:**
```python
# Line 112: ❌ BEFORE
text = unicodedata.normalize('NFD', text)

# ✅ AFTER
text = unicodedata.normalize('NFC', text)
```

**Giải thích:**
- NFD: Tách dấu → "à" = "a" + "̀" (2 chars)
- NFC: Giữ nguyên → "à" = 1 char
- NFC là standard cho tiếng Việt trong ML/NLP

**Expected Result:**
- Vocab size: 34 → ~120-150 chars (chuẩn cho tiếng Việt)
- Dấu không bị tách riêng

---

#### ✅ Fix 1.2: VAD Duration Filter (Cell 06)
**File:** `06_segment_audio.py`
**Priority:** 🔴 CRITICAL
**Thời gian:** 5 phút

**Thay đổi:**
```python
# Line 201: ❌ BEFORE
if 3.0 <= duration <= 10.0:

# ✅ AFTER - Option 1: More flexible range
if 1.0 <= duration <= 30.0:

# ✅ AFTER - Option 2: Configurable with warning
MIN_DURATION = 1.0  # Configurable
MAX_DURATION = 30.0  # Configurable
WARN_IF_FILTERED_RATE_ABOVE = 0.7  # Warn if >70% filtered

if MIN_DURATION <= duration <= MAX_DURATION:
```

**Thêm validation:**
```python
# After line 208, add:
filtered_rate = 1 - (len(segments) / len(speech_timestamps)) if speech_timestamps else 0
if filtered_rate > WARN_IF_FILTERED_RATE_ABOVE:
    print(f"  ⚠️  WARNING: {filtered_rate*100:.1f}% segments filtered out!")
    print(f"  Original: {len(speech_timestamps)} → After filter: {len(segments)}")
    print(f"  💡 Consider adjusting MIN_DURATION={MIN_DURATION}s, MAX_DURATION={MAX_DURATION}s")
```

**Expected Result:**
- 30 phút → giữ ~20-25 phút (thay vì 2.4 phút)
- Retention rate: ~70-80% (thay vì 7%)

---

#### ✅ Fix 1.3: Critical Error Handling
**Files:** Cell 06, 07, 08
**Priority:** 🔴 CRITICAL
**Thời gian:** 15 phút

**Nguyên tắc:**
- **Silent errors** (logging only) → Dùng `continue` ✅
- **Critical errors** (ảnh hưởng training) → Dùng `sys.exit(1)` hoặc `raise` ❌

**Cell 06 - Thay đổi:**
```python
# Line 220-224: ❌ BEFORE
except Exception as e:
    print(f"  ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    continue

# ✅ AFTER
except Exception as e:
    print(f"\n{'='*70}")
    print(f"❌ CRITICAL ERROR in VAD processing!")
    print(f"{'='*70}")
    print(f"File: {audio_path}")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    print(f"\n💡 This error prevents proper data preparation.")
    print(f"   Please fix the issue and re-run this cell.")
    print(f"{'='*70}")
    sys.exit(1)  # ✅ STOP HERE!
```

**Cell 06 - Thêm validation cuối cell:**
```python
# After line 304, add validation
if len(extracted_segments) == 0:
    print(f"\n{'='*70}")
    print(f"❌ CRITICAL ERROR: No segments extracted!")
    print(f"{'='*70}")
    print(f"⚠️  Possible causes:")
    print(f"   1. VAD filter too strict (try adjusting MIN/MAX_DURATION)")
    print(f"   2. Audio has no speech detected")
    print(f"   3. Audio format not supported")
    print(f"\n💡 Cannot proceed without segments. Please investigate.")
    print(f"{'='*70}")
    sys.exit(1)

# Calculate retention rate
total_original_duration = sum(
    info['total_duration'] 
    for info in all_segments.values()
)
for audio_path, info in all_segments.items():
    file_duration = torchaudio.info(info['audio_file']).num_frames / torchaudio.info(info['audio_file']).sample_rate / 60
    retention_rate = (info['total_duration'] / 60) / file_duration if file_duration > 0 else 0
    
    if retention_rate < 0.3:
        print(f"\n⚠️  WARNING: Low retention rate for {Path(audio_path).name}")
        print(f"   Original: {file_duration:.1f} min → Kept: {info['total_duration']/60:.1f} min ({retention_rate*100:.1f}%)")
        print(f"   💡 Consider adjusting VAD filter parameters")
```

**Cell 07 - Thêm validation:**
```python
# After line 183, add:
if len(transcriptions) == 0:
    print(f"\n{'='*70}")
    print(f"❌ CRITICAL ERROR: No successful transcriptions!")
    print(f"{'='*70}")
    print(f"   All {len(extracted_segments)} segments failed transcription.")
    print(f"   Cannot proceed without transcriptions.")
    print(f"{'='*70}")
    sys.exit(1)

success_rate = len(transcriptions) / len(extracted_segments) if extracted_segments else 0
if success_rate < 0.5:
    print(f"\n{'='*70}")
    print(f"⚠️  WARNING: Low transcription success rate!")
    print(f"{'='*70}")
    print(f"   Success: {len(transcriptions)}/{len(extracted_segments)} ({success_rate*100:.1f}%)")
    print(f"   This may indicate audio quality issues.")
    print(f"{'='*70}")
    
    proceed = input("\nContinue anyway? (y/n, default=n): ").strip().lower()
    if proceed != 'y':
        print("Stopping. Please check audio quality and re-run.")
        sys.exit(1)
```

**Cell 08 - Thêm validation:**
```python
# After line 123, add vocab validation:
if len(new_vocab) < 50:
    print(f"\n{'='*70}")
    print(f"❌ CRITICAL ERROR: Vocab size too small!")
    print(f"{'='*70}")
    print(f"   Expected for Vietnamese: 100-200 characters")
    print(f"   Got: {len(new_vocab)} characters")
    print(f"   Dataset vocab: {len(dataset_tokens)} characters")
    print(f"\n⚠️  This indicates a serious problem with text processing:")
    print(f"   1. Transcription failed")
    print(f"   2. Text normalization removed too much")
    print(f"   3. Unicode encoding issue (check NFD vs NFC)")
    print(f"\n💡 Please check Cell 07 output and transcriptions.")
    print(f"{'='*70}")
    sys.exit(1)

# After line 192, add duration validation:
if arrow_size < 0.1:
    print(f"\n{'='*70}")
    print(f"❌ CRITICAL ERROR: raw.arrow file too small!")
    print(f"{'='*70}")
    print(f"   Size: {arrow_size:.2f} MB (expected: >5 MB for 30 min audio)")
    print(f"   This indicates feature extraction failed or no data.")
    print(f"{'='*70}")
    sys.exit(1)

# After line 364, add duration validation:
if total_duration > 0 and total_duration < 5:
    print(f"\n{'='*70}")
    print(f"⚠️  WARNING: Very low total duration!")
    print(f"{'='*70}")
    print(f"   Expected: >10 minutes for quality training")
    print(f"   Got: {total_duration:.1f} minutes")
    print(f"   Original audio was likely much longer.")
    print(f"\n   Possible causes:")
    print(f"   1. VAD filter too strict (Cell 06)")
    print(f"   2. Transcription failures (Cell 07)")
    print(f"   3. Feature extraction issues")
    print(f"{'='*70}")
    
    proceed = input("\nContinue with this small dataset? (y/n, default=n): ").strip().lower()
    if proceed != 'y':
        print("Stopping. Please check previous cells.")
        sys.exit(1)
```

---

### **Phase 2: Validation & Warnings** ⏱️ ~45 phút

#### ✅ Fix 2.1: Cell 04 - Multi-file Upload UI
**File:** `04_upload_and_prepare.py`
**Priority:** 🟡 MEDIUM
**Thời gian:** 10 phút

**Thay đổi:**
```python
# Line 32-42: Improve instructions
print("""
📝 Instructions:
   You can upload MULTIPLE files for the SAME speaker:
   
   [Option 1] Upload from computer:
   - Click 'Choose Files' button below
   - Hold Ctrl/Cmd to select MULTIPLE files
   - All files will be used for training the same voice
   
   [Option 2] Use files from Google Drive:
   - Upload all files to: /content/drive/MyDrive/F5TTS_Vietnamese/uploads
   - All files in this folder will be processed automatically
   
⚠️  Notes:
   - Max file size per upload: ~200MB (Colab limit)
   - For larger files: use Google Drive (Option 2)
   - Supported formats: MP3, WAV, FLAC
   - More audio = Better voice quality (recommended: 30-60 min total)
   
🎯 Recommendation:
   - Minimum: 10 minutes of clean audio
   - Good: 30-60 minutes
   - Best: 1-3 hours
""")
```

**Thêm summary sau upload:**
```python
# After line 135, add summary:
total_duration_min = sum(
    float(subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', str(f)],
        capture_output=True, text=True
    ).stdout.strip() or 0) / 60
    for f in audio_files
)

print(f"\n📊 Upload Summary:")
print(f"   Total files: {len(audio_files)}")
print(f"   Total duration: ~{total_duration_min:.1f} minutes")
print(f"   Total size: {sum(f.stat().st_size for f in audio_files) / (1024**2):.1f} MB")

if total_duration_min < 10:
    print(f"\n⚠️  WARNING: Low total duration!")
    print(f"   Recommended: At least 30 minutes for quality results")
    print(f"   Current: {total_duration_min:.1f} minutes")
    print(f"   💡 Consider uploading more audio files")
```

---

#### ✅ Fix 2.2: Add Progress Summary Between Cells
**Files:** End of each cell
**Priority:** 🟡 MEDIUM
**Thời gian:** 20 phút

**Template to add at end of each cell:**
```python
# End of Cell 06
print(f"\n{'='*70}")
print(f"📊 DATA QUALITY CHECK - Cell 06")
print(f"{'='*70}")

total_input_duration = 0
total_output_duration = 0

for audio_path, info in all_segments.items():
    file_duration = torchaudio.info(info['audio_file']).num_frames / torchaudio.info(info['audio_file']).sample_rate
    total_input_duration += file_duration
    total_output_duration += info['total_duration']

retention_rate = total_output_duration / total_input_duration if total_input_duration > 0 else 0

print(f"Input audio: {total_input_duration/60:.1f} minutes")
print(f"Output segments: {total_output_duration/60:.1f} minutes")
print(f"Retention rate: {retention_rate*100:.1f}%")
print(f"Segments extracted: {len(extracted_segments)}")

if retention_rate < 0.5:
    print(f"\n⚠️  LOW RETENTION WARNING!")
    print(f"   Expected: 60-80% retention")
    print(f"   Got: {retention_rate*100:.1f}%")
    print(f"   💡 Check VAD filter parameters (MIN/MAX_DURATION)")
else:
    print(f"\n✅ Retention rate looks good!")

print(f"{'='*70}")
```

```python
# End of Cell 07
print(f"\n{'='*70}")
print(f"📊 DATA QUALITY CHECK - Cell 07")
print(f"{'='*70}")

# Collect all unique chars
all_chars = set()
for trans in transcriptions:
    all_chars.update(trans['text'])

print(f"Transcribed segments: {len(transcriptions)}/{len(extracted_segments)}")
print(f"Success rate: {len(transcriptions)/len(extracted_segments)*100:.1f}%")
print(f"Unique characters: {len(all_chars)}")
print(f"Sample chars: {''.join(sorted(all_chars)[:50])}")

if len(all_chars) < 50:
    print(f"\n⚠️  LOW VOCAB WARNING!")
    print(f"   Expected for Vietnamese: 100-150 characters")
    print(f"   Got: {len(all_chars)} characters")
    print(f"   💡 Check text normalization (should use NFC, not NFD)")
elif len(all_chars) > 200:
    print(f"\n⚠️  HIGH VOCAB WARNING!")
    print(f"   This may include special characters or emojis")
    print(f"   💡 Check transcriptions for unexpected characters")
else:
    print(f"\n✅ Vocab size looks good!")

print(f"{'='*70}")
```

---

#### ✅ Fix 2.3: Add Resume Capability
**Priority:** 🟢 LOW
**Thời gian:** 15 phút

**Add to start of Cell 06, 07, 08:**
```python
# Check if already processed
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if config.get('cell_06_complete', False):
        print(f"\n{'='*70}")
        print(f"ℹ️  Cell 06 already completed")
        print(f"{'='*70}")
        print(f"Segments: {config.get('total_segments', 0)}")
        print(f"Duration: {sum(s['duration'] for s in config.get('extracted_segments', []))/60:.1f} min")
        
        rerun = input("\nRe-run anyway? (y/n, default=n): ").strip().lower()
        if rerun != 'y':
            print("Skipping to next cell...")
            sys.exit(0)
```

---

### **Phase 3: Optimization & Polish** ⏱️ ~30 phút

#### ✅ Fix 3.1: Smarter VAD Parameters
**File:** `06_segment_audio.py`
**Priority:** 🟢 LOW
**Thời gian:** 15 phút

**Add adaptive parameters based on audio type:**
```python
# Auto-detect audio type and adjust parameters
def detect_audio_type(audio_path):
    """Detect if audio is podcast/audiobook (long speech) or conversation"""
    # Simple heuristic: check average segment length in first minute
    # Implementation details...
    pass

# Use different parameters for different audio types
if audio_type == "podcast":
    MIN_DURATION = 2.0
    MAX_DURATION = 30.0
    min_speech_duration_ms = 2000
elif audio_type == "conversation":
    MIN_DURATION = 1.0
    MAX_DURATION = 15.0
    min_speech_duration_ms = 1000
else:
    MIN_DURATION = 1.0
    MAX_DURATION = 30.0
    min_speech_duration_ms = 1500
```

---

#### ✅ Fix 3.2: Better Error Messages
**All files**
**Priority:** 🟢 LOW
**Thời gian:** 15 phút

**Standardize error format:**
```python
def print_critical_error(title, details, suggestions):
    print(f"\n{'='*70}")
    print(f"❌ CRITICAL ERROR: {title}")
    print(f"{'='*70}")
    for detail in details:
        print(f"   {detail}")
    print(f"\n💡 Suggestions:")
    for suggestion in suggestions:
        print(f"   • {suggestion}")
    print(f"{'='*70}\n")

# Usage:
print_critical_error(
    title="Vocab size too small",
    details=[
        f"Expected: 100-200 characters for Vietnamese",
        f"Got: {len(vocab)} characters",
        f"This prevents proper training"
    ],
    suggestions=[
        "Check Cell 07 transcription output",
        "Verify unicode normalization uses NFC (not NFD)",
        "Check if transcriptions contain Vietnamese text"
    ]
)
```

---

## 📋 Implementation Checklist

### **🔴 Critical (Do First)**
- [ ] Fix Unicode normalization NFD → NFC (Cell 07)
- [ ] Fix VAD filter 3-10s → 1-30s (Cell 06)
- [ ] Add critical error handling with sys.exit() (Cells 06, 07, 08)
- [ ] Add vocab size validation < 50 (Cell 08)
- [ ] Add duration validation (Cell 08)
- [ ] Add arrow file size check (Cell 08)

### **🟠 High Priority**
- [ ] Add retention rate warning (Cell 06)
- [ ] Add transcription success rate check (Cell 07)
- [ ] Add data quality summary at end of each cell
- [ ] Improve multi-file upload instructions (Cell 04)

### **🟡 Medium Priority**
- [ ] Add resume capability (Cells 06, 07, 08)
- [ ] Add progress summary between cells
- [ ] Standardize error message format
- [ ] Add config validation at cell start

### **🟢 Low Priority (Nice to Have)**
- [ ] Adaptive VAD parameters
- [ ] Auto-detect audio type
- [ ] Better progress bars
- [ ] Colored console output

---

## 🧪 Testing Plan

### **Test Case 1: Normal Flow (30 min podcast)**
**Expected Results:**
- Vocab size: 120-150
- Duration retention: 60-80%
- Total duration: 15-25 minutes
- No critical errors

### **Test Case 2: Multiple Files**
**Input:** 3 files × 10 minutes each
**Expected Results:**
- All files processed
- Total duration: 18-25 minutes
- Single speaker model

### **Test Case 3: Error Scenarios**
- **Empty transcriptions** → Should exit with clear error
- **Low vocab** → Should exit with clear error
- **Low retention** → Should warn and ask to continue

---

## 📊 Expected Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Vocab size | 34 | 120-150 | +353% |
| Data retention | 7% (0.04h/30min) | 60-80% | +857% |
| Duration kept | 2.4 min | 18-24 min | +750% |
| Error detection | Silent failures | Caught & reported | 100% |
| User experience | Confusing | Clear warnings | Much better |

---

## ⏱️ Total Implementation Time

- **Phase 1 (Critical):** ~30 minutes ⚡
- **Phase 2 (Validation):** ~45 minutes
- **Phase 3 (Polish):** ~30 minutes
- **Testing:** ~30 minutes

**Total:** ~2.5 hours for complete fix

---

## 🎯 Priority Order

1. **NGAY LẬP TỨC:** Fix 1.1 (NFD→NFC) + Fix 1.2 (VAD filter)
2. **Trong 1 giờ:** Fix 1.3 (Error handling) + Cell 08 validation
3. **Trong 2 giờ:** Phase 2 (Warnings & validation)
4. **Khi rảnh:** Phase 3 (Optimization)

---

## ✅ Success Criteria

Sau khi fix, với 30 phút podcast:
- ✅ Vocab size: 100-200 characters
- ✅ Duration: 15-25 minutes (50-80% retention)
- ✅ No silent failures
- ✅ Clear error messages
- ✅ Warning when anomalies detected

---

**📌 Ghi chú quan trọng:**
- **Fix 1.1 và 1.2 là quan trọng nhất** - Giải quyết 90% vấn đề
- Các fix khác là để improve UX và prevent future issues
- Recommend test lại toàn bộ pipeline sau khi fix

---

**Created:** 2025-11-07  
**Status:** Ready for Implementation  
**Priority:** 🔴 CRITICAL
