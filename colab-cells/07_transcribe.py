"""
Cell 07: Automatic Transcription using Whisper
Mục đích:
  - Transcribe tất cả segments bằng Whisper large-v3
  - Tạo text files cho training
  - Normalize Vietnamese text
"""

# ============================================================================
# CELL 07: AUTOMATIC TRANSCRIPTION
# ============================================================================

print("🎤 Automatic Transcription using Whisper...")

import os
import sys
import json
import re
import unicodedata
from pathlib import Path
from tqdm import tqdm

# Use venv
venv_python = "/content/venv/bin/python"
venv_pip = "/content/venv/bin/pip"

# ✅ IMPORTANT: Add venv packages to path BEFORE importing whisper
# This fixes the import error when whisper is installed in venv

# Method 1: Detect Python version dynamically
import subprocess
try:
    result = subprocess.run([venv_python, "--version"], capture_output=True, text=True, check=True)
    python_version = result.stdout.strip().split()[1]  # "3.10.12"
    python_major_minor = '.'.join(python_version.split('.')[:2])  # "3.10"
    venv_site_packages = f'/content/venv/lib/python{python_major_minor}/site-packages'
except:
    python_major_minor = "3.10"  # Fallback
    venv_site_packages = f'/content/venv/lib/python{python_major_minor}/site-packages'

# Method 2: Verify path exists, if not find correct one
import glob
if not os.path.exists(venv_site_packages):
    possible_paths = glob.glob('/content/venv/lib/python*/site-packages')
    if possible_paths:
        venv_site_packages = possible_paths[0]
        print(f"⚠️  Using detected venv path: {venv_site_packages}")
    else:
        print("❌ venv site-packages not found!")
        print("💡 Make sure Cell 03 ran successfully")
        raise RuntimeError("venv site-packages not found. Please run Cell 03 first!")

# Add to path
sys.path.insert(0, venv_site_packages)

# Verify whisper is installed
try:
    import whisper
    import torch  # Required for fp16 inference
    print(f"✅ Whisper imported successfully from: {venv_site_packages}")
except ImportError as e:
    print(f"❌ Failed to import whisper: {e}")
    print(f"💡 Attempting to install whisper...")
    
    # Try to install if missing
    subprocess.run([venv_pip, "install", "openai-whisper"], check=True)
    
    # Try import again
    import whisper
    import torch
    print("✅ Whisper installed and imported successfully!")

# Load config
config_path = "/content/processing_config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

extracted_segments = config['extracted_segments']

# ------------------------------------------------------------------------------
# 0. Check for existing transcriptions backup in Drive
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🔍 Checking for existing transcriptions backup in Drive...")
print("="*70)

transcriptions_backup_dir = "/content/drive/MyDrive/F5TTS_Vietnamese/transcriptions_backup"
config_backup_path = "/content/drive/MyDrive/F5TTS_Vietnamese/transcriptions_config_backup.json"
use_backup = False

if os.path.exists(transcriptions_backup_dir) and os.path.exists(config_backup_path):
    print(f"✅ Found transcriptions backup in Drive: {transcriptions_backup_dir}")
    
    # Check backup contents
    num_backup_files = sum(
        len(filenames)
        for dirpath, dirnames, filenames in os.walk(transcriptions_backup_dir)
    )
    
    backup_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, dirnames, filenames in os.walk(transcriptions_backup_dir)
        for filename in filenames
    ) / (1024 * 1024)  # MB
    
    print(f"📊 Backup info:")
    print(f"   Files: {num_backup_files}")
    print(f"   Size: {backup_size:.1f} MB")
    print()
    print("="*70)
    print("💬 Bạn muốn sử dụng transcriptions backup hay chạy lại?")
    print("="*70)
    print("   1. Sử dụng backup từ Drive (nhanh, tiết kiệm thời gian)")
    print("   2. Chạy lại Whisper transcription (sẽ ghi đè backup cũ)")
    print("="*70)
    
    user_choice = input("Lựa chọn của bạn (1/2, mặc định=1): ").strip()
    
    if user_choice == "2":
        print("\n✅ Sẽ chạy lại Whisper transcription và tạo backup mới...")
        use_backup = False
    else:
        print("\n✅ Sử dụng transcriptions backup từ Drive...")
        use_backup = True
else:
    print("📝 Không tìm thấy transcriptions backup. Sẽ transcribe từ đầu...")
    use_backup = False

# ------------------------------------------------------------------------------
# Restore from backup if user chose to
# ------------------------------------------------------------------------------
if use_backup:
    print("\n" + "="*70)
    print("📦 Restoring transcriptions from Drive backup...")
    print("="*70)
    
    import shutil
    
    # Load config from backup
    with open(config_backup_path, 'r') as f:
        backup_config = json.load(f)
    
    print("⏳ Copying transcription files...")
    
    # Copy all transcription files (.txt, metadata.csv, vocab.txt)
    segments_dir = config['segments_dir']
    
    # For each speaker in backup
    for speaker_backup_dir in Path(transcriptions_backup_dir).iterdir():
        if speaker_backup_dir.is_dir():
            speaker_name = speaker_backup_dir.name
            target_speaker_dir = Path(segments_dir) / speaker_name
            
            # Copy metadata.csv and vocab.txt
            for file_name in ['metadata.csv', 'vocab.txt']:
                backup_file = speaker_backup_dir / file_name
                target_file = target_speaker_dir / file_name
                if backup_file.exists():
                    shutil.copy2(backup_file, target_file)
                    print(f"   Copied {speaker_name}/{file_name}")
            
            # Copy all .txt files in wavs/
            backup_wavs = speaker_backup_dir / 'wavs'
            target_wavs = target_speaker_dir / 'wavs'
            
            if backup_wavs.exists():
                for txt_file in backup_wavs.glob('*.txt'):
                    target_txt = target_wavs / txt_file.name
                    shutil.copy2(txt_file, target_txt)
                
                txt_count = len(list(backup_wavs.glob('*.txt')))
                print(f"   Copied {txt_count} .txt files for {speaker_name}")
    
    # Update current config with backup data
    config['transcriptions'] = backup_config.get('transcriptions', [])
    config['failed_transcriptions'] = backup_config.get('failed_transcriptions', [])
    config['metadata_files'] = backup_config.get('metadata_files', {})
    config['total_transcribed'] = backup_config.get('total_transcribed', 0)
    
    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Also backup to main drive config
    drive_config = "/content/drive/MyDrive/F5TTS_Vietnamese/processing_config.json"
    with open(drive_config, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Restored {config['total_transcribed']} transcriptions from backup")
    
    # Display statistics
    print("\n" + "="*70)
    print("📊 Restored Transcriptions Summary:")
    print("="*70)
    
    # Group by speaker
    speaker_stats = {}
    for trans in config['transcriptions']:
        speaker = trans['speaker']
        if speaker not in speaker_stats:
            speaker_stats[speaker] = {'count': 0}
        speaker_stats[speaker]['count'] += 1
    
    for speaker, stats in speaker_stats.items():
        print(f"\n   {speaker}:")
        print(f"      Transcriptions: {stats['count']}")
        
        # Count vocab size
        vocab_path = Path(config['segments_dir']) / speaker / 'vocab.txt'
        if vocab_path.exists():
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_size = len(f.readlines())
            print(f"      Vocabulary: {vocab_size} characters")
    
    # Show sample transcriptions
    print(f"\n📝 Sample Transcriptions:")
    for trans in config['transcriptions'][:3]:
        filename = Path(trans['audio_path']).name
        text = trans['text']
        print(f"   {filename}")
        print(f"   → \"{text}\"")
        print()
    
    print("\n" + "="*70)
    print("✅ RESTORE COMPLETE! Skipping to next cell...")
    print("="*70)
    print(f"📝 Next Steps:")
    print(f"   Run Cell 08 to prepare final training dataset")
    print("="*70)
    
    # Skip the rest of this cell
    import sys
    sys.exit(0)

# ------------------------------------------------------------------------------
# 1. Load Whisper Model
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("📥 Loading Whisper large-v3...")
print("="*70)

print("⏳ This may take a minute...")

model = whisper.load_model("large-v3")

print("✅ Whisper model loaded!")

# ------------------------------------------------------------------------------
# 2. Define Text Normalization Function
# ------------------------------------------------------------------------------

def normalize_vietnamese_text(text):
    """
    Normalize Vietnamese text for training
    """
    # Lowercase
    text = text.lower()
    
    # Remove special characters (keep Vietnamese diacritics and punctuation)
    text = re.sub(
        r'[^a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ\s,.!?]',
        '',
        text
    )
    
    # Unicode normalization (NFC) - CRITICAL: Use NFC to keep Vietnamese diacritics intact
    # NFD splits diacritics: "à" → "a" + "̀" (2 chars) → vocab size = 34
    # NFC keeps intact: "à" → 1 char → vocab size = 120-150 ✓
    text = unicodedata.normalize('NFC', text)
    
    # Clean multiple spaces
    text = ' '.join(text.split())
    
    # Remove leading/trailing punctuation and spaces
    text = text.strip(' ,.!?')
    
    return text

# ------------------------------------------------------------------------------
# 3. Transcribe All Segments
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🎤 Transcribing segments...")
print("="*70)

print(f"Total segments to transcribe: {len(extracted_segments)}")
print("⏳ Estimated time: ~0.5-1s per segment")
print()

transcriptions = []
failed_transcriptions = []

for seg_info in tqdm(extracted_segments, desc="Transcribing"):
    audio_path = seg_info['path']
    speaker = seg_info['speaker']
    
    try:
        # Transcribe
        result = model.transcribe(
            audio_path,
            language="vi",  # Vietnamese
            task="transcribe",
            fp16=torch.cuda.is_available()
        )
        
        # Get transcription
        text = result['text'].strip()
        
        if not text or len(text) < 3:
            print(f"⚠️  Empty transcription: {Path(audio_path).name}")
            failed_transcriptions.append(audio_path)
            continue
        
        # Normalize text
        normalized_text = normalize_vietnamese_text(text)
        
        if not normalized_text:
            print(f"⚠️  Empty after normalization: {Path(audio_path).name}")
            failed_transcriptions.append(audio_path)
            continue
        
        # Save transcription to .txt file
        txt_path = audio_path.replace('.wav', '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(normalized_text)
        
        transcriptions.append({
            'audio_path': audio_path,
            'text': normalized_text,
            'speaker': speaker,
            'txt_path': txt_path
        })
        
    except Exception as e:
        print(f"⚠️  Warning: Failed to transcribe {Path(audio_path).name}: {e}")
        failed_transcriptions.append(audio_path)
        # Non-critical: continue with other segments
        continue

print(f"\n✅ Transcribed {len(transcriptions)} segments")
print(f"❌ Failed: {len(failed_transcriptions)}")

# Critical validation: Check if we have any transcriptions
if len(transcriptions) == 0:
    print(f"\n{'='*70}")
    print(f"❌ CRITICAL ERROR: No successful transcriptions!")
    print(f"{'='*70}")
    print(f"   All {len(extracted_segments)} segments failed transcription.")
    print(f"   Cannot proceed without transcriptions.")
    print(f"\n⚠️  Possible causes:")
    print(f"   1. Audio quality too poor")
    print(f"   2. Audio format not supported by Whisper")
    print(f"   3. No speech detected in segments")
    print(f"   4. Whisper model not loaded correctly")
    print(f"{'='*70}")
    sys.exit(1)

# Check transcription success rate
success_rate = len(transcriptions) / len(extracted_segments) if extracted_segments else 0
if success_rate < 0.5:
    print(f"\n{'='*70}")
    print(f"⚠️  WARNING: Low transcription success rate!")
    print(f"{'='*70}")
    print(f"   Success: {len(transcriptions)}/{len(extracted_segments)} ({success_rate*100:.1f}%)")
    print(f"   This may indicate audio quality issues.")
    print(f"\n   💡 Recommendations:")
    print(f"      1. Check audio quality of failed segments")
    print(f"      2. Consider re-recording or using different audio")
    print(f"      3. Verify segments contain clear speech")
    print(f"{'='*70}")
    
    proceed = input("\nContinue with low success rate? (y/n, default=n): ").strip().lower()
    if proceed != 'y':
        print("Stopping. Please check audio quality and re-run.")
        sys.exit(1)

# Collect all unique characters for vocab check
all_chars = set()
for trans in transcriptions:
    all_chars.update(trans['text'])

print(f"\n📊 Text Statistics:")
print(f"   Unique characters: {len(all_chars)}")
print(f"   Sample: {''.join(sorted(all_chars)[:50])}...")

if len(all_chars) < 50:
    print(f"\n{'='*70}")
    print(f"⚠️  WARNING: Very low character diversity!")
    print(f"{'='*70}")
    print(f"   Expected for Vietnamese: 100-150 characters")
    print(f"   Got: {len(all_chars)} characters")
    print(f"\n   This may indicate:")
    print(f"   1. Text normalization too aggressive")
    print(f"   2. Transcriptions are too short or repetitive")
    print(f"   3. Unicode encoding issue (should use NFC)")
    print(f"{'='*70}")
elif len(all_chars) > 200:
    print(f"\n⚠️  Note: High character count ({len(all_chars)})")
    print(f"   This may include special characters or emojis")
    print(f"   💡 Review transcriptions if unexpected")

# ------------------------------------------------------------------------------
# 4. Create Metadata Files
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("📝 Creating metadata files...")
print("="*70)

# Group by speaker
speaker_transcriptions = {}
for trans in transcriptions:
    speaker = trans['speaker']
    if speaker not in speaker_transcriptions:
        speaker_transcriptions[speaker] = []
    speaker_transcriptions[speaker].append(trans)

# Create metadata.csv for each speaker
metadata_files = {}

for speaker, trans_list in speaker_transcriptions.items():
    speaker_dir = Path(config['segments_dir']) / speaker
    metadata_path = speaker_dir / "metadata.csv"
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("audio_path|text\n")
        
        # Data
        for trans in trans_list:
            # Get relative path
            audio_rel_path = str(Path(trans['audio_path']).relative_to(speaker_dir))
            f.write(f"{audio_rel_path}|{trans['text']}\n")
    
    metadata_files[speaker] = str(metadata_path)
    print(f"✅ Created metadata for {speaker}: {len(trans_list)} entries")

# ------------------------------------------------------------------------------
# 5. Create Vocabulary File
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("📚 Creating vocabulary...")
print("="*70)

# Collect all unique characters
all_chars = set()
for trans in transcriptions:
    all_chars.update(trans['text'])

# Sort and save
vocab_chars = sorted(all_chars)

for speaker, trans_list in speaker_transcriptions.items():
    speaker_dir = Path(config['segments_dir']) / speaker
    vocab_path = speaker_dir / "vocab.txt"
    
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for char in vocab_chars:
            f.write(f"{char}\n")
    
    print(f"✅ Vocab for {speaker}: {len(vocab_chars)} characters")

# ------------------------------------------------------------------------------
# 6. Update Configuration
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("💾 Updating configuration...")
print("="*70)

config['transcriptions'] = transcriptions
config['failed_transcriptions'] = failed_transcriptions
config['metadata_files'] = metadata_files
config['total_transcribed'] = len(transcriptions)

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

# Backup to Drive
drive_config = "/content/drive/MyDrive/F5TTS_Vietnamese/processing_config.json"
with open(drive_config, 'w') as f:
    json.dump(config, f, indent=2)

print("✅ Configuration saved")

# ------------------------------------------------------------------------------
# 7. Display Results
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("✅ TRANSCRIPTION COMPLETE!")
print("="*70)

print(f"""
📊 Statistics:
   Total Segments: {len(extracted_segments)}
   Successfully Transcribed: {len(transcriptions)}
   Failed: {len(failed_transcriptions)}
   Success Rate: {len(transcriptions)/len(extracted_segments)*100:.1f}%
   
👥 Per Speaker:
""")

for speaker, trans_list in speaker_transcriptions.items():
    total_duration = sum(
        seg['duration'] for seg in extracted_segments 
        if seg['speaker'] == speaker and seg['path'] in [t['audio_path'] for t in trans_list]
    )
    
    print(f"\n   {speaker}:")
    print(f"      Transcriptions: {len(trans_list)}")
    print(f"      Duration: {total_duration / 60:.1f} minutes")
    print(f"      Vocabulary: {len(vocab_chars)} unique characters")

print(f"""
📝 Sample Transcriptions:
""")

# Show first 5 samples
for trans in transcriptions[:5]:
    filename = Path(trans['audio_path']).name
    text = trans['text']
    print(f"   {filename}")
    print(f"   → \"{text}\"")
    print()

print(f"""
📁 Output Files:
   For each speaker:
   ├── wavs/
   │   ├── segment_0001.wav
   │   ├── segment_0001.txt  ← transcription
   │   └── ...
   ├── metadata.csv  ← training metadata
   └── vocab.txt     ← character vocabulary

📝 Next Steps:
   → Run Cell 08 to prepare final training dataset
   → This will organize files for F5-TTS training
   
⚠️  Notes:
   - All transcriptions normalized for Vietnamese
   - Empty or short transcriptions filtered out
   - Ready for training pipeline
""")

# ------------------------------------------------------------------------------
# 8. Backup Transcriptions to Drive
# ------------------------------------------------------------------------------
print("="*70)
print("💾 Backing up transcriptions to Drive...")
print("="*70)

import shutil

transcriptions_backup_dir = "/content/drive/MyDrive/F5TTS_Vietnamese/transcriptions_backup"
config_backup_path = "/content/drive/MyDrive/F5TTS_Vietnamese/transcriptions_config_backup.json"

# Remove old backup if exists
if os.path.exists(transcriptions_backup_dir):
    print("⏳ Removing old backup...")
    shutil.rmtree(transcriptions_backup_dir)

# Create backup directory
os.makedirs(transcriptions_backup_dir, exist_ok=True)

print("⏳ Copying transcription files to Drive...")

# For each speaker, backup .txt files, metadata.csv, and vocab.txt
for speaker, trans_list in speaker_transcriptions.items():
    speaker_dir = Path(config['segments_dir']) / speaker
    speaker_backup_dir = Path(transcriptions_backup_dir) / speaker
    
    # Create speaker backup directory structure
    (speaker_backup_dir / 'wavs').mkdir(parents=True, exist_ok=True)
    
    # Copy metadata.csv and vocab.txt
    for file_name in ['metadata.csv', 'vocab.txt']:
        src_file = speaker_dir / file_name
        dst_file = speaker_backup_dir / file_name
        if src_file.exists():
            shutil.copy2(src_file, dst_file)
    
    # Copy all .txt files
    wavs_dir = speaker_dir / 'wavs'
    backup_wavs_dir = speaker_backup_dir / 'wavs'
    
    txt_files = list(wavs_dir.glob('*.txt'))
    for txt_file in txt_files:
        dst_txt = backup_wavs_dir / txt_file.name
        shutil.copy2(txt_file, dst_txt)
    
    print(f"   Backed up {speaker}: {len(txt_files)} .txt files + metadata + vocab")

# Save config backup
backup_config = {
    'transcriptions': transcriptions,
    'failed_transcriptions': failed_transcriptions,
    'metadata_files': metadata_files,
    'total_transcribed': len(transcriptions)
}

with open(config_backup_path, 'w') as f:
    json.dump(backup_config, f, indent=2)

# Calculate backup size
backup_size = sum(
    os.path.getsize(os.path.join(dirpath, filename))
    for dirpath, dirnames, filenames in os.walk(transcriptions_backup_dir)
    for filename in filenames
) / (1024 * 1024)  # MB

num_backup_files = sum(
    len(filenames)
    for dirpath, dirnames, filenames in os.walk(transcriptions_backup_dir)
)

print(f"\n✅ Backup complete!")
print(f"   Location: {transcriptions_backup_dir}")
print(f"   Files: {num_backup_files}")
print(f"   Size: {backup_size:.1f} MB")

print("="*70)
print("🎉 Ready to proceed to Cell 08!")
print("="*70)



