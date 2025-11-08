"""
Cell 08: Prepare Training Dataset
Mục đích:
  - Organize data theo format F5-TTS yêu cầu
  - Check và extend vocabulary
  - Prepare features (raw.arrow, duration.json)
  - Ready for training
"""

# ============================================================================
# CELL 08: PREPARE TRAINING DATASET
# ============================================================================

print("📦 Preparing Training Dataset...")

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

# Use venv
venv_python = "/content/venv/bin/python"

# Load config
config_path = "/content/processing_config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

# Change to F5-TTS directory
os.chdir("/content/F5-TTS-Vietnamese")

# ------------------------------------------------------------------------------
# 0. Check for Existing Training Data in Drive
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🔍 Checking for existing training data in Drive...")
print("="*70)

drive_training_backup = "/content/drive/MyDrive/F5TTS_Vietnamese/training_data"
use_backup = False

# Check if training data exists in Drive
if os.path.exists(drive_training_backup) and config.get('transcriptions'):
    # List speakers in backup
    backup_speakers = [d for d in os.listdir(drive_training_backup) 
                      if os.path.isdir(os.path.join(drive_training_backup, d)) 
                      and d.endswith('_training')]
    
    if backup_speakers:
        print(f"✅ Found training data backup in Drive!")
        print(f"📊 Backup info:")
        print(f"   Speakers: {len(backup_speakers)}")
        
        # Check if backup has required files
        complete_backup = True
        for speaker_dir in backup_speakers:
            backup_path = os.path.join(drive_training_backup, speaker_dir)
            required_files = ['metadata.csv', 'vocab.txt', 'raw.arrow', 'duration.json']
            
            for req_file in required_files:
                if not os.path.exists(os.path.join(backup_path, req_file)):
                    print(f"   ⚠️  Missing {req_file} in {speaker_dir}")
                    complete_backup = False
        
        if complete_backup:
            print(f"   Status: ✅ Complete")
            print()
            print("="*70)
            print("💬 Bạn muốn sử dụng training data backup hay chuẩn bị lại?")
            print("="*70)
            print("   1. Sử dụng backup từ Drive (nhanh, tiết kiệm thời gian)")
            print("   2. Chuẩn bị lại training data (sẽ ghi đè backup cũ)")
            print("="*70)
            
            user_choice = input("Lựa chọn của bạn (1/2, mặc định=1): ").strip()
            
            if user_choice == "2":
                print("\n✅ Sẽ chuẩn bị lại training data...")
                use_backup = False
            else:
                print("\n✅ Sử dụng training data backup từ Drive...")
                use_backup = True
        else:
            print("   Status: ⚠️  Incomplete backup. Will prepare new data...")
            use_backup = False
    else:
        print("📝 Backup folder exists but empty. Will prepare new data...")
        use_backup = False
else:
    print("📝 No training data backup found. Will prepare new data...")
    use_backup = False

# ------------------------------------------------------------------------------
# Restore from backup if user chose to
# ------------------------------------------------------------------------------
if use_backup:
    print("\n" + "="*70)
    print("📦 Restoring training data from Drive backup...")
    print("="*70)
    
    # Create local data directory
    local_data_dir = "/content/data"
    os.makedirs(local_data_dir, exist_ok=True)
    
    # Copy each speaker's training data
    backup_speakers = [d for d in os.listdir(drive_training_backup) 
                      if os.path.isdir(os.path.join(drive_training_backup, d))]
    
    training_dirs = {}
    speakers_list = []
    
    for speaker_dir in backup_speakers:
        backup_path = os.path.join(drive_training_backup, speaker_dir)
        local_path = os.path.join(local_data_dir, speaker_dir)
        
        print(f"⏳ Copying {speaker_dir}...")
        
        # Remove old local data if exists
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
        
        # Copy entire directory
        shutil.copytree(backup_path, local_path)
        
        # Extract speaker name (remove _training suffix)
        speaker_name = speaker_dir.replace('_training', '')
        speakers_list.append(speaker_name)
        training_dirs[speaker_name] = local_path
        
        print(f"   ✅ Restored to: {local_path}")
    
    # Update config
    config['training_dirs'] = training_dirs
    config['speakers_list'] = speakers_list
    config['ready_for_training'] = True
    
    # Save config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Backup to Drive
    drive_config = "/content/drive/MyDrive/F5TTS_Vietnamese/processing_config.json"
    with open(drive_config, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Restored training data for {len(speakers_list)} speaker(s)")
    
    # Display summary
    print("\n" + "="*70)
    print("📊 Restored Training Data Summary:")
    print("="*70)
    
    for speaker in speakers_list:
        training_dir = training_dirs[speaker]
        
        # Count files
        wav_count = len(list(Path(training_dir).glob("wavs/*.wav")))
        
        # Get duration if available
        duration_file = os.path.join(training_dir, "duration.json")
        if os.path.exists(duration_file):
            with open(duration_file, 'r') as f:
                duration_data = json.load(f)
                total_duration = sum(duration_data['duration']) / 60  # minutes
        else:
            total_duration = 0
        
        # Get vocab size
        vocab_file = os.path.join(training_dir, "vocab.txt")
        if os.path.exists(vocab_file):
            with open(vocab_file, 'r') as f:
                vocab_size = len(f.readlines())
        else:
            vocab_size = 0
        
        print(f"\n   {speaker}:")
        print(f"      Audio files: {wav_count}")
        print(f"      Duration: {total_duration:.1f} minutes")
        print(f"      Vocab size: {vocab_size}")
        print(f"      Location: {training_dir}")
    
    print("\n" + "="*70)
    print("✅ RESTORE COMPLETE! Ready for training...")
    print("="*70)
    print("📝 Next Steps:")
    print("   → Run Cell 09 to start training!")
    print("="*70)
    
    # Skip the rest of this cell
    sys.exit(0)

# ------------------------------------------------------------------------------
# 1. Organize Data Structure
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("📁 Organizing data structure...")
print("="*70)

# Get speakers
speakers = set(trans['speaker'] for trans in config['transcriptions'])

print(f"Preparing data for {len(speakers)} speaker(s): {', '.join(speakers)}")

for speaker in speakers:
    print(f"\n👤 Processing speaker: {speaker}")
    
    # Create directories
    dataset_dir = f"/content/data/{speaker}_dataset"
    training_dir = f"/content/data/{speaker}_training"
    
    os.makedirs(f"{training_dir}/wavs", exist_ok=True)
    
    # Copy segments to training directory
    source_dir = Path(config['segments_dir']) / speaker / "wavs"
    target_dir = Path(training_dir) / "wavs"
    
    # Copy all wav and txt files
    wav_files = list(source_dir.glob("*.wav"))
    print(f"   Copying {len(wav_files)} audio files...")
    
    for wav_file in wav_files:
        shutil.copy(str(wav_file), str(target_dir))
        
        # Also copy txt file
        txt_file = wav_file.with_suffix('.txt')
        if txt_file.exists():
            shutil.copy(str(txt_file), str(target_dir))
    
    # Copy metadata.csv
    metadata_src = Path(config['segments_dir']) / speaker / "metadata.csv"
    metadata_dst = Path(training_dir) / "metadata.csv"
    shutil.copy(str(metadata_src), str(metadata_dst))
    
    print(f"   ✅ Data organized in: {training_dir}")

# ------------------------------------------------------------------------------
# 2. Check Vocabulary
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("📚 Checking vocabulary...")
print("="*70)

# Pretrained vocab path
pretrained_vocab_path = "/content/F5-TTS-Vietnamese/data/Emilia_ZH_EN_pinyin/vocab.txt"

if not os.path.exists(pretrained_vocab_path):
    print("⚠️  Pretrained vocab not found, will create new vocab")
    pretrained_tokens = set()
else:
    with open(pretrained_vocab_path, 'r', encoding='utf-8') as f:
        pretrained_tokens = set(line.strip() for line in f)
    print(f"   Pretrained vocab size: {len(pretrained_tokens)}")

for speaker in speakers:
    print(f"\n👤 {speaker}:")
    
    training_dir = f"/content/data/{speaker}_training"
    
    # Read dataset vocab
    dataset_vocab_path = Path(config['segments_dir']) / speaker / "vocab.txt"
    with open(dataset_vocab_path, 'r', encoding='utf-8') as f:
        dataset_tokens = set(line.strip() for line in f)
    
    print(f"   Dataset vocab size: {len(dataset_tokens)}")
    
    # Find missing tokens
    missing_tokens = dataset_tokens - pretrained_tokens
    print(f"   Missing tokens: {len(missing_tokens)}")
    
    if missing_tokens:
        print(f"   Missing chars: {sorted(missing_tokens)[:20]}")  # Show first 20
    
    # Create extended vocab
    new_vocab = sorted(pretrained_tokens | dataset_tokens)
    
    # Save extended vocab
    new_vocab_path = Path(training_dir) / "vocab.txt"
    with open(new_vocab_path, 'w', encoding='utf-8') as f:
        for token in new_vocab:
            f.write(f"{token}\n")
    
    print(f"   ✅ Extended vocab saved: {len(new_vocab)} tokens")
    
    # 🔴 CRITICAL VALIDATION: Check vocab size
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

# ------------------------------------------------------------------------------
# 2.5. Fix prepare_csv_wavs.py vocab path issue
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🔧 Setting up vocab for prepare_csv_wavs.py...")
print("="*70)

# The prepare_csv_wavs.py script looks for vocab at a hardcoded path
# We need to copy the extended vocab from speaker_training to that location
script_expected_vocab_dir = "/content/F5-TTS-Vietnamese/data/your_training_dataset"
script_expected_vocab_path = f"{script_expected_vocab_dir}/vocab.txt"

os.makedirs(script_expected_vocab_dir, exist_ok=True)

# Use the first speaker's extended vocab (already created in step 2)
first_speaker = list(speakers)[0]
training_dir = f"/content/data/{first_speaker}_training"
speaker_vocab = f"{training_dir}/vocab.txt"

if os.path.exists(speaker_vocab):
    shutil.copy(speaker_vocab, script_expected_vocab_path)
    print(f"✅ Copied extended vocab to: {script_expected_vocab_path}")
    print(f"   Source: {speaker_vocab}")
else:
    print(f"❌ ERROR: Vocab not found at {speaker_vocab}")
    sys.exit(1)

# ------------------------------------------------------------------------------
# 3. Run Feature Extraction
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🎨 Running feature extraction...")
print("="*70)

print("⏳ This may take 5-10 minutes depending on data size...")

for speaker in speakers:
    print(f"\n👤 Processing {speaker}...")
    
    training_dir = f"/content/data/{speaker}_training"
    
    # Run prepare_csv_wavs.py
    cmd = [
        venv_python,
        "src/f5_tts/train/datasets/prepare_csv_wavs.py",
        training_dir,  # input
        training_dir,  # output
        "--workers", "4"
    ]
    
    print(f"   Running: prepare_csv_wavs.py")
    print(f"   Input/Output: {training_dir}")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd="/content/F5-TTS-Vietnamese"
    )
    
    if result.returncode != 0:
        print(f"\n{'='*70}")
        print(f"❌ CRITICAL ERROR: Feature extraction failed for {speaker}!")
        print(f"{'='*70}")
        print(f"Error output:")
        print(result.stderr)
        print(f"\n💡 This prevents proper training data preparation.")
        print(f"   Please check the error above and fix the issue.")
        print(f"{'='*70}")
        sys.exit(1)
    
    # Print output
    print(result.stdout)
    
    # Verify outputs
    required_files = [
        f"{training_dir}/raw.arrow",
        f"{training_dir}/duration.json",
        f"{training_dir}/vocab.txt"
    ]
    
    all_exist = all(os.path.exists(f) for f in required_files)
    
    if all_exist:
        print(f"   ✅ Feature extraction complete!")
        
        # Show file sizes
        arrow_size = os.path.getsize(f"{training_dir}/raw.arrow") / (1024**2)
        print(f"   raw.arrow: {arrow_size:.1f} MB")
        
        # 🔴 CRITICAL VALIDATION: Check arrow file size
        if arrow_size < 0.1:
            print(f"\n{'='*70}")
            print(f"❌ CRITICAL ERROR: raw.arrow file too small!")
            print(f"{'='*70}")
            print(f"   Size: {arrow_size:.2f} MB (expected: >5 MB for 30 min audio)")
            print(f"   This indicates feature extraction failed or no data.")
            print(f"{'='*70}")
            sys.exit(1)
    else:
        print(f"\n{'='*70}")
        print(f"❌ CRITICAL ERROR: Required output files missing!")
        print(f"{'='*70}")
        for f in required_files:
            exists = "✅" if os.path.exists(f) else "❌"
            print(f"   {exists} {Path(f).name}")
        print(f"\n💡 Feature extraction did not complete successfully.")
        print(f"   Please check the error above and re-run the cell.")
        print(f"{'='*70}")
        sys.exit(1)

# ------------------------------------------------------------------------------
# 4. Download Pretrained Model
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("📥 Downloading pretrained model...")
print("="*70)

for speaker in speakers:
    print(f"\n👤 Setting up for {speaker}...")
    
    # Create checkpoint directory
    ckpt_dir = f"/content/ckpts/{speaker}_training"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Download F5-TTS Base model if not exists
    pretrained_path = f"{ckpt_dir}/pretrained_model_1200000.pt"
    
    if not os.path.exists(pretrained_path):
        print("   📥 Downloading F5-TTS Base model (~800MB)...")
        print("   ⏳ This may take 5-10 minutes...")
        
        # Use cached_path to download
        download_script = f'''
import os
from cached_path import cached_path

url = "hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.pt"
ckpt_path = str(cached_path(url))
print(f"Downloaded to: {{ckpt_path}}")

# Copy to our checkpoint directory
import shutil
shutil.copy(ckpt_path, "{pretrained_path}")
print("✅ Model ready!")
'''
        
        result = subprocess.run(
            [venv_python, "-c", download_script],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("❌ Download failed!")
            print(result.stderr)
    else:
        print(f"   ✅ Pretrained model already exists")
    
    # Check if we need to extend embedding
    training_dir = f"/content/data/{speaker}_training"
    vocab_path = f"{training_dir}/vocab.txt"
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        new_vocab_size = len(f.readlines())
    
    # Pretrained vocab size
    with open(pretrained_vocab_path, 'r', encoding='utf-8') as f:
        pretrained_vocab_size = len(f.readlines())
    
    num_new_tokens = new_vocab_size - pretrained_vocab_size
    
    if num_new_tokens > 0:
        print(f"   📝 Need to extend embedding: +{num_new_tokens} tokens")
        # Note: We'll handle this in training script
    else:
        print(f"   ✅ Vocab compatible, no extension needed")

# ------------------------------------------------------------------------------
# 5. Update Configuration & Save to Drive
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("💾 Saving configuration...")
print("="*70)

# Update config
training_dirs = {}
for speaker in speakers:
    training_dirs[speaker] = f"/content/data/{speaker}_training"

config['training_dirs'] = training_dirs
config['speakers_list'] = list(speakers)
config['ready_for_training'] = True

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

# Backup everything to Drive
print("\n📤 Backing up to Google Drive...")

drive_base = "/content/drive/MyDrive/F5TTS_Vietnamese"

for speaker in speakers:
    training_dir = f"/content/data/{speaker}_training"
    drive_training = f"{drive_base}/training_data/{speaker}"
    
    os.makedirs(drive_training, exist_ok=True)
    
    # Copy important files
    for file in ['metadata.csv', 'vocab.txt', 'raw.arrow', 'duration.json']:
        src = f"{training_dir}/{file}"
        if os.path.exists(src):
            shutil.copy(src, f"{drive_training}/{file}")
    
    print(f"   ✅ Backed up {speaker} to Drive")

# Save config to Drive
drive_config = f"{drive_base}/processing_config.json"
with open(drive_config, 'w') as f:
    json.dump(config, f, indent=2)

print("✅ All data backed up to Google Drive")

# ------------------------------------------------------------------------------
# 6. Display Summary
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("✅ TRAINING DATA READY!")
print("="*70)

print(f"""
📊 Preparation Summary:
   Speakers: {len(speakers)}
   
👥 Per Speaker:
""")

for speaker in speakers:
    training_dir = f"/content/data/{speaker}_training"
    
    # Verify files exist before reading
    duration_file = f"{training_dir}/duration.json"
    vocab_file = f"{training_dir}/vocab.txt"
    
    if not os.path.exists(duration_file):
        print(f"\n⚠️  WARNING: Cannot display summary for {speaker}")
        print(f"   Missing file: duration.json")
        print(f"   Feature extraction may have failed.")
        continue
    
    # Count files
    wav_count = len(list(Path(f"{training_dir}/wavs").glob("*.wav")))
    
    # Get vocab size
    with open(vocab_file, 'r') as f:
        vocab_size = len(f.readlines())
    
    # Get duration
    with open(duration_file, 'r') as f:
        duration_data = json.load(f)
        total_duration = sum(duration_data['duration']) / 60  # minutes
    
    print(f"""
   {speaker}:
      Audio files: {wav_count}
      Duration: {total_duration:.1f} minutes
      Vocab size: {vocab_size}
      Ready: ✅
      
      Location: {training_dir}
      Files:
        ✅ wavs/ ({wav_count} files)
        ✅ metadata.csv
        ✅ vocab.txt
        ✅ raw.arrow
        ✅ duration.json
""")
    
    # 🟠 WARNING: Check if duration is too low
    if total_duration > 0 and total_duration < 5:
        print(f"\n{'='*70}")
        print(f"⚠️  WARNING: Very low total duration for {speaker}!")
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

print(f"""
📝 Next Steps:
   → Run Cell 09 to start training!
   → Training will take 2-4 hours for ~30 min of audio
   
⚠️  Before training:
   ✅ All data prepared
   ✅ Pretrained model downloaded
   ✅ Vocabulary extended
   ✅ Features extracted
   ✅ Backed up to Drive
   
🎯 Training Configuration:
   Model: F5TTS_Base (DiT, 200M params)
   Batch size: 7000 frames (adjust based on GPU)
   Epochs: 50-100 (will auto-stop if needed)
   Checkpoint: Every 10000 steps
""")

print("="*70)
print("🎉 Ready to train! Proceed to Cell 09!")
print("="*70)



