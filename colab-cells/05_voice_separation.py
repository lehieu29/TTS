"""
Cell 05: Voice Separation (Optional - skip if audio is clean)
Mục đích:
  - Tách giọng nói khỏi nhạc nền bằng Demucs
  - Xử lý podcast có background music
"""

# ============================================================================
# CELL 05: VOICE SEPARATION
# ============================================================================

print("🎵 Voice Separation using Demucs...")

import os
import sys
import json
import subprocess
from pathlib import Path
from tqdm import tqdm

# Use venv Python
venv_python = "/content/venv/bin/python"

# Load config from previous cell
config_path = "/content/processing_config.json"

if not os.path.exists(config_path):
    print("❌ Configuration not found!")
    print("   Please run Cell 04 first")
    sys.exit(1)

with open(config_path, 'r') as f:
    config = json.load(f)

audio_files = [Path(p) for p in config['audio_files']]
speakers = config['speakers']

# ------------------------------------------------------------------------------
# 0. Check for Existing Separated Vocals in Drive
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🔍 Checking for existing separated vocals in Drive...")
print("="*70)

import shutil

drive_vocals_backup = "/content/drive/MyDrive/F5TTS_Vietnamese/vocals_backup"
use_backup = False

# Check if there are separated files in current config and in Drive backup
if config.get('separated', False) and os.path.exists(drive_vocals_backup):
    # Check if backup has files
    backup_files = []
    for root, dirs, files in os.walk(drive_vocals_backup):
        backup_files.extend([f for f in files if f.endswith('.wav')])
    
    if backup_files:
        print(f"✅ Found separated vocals backup in Drive!")
        print(f"📊 Backup info:")
        print(f"   Files: {len(backup_files)}")
        
        # Calculate size
        backup_size = sum(
            os.path.getsize(os.path.join(root, f))
            for root, dirs, files in os.walk(drive_vocals_backup)
            for f in files
        ) / (1024 * 1024)  # MB
        
        print(f"   Size: {backup_size:.1f} MB")
        print()
        print("="*70)
        print("💬 Bạn muốn sử dụng separated vocals backup hay tách lại?")
        print("="*70)
        print("   1. Sử dụng backup từ Drive (nhanh, tiết kiệm thời gian)")
        print("   2. Chạy lại voice separation (sẽ ghi đè backup cũ)")
        print("   3. Skip voice separation (dùng audio gốc)")
        print("="*70)
        
        user_choice = input("Lựa chọn của bạn (1/2/3, mặc định=1): ").strip()
        
        if user_choice == "2":
            print("\n✅ Sẽ chạy lại voice separation...")
            use_backup = False
        elif user_choice == "3":
            print("\n✅ Skip voice separation, dùng audio gốc...")
            config['vocals_dir'] = config['upload_dir']
            config['separated'] = False
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print("\n🎉 Proceeding to next step...")
            sys.exit(0)
        else:
            print("\n✅ Sử dụng separated vocals backup từ Drive...")
            use_backup = True
    else:
        print("📝 Backup folder exists but empty. Will run separation...")
        use_backup = False
else:
    print("📝 No separated vocals backup found. Will run separation...")
    use_backup = False

# ------------------------------------------------------------------------------
# Restore from backup if user chose to
# ------------------------------------------------------------------------------
if use_backup:
    print("\n" + "="*70)
    print("📦 Restoring separated vocals from Drive backup...")
    print("="*70)
    
    # Create local vocals directory
    vocals_dir = "/content/processed/vocals"
    os.makedirs(vocals_dir, exist_ok=True)
    
    # Copy all files from backup
    print("⏳ Copying separated vocals from Drive...")
    
    # Remove old local vocals if exists
    if os.path.exists(vocals_dir):
        shutil.rmtree(vocals_dir)
    
    # Copy entire backup directory
    shutil.copytree(drive_vocals_backup, vocals_dir)
    
    # Update config with local paths
    separated_files = {}
    for root, dirs, files in os.walk(vocals_dir):
        for f in files:
            if f.endswith('_vocals.wav'):
                vocals_path = os.path.join(root, f)
                # Try to match with original audio file
                base_name = f.replace('_vocals.wav', '')
                for audio_path in config['audio_files']:
                    if Path(audio_path).stem == base_name:
                        separated_files[audio_path] = vocals_path
                        break
    
    # Update config
    config['vocals_dir'] = vocals_dir
    config['separated_files'] = separated_files
    config['separated'] = True
    
    # Save config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Backup to Drive
    drive_config = "/content/drive/MyDrive/F5TTS_Vietnamese/processing_config.json"
    with open(drive_config, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Restored {len(separated_files)} separated vocals files")
    print(f"📁 Location: {vocals_dir}")
    
    # Display summary
    print("\n" + "="*70)
    print("📊 Restored Separated Vocals Summary:")
    print("="*70)
    
    for original, vocals in list(separated_files.items())[:5]:
        print(f"   {Path(original).name}")
        print(f"   → {Path(vocals).name}")
        print()
    
    if len(separated_files) > 5:
        print(f"   ... and {len(separated_files) - 5} more")
    
    print("\n" + "="*70)
    print("✅ RESTORE COMPLETE! Skipping to next cell...")
    print("="*70)
    print("📝 Next Steps:")
    print("   → Run Cell 06 to detect speech segments (VAD)")
    print("="*70)
    
    # Skip the rest of this cell
    sys.exit(0)

# ------------------------------------------------------------------------------
# 1. User Choice: Separate or Skip
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🎵 Voice Separation Options...")
print("="*70)

print("""
⚠️  Voice separation using Demucs:

Pros:
  ✅ Removes background music
  ✅ Extracts clean vocals
  ✅ Better training quality

Cons:
  ⚠️  Takes time (~5-10 min per 30-min file)
  ⚠️  Downloads ~2GB model on first run
  ⚠️  May reduce audio quality slightly

📝 Skip if your audio is already clean (no music/noise)
""")

do_separation = input("\nRun voice separation? (y/n, default=y): ").strip().lower()
do_separation = do_separation != 'n'

if not do_separation:
    print("\n✅ Skipping voice separation")
    print("   Using original audio files")
    
    # Update config to use original files
    config['vocals_dir'] = config['upload_dir']
    config['separated'] = False
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n🎉 Proceeding to next step...")
    sys.exit(0)

# ------------------------------------------------------------------------------
# 2. Setup Output Directory
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("📁 Setting up output directory...")
print("="*70)

vocals_dir = "/content/processed/vocals"
os.makedirs(vocals_dir, exist_ok=True)

print(f"✅ Output directory: {vocals_dir}")

# ------------------------------------------------------------------------------
# 3. Process Each File
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🎵 Separating vocals...")
print("="*70)

separated_files = {}

for audio_file in tqdm(audio_files, desc="Processing files"):
    print(f"\n📁 Processing: {audio_file.name}")
    print("-" * 70)
    
    speaker_name = speakers[str(audio_file)]['name']
    
    # Create speaker-specific output dir
    speaker_output_dir = os.path.join(vocals_dir, speaker_name)
    os.makedirs(speaker_output_dir, exist_ok=True)
    
    try:
        # Run Demucs
        print("⏳ Running Demucs (this may take 5-10 minutes)...")
        
        # Demucs command
        cmd = [
            venv_python, "-m", "demucs.separate",
            "-n", "htdemucs",           # Model name
            "--two-stems", "vocals",    # Extract only vocals
            "-o", speaker_output_dir,   # Output directory
            str(audio_file)             # Input file
        ]
        
        # Run with progress
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            if 'it/s' in line or '%' in line:
                print(f"  {line.strip()}", end='\r')
        
        process.wait()
        
        if process.returncode != 0:
            print(f"\n❌ Demucs failed for {audio_file.name}")
            continue
        
        # Find output file
        basename = audio_file.stem
        vocals_path = Path(speaker_output_dir) / "htdemucs" / basename / "vocals.wav"
        
        if not vocals_path.exists():
            print(f"❌ Output not found: {vocals_path}")
            continue
        
        # Move to speaker directory with clean name
        final_path = Path(speaker_output_dir) / f"{basename}_vocals.wav"
        vocals_path.rename(final_path)
        
        separated_files[str(audio_file)] = str(final_path)
        
        print(f"\n✅ Vocals saved: {final_path.name}")
        
        # Get file size
        size_mb = final_path.stat().st_size / (1024 * 1024)
        print(f"   Size: {size_mb:.1f} MB")
        
    except Exception as e:
        print(f"\n❌ Error processing {audio_file.name}: {e}")
        continue

# ------------------------------------------------------------------------------
# 4. Update Configuration
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("💾 Updating configuration...")
print("="*70)

config['vocals_dir'] = vocals_dir
config['separated_files'] = separated_files
config['separated'] = True
config['separation_success'] = len(separated_files)
config['separation_failed'] = len(audio_files) - len(separated_files)

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

# Backup to Drive
drive_config = "/content/drive/MyDrive/F5TTS_Vietnamese/processing_config.json"
with open(drive_config, 'w') as f:
    json.dump(config, f, indent=2)

print(f"✅ Configuration updated")

# ------------------------------------------------------------------------------
# 4.5. Backup Separated Vocals to Drive
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("💾 Backing up separated vocals to Drive...")
print("="*70)

drive_vocals_backup = "/content/drive/MyDrive/F5TTS_Vietnamese/vocals_backup"

# Remove old backup if exists
if os.path.exists(drive_vocals_backup):
    print("⏳ Removing old backup...")
    shutil.rmtree(drive_vocals_backup)

# Copy vocals to Drive
print("⏳ Copying separated vocals to Drive...")
shutil.copytree(vocals_dir, drive_vocals_backup)

# Calculate backup size
backup_size = sum(
    os.path.getsize(os.path.join(dirpath, filename))
    for dirpath, dirnames, filenames in os.walk(drive_vocals_backup)
    for filename in filenames
) / (1024 * 1024)  # MB

print(f"✅ Backup complete!")
print(f"   Location: {drive_vocals_backup}")
print(f"   Files: {len(separated_files)}")
print(f"   Size: {backup_size:.1f} MB")

# ------------------------------------------------------------------------------
# 5. Display Summary
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("✅ VOICE SEPARATION COMPLETE!")
print("="*70)

print(f"""
📊 Summary:
   Total Files: {len(audio_files)}
   Successfully Separated: {len(separated_files)}
   Failed: {len(audio_files) - len(separated_files)}
   
📁 Output Directory:
   {vocals_dir}
   
📝 Separated Files:
""")

for original, vocals in separated_files.items():
    print(f"   {Path(original).name}")
    print(f"   → {Path(vocals).name}")
    print()

# Show audio samples
if separated_files:
    print("\n🎧 Play sample to verify quality:")
    
    # Get first vocals file
    first_vocals = list(separated_files.values())[0]
    
    print(f"\nSample: {Path(first_vocals).name}")
    
    # Try to display audio player
    try:
        from IPython.display import Audio, display
        display(Audio(first_vocals, rate=24000))
    except:
        print(f"   File: {first_vocals}")

print(f"""
📝 Next Steps:
   → Run Cell 06 to detect speech segments (VAD)
   → This will split audio into manageable chunks
   
⚠️  Note:
   - Separated files saved to: {vocals_dir}
   - Original files kept in: {config['upload_dir']}
   - All changes backed up to Google Drive
""")

print("="*70)
print("🎉 Ready to proceed to Cell 06!")
print("="*70)



