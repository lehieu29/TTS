"""
Cell 04: Upload Audio & Initial Preparation
Mục đích:
  - Upload podcast/audio files
  - Basic validation
  - Organize files
"""

# ============================================================================
# CELL 04: UPLOAD AND PREPARE AUDIO FILES
# ============================================================================

print("📤 Upload and Prepare Audio Files...")

import os
import sys
from pathlib import Path
from google.colab import files
import shutil

# Load helper functions
sys.path.insert(0, '/content')
from colab_helpers import *

# ------------------------------------------------------------------------------
# 0. Check for Existing Configuration in Drive
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🔍 Checking for existing configuration in Drive...")
print("="*70)

import json

drive_config_path = "/content/drive/MyDrive/F5TTS_Vietnamese/processing_config.json"
drive_audio_dir = "/content/drive/MyDrive/F5TTS_Vietnamese/uploads"
use_existing = False

if os.path.exists(drive_config_path) and os.path.exists(drive_audio_dir):
    # Load existing config
    with open(drive_config_path, 'r', encoding='utf-8') as f:
        existing_config = json.load(f)
    
    # Check if there are audio files
    drive_files = list(Path(drive_audio_dir).glob("*.mp3")) + \
                  list(Path(drive_audio_dir).glob("*.wav")) + \
                  list(Path(drive_audio_dir).glob("*.flac"))
    
    if drive_files and existing_config.get('speakers'):
        print(f"✅ Found existing configuration in Drive!")
        print(f"📊 Configuration info:")
        print(f"   Audio files: {len(drive_files)}")
        print(f"   Speakers: {len(set(s['name'] for s in existing_config['speakers'].values()))}")
        
        if 'upload_stats' in existing_config:
            stats = existing_config['upload_stats']
            print(f"   Total duration: ~{stats.get('total_duration_min', 0):.1f} minutes")
        
        print()
        print("="*70)
        print("💬 Bạn muốn sử dụng config cũ hay upload audio mới?")
        print("="*70)
        print("   1. Sử dụng config và audio từ Drive (nhanh)")
        print("   2. Upload audio files mới (sẽ ghi đè config cũ)")
        print("="*70)
        
        user_choice = input("Lựa chọn của bạn (1/2, mặc định=1): ").strip()
        
        if user_choice == "2":
            print("\n✅ Sẽ upload audio mới và tạo config mới...")
            use_existing = False
        else:
            print("\n✅ Sử dụng config và audio từ Drive...")
            use_existing = True
    else:
        print("📝 Config exists but incomplete. Will create new one...")
        use_existing = False
else:
    print("📝 No existing configuration found. Will create new one...")
    use_existing = False

# ------------------------------------------------------------------------------
# Restore from Drive if user chose to
# ------------------------------------------------------------------------------
if use_existing:
    print("\n" + "="*70)
    print("📦 Restoring configuration and audio files from Drive...")
    print("="*70)
    
    # Create upload directory
    upload_dir = "/content/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    # Copy audio files from Drive
    drive_files = list(Path(drive_audio_dir).glob("*.mp3")) + \
                  list(Path(drive_audio_dir).glob("*.wav")) + \
                  list(Path(drive_audio_dir).glob("*.flac"))
    
    print(f"⏳ Copying {len(drive_files)} audio files from Drive...")
    for f in drive_files:
        dest = Path(upload_dir) / f.name
        if not dest.exists():
            shutil.copy(str(f), str(dest))
    
    print(f"✅ Copied {len(drive_files)} files to {upload_dir}")
    
    # Update paths in config to local paths
    config = existing_config.copy()
    
    # Update audio_files paths to local
    local_audio_files = [str(Path(upload_dir) / Path(p).name) for p in config['audio_files']]
    config['audio_files'] = local_audio_files
    config['upload_dir'] = upload_dir
    
    # Update speaker paths
    new_speakers = {}
    for old_path, speaker_info in config['speakers'].items():
        filename = Path(old_path).name
        new_path = str(Path(upload_dir) / filename)
        new_speakers[new_path] = speaker_info
    config['speakers'] = new_speakers
    
    # Save local config
    config_path = "/content/processing_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Configuration restored to: {config_path}")
    
    # Display summary
    print("\n" + "="*70)
    print("📊 Restored Configuration Summary:")
    print("="*70)
    
    unique_speakers = {}
    for audio_path, info in config['speakers'].items():
        name = info['name']
        if name not in unique_speakers:
            unique_speakers[name] = []
        unique_speakers[name].append(Path(audio_path).name)
    
    print(f"\n👥 Speakers:")
    for speaker, files in unique_speakers.items():
        print(f"   {speaker}: {len(files)} file(s)")
    
    if 'upload_stats' in config:
        stats = config['upload_stats']
        print(f"\n📊 Upload Stats:")
        print(f"   Total files: {stats.get('total_files', 0)}")
        print(f"   Total size: {stats.get('total_size_mb', 0):.1f} MB")
        print(f"   Total duration: ~{stats.get('total_duration_min', 0):.1f} minutes")
    
    print("\n" + "="*70)
    print("✅ RESTORE COMPLETE! Skipping to next cell...")
    print("="*70)
    print("📝 Next Steps:")
    print("   → Run Cell 05 to separate vocals (if needed)")
    print("   → Or skip to Cell 06 if audio is already clean")
    print("="*70)
    
    # Skip the rest of this cell
    import sys
    sys.exit(0)

# ------------------------------------------------------------------------------
# 1. Upload Audio Files
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("📤 Upload Audio Files...")
print("="*70)

print("""
📝 Instructions:
   You can upload MULTIPLE files for the SAME speaker:
   
   [Option 1] Upload from computer:
   - Click 'Choose Files' button below
   - Hold Ctrl/Cmd to select MULTIPLE files at once
   - All files will be used for training the same voice
   
   [Option 2] Use files from Google Drive:
   - Upload all files to: /content/drive/MyDrive/F5TTS_Vietnamese/uploads
   - All files in this folder will be processed automatically
   
⚠️  Notes:
   - Max file size per upload: ~200MB (Colab limit)
   - For larger files: use Google Drive (Option 2)
   - Supported formats: MP3, WAV, FLAC
   - More audio = Better voice quality!
   
🎯 Recommendations:
   - Minimum: 10 minutes of clean audio
   - Good: 30-60 minutes
   - Best: 1-3 hours
   - Quality > Quantity (clean audio is more important)
""")

# Option 1: Upload from local computer
print("\n[Option 1] Upload from computer:")
uploaded = files.upload()

if uploaded:
    print(f"\n✅ Uploaded {len(uploaded)} file(s)")
    
    # Move to uploads directory
    upload_dir = "/content/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    for filename, content in uploaded.items():
        filepath = os.path.join(upload_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(content)
        print(f"  ✓ Saved: {filename}")
else:
    print("⚠️  No files uploaded via Option 1")

# Option 2: Use files from Google Drive
print("\n[Option 2] Or use files from Google Drive:")
drive_audio_dir = "/content/drive/MyDrive/F5TTS_Vietnamese/uploads"

if os.path.exists(drive_audio_dir):
    drive_files = list(Path(drive_audio_dir).glob("*.mp3")) + \
                  list(Path(drive_audio_dir).glob("*.wav")) + \
                  list(Path(drive_audio_dir).glob("*.flac"))
    
    if drive_files:
        print(f"Found {len(drive_files)} audio files in Google Drive:")
        for f in drive_files[:10]:  # Show first 10
            print(f"  📁 {f.name}")
        if len(drive_files) > 10:
            print(f"  ... and {len(drive_files) - 10} more")
        
        # Copy to working directory
        upload_dir = "/content/uploads"
        for f in drive_files:
            shutil.copy(str(f), upload_dir)
        print(f"\n✅ Copied {len(drive_files)} files to working directory")
    else:
        print("⚠️  No audio files found in Drive uploads folder")
        print(f"   Upload files to: {drive_audio_dir}")
else:
    print(f"⚠️  Drive folder not found: {drive_audio_dir}")
    print("   It will be created after you upload files")

# ------------------------------------------------------------------------------
# 2. List Uploaded Files & Calculate Statistics
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("📁 Listing uploaded files...")
print("="*70)

upload_dir = "/content/uploads"
audio_files = list(Path(upload_dir).glob("*.mp3")) + \
              list(Path(upload_dir).glob("*.wav")) + \
              list(Path(upload_dir).glob("*.flac"))

if not audio_files:
    print("❌ No audio files found!")
    print("   Please upload files and try again")
    sys.exit(1)

print(f"\nFound {len(audio_files)} audio file(s):")

# Calculate total size and duration
import subprocess
total_size_mb = 0
total_duration_sec = 0

for f in audio_files:
    size_mb = f.stat().st_size / (1024 * 1024)
    total_size_mb += size_mb
    
    # Get duration using ffprobe
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', str(f)],
            capture_output=True, text=True, timeout=10
        )
        duration = float(result.stdout.strip() or 0)
        total_duration_sec += duration
        print(f"  🎵 {f.name} ({size_mb:.1f} MB, {duration/60:.1f} min)")
    except Exception as e:
        print(f"  🎵 {f.name} ({size_mb:.1f} MB, duration unknown)")

total_duration_min = total_duration_sec / 60

print(f"\n📊 Upload Summary:")
print(f"   Total files: {len(audio_files)}")
print(f"   Total size: {total_size_mb:.1f} MB")
print(f"   Total duration: ~{total_duration_min:.1f} minutes")

# Validation warnings
if total_duration_min < 10:
    print(f"\n⚠️  WARNING: Very low total duration!")
    print(f"   Recommended: At least 30 minutes for quality results")
    print(f"   Current: {total_duration_min:.1f} minutes")
    print(f"   💡 Consider uploading more audio files for better quality")
elif total_duration_min < 30:
    print(f"\n⚠️  Note: Duration is on the low side")
    print(f"   Recommended: 30-60 minutes for best results")
    print(f"   Current: {total_duration_min:.1f} minutes")
else:
    print(f"\n✅ Duration looks good for training!")

# ------------------------------------------------------------------------------
# 3. Collect Speaker Information
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("👤 Speaker Information...")
print("="*70)

# Dictionary to store speaker info
speakers = {}

print("""
📝 For each audio file, please provide:
   - Speaker name (e.g., "nguyen_van_a")
   - Description (optional)
   
⚠️  Note: Speaker name will be used for model naming
   Use lowercase, no spaces (use underscore)
""")

# Simple input for now
default_speaker = input("\nEnter default speaker name for all files (or press Enter to skip): ").strip()

if default_speaker:
    print(f"\n✅ Will use speaker name: '{default_speaker}' for all files")
    
    for audio_file in audio_files:
        speakers[str(audio_file)] = {
            'name': default_speaker,
            'source_file': audio_file.name,
            'description': ''
        }
else:
    # Manual input for each file
    for audio_file in audio_files:
        print(f"\n📁 File: {audio_file.name}")
        speaker_name = input("  Speaker name: ").strip()
        
        if not speaker_name:
            speaker_name = audio_file.stem  # Use filename as default
        
        speakers[str(audio_file)] = {
            'name': speaker_name,
            'source_file': audio_file.name,
            'description': ''
        }

# ------------------------------------------------------------------------------
# 4. Save Configuration
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("💾 Saving Configuration...")
print("="*70)

import json

config = {
    'upload_dir': upload_dir,
    'audio_files': [str(f) for f in audio_files],
    'speakers': speakers,
    'total_files': len(audio_files),
    'upload_stats': {
        'total_files': len(audio_files),
        'total_size_mb': total_size_mb,
        'total_duration_min': total_duration_min
    }
}

config_path = "/content/processing_config.json"
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(f"✅ Configuration saved to: {config_path}")

# Also save to Drive for persistence
drive_config_path = "/content/drive/MyDrive/F5TTS_Vietnamese/processing_config.json"
shutil.copy(config_path, drive_config_path)
print(f"✅ Backup saved to Drive: {drive_config_path}")

# ------------------------------------------------------------------------------
# 5. Display Summary
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("✅ UPLOAD AND PREPARATION COMPLETE!")
print("="*70)

print(f"""
📊 Summary:
   Total Files: {len(audio_files)}
   Speakers: {len(set(s['name'] for s in speakers.values()))}
   
📁 Files Location:
   Working Dir: {upload_dir}
   Config: {config_path}
   
👥 Speakers:
""")

# List unique speakers
unique_speakers = {}
for audio_path, info in speakers.items():
    name = info['name']
    if name not in unique_speakers:
        unique_speakers[name] = []
    unique_speakers[name].append(Path(audio_path).name)

for speaker, files in unique_speakers.items():
    print(f"   {speaker}: {len(files)} file(s)")
    for f in files[:3]:  # Show first 3
        print(f"     - {f}")
    if len(files) > 3:
        print(f"     ... and {len(files) - 3} more")

print(f"""
📝 Next Steps:
   → Run Cell 05 to separate vocals (if audio has background music)
   → Or skip to Cell 06 if audio is already clean
   
⚠️  Important:
   - Voice separation takes ~5-10 min per 30-min file
   - Files are saved in working directory and Drive
   - You can stop and resume processing
""")

print("="*70)
print("🎉 Ready to proceed to Cell 05!")
print("="*70)

# Export config for next cells
print("\n📌 Config loaded in memory for next cells")



