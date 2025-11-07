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



