"""
Backup Cell 9 Checkpoints to Google Drive
==========================================
Script tự động tìm và backup checkpoints từ training vào Google Drive

Usage in Colab:
    %run /content/F5-TTS-Vietnamese/colab-cells/backup_checkpoints_to_drive.py
"""

import os
import shutil
import json
from pathlib import Path

print("=" * 70)
print("📦 BACKUP CHECKPOINTS TO GOOGLE DRIVE")
print("=" * 70)

# ------------------------------------------------------------------------------
# 1. Check Drive is Mounted
# ------------------------------------------------------------------------------
drive_base = "/content/drive/MyDrive/F5TTS_Vietnamese"

if not os.path.exists("/content/drive"):
    print("\n❌ ERROR: Google Drive not mounted!")
    print("   Please run: from google.colab import drive; drive.mount('/content/drive')")
    exit(1)

print(f"\n✅ Drive mounted: {drive_base}")

# ------------------------------------------------------------------------------
# 2. Load Config to Get Speaker Names
# ------------------------------------------------------------------------------
config_path = "/content/processing_config.json"

if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    trained_speakers = config.get('trained_speakers', [])
    print(f"✅ Found {len(trained_speakers)} trained speaker(s): {trained_speakers}")
else:
    print("\n⚠️  Config not found, will search for all speakers")
    trained_speakers = []

# ------------------------------------------------------------------------------
# 3. Find All Checkpoint Directories
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("🔍 SEARCHING FOR CHECKPOINTS...")
print("=" * 70)

checkpoint_sources = []

# Search 1: Training checkpoints
ckpt_base = "/content/F5-TTS-Vietnamese/ckpts"
if os.path.exists(ckpt_base):
    for item in os.listdir(ckpt_base):
        item_path = os.path.join(ckpt_base, item)
        if os.path.isdir(item_path) and "_training" in item:
            speaker_name = item.replace("_training", "")
            checkpoint_sources.append({
                'speaker': speaker_name,
                'source_dir': item_path,
                'type': 'checkpoints'
            })
            print(f"\n✅ Found training checkpoints: {item}")
            
            # Count files
            pt_files = list(Path(item_path).glob("*.pt"))
            total_size = sum(f.stat().st_size for f in pt_files) / (1024**2)
            print(f"   Files: {len(pt_files)} checkpoints ({total_size:.1f} MB)")

# Search 2: Organized models
models_base = "/content/models"
if os.path.exists(models_base):
    for item in os.listdir(models_base):
        item_path = os.path.join(models_base, item)
        if os.path.isdir(item_path):
            checkpoint_sources.append({
                'speaker': item,
                'source_dir': item_path,
                'type': 'models'
            })
            print(f"\n✅ Found organized model: {item}")
            
            # List files
            for file in ["model.pt", "vocab.txt", "config.json"]:
                filepath = os.path.join(item_path, file)
                if os.path.exists(filepath):
                    size_mb = os.path.getsize(filepath) / (1024**2)
                    print(f"   - {file}: {size_mb:.2f} MB")

if not checkpoint_sources:
    print("\n❌ No checkpoints found!")
    print("\n💡 Possible locations:")
    print("   - /content/F5-TTS-Vietnamese/ckpts/{speaker}_training/")
    print("   - /content/models/{speaker}/")
    exit(1)

# ------------------------------------------------------------------------------
# 4. Backup to Drive
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("📤 BACKING UP TO DRIVE...")
print("=" * 70)

backup_summary = []

for source in checkpoint_sources:
    speaker = source['speaker']
    source_dir = source['source_dir']
    backup_type = source['type']
    
    print(f"\n📁 Processing: {speaker} ({backup_type})")
    
    # Determine destination
    if backup_type == 'checkpoints':
        dest_dir = f"{drive_base}/checkpoints/{speaker}"
    else:  # models
        dest_dir = f"{drive_base}/models/{speaker}"
    
    # Create destination directory
    os.makedirs(dest_dir, exist_ok=True)
    
    # Copy files
    files_copied = 0
    total_size = 0
    
    try:
        for item in os.listdir(source_dir):
            source_file = os.path.join(source_dir, item)
            dest_file = os.path.join(dest_dir, item)
            
            if os.path.isfile(source_file):
                # Check if already exists and same size
                if os.path.exists(dest_file):
                    source_size = os.path.getsize(source_file)
                    dest_size = os.path.getsize(dest_file)
                    
                    if source_size == dest_size:
                        print(f"   ⏭️  Skip (exists): {item}")
                        continue
                
                # Copy file
                print(f"   📄 Copying: {item}...", end=" ")
                shutil.copy2(source_file, dest_file)
                
                file_size = os.path.getsize(dest_file) / (1024**2)
                total_size += file_size
                files_copied += 1
                print(f"✅ ({file_size:.1f} MB)")
        
        backup_summary.append({
            'speaker': speaker,
            'type': backup_type,
            'files': files_copied,
            'size_mb': total_size,
            'destination': dest_dir,
            'success': True
        })
        
        print(f"   ✅ Backed up {files_copied} files ({total_size:.1f} MB)")
        print(f"   📂 Destination: {dest_dir}")
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        backup_summary.append({
            'speaker': speaker,
            'type': backup_type,
            'success': False,
            'error': str(e)
        })

# ------------------------------------------------------------------------------
# 5. Summary
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("📊 BACKUP SUMMARY")
print("=" * 70)

success_count = sum(1 for item in backup_summary if item.get('success', False))
total_files = sum(item.get('files', 0) for item in backup_summary if item.get('success', False))
total_size = sum(item.get('size_mb', 0) for item in backup_summary if item.get('success', False))

print(f"\n✅ Successfully backed up: {success_count}/{len(backup_summary)} speakers")
print(f"📄 Total files copied: {total_files}")
print(f"💾 Total size: {total_size:.1f} MB")

print("\n📂 Backup locations on Drive:")
for item in backup_summary:
    if item.get('success', False):
        print(f"   ✅ {item['speaker']} ({item['type']})")
        print(f"      {item['destination']}")
        print(f"      {item['files']} files, {item['size_mb']:.1f} MB")

if success_count < len(backup_summary):
    print("\n⚠️  Some backups failed:")
    for item in backup_summary:
        if not item.get('success', False):
            print(f"   ❌ {item['speaker']}: {item.get('error', 'Unknown error')}")

# ------------------------------------------------------------------------------
# 6. Verification
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("🔍 VERIFYING BACKUP...")
print("=" * 70)

for item in backup_summary:
    if item.get('success', False):
        dest_dir = item['destination']
        speaker = item['speaker']
        
        print(f"\n📁 {speaker}:")
        
        if item['type'] == 'checkpoints':
            # Check for important checkpoint files
            model_last = os.path.join(dest_dir, "model_last.pt")
            if os.path.exists(model_last):
                size_mb = os.path.getsize(model_last) / (1024**2)
                print(f"   ✅ model_last.pt: {size_mb:.1f} MB")
            else:
                # Check for any .pt files
                pt_files = list(Path(dest_dir).glob("*.pt"))
                if pt_files:
                    print(f"   ✅ {len(pt_files)} checkpoint files found")
                else:
                    print(f"   ⚠️  No .pt files found!")
        
        else:  # models
            # Check for essential files
            for filename in ["model.pt", "vocab.txt"]:
                filepath = os.path.join(dest_dir, filename)
                if os.path.exists(filepath):
                    size_mb = os.path.getsize(filepath) / (1024**2)
                    print(f"   ✅ {filename}: {size_mb:.2f} MB")
                else:
                    print(f"   ⚠️  {filename}: NOT FOUND")

# ------------------------------------------------------------------------------
# 7. Next Steps
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("✅ BACKUP COMPLETE!")
print("=" * 70)

print("\n💡 Next steps:")
print("   1. Verify files on Google Drive web interface")
print("   2. When restarting Colab, run Cell 10 or 11")
print("   3. Models will auto-load from Drive")

print("\n📂 Drive structure:")
print(f"   {drive_base}/")
print(f"   ├── checkpoints/")
for item in backup_summary:
    if item.get('success', False) and item['type'] == 'checkpoints':
        print(f"   │   └── {item['speaker']}/")
print(f"   └── models/")
for item in backup_summary:
    if item.get('success', False) and item['type'] == 'models':
        print(f"       └── {item['speaker']}/")

print("\n" + "=" * 70)
