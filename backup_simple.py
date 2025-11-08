"""
SIMPLE BACKUP - Copy Checkpoints to Drive
==========================================
Chạy trong Colab để backup checkpoints nhanh
"""

import os
import shutil
from pathlib import Path

# Config
DRIVE_BASE = "/content/drive/MyDrive/F5TTS_Vietnamese"

print("📦 BACKING UP CHECKPOINTS...")

# Mount Drive check
if not os.path.exists("/content/drive"):
    print("❌ Please mount Drive first!")
    print("Run: from google.colab import drive; drive.mount('/content/drive')")
    exit(1)

# Find and copy checkpoints
copied = 0

# 1. Training checkpoints
ckpts_dir = "/content/F5-TTS-Vietnamese/ckpts"
if os.path.exists(ckpts_dir):
    for speaker_dir in os.listdir(ckpts_dir):
        if "_training" in speaker_dir:
            speaker = speaker_dir.replace("_training", "")
            source = os.path.join(ckpts_dir, speaker_dir)
            dest = f"{DRIVE_BASE}/checkpoints/{speaker}"
            
            os.makedirs(dest, exist_ok=True)
            
            for file in os.listdir(source):
                if file.endswith('.pt'):
                    shutil.copy2(os.path.join(source, file), os.path.join(dest, file))
                    copied += 1
                    print(f"✅ {speaker}/{file}")

# 2. Organized models
models_dir = "/content/models"
if os.path.exists(models_dir):
    for speaker in os.listdir(models_dir):
        source = os.path.join(models_dir, speaker)
        dest = f"{DRIVE_BASE}/models/{speaker}"
        
        os.makedirs(dest, exist_ok=True)
        
        for file in os.listdir(source):
            shutil.copy2(os.path.join(source, file), os.path.join(dest, file))
            copied += 1
            print(f"✅ {speaker}/{file}")

print(f"\n✅ Done! Copied {copied} files to Drive")
print(f"📂 Location: {DRIVE_BASE}/")
