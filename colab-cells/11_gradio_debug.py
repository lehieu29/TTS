"""
Cell 11 Debug - Test Gradio Inference Logic
===========================================
Mục đích: Debug lỗi inference bằng cách chạy trực tiếp trên terminal
"""

import os
import sys
import json
import subprocess
import time
import shutil
from pathlib import Path

print("🔍 Cell 11 Debug - Testing Inference Logic")
print("="*70)

# ------------------------------------------------------------------------------
# 1. Load Configuration
# ------------------------------------------------------------------------------
config_path = "/content/processing_config.json"
if not os.path.exists(config_path):
    print("❌ Config file not found!")
    print("   Please run previous cells first")
    sys.exit(1)

with open(config_path, 'r') as f:
    config = json.load(f)

trained_speakers = config.get('trained_speakers', [])
if not trained_speakers:
    print("❌ No trained speakers found in config")
    sys.exit(1)

print(f"✅ Found {len(trained_speakers)} trained speaker(s): {trained_speakers}")

# ------------------------------------------------------------------------------
# 2. Load Model and Prepare Speaker Data
# ------------------------------------------------------------------------------
local_models_dir = "/content/models"
venv_python = "/content/venv/bin/python"

# Check if model exists, if not try to load from training checkpoints
speaker = trained_speakers[0]  # Use first speaker for test
model_dir = f"{local_models_dir}/{speaker}"
model_path = f"{model_dir}/model.pt"
vocab_path = f"{model_dir}/vocab.txt"

if not os.path.exists(model_path):
    print(f"\n⚠️  Model not found at: {model_path}")
    print("   Attempting to load from training checkpoints...")
    
    # Try to load from training checkpoints
    ckpt_path = f"/content/F5-TTS-Vietnamese/ckpts/{speaker}_training/model_last.pt"
    training_dir = f"/content/data/{speaker}_training"
    training_vocab_path = f"{training_dir}/vocab.txt"
    
    if os.path.exists(ckpt_path):
        print(f"   ✅ Found checkpoint: {ckpt_path}")
        
        # Copy to models directory
        os.makedirs(model_dir, exist_ok=True)
        shutil.copy(ckpt_path, model_path)
        if os.path.exists(training_vocab_path):
            shutil.copy(training_vocab_path, vocab_path)
        
        print(f"   ✅ Copied to: {model_path}")
    else:
        print(f"   ❌ Checkpoint not found at: {ckpt_path}")
        sys.exit(1)

# Verify files exist
print(f"\n📋 Checking files:")
print(f"   Model: {model_path} - {'✅' if os.path.exists(model_path) else '❌'}")
print(f"   Vocab: {vocab_path} - {'✅' if os.path.exists(vocab_path) else '❌'}")

if not os.path.exists(model_path) or not os.path.exists(vocab_path):
    print("\n❌ Required files missing!")
    sys.exit(1)

# ------------------------------------------------------------------------------
# 3. Get Reference Audio
# ------------------------------------------------------------------------------
print(f"\n🎧 Preparing reference audio...")
segments_dir = f"/content/data/{speaker}_training/wavs"

if not os.path.exists(segments_dir):
    print(f"❌ Segments directory not found: {segments_dir}")
    sys.exit(1)

ref_audios = list(Path(segments_dir).glob("*.wav"))
if not ref_audios:
    print(f"❌ No audio files found in: {segments_dir}")
    sys.exit(1)

ref_audio = str(ref_audios[0])
print(f"   ✅ Reference audio: {Path(ref_audio).name}")

# Get reference text from metadata
metadata_path = f"/content/data/{speaker}_training/metadata.csv"
ref_text = ""

if os.path.exists(metadata_path):
    import csv
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            if len(row) >= 2:
                audio_file = row[0]
                if Path(ref_audio).stem in audio_file:
                    ref_text = row[1]
                    break

if not ref_text:
    print("   ⚠️  Reference text not found in metadata, using placeholder")
    ref_text = "xin chào"
else:
    print(f"   ✅ Reference text: {ref_text[:50]}...")

# ------------------------------------------------------------------------------
# 4. Test Inference
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🎙️  TESTING INFERENCE")
print("="*70)

# Test text
gen_text = "xin chào các bạn, hôm nay tôi sẽ giới thiệu về trí tuệ nhân tạo"
print(f"\n📝 Text to generate:")
print(f"   {gen_text}")

# Output
output_dir = "/content/outputs"
os.makedirs(output_dir, exist_ok=True)
output_filename = f"{speaker}_debug_test.wav"

# Build inference command
cmd = [
    venv_python, "-m", "f5_tts.infer.infer_cli",
    "--model", "F5TTS_Base",
    "--ref_audio", ref_audio,
    "--ref_text", ref_text,
    "--gen_text", gen_text,
    "--output_dir", output_dir,
    "--output_file", output_filename,
    "--vocab_file", vocab_path,
    "--ckpt_file", model_path,
    "--speed", "1.0",
    "--nfe_step", "32"
]

print("\n⚙️  Inference command:")
print("   " + " ".join([f'"{arg}"' if " " in arg else arg for arg in cmd]))

# Run inference
print("\n⏳ Running inference...")
print("-" * 70)

try:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        cwd="/content/F5-TTS-Vietnamese"
    )
    
    # Print stdout
    if result.stdout:
        print("📤 STDOUT:")
        print(result.stdout)
    
    # Print stderr
    if result.stderr:
        print("\n📤 STDERR:")
        print(result.stderr)
    
    # Check result
    print("\n" + "-" * 70)
    print(f"Return code: {result.returncode}")
    
    output_file = f"{output_dir}/{output_filename}"
    
    if result.returncode == 0:
        if os.path.exists(output_file):
            size_mb = os.path.getsize(output_file) / (1024**2)
            print(f"\n✅ SUCCESS!")
            print(f"   Output file: {output_file}")
            print(f"   Size: {size_mb:.2f} MB")
            
            # Try to play audio in Colab
            try:
                from IPython.display import Audio, display
                print("\n🔊 Playing audio:")
                display(Audio(output_file, rate=24000))
            except Exception as e:
                print(f"   ⚠️  Cannot display audio: {e}")
        else:
            print(f"\n⚠️  WARNING: Command succeeded but output file not found!")
            print(f"   Expected: {output_file}")
    else:
        print(f"\n❌ FAILED!")
        print(f"   Inference command returned error code: {result.returncode}")
        
except subprocess.TimeoutExpired:
    print("\n❌ TIMEOUT!")
    print("   Inference took longer than 120 seconds")
    
except Exception as e:
    print(f"\n❌ EXCEPTION!")
    print(f"   Error type: {type(e).__name__}")
    print(f"   Error message: {str(e)}")
    
    # Print traceback
    import traceback
    print("\n📋 Full traceback:")
    traceback.print_exc()

print("\n" + "="*70)
print("🏁 Debug test complete!")
print("="*70)
print("\n💡 Next steps:")
print("   1. Check the output above for errors")
print("   2. If successful, the issue is with Gradio integration")
print("   3. If failed, fix the inference command/parameters")
print("   4. Once working, you can use Cell 11 Gradio normally")
