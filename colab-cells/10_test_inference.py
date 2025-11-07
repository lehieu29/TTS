"""
Cell 10: Test Inference
Mục đích:
  - Test trained model với inference
  - Generate speech từ text
  - Verify quality
"""

# ============================================================================
# CELL 10: TEST INFERENCE
# ============================================================================

print("🎤 Testing Trained Model...")

import os
import sys
import json
import subprocess
from pathlib import Path

# Use venv
venv_python = "/content/venv/bin/python"

# Load config
config_path = "/content/processing_config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

if not config.get('training_complete', False):
    print("❌ Training not complete!")
    print("   Please run Cell 09 first")
    sys.exit(1)

speakers = config['trained_speakers']

# Change to F5-TTS directory
os.chdir("/content/F5-TTS-Vietnamese")

# ------------------------------------------------------------------------------
# 1. Select Speaker
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("👤 Available Trained Speakers:")
print("="*70)

for i, speaker in enumerate(speakers, 1):
    model_path = f"/content/models/{speaker}/model.pt"
    if os.path.exists(model_path):
        print(f"   {i}. {speaker} ✅")
    else:
        print(f"   {i}. {speaker} ❌ (model not found)")

if len(speakers) == 1:
    selected_speaker = speakers[0]
    print(f"\n✅ Using: {selected_speaker}")
else:
    selection = input(f"\nSelect speaker (1-{len(speakers)}, default=1): ").strip()
    idx = int(selection) - 1 if selection.isdigit() else 0
    selected_speaker = speakers[idx]
    print(f"✅ Selected: {selected_speaker}")

# ------------------------------------------------------------------------------
# 2. Get Reference Audio
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🎧 Preparing Reference Audio...")
print("="*70)

model_dir = f"/content/models/{selected_speaker}"

# Use one of the training segments as reference
segments_dir = f"/content/data/{selected_speaker}_training/wavs"
ref_audio_files = list(Path(segments_dir).glob("*.wav"))[:5]  # Get first 5

if not ref_audio_files:
    print("❌ No reference audio found!")
    sys.exit(1)

print("📁 Available reference audio:")
for i, ref in enumerate(ref_audio_files, 1):
    print(f"   {i}. {ref.name}")

# Use first one as default
ref_audio = str(ref_audio_files[0])
print(f"\n✅ Using reference: {ref_audio_files[0].name}")

# Get reference text
ref_text_file = Path(ref_audio).with_suffix('.txt')
if ref_text_file.exists():
    with open(ref_text_file, 'r', encoding='utf-8') as f:
        ref_text = f.read().strip()
    print(f"   Reference text: \"{ref_text}\"")
else:
    ref_text = ""
    print("   ⚠️  No reference text found, will auto-transcribe")

# ------------------------------------------------------------------------------
# 3. Input Text for Generation
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("✍️  Text to Generate:")
print("="*70)

default_texts = [
    "xin chào các bạn, hôm nay tôi sẽ giới thiệu về trí tuệ nhân tạo",
    "việt nam là một đất nước xinh đẹp với văn hóa phong phú",
    "công nghệ đang phát triển rất nhanh trong những năm gần đây",
    "tôi rất vui được chia sẻ kiến thức với mọi người",
]

print("\n📝 Example texts:")
for i, text in enumerate(default_texts, 1):
    print(f"   {i}. {text}")

print(f"\n💡 Or enter your own Vietnamese text:")
gen_text_input = input("Enter text (or number 1-4, or Enter for example 1): ").strip()

if gen_text_input.isdigit() and 1 <= int(gen_text_input) <= len(default_texts):
    gen_text = default_texts[int(gen_text_input) - 1]
elif gen_text_input:
    gen_text = gen_text_input
else:
    gen_text = default_texts[0]

print(f"\n✅ Generating: \"{gen_text}\"")

# ------------------------------------------------------------------------------
# 4. Run Inference
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🎙️  Generating Speech...")
print("="*70)

output_dir = "/content/outputs"
os.makedirs(output_dir, exist_ok=True)

output_file = f"{output_dir}/{selected_speaker}_generated.wav"

# Inference command
cmd = [
    venv_python, "-m", "f5_tts.infer.infer_cli",
    "--model", "F5TTS_Base",
    "--ref_audio", ref_audio,
    "--ref_text", ref_text if ref_text else "",
    "--gen_text", gen_text,
    "--gen_file", output_file,
    "--vocab_file", f"{model_dir}/vocab.txt",
    "--ckpt_file", f"{model_dir}/model.pt",
    "--speed", "1.0",
    "--nfe_step", "32"
]

print("⏳ Generating... (this may take 5-10 seconds)")
print()

try:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120
    )
    
    if result.returncode == 0:
        print("✅ Speech generated successfully!")
        print(result.stdout)
    else:
        print("❌ Inference failed!")
        print(result.stderr)
        sys.exit(1)
        
except subprocess.TimeoutExpired:
    print("❌ Inference timeout!")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# ------------------------------------------------------------------------------
# 5. Play Generated Audio
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🎧 Generated Audio:")
print("="*70)

if os.path.exists(output_file):
    size_mb = os.path.getsize(output_file) / (1024**2)
    print(f"✅ File saved: {output_file}")
    print(f"   Size: {size_mb:.2f} MB")
    
    # Display audio player
    try:
        from IPython.display import Audio, display
        print("\n🔊 Playing audio:")
        display(Audio(output_file, rate=24000))
    except Exception as e:
        print(f"\n⚠️  Could not display audio player: {e}")
        print(f"   File saved at: {output_file}")
    
    # Save to Drive
    drive_output_dir = f"/content/drive/MyDrive/F5TTS_Vietnamese/outputs/{selected_speaker}"
    os.makedirs(drive_output_dir, exist_ok=True)
    
    import shutil
    drive_output_file = f"{drive_output_dir}/generated_{int(time.time())}.wav"
    shutil.copy(output_file, drive_output_file)
    
    print(f"\n✅ Also saved to Drive: {drive_output_file}")
    
else:
    print("❌ Output file not generated!")

# ------------------------------------------------------------------------------
# 6. Multiple Generations (Optional)
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🎨 Generate More?")
print("="*70)

generate_more = input("\nGenerate another text? (y/n, default=n): ").strip().lower()

if generate_more == 'y':
    print("\n📝 Enter text to generate:")
    new_text = input("> ").strip()
    
    if new_text:
        new_output = f"{output_dir}/{selected_speaker}_generated_{int(time.time())}.wav"
        
        cmd[-2] = new_output  # Update output file
        cmd[8] = new_text     # Update gen_text
        
        print(f"\n🎙️  Generating: \"{new_text}\"")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Generated!")
            
            try:
                from IPython.display import Audio, display
                display(Audio(new_output, rate=24000))
            except:
                pass
        else:
            print("❌ Failed!")

# ------------------------------------------------------------------------------
# 7. Display Summary
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("✅ INFERENCE TEST COMPLETE!")
print("="*70)

print(f"""
📊 Summary:
   Speaker: {selected_speaker}
   Model: {model_dir}/model.pt
   Reference: {Path(ref_audio).name}
   Generated: {gen_text}
   Output: {output_file}
   
📁 Output Files:
   Local: {output_dir}/
   Drive: /content/drive/MyDrive/F5TTS_Vietnamese/outputs/
   
🎯 Model Quality:
   Listen to the generated audio above
   ✅ Clear pronunciation?
   ✅ Natural prosody?
   ✅ Sounds like the speaker?
   
📝 Next Steps:
   → If quality is good: Run Cell 11 for Gradio UI!
   → If quality is poor: May need more training data
   
💡 Tips for Better Quality:
   - Use 50-100 hours of training data
   - Ensure accurate transcriptions
   - Use clean audio (no background noise)
   - Reference audio should be clear
""")

print("="*70)
print("🎉 Ready for Gradio UI! Proceed to Cell 11!")
print("="*70)



