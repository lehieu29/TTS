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

# Change to F5-TTS directory
os.chdir("/content/F5-TTS-Vietnamese")

# ------------------------------------------------------------------------------
# 0. Auto-Load Checkpoints from Drive
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🔍 Checking for trained models...")
print("="*70)

import shutil
import time

drive_checkpoints_dir = "/content/drive/MyDrive/F5TTS_Vietnamese/checkpoints"
local_models_dir = "/content/models"

# Check if we have trained_speakers in config or checkpoints in Drive
trained_speakers = config.get('trained_speakers', [])
speakers_list = config.get('speakers_list', [])

# If no trained_speakers but we have checkpoints in Drive, auto-load them
if not trained_speakers and os.path.exists(drive_checkpoints_dir):
    print("📝 No local models found. Checking Drive for checkpoints...")
    
    # List speakers with checkpoints in Drive
    drive_speakers = []
    if os.path.exists(drive_checkpoints_dir):
        for speaker_dir in os.listdir(drive_checkpoints_dir):
            speaker_path = os.path.join(drive_checkpoints_dir, speaker_dir)
            if os.path.isdir(speaker_path):
                # Check if there are .pt checkpoint files
                checkpoints = list(Path(speaker_path).glob("*.pt"))
                if checkpoints:
                    drive_speakers.append(speaker_dir)
    
    if drive_speakers:
        print(f"✅ Found checkpoints in Drive for {len(drive_speakers)} speaker(s)")
        print(f"   Speakers: {', '.join(drive_speakers)}")
        print()
        print("="*70)
        print("💬 Tải checkpoint từ Drive xuống local?")
        print("="*70)
        print("   Checkpoint sẽ được copy vào /content/models/")
        print()
        
        load_checkpoints = input("Load checkpoints? (y/n, default=y): ").strip().lower()
        
        if load_checkpoints != 'n':
            print("\n📦 Loading checkpoints from Drive...")
            
            os.makedirs(local_models_dir, exist_ok=True)
            loaded_speakers = []
            
            for speaker in drive_speakers:
                speaker_checkpoint_dir = os.path.join(drive_checkpoints_dir, speaker)
                speaker_model_dir = os.path.join(local_models_dir, speaker)
                
                # Find latest checkpoint
                checkpoints = sorted(
                    Path(speaker_checkpoint_dir).glob("*.pt"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                )
                
                if checkpoints:
                    latest_checkpoint = checkpoints[0]
                    
                    # Create speaker model directory
                    os.makedirs(speaker_model_dir, exist_ok=True)
                    
                    # Copy checkpoint as model.pt
                    dest_model = os.path.join(speaker_model_dir, "model.pt")
                    print(f"⏳ Copying {speaker}: {latest_checkpoint.name}...")
                    shutil.copy(str(latest_checkpoint), dest_model)
                    
                    # Also copy vocab.txt if exists in training_data
                    training_dir = f"/content/data/{speaker}_training"
                    if not os.path.exists(training_dir):
                        # Try to restore from Drive backup
                        drive_training_backup = f"/content/drive/MyDrive/F5TTS_Vietnamese/training_data/{speaker}_training"
                        if os.path.exists(drive_training_backup):
                            print(f"   Restoring training data for vocab...")
                            os.makedirs(training_dir, exist_ok=True)
                            # Copy only vocab.txt and a few wavs for reference
                            vocab_src = os.path.join(drive_training_backup, "vocab.txt")
                            if os.path.exists(vocab_src):
                                shutil.copy(vocab_src, training_dir)
                            
                            # Copy wavs directory for reference audio
                            wavs_src = os.path.join(drive_training_backup, "wavs")
                            wavs_dst = os.path.join(training_dir, "wavs")
                            if os.path.exists(wavs_src):
                                shutil.copytree(wavs_src, wavs_dst)
                    
                    # Copy vocab.txt to model dir
                    vocab_src = os.path.join(training_dir, "vocab.txt")
                    vocab_dst = os.path.join(speaker_model_dir, "vocab.txt")
                    if os.path.exists(vocab_src):
                        shutil.copy(vocab_src, vocab_dst)
                    
                    print(f"   ✅ Loaded: {dest_model}")
                    loaded_speakers.append(speaker)
            
            # Update config
            config['trained_speakers'] = loaded_speakers
            config['training_complete'] = True
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            trained_speakers = loaded_speakers
            
            print(f"\n✅ Loaded {len(loaded_speakers)} model(s) from Drive")
        else:
            print("\n❌ Cannot proceed without models. Please train a model first (Cell 09)")
            sys.exit(1)
    else:
        print("❌ No checkpoints found in Drive!")
        print("   Please train a model first (Cell 09)")
        sys.exit(1)

elif not trained_speakers:
    print("❌ No trained models found!")
    print("   Please run Cell 09 to train a model first")
    print("   Or check if checkpoints exist in Drive")
    sys.exit(1)
else:
    print(f"✅ Found {len(trained_speakers)} trained speaker(s) in config")

# Final check: verify models exist locally
speakers = trained_speakers
missing_models = []

for speaker in speakers:
    model_path = f"{local_models_dir}/{speaker}/model.pt"
    if not os.path.exists(model_path):
        missing_models.append(speaker)

if missing_models:
    print(f"\n⚠️  WARNING: Some models missing locally:")
    for speaker in missing_models:
        print(f"   - {speaker}")
    
    # Try to load from ckpts directory first (from recent training)
    print("\n   Attempting to load from training checkpoints...")
    for speaker in missing_models[:]:  # Use slice to allow removal during iteration
        ckpt_path = f"/content/F5-TTS-Vietnamese/ckpts/{speaker}_training/model_last.pt"
        if os.path.exists(ckpt_path):
            training_dir = f"/content/data/{speaker}_training"
            vocab_path = f"{training_dir}/vocab.txt"
            
            # Copy checkpoint and vocab to models directory
            os.makedirs(f"{local_models_dir}/{speaker}", exist_ok=True)
            shutil.copy(ckpt_path, f"{local_models_dir}/{speaker}/model.pt")
            if os.path.exists(vocab_path):
                shutil.copy(vocab_path, f"{local_models_dir}/{speaker}/vocab.txt")
            
            print(f"   ✅ Loaded {speaker} from training checkpoints")
            missing_models.remove(speaker)
    
    # Try to load remaining missing models from Drive
    if missing_models:
        print("\n   Attempting to load from Drive...")
        for speaker in missing_models:
            drive_checkpoint_path = f"{drive_checkpoints_dir}/{speaker}"
            if os.path.exists(drive_checkpoint_path):
                checkpoints = sorted(
                    Path(drive_checkpoint_path).glob("*.pt"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                )
                if checkpoints:
                    os.makedirs(f"{local_models_dir}/{speaker}", exist_ok=True)
                    shutil.copy(str(checkpoints[0]), f"{local_models_dir}/{speaker}/model.pt")
                    print(f"   ✅ Loaded {speaker} from Drive")

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

output_filename = f"{selected_speaker}_generated.wav"

# Inference command
cmd = [
    venv_python, "-m", "f5_tts.infer.infer_cli",
    "--model", "F5TTS_Base",
    "--ref_audio", ref_audio,
    "--ref_text", ref_text if ref_text else "",
    "--gen_text", gen_text,
    "--output_dir", output_dir,          # FIX: Correct output directory parameter
    "--output_file", output_filename,     # FIX: Correct output filename parameter
    "--vocab_file", f"{model_dir}/vocab.txt",
    "--ckpt_file", f"{model_dir}/model.pt",
    "--speed", "1.0",
    "--nfe_step", "32"
]

# Full output path for checking later
output_file = f"{output_dir}/{output_filename}"

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



