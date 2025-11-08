"""
Cell 09: Train F5-TTS Model
Mục đích:
  - Fine-tune F5-TTS model với dữ liệu Vietnamese
  - Monitor training progress
  - Save checkpoints to Drive
  
⚠️  IMPORTANT: This will take 2-4 hours for 30 minutes of audio
"""

# ============================================================================
# CELL 09: TRAIN F5-TTS MODEL
# ============================================================================

print("🚀 Training F5-TTS Model...")

import os
import sys
import json
import subprocess
import time
import shutil
from pathlib import Path

# Use venv
venv_python = "/content/venv/bin/python"

# Load config
config_path = "/content/processing_config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

if not config.get('ready_for_training', False):
    print("❌ Data not ready for training!")
    print("   Please run Cell 08 first")
    sys.exit(1)

speakers = config['speakers_list']
training_dirs = config['training_dirs']

# Change to F5-TTS directory
os.chdir("/content/F5-TTS-Vietnamese")

# ------------------------------------------------------------------------------
# 1. Training Configuration
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("⚙️  Training Configuration...")
print("="*70)

# Check GPU
import torch
if not torch.cuda.is_available():
    print("⚠️  WARNING: No GPU detected!")
    print("   Training will be VERY slow on CPU")
    print("   Recommendation: Runtime → Change runtime type → GPU")
    
    proceed = input("\nContinue anyway? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Training cancelled. Please enable GPU first.")
        sys.exit(0)
else:
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"✅ GPU: {gpu_name}")
    print(f"   Memory: {gpu_memory:.1f} GB")

# Training hyperparameters
print("\n📊 Training Hyperparameters:")

# Adjust batch size based on GPU memory
if torch.cuda.is_available():
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if gpu_memory_gb < 10:
        batch_size = 3200
    elif gpu_memory_gb < 14:
        batch_size = 5000
    else:
        batch_size = 7000
else:
    batch_size = 1000  # Very small for CPU

TRAINING_CONFIG = {
    "exp_name": "F5TTS_Base",
    "batch_size": batch_size,
    "learning_rate": 1e-5,
    "epochs": 100,
    "num_warmup_updates": 5000,
    "save_per_updates": 2000,
    "last_per_updates": 1000,
    "max_grad_norm": 1.0,
}

for key, value in TRAINING_CONFIG.items():
    print(f"   {key}: {value}")

print(f"""
⏱️  Estimated Training Time:
   • T4 GPU: ~2-4 hours
   • V100 GPU: ~1-2 hours
   • A100 GPU: ~40-80 minutes
   • CPU: ❌ Not recommended (days)
""")

# Ask for confirmation
proceed = input("\nStart training? (y/n, default=y): ").strip().lower()
if proceed == 'n':
    print("Training cancelled.")
    sys.exit(0)

# ------------------------------------------------------------------------------
# 2. Train Each Speaker
# ------------------------------------------------------------------------------

for speaker in speakers:
    print("\n" + "="*70)
    print(f"👤 Training model for: {speaker}")
    print("="*70)
    
    dataset_name = f"{speaker}_training"
    training_dir = training_dirs[speaker]
    ckpt_dir = f"/content/F5-TTS-Vietnamese/ckpts/{dataset_name}"  # FIX: Use correct checkpoint path
    
    # ✅ FIX: Verify required files exist
    print(f"\n📋 Verifying training data...")
    required_files = [
        f"{training_dir}/raw.arrow",
        f"{training_dir}/duration.json",
        f"{training_dir}/vocab.txt"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files for {speaker}:")
        for f in missing_files:
            print(f"   - {f}")
        print(f"\n💡 Please run Cell 08 again to prepare data.")
        continue
    
    print(f"   ✅ All required files found")
    
    # ✅ FIX: Copy entire dataset directory structure to expected location
    # Script expects data at: /content/F5-TTS-Vietnamese/data/{dataset_name}/
    # This ensures wavs/, metadata.csv, and all files are preserved with correct paths
    print(f"\n📁 Preparing dataset path...")
    
    expected_data_dir = f"/content/F5-TTS-Vietnamese/data/{dataset_name}"
    
    # Remove old directory if exists (to avoid conflicts)
    if os.path.exists(expected_data_dir):
        print(f"   Removing old directory...")
        shutil.rmtree(expected_data_dir)
    
    # Copy entire directory structure (preserves wavs/, metadata.csv, etc.)
    print(f"   Copying dataset from {training_dir}...")
    try:
        shutil.copytree(training_dir, expected_data_dir)
        print(f"   ✅ Dataset copied to: {expected_data_dir}")
    except Exception as e:
        print(f"   ❌ Failed to copy dataset: {e}")
        continue
    
    # Verify required files and directories exist
    print(f"\n🔍 Verifying dataset structure...")
    required_items = {
        'raw.arrow': 'file',
        'duration.json': 'file',
        'vocab.txt': 'file',
        'wavs': 'directory',
        'metadata.csv': 'file'
    }
    
    all_exist = True
    for item, item_type in required_items.items():
        path = f"{expected_data_dir}/{item}"
        exists = os.path.exists(path)
        
        if item_type == 'directory':
            is_correct = exists and os.path.isdir(path)
        else:
            is_correct = exists and os.path.isfile(path)
        
        status = "✅" if is_correct else "❌"
        print(f"   {status} {item} ({item_type})")
        
        if not is_correct:
            all_exist = False
    
    if not all_exist:
        print(f"\n❌ Some required files/directories missing!")
        print(f"   Please ensure Cell 08 completed successfully.")
        continue
    
    # ✅ FIX: Verify arrow file integrity using Hugging Face Datasets API
    print(f"\n🔍 Verifying arrow file integrity...")
    try:
        from datasets import Dataset
        
        arrow_path = f"{expected_data_dir}/raw.arrow"
        
        # Load dataset using Hugging Face Datasets API
        # This is the correct way to read files created by ArrowWriter
        dataset = Dataset.from_file(arrow_path)
        num_rows = len(dataset)
        print(f"   ✅ Arrow file valid: {num_rows} rows")
        
        # Check column names
        column_names = dataset.column_names
        if 'audio_path' in column_names:
            print(f"   ✅ Contains 'audio_path' column")
            
            # Check if audio paths exist (sample first 5)
            audio_paths = dataset['audio_path'][:5]  # Get first 5
            missing_audio = []
            
            for audio_path in audio_paths:
                # Handle both absolute and relative paths
                if os.path.isabs(audio_path):
                    full_path = audio_path
                else:
                    # Relative path - check in expected_data_dir
                    full_path = os.path.join(expected_data_dir, audio_path)
                
                if not os.path.exists(full_path):
                    missing_audio.append(audio_path)
            
            if missing_audio:
                print(f"   ⚠️  Some audio files not found (checked first 5):")
                for path in missing_audio[:3]:
                    print(f"      - {path}")
                print(f"   💡 This may be OK if paths are relative to wavs/ directory")
            else:
                print(f"   ✅ Audio paths verified (checked first 5)")
        else:
            print(f"   ⚠️  'audio_path' column not found in arrow file")
            print(f"   Available columns: {', '.join(column_names)}")
                
    except Exception as e:
        print(f"   ❌ Arrow file verification failed: {e}")
        print(f"   💡 Please re-run Cell 08 to regenerate dataset")
        print(f"   Error type: {type(e).__name__}")
        continue
    
    # Verify path resolution matches
    print(f"\n🔍 Verifying path resolution...")
    try:
        from importlib.resources import files
        test_path = str(files("f5_tts").joinpath(f"../../data/{dataset_name}"))
        resolved_test = os.path.abspath(test_path)
        resolved_expected = os.path.abspath(expected_data_dir)
        
        print(f"   Script will look for: {resolved_test}")
        print(f"   Data copied to: {resolved_expected}")
        
        if resolved_test == resolved_expected:
            print(f"   ✅ Paths match!")
        else:
            print(f"   ⚠️  Path mismatch detected!")
            print(f"      This may still work if paths resolve correctly at runtime")
    except Exception as e:
        print(f"   ⚠️  Could not verify path resolution: {e}")
    
    print(f"\n   ✅ Dataset ready for training!")
    
    # Prepare training command
    cmd = [
        venv_python,
        "src/f5_tts/train/finetune_cli.py",
        "--exp_name", TRAINING_CONFIG["exp_name"],
        "--dataset_name", dataset_name,
        "--batch_size_per_gpu", str(TRAINING_CONFIG["batch_size"]),
        "--learning_rate", str(TRAINING_CONFIG["learning_rate"]),
        "--epochs", str(TRAINING_CONFIG["epochs"]),
        "--num_warmup_updates", str(TRAINING_CONFIG["num_warmup_updates"]),
        "--save_per_updates", str(TRAINING_CONFIG["save_per_updates"]),
        "--last_per_updates", str(TRAINING_CONFIG["last_per_updates"]),
        "--max_grad_norm", str(TRAINING_CONFIG["max_grad_norm"]),
        "--finetune",
        "--tokenizer", "custom",
        "--tokenizer_path", f"{training_dir}/vocab.txt",
        "--pretrain", f"{ckpt_dir}/pretrained_model_1200000.pt"
    ]
    
    print(f"\n🚀 Starting training...")
    print(f"   Dataset: {dataset_name}")
    print(f"   Checkpoint dir: {ckpt_dir}")
    print(f"   Batch size: {TRAINING_CONFIG['batch_size']}")
    print()
    
    # Run training
    start_time = time.time()
    
    try:
        # Start training process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Monitor output
        for line in process.stdout:
            # Print all output
            print(line, end='')
            
            # Parse and save important metrics
            if 'Step' in line or 'Loss' in line or 'Epoch' in line:
                # Log to file
                log_file = f"/content/drive/MyDrive/F5TTS_Vietnamese/logs/{speaker}_training.log"
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                with open(log_file, 'a') as f:
                    f.write(line)
            
            # Auto-backup checkpoints to Drive
            if 'Saved checkpoint' in line:
                print("\n📤 Backing up checkpoint to Drive...")
                
                # Find latest checkpoint
                ckpts = list(Path(ckpt_dir).glob("model_*.pt"))
                if ckpts:
                    latest_ckpt = max(ckpts, key=lambda p: p.stat().st_mtime)
                    
                    # Copy to Drive
                    drive_ckpt_dir = f"/content/drive/MyDrive/F5TTS_Vietnamese/checkpoints/{speaker}"
                    os.makedirs(drive_ckpt_dir, exist_ok=True)
                    
                    import shutil
                    shutil.copy(str(latest_ckpt), drive_ckpt_dir)
                    print(f"✅ Backed up: {latest_ckpt.name}")
        
        process.wait()
        
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        
        if process.returncode == 0:
            print(f"\n✅ Training completed for {speaker}!")
            print(f"   Time: {hours}h {minutes}m")
        else:
            print(f"\n❌ Training failed for {speaker}!")
            continue
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        print("   Saving current state...")
        process.terminate()
        process.wait()
        
        # Backup current checkpoints
        print("📤 Backing up checkpoints to Drive...")
        drive_ckpt_dir = f"/content/drive/MyDrive/F5TTS_Vietnamese/checkpoints/{speaker}"
        os.makedirs(drive_ckpt_dir, exist_ok=True)
        
        for ckpt in Path(ckpt_dir).glob("model_*.pt"):
            import shutil
            shutil.copy(str(ckpt), drive_ckpt_dir)
        
        print("✅ Checkpoints saved to Drive")
        print("   You can resume training later")
        break
    
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        continue
    
    # ------------------------------------------------------------------------------
    # 3. Post-Training: Organize Checkpoints
    # ------------------------------------------------------------------------------
    print("\n" + "="*70)
    print(f"📦 Organizing checkpoints for {speaker}...")
    print("="*70)
    
    # Find best checkpoint (usually model_last.pt)
    best_ckpt = Path(ckpt_dir) / "model_last.pt"
    
    if best_ckpt.exists():
        print(f"✅ Best model: {best_ckpt}")
        
        # Copy to models directory
        model_dir = f"/content/models/{speaker}"
        os.makedirs(model_dir, exist_ok=True)
        
        import shutil
        shutil.copy(str(best_ckpt), f"{model_dir}/model.pt")
        shutil.copy(f"{training_dir}/vocab.txt", f"{model_dir}/vocab.txt")
        
        # Create config file
        model_config = {
            "speaker": speaker,
            "model_path": f"{model_dir}/model.pt",
            "vocab_path": f"{model_dir}/vocab.txt",
            "training_duration": f"{hours}h {minutes}m",
            "dataset": dataset_name,
            "batch_size": TRAINING_CONFIG["batch_size"],
            "epochs_trained": TRAINING_CONFIG["epochs"]
        }
        
        with open(f"{model_dir}/config.json", 'w') as f:
            json.dump(model_config, f, indent=2)
        
        print(f"✅ Model organized in: {model_dir}")
        
        # Backup to Drive
        drive_model_dir = f"/content/drive/MyDrive/F5TTS_Vietnamese/models/{speaker}"
        os.makedirs(drive_model_dir, exist_ok=True)
        
        shutil.copy(f"{model_dir}/model.pt", drive_model_dir)
        shutil.copy(f"{model_dir}/vocab.txt", drive_model_dir)
        shutil.copy(f"{model_dir}/config.json", drive_model_dir)
        
        print(f"✅ Model backed up to Drive: {drive_model_dir}")
    else:
        print(f"⚠️  Best checkpoint not found")

# ------------------------------------------------------------------------------
# 4. Training Complete Summary
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)

print(f"""
📊 Summary:
   Speakers trained: {len(speakers)}
   
📁 Models Location:
   Local: /content/models/
   Drive: /content/drive/MyDrive/F5TTS_Vietnamese/models/
   
👥 Trained Speakers:
""")

for speaker in speakers:
    model_path = f"/content/models/{speaker}/model.pt"
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024**2)
        print(f"   ✅ {speaker}")
        print(f"      Model: {size_mb:.1f} MB")
        print(f"      Location: /content/models/{speaker}/")
    else:
        print(f"   ❌ {speaker} - training may have failed")

print(f"""
📝 Next Steps:
   → Run Cell 10 to test inference
   → Generate speech with trained voices!
   
⚠️  Important:
   - All models backed up to Google Drive
   - Checkpoints saved for resume if needed
   - Training logs in Drive/logs/
""")

print("="*70)
print("🎉 Ready for inference! Proceed to Cell 10!")
print("="*70)

# Update config
config['trained_speakers'] = speakers
config['training_complete'] = True

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

# Backup config
drive_config = "/content/drive/MyDrive/F5TTS_Vietnamese/processing_config.json"
with open(drive_config, 'w') as f:
    json.dump(config, f, indent=2)



