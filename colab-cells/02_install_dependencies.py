"""
Cell 02: Install Core Dependencies in Virtual Environment
Mục đích:
  - Clone F5-TTS-Vietnamese repository
  - Install PyTorch with CUDA support
  - Install F5-TTS and core dependencies
  - Verify installation
"""

# ============================================================================
# CELL 02: INSTALL CORE DEPENDENCIES
# ============================================================================

print("📦 Installing Core Dependencies...")

import subprocess
import sys
import os

# ------------------------------------------------------------------------------
# 1. Activate Virtual Environment
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🐍 Activating Virtual Environment...")
print("="*70)

venv_python = "/content/venv/bin/python"
venv_pip = "/content/venv/bin/pip"

# Verify venv exists
if not os.path.exists(venv_python):
    print("❌ Virtual environment not found!")
    print("Please run Cell 01 first!")
    sys.exit(1)

print(f"✅ Using Python: {venv_python}")
print(f"✅ Using Pip: {venv_pip}")

# Check Python version
result = subprocess.run([venv_python, "--version"], capture_output=True, text=True)
print(f"Python version: {result.stdout.strip()}")

# ------------------------------------------------------------------------------
# 2. Upgrade pip, setuptools, wheel
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("⬆️  Upgrading pip, setuptools, wheel...")
print("="*70)

subprocess.run([
    venv_pip, "install", "--upgrade",
    "pip", "setuptools", "wheel"
], check=True)

print("✅ Upgraded successfully!")

# ------------------------------------------------------------------------------
# 3. Clone F5-TTS-Vietnamese Repository
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("📥 Cloning F5-TTS-Vietnamese Repository...")
print("="*70)

repo_path = "/content/F5-TTS-Vietnamese"

if os.path.exists(repo_path):
    print(f"⚠️  Repository already exists at {repo_path}")
    print("Pulling latest changes...")
    subprocess.run(["git", "-C", repo_path, "pull"], check=True)
else:
    print("Cloning repository...")
    subprocess.run([
        "git", "clone",
        "https://github.com/nguyenthienhy/F5-TTS-Vietnamese.git",
        repo_path
    ], check=True)

print(f"✅ Repository ready at: {repo_path}")

# Change to repo directory
os.chdir(repo_path)
print(f"📁 Changed directory to: {os.getcwd()}")

# ------------------------------------------------------------------------------
# 4. Install PyTorch with CUDA Support
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🔥 Installing PyTorch with CUDA 12.1...")
print("="*70)

print("⏳ This may take 5-10 minutes...")

subprocess.run([
    venv_pip, "install",
    "torch==2.4.0",
    "torchaudio==2.4.0",
    "--index-url", "https://download.pytorch.org/whl/cu121"
], check=True)

print("✅ PyTorch installed!")

# Verify CUDA
print("\n🔍 Verifying CUDA...")
cuda_check = subprocess.run([
    venv_python, "-c",
    "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
], capture_output=True, text=True)

print(cuda_check.stdout)

if "CUDA available: True" not in cuda_check.stdout:
    print("⚠️  CUDA not available! Training will be slow.")
    print("💡 Check: Runtime → Change runtime type → GPU")

# ------------------------------------------------------------------------------
# 5. Install numpy < 2.0 (Critical for F5-TTS)
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🔢 Installing numpy < 2.0...")
print("="*70)

subprocess.run([
    venv_pip, "install",
    "numpy<2.0,>=1.24.0"
], check=True)

# Verify numpy version
numpy_check = subprocess.run([
    venv_python, "-c",
    "import numpy; print(f'numpy version: {numpy.__version__}')"
], capture_output=True, text=True)

print(numpy_check.stdout)

if "numpy version: 1." not in numpy_check.stdout:
    print("❌ numpy version incorrect!")
    sys.exit(1)

print("✅ numpy < 2.0 installed correctly!")

# ------------------------------------------------------------------------------
# 6. Install F5-TTS and Dependencies
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🎙️  Installing F5-TTS...")
print("="*70)

print("⏳ This may take 10-15 minutes...")

# Install in editable mode
subprocess.run([
    venv_pip, "install", "-e", "."
], cwd=repo_path, check=True)

print("✅ F5-TTS installed!")

# ------------------------------------------------------------------------------
# 7. Install Audio Processing Tools
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🎵 Installing Audio Processing Tools...")
print("="*70)

# Install sox and ffmpeg (system packages)
print("Installing sox and ffmpeg...")
subprocess.run(["apt-get", "update", "-qq"], check=True)
subprocess.run([
    "apt-get", "install", "-y", "-qq",
    "sox", "libsox-fmt-all", "ffmpeg"
], check=True)

print("✅ sox and ffmpeg installed!")

# Verify installations
sox_check = subprocess.run(["sox", "--version"], capture_output=True, text=True)
print(f"sox: {sox_check.stdout.split()[1] if sox_check.returncode == 0 else 'Not found'}")

ffmpeg_check = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
print(f"ffmpeg: {ffmpeg_check.stdout.split()[2] if ffmpeg_check.returncode == 0 else 'Not found'}")

# ------------------------------------------------------------------------------
# 8. Verify Installation
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("✅ Verifying Installation...")
print("="*70)

# Test import
verification_script = """
import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

import torchaudio
print(f"TorchAudio: {torchaudio.__version__}")

import numpy
print(f"NumPy: {numpy.__version__}")

import f5_tts
print(f"F5-TTS: Installed ✓")

from f5_tts.model import DiT, UNetT
print(f"F5-TTS Models: Imported ✓")

print("\\n✅ All core dependencies verified!")
"""

result = subprocess.run(
    [venv_python, "-c", verification_script],
    capture_output=True,
    text=True
)

print(result.stdout)

if result.returncode != 0:
    print("❌ Verification failed!")
    print(result.stderr)
    sys.exit(1)

# ------------------------------------------------------------------------------
# 9. Display Summary
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("✅ INSTALLATION COMPLETE!")
print("="*70)

print("""
📦 Installed Packages:
   ✅ PyTorch 2.4.0 + CUDA 12.1
   ✅ TorchAudio 2.4.0
   ✅ NumPy < 2.0
   ✅ F5-TTS (editable mode)
   ✅ sox, ffmpeg

📁 Repository:
   Path: /content/F5-TTS-Vietnamese/
   Status: Ready

🐍 Virtual Environment:
   Python: /content/venv/bin/python
   Pip: /content/venv/bin/pip

📝 Next Steps:
   → Run Cell 03 to install preprocessing tools
   → (Demucs, Whisper, Silero VAD)

⚠️  Remember:
   - Always use venv_python and venv_pip
   - Keep this cell's output for reference
""")

# Save environment info
env_info = f"""
# Environment Information
Timestamp: {subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}
Python: {result.stdout}
Repository: {repo_path}
Virtual Environment: /content/venv/
"""

with open("/content/drive/MyDrive/F5TTS_Vietnamese/environment_info.txt", "w") as f:
    f.write(env_info)

print("="*70)
print("🎉 Ready to proceed to Cell 03!")
print("="*70)



