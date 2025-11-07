"""
Cell 03: Install Preprocessing Tools
Mục đích:
  - Install Demucs (voice separation)
  - Install Whisper (transcription)
  - Install Silero VAD (voice activity detection)
  - Install enhancement tools
"""

# ============================================================================
# CELL 03: INSTALL PREPROCESSING TOOLS
# ============================================================================

print("🔧 Installing Preprocessing Tools...")

import subprocess
import sys
import os

venv_python = "/content/venv/bin/python"
venv_pip = "/content/venv/bin/pip"

# ------------------------------------------------------------------------------
# 1. Install Demucs (Voice Separation)
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🎵 Installing Demucs (Voice Separation)...")
print("="*70)

print("⏳ This may take 5-10 minutes...")

subprocess.run([
    venv_pip, "install",
    "demucs"
], check=True)

print("✅ Demucs installed!")

# Verify
demucs_check = subprocess.run([
    venv_python, "-c",
    "import demucs; print(f'Demucs version: {demucs.__version__}')"
], capture_output=True, text=True)
print(demucs_check.stdout)

# ------------------------------------------------------------------------------
# 2. Install Whisper (Transcription)
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🎤 Installing Whisper (Transcription)...")
print("="*70)

print("⏳ This may take 5 minutes...")

subprocess.run([
    venv_pip, "install",
    "openai-whisper"
], check=True)

print("✅ Whisper installed!")

# Verify
whisper_check = subprocess.run([
    venv_python, "-c",
    "import whisper; print(f'Whisper installed successfully')"
], capture_output=True, text=True)
print(whisper_check.stdout)

# Download Whisper model (large-v3 for Vietnamese)
print("\n📥 Downloading Whisper large-v3 model...")
print("⏳ This may take 5-10 minutes (2.9 GB)...")

download_script = """
import whisper
print("Downloading Whisper large-v3...")
model = whisper.load_model("large-v3")
print("✅ Model downloaded and cached!")
"""

subprocess.run([
    venv_python, "-c", download_script
], check=True)

print("✅ Whisper model ready!")

# ------------------------------------------------------------------------------
# 3. Install Silero VAD (Voice Activity Detection)
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🔊 Installing Silero VAD...")
print("="*70)

# Silero VAD uses torch hub, just need to download
print("📥 Downloading Silero VAD model...")

vad_script = """
import torch
print("Loading Silero VAD model...")
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)
print("✅ Silero VAD model downloaded and cached!")
"""

subprocess.run([
    venv_python, "-c", vad_script
], check=True)

print("✅ Silero VAD ready!")

# ------------------------------------------------------------------------------
# 4. Install Audio Enhancement Tools (Optional)
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("✨ Installing Audio Enhancement Tools...")
print("="*70)

# Install audio processing libraries
enhancement_packages = [
    "librosa",
    "soundfile", 
    "pydub",
    "noisereduce",
    "scipy"
]

print(f"Installing: {', '.join(enhancement_packages)}")
subprocess.run([
    venv_pip, "install"
] + enhancement_packages, check=True)

print("✅ Enhancement tools installed!")

# ------------------------------------------------------------------------------
# 5. Install Vietnamese Text Processing
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🇻🇳 Installing Vietnamese Text Processing...")
print("="*70)

subprocess.run([
    venv_pip, "install",
    "underthesea",
    "num2words"
], check=True)

print("✅ Vietnamese NLP tools installed!")

# ------------------------------------------------------------------------------
# 6. Install Additional Utilities
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🛠️  Installing Additional Utilities...")
print("="*70)

utilities = [
    "tqdm",          # Progress bars
    "matplotlib",    # Plotting
    "ipywidgets",    # Colab widgets
    "pandas",        # Data handling
]

subprocess.run([
    venv_pip, "install"
] + utilities, check=True)

print("✅ Utilities installed!")

# ------------------------------------------------------------------------------
# 7. Create Preprocessing Module
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("📝 Creating Preprocessing Helper Module...")
print("="*70)

preprocessing_code = '''
"""
Preprocessing helpers for Google Colab
"""
import os
import subprocess
import torch
import torchaudio
import whisper
from pathlib import Path

# Singleton instances
_whisper_model = None
_vad_model = None
_vad_utils = None

def load_whisper(model_size="large-v3"):
    """Load Whisper model (singleton)"""
    global _whisper_model
    if _whisper_model is None:
        print(f"Loading Whisper {model_size}...")
        _whisper_model = whisper.load_model(model_size)
    return _whisper_model

def load_vad():
    """Load Silero VAD model (singleton)"""
    global _vad_model, _vad_utils
    if _vad_model is None:
        print("Loading Silero VAD...")
        _vad_model, _vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
    return _vad_model, _vad_utils

def separate_vocals(audio_path, output_dir="/content/temp/separated"):
    """
    Separate vocals using Demucs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Separating vocals from: {audio_path}")
    
    cmd = [
        "python", "-m", "demucs.separate",
        "-n", "htdemucs",
        "--two-stems", "vocals",
        "-o", output_dir,
        audio_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed: {result.stderr}")
    
    # Find output
    basename = Path(audio_path).stem
    vocals_path = Path(output_dir) / "htdemucs" / basename / "vocals.wav"
    
    if not vocals_path.exists():
        raise FileNotFoundError(f"Expected output not found: {vocals_path}")
    
    print(f"✅ Vocals saved to: {vocals_path}")
    return str(vocals_path)

def detect_speech_segments(audio_path, threshold=0.5):
    """
    Detect speech segments using Silero VAD
    """
    model, utils = load_vad()
    get_speech_timestamps, _, _, _, _ = utils
    
    # Load audio
    wav, sr = torchaudio.load(audio_path)
    
    # Ensure mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    
    # Resample to 16kHz for VAD
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        wav = resampler(wav)
        sr = 16000
    
    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(
        wav.squeeze(),
        model,
        sampling_rate=sr,
        threshold=threshold,
        min_speech_duration_ms=500,
        min_silence_duration_ms=500
    )
    
    # Convert to seconds
    segments = []
    for ts in speech_timestamps:
        segments.append({
            'start': ts['start'] / sr,
            'end': ts['end'] / sr,
            'duration': (ts['end'] - ts['start']) / sr
        })
    
    print(f"✅ Detected {len(segments)} speech segments")
    return segments

def transcribe_audio(audio_path, language="vi"):
    """
    Transcribe audio using Whisper
    """
    model = load_whisper()
    
    result = model.transcribe(
        audio_path,
        language=language,
        task="transcribe"
    )
    
    return result['text'].strip()

def normalize_vietnamese_text(text):
    """
    Normalize Vietnamese text
    """
    import re
    import unicodedata
    
    # Lowercase
    text = text.lower()
    
    # Remove special characters (keep Vietnamese chars)
    text = re.sub(r'[^a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ\\s,.!?]', '', text)
    
    # Unicode normalization
    text = unicodedata.normalize('NFD', text)
    
    # Clean whitespace
    text = ' '.join(text.split())
    
    return text

print("✅ Preprocessing helpers loaded!")
'''

helper_path = "/content/preprocessing_helpers.py"
with open(helper_path, "w") as f:
    f.write(preprocessing_code)

print(f"✅ Helper module created at: {helper_path}")

# ------------------------------------------------------------------------------
# 8. Verify All Installations
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("✅ Verifying All Installations...")
print("="*70)

verification_script = """
print("Verifying preprocessing tools...\\n")

# Demucs
import demucs
print(f"✅ Demucs: {demucs.__version__}")

# Whisper
import whisper
print(f"✅ Whisper: installed")

# Silero VAD
import torch
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)
print(f"✅ Silero VAD: loaded")

# Audio processing
import librosa
print(f"✅ Librosa: {librosa.__version__}")

import soundfile
print(f"✅ SoundFile: {soundfile.__version__}")

import pydub
print(f"✅ Pydub: installed")

# Vietnamese NLP
import underthesea
print(f"✅ Underthesea: {underthesea.__version__}")

# Utilities
import tqdm
print(f"✅ tqdm: {tqdm.__version__}")

print("\\n✅ All preprocessing tools verified!")
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
print("✅ PREPROCESSING TOOLS INSTALLED!")
print("="*70)

print("""
🔧 Installed Tools:
   ✅ Demucs (voice separation)
   ✅ Whisper large-v3 (transcription)
   ✅ Silero VAD (voice activity detection)
   ✅ Librosa, SoundFile, Pydub (audio processing)
   ✅ Underthesea (Vietnamese NLP)
   ✅ Enhancement tools

📦 Downloaded Models:
   ✅ Whisper large-v3 (~2.9 GB)
   ✅ Silero VAD
   ✅ Demucs htdemucs (will download on first use)

📝 Helper Module:
   File: /content/preprocessing_helpers.py
   Import: from preprocessing_helpers import *

📁 Ready For:
   → Voice separation (Demucs)
   → Speech detection (Silero VAD)
   → Auto transcription (Whisper)
   → Text normalization

📝 Next Steps:
   → Run Cell 04 to upload audio files
   → Then run preprocessing pipeline

⚠️  Notes:
   - Demucs downloads ~2GB on first use
   - Whisper large-v3 for best Vietnamese accuracy
   - All tools ready in virtual environment
""")

print("="*70)
print("🎉 Ready to proceed to Cell 04!")
print("="*70)



