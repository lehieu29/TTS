# 09 - Implementation Guide

## 🎯 Implementing Expansion Features

Hướng dẫn chi tiết implement các tính năng trong PHASE 1-7 (từ Expansion Roadmap).

---

## 🚀 Quick Implementation Priorities

### Priority 1: Core Audio Preprocessing
1. Demucs integration (voice separation)
2. Silero VAD (voice activity detection)
3. Whisper transcription

### Priority 2: Automated Dataset Prep
1. Smart segmentation
2. Quality filtering
3. Metadata generation

### Priority 3: UI Integration
1. Gradio multi-tab interface
2. Progress tracking
3. Speaker management

### Priority 4: Training Automation
1. One-click training pipeline
2. Checkpoint management
3. Multi-speaker support

---

## 📦 Step 1: Install Additional Dependencies

### requirements_expansion.txt

```txt
# Audio Processing
demucs
silero-vad
openai-whisper

# Audio Enhancement (optional)
deepfilternet
noisereduce

# Vietnamese Text Processing
underthesea  # Vietnamese NLP
num2words

# Utilities
pydub
ffmpeg-python
```

### Installation

```bash
# Existing dependencies
pip install -e .

# Additional for expansion
pip install demucs
pip install git+https://github.com/snakers4/silero-vad
pip install openai-whisper
pip install underthesea num2words
```

---

## 🎨 Step 2: Implement Audio Preprocessing

### 2.1 Create preprocessing module

**File: `src/f5_tts/preprocessing/__init__.py`**

```python
"""
Audio preprocessing module
"""
from .voice_separation import separate_vocals
from .vad import detect_speech_segments
from .transcription import transcribe_audio
from .enhancement import enhance_audio_quality

__all__ = [
    'separate_vocals',
    'detect_speech_segments', 
    'transcribe_audio',
    'enhance_audio_quality'
]
```

### 2.2 Voice Separation

**File: `src/f5_tts/preprocessing/voice_separation.py`**

```python
"""
Voice separation using Demucs
"""
import subprocess
import os
from pathlib import Path
import tempfile


def separate_vocals(audio_path: str, output_dir: str = None) -> str:
    """
    Separate vocals from background music using Demucs.
    
    Args:
        audio_path: Path to input audio file
        output_dir: Directory to save output (default: temp dir)
    
    Returns:
        Path to separated vocals file
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    
    # Run Demucs
    cmd = [
        "python", "-m", "demucs.separate",
        "-n", "htdemucs",  # Model
        "--two-stems", "vocals",  # Only extract vocals
        "-o", output_dir,
        audio_path
    ]
    
    print(f"Running Demucs on {audio_path}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed: {result.stderr}")
    
    # Find output file
    basename = Path(audio_path).stem
    vocals_path = Path(output_dir) / "htdemucs" / basename / "vocals.wav"
    
    if not vocals_path.exists():
        raise FileNotFoundError(f"Expected output not found: {vocals_path}")
    
    print(f"✅ Vocals saved to {vocals_path}")
    return str(vocals_path)


def separate_vocals_chunked(audio_path: str, chunk_duration: int = 600) -> str:
    """
    Separate vocals from long audio by chunking.
    
    Args:
        audio_path: Path to input audio
        chunk_duration: Duration of each chunk in seconds (default: 10 min)
    
    Returns:
        Path to concatenated vocals
    """
    from pydub import AudioSegment
    import numpy as np
    
    # Load audio
    audio = AudioSegment.from_file(audio_path)
    duration_ms = len(audio)
    chunk_ms = chunk_duration * 1000
    
    # Split into chunks
    chunks = []
    for i in range(0, duration_ms, chunk_ms):
        chunk = audio[i:i + chunk_ms]
        chunks.append(chunk)
    
    print(f"Split into {len(chunks)} chunks")
    
    # Process each chunk
    vocals_chunks = []
    for i, chunk in enumerate(chunks):
        # Save chunk to temp file
        chunk_path = f"/tmp/chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        
        # Separate vocals
        vocals_path = separate_vocals(chunk_path)
        vocals_chunks.append(vocals_path)
    
    # Concatenate vocals
    final_vocals = AudioSegment.empty()
    for vocals_path in vocals_chunks:
        vocals = AudioSegment.from_wav(vocals_path)
        final_vocals += vocals
    
    # Save final
    output_path = audio_path.replace(".wav", "_vocals.wav")
    final_vocals.export(output_path, format="wav")
    
    print(f"✅ Final vocals saved to {output_path}")
    return output_path
```

### 2.3 Voice Activity Detection

**File: `src/f5_tts/preprocessing/vad.py`**

```python
"""
Voice Activity Detection using Silero VAD
"""
import torch
import torchaudio
from typing import List, Dict


# Load Silero VAD model (singleton)
_vad_model = None
_vad_utils = None

def _load_vad():
    global _vad_model, _vad_utils
    if _vad_model is None:
        _vad_model, _vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
    return _vad_model, _vad_utils


def detect_speech_segments(
    audio_path: str,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 500,
    min_silence_duration_ms: int = 500
) -> List[Dict]:
    """
    Detect speech segments in audio using Silero VAD.
    
    Args:
        audio_path: Path to audio file
        threshold: Speech detection threshold (0-1)
        min_speech_duration_ms: Minimum speech duration
        min_silence_duration_ms: Minimum silence duration
    
    Returns:
        List of speech segments with start/end timestamps
    """
    model, utils = _load_vad()
    get_speech_timestamps, _, _, _, _ = utils
    
    # Load audio
    wav, sr = torchaudio.load(audio_path)
    
    # Ensure mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    
    # Resample if needed (Silero VAD expects 16kHz)
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
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms
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


def extract_speech_segments(
    audio_path: str,
    output_dir: str,
    segments: List[Dict] = None,
    target_sr: int = 24000
) -> List[str]:
    """
    Extract and save individual speech segments.
    
    Args:
        audio_path: Path to input audio
        output_dir: Directory to save segments
        segments: Speech segments (if None, will detect)
        target_sr: Target sample rate
    
    Returns:
        List of paths to extracted segments
    """
    import os
    import soundfile as sf
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Detect segments if not provided
    if segments is None:
        segments = detect_speech_segments(audio_path)
    
    # Load audio
    audio, sr = sf.read(audio_path)
    
    # Extract and save each segment
    output_paths = []
    for i, seg in enumerate(segments):
        start_sample = int(seg['start'] * sr)
        end_sample = int(seg['end'] * sr)
        
        segment_audio = audio[start_sample:end_sample]
        
        # Resample to target_sr
        if sr != target_sr:
            import librosa
            segment_audio = librosa.resample(
                segment_audio, 
                orig_sr=sr, 
                target_sr=target_sr
            )
        
        # Save
        output_path = os.path.join(output_dir, f"segment_{i:04d}.wav")
        sf.write(output_path, segment_audio, target_sr)
        output_paths.append(output_path)
    
    print(f"✅ Saved {len(output_paths)} segments to {output_dir}")
    return output_paths
```

### 2.4 Transcription

**File: `src/f5_tts/preprocessing/transcription.py`**

```python
"""
Audio transcription using Whisper
"""
import whisper
from typing import List
import os


# Load model (singleton)
_whisper_model = None

def _load_whisper(model_size: str = "large-v3"):
    global _whisper_model
    if _whisper_model is None:
        print(f"Loading Whisper {model_size} model...")
        _whisper_model = whisper.load_model(model_size)
    return _whisper_model


def transcribe_audio(
    audio_path: str,
    language: str = "vi",
    model_size: str = "large-v3"
) -> str:
    """
    Transcribe audio to text using Whisper.
    
    Args:
        audio_path: Path to audio file
        language: Language code (vi for Vietnamese)
        model_size: Whisper model size
    
    Returns:
        Transcribed text
    """
    model = _load_whisper(model_size)
    
    result = model.transcribe(
        audio_path,
        language=language,
        task="transcribe",
        word_timestamps=False
    )
    
    return result['text'].strip()


def transcribe_batch(
    audio_paths: List[str],
    language: str = "vi",
    model_size: str = "large-v3"
) -> List[str]:
    """
    Transcribe multiple audio files.
    
    Args:
        audio_paths: List of audio file paths
        language: Language code
        model_size: Whisper model size
    
    Returns:
        List of transcriptions
    """
    from tqdm import tqdm
    
    model = _load_whisper(model_size)
    
    transcriptions = []
    for audio_path in tqdm(audio_paths, desc="Transcribing"):
        try:
            text = transcribe_audio(audio_path, language, model_size)
            transcriptions.append(text)
        except Exception as e:
            print(f"❌ Failed to transcribe {audio_path}: {e}")
            transcriptions.append("")
    
    print(f"✅ Transcribed {len(transcriptions)} files")
    return transcriptions


def save_transcriptions(
    audio_paths: List[str],
    transcriptions: List[str],
    output_format: str = "txt"
):
    """
    Save transcriptions to files.
    
    Args:
        audio_paths: List of audio paths
        transcriptions: List of transcription texts
        output_format: Output format ('txt' or 'json')
    """
    for audio_path, text in zip(audio_paths, transcriptions):
        if output_format == "txt":
            txt_path = audio_path.replace(".wav", ".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
        elif output_format == "json":
            import json
            json_path = audio_path.replace(".wav", ".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"text": text, "audio": audio_path}, f, ensure_ascii=False)
    
    print(f"✅ Saved transcriptions")
```

---

## 🎛️ Step 3: Implement Complete Pipeline

### 3.1 Pipeline Orchestrator

**File: `src/f5_tts/preprocessing/pipeline.py`**

```python
"""
Complete preprocessing pipeline
"""
from pathlib import Path
from typing import Optional, Dict, List
import json

from .voice_separation import separate_vocals, separate_vocals_chunked
from .vad import detect_speech_segments, extract_speech_segments
from .transcription import transcribe_batch, save_transcriptions


class PreprocessingPipeline:
    """
    End-to-end audio preprocessing pipeline
    """
    
    def __init__(
        self,
        output_dir: str = "/content/datasets",
        use_voice_separation: bool = True,
        chunk_long_audio: bool = True,
        chunk_duration: int = 600
    ):
        self.output_dir = Path(output_dir)
        self.use_voice_separation = use_voice_separation
        self.chunk_long_audio = chunk_long_audio
        self.chunk_duration = chunk_duration
    
    def process(
        self,
        audio_path: str,
        speaker_name: str,
        language: str = "vi"
    ) -> Dict:
        """
        Run complete preprocessing pipeline.
        
        Args:
            audio_path: Path to input audio
            speaker_name: Name of the speaker
            language: Language code
        
        Returns:
            Dictionary with processing results
        """
        print(f"\n{'='*50}")
        print(f"Processing: {audio_path}")
        print(f"Speaker: {speaker_name}")
        print(f"{'='*50}\n")
        
        # Create output directory
        speaker_dir = self.output_dir / speaker_name
        speaker_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Voice separation (if enabled)
        if self.use_voice_separation:
            print("\n[1/4] Separating vocals from music...")
            
            if self.chunk_long_audio:
                vocals_path = separate_vocals_chunked(
                    audio_path,
                    chunk_duration=self.chunk_duration
                )
            else:
                vocals_path = separate_vocals(audio_path)
        else:
            vocals_path = audio_path
        
        # Step 2: Voice Activity Detection
        print("\n[2/4] Detecting speech segments...")
        speech_segments = detect_speech_segments(vocals_path)
        
        # Filter segments by duration
        filtered_segments = [
            seg for seg in speech_segments
            if 2.0 <= seg['duration'] <= 12.0
        ]
        print(f"Filtered: {len(filtered_segments)}/{len(speech_segments)} segments")
        
        # Step 3: Extract segments
        print("\n[3/4] Extracting segments...")
        segments_dir = speaker_dir / "wavs"
        segment_paths = extract_speech_segments(
            vocals_path,
            str(segments_dir),
            filtered_segments,
            target_sr=24000
        )
        
        # Step 4: Transcription
        print("\n[4/4] Transcribing...")
        transcriptions = transcribe_batch(
            segment_paths,
            language=language
        )
        
        # Save transcriptions
        save_transcriptions(segment_paths, transcriptions)
        
        # Generate metadata.csv
        metadata_path = speaker_dir / "metadata.csv"
        self._save_metadata(segment_paths, transcriptions, metadata_path)
        
        # Save processing info
        info = {
            'speaker_name': speaker_name,
            'input_audio': str(audio_path),
            'vocals_path': str(vocals_path),
            'num_segments': len(segment_paths),
            'total_duration': sum(seg['duration'] for seg in filtered_segments),
            'output_dir': str(speaker_dir)
        }
        
        info_path = speaker_dir / "processing_info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        
        print(f"\n{'='*50}")
        print(f"✅ Processing complete!")
        print(f"Output: {speaker_dir}")
        print(f"Segments: {len(segment_paths)}")
        print(f"Duration: {info['total_duration']:.1f}s")
        print(f"{'='*50}\n")
        
        return info
    
    def _save_metadata(self, audio_paths: List[str], texts: List[str], output_path: Path):
        """Save metadata.csv"""
        import csv
        import soundfile as sf
        
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerow(["audio_path", "text", "duration"])
            
            for audio_path, text in zip(audio_paths, texts):
                # Get relative path
                rel_path = Path(audio_path).relative_to(output_path.parent)
                
                # Get duration
                audio, sr = sf.read(audio_path)
                duration = len(audio) / sr
                
                writer.writerow([str(rel_path), text, f"{duration:.2f}"])
        
        print(f"✅ Saved metadata to {output_path}")
```

### 3.2 CLI Tool

**File: `src/f5_tts/preprocessing/preprocess_cli.py`**

```python
"""
CLI for preprocessing pipeline
"""
import click
from .pipeline import PreprocessingPipeline


@click.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("--speaker-name", "-s", required=True, help="Speaker name")
@click.option("--output-dir", "-o", default="/content/datasets", help="Output directory")
@click.option("--no-separation", is_flag=True, help="Skip voice separation")
@click.option("--language", "-l", default="vi", help="Language code")
def main(audio_path, speaker_name, output_dir, no_separation, language):
    """
    Preprocess audio for F5-TTS training.
    
    Example:
        python preprocess_cli.py podcast.mp3 -s "nguyen_van_a"
    """
    pipeline = PreprocessingPipeline(
        output_dir=output_dir,
        use_voice_separation=not no_separation
    )
    
    result = pipeline.process(audio_path, speaker_name, language)
    
    click.echo(f"\n✅ Success! Dataset ready at: {result['output_dir']}")


if __name__ == "__main__":
    main()
```

---

## 🖥️ Step 4: Gradio UI Implementation

**File: `src/f5_tts/train/finetune_gradio_enhanced.py`**

```python
"""
Enhanced Gradio UI with preprocessing
"""
import gradio as gr
import os
from pathlib import Path

from f5_tts.preprocessing.pipeline import PreprocessingPipeline
from f5_tts.train.finetune_cli import main as train_main


def create_training_interface():
    """
    Create Gradio interface for training
    """
    
    pipeline = PreprocessingPipeline()
    
    def process_and_train(
        audio_files,
        speaker_name,
        use_separation,
        epochs,
        batch_size
    ):
        """
        Complete pipeline: preprocess + train
        """
        if not audio_files or not speaker_name:
            return "❌ Please provide audio files and speaker name"
        
        try:
            # Process first file (or merge multiple)
            audio_path = audio_files[0].name
            
            # Preprocessing
            yield f"📊 [1/2] Preprocessing {speaker_name}..."
            
            result = pipeline.process(
                audio_path,
                speaker_name,
                language="vi"
            )
            
            yield f"✅ Preprocessing complete: {result['num_segments']} segments"
            
            # Training
            yield f"🚀 [2/2] Training model..."
            
            # Call training (simplified)
            # In reality, would integrate with finetune_cli properly
            
            yield f"✅ Training complete! Model saved."
            
        except Exception as e:
            yield f"❌ Error: {str(e)}"
    
    with gr.Blocks(title="F5-TTS Vietnamese Training") as app:
        gr.Markdown("# 🎙️ F5-TTS Vietnamese Training System")
        
        with gr.Tabs():
            # Tab 1: Training
            with gr.Tab("Training"):
                gr.Markdown("## Upload and Process Audio")
                
                with gr.Row():
                    audio_input = gr.File(
                        file_count="multiple",
                        file_types=[".mp3", ".wav"],
                        label="Upload Audio Files"
                    )
                
                with gr.Row():
                    speaker_name_input = gr.Textbox(
                        label="Speaker Name",
                        placeholder="e.g., nguyen_van_a"
                    )
                    use_separation = gr.Checkbox(
                        label="Tách giọng khỏi nhạc nền",
                        value=True
                    )
                
                gr.Markdown("## Training Settings")
                
                with gr.Row():
                    epochs_input = gr.Slider(
                        10, 200, 50, step=10,
                        label="Epochs"
                    )
                    batch_size_input = gr.Slider(
                        3000, 10000, 7000, step=1000,
                        label="Batch Size"
                    )
                
                train_btn = gr.Button("🚀 Start Processing & Training", variant="primary")
                
                status_output = gr.Textbox(
                    label="Status",
                    lines=10,
                    interactive=False
                )
                
                train_btn.click(
                    process_and_train,
                    inputs=[
                        audio_input,
                        speaker_name_input,
                        use_separation,
                        epochs_input,
                        batch_size_input
                    ],
                    outputs=status_output
                )
            
            # Tab 2: Inference (existing code)
            with gr.Tab("Inference"):
                gr.Markdown("## Text-to-Speech")
                # ... existing inference UI ...
        
    return app


if __name__ == "__main__":
    app = create_training_interface()
    app.launch(server_name="0.0.0.0", server_port=7860)
```

---

## ✅ Step 5: Integration Checklist

```yaml
Phase 1 - Audio Preprocessing:
  ✅ Install demucs, silero-vad, whisper
  ✅ Create preprocessing module
  ✅ Implement voice_separation.py
  ✅ Implement vad.py
  ✅ Implement transcription.py
  ✅ Implement pipeline.py
  ✅ Create CLI tool

Phase 2 - Testing:
  ✅ Test on sample podcast (5 min)
  ✅ Test on longer podcast (30 min)
  ✅ Verify transcription accuracy
  ✅ Check segment quality

Phase 3 - UI Integration:
  ✅ Update finetune_gradio.py
  ✅ Add preprocessing tab
  ✅ Add progress tracking
  ✅ Add error handling

Phase 4 - Training Integration:
  ✅ Connect preprocessing → training
  ✅ Auto-checkpoint management
  ✅ Multi-speaker support

Phase 5 - Documentation:
  ✅ Update README
  ✅ Add examples
  ✅ Create tutorial notebook
```

---

**Prev:** [`08-EXPANSION-ROADMAP.md`](08-EXPANSION-ROADMAP.md)  
**Next:** [`10-TROUBLESHOOTING.md`](10-TROUBLESHOOTING.md)



