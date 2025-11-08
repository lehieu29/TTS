"""
Cell 11 Simple Test - Minimal Gradio Test
==========================================
Test Gradio với 1 speaker, 1 text đơn giản
"""

import gradio as gr
import os
import sys
import json
import subprocess
import time
import shutil
from pathlib import Path

print("🧪 Simple Gradio Test")
print("="*70)

# ------------------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------------------
venv_python = "/content/venv/bin/python"
local_models_dir = "/content/models"

# Load config
config_path = "/content/processing_config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

trained_speakers = config.get('trained_speakers', [])
speaker = trained_speakers[0]

print(f"Testing with speaker: {speaker}")

# Ensure model exists
model_dir = f"{local_models_dir}/{speaker}"
model_path = f"{model_dir}/model.pt"
vocab_path = f"{model_dir}/vocab.txt"

if not os.path.exists(model_path):
    # Load from training checkpoints
    ckpt_path = f"/content/F5-TTS-Vietnamese/ckpts/{speaker}_training/model_last.pt"
    training_vocab = f"/content/data/{speaker}_training/vocab.txt"
    
    os.makedirs(model_dir, exist_ok=True)
    shutil.copy(ckpt_path, model_path)
    shutil.copy(training_vocab, vocab_path)
    print(f"✅ Loaded model from training checkpoints")

# Get reference audio
segments_dir = f"/content/data/{speaker}_training/wavs"
ref_audio = str(list(Path(segments_dir).glob("*.wav"))[0])

# Get reference text
metadata_path = f"/content/data/{speaker}_training/metadata.csv"
ref_text = "xin chào"

if os.path.exists(metadata_path):
    import csv
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            if len(row) >= 2:
                if Path(ref_audio).stem in row[0]:
                    ref_text = row[1]
                    break

print(f"✅ Reference audio: {Path(ref_audio).name}")
print(f"✅ Reference text: {ref_text[:50]}...")

# ------------------------------------------------------------------------------
# 2. Inference Function
# ------------------------------------------------------------------------------
def generate_speech(input_text):
    """Simple inference function for Gradio"""
    
    print(f"\n{'='*70}")
    print(f"🎙️  GRADIO INFERENCE REQUEST")
    print(f"{'='*70}")
    print(f"Input text: {input_text}")
    
    if not input_text or len(input_text.strip()) == 0:
        error_msg = "⚠️ Please enter some text"
        print(error_msg)
        return None, error_msg
    
    # Output setup
    output_dir = "/content/outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"gradio_test_{int(time.time())}.wav"
    
    # Build command
    cmd = [
        venv_python, "-m", "f5_tts.infer.infer_cli",
        "--model", "F5TTS_Base",
        "--ref_audio", ref_audio,
        "--ref_text", ref_text,
        "--gen_text", input_text,
        "--output_dir", output_dir,
        "--output_file", output_filename,
        "--vocab_file", vocab_path,
        "--ckpt_file", model_path,
        "--speed", "1.0",
        "--nfe_step", "32"
    ]
    
    print(f"\nCommand: {' '.join(cmd[:5])}...")
    print(f"Generating audio...")
    
    try:
        # Run inference
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd="/content/F5-TTS-Vietnamese"
        )
        
        print(f"\nReturn code: {result.returncode}")
        
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        
        output_file = f"{output_dir}/{output_filename}"
        
        if result.returncode == 0 and os.path.exists(output_file):
            size_mb = os.path.getsize(output_file) / (1024**2)
            success_msg = f"✅ Success! Generated {size_mb:.2f} MB audio"
            print(success_msg)
            return output_file, success_msg
        else:
            error_msg = f"❌ Failed! Return code: {result.returncode}\n"
            error_msg += f"STDERR: {result.stderr[:200]}"
            print(error_msg)
            return None, error_msg
            
    except subprocess.TimeoutExpired:
        error_msg = "❌ Timeout! Inference took too long"
        print(error_msg)
        return None, error_msg
        
    except Exception as e:
        error_msg = f"❌ Error: {type(e).__name__}: {str(e)}"
        print(error_msg)
        
        # Print full traceback to console
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        
        return None, error_msg

# ------------------------------------------------------------------------------
# 3. Create Gradio Interface
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🎨 Creating Gradio Interface")
print("="*70)

# Example texts
example_texts = [
    "xin chào các bạn",
    "hôm nay tôi sẽ giới thiệu về trí tuệ nhân tạo",
    "việt nam là một đất nước xinh đẹp",
]

# Create interface
with gr.Blocks(title=f"F5-TTS Test - {speaker}") as demo:
    gr.Markdown(f"# 🎙️ F5-TTS Simple Test\n## Speaker: {speaker}")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Enter Vietnamese text",
                placeholder="Nhập văn bản tiếng Việt...",
                lines=3
            )
            
            generate_btn = gr.Button("🎙️ Generate Speech", variant="primary")
            
            gr.Examples(
                examples=example_texts,
                inputs=input_text
            )
        
        with gr.Column():
            output_audio = gr.Audio(
                label="Generated Audio",
                type="filepath"
            )
            
            status_text = gr.Textbox(
                label="Status",
                lines=3
            )
    
    # Connect button
    generate_btn.click(
        fn=generate_speech,
        inputs=[input_text],
        outputs=[output_audio, status_text]
    )

# ------------------------------------------------------------------------------
# 4. Launch
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🚀 Launching Gradio")
print("="*70)

try:
    demo.launch(
        share=True,
        debug=True,  # Enable debug mode to see errors
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True  # Show errors in UI
    )
except Exception as e:
    print(f"\n❌ Failed to launch Gradio!")
    print(f"Error: {type(e).__name__}: {str(e)}")
    
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
