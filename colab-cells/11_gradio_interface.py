"""
Cell 11: Gradio Web Interface
Mục đích:
  - Tạo web UI để test model dễ dàng
  - Multi-speaker support
  - Text input và audio output
  - Share link để demo
"""

# ============================================================================
# CELL 11: GRADIO WEB INTERFACE
# ============================================================================

print("🌐 Starting Gradio Web Interface...")

import os
import sys
import json
import gradio as gr
from pathlib import Path
import subprocess
import time

# Use venv
venv_python = "/content/venv/bin/python"
sys.path.insert(0, '/content/venv/lib/python3.10/site-packages')

# Load config
config_path = "/content/processing_config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

speakers = config.get('trained_speakers', [])

if not speakers:
    print("❌ No trained speakers found!")
    print("   Please train a model first (Cell 09)")
    sys.exit(1)

# Change to F5-TTS directory
os.chdir("/content/F5-TTS-Vietnamese")

# ------------------------------------------------------------------------------
# 1. Prepare Speaker Data
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("📊 Loading Speaker Models...")
print("="*70)

speaker_data = {}

for speaker in speakers:
    model_dir = f"/content/models/{speaker}"
    model_path = f"{model_dir}/model.pt"
    vocab_path = f"{model_dir}/vocab.txt"
    
    if os.path.exists(model_path):
        # Get reference audio
        segments_dir = f"/content/data/{speaker}_training/wavs"
        ref_audios = list(Path(segments_dir).glob("*.wav"))[:1]
        
        if ref_audios:
            ref_audio = str(ref_audios[0])
            ref_text_file = Path(ref_audio).with_suffix('.txt')
            
            if ref_text_file.exists():
                with open(ref_text_file, 'r', encoding='utf-8') as f:
                    ref_text = f.read().strip()
            else:
                ref_text = ""
            
            speaker_data[speaker] = {
                'model_path': model_path,
                'vocab_path': vocab_path,
                'ref_audio': ref_audio,
                'ref_text': ref_text
            }
            
            print(f"✅ {speaker}: Model loaded")
        else:
            print(f"⚠️  {speaker}: No reference audio found")
    else:
        print(f"❌ {speaker}: Model not found")

if not speaker_data:
    print("\n❌ No valid speaker models found!")
    sys.exit(1)

# ------------------------------------------------------------------------------
# 2. Define Inference Function
# ------------------------------------------------------------------------------

def generate_speech(speaker_name, input_text, speed=1.0):
    """
    Generate speech using trained model
    """
    if not speaker_name or not input_text:
        return None, "⚠️ Please select speaker and enter text"
    
    if speaker_name not in speaker_data:
        return None, f"❌ Speaker {speaker_name} not found"
    
    # Get speaker info
    speaker_info = speaker_data[speaker_name]
    
    # Output file
    output_file = f"/content/outputs/{speaker_name}_{int(time.time())}.wav"
    os.makedirs("/content/outputs", exist_ok=True)
    
    # Inference command
    cmd = [
        venv_python, "-m", "f5_tts.infer.infer_cli",
        "--model", "F5TTS_Base",
        "--ref_audio", speaker_info['ref_audio'],
        "--ref_text", speaker_info['ref_text'],
        "--gen_text", input_text,
        "--gen_file", output_file,
        "--vocab_file", speaker_info['vocab_path'],
        "--ckpt_file", speaker_info['model_path'],
        "--speed", str(speed),
        "--nfe_step", "32"
    ]
    
    status_msg = f"🎙️ Generating speech for {speaker_name}...\n"
    status_msg += f"Text: {input_text[:50]}...\n"
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0 and os.path.exists(output_file):
            status_msg += f"\n✅ Generation complete!\n"
            status_msg += f"File: {output_file}\n"
            status_msg += f"Size: {os.path.getsize(output_file) / 1024:.1f} KB"
            
            return output_file, status_msg
        else:
            status_msg += f"\n❌ Generation failed!\n{result.stderr}"
            return None, status_msg
            
    except subprocess.TimeoutExpired:
        return None, status_msg + "\n❌ Timeout!"
    except Exception as e:
        return None, status_msg + f"\n❌ Error: {e}"

# ------------------------------------------------------------------------------
# 3. Create Gradio Interface
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🎨 Creating Gradio Interface...")
print("="*70)

# Example texts
example_texts = [
    "xin chào các bạn, hôm nay tôi sẽ giới thiệu về trí tuệ nhân tạo",
    "việt nam là một đất nước xinh đẹp với văn hóa phong phú",
    "công nghệ đang phát triển rất nhanh trong những năm gần đây",
    "tôi rất vui được chia sẻ kiến thức với mọi người",
    "học máy và trí tuệ nhân tạo đang thay đổi cuộc sống của chúng ta"
]

# Create Gradio interface
with gr.Blocks(title="F5-TTS Vietnamese Voice Cloning") as demo:
    gr.Markdown("""
    # 🎙️ F5-TTS Vietnamese Voice Cloning
    
    Generate speech in Vietnamese using trained voices!
    """)
    
    with gr.Row():
        with gr.Column():
            # Speaker selection
            speaker_dropdown = gr.Dropdown(
                choices=list(speaker_data.keys()),
                value=list(speaker_data.keys())[0],
                label="👤 Select Speaker",
                info="Choose which voice to use"
            )
            
            # Text input
            text_input = gr.Textbox(
                label="✍️ Enter Vietnamese Text",
                placeholder="Nhập văn bản tiếng Việt...",
                lines=5,
                info="Enter the text you want to convert to speech"
            )
            
            # Speed control
            speed_slider = gr.Slider(
                minimum=0.5,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="⚡ Speed",
                info="Adjust speech speed"
            )
            
            # Generate button
            generate_btn = gr.Button(
                "🎙️ Generate Speech",
                variant="primary",
                size="lg"
            )
            
            # Example buttons
            gr.Markdown("### 📝 Quick Examples:")
            example_buttons = []
            for i, text in enumerate(example_texts[:3], 1):
                btn = gr.Button(f"Example {i}", size="sm")
                btn.click(
                    fn=lambda t=text: t,
                    outputs=text_input
                )
        
        with gr.Column():
            # Audio output
            audio_output = gr.Audio(
                label="🔊 Generated Audio",
                type="filepath"
            )
            
            # Status output
            status_output = gr.Textbox(
                label="📊 Status",
                lines=8,
                interactive=False
            )
            
            # Info
            gr.Markdown(f"""
            ### ℹ️ Information:
            - **Available Speakers:** {len(speaker_data)}
            - **Model:** F5-TTS Base
            - **Language:** Vietnamese
            - **Quality:** Depends on training data
            
            ### 💡 Tips:
            - Use proper Vietnamese diacritics
            - Add punctuation for better prosody
            - Shorter texts (< 100 words) work best
            - Adjust speed if needed
            """)
    
    # Connect generate button
    generate_btn.click(
        fn=generate_speech,
        inputs=[speaker_dropdown, text_input, speed_slider],
        outputs=[audio_output, status_output]
    )
    
    gr.Markdown("""
    ---
    ### 🎯 Next Steps:
    - Test with different texts
    - Try different speakers
    - Adjust speed for natural speech
    - Share the link with others!
    
    ### 📝 Notes:
    - Generation takes 5-10 seconds
    - All audio saved to `/content/outputs/`
    - Models backed up to Google Drive
    """)

# ------------------------------------------------------------------------------
# 4. Launch Interface
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🚀 Launching Gradio Interface...")
print("="*70)

print("""
⚙️  Launching Settings:
   - Server: 0.0.0.0:7860
   - Share: Yes (public link)
   - Debug: False
   
⏳ Starting server...
""")

try:
    # Launch Gradio with share=True for public link
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Create public link
        debug=False,
        show_error=True
    )
    
except Exception as e:
    print(f"\n❌ Failed to launch Gradio: {e}")
    print("\n💡 Alternative: Run inference via Cell 10")



