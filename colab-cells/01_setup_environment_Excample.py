"""
Cell 01: Setup Environment + Mount Google Drive
Mục đích: 
  - Mount Google Drive
  - Tạo virtual environment (venv) 
  - Setup thư mục làm việc
"""

# ============================================================================
# CELL 01: SETUP ENVIRONMENT
# ============================================================================

print("🚀 Starting Environment Setup...")

# ------------------------------------------------------------------------------
# 1. Check GPU
# ------------------------------------------------------------------------------
import subprocess
import os

print("\n" + "="*70)
print("📊 Checking GPU...")
print("="*70)

gpu_info = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
if gpu_info.returncode == 0:
    print("✅ GPU Available!")
    print(gpu_info.stdout)
else:
    print("⚠️  No GPU detected. Training sẽ rất chậm!")
    print("💡 Tip: Runtime → Change runtime type → GPU")

# ------------------------------------------------------------------------------
# 2. Mount Google Drive
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("💾 Mounting Google Drive...")
print("="*70)

from google.colab import drive

try:
    drive.mount('/content/drive', force_remount=False)
    print("✅ Google Drive mounted successfully!")
except Exception as e:
    print(f"❌ Failed to mount Drive: {e}")
    print("Please authorize and try again")

# ------------------------------------------------------------------------------
# 3. Create Working Directories
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("📁 Creating Working Directories...")
print("="*70)

# Thư mục trên Drive (persistent)
drive_base = "/content/drive/MyDrive/F5TTS_Vietnamese"
directories = {
    "base": drive_base,
    "models": f"{drive_base}/models",
    "datasets": f"{drive_base}/datasets", 
    "outputs": f"{drive_base}/outputs",
    "checkpoints": f"{drive_base}/checkpoints",
    "uploads": f"{drive_base}/uploads",
    "logs": f"{drive_base}/logs"
}

for name, path in directories.items():
    os.makedirs(path, exist_ok=True)
    print(f"✅ Created: {path}")

# Thư mục local (faster access)
local_dirs = [
    "/content/uploads",
    "/content/temp",
    "/content/processed"
]

for path in local_dirs:
    os.makedirs(path, exist_ok=True)
    print(f"✅ Created: {path}")

# ------------------------------------------------------------------------------
# 4. Setup Virtual Environment
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🐍 Setting up Virtual Environment (for numpy compatibility)...")
print("="*70)

venv_path = "/content/venv"

# Create venv if not exists
if not os.path.exists(venv_path):
    print("Creating virtual environment...")
    subprocess.run(["python", "-m", "venv", venv_path], check=True)
    print("✅ Virtual environment created!")
else:
    print("✅ Virtual environment already exists")

# Create activation helper script
activate_script = f"""
# Activate venv
source {venv_path}/bin/activate

# Verify activation
echo "✅ Virtual environment activated!"
echo "Python: $(which python)"
echo "Pip: $(which pip)"
"""

with open("/content/activate_venv.sh", "w") as f:
    f.write(activate_script)

print(f"\n📝 Virtual environment created at: {venv_path}")
print(f"📝 Activation script saved at: /content/activate_venv.sh")

# ------------------------------------------------------------------------------
# 5. Create Helper Functions File
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🔧 Creating Helper Functions...")
print("="*70)

helper_code = '''
"""
Helper functions for Colab cells
"""
import os
import subprocess
import sys

def activate_venv():
    """Activate virtual environment"""
    venv_path = "/content/venv"
    activate_script = os.path.join(venv_path, "bin", "activate_this.py")
    
    # Alternative: modify sys.path
    venv_python = os.path.join(venv_path, "bin", "python")
    venv_site_packages = os.path.join(venv_path, "lib", "python3.10", "site-packages")
    
    if venv_site_packages not in sys.path:
        sys.path.insert(0, venv_site_packages)
    
    print(f"✅ Using Python: {sys.executable}")
    print(f"✅ Using packages from: {venv_site_packages}")
    
    return venv_python

def run_in_venv(command):
    """Run command in virtual environment"""
    venv_python = "/content/venv/bin/python"
    venv_pip = "/content/venv/bin/pip"
    
    if command.startswith("pip "):
        cmd = command.replace("pip", venv_pip)
    elif command.startswith("python "):
        cmd = command.replace("python", venv_python)
    else:
        cmd = f"{venv_python} -c '{command}'"
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result

def check_gpu():
    """Check GPU status"""
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', 
                           '--format=csv,noheader'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    return None

def get_disk_usage():
    """Get disk usage"""
    result = subprocess.run(['df', '-h', '/content'], capture_output=True, text=True)
    return result.stdout

def save_checkpoint(model_path, drive_path):
    """Save checkpoint to Google Drive"""
    import shutil
    os.makedirs(os.path.dirname(drive_path), exist_ok=True)
    shutil.copy2(model_path, drive_path)
    print(f"✅ Checkpoint saved to: {drive_path}")
'''

with open("/content/colab_helpers.py", "w") as f:
    f.write(helper_code)

print("✅ Helper functions created at: /content/colab_helpers.py")

# ------------------------------------------------------------------------------
# 6. Display Summary
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("✅ SETUP COMPLETE!")
print("="*70)

print("""
📁 Working Directories:
   Google Drive: /content/drive/MyDrive/F5TTS_Vietnamese/
   Local Temp: /content/uploads/, /content/temp/

🐍 Virtual Environment:
   Path: /content/venv/
   Activation: Source in each cell that needs it

🔧 Helper Functions:
   File: /content/colab_helpers.py
   Import: from colab_helpers import *

📝 Next Steps:
   → Run Cell 02 to install dependencies
   → Make sure to activate venv in each cell!

⚠️  Important:
   - Always activate venv before installing packages
   - Save checkpoints to Drive frequently
   - Monitor GPU memory with !nvidia-smi
""")

print("="*70)
print("🎉 Ready to proceed to Cell 02!")
print("="*70)



