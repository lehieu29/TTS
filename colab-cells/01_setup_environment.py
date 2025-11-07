"""
Cell 01: Setup Environment + Mount Google Drive
Mục đích: 
  - Mount Google Drive
  - Tạo virtual environment (venv) tại /content/venv
  - Setup thư mục làm việc
"""

# ============================================================================
# CELL 01: SETUP ENVIRONMENT (Google Colab friendly)
# ============================================================================

print("🚀 Starting Environment Setup...")

import subprocess
import os
import sys

# ------------------------------------------------------------------------------
# 1. Check GPU
# ------------------------------------------------------------------------------
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
# 4. Setup Virtual Environment (Google Colab compatible)
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("🐍 Setting up Virtual Environment (for numpy compatibility)...")
print("="*70)

venv_path = "/content/venv"

def create_venv(path: str):
    """Tạo venv, nếu fail lần 1 thì cài python3-venv rồi thử lại"""
    print(f"🔧 Creating virtual environment at: {path}")
    # Lần 1: thử tạo venv trực tiếp bằng Python hiện tại (Colab)
    result = subprocess.run(
        [sys.executable, "-m", "venv", path],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("✅ Virtual environment created (first attempt)!")
        return

    print("⚠️ First attempt to create venv failed.")
    if result.stderr:
        print("---- stderr (truncated) ----")
        print(result.stderr[:500])
        print("-----------------------------")

    # Lần 2: cài python3-venv, pythonX.Y-venv, rồi thử lại
    print("🔧 Installing python3-venv & retrying...")
    try:
        subprocess.run(["apt-get", "update", "-qq"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        subprocess.run(
            ["apt-get", "install", "-y", "-qq", "python3-venv", f"python{py_ver}-venv"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except Exception as e:
        print(f"❌ apt-get install error: {e}")

    # Thử tạo lại
    result2 = subprocess.run(
        [sys.executable, "-m", "venv", path],
        capture_output=True,
        text=True
    )
    if result2.returncode != 0:
        print("❌ Failed to create venv even after installing python3-venv.")
        if result2.stderr:
            print("---- stderr (truncated) ----")
            print(result2.stderr[:500])
            print("-----------------------------")
        raise RuntimeError(f"Cannot create virtual environment at {path}")
    else:
        print("✅ Virtual environment created successfully after installing python3-venv!")

def ensure_pip_in_venv(path: str):
    """Đảm bảo trong venv có pip (fix lỗi FileNotFoundError: .../venv/bin/pip)"""
    venv_python = os.path.join(path, "bin", "python")
    venv_pip = os.path.join(path, "bin", "pip")

    if not os.path.exists(venv_python):
        raise RuntimeError(f"❌ venv python not found at: {venv_python}. Venv creation failed.")

    if os.path.exists(venv_pip):
        print(f"✅ pip already exists in venv: {venv_pip}")
        return

    print("⚙️  pip not found in venv. Installing pip with ensurepip...")

    # Thử ensurepip
    result = subprocess.run(
        [venv_python, "-m", "ensurepip", "--upgrade"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("⚠️ ensurepip failed. Fallback to get-pip.py")
        print("---- ensurepip stderr (truncated) ----")
        print(result.stderr[:500])
        print("--------------------------------------")

        # Fallback: tải get-pip.py rồi chạy
        get_pip_path = "/content/get-pip.py"
        try:
            subprocess.run(
                ["wget", "-q", "https://bootstrap.pypa.io/get-pip.py", "-O", get_pip_path],
                check=True
            )
            subprocess.run([venv_python, get_pip_path], check=True)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to install pip via get-pip.py: {e}")

    # Sau khi chạy ensurepip/get-pip, kiểm tra lại pip
    if not os.path.exists(venv_pip):
        raise RuntimeError(f"❌ pip still not found in venv at: {venv_pip}")

    print("✅ pip installed inside venv successfully!")
    # Nâng cấp pip, setuptools, wheel cơ bản (nhẹ, tránh conflict)
    subprocess.run(
        [venv_pip, "install", "--upgrade", "pip", "setuptools", "wheel"],
        check=True
    )
    print("✅ pip, setuptools, wheel upgraded inside venv!")

# Tạo venv nếu chưa có
if not os.path.exists(venv_path):
    create_venv(venv_path)
else:
    print(f"✅ Virtual environment already exists at: {venv_path}")

# Đảm bảo có pip trong venv
ensure_pip_in_venv(venv_path)

# Tạo script kích hoạt venv (dùng trong shell cell: `!bash /content/activate_venv.sh`)
activate_script = f"""
# Activate venv (for use in shell cells)
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
print("➡️  Dùng trong shell cell: !bash /content/activate_venv.sh")

# ------------------------------------------------------------------------------
# 5. Create Helper Functions File (optional, nhưng tiện cho các cell sau)
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

def _get_venv_paths():
    venv_path = "/content/venv"
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    venv_python = os.path.join(venv_path, "bin", "python")
    venv_pip = os.path.join(venv_path, "bin", "pip")
    venv_site_packages = os.path.join(venv_path, "lib", f"python{py_ver}", "site-packages")
    return venv_path, venv_python, venv_pip, venv_site_packages

def activate_venv():
    """
    'Kích hoạt' venv theo kiểu Colab:
    - Không đổi interpreter, nhưng thêm site-packages của venv vào sys.path
    - Dùng khi bạn muốn import các package đã cài trong venv
    """
    venv_path, venv_python, venv_pip, venv_site_packages = _get_venv_paths()

    if not os.path.exists(venv_site_packages):
        print(f"⚠️ venv site-packages not found at: {venv_site_packages}")
    else:
        if venv_site_packages not in sys.path:
            sys.path.insert(0, venv_site_packages)

    print(f"✅ Using base Python interpreter: {sys.executable}")
    print(f"✅ Extra packages from venv: {venv_site_packages}")
    return venv_python

def run_in_venv(command):
    """
    Chạy lệnh trong venv:
      - 'pip ...'  -> dùng pip của venv
      - 'python ...' -> dùng python của venv
      - còn lại -> python -c '...'
    """
    venv_path, venv_python, venv_pip, venv_site_packages = _get_venv_paths()

    if command.startswith("pip "):
        cmd = command.replace("pip", venv_pip, 1)
    elif command.startswith("python "):
        cmd = command.replace("python", venv_python, 1)
    else:
        cmd = f"{venv_python} -c '{command}'"

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result

def check_gpu():
    """Check GPU status"""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return None

def get_disk_usage():
    """Get disk usage"""
    result = subprocess.run(["df", "-h", "/content"], capture_output=True, text=True)
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

print(f"""
📁 Working Directories:
   Google Drive: {drive_base}/
   Local Temp: /content/uploads/, /content/temp/, /content/processed/

🐍 Virtual Environment:
   Path: /content/venv/
   Shell Activation: !bash /content/activate_venv.sh
   Python in venv: /content/venv/bin/python
   Pip in venv:    /content/venv/bin/pip

🔧 Helper Functions:
   File: /content/colab_helpers.py
   Import: from colab_helpers import *
   - Gợi ý dùng thêm (tuỳ chọn, không bắt buộc cho Cell 02):
       from colab_helpers import activate_venv, run_in_venv
       activate_venv()  # trước khi import các package trong venv

📝 Next Steps:
   → Run Cell 02 (của bạn) để cài dependencies bằng:
       venv_python = "/content/venv/bin/python"
       venv_pip    = "/content/venv/bin/pip"
   → Cell 02 của bạn đã đúng đường dẫn này rồi, không cần sửa thêm.

⚠️  Important:
   - Nếu đổi đường dẫn venv, nhớ sửa lại cả Cell 02
   - Luôn lưu checkpoint ra Google Drive
   - Monitor GPU memory với !nvidia-smi
""")

print("="*70)
print("🎉 Ready to proceed to Cell 02!")
print("="*70)
