# Hướng Dẫn Backup Checkpoints

## 🎯 Mục Đích

Backup checkpoints từ Cell 9 vào Google Drive để:
- Không mất checkpoint khi Colab timeout
- Có thể restore lại khi restart Colab
- Backup an toàn trên Drive

---

## 📦 Có 2 Scripts Backup:

### **1. `backup_simple.py` - Script Đơn Giản (KHUYẾN NGHỊ)**

**Đặc điểm:**
- ✅ Siêu đơn giản (~50 lines)
- ✅ Tự động tìm và copy tất cả files
- ✅ Không cần config gì
- ✅ Nhanh

**Cách dùng trong Colab:**

```python
# Mount Drive (nếu chưa)
from google.colab import drive
drive.mount('/content/drive')

# Chạy backup
%run /content/F5-TTS-Vietnamese/backup_simple.py
```

**Output:**
```
📦 BACKING UP CHECKPOINTS...
✅ Podcast_Thuan(3)/model_last.pt
✅ Podcast_Thuan(3)/model_100000.pt
✅ Podcast_Thuan(3)/vocab.txt
✅ Podcast_Thuan(3)/model.pt

✅ Done! Copied 4 files to Drive
📂 Location: /content/drive/MyDrive/F5TTS_Vietnamese/
```

---

### **2. `backup_checkpoints_to_drive.py` - Script Đầy Đủ**

**Đặc điểm:**
- ✅ Có verification
- ✅ Có summary chi tiết
- ✅ Skip files đã backup
- ✅ Hiển thị file size
- ✅ Error handling

**Cách dùng trong Colab:**

```python
# Mount Drive (nếu chưa)
from google.colab import drive
drive.mount('/content/drive')

# Chạy backup
%run /content/F5-TTS-Vietnamese/colab-cells/backup_checkpoints_to_drive.py
```

**Output:**
```
======================================================================
📦 BACKUP CHECKPOINTS TO GOOGLE DRIVE
======================================================================

✅ Drive mounted: /content/drive/MyDrive/F5TTS_Vietnamese
✅ Found 1 trained speaker(s): ['Podcast_Thuan(3)']

======================================================================
🔍 SEARCHING FOR CHECKPOINTS...
======================================================================

✅ Found training checkpoints: Podcast_Thuan(3)_training
   Files: 3 checkpoints (412.5 MB)

✅ Found organized model: Podcast_Thuan(3)
   - model.pt: 206.32 MB
   - vocab.txt: 0.00 MB
   - config.json: 0.00 MB

======================================================================
📤 BACKING UP TO DRIVE...
======================================================================

📁 Processing: Podcast_Thuan(3) (checkpoints)
   📄 Copying: model_last.pt... ✅ (206.3 MB)
   📄 Copying: model_100000.pt... ✅ (206.2 MB)
   ✅ Backed up 2 files (412.5 MB)
   📂 Destination: /content/drive/MyDrive/F5TTS_Vietnamese/checkpoints/Podcast_Thuan(3)

📁 Processing: Podcast_Thuan(3) (models)
   📄 Copying: model.pt... ✅ (206.3 MB)
   📄 Copying: vocab.txt... ✅ (0.0 MB)
   📄 Copying: config.json... ✅ (0.0 MB)
   ✅ Backed up 3 files (206.3 MB)
   📂 Destination: /content/drive/MyDrive/F5TTS_Vietnamese/models/Podcast_Thuan(3)

======================================================================
📊 BACKUP SUMMARY
======================================================================

✅ Successfully backed up: 2/2 speakers
📄 Total files copied: 5
💾 Total size: 618.8 MB

📂 Backup locations on Drive:
   ✅ Podcast_Thuan(3) (checkpoints)
      /content/drive/MyDrive/F5TTS_Vietnamese/checkpoints/Podcast_Thuan(3)
      2 files, 412.5 MB
   ✅ Podcast_Thuan(3) (models)
      /content/drive/MyDrive/F5TTS_Vietnamese/models/Podcast_Thuan(3)
      3 files, 206.3 MB

======================================================================
🔍 VERIFYING BACKUP...
======================================================================

📁 Podcast_Thuan(3):
   ✅ model_last.pt: 206.3 MB

📁 Podcast_Thuan(3):
   ✅ model.pt: 206.32 MB
   ✅ vocab.txt: 0.00 MB

======================================================================
✅ BACKUP COMPLETE!
======================================================================

💡 Next steps:
   1. Verify files on Google Drive web interface
   2. When restarting Colab, run Cell 10 or 11
   3. Models will auto-load from Drive

📂 Drive structure:
   /content/drive/MyDrive/F5TTS_Vietnamese/
   ├── checkpoints/
   │   └── Podcast_Thuan(3)/
   └── models/
       └── Podcast_Thuan(3)/
```

---

## 🚀 Quick Start

### **Cách Nhanh Nhất (Trong Colab):**

```python
# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Backup (chọn 1 trong 2)
# Simple version (khuyến nghị):
%run /content/F5-TTS-Vietnamese/backup_simple.py

# Hoặc full version:
%run /content/F5-TTS-Vietnamese/colab-cells/backup_checkpoints_to_drive.py
```

---

## 📂 Cấu Trúc Backup Trên Drive

```
Google Drive/
└── My Drive/
    └── F5TTS_Vietnamese/
        ├── checkpoints/
        │   └── Podcast_Thuan(3)/
        │       ├── model_last.pt           ← Checkpoint cuối (quan trọng nhất)
        │       ├── model_100000.pt         ← Checkpoint theo step
        │       ├── model_200000.pt
        │       └── pretrained_model_1200000.pt
        │
        └── models/
            └── Podcast_Thuan(3)/
                ├── model.pt                ← Model ready-to-use
                ├── vocab.txt               ← Vocabulary
                └── config.json             ← Config
```

---

## 🔍 Verify Backup

### **Cách 1: Check Trong Colab**

```python
import os
from pathlib import Path

drive_base = "/content/drive/MyDrive/F5TTS_Vietnamese"
speaker = "Podcast_Thuan(3)"

# Check checkpoints
ckpt_dir = f"{drive_base}/checkpoints/{speaker}"
if os.path.exists(ckpt_dir):
    files = list(Path(ckpt_dir).glob("*.pt"))
    print(f"✅ {len(files)} checkpoints found")
    for f in files:
        size_mb = f.stat().st_size / (1024**2)
        print(f"   - {f.name}: {size_mb:.1f} MB")
else:
    print("❌ Checkpoints not found")

# Check models
model_dir = f"{drive_base}/models/{speaker}"
if os.path.exists(model_dir):
    for filename in ["model.pt", "vocab.txt"]:
        filepath = os.path.join(model_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024**2)
            print(f"   ✅ {filename}: {size_mb:.2f} MB")
else:
    print("❌ Models not found")
```

### **Cách 2: Check Trên Web**

1. Mở https://drive.google.com
2. Navigate: **My Drive → F5TTS_Vietnamese**
3. Check folders:
   - `checkpoints/Podcast_Thuan(3)/` → Có file `.pt`?
   - `models/Podcast_Thuan(3)/` → Có `model.pt` và `vocab.txt`?

---

## ⚠️ Lưu Ý

### **Files Quan Trọng Nhất:**
1. ✅ `model_last.pt` hoặc `model.pt` (~200-400 MB)
2. ✅ `vocab.txt` (~1-2 KB)

Chỉ cần 2 files này là đủ để chạy inference!

### **Khi Nào Backup?**

✅ **Backup ngay sau khi:**
- Cell 09 training xong
- Trước khi đóng Colab
- Định kỳ (mỗi vài giờ nếu training lâu)

### **Nếu Drive Đầy?**

Chỉ cần backup files quan trọng:
```python
# Backup minimal - chỉ model.pt và vocab.txt
import shutil
import os

speaker = "Podcast_Thuan(3)"
drive_base = "/content/drive/MyDrive/F5TTS_Vietnamese/models"

os.makedirs(f"{drive_base}/{speaker}", exist_ok=True)

# Copy 2 files quan trọng
shutil.copy2(
    f"/content/models/{speaker}/model.pt",
    f"{drive_base}/{speaker}/model.pt"
)
shutil.copy2(
    f"/content/models/{speaker}/vocab.txt",
    f"{drive_base}/{speaker}/vocab.txt"
)

print("✅ Backed up essential files only")
```

---

## 🔄 Restore Khi Restart Colab

**KHÔNG CẦN làm gì!**

Cell 10 và 11 đã có logic tự động load từ Drive:
```python
# Cell 10/11 sẽ tự động:
# 1. Check local /content/models/
# 2. Nếu không có → Load từ Drive
# 3. Copy về local
# 4. Chạy inference
```

Chỉ cần:
1. Mount Drive (Cell 03)
2. Chạy Cell 10 hoặc 11
3. Done! 🎉

---

## 💡 Tips

### **1. Backup Tự Động (Trong Cell 09):**

Thêm vào cuối Cell 09:
```python
# Auto backup after training
print("\n📦 Auto-backing up to Drive...")
%run /content/F5-TTS-Vietnamese/backup_simple.py
```

### **2. Backup Định Kỳ:**

Nếu training lâu (>2 giờ), thêm vào training loop:
```python
# Trong Cell 09, sau mỗi X steps
if step % 10000 == 0:
    print(f"\n📦 Backup checkpoint at step {step}...")
    %run /content/F5-TTS-Vietnamese/backup_simple.py
```

### **3. Check Drive Space:**

```python
!df -h /content/drive
```

---

## 📞 Troubleshooting

### **Q: Script báo "Drive not mounted"?**
A: Chạy trước: `from google.colab import drive; drive.mount('/content/drive')`

### **Q: Script không tìm thấy checkpoints?**
A: Check đường dẫn:
```python
!ls -lh /content/F5-TTS-Vietnamese/ckpts/
!ls -lh /content/models/
```

### **Q: Files đã backup nhưng vẫn copy lại?**
A: Dùng full version script, nó sẽ skip files đã backup

### **Q: Backup quá chậm?**
A: Chỉ backup files cần thiết (model.pt + vocab.txt)

---

## ✅ Checklist

- [ ] Mount Google Drive
- [ ] Chạy backup script
- [ ] Verify files trên Drive
- [ ] Test restore (restart Colab → chạy Cell 10)
- [ ] Confirm inference works

---

**Khuyến nghị: Dùng `backup_simple.py` - Nhanh và đủ dùng!** 🚀
