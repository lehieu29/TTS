# Cell 11 Debug Guide - Hướng Dẫn Debug Gradio

## 🎯 Mục Đích

Khi Cell 11 (Gradio) chạy nhưng không hiển thị lỗi rõ ràng, sử dụng các cell debug này để xác định nguyên nhân.

---

## 📋 Các File Debug

### 1. **11_gradio_debug.py** - Terminal Test (Khuyến nghị chạy TRƯỚC)

**Mục đích:** Test inference trực tiếp trên terminal, không qua Gradio

**Khi nào dùng:**
- Gradio khởi chạy nhưng không generate audio
- Muốn xem chi tiết lỗi inference
- Test xem model có load được không

**Cách chạy trong Colab:**
```python
%run /content/F5-TTS-Vietnamese/colab-cells/11_gradio_debug.py
```

**Output mong đợi:**
```
✅ SUCCESS!
   Output file: /content/outputs/Podcast_Thuan(3)_debug_test.wav
   Size: 0.XX MB
```

**Nếu LỖI, bạn sẽ thấy:**
- `STDOUT`: Output của inference script
- `STDERR`: Thông báo lỗi chi tiết
- `Full traceback`: Stack trace đầy đủ

---

### 2. **11_gradio_simple_test.py** - Minimal Gradio Test

**Mục đích:** Test Gradio với giao diện đơn giản nhất, dễ debug

**Khi nào dùng:**
- File debug #1 chạy OK (terminal test thành công)
- Muốn test xem lỗi có phải do Gradio integration không
- Cần debug Gradio function trực tiếp

**Cách chạy trong Colab:**
```python
%run /content/F5-TTS-Vietnamese/colab-cells/11_gradio_simple_test.py
```

**Đặc điểm:**
- ✅ `debug=True` - Hiển thị lỗi trên console
- ✅ `show_error=True` - Hiển thị lỗi trên UI
- ✅ Print mọi bước trong function
- ✅ Giao diện đơn giản, 1 speaker duy nhất

---

## 🔍 Quy Trình Debug

### **Bước 1: Chạy Terminal Test**

```python
# Trong Colab cell
%run /content/F5-TTS-Vietnamese/colab-cells/11_gradio_debug.py
```

#### Kết quả:

**A. Nếu THÀNH CÔNG ✅:**
```
✅ SUCCESS!
   Output file: /content/outputs/...
```
→ **Model và inference hoạt động OK**  
→ Lỗi nằm ở Gradio integration  
→ Chuyển sang **Bước 2**

**B. Nếu THẤT BẠI ❌:**

**Lỗi 1: Model không load được**
```
❌ Model not found at: /content/models/...
❌ Checkpoint not found at: /content/F5-TTS-Vietnamese/ckpts/...
```
→ **Nguyên nhân:** Cell 09 chưa chạy hoặc checkpoint không save  
→ **Giải pháp:** Chạy lại Cell 09 (Training)

**Lỗi 2: Vocab size mismatch**
```
RuntimeError: size mismatch for text_embed.weight
```
→ **Nguyên nhân:** Vocab không khớp với checkpoint  
→ **Giải pháp:** Chạy lại Cell 08 với fix vocab đã sửa

**Lỗi 3: FileNotFoundError trong inference**
```
FileNotFoundError: [Errno 2] No such file or directory
```
→ **Nguyên nhân:** Reference audio hoặc file path sai  
→ **Giải pháp:** Check đường dẫn trong output STDERR

**Lỗi 4: Module import error**
```
ModuleNotFoundError: No module named 'f5_tts'
```
→ **Nguyên nhân:** Virtual environment chưa được activate  
→ **Giải pháp:** Check venv_python path hoặc cài lại dependencies

---

### **Bước 2: Chạy Simple Gradio Test**

```python
# Sau khi Bước 1 thành công
%run /content/F5-TTS-Vietnamese/colab-cells/11_gradio_simple_test.py
```

#### Kết quả:

**A. Nếu THÀNH CÔNG ✅:**
- Gradio UI xuất hiện
- Click "Generate Speech" → Audio được tạo
- Status hiển thị "✅ Success!"

→ **Gradio hoạt động OK**  
→ Lỗi ở Cell 11 chính là do code phức tạp hơn  
→ So sánh code giữa `11_gradio_simple_test.py` và `11_gradio_interface.py`

**B. Nếu THẤT BẠI ❌:**

**Lỗi 1: Gradio không launch**
```
❌ Failed to launch Gradio!
Error: ...
```
→ Check traceback để xem lỗi cụ thể  
→ Thường do: port conflict, network issue

**Lỗi 2: Generate button không hoạt động**
- Click button nhưng không có gì xảy ra
- Status không update

→ Check console output trong Colab  
→ Lỗi thường được print ra console với `debug=True`

**Lỗi 3: Audio không play được**
- File được tạo nhưng không nghe được
- Gradio Audio component trống

→ Check file có tồn tại: `!ls -lh /content/outputs/`  
→ Check file size > 0  
→ Thử download file về máy để test

---

## 🛠️ Các Lệnh Debug Hữu Ích

### Check Model Files
```bash
# Check model exists
!ls -lh /content/models/*/

# Check training checkpoints
!ls -lh /content/F5-TTS-Vietnamese/ckpts/*/

# Check model file size
!du -h /content/models/*/model.pt
```

### Check Output Files
```bash
# List generated audio files
!ls -lh /content/outputs/

# Play audio in Colab
from IPython.display import Audio, display
display(Audio('/content/outputs/YOUR_FILE.wav', rate=24000))
```

### Check Gradio Process
```bash
# Check if Gradio is running
!ps aux | grep gradio

# Check port 7860
!netstat -tuln | grep 7860
```

### Kill Gradio Process
```bash
# If Gradio stuck, kill it
!pkill -f gradio

# Or kill by port
!fuser -k 7860/tcp
```

---

## 📊 Troubleshooting Table

| Triệu Chứng | Nguyên Nhân Có Thể | Debug Step | Giải Pháp |
|-------------|-------------------|------------|-----------|
| Gradio launch OK, nhưng không generate audio | Inference command sai | Bước 1 | Check STDERR trong terminal test |
| "Model not found" | Checkpoint không có | Bước 1 | Chạy lại Cell 09 |
| "Vocab size mismatch" | Vocab không khớp | Bước 1 | Chạy lại Cell 08 |
| Gradio không launch | Port conflict | Bước 2 | Kill process hoặc đổi port |
| Generate button không làm gì | Function error | Bước 2 | Check console với debug=True |
| Audio file trống (0 bytes) | Inference failed silently | Bước 1+2 | Check return code và STDERR |

---

## 💡 Tips

1. **Luôn chạy Terminal Test (Bước 1) TRƯỚC:**
   - Nhanh hơn
   - Lỗi rõ ràng hơn
   - Không cần đợi Gradio UI load

2. **Enable debug mode:**
   - Simple test đã có `debug=True`
   - Xem tất cả print statements trong console

3. **Check console output:**
   - Colab console thường có nhiều thông tin hơn UI
   - Scroll lên xem các dòng print trước đó

4. **Test với text ngắn trước:**
   - "xin chào" (2 từ)
   - Nếu OK → test text dài hơn
   - Nếu fail → lỗi không phải do text length

5. **So sánh với Cell 10:**
   - Cell 10 chạy OK nhưng Cell 11 fail
   - → Lỗi ở Gradio integration
   - Compare code giữa 2 cells

---

## 🎯 Success Criteria

### Terminal Test (Bước 1) Thành Công Khi:
- ✅ Return code: 0
- ✅ Output file exists
- ✅ File size > 0.1 MB
- ✅ Có thể play audio

### Gradio Test (Bước 2) Thành Công Khi:
- ✅ Gradio UI loads
- ✅ Public link accessible
- ✅ Click Generate → Audio appears
- ✅ Status shows "✅ Success!"
- ✅ Can play audio in browser

---

## 📞 Next Steps

Sau khi debug xong:

1. **Nếu cả 2 test đều OK:**
   - Lỗi ở Cell 11 chính là do code phức tạp
   - Copy logic từ `11_gradio_simple_test.py` sang `11_gradio_interface.py`
   - Hoặc sử dụng simple version thay vì full version

2. **Nếu Terminal test OK, Gradio test fail:**
   - Lỗi ở Gradio integration
   - Check Gradio version: `!pip show gradio`
   - Thử update Gradio: `!pip install --upgrade gradio`

3. **Nếu cả 2 đều fail:**
   - Quay lại check Cell 08 và Cell 09
   - Đảm bảo training hoàn thành thành công
   - Verify checkpoint files tồn tại và có kích thước đúng

---

## 📝 Report Bug Template

Nếu vẫn gặp lỗi sau khi debug, report theo format:

```
**Environment:**
- Colab: Free / Pro
- GPU: T4 / V100 / etc
- Python: 3.10 / 3.11 / etc

**Steps:**
1. Ran Cell 09 - Training
2. Ran 11_gradio_debug.py
3. Got error: ...

**Terminal Test Output:**
[Paste full output here]

**Gradio Test Output:**
[Paste console output here]

**Error Message:**
[Paste specific error]

**Files Status:**
- Model: exists / not exists
- Vocab: exists / not exists
- Checkpoint size: XXX MB
```
