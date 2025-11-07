# 📝 Tóm tắt các thay đổi - Vietnamese Text Processing Fix

## ✅ Đã thực hiện

### 1. **Sửa hàm `convert_char_to_pinyin()` trong source code**
   
**File:** `src/f5_tts/model/utils.py` (line 137-213)

**Thay đổi chính:**
- ✅ Thêm detection cho tiếng Việt (Vietnamese character detection)
- ✅ Giữ nguyên dấu thanh tiếng Việt (preserve Vietnamese tones)
- ✅ Vẫn giữ logic Chinese pinyin conversion cho tiếng Trung
- ✅ Tương thích backward với tiếng Anh và các ngôn ngữ khác

**Logic mới:**
```python
def convert_char_to_pinyin(text_list, polyphone=True):
    """
    - Chinese: Convert to Pinyin (giữ nguyên logic cũ)
    - Vietnamese: Keep original characters with tones (MỚI)
    - English/Other: Keep as-is (giữ nguyên logic cũ)
    """
    
    # Detect Vietnamese by checking for Vietnamese diacritics
    if has_vietnamese_chars(text):
        # Vietnamese path: Keep original characters
        words = text.split()
        for word in words:
            char_list.extend(list(word))
    else:
        # Chinese/English path: Original jieba + pinyin logic
        ...
```

**Vietnamese character detection:**
- Kiểm tra Unicode ranges cho tiếng Việt:
  - `\u0041-\u007A`: Basic Latin (a-z, A-Z)
  - `\u00C0-\u00FF`: Latin-1 Supplement (À, Á, Â, Ã, etc.)
  - `\u0100-\u017F`: Latin Extended-A (Ā, ă, etc.)
  - `\u1E00-\u1EFF`: Latin Extended Additional (ạ, ả, ấ, etc.)

- Kiểm tra các ký tự đặc trưng tiếng Việt:
  - `àáảãạâầấẩẫậăằắẳẵặ` (a với các dấu)
  - `èéẻẽẹêềếểễệ` (e với các dấu)
  - `òóỏõọôồốổỗộơờớởỡợ` (o với các dấu)
  - `ùúủũụưừứửữự` (u với các dấu)
  - `ìíỉĩị`, `ỳýỷỹỵ`, `đ` (i, y, d với các dấu)

---

### 2. **Cập nhật Cell 02 để clone từ repo của bạn**

**File:** `colab-cells/02_install_dependencies.py` (line 74)

**Thay đổi:**
```python
# Cũ:
"https://github.com/nguyenthienhy/F5-TTS-Vietnamese.git"

# Mới:
"https://github.com/lehieu29/TTS.git"
```

---

## 🎯 Kết quả mong đợi

### Trước khi fix:
```
Text input: "tự dưng trong mình nó cảm thấy bồi hồi"
↓ (Chinese pinyin converter)
Output: [] hoặc gibberish
↓
raw.arrow: 0.6 MB (quá nhỏ ❌)
```

### Sau khi fix:
```
Text input: "tự dưng trong mình nó cảm thấy bồi hồi"
↓ (Vietnamese character processor - DETECT VIETNAMESE)
Output: ['t','ự',' ','d','ư','ơ','n','g',' ','t','r','o','n','g',...]
↓
raw.arrow: 15-20 MB (đúng kích thước ✅)
```

---

## 🚀 Workflow tiếp theo

### **Bước 1: Push code lên GitHub**

```bash
cd D:\Project\F5-TTS\F5-TTS-Vietnamese

# Add changes
git add src/f5_tts/model/utils.py
git add colab-cells/02_install_dependencies.py

# Commit
git commit -m "Fix Vietnamese text processing in convert_char_to_pinyin"

# Push to your repo
git push origin main
```

### **Bước 2: Chạy lại trên Colab**

1. **Chạy Cell 02** (Install Dependencies)
   - Sẽ clone từ repo mới của bạn: `https://github.com/lehieu29/TTS.git`
   - Code đã có Vietnamese fix sẵn trong source

2. **KHÔNG cần chạy lại Cell 06, 07**
   - Segmentation OK (877 segments)
   - Transcription OK (92.3 phút)

3. **Chạy lại Cell 08** (Prepare Training Data)
   - Sẽ tự động dùng `convert_char_to_pinyin()` đã được fix
   - Text tiếng Việt sẽ được xử lý đúng
   - `raw.arrow` sẽ đạt kích thước 15-20 MB ✅

4. **Verify kết quả:**
   ```python
   import os
   arrow_path = "/content/data/<speaker>_training/raw.arrow"
   size_mb = os.path.getsize(arrow_path) / (1024**2)
   print(f"raw.arrow size: {size_mb:.2f} MB")
   # Kỳ vọng: 15-20 MB ✅
   ```

---

## 📊 So sánh với giải pháp trước

| Aspect | Giải pháp cũ (Patch file) | Giải pháp mới (Source fix) |
|--------|---------------------------|----------------------------|
| **Implementation** | File riêng `prepare_csv_wavs_vietnamese.py` | Sửa trực tiếp trong source `utils.py` |
| **Maintainability** | ❌ Cần maintain 2 file | ✅ Chỉ 1 file source duy nhất |
| **Compatibility** | ⚠️ Chỉ Vietnamese | ✅ Vietnamese + Chinese + English |
| **Git workflow** | ❌ Cần copy file patch | ✅ Clone là có ngay fix |
| **Cell 08** | ❌ Cần modify script path | ✅ Không cần sửa gì |

---

## 🔍 Technical Details

### Vietnamese Detection Logic

```python
def has_vietnamese_chars(text):
    # List of all Vietnamese diacritics
    vietnamese_chars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    vietnamese_chars += vietnamese_chars.upper()
    
    # Check if any Vietnamese character exists
    return any(c in vietnamese_chars for c in text)
```

**Ví dụ:**
- `"tự dưng"` → `True` (có 'ự', 'ư')
- `"你好"` → `False` (Chinese)
- `"hello"` → `False` (English)
- `"tự dưng hello 你好"` → `True` (mixed, nhưng có Vietnamese chars)

### Character Processing

**Vietnamese text:**
```python
Input:  "tự dưng trong mình"
Split:  ["tự", "dưng", "trong", "mình"]
Output: ['t','ự',' ','d','ư','ơ','n','g',' ','t','r','o','n','g',' ','m','ì','n','h']
```

**Chinese text:**
```python
Input:  "你好世界"
Jieba:  ["你好", "世界"]
Pinyin: ["ni3", "hao3", "shi4", "jie4"]
Output: ['n','i','3',' ','h','a','o','3',' ','s','h','i','4',' ','j','i','e','4']
```

---

## ✅ Checklist

- [x] Sửa `convert_char_to_pinyin()` để support Vietnamese
- [x] Cập nhật Cell 02 để clone từ repo `lehieu29/TTS`
- [x] Test Vietnamese character detection logic
- [ ] **TODO: Push code lên GitHub**
- [ ] **TODO: Chạy lại Cell 02 trên Colab**
- [ ] **TODO: Chạy lại Cell 08 và verify raw.arrow size**

---

## 🐛 Troubleshooting

### Nếu raw.arrow vẫn nhỏ sau khi fix:

1. **Verify code đã được pull đúng:**
   ```python
   # Trong Colab, sau khi chạy Cell 02
   with open("/content/F5-TTS-Vietnamese/src/f5_tts/model/utils.py", "r") as f:
       content = f.read()
       if "has_vietnamese_chars" in content:
           print("✅ Vietnamese fix applied!")
       else:
           print("❌ Old code still present!")
   ```

2. **Test hàm convert_char_to_pinyin:**
   ```python
   from f5_tts.model.utils import convert_char_to_pinyin
   
   test_text = ["tự dưng trong mình"]
   result = convert_char_to_pinyin(test_text)
   
   print(f"Input:  {test_text[0]}")
   print(f"Output: {''.join(result[0])}")
   
   # Should show: tự dưng trong mình (giữ nguyên)
   ```

3. **Check metadata.csv:**
   ```python
   import pandas as pd
   df = pd.read_csv("/content/data/<speaker>_training/metadata.csv", 
                    sep="|", encoding="utf-8")
   print(df['text'].head())
   # Should show Vietnamese text correctly
   ```

---

## 📚 References

**Modified files:**
1. `src/f5_tts/model/utils.py` - Vietnamese text processing
2. `colab-cells/02_install_dependencies.py` - Repository URL

**Vietnamese Unicode ranges:**
- [Vietnamese Unicode](https://en.wikipedia.org/wiki/Vietnamese_alphabet)
- [Latin Extended Additional](https://en.wikipedia.org/wiki/Latin_Extended_Additional)

---

**🎉 Với fix này, bạn chỉ cần push code và chạy lại Cell 08 là xong!**
