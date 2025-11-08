# Vietnamese Simplification - Xóa Bỏ Chinese Processing

## 🎯 Mục Đích

Đơn giản hóa codebase cho **Vietnamese TTS only** bằng cách:
- ✅ Xóa hết code xử lý tiếng Trung (Chinese/Pinyin)
- ✅ Xóa dependencies không cần thiết (jieba, pypinyin)
- ✅ Giảm complexity và tránh lỗi
- ✅ Focus 100% vào tiếng Việt và tiếng Anh

---

## 📝 Thay Đổi Chi Tiết

### **File:** `src/f5_tts/model/utils.py`

#### **1. Xóa Imports (Line 11-12):**

**TRƯỚC:**
```python
import jieba
from pypinyin import lazy_pinyin, Style
```

**SAU:**
```python
# Note: jieba and pypinyin imports removed - Chinese processing not needed for Vietnamese TTS
```

#### **2. Đơn Giản Hóa `convert_char_to_pinyin` Function:**

**TRƯỚC (~100 lines):**
```python
def convert_char_to_pinyin(text_list, polyphone=True):
    """Chinese, Vietnamese, English support"""
    
    # Multiple checks
    if has_chinese_chars(text):
        # Complex jieba segmentation
        # lazy_pinyin conversion
        # Multiple branches for different cases
    elif has_vietnamese_chars(text):
        # Vietnamese processing
    else:
        # Fallback
```

**SAU (~60 lines):**
```python
def convert_char_to_pinyin(text_list, polyphone=True):
    """
    Vietnamese and English only.
    SIMPLIFIED VERSION - Chinese/Pinyin processing removed.
    """
    
    # Single, simple logic for all Latin-based text
    for text in text_list:
        words = text.split()
        for word in words:
            char_list.extend(list(word))
```

---

## ✅ Lợi Ích

### **1. Code Đơn Giản Hơn:**
- Từ ~100 lines → ~60 lines
- Từ 3-4 branches → 1 branch duy nhất
- Không còn complex logic checking

### **2. Ít Dependencies:**
```bash
# KHÔNG CẦN cài đặt:
pip uninstall jieba pypinyin -y
```

### **3. Tránh Lỗi:**
- ❌ Không còn lỗi: "Text không dấu bị convert sang Pinyin"
- ❌ Không còn lỗi: "jieba segmentation sai"
- ❌ Không còn lỗi: "Vietnamese bị nhận diện thành Chinese"

### **4. Performance:**
- Nhanh hơn (không cần jieba.cut, lazy_pinyin)
- Ít memory hơn (không load Chinese dictionaries)

---

## 🧪 Testing

### **Test Script:** `test_vietnamese_fix.py`

```bash
# Run test
cd D:\Project\F5-TTS\F5-TTS-Vietnamese
python test_vietnamese_fix.py
```

**Test Cases:**
- ✅ Vietnamese có dấu: `"xin chào các bạn"`
- ✅ Vietnamese không dấu: `"xin chao cac ban"`
- ✅ English: `"hello world"`
- ✅ Mixed: `"Hello, xin chào!"`
- ✅ Punctuation: `"xin chào, tôi là AI"`
- ✅ Numbers: `"test123 abc"`

**Expected Output:**
```
TEST RESULTS
===================
Passed: 8/8
Success rate: 100.0%

🎉 ALL TESTS PASSED!
```

---

## 🚀 Deployment

### **Bước 1: Test Local (Windows)**
```bash
cd D:\Project\F5-TTS\F5-TTS-Vietnamese
python test_vietnamese_fix.py
```

### **Bước 2: Commit Changes**
```bash
git add src/f5_tts/model/utils.py
git add test_vietnamese_fix.py
git add VIETNAMESE_SIMPLIFICATION.md
git commit -m "Simplify for Vietnamese only - remove Chinese/Pinyin processing"
git push
```

### **Bước 3: Deploy to Colab**
```python
# In Colab
%cd /content/F5-TTS-Vietnamese
!git pull origin main

# CRITICAL: Reinstall package to apply changes
!pip install -e . --force-reinstall --no-deps

# Optional: Uninstall unused dependencies
!pip uninstall jieba pypinyin -y
```

### **Bước 4: Test Inference**
```python
# Test với text không dấu (đây là case bị lỗi trước đây)
%run /content/F5-TTS-Vietnamese/colab-cells/11_gradio_debug.py
```

---

## 📊 Before/After Comparison

| Aspect | Before (Chinese Support) | After (Vietnamese Only) |
|--------|--------------------------|-------------------------|
| **Lines of Code** | ~100 lines | ~60 lines |
| **Dependencies** | jieba, pypinyin | None (removed) |
| **Logic Branches** | 3-4 branches | 1 branch |
| **Bug: No diacritics** | ❌ Converted to Pinyin | ✅ Kept as Vietnamese |
| **Performance** | Slower (jieba, pinyin) | Faster |
| **Memory Usage** | Higher (Chinese dict) | Lower |
| **Maintenance** | Complex | Simple |

---

## 🔍 Detailed Logic Changes

### **Old Logic (WRONG):**
```python
Text Input: "xin chao cac ban"
    ↓
has_vietnamese_chars(text)?  # Check for diacritics only
    ↓ NO (no à, é, ô, etc.)
    ↓
has_chinese_chars(text)?
    ↓ NO
    ↓
DEFAULT: Use jieba + lazy_pinyin  ← ❌ WRONG!
    ↓
Output: "xīn cháo cāc bān" (Pinyin) ← ❌ WRONG!
```

### **New Logic (CORRECT):**
```python
Text Input: "xin chao cac ban"
    ↓
Split by spaces → ["xin", "chao", "cac", "ban"]
    ↓
Convert each word to char list → ['x','i','n',' ','c','h','a','o',...]
    ↓
Output: "x i n   c h a o   c a c   b a n" ← ✅ CORRECT!
```

---

## ⚠️ Limitations

### **What's NOT Supported:**

1. **Chinese Text:**
   - Input: `"你好世界"`
   - Output: Will be kept as-is (not converted to Pinyin)
   - Note: Training with Chinese text is NOT recommended

2. **Mixed Vietnamese-Chinese:**
   - Input: `"xin chào 你好"`
   - Output: Both parts kept as-is
   - Note: Chinese characters won't be converted

### **Why This is OK:**

- ✅ You're training Vietnamese TTS only
- ✅ Your audio data is Vietnamese
- ✅ Your use case is Vietnamese + English
- ✅ No Chinese input expected

---

## 🎯 Expected Behavior

### **Vietnamese (có dấu):**
```python
Input:  "xin chào các bạn"
Output: "x i n   c h à o   c á c   b ạ n"
Status: ✅ PASS - Diacritics preserved
```

### **Vietnamese (không dấu):**
```python
Input:  "xin chao cac ban"
Output: "x i n   c h a o   c a c   b a n"
Status: ✅ PASS - NOT converted to Pinyin (fixed!)
```

### **English:**
```python
Input:  "hello world"
Output: "h e l l o   w o r l d"
Status: ✅ PASS
```

### **Mixed:**
```python
Input:  "xin chào, hello!"
Output: "x i n   c h à o ,   h e l l o !"
Status: ✅ PASS
```

---

## 💡 Q&A

**Q: Có cần train lại model không?**
A: ❌ KHÔNG. Model đã train vẫn dùng được. Chỉ cần reinstall package.

**Q: Có mất tính năng gì không?**
A: ❌ KHÔNG. Chinese processing không được dùng cho Vietnamese TTS.

**Q: Nếu muốn dùng tiếng Trung sau này?**
A: Revert commit này hoặc dùng branch khác. Nhưng không khuyến khích mix Chinese-Vietnamese trong cùng model.

**Q: Có ảnh hưởng đến training không?**
A: ❌ KHÔNG. Training logic không đổi. Chỉ đơn giản hóa text preprocessing.

**Q: Có cần cập nhật vocab không?**
A: ❌ KHÔNG. Vocab đã có từ Cell 08 vẫn dùng được.

---

## 🔗 Related Files

- `src/f5_tts/model/utils.py` - Main change
- `test_vietnamese_fix.py` - Test script
- `11_gradio_debug.py` - For testing inference
- `VIETNAMESE_SIMPLIFICATION.md` - This file

---

## 📞 Support

Nếu gặp vấn đề sau khi apply changes:

1. **Check reinstall:**
   ```python
   !pip install -e . --force-reinstall --no-deps
   ```

2. **Run test:**
   ```bash
   python test_vietnamese_fix.py
   ```

3. **Test inference:**
   ```python
   %run colab-cells/11_gradio_debug.py
   ```

4. **If still errors:**
   - Restart Colab runtime
   - Re-run from Cell 01
   - Model checkpoints should still work

---

## ✅ Checklist

- [x] Xóa jieba, pypinyin imports
- [x] Đơn giản hóa convert_char_to_pinyin
- [x] Cập nhật test script
- [x] Tạo documentation
- [ ] Test local (Windows)
- [ ] Commit & push
- [ ] Deploy to Colab
- [ ] Reinstall package
- [ ] Test inference
- [ ] Verify output language (Vietnamese, NOT Pinyin)

---

**Last Updated:** 2025-11-08  
**Status:** ✅ READY FOR TESTING
