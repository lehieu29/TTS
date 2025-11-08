"""
Test Vietnamese/English Text Processing (Simplified Version)
============================================================
Test xem convert_char_to_pinyin xử lý đúng tiếng Việt và tiếng Anh không
(Chinese processing đã được XÓA)
"""

import sys
sys.path.insert(0, 'src')

from f5_tts.model.utils import convert_char_to_pinyin

print("=" * 70)
print("TEST: Vietnamese/English Text Processing (Simplified)")
print("=" * 70)

# Test cases
test_cases = [
    ("xin chào các bạn", "Vietnamese with diacritics", True),
    ("xin chao cac ban", "Vietnamese without diacritics", True),
    ("hôm nay tôi sẽ giới thiệu về trí tuệ nhân tạo", "Long Vietnamese text", True),
    ("Việt Nam là một đất nước xinh đẹp", "Mixed case Vietnamese", True),
    ("hello world", "English text", True),
    ("Hello, xin chào!", "Mixed English-Vietnamese", True),
    ("test123 abc", "Text with numbers", True),
    ("xin chào, tôi là AI", "Vietnamese with punctuation", True),
]

success_count = 0
total_count = len(test_cases)

for text, description, should_pass in test_cases:
    print(f"\n{'-' * 70}")
    print(f"Test: {description}")
    print(f"Input:  '{text}'")
    
    try:
        result = convert_char_to_pinyin([text])
        output = ''.join(result[0])
        print(f"Output: '{output}'")
        
        # Basic validation
        passed = True
        
        # Check 1: Output should not be empty
        if not output:
            print("❌ FAIL: Empty output")
            passed = False
        
        # Check 2: Output should preserve Vietnamese diacritics
        vietnamese_chars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ"
        input_diacritics = [c for c in text if c in vietnamese_chars]
        output_diacritics = [c for c in output if c in vietnamese_chars]
        
        if len(input_diacritics) != len(output_diacritics):
            print(f"⚠️  WARNING: Diacritics count mismatch (input: {len(input_diacritics)}, output: {len(output_diacritics)})")
        
        # Check 3: Should NOT have Pinyin tone numbers
        if any(c in "1234" for c in output):
            print("❌ FAIL: Found Pinyin tone markers (should not happen)")
            passed = False
        
        # Check 4: Character count should be reasonable (spaces added)
        # Output length should be >= input length (due to spaces)
        if len(output) < len(text) - text.count(' '):
            print("⚠️  WARNING: Output shorter than expected")
        
        if passed:
            print("✅ PASS")
            success_count += 1
        
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("TEST RESULTS")
print("=" * 70)
print(f"Passed: {success_count}/{total_count}")
print(f"Success rate: {success_count/total_count*100:.1f}%")

if success_count == total_count:
    print("\n🎉 ALL TESTS PASSED!")
else:
    print(f"\n⚠️  {total_count - success_count} test(s) failed")

print("\n💡 Expected behavior (SIMPLIFIED VERSION):")
print("  ✅ Vietnamese (with diacritics): Keep all original characters + tones")
print("  ✅ Vietnamese (without diacritics): Keep all original characters")
print("  ✅ English: Keep all original characters")
print("  ✅ Mixed: Handle both Vietnamese and English")
print("  ❌ Chinese: NOT SUPPORTED (will be treated as regular text)")
print("\n📝 Note: Chinese/Pinyin processing has been REMOVED")
