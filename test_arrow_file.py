"""
Test script to verify raw.arrow file can be read with Hugging Face Datasets
"""

import sys
from pathlib import Path

# Test file path
arrow_file = Path("D:/Project/F5-TTS/F5-TTS-Vietnamese/raw.arrow")

if not arrow_file.exists():
    print(f"[ERROR] File not found: {arrow_file}")
    sys.exit(1)

print(f"[*] Testing file: {arrow_file}")
print(f"   Size: {arrow_file.stat().st_size / 1024:.2f} KB")
print()

# Test 1: Try with Hugging Face Datasets (correct way)
print("="*70)
print("Test 1: Reading with Hugging Face Datasets API")
print("="*70)
try:
    from datasets import Dataset
    
    dataset = Dataset.from_file(str(arrow_file))
    print(f"[OK] Success!")
    print(f"   Rows: {len(dataset)}")
    print(f"   Columns: {dataset.column_names}")
    
    if len(dataset) > 0:
        print(f"\n[*] First row sample:")
        first_row = dataset[0]
        for key, value in first_row.items():
            if isinstance(value, str) and len(value) > 50:
                value = value[:50] + "..."
            print(f"   {key}: {value}")
    
except Exception as e:
    print(f"[FAIL] Failed: {type(e).__name__}")
    print(f"   {e}")

print()

# Test 2: Try with PyArrow IPC (wrong way - for comparison)
print("="*70)
print("Test 2: Reading with PyArrow IPC (wrong way)")
print("="*70)
try:
    import pyarrow as pa
    
    with pa.ipc.open_file(str(arrow_file)) as reader:
        table = reader.read_all()
    
    print(f"[OK] Success! (unexpected)")
    print(f"   Rows: {len(table)}")
    
except Exception as e:
    print(f"[FAIL] Failed (expected): {type(e).__name__}")
    print(f"   {e}")

print()
print("="*70)
print("[*] Conclusion:")
print("="*70)
print("If Test 1 succeeds: Cell 09 will work after the fix")
print("If Test 1 fails: Need to regenerate raw.arrow with Cell 08")
