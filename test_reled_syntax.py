#!/usr/bin/env python3
"""
Syntax and structure test for RelEdit implementation
Checks code can be parsed without running it
"""

import ast
import json
from pathlib import Path

def check_python_syntax(file_path):
    """Check if a Python file has valid syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)

# Test all RelEdit Python files
print("Testing RelEdit Python files syntax...")
print("="*50)

files_to_test = [
    "RelEdit/__init__.py",
    "RelEdit/RelEdit_hparams.py",
    "RelEdit/kg_sampler.py",
    "RelEdit/compute_projection.py",
    "RelEdit/RelEdit_main.py",
]

all_passed = True
for file_path in files_to_test:
    path = Path(file_path)
    if not path.exists():
        print(f"✗ {file_path}: File not found")
        all_passed = False
        continue

    valid, error = check_python_syntax(file_path)
    if valid:
        print(f"[OK] {file_path}: Valid syntax")
    else:
        print(f"[FAIL] {file_path}: Syntax error")
        print(f"  {error}")
        all_passed = False

# Test JSON configuration files
print("\n" + "="*50)
print("Testing RelEdit configuration files...")
print("="*50)

config_files = [
    "hparams/RelEdit/Llama3-8B.json",
    "hparams/RelEdit/gpt2-xl.json",
    "hparams/RelEdit/EleutherAI_gpt-j-6B.json",
    "hparams/RelEdit/phi-1.5.json",
]

for config_file in config_files:
    path = Path(config_file)
    if not path.exists():
        print(f"[FAIL] {config_file}: File not found")
        all_passed = False
        continue

    try:
        with open(config_file, 'r') as f:
            config = json.load(f)

        # Check required RelEdit parameters
        required_params = ["num_paths", "max_path_length", "alpha", "use_kg_sampling", "kg_cache_dir"]
        missing = [p for p in required_params if p not in config]

        if missing:
            print(f"[FAIL] {config_file}: Missing parameters: {missing}")
            all_passed = False
        else:
            print(f"[OK] {config_file}: Valid (alpha={config['alpha']}, num_paths={config['num_paths']})")
    except json.JSONDecodeError as e:
        print(f"[FAIL] {config_file}: JSON error - {e}")
        all_passed = False

# Check script files exist
print("\n" + "="*50)
print("Testing experiment scripts...")
print("="*50)

script_files = [
    "run_reled_demo.sh",
    "run_reled_full.sh",
    "run_reled_ablation.sh",
    "run_reled_comparison.sh",
]

for script_file in script_files:
    path = Path(script_file)
    if path.exists():
        print(f"[OK] {script_file}: Exists")
    else:
        print(f"[FAIL] {script_file}: Not found")
        all_passed = False

# Check evaluation script integration
print("\n" + "="*50)
print("Checking evaluation script integration...")
print("="*50)

eval_script = Path("experiments/evaluate.py")
if eval_script.exists():
    with open(eval_script, 'r', encoding='utf-8') as f:
        content = f.read()

    checks = [
        ("RelEdit import", "from RelEdit import RelEditHyperParams" in content),
        ("RelEdit function import", "from RelEdit.RelEdit_main import apply_RelEdit_to_model" in content),
        ("RelEdit in ALG_DICT", '"RelEdit"' in content and 'RelEditHyperParams' in content),
    ]

    for check_name, passed in checks:
        if passed:
            print(f"[OK] {check_name}")
        else:
            print(f"[FAIL] {check_name}")
            all_passed = False
else:
    print(f"[FAIL] experiments/evaluate.py: Not found")
    all_passed = False

# Final summary
print("\n" + "="*50)
if all_passed:
    print("All structure tests passed!")
    print("="*50)
    print("\nRelEdit implementation structure is correct.")
    print("\nTo run functional tests (requires torch installation):")
    print("  python test_reled.py")
    print("\nTo run experiments:")
    print("  1. Demo: bash run_reled_demo.sh")
    print("  2. Full: bash run_reled_full.sh")
    print("  3. Comparison: bash run_reled_comparison.sh")
else:
    print("Some tests failed!")
    print("="*50)
    print("Please fix the issues above before running experiments.")

exit(0 if all_passed else 1)