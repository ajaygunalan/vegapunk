#!/usr/bin/env python3

import os
import sys
import time
import subprocess
import shutil
from pathlib import Path

# Add project to path
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

# Setup paths
INPUT_DIR = BASE_DIR / "input/markdown/test_samples"
OUTPUT_DIR = BASE_DIR / "output"

# Clear output directory
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir()

# Track results
results = []
total_start = time.time()

# Process each paper
for paper_dir in sorted(INPUT_DIR.iterdir()):
    if paper_dir.is_dir():
        main_file = paper_dir / "main.md"
        if main_file.exists():
            paper_name = paper_dir.name
            print(f"\nProcessing: {paper_name}")
            
            start_time = time.time()
            
            # Run paper2code.py
            try:
                # Set env vars for API keys
                env = os.environ.copy()
                env_file = BASE_DIR / ".env"
                if env_file.exists():
                    with open(env_file) as f:
                        for line in f:
                            if '=' in line:
                                key, value = line.strip().split('=', 1)
                                env[key] = value
                
                result = subprocess.run([
                    sys.executable, 
                    str(BASE_DIR / "src/paper2code.py"), 
                    str(main_file)
                ], capture_output=True, text=True, timeout=120, env=env)
                
                if result.returncode != 0:
                    print(f"Error: {result.stderr}")
                
                elapsed = time.time() - start_time
                results.append((paper_name, elapsed, "Success"))
                print(f"✓ Completed in {elapsed:.1f}s")
                
            except subprocess.TimeoutExpired:
                elapsed = time.time() - start_time
                results.append((paper_name, elapsed, "Timeout"))
                print(f"✗ Timeout after {elapsed:.1f}s")
                
            except Exception:
                elapsed = time.time() - start_time
                results.append((paper_name, elapsed, "Failed"))
                print(f"✗ Failed after {elapsed:.1f}s")

# Summary
total_time = time.time() - total_start
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

for i, (paper, elapsed, status) in enumerate(results, 1):
    mins = int(elapsed // 60)
    secs = elapsed % 60
    time_str = f"{mins}m {secs:.1f}s" if mins > 0 else f"{secs:.1f}s"
    print(f"{i}. {paper:<47} {time_str:>10}  {status}")

print("-"*60)
total_mins = int(total_time // 60)
total_secs = total_time % 60
print(f"Total time: {total_mins}m {total_secs:.1f}s ({total_time:.1f}s)")
print(f"Average time: {total_time/len(results):.1f}s per paper")