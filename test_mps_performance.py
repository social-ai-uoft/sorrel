#!/usr/bin/env python3
"""Quick Treasure Hunt MPS vs CPU comparison."""

import time
import subprocess
from pathlib import Path

def run_treasurehunt(mode_name, epochs=100):
    print(f"\n{'='*60}")
    print(f"Running: {mode_name}")
    print(f"{'='*60}")
    
    cmd = [
        "poetry", "run", "python", 
        "sorrel/examples/treasurehunt/main.py",
        f"epochs={epochs}"
    ]
    
    start = time.time()
    result = subprocess.run(cmd, cwd="/Users/wil/git/sorrel/sorrel", capture_output=True, text=True)
    elapsed = time.time() - start
    
    print(f"{mode_name} completed in {elapsed:.2f}s")
    print(f"Exit code: {result.returncode}")
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    
    return elapsed

if __name__ == "__main__":
    # With MPS auto-detection enabled, this should use GPU
    mps_time = run_treasurehunt("Treasure Hunt with MPS (auto-detected)", epochs=100)
    
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"MPS Time: {mps_time:.2f}s")
    print(f"{'='*60}")
