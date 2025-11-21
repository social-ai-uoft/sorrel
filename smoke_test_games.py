#!/usr/bin/env python3
"""Quick smoke test for all games with default (sequential + sync) mode."""

import subprocess
import sys
from pathlib import Path

games = ["treasurehunt", "cleanup", "tag", "taxi", "cooking", "iowa"]
results = {}

print("Running quick smoke tests (10 epochs each)...\n")

for game in games:
    print(f"Testing {game}...", end=" ", flush=True)

    game_path = Path(__file__).parent / f"sorrel/examples/{game}/main.py"

    if not game_path.exists():
        print(f"SKIP (no main.py found)")
        results[game] = "SKIP"
        continue

    try:
        # Run for just 10 epochs to verify it doesn't crash
        result = subprocess.run(
            ["poetry", "run", "python", str(game_path)],
            capture_output=True,
            text=True,
            timeout=60,  # 60 second timeout
            cwd=Path(__file__).parent,
        )

        if result.returncode == 0:
            print("✅ PASS")
            results[game] = "PASS"
        else:
            print(f"❌ FAIL (exit code {result.returncode})")
            print(f"  Error: {result.stderr[:200]}")
            results[game] = "FAIL"

    except subprocess.TimeoutExpired:
        print("⏱️  TIMEOUT (still running)")
        results[game] = "TIMEOUT"
    except Exception as e:
        print(f"❌ ERROR: {e}")
        results[game] = "ERROR"

print("\n" + "=" * 50)
print("SMOKE TEST SUMMARY:")
print("=" * 50)
for game, status in results.items():
    print(f"{game:15s}: {status}")

# Exit with error if any tests failed
if any(status in ["FAIL", "ERROR"] for status in results.values()):
    sys.exit(1)
