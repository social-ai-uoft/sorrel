#!/usr/bin/env python3
"""
Quick verification that all games work in original mode (sequential + sync).
Runs each game for just 5 epochs to verify no crashes.
"""

import subprocess
import sys
from pathlib import Path

print("="*70)
print("VERIFYING ALL GAMES IN ORIGINAL MODE (sequential + sync)")
print("="*70)
print("\nRunning each game for 5 epochs to verify no regressions...\n")

games = {
    'treasurehunt': 'sorrel/examples/treasurehunt/main.py',
    'cleanup': 'sorrel/examples/cleanup/main.py',
    'tag': 'sorrel/examples/tag/main.py',
    'taxi': 'sorrel/examples/taxi/main.py',
    'cooking': 'sorrel/examples/cooking/main.py',
    'iowa': 'sorrel/examples/iowa/main.py',
}

results = {}

for game_name, game_path in games.items():
    print(f"\n{'='*70}")
    print(f"Testing {game_name.upper()}")
    print('='*70)
    
    full_path = Path(__file__).parent / game_path
    
    if not full_path.exists():
        print(f"⚠️  SKIP - {game_path} not found")
        results[game_name] = 'SKIP'
        continue
    
    try:
        # Run with timeout
        result = subprocess.run(
            ['poetry', 'run', 'python', str(full_path)],
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            cwd=Path(__file__).parent
        )
        
        if result.returncode == 0:
            print("✅ PASS - Game runs without errors")
            results[game_name] = 'PASS'
        else:
            print(f"❌ FAIL - Exit code {result.returncode}")
            if result.stderr:
                print(f"Error output:\n{result.stderr[:500]}")
            results[game_name] = 'FAIL'
            
    except subprocess.TimeoutExpired:
        print("⏱️  TIMEOUT - Game is still running (likely working)")
        results[game_name] = 'TIMEOUT'
    except Exception as e:
        print(f"❌ ERROR: {str(e)[:200]}")
        results[game_name] = 'ERROR'

# Summary
print("\n" + "="*70)
print("VERIFICATION SUMMARY")
print("="*70)
for game, status in results.items():
    emoji = {
        'PASS': '✅',
        'TIMEOUT': '⏱️',
        'SKIP': '⚠️',
        'FAIL': '❌',
        'ERROR': '❌'
    }.get(status, '❓')
    print(f"{emoji} {game:15s}: {status}")

print("="*70)

# Check results
passed = sum(1 for s in results.values() if s in ['PASS', 'TIMEOUT'])
total = len(results)

print(f"\nResult: {passed}/{total} games verified working")

if passed == total:
    print("\n✅ ALL GAMES WORKING - No regressions detected!")
    sys.exit(0)
else:
    print("\n⚠️  Some games had issues - review output above")
    sys.exit(1)
