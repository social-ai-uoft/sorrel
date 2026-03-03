#!/bin/bash
# Check status of CPC training with recent-first sampling

echo "=== CPC Training Status Check ==="
echo ""

# Check if process is running
if pgrep -f "train_cpc_500_epochs.*recent_first" > /dev/null; then
    echo "✓ Training process is RUNNING"
    echo ""
    ps aux | grep -E "train_cpc_500_epochs.*recent_first" | grep -v grep | awk '{print "  PID: "$2", CPU: "$3"%, Memory: "$4"%"}'
    echo ""
else
    echo "✗ Training process is NOT running"
    echo ""
fi

# Check output file
OUTPUT_FILE="./cpc_report/training_500_epochs_cpc_start_10_memory1000_sample64_recent_first.txt"
if [ -f "$OUTPUT_FILE" ]; then
    echo "✓ Output file exists: $OUTPUT_FILE"
    echo ""
    echo "Last 20 lines of output:"
    echo "---"
    tail -20 "$OUTPUT_FILE" 2>/dev/null || echo "File is empty or not readable"
    echo "---"
    echo ""
    echo "File size: $(ls -lh "$OUTPUT_FILE" | awk '{print $5}')"
    echo "Last modified: $(ls -l "$OUTPUT_FILE" | awk '{print $6, $7, $8}')"
else
    echo "✗ Output file not found: $OUTPUT_FILE"
    echo "  (Training may still be in early stages)"
fi

echo ""
echo "=== End Status Check ==="

