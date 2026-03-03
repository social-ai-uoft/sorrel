#!/bin/bash
# Script to check if training is still running

echo "=== Training Status Check ==="
echo ""

# Check if process is running
PROCESS_COUNT=$(ps aux | grep "train_cpc_500_epochs" | grep -v grep | wc -l | tr -d ' ')
if [ "$PROCESS_COUNT" -gt 0 ]; then
    echo "✓ Training is RUNNING ($PROCESS_COUNT process(es))"
    ps aux | grep "train_cpc_500_epochs" | grep -v grep | awk '{print "  PID:", $2, "CPU:", $3"%", "MEM:", $4"%"}'
else
    echo "✗ Training is NOT running"
fi

echo ""

# Check output log
if [ -f "training_output.log" ]; then
    echo "=== Latest Output (last 10 lines) ==="
    tail -10 training_output.log
    echo ""
    echo "Log file size: $(ls -lh training_output.log | awk '{print $5}')"
    echo "Last modified: $(stat -f "%Sm" training_output.log 2>/dev/null || stat -c "%y" training_output.log 2>/dev/null)"
else
    echo "No training_output.log found"
fi

echo ""

# Check results file
if [ -f "cpc_report/training_500_epochs_cpc_start_10.txt" ]; then
    echo "=== Results File Status ==="
    echo "File exists: ✓"
    echo "File size: $(ls -lh cpc_report/training_500_epochs_cpc_start_10.txt | awk '{print $5}')"
    echo "Last modified: $(stat -f "%Sm" cpc_report/training_500_epochs_cpc_start_10.txt 2>/dev/null || stat -c "%y" cpc_report/training_500_epochs_cpc_start_10.txt 2>/dev/null)"
    echo ""
    echo "=== Latest Epochs (last 5 lines) ==="
    tail -5 cpc_report/training_500_epochs_cpc_start_10.txt | grep -E "^[0-9]" | tail -5
else
    echo "Results file not created yet (training may be in early stages)"
fi

echo ""

# Check for plot file
if [ -f "cpc_report/loss_plots.png" ]; then
    echo "✓ Plot file exists: cpc_report/loss_plots.png"
    echo "  This means training has completed!"
else
    echo "Plot file not found yet (training still in progress)"
fi

echo ""
echo "=== Quick Status ==="
if [ "$PROCESS_COUNT" -gt 0 ]; then
    echo "Status: RUNNING"
    echo "To monitor: tail -f training_output.log"
    echo "To stop: pkill -f train_cpc_500_epochs"
else
    if [ -f "cpc_report/loss_plots.png" ]; then
        echo "Status: COMPLETED ✓"
    else
        echo "Status: NOT RUNNING (may have finished or crashed)"
    fi
fi

