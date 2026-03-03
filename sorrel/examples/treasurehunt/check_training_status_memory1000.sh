#!/bin/bash

LOG_FILE="training_output.log"
OUTPUT_FILE="./cpc_report/training_500_epochs_cpc_start_10_memory1000_sample64.txt"

# Check if the training script is running
PID=$(pgrep -f "train_cpc_500_epochs.py.*memory1000_sample64")

if [ -z "$PID" ]; then
    echo "Training script is NOT running."
    # Check if the output file exists and has content, implying it finished
    if [ -f "$OUTPUT_FILE" ] && [ -s "$OUTPUT_FILE" ]; then
        echo "Training appears to have completed. Check '$OUTPUT_FILE' for full output."
        echo "=== Last 10 lines of output ==="
        tail -n 10 "$OUTPUT_FILE"
    else
        echo "No output file found or it's empty, implying training did not start or finished without output."
    fi
else
    echo "Training script is running with PID: $PID"
    echo "=== Latest Progress (last 10 lines of log) ==="
    # Display the last few lines of the log file
    if [ -f "$LOG_FILE" ]; then
        tail -n 10 "$LOG_FILE" | grep -v "UserWarning\|warnings.warn" || tail -n 10 "$LOG_FILE"
    else
        echo "Log file '$LOG_FILE' not found yet."
    fi
    
    # Check if output file exists and show progress
    if [ -f "$OUTPUT_FILE" ]; then
        echo ""
        echo "=== Training Progress (from output file) ==="
        tail -n 5 "$OUTPUT_FILE" | grep -E "Epoch|Total Loss|CPC Loss" || tail -n 5 "$OUTPUT_FILE"
    fi
fi

