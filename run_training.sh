#!/bin/bash

# Script to run train.py with real-time output streaming to debug_output.txt
# This allows running from anywhere without needing ccv-vscode-node environment

# Set the working directory to the script's location
cd "$(dirname "$0")"

# Clear any existing debug output
> debug_output.txt

echo "Starting training run at $(date)" | tee -a debug_output.txt
echo "=========================================" | tee -a debug_output.txt

# Run train.py and stream output to both terminal and debug_output.txt
# Using unbuffer to ensure real-time streaming (if available)
if command -v unbuffer >/dev/null 2>&1; then
    echo "Using unbuffer for real-time output streaming..." | tee -a debug_output.txt
    unbuffer python train.py 2>&1 | tee -a debug_output.txt
else
    echo "Using standard tee for output streaming..." | tee -a debug_output.txt
    python train.py 2>&1 | tee -a debug_output.txt
fi

# Capture the exit code
EXIT_CODE=${PIPESTATUS[0]}

echo "=========================================" | tee -a debug_output.txt
echo "Training run completed at $(date)" | tee -a debug_output.txt
echo "Exit code: $EXIT_CODE" | tee -a debug_output.txt

# Exit with the same code as the Python script
exit $EXIT_CODE
