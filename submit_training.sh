#!/bin/bash

# Simple script to submit the training batch job
# Run this from anywhere - it will navigate to the correct directory

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Navigate to the project directory
cd "$SCRIPT_DIR"

echo "Submitting training job from: $(pwd)"
echo "Batch script: train_batch.slurm"

# Submit the batch job
sbatch train_batch.slurm

echo "Job submitted! Check debug_output.txt for real-time output."
echo "You can monitor job status with: squeue -u $USER"
