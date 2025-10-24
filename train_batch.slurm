#!/bin/bash

#SBATCH -J particle_graph_training
#SBATCH --account=lgouskos-gcondo
#SBATCH --partition=lgouskos-h100-gcondo
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -t 12:00:00
#SBATCH --mem=80g
#SBATCH -o debug_output.txt
#SBATCH -e debug_output.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bpbradle@brown.edu

# Set the working directory to the project directory
cd /oscar/scratch/bpbradle/particle_graph

echo "Starting training run at $(date)"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: H100 (1 GPU)"
echo "Memory: 80GB"
echo "Queue: lgouskos-h100-gcondo"
echo "Account: lgouskos-gcondo"
echo "Current directory: $(pwd)"
echo "Files in directory: $(ls -la)"
echo "========================================="

# Load any necessary modules (uncomment if needed)
# module load python/3.9
# module load cuda/11.8

# Run train.py with unbuffered output
python -u train.py

# Capture the exit code
EXIT_CODE=$?

echo "========================================="
echo "Training run completed at $(date)"
echo "Exit code: $EXIT_CODE"

# Exit with the same code as the Python script
exit $EXIT_CODE
