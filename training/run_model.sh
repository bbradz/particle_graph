#!/bin/bash

#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 16
#SBATCH --mem=8G
#SBATCH -t 00:10:00

#SBATCH --qos=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user qian_niu@brown.edu
#BATCH --mail-type END, FAIL
#SBATCH --output=/users/qniu3/physics/RL_model_builder/training/run_model.%j.out

module load mathematica
python /users/qniu3/physics/RL_model_builder/training/run_model.py
