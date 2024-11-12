#!/bin/bash
#SBATCH -A t24_ngpfc_g
#SBATCH --job-name=artemis_ci_chicoma_gpu_job
#SBATCH --output=ci_gpu_%j.out
#SBATCH --error=ci_gpu_%j.err
#SBATCH --time=00:10:00
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --qos=standard
#SBATCH -C gpu40

# Node options
#SBATCH --tasks-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0

# Set artemis path

# Check github token
echo "Github token:"
echo $ARTEMIS_GITHUB_TOKEN

# Load environment
source env/bash

# Ensure GITHUB_TOKEN is set
#export GITHUB_TOKEN=<your_token_here>  # Or source it from a secure file

module list

# Run the CI Python script
#srun python3 ci_runner.py:w

