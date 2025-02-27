#!/bin/bash
#SBATCH --job-name=LLM_testing     # Job name
#SBATCH --output=/projects/bdlo/download_output.log  # Output log file
#SBATCH --error=/projects/bdlo/download_error.log    # Error log file
#SBATCH --time=1:00:00                    # Max runtime (hh:mm:ss)
#SBATCH --ntasks=1                         # Number of tasks (single wget process)
#SBATCH --cpus-per-task=1                  # Single-core usage (wget is not CPU-intensive)
#SBATCH --mem=1G                           # Memory allocation (1GB should be sufficient)
#SBATCH --partition=gpuA100x4                 # Partition to submit to (adjust based on your system)
#SBATCH --account=bdlo-delta-gpu           # Specify the correct account

# Load necessary modules
module load cuda/11.8 # Load CUDA

# Test LLM
python knowledge_distillation.py