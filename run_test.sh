#!/bin/bash
#SBATCH --job-name=mistral-zero
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6

echo "Job started at $(date)"
nvidia-smi || true

# Caches off $HOME
export HF_HOME=/scratch/s5473535/hf
export TRANSFORMERS_CACHE=/scratch/s5473535/hf/transformers
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

# Required for gated models (Mistral Instruct)
export HUGGING_FACE_HUB_TOKEN=$(cat /scratch/s5473535/.hf_token)

module load Python/3.10.8-GCCcore-12.2.0
source /scratch/s5473535/xed-llm/.venv/bin/activate

python -u /scratch/s5473535/xed-llm/peft_try.py
echo "Job finished at $(date)"
