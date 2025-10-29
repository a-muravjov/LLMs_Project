#!/bin/bash
#SBATCH --job-name=peft-train
#SBATCH --partition=gpublong
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/s5473535/logs/peft-train-%j.out
#SBATCH --error=/scratch/s5473535/logs/peft-train-%j.err

echo "Job started at $(date)"
nvidia-smi || true

# --- Hugging Face cache setup ---
export HF_HOME=/scratch/s5473535/hf
mkdir -p "$HF_HOME"

# If you want to avoid the deprecation warning, remove TRANSFORMERS_CACHE entirely
# export TRANSFORMERS_CACHE=/scratch/s5473535/hf/transformers

# --- Auth for gated models (Mistral Instruct) ---
export HUGGING_FACE_HUB_TOKEN=$(tr -d '\n' < /scratch/s5473535/.hf_token)

# --- Environment setup ---
module load Python/3.10.8-GCCcore-12.2.0
source /scratch/s5473535/xed-llm/.venv/bin/activate

# --- Run training script ---
python -u /scratch/s5473535/xed-llm/LLMs_Project/PEFT_config/main_run_peft.py

echo "Job finished at $(date)"