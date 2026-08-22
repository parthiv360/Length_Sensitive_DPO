#!/bin/bash

SCRIPT_NAME="experiments/baseline.py"
CONDA_ENV_NAME="base"  # Use the base environment

PROJECT_DIR="/home/pasa00007/Seminar/Length_Sensitive_DPO/" 
CONDA_PYTHON="/home/pasa00007/.conda/envs/agentic-eval/bin/python"
MODULE_NAME="experiments.baseline"

# Hugging Face cache on scratch
export HF_HOME="/scratch/compuling/pasa00007/HF_DATA"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

# Navigate to the project directory
cd "$PROJECT_DIR" || { echo "Failed to change directory to $PROJECT_DIR"; exit 1; }

echo "=========================================="
echo "Starting Baseline Code Execution"
echo "Script: $SCRIPT_NAME"
echo "Conda Environment: $CONDA_ENV_NAME"
echo "=========================================="

"$CONDA_PYTHON" -m "$MODULE_NAME" \
    --model_name "allenai/open-instruct-pythia-6.9b-tulu" \
    --dataset_name "UCL-DARK/ludwig" \
    --max_length 512

echo "=========================================="
echo "Baseline Code Execution Completed"
echo "=========================================="