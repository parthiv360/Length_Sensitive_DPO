#!/bin/bash

set -Eeuo pipefail

echo "HOME=$HOME"
export HF_DATASETS_CACHE="/scratch/compuling/pasa00007/HF_DATA/datasets"
export HUGGINGFACE_HUB_CACHE="/scratch/compuling/pasa00007/HF_DATA/hub"

echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"

if [ -f "$HOME/.cache/huggingface/token" ]; then
    echo "HF token visible inside job"
else
    echo "HF token NOT visible inside job"
fi


SCRIPT_NAME="evaluation/lnrs.py"
CONDA_ENV_NAME="base"  # Use the base environment

PROJECT_DIR="/home/pasa00007/Seminar/Length_Sensitive_DPO/" 
CONDA_PYTHON="/home/pasa00007/.conda/envs/agentic-eval/bin/python"
MODULE_NAME="evaluation.lnrs"

# Hugging Face cache on scratch
# export HF_HOME="/scratch/compuling/pasa00007/HF_DATA"
# export HF_DATASETS_CACHE="$HF_HOME/datasets"
# export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

# Navigate to the project directory
cd "$PROJECT_DIR" || { echo "Failed to change directory to $PROJECT_DIR"; exit 1; }

echo "=========================================="
echo "Starting MCQA Execution"
echo "Script: $SCRIPT_NAME"
echo "Conda Environment: $CONDA_ENV_NAME"
echo "=========================================="

# "$CONDA_PYTHON" -m "$MODULE_NAME" \
#     --model-name "ls_dpo_model_output/pythia-6.9b-tulu" \
#     --dataset-name "allenai/social_i_qa"

# "$CONDA_PYTHON" -m "$MODULE_NAME" \
#     --model-name "ls_dpo_model_output/llama-2-7b-chat" \
#     --dataset-name "allenai/social_i_qa"

# "$CONDA_PYTHON" -m "$MODULE_NAME" \
#     --model-name "ls_dpo_model_output/llama-2-13b-chat" \
#     --dataset-name "allenai/social_i_qa"

# "$CONDA_PYTHON" -m "$MODULE_NAME" \
#     --model-name "ls_dpo_model_output/pythia-6.9b-tulu" \
#     --dataset-name "lm-pragmatics"

# "$CONDA_PYTHON" -m "$MODULE_NAME" \
#     --model-name "ls_dpo_model_output/llama-2-7b-chat" \
#     --dataset-name "lm-pragmatics"

# "$CONDA_PYTHON" -m "$MODULE_NAME" \
#     --model-name "ls_dpo_model_output/llama-2-13b-chat" \
#     --dataset-name "lm-pragmatics"

# "$CONDA_PYTHON" -m "$MODULE_NAME" \
#     --model-name "ls_dpo_model_output/pythia-6.9b-tulu" \
#     --dataset-name "UCL-DARK/ludwig"

# "$CONDA_PYTHON" -m "$MODULE_NAME" \
#     --model-name "ls_dpo_model_output/llama-2-7b-chat" \
#     --dataset-name "UCL-DARK/ludwig"

"$CONDA_PYTHON" -m "$MODULE_NAME" \
    --model-name "sft_model_output/llama-2-13b-chat" \
    --dataset-name "lm-pragmatics"

# "$CONDA_PYTHON" -m "$MODULE_NAME" \
#     --model-name "dpo_model_output/llama-2-13b-chat" \
#     --dataset-name "lm-pragmatics"

# "$CONDA_PYTHON" -m "$MODULE_NAME" \
#     --model-name "sft_model_output/llama-2-13b-chat" \
#     --dataset-name "lm-pragmatics"


echo "=========================================="
echo "MCQA Execution Completed"
echo "=========================================="