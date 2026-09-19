#!/bin/bash
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


SCRIPT_NAME="training/sft.py"
CONDA_ENV_NAME="base"  # Use the base environment

PROJECT_DIR="/home/pasa00007/Seminar/Length_Sensitive_DPO/" 
CONDA_PYTHON="/home/pasa00007/.conda/envs/agentic-eval/bin/python"
MODULE_NAME="training.sft"

# Hugging Face cache on scratch
# export HF_HOME="/scratch/compuling/pasa00007/HF_DATA"
# export HF_DATASETS_CACHE="$HF_HOME/datasets"
# export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

# Navigate to the project directory
cd "$PROJECT_DIR" || { echo "Failed to change directory to $PROJECT_DIR"; exit 1; }

echo "=========================================="
echo "Starting SFT Training"
echo "Script: $SCRIPT_NAME"
echo "Conda Environment: $CONDA_ENV_NAME"
echo "=========================================="

"$CONDA_PYTHON" -m "$MODULE_NAME" \
    --model_name "meta-llama/Llama-2-7b-chat-hf" \
    --dataset_name "allenai/social_i_qa,cfilt/PUB" \
    --output_dir "sft_model_output/llama-2-7b-chat" \
    --num_train_epochs 1 \
    --batch_size 64 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-7 \
    --optimizer "rmsprop" \
    --max_grad_norm 10.0 \
    --warmup_steps 150 \
    --max_steps -1 \
    --logging_steps 10 \
    --save_strategy "epoch" \
    --gradient_checkpointing \
    --dataloader_num_workers 4 \
    --use_wandb \
    --max_length 512 

echo "=========================================="
echo "SFT Training Completed"
echo "=========================================="