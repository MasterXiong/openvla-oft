#!/bin/bash
# HN-LoRA Evaluation Script for LIBERO
# Usage:
#   Default: bash run_hn_lora_eval.sh [task_suite] [gpu_id] [num_trials]
#   Example: bash run_hn_lora_eval.sh libero_spatial 0 50
#
#   task_suite: libero_spatial, libero_object, libero_goal, libero_10 - default: libero_spatial
#   gpu_id: GPU ID to use - default: 0
#   num_trials: Number of trials per task - default: 50

# Parse command line arguments
TASK_SUITE=${1:-libero_spatial}
GPU_ID=${2:-0}
NUM_TRIALS=${3:-50}

# Define checkpoint path - using your trained checkpoint
CHECKPOINT_PATH="/homes/80/kang/zheng_openvla_oft/openvla-oft/runs/20250917_222219+openvla-7b+libero_spatial_no_noops+b1+lr-0.0005+hn_lora-r32--image_aug--hn_lora_v9_optimized_gpu7/10_chkpt"

# Set environment variables
export PYTHONPATH=/homes/80/kang/zheng_openvla_oft/openvla-oft:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Change to the correct directory
cd /homes/80/kang/zheng_openvla_oft/openvla-oft

echo ""
echo "========================================="
echo "HN-LoRA Evaluation for LIBERO"
echo "========================================="
echo ""
echo "Evaluation Configuration:"
echo "  - Task Suite: ${TASK_SUITE}"
echo "  - GPU ID: ${GPU_ID}"
echo "  - Trials per task: ${NUM_TRIALS}"
echo "  - Checkpoint path: ${CHECKPOINT_PATH}"
echo ""

# Activate conda environment and check if LIBERO is installed
source /homes/80/kang/anaconda3/etc/profile.d/conda.sh
conda activate openvla-oft

echo "Using Python: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo ""

# Check if LIBERO is installed in the correct environment
/homes/80/kang/anaconda3/envs/openvla-oft/bin/python -c "import libero" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "LIBERO is not installed!"
    echo ""
    echo "To install LIBERO in the openvla-oft environment, please run:"
    echo "  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git"
    echo "  /homes/80/kang/anaconda3/envs/openvla-oft/bin/pip install -e LIBERO"
    echo "  /homes/80/kang/anaconda3/envs/openvla-oft/bin/pip install -r experiments/robot/libero/libero_requirements.txt"
    echo ""
    echo "Do you want to install LIBERO now? (y/n)"
    read -r response
    if [[ "$response" == "y" ]]; then
        echo "Installing LIBERO..."
        # Clone LIBERO if not exists
        if [ ! -d "LIBERO" ]; then
            git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
        fi
        /homes/80/kang/anaconda3/envs/openvla-oft/bin/pip install -e LIBERO
        /homes/80/kang/anaconda3/envs/openvla-oft/bin/pip install -r experiments/robot/libero/libero_requirements.txt
        echo " LIBERO installation completed!"
    else
        echo " Exiting. Please install LIBERO before running evaluation."
        exit 1
    fi
fi

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo " Error: Checkpoint directory not found at: $CHECKPOINT_PATH"
    echo ""
    echo "Available checkpoints:"
    ls -d ${CHECKPOINT_BASE}/*_chkpt 2>/dev/null | xargs -I {} basename {}
    exit 1
fi

# Setup checkpoint files if needed
echo "🔧 Checking checkpoint setup..."
BASE_MODEL="/homes/80/kang/.cache/huggingface/hub/models--openvla--openvla-7b/snapshots/31f090d05236101ebfc381b61c674dd4746d4ce0"

# Copy config.json if missing
if [ ! -f "${CHECKPOINT_PATH}/config.json" ]; then
    echo "  - Copying config.json..."
    cp "${BASE_MODEL}/config.json" "${CHECKPOINT_PATH}/"
fi

# Copy model index if missing
if [ ! -f "${CHECKPOINT_PATH}/model.safetensors.index.json" ]; then
    echo "  - Copying model.safetensors.index.json..."
    cp "${BASE_MODEL}/model.safetensors.index.json" "${CHECKPOINT_PATH}/"
fi

# Create symbolic links to model weights if missing
for i in 1 2 3; do
    file="model-0000${i}-of-00003.safetensors"
    if [ ! -e "${CHECKPOINT_PATH}/$file" ]; then
        echo "  - Creating symlink for $file..."
        ln -s "${BASE_MODEL}/$file" "${CHECKPOINT_PATH}/$file"
    fi
done

echo " Checkpoint setup complete"

# Check for action_head and proprio_projector files
echo "🔧 Checking for component files..."
if [ ! -f "${CHECKPOINT_PATH}/action_head--10_checkpoint.pt" ] && [ ! -f "${CHECKPOINT_PATH}/action_head--latest_checkpoint.pt" ]; then
    echo "Warning: action_head checkpoint not found"
fi

if [ ! -f "${CHECKPOINT_PATH}/proprio_projector--10_checkpoint.pt" ] && [ ! -f "${CHECKPOINT_PATH}/proprio_projector--latest_checkpoint.pt" ]; then
    echo "Warning: proprio_projector checkpoint not found"
fi

echo ""

# Check HN-LoRA checkpoint
if [ ! -d "${CHECKPOINT_PATH}/hn_lora_hypernet" ]; then
    echo "⚠️  Warning: HN-LoRA checkpoint directory not found"
    echo "  Missing: ${CHECKPOINT_PATH}/hn_lora_hypernet"
    echo "  This checkpoint may not be properly configured for HN-LoRA evaluation"
    echo ""
    echo "Continue anyway? (y/n)"
    read -r response
    if [[ "$response" != "y" ]]; then
        exit 1
    fi
else
    echo "✅ HN-LoRA checkpoint detected"
    echo "  Found: ${CHECKPOINT_PATH}/hn_lora_hypernet/"

    # Check for required HN-LoRA files
    if [ -f "${CHECKPOINT_PATH}/hn_lora_hypernet/hypernet_state.pt" ]; then
        echo "  ✓ hypernet_state.pt found"
    else
        echo "  ✗ hypernet_state.pt missing!"
    fi

    if [ -f "${CHECKPOINT_PATH}/hn_lora_hypernet/layer_dims.json" ]; then
        echo "  ✓ layer_dims.json found"
    else
        echo "  ✗ layer_dims.json missing!"
    fi

    if [ -f "${CHECKPOINT_PATH}/hn_lora_hypernet/hn_lora_config.json" ]; then
        echo "  ✓ hn_lora_config.json found"
    else
        echo "  ✗ hn_lora_config.json missing!"
    fi
fi

echo ""
echo "🚀 Starting HN-LoRA evaluation..."
echo "Command: python experiments/robot/libero/run_libero_eval_hn_lora.py \\"
echo "  --pretrained_checkpoint ${CHECKPOINT_PATH} \\"
echo "  --task_suite_name ${TASK_SUITE} \\"
echo "  --use_hn_lora True \\"
echo "  --center_crop True \\"
echo "  --lora_rank 32 \\"
echo "  --num_trials_per_task ${NUM_TRIALS} \\"
echo "  --num_images_in_input 2 \\"
echo "  --use_proprio True \\"
echo "  --use_l1_regression True \\"
echo "  --seed 7"
echo ""
echo "========================================="
echo ""


/homes/80/kang/anaconda3/envs/openvla-oft/bin/python experiments/robot/libero/run_libero_eval_hn_lora.py \
    --pretrained_checkpoint ${CHECKPOINT_PATH} \
    --task_suite_name ${TASK_SUITE} \
    --use_hn_lora True \
    --center_crop True \
    --lora_rank 32 \
    --num_trials_per_task ${NUM_TRIALS} \
    --num_images_in_input 2 \
    --use_proprio True \
    --use_l1_regression True \
    --seed 7

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "Evaluation completed successfully!"
else
    echo ""
    echo "Evaluation failed. Please check the error messages above."
    exit 1
fi