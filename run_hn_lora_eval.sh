#!/bin/bash
# HN-LoRA Evaluation Script for LIBERO
# Usage:
#   Default: bash run_hn_lora_eval.sh [step] [task_suite] [gpu_id]
#   Example: bash run_hn_lora_eval.sh 30 libero_spatial 7
#
#   step: checkpoint step number (10, 20, 30, etc.) - default: 30
#   task_suite: libero_spatial, libero_object, libero_goal, libero_10 - default: libero_spatial
#   gpu_id: GPU ID to use - default: 7

# Parse command line arguments
CHECKPOINT_STEP=${1:-30}
TASK_SUITE=${2:-libero_spatial}
GPU_ID=${3:-7}
NUM_TRIALS=${4:-50}

# Define checkpoint base path
CHECKPOINT_BASE="runs/20250916_144455+openvla-7b+libero_spatial_no_noops+b1+lr-0.0005+hn_lora-r32--image_aug--hn_lora_v9_optimized_gpu6_7"
CHECKPOINT_PATH="${CHECKPOINT_BASE}/${CHECKPOINT_STEP}_chkpt"

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
echo "  - Checkpoint: Step ${CHECKPOINT_STEP}"
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
if [ ! -f "${CHECKPOINT_PATH}/action_head--${CHECKPOINT_STEP}_checkpoint.pt" ]; then
    echo "Warning: action_head checkpoint not found for step ${CHECKPOINT_STEP}"
    # Try to find the latest available checkpoint
    for step in 20 10; do
        if [ -f "${CHECKPOINT_BASE}/${step}_chkpt/action_head--${step}_checkpoint.pt" ]; then
            echo "  - Using action_head from step ${step} as fallback"
            cp "${CHECKPOINT_BASE}/${step}_chkpt/action_head--${step}_checkpoint.pt" \
               "${CHECKPOINT_PATH}/action_head--${CHECKPOINT_STEP}_checkpoint.pt"
            break
        fi
    done
fi

if [ ! -f "${CHECKPOINT_PATH}/proprio_projector--${CHECKPOINT_STEP}_checkpoint.pt" ]; then
    echo "Warning: proprio_projector checkpoint not found for step ${CHECKPOINT_STEP}"
    # Try to find the latest available checkpoint
    for step in 20 10; do
        if [ -f "${CHECKPOINT_BASE}/${step}_chkpt/proprio_projector--${step}_checkpoint.pt" ]; then
            echo "  - Using proprio_projector from step ${step} as fallback"
            cp "${CHECKPOINT_BASE}/${step}_chkpt/proprio_projector--${step}_checkpoint.pt" \
               "${CHECKPOINT_PATH}/proprio_projector--${CHECKPOINT_STEP}_checkpoint.pt"
            break
        fi
    done
fi

echo ""

# Check checkpoint type and validate
if [ "$LORA_TYPE" == "hn_lora" ]; then
    # Check if HN-LoRA checkpoint exists
    if [ ! -d "${CHECKPOINT_PATH}/hn_lora_hypernet" ]; then
        echo "Warning: This doesn't appear to be a HN-LoRA checkpoint"
        echo "  Missing: ${CHECKPOINT_PATH}/hn_lora_hypernet"
        echo ""
        echo "Continue anyway? (y/n)"
        read -r response
        if [[ "$response" != "y" ]]; then
            exit 1
        fi
    else
        echo "HN-LoRA checkpoint detected"
    fi
else
    # For standard LoRA, check for lora_adapter directory or adapter files
    if [ ! -d "${CHECKPOINT_PATH}/lora_adapter" ] && [ ! -f "${CHECKPOINT_PATH}/adapter_config.json" ]; then
        echo "Warning: This doesn't appear to be a standard LoRA checkpoint"
        echo "  Missing: ${CHECKPOINT_PATH}/lora_adapter or adapter_config.json"
        echo ""
        echo "Continue anyway? (y/n)"
        read -r response
        if [[ "$response" != "y" ]]; then
            exit 1
        fi
    else
        echo "Standard LoRA checkpoint detected"
    fi
fi

echo ""
echo "🚀 Starting evaluation..."
echo "Command: python experiments/robot/libero/run_libero_eval.py \\"
echo "  --pretrained_checkpoint ${CHECKPOINT_PATH} \\"
echo "  --task_suite_name ${TASK_SUITE} \\"
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


/homes/80/kang/anaconda3/envs/openvla-oft/bin/python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint ${CHECKPOINT_PATH} \
    --task_suite_name ${TASK_SUITE} \
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