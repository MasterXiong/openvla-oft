#!/bin/bash
# HN-LoRA v9 training - Optimized with vmap + torch.compile
# 3-phase optimization for maximum performance
# Usage: bash run_hn_lora_v9.sh [num_gpus N]
#   Default: single GPU (GPU 7)
#   Multi-GPU: bash run_hn_lora_v9.sh num_gpus 8

# Parse command line arguments
NUM_GPUS=1
CUDA_DEVICES="0"
BATCH_SIZE=1
GRAD_ACCUM=1 # only for testing to be DELETED !!!
# GRAD_ACCUM=64
RUN_NOTE="hn_lora_v9_optimized_gpu7"

if [[ "$1" == "num_gpus" ]] && [[ -n "$2" ]]; then
    if [[ "$2" == "8" ]]; then
        NUM_GPUS=8
        CUDA_DEVICES="0,1,2,3,4,5,6,7"
        BATCH_SIZE=1  # Each GPU processes 8 samples
        GRAD_ACCUM=1   
        # GRAD_ACCUM=8   
        RUN_NOTE="hn_lora_v9_optimized_8gpu"
        echo " Multi-GPU mode: Using 8 GPUs (batch_size=8 per GPU)"
    else
        echo " Error: Only 'num_gpus 8' is supported for multi-GPU training"
        echo "Usage: $0 [num_gpus 8]"
        exit 1
    fi
else
    echo " Single-GPU mode: Using GPU 7 (batch_size=1, grad_accum=8)"
fi

export PYTHONPATH=/homes/80/kang/openvla-oft:$PYTHONPATH
cd /homes/80/kang/openvla-oft

echo ""
echo "Starting HN-LoRA v9 Optimized training..."
echo "========================================="
echo "🖥️  GPU Configuration:"
echo "  - Number of GPUs: $NUM_GPUS"
echo "  - CUDA devices: $CUDA_DEVICES"
echo "  - Batch size per GPU: $BATCH_SIZE"
echo "  - Gradient accumulation: $GRAD_ACCUM"
echo "  - Effective batch size: $((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM))"
echo ""
echo "🚀 Optimizations Applied:"
echo "  Phase 1: Output heads parallelization with vmap"
echo "  Phase 2: Batch processing with einsum"
echo "  Phase 3: torch.compile for kernel fusion"
echo ""
echo "Expected Performance:"
echo "  - v8: ~8-9 seconds/iteration"
echo "  - v9: ~1.5-2 seconds/iteration (4-6x speedup)"
if [[ "$NUM_GPUS" == "8" ]]; then
    echo "  - Multi-GPU: Additional speedup from data parallelism"
fi
echo ""
echo "HyperNetwork Configuration:"
echo "- Context embedding dim: 128 (matches paper)"
echo "- Expected HyperNetwork size: ~32M parameters"
echo "- Optimization mode: reduce-overhead"
echo ""

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES /homes/80/kang/anaconda3/envs/openvla-oft/bin/torchrun --standalone --nnodes 1 --nproc-per-node $NUM_GPUS vla-scripts/finetune.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir /homes/80/kang/modified_libero_rlds_new \
    --dataset_name libero_spatial_no_noops \
    --run_root_dir runs \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --batch_size $BATCH_SIZE \
    --grad_accumulation_steps $GRAD_ACCUM \
    --learning_rate 5e-4 \
    --num_steps_before_decay 100000 \
    --max_steps 150005 \
    --save_freq 100 \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --use_lora True \
    --use_hn_lora True \
    --lora_rank 32 \
    --lora_dropout 0.0 \
    --hn_context_dim 128 \
    --hn_encoder_type transformer \
    --hn_encoder_layers 1 \
    --hn_encoder_heads 2 \
    --hn_mlp_dim 128 \
    --hn_embedding_dropout 0.1 \
    --wandb_entity "kang-oxford" \
    --wandb_project "openvla-hn-lora-v8" \
    --run_id_note $RUN_NOTE