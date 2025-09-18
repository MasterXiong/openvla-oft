OPENVLA_DEBUG_INPUT_IDS=${OPENVLA_DEBUG_INPUT_IDS:-1} \
OPENVLA_DEBUG_MAX=${OPENVLA_DEBUG_MAX:-1000000} \
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint finetune_saves/openvla-oft/20250918_091937+openvla-7b+libero_goal_no_noops+b1+lr-0.0005+hn_lora-r4--image_aug--hyperlora--200_chkpt \
  --task_suite_name libero_goal \
  --num_images_in_input 1 \
  --use_proprio False \
  --use_hn_lora True \
  --num_open_loop_steps 1

# 20250918_091937+openvla-7b+libero_goal_no_noops+b1+lr-0.0005+hn_lora-r4--image_aug--hyperlora--200_chkpt


# python experiments/robot/libero/run_libero_eval.py --pretrained_checkpoint finetune_saves/openvla-oft/20250916_052411+openvla-7b+libero_goal_no_noops+b8+lr-0.0005+hn_lora-r4--image_aug--hyperlora/50000_chkpt --task_suite_name libero_goal --num_images_in_input 1 --use_proprio False --use_hn_lora True 
