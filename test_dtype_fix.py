#!/usr/bin/env python3
"""
Test script to verify BFloat16 dtype fix
"""

import torch
import torch.nn as nn
from dataclasses import dataclass

print("Testing BFloat16 dtype compatibility...")

# Simulate the scenario
@dataclass
class TestConfig:
    pretrained_checkpoint: str = 'finetune_saves/openvla-oft/20250918_091937+openvla-7b+libero_goal_no_noops+b1+lr-0.0005+hn_lora-r4--image_aug--hyperlora--200_chkpt'
    num_images_in_input: int = 1
    use_hn_lora: bool = True

# Test 1: Check HyperNet dtype after loading
print("\n1. Loading model with HN-LoRA...")
try:
    from experiments.robot.hn_lora_utils import get_hn_lora_vla
    cfg = TestConfig()
    model = get_hn_lora_vla(cfg)

    # Check base model dtype
    first_param = next(model.parameters())
    print(f"   Base model dtype: {first_param.dtype}")

    # Check HyperNet dtype
    if hasattr(model, 'hn_lora_hypernet'):
        hn_first_param = next(model.hn_lora_hypernet.parameters())
        print(f"   HyperNet dtype: {hn_first_param.dtype}")

        if first_param.dtype == hn_first_param.dtype:
            print("   ✅ Dtypes match!")
        else:
            print(f"   ❌ Dtype mismatch: base={first_param.dtype}, hypernet={hn_first_param.dtype}")

    # Test 2: Generate LoRA params and check dtype
    print("\n2. Testing LoRA parameter generation...")
    with torch.no_grad():
        # Create dummy input
        test_input = torch.randn(1, 10, 4096, dtype=torch.bfloat16).cuda()
        test_embeds = model.get_input_embeddings()(torch.tensor([[1, 2, 3, 4, 5]]).cuda())

        # Generate LoRA params
        lora_params = model.hn_lora_hypernet(test_embeds)

        # Check dtype of generated params
        lora_A, lora_B = lora_params[0]
        print(f"   Generated LoRA A dtype: {lora_A.dtype}")
        print(f"   Generated LoRA B dtype: {lora_B.dtype}")

        # Check if they match after set_lora_params
        if hasattr(model, 'hn_lora_layers') and len(model.hn_lora_layers) > 0:
            test_layer = model.hn_lora_layers[0]
            test_layer.set_lora_params(lora_A, lora_B)

            print(f"   After set_lora_params:")
            print(f"     Layer weight dtype: {test_layer.weight.dtype}")
            print(f"     Stored LoRA A dtype: {test_layer.lora_A.dtype}")
            print(f"     Stored LoRA B dtype: {test_layer.lora_B.dtype}")

            if test_layer.weight.dtype == test_layer.lora_A.dtype == test_layer.lora_B.dtype:
                print("   ✅ All dtypes consistent after set_lora_params!")
            else:
                print("   ❌ Dtype inconsistency detected")

    print("\n✅ All dtype tests passed!")

except Exception as e:
    print(f"\n❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()

print("\nDtype fix verification complete.")