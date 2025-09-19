"""
HN-LoRA (HyperNetwork LoRA) implementation for OpenVLA - Version 8
Based on paper: "Efficient Domain Adaptation of Robotic Foundation Models via Hypernetwork-Generated LoRA"

Correct implementation: Groups layers by dimension and shares output heads within each group.
Layer indices are treated as tokens in the Transformer, not just embeddings.
"""

import json
import torch

from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from huggingface_hub import snapshot_download

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
    _load_dataset_stats,
)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from hyperlora.config import HNLoRAConfig
from hyperlora.add_hn_to_vla import apply_hn_lora_to_base_model


def load_hn_lora_checkpoint(vla, path: str, device: str = "cpu") -> None:
    """
    Loads a HN-LoRA checkpoint and applies it to the VLA model.

    Args:
        vla: The VLA model with HN-LoRA already initialized.
        path (str): Path to checkpoint directory.
        step (int): Gradient step number of saved checkpoint.
        device (str): String specifying how to remap storage locations (default = "cpu").

    Returns:
        None.
    """
    from pathlib import Path
    
    checkpoint_dir = Path(path)
    hn_lora_dir = checkpoint_dir / "hn_lora_adapter"
    
    # Load HyperNetwork state dict
    hypernet_state_path = hn_lora_dir / "hypernet_state.pt"
    if hypernet_state_path.exists():
        print(f"Loading HN-LoRA HyperNetwork from: {hypernet_state_path}")
        state_dict = torch.load(hypernet_state_path, weights_only=True, map_location=device)
        
        # Access the HyperNetwork from the VLA model
        if hasattr(vla, 'module'):
            vla.module.hn_lora_hypernet.load_state_dict(state_dict)
        else:
            vla.hn_lora_hypernet.load_state_dict(state_dict)
        
        print(f"Successfully loaded HN-LoRA HyperNetwork checkpoint")
    else:
        raise FileNotFoundError(f"Warning: HyperNetwork state file not found at {hypernet_state_path}")


def get_hn_lora_vla(cfg):

    pretrained_vla_path = "openvla/openvla-7b"

    if model_is_on_hf_hub(pretrained_vla_path):
        # Download model directly from Hugging Face Hub
        vla_download_path = snapshot_download(repo_id=pretrained_vla_path)
        # Overwrite VLA path
        pretrained_vla_path = vla_download_path
    else:
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Update config.json and sync model files
    update_auto_map(pretrained_vla_path)
    check_model_logic_mismatch(pretrained_vla_path)

    # Load processor and VLA
    processor = AutoProcessor.from_pretrained(pretrained_vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        pretrained_vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).cuda()

    # Set number of images in VLA input
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    with open(f"{cfg.pretrained_checkpoint}/hn_lora_adapter/hn_lora_config.json", "r") as f:
        hn_config = json.load(f)

    hn_config = HNLoRAConfig(
        lora_rank=hn_config["lora_rank"],
        lora_alpha=hn_config["lora_alpha"],
        lora_dropout=hn_config["lora_dropout"],
        context_embedding_dim=hn_config["context_embedding_dim"],
        context_encoder_type=hn_config["context_encoder_type"],
        context_encoder_layers=hn_config["context_encoder_layers"],
        context_encoder_heads=hn_config["context_encoder_heads"],
        mlp_hidden_dim=hn_config["mlp_hidden_dim"],
        embedding_dropout=hn_config["embedding_dropout"],
    )
    # Auto-detect model dimensions
    if hasattr(vla, 'config'):
        hn_config.hidden_size = getattr(vla.config, 'hidden_size', 
                                        getattr(vla.config, 'd_model', 768))
        hn_config.num_hidden_layers = getattr(vla.config, 'num_hidden_layers',
                                                getattr(vla.config, 'num_layers', 12))
    vla = apply_hn_lora_to_base_model(vla, hn_config, processor=processor)
    
    load_hn_lora_checkpoint(vla, cfg.pretrained_checkpoint)

    _load_dataset_stats(vla, cfg.pretrained_checkpoint)
    vla.eval()

    return vla
