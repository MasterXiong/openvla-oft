import os
import torch
import torch.nn as nn

from .config import HNLoRAConfig
from .generated_lora_layer import HNLoRALinear
from .hypernet import HyperNetwork


def apply_hn_lora_to_base_model(base_model, config: HNLoRAConfig):
    """
    Apply HN-LoRA directly to the base model by:
    1. Replacing target Linear layers with HNLoRALinear
    2. Adding HyperNetwork as a module to the base model
    3. Modifying the forward method to use HN-LoRA
    4. Returning the modified base model itself
    """
    
    # Auto-detect dimensions
    if config.hidden_size is None:
        if hasattr(base_model, 'llm_dim'):
            config.hidden_size = base_model.llm_dim
        else:
            config.hidden_size = 4096  # OpenVLA/LLaMA default
        config.num_hidden_layers = 32  # LLaMA default
    
    # Freeze base model parameters
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Find and replace linear layers
    lora_layers = []
    layer_dims = []
    
    def replace_linear_layers(module, prefix=""):
        nonlocal lora_layers, layer_dims
        
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, nn.Linear):
                if "lm_head" in full_name:
                    continue
                # Create HNLoRALinear to replace the original
                hn_linear = HNLoRALinear(child, config.lora_rank, config.lora_alpha, config.lora_dropout)
                # Replace in parent module
                setattr(module, name, hn_linear)
                # Track layer
                lora_layers.append(hn_linear)
                shared_layer_id = full_name.split('.')
                shared_layer_id = '.'.join([x for x in shared_layer_id if not x.isdigit()])
                shared_layer_id = shared_layer_id.replace('.', '/')
                layer_dims.append((shared_layer_id, (child.in_features, child.out_features)))
            else:
                replace_linear_layers(child, full_name)
    
    replace_linear_layers(base_model)
    print(f"Total HN-LoRA layers: {len(lora_layers)}")
    
    # Create and attach HyperNetwork
    hypernet = HyperNetwork(config, layer_dims)
    
    # Move HyperNetwork to same device as base model
    if next(base_model.parameters(), None) is not None:
        device = next(base_model.parameters()).device
        hypernet = hypernet.to(device)
    
    base_model.hn_lora_hypernet = hypernet
    base_model.hn_lora_layers = lora_layers
    
    # Find embedding layer
    embedding_layer = None
    if hasattr(base_model, 'get_input_embeddings'):
        embedding_layer = base_model.get_input_embeddings()
    elif hasattr(base_model, 'llm_backbone') and hasattr(base_model.llm_backbone, 'get_input_embeddings'):
        embedding_layer = base_model.llm_backbone.get_input_embeddings()
    else:
        # Search for embedding layer
        for module in base_model.modules():
            if isinstance(module, nn.Embedding) and module.num_embeddings > 10000:  # Vocabulary size check
                embedding_layer = module
                break
    
    # Store original forward method
    original_forward = base_model.forward

    # Create new forward method that uses HN-LoRA
    def _get_instruction_embeds_from_kwargs(kwargs):
        # Prefer explicit HN inputs if provided
        tokenizer = None
        try:
            tokenizer = base_model.llm_backbone.tokenizer if hasattr(base_model, 'llm_backbone') else None
        except Exception:
            tokenizer = None

        if 'hn_input_ids' in kwargs and kwargs['hn_input_ids'] is not None:
            hn_ids = kwargs['hn_input_ids']
            if isinstance(hn_ids, (list, tuple)):
                hn_ids = torch.tensor(hn_ids, dtype=torch.long, device=base_model.device)
            elif isinstance(hn_ids, torch.Tensor):
                hn_ids = hn_ids.to(device=base_model.device, dtype=torch.long)
            else:
                raise ValueError("hn_input_ids must be a list, tuple, or torch.Tensor")
            return embedding_layer(hn_ids)

        if 'hn_instruction_text' in kwargs and kwargs['hn_instruction_text'] is not None and tokenizer is not None:
            text = kwargs['hn_instruction_text']
            if isinstance(text, str):
                # 完全模仿训练侧的处理
                ids = tokenizer(text, add_special_tokens=True).input_ids  # 返回 list
                ids = torch.tensor([ids], dtype=torch.long, device=base_model.device)  # 加 batch 维度 [1, seq_len]
            else:
                # 批量处理（我觉得基本不会发生）
                texts = list(text)
                ids_list = [tokenizer(t, add_special_tokens=True).input_ids for t in texts]
                ids = torch.tensor(ids_list, dtype=torch.long, device=base_model.device)
            return embedding_layer(ids)

        return None

    def hn_lora_forward(**kwargs):
        # Get instruction embeddings (prefer explicit HN inputs if provided)
        with torch.no_grad():
            instruction_embeds = _get_instruction_embeds_from_kwargs(kwargs)
            # Ensure dtype matches the hypernet
            if hasattr(base_model.hn_lora_hypernet, 'dtype'):
                instruction_embeds = instruction_embeds.to(dtype=base_model.hn_lora_hypernet.dtype)
            elif hasattr(base_model.hn_lora_hypernet, 'parameters'):
                # Get dtype from first parameter
                first_param = next(base_model.hn_lora_hypernet.parameters(), None)
                if first_param is not None:
                    instruction_embeds = instruction_embeds.to(dtype=first_param.dtype)

        # Generate LoRA parameters
        lora_params = base_model.hn_lora_hypernet(instruction_embeds)

        # Set parameters in layers
        for layer, (lora_A, lora_B) in zip(base_model.hn_lora_layers, lora_params):
            layer.set_lora_params(lora_A, lora_B)

        # Remove HN-only kwargs before calling original forward
        kwargs.pop('hn_input_ids', None)
        kwargs.pop('hn_instruction_text', None)
        # Call original forward
        return original_forward(**kwargs)

    # Replace forward method
    base_model.forward = hn_lora_forward

    # CRITICAL: Also wrap predict_action method since that's what evaluation actually calls
    original_predict_action = base_model.predict_action

    def hn_lora_predict_action(**kwargs):
        # Get instruction embeddings (prefer explicit HN inputs if provided)
        with torch.no_grad():
            instruction_embeds = _get_instruction_embeds_from_kwargs(kwargs)
            # Ensure dtype matches the hypernet
            if hasattr(base_model.hn_lora_hypernet, 'parameters'):
                first_param = next(base_model.hn_lora_hypernet.parameters(), None)
                if first_param is not None:
                    instruction_embeds = instruction_embeds.to(dtype=first_param.dtype)

            # Generate LoRA parameters
            lora_params = base_model.hn_lora_hypernet(instruction_embeds)

            # Set parameters in layers
            for layer, (lora_A, lora_B) in zip(base_model.hn_lora_layers, lora_params):
                layer.set_lora_params(lora_A, lora_B)

        # Remove HN-only kwargs before calling original predict_action
        kwargs.pop('hn_input_ids', None)
        kwargs.pop('hn_instruction_text', None)
        # Call original predict_action
        return original_predict_action(**kwargs)

    # Replace predict_action method
    base_model.predict_action = hn_lora_predict_action

    def print_trainable_parameters():
        trainable = sum(p.numel() for p in base_model.hn_lora_hypernet.parameters() if p.requires_grad)
        total = sum(p.numel() for p in base_model.parameters())
        print(f"HN-LoRA trainable params: {trainable:,} || "
              f"all params: {total:,} || "
              f"trainable%: {100 * trainable / total:.2f}")
        print(f"Note: Only HyperNetwork parameters ({trainable:,}) are trainable")
    
    base_model.print_trainable_parameters = print_trainable_parameters

    return base_model