import os
import torch
import torch.nn as nn

from .config import HNLoRAConfig
from .generated_lora_layer import HNLoRALinear
from .hypernet import HyperNetwork


def apply_hn_lora_to_base_model(base_model, config: HNLoRAConfig, processor=None):
    """
    Apply HN-LoRA directly to the base model by:
    1. Replacing target Linear layers with HNLoRALinear
    2. Adding HyperNetwork as a module to the base model
    3. Modifying the forward method to use HN-LoRA
    4. Returning the modified base model itself

    Args:
        base_model: The base VLA model
        config: HN-LoRA configuration
        processor: Optional processor containing tokenizer (for evaluation)
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

    # Move HyperNetwork to same device and dtype as base model
    if next(base_model.parameters(), None) is not None:
        first_param = next(base_model.parameters())
        device = first_param.device
        dtype = first_param.dtype
        hypernet = hypernet.to(device=device, dtype=dtype)
        print(f"HyperNetwork moved to device={device}, dtype={dtype}")
    
    base_model.hn_lora_hypernet = hypernet
    base_model.hn_lora_layers = lora_layers

    # Store processor if provided (for evaluation)
    if processor is not None:
        base_model._hn_processor = processor  # 使用 _hn_ 前缀避免覆盖原有的 processor
    
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
        import os

        # 修复 tokenizer 查找
        tokenizer = None
        tokenizer_source = None

        # 优先级查找 tokenizer
        # 1. 优先查找我们存储的 processor
        if hasattr(base_model, '_hn_processor') and hasattr(base_model._hn_processor, 'tokenizer'):
            tokenizer = base_model._hn_processor.tokenizer
            tokenizer_source = "base_model._hn_processor.tokenizer (stored)"
        # 2. 然后查找标准位置
        elif hasattr(base_model, 'processor') and hasattr(base_model.processor, 'tokenizer'):
            tokenizer = base_model.processor.tokenizer
            tokenizer_source = "base_model.processor.tokenizer"
        # 3. 最后查找 LLM backbone
        elif hasattr(base_model, 'llm_backbone'):
            llm = base_model.llm_backbone
            if hasattr(llm, 'tokenizer'):
                tokenizer = llm.tokenizer
                tokenizer_source = "base_model.llm_backbone.tokenizer"
            elif hasattr(llm, 'model') and hasattr(llm.model, 'tokenizer'):
                tokenizer = llm.model.tokenizer
                tokenizer_source = "base_model.llm_backbone.model.tokenizer"

        # 打印 tokenizer 来源
        if tokenizer is not None and os.environ.get("OPENVLA_DEBUG_INPUT_IDS", "0") == "1":
            print(f"\033[96m[HN][TOKENIZER] Found from: {tokenizer_source}\033[0m")
            print(f"\033[96m[HN][TOKENIZER] Type: {type(tokenizer).__name__}\033[0m")

        # 处理 hn_input_ids（训练传入的）
        if 'hn_input_ids' in kwargs and kwargs['hn_input_ids'] is not None:
            hn_ids = kwargs['hn_input_ids']
            if isinstance(hn_ids, (list, tuple)):
                hn_ids = torch.tensor(hn_ids, dtype=torch.long, device=base_model.device)
            elif isinstance(hn_ids, torch.Tensor):
                hn_ids = hn_ids.to(device=base_model.device, dtype=torch.long)
            else:
                raise ValueError("hn_input_ids must be a list, tuple, or torch.Tensor")

            if os.environ.get("OPENVLA_DEBUG_INPUT_IDS", "0") == "1":
                print(f"\033[92m[HN][INPUT] Using hn_input_ids, shape: {hn_ids.shape}\033[0m")

            return embedding_layer(hn_ids)

        # 处理 hn_instruction_text（评估传入的）
        if 'hn_instruction_text' in kwargs and kwargs['hn_instruction_text'] is not None:
            if tokenizer is None:
                error_msg = (
                    "\033[91m[HN][ERROR] Cannot process hn_instruction_text: tokenizer not found!\033[0m\n"
                    "Tried: processor.tokenizer, llm_backbone.tokenizer, etc.\n"
                )
                # 添加诊断信息
                if hasattr(base_model, 'processor'):
                    error_msg += f"Processor type: {type(base_model.processor)}\n"
                    error_msg += f"Processor attrs with 'token': {[a for a in dir(base_model.processor) if 'token' in a.lower()]}\n"
                if hasattr(base_model, 'llm_backbone'):
                    error_msg += f"LLM backbone type: {type(base_model.llm_backbone)}\n"
                raise RuntimeError(error_msg)

            from prismatic.vla.constants import MAX_INSTRUCTION_LENGTH
            text = kwargs['hn_instruction_text']

            if os.environ.get("OPENVLA_DEBUG_INPUT_IDS", "0") == "1":
                print(f"\033[92m[HN][TEXT] Processing: '{text[:50]}{'...' if len(text) > 50 else ''}'\033[0m")

            # 使用固定长度 tokenization（与训练一致）
            if isinstance(text, str):
                tokenized = tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=MAX_INSTRUCTION_LENGTH,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                    return_attention_mask=True
                )
                ids = tokenized.input_ids.to(device=base_model.device)
                mask = tokenized.attention_mask.to(device=base_model.device)
                kwargs['hn_attention_mask'] = mask

                if os.environ.get("OPENVLA_DEBUG_INPUT_IDS", "0") == "1":
                    print(f"\033[92m[HN][TOKENS] shape: {ids.shape}, mask sum: {mask.sum().item()}\033[0m")
            else:
                # 批量处理
                texts = list(text)
                tokenized = tokenizer(
                    texts,
                    add_special_tokens=True,
                    max_length=MAX_INSTRUCTION_LENGTH,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                    return_attention_mask=True
                )
                ids = tokenized.input_ids.to(device=base_model.device)
                mask = tokenized.attention_mask.to(device=base_model.device)
                kwargs['hn_attention_mask'] = mask

                if os.environ.get("OPENVLA_DEBUG_INPUT_IDS", "0") == "1":
                    print(f"\033[92m[HN][TOKENS] Batch size: {len(texts)}, shape: {ids.shape}\033[0m")

            return embedding_layer(ids)

        return None

    def hn_lora_forward(**kwargs):
        # Get instruction embeddings (prefer explicit HN inputs if provided)
        with torch.no_grad():
            instruction_embeds = _get_instruction_embeds_from_kwargs(kwargs)
            attention_mask = kwargs.get('hn_attention_mask', None)

            # Ensure dtype matches the hypernet
            if hasattr(base_model.hn_lora_hypernet, 'dtype'):
                instruction_embeds = instruction_embeds.to(dtype=base_model.hn_lora_hypernet.dtype)
            elif hasattr(base_model.hn_lora_hypernet, 'parameters'):
                # Get dtype from first parameter
                first_param = next(base_model.hn_lora_hypernet.parameters(), None)
                if first_param is not None:
                    instruction_embeds = instruction_embeds.to(dtype=first_param.dtype)

        # Generate LoRA parameters with mask
        if attention_mask is not None:
            lora_params = base_model.hn_lora_hypernet(instruction_embeds, attention_mask)
        else:
            lora_params = base_model.hn_lora_hypernet(instruction_embeds)

        # Set parameters in layers
        for layer, (lora_A, lora_B) in zip(base_model.hn_lora_layers, lora_params):
            layer.set_lora_params(lora_A, lora_B)

        # Remove HN-only kwargs before calling original forward
        kwargs.pop('hn_input_ids', None)
        kwargs.pop('hn_instruction_text', None)
        kwargs.pop('hn_attention_mask', None)
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
            attention_mask = kwargs.get('hn_attention_mask', None)

            # Ensure dtype matches the hypernet
            if hasattr(base_model.hn_lora_hypernet, 'parameters'):
                first_param = next(base_model.hn_lora_hypernet.parameters(), None)
                if first_param is not None:
                    instruction_embeds = instruction_embeds.to(dtype=first_param.dtype)

            # Generate LoRA parameters with mask
            if attention_mask is not None:
                lora_params = base_model.hn_lora_hypernet(instruction_embeds, attention_mask)
            else:
                lora_params = base_model.hn_lora_hypernet(instruction_embeds)

            # Set parameters in layers
            for layer, (lora_A, lora_B) in zip(base_model.hn_lora_layers, lora_params):
                layer.set_lora_params(lora_A, lora_B)

        # Remove HN-only kwargs before calling original predict_action
        kwargs.pop('hn_input_ids', None)
        kwargs.pop('hn_instruction_text', None)
        kwargs.pop('hn_attention_mask', None)
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