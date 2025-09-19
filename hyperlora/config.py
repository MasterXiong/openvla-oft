from dataclasses import dataclass


@dataclass 
class HNLoRAConfig:
    """Configuration for HN-LoRA"""
    # LoRA parameters
    lora_rank: int = 32
    lora_alpha: float = 1.0
    lora_dropout: float = 0.0
    
    # HyperNetwork parameters
    context_embedding_dim: int = 128
    context_encoder_type: str = "transformer"  # "transformer" or "mlp"
    context_encoder_layers: int = 2  # Paper uses 2 layers
    context_encoder_heads: int = 4   # Paper uses 4 heads
    mlp_hidden_dim: int = 256        # Paper uses 256
    embedding_dropout: float = 0.1    # Paper uses 0.1 for generalization
    
    # Model dimensions (will be auto-detected)
    hidden_size: int = None
    num_hidden_layers: int = None
