import torch
import torch.nn as nn
import torch.nn.functional as F


class HNLoRALinear(nn.Module):
    """Modified Linear layer with HN-LoRA support"""
    
    def __init__(self, original_linear: nn.Linear, lora_rank: int, lora_alpha: float, lora_dropout: float = 0.0):
        super().__init__()
        # Store original layer parameters
        self.weight = original_linear.weight
        self.bias = original_linear.bias
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # LoRA configuration
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout)
        
        # Storage for dynamically generated LoRA params (not trainable)
        self.register_buffer('lora_A', None)
        self.register_buffer('lora_B', None)
    
    def extra_repr(self) -> str:
        """Extra representation for printing"""
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'lora_rank={self.lora_rank}, lora_alpha={self.lora_alpha}')
    
    def set_lora_params(self, lora_A: torch.Tensor, lora_B: torch.Tensor):
        """Set the LoRA parameters for this layer"""
        self.lora_A = lora_A
        self.lora_B = lora_B
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward
        result = F.linear(x, self.weight, self.bias)
        
        # Add LoRA if parameters are set
        if self.lora_A is not None and self.lora_B is not None:
            if self.lora_A.dim() == 3:  # Batch-specific params
                # Handle different input shapes
                orig_shape = x.shape
                if x.dim() > 2:
                    # Flatten all but last dim
                    x_flat = x.reshape(-1, x.shape[-1])
                    batch_size = self.lora_A.shape[0]
                    
                    # Expand LoRA params if needed
                    if x_flat.shape[0] != batch_size:
                        # Assuming x has sequence dimension
                        seq_len = x_flat.shape[0] // batch_size
                        lora_A_exp = self.lora_A.unsqueeze(1).expand(-1, seq_len, -1, -1).reshape(x_flat.shape[0], self.in_features, self.lora_rank)
                        lora_B_exp = self.lora_B.unsqueeze(1).expand(-1, seq_len, -1, -1).reshape(x_flat.shape[0], self.lora_rank, self.out_features)
                    else:
                        lora_A_exp = self.lora_A
                        lora_B_exp = self.lora_B
                    
                    # Apply LoRA
                    lora_out = torch.bmm(x_flat.unsqueeze(1), lora_A_exp).squeeze(1)
                    lora_out = torch.bmm(lora_out.unsqueeze(1), lora_B_exp).squeeze(1)
                    
                    # Reshape back
                    lora_out = lora_out.reshape(*orig_shape[:-1], self.out_features)
                else:
                    # Simple 2D case
                    lora_out = torch.bmm(x.unsqueeze(1), self.lora_A).squeeze(1)
                    lora_out = torch.bmm(lora_out.unsqueeze(1), self.lora_B).squeeze(1)
            else:  # Shared params (2D)
                lora_out = x @ self.lora_A @ self.lora_B
            
            # Apply dropout and scaling
            lora_out = self.lora_dropout(lora_out)
            result = result + lora_out * self.lora_alpha / self.lora_rank
        
        return result
