"""
HN-LoRA (HyperNetwork LoRA) implementation for OpenVLA - Version 8
Based on paper: "Efficient Domain Adaptation of Robotic Foundation Models via Hypernetwork-Generated LoRA"

Correct implementation: Groups layers by dimension and shares output heads within each group.
Layer indices are treated as tokens in the Transformer, not just embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple, Any
import math
from dataclasses import dataclass

# Try to import torchinfo for model summaries
try:
    from torchinfo import summary
    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False
    print("torchinfo not available. Install with: pip install torchinfo")

import time
import os
import json
import datetime
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import draccus
import torch
import torch.distributed as dist
import torch.nn as nn

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

'''
HyperNetwork的输入是如下两部分：

1. 语言指令的tokens

- 来源：用户的语言指令（如"pick up the red block"）
- 处理：通过LLM的embedding层转为embeddings
- 作用：提供任务的语义信息

2. Layer indices

- 来源：96个层的索引 [0,1,2...95]
- 处理：通过learned embedding层转为embeddings
- 作用：区分不同层，生成层特定的LoRA参数

核心架构（符合论文Figure 1）：

语言指令 + Layer indices
      ↓
Transformer Encoder
      ↓
Layer-specific contexts
      ↓
Output heads
      ↓
96组LoRA参数 (W_up, W_down)

论文：
- 不是为每个任务存储固定的LoRA参数
- 而是根据语言指令动态生成LoRA参数
'''


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
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        # Storage for dynamically generated LoRA params (not trainable)
        self.register_buffer('lora_A', None)
        self.register_buffer('lora_B', None)
    
    def extra_repr(self) -> str:
        """Extra representation for printing"""
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'lora_rank={self.lora_rank}, lora_alpha={self.lora_alpha}')
    
    def set_lora_params(self, lora_A: torch.Tensor, lora_B: torch.Tensor):
        """Set the LoRA parameters for this layer"""
        # Store with the same dtype as the weight matrix for compatibility
        if lora_A is not None and self.weight is not None:
            lora_A = lora_A.to(dtype=self.weight.dtype, device=self.weight.device)
        if lora_B is not None and self.weight is not None:
            lora_B = lora_B.to(dtype=self.weight.dtype, device=self.weight.device)
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


class HyperNetwork(nn.Module):
    """HyperNetwork that generates LoRA parameters"""
    
    def __init__(self, config: HNLoRAConfig, layer_dims: List[Tuple[int, int]]):
        super().__init__()
        self.config = config
        self.layer_dims = layer_dims  # List of (layer_name, (in_features, out_features)) for each layer
        self.num_layers = len(layer_dims)
        
        # Layer embeddings
        self.layer_embedding = nn.Embedding(self.num_layers, config.context_embedding_dim)
        
        # Instruction projection (OpenVLA uses 4096-dim embeddings)
        actual_hidden_size = 4096 if config.hidden_size == 768 else config.hidden_size
        self.instruction_projection = nn.Linear(
            actual_hidden_size,
            config.context_embedding_dim
        )
        
        # Context encoder
        if config.context_encoder_type == "mlp":
            self.context_encoder = nn.Sequential(
                nn.Linear(config.context_embedding_dim, config.mlp_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.lora_dropout),
                nn.Linear(config.mlp_hidden_dim, config.context_embedding_dim),
                nn.GELU()
            )
        else:  # transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.context_embedding_dim,
                nhead=config.context_encoder_heads,
                dim_feedforward=config.mlp_hidden_dim,
                dropout=config.lora_dropout,
                activation='gelu',
                batch_first=True
            )
            self.context_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=config.context_encoder_layers
            )
        
        # Dropout for embeddings
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        
        # Group layers by dimensions and create shared output heads for each group
        # This is the key insight from the paper - reuse output heads for layers with same dimensions
        self.output_head_groups = {}  # Maps (in_dim, out_dim) -> list of layer indices
        self.output_head_dim = {}
        for idx, (layer_name, (in_dim, out_dim)) in enumerate(layer_dims):
            if layer_name not in self.output_head_groups:
                self.output_head_groups[layer_name] = []
                self.output_head_dim[layer_name] = (in_dim, out_dim)
            self.output_head_groups[layer_name].append(idx)
        
        # Create shared output heads for each unique dimension pair
        self.output_heads = nn.ModuleDict()
        for layer_name in self.output_head_groups:
            in_dim, out_dim = self.output_head_dim[layer_name]
            layer_indices = self.output_head_groups[layer_name]
            self.output_heads[layer_name + "_down"] = nn.Linear(
                config.context_embedding_dim, in_dim * config.lora_rank
            )
            self.output_heads[layer_name + "_up"] = nn.Linear(
                config.context_embedding_dim, config.lora_rank * out_dim
            )
            # Initialize
            nn.init.zeros_(self.output_heads[layer_name + "_down"].weight)
            nn.init.normal_(self.output_heads[layer_name + "_down"].bias, std=0.001)
            nn.init.zeros_(self.output_heads[layer_name + "_up"].weight)
            nn.init.zeros_(self.output_heads[layer_name + "_up"].bias)
        
            print(f"\nGroup {layer_name}: Dimension ({in_dim}, {out_dim})")
            print(f"  Number of layers in this group: {len(layer_indices)}")
            print(f"  Layer indices: {layer_indices}")
            
            # Print output head information for this group
            down_head = self.output_heads[layer_name + "_down"]
            up_head = self.output_heads[layer_name + "_up"]
            
            print(f"\n  Output heads for this group:")
            print(f"    Down projection head '{layer_name}_down':")
            print(f"      Input: {down_head.in_features} → Output: {down_head.out_features}")
            print(f"      Parameters: {sum(p.numel() for p in down_head.parameters()):,}")
            print(f"      Generates: LoRA A matrices of shape ({in_dim}, {config.lora_rank})")
            
            print(f"    Up projection head '{layer_name}_up':")
            print(f"      Input: {up_head.in_features} → Output: {up_head.out_features}")
            print(f"      Parameters: {sum(p.numel() for p in up_head.parameters()):,}")
            print(f"      Generates: LoRA B matrices of shape ({config.lora_rank}, {out_dim})")
            
            print(f"\n  These heads will generate LoRA parameters for {len(layer_indices)} layers")
            print(f"  Total parameters for this dimension group: {sum(p.numel() for p in down_head.parameters()) + sum(p.numel() for p in up_head.parameters()):,}")
        
        print("\n" + "="*80)
        print(f"Summary: {self.num_layers} layers grouped into {len(self.output_head_dim)} dimension groups")
        total_head_params = sum(sum(p.numel() for p in self.output_heads[name].parameters()) 
                               for name in self.output_heads)
        print(f"Total output head parameters: {total_head_params:,}")
        print("="*80)
        
        # Print detailed HyperNetwork parameter breakdown
        print("\n" + "="*80)
        print("HyperNetwork Parameter Breakdown")
        print("="*80)
        
        # Calculate each component's parameters
        instruction_proj_params = sum(p.numel() for p in self.instruction_projection.parameters())
        layer_embed_params = sum(p.numel() for p in self.layer_embedding.parameters())
        encoder_params = sum(p.numel() for p in self.context_encoder.parameters())
        
        total_hypernet_params = instruction_proj_params + layer_embed_params + encoder_params + total_head_params
        
        print(f"\nTotal HyperNetwork size: {total_hypernet_params:,} parameters (~{total_hypernet_params/1e6:.1f}M)")
        print("\nDetailed composition:")
        print(f"1. Instruction Projection: {self.instruction_projection.in_features} → {self.instruction_projection.out_features}")
        print(f"   - {instruction_proj_params:,} parameters")
        print(f"2. Layer Embedding: {self.num_layers} layers × {config.context_embedding_dim} dim")
        print(f"   - {layer_embed_params:,} parameters")
        print(f"3. Transformer Encoder: {config.context_encoder_layers} layer(s), {config.context_encoder_heads} heads")
        print(f"   - {encoder_params:,} parameters")
        print(f"4. Output Heads (largest component): {total_head_params:,} parameters")
        
        # # Detail for each group's output heads
        # for (in_dim, out_dim), layer_indices in self.output_head_dim.items():
        #     key_str = f"{in_dim}_{out_dim}"
        #     down_params = sum(p.numel() for p in self.output_heads[key_str + "_down"].parameters())
        #     up_params = sum(p.numel() for p in self.output_heads[key_str + "_up"].parameters())
        #     print(f"   - Group ({in_dim}→{out_dim}): {len(layer_indices)} layers shared")
        #     print(f"     • Down head: {self.output_heads[key_str + '_down'].in_features} → {self.output_heads[key_str + '_down'].out_features} = {down_params:,} parameters")
        #     print(f"     • Up head: {self.output_heads[key_str + '_up'].in_features} → {self.output_heads[key_str + '_up'].out_features} = {up_params:,} parameters")
        
        # print(f"\nKey observations:")
        # print(f"• Output heads占了总参数的{100*total_head_params/total_hypernet_params:.1f}% ({total_head_params/1e6:.1f}M/{total_hypernet_params/1e6:.1f}M)")
        # print(f"• 这是性能瓶颈的主要原因")
        # print(f"• 虽然共享了heads（只有{len(self.output_heads)}个而不是{self.num_layers*2}个），但每个head仍然很大")
        
        # Calculate generated LoRA parameters
        print("\n" + "="*80)
        print("Generated LoRA Parameters (in Base Network)")
        print("="*80)
        
        total_lora_params = 0
        for layer_name in self.output_head_groups:
            in_dim, out_dim = self.output_head_dim[layer_name]
            layer_indices = self.output_head_groups[layer_name]
            group_lora_A = in_dim * config.lora_rank * len(layer_indices)
            group_lora_B = config.lora_rank * out_dim * len(layer_indices)
            group_total = group_lora_A + group_lora_B
            total_lora_params += group_total
            
            print(f"\nGroup {layer_name}: ({in_dim}, {out_dim}) - {len(layer_indices)} layers")
            print(f"  Layer indices: {layer_indices}")
            print(f"  Output heads:")
            print(f"    • Down head: {config.context_embedding_dim}→{in_dim * config.lora_rank} ({sum(p.numel() for p in self.output_heads[f'{layer_name}_down'].parameters())/1e6:.1f}M params) 生成{in_dim}×{config.lora_rank}的LoRA A")
            print(f"    • Up head: {config.context_embedding_dim}→{config.lora_rank * out_dim} ({sum(p.numel() for p in self.output_heads[f'{layer_name}_up'].parameters())/1e6:.1f}M params) 生成{config.lora_rank}×{out_dim}的LoRA B")
            print(f"  LoRA parameters to generate:")
            print(f"    • LoRA A (W_down): {group_lora_A:,} params = {in_dim}×{config.lora_rank}×{len(layer_indices)}")
            print(f"    • LoRA B (W_up): {group_lora_B:,} params = {config.lora_rank}×{out_dim}×{len(layer_indices)}")
            print(f"  Subtotal: {group_total:,} parameters")
        
        print("\n" + "="*80)
        print("关键数字总结")
        print("="*80)
        print("\n| 组件 | 参数量 | 说明 |")
        print("|------|--------|------|")
        print(f"| **HyperNetwork总参数** | {total_hypernet_params/1e6:.1f}M | Trainable |")
        print(f"| - Instruction/Layer/Transformer | {(instruction_proj_params + layer_embed_params + encoder_params)/1e6:.1f}M | {100*(instruction_proj_params + layer_embed_params + encoder_params)/total_hypernet_params:.0f}% |")
        print(f"| - Output Heads | {total_head_params/1e6:.1f}M | {100*total_head_params/total_hypernet_params:.0f}% |")
        print(f"| **生成的LoRA参数** | {total_lora_params/1e6:.1f}M | Non-trainable，动态生成 |")
        print(f"| **效率比** | {total_hypernet_params/total_lora_params:.2f}:1 | 用{total_hypernet_params/1e6:.0f}M生成{total_lora_params/1e6:.1f}M |")
        print("="*80)
    
    def print_summary(self, batch_size: int = 1, device: str = 'cpu'):
        """Print a detailed summary of the HyperNetwork using torchinfo"""
        if not TORCHINFO_AVAILABLE:
            print("torchinfo not installed. Install with: pip install torchinfo")
            print("\nHyperNetwork Structure:")
            print(f"  Total layers to generate LoRA for: {self.num_layers}")
            print(f"  Unique dimension groups: {len(self.output_head_groups)}")
            print(f"  Context embedding dim: {self.config.context_embedding_dim}")
            print(f"  LoRA rank: {self.config.lora_rank}")
            print(f"  Encoder type: {self.config.context_encoder_type}")
            if self.config.context_encoder_type == "transformer":
                print(f"  Transformer layers: {self.config.context_encoder_layers}")
                print(f"  Transformer heads: {self.config.context_encoder_heads}")
            print(f"  MLP hidden dim: {self.config.mlp_hidden_dim}")
            
            # Count parameters
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"\n  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            return
        
        print("\n" + "="*80)
        print("HyperNetwork Architecture Summary")
        print("="*80)
        
        # Create sample input - assuming OpenVLA uses 4096-dim embeddings
        actual_hidden_size = 4096 if self.config.hidden_size == 768 else self.config.hidden_size
        sample_input = torch.randn(batch_size, 32, actual_hidden_size).to(device)  # 32 is typical sequence length
        
        # Display the summary
        model_stats = summary(
            self.to(device),
            input_data=sample_input,
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"],
            verbose=2
        )
        
        # print("\nDimension Groups:")
        # for key, indices in self.dim_groups.items():
        #     print(f"  {key}: {len(indices)} layers - indices {indices[:3]}..." if len(indices) > 3 else f"  {key}: {len(indices)} layers - indices {indices}")
    
    def forward(self, instruction_embeds: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Generate LoRA parameters for all layers - optimized batch processing"""
        batch_size = instruction_embeds.shape[0]
        device = instruction_embeds.device
        
        # Project instructions
        instruction_context = self.instruction_projection(instruction_embeds)
        
        # Get all layer embeddings at once
        all_layer_indices = torch.arange(self.num_layers, device=device)
        all_layer_embeds = self.layer_embedding(all_layer_indices)  # (num_layers, dim)
        
        if self.config.context_encoder_type == "transformer":
            # Batch process all layers through Transformer encoder
            # Expand layer embeddings for batch
            layer_embeds_batch = all_layer_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, num_layers, dim)
            
            # Concatenate instruction context with all layer embeddings as a sequence
            combined = torch.cat([
                instruction_context,  # (batch, seq_len, dim)
                layer_embeds_batch   # (batch, num_layers, dim)
            ], dim=1)  # (batch, seq_len + num_layers, dim)
            
            # Single forward pass through Transformer encoder
            encoded = self.context_encoder(combined)
            
            # Extract contexts for all layers
            layer_contexts = encoded[:, -self.num_layers:, :]  # (batch, num_layers, dim)
        else:  # MLP
            # For MLP, we can still batch process
            instruction_pooled = instruction_context.mean(dim=1)  # (batch, dim)
            # Broadcast addition
            instruction_pooled_exp = instruction_pooled.unsqueeze(1).expand(-1, self.num_layers, -1)  # (batch, num_layers, dim)
            layer_embeds_batch = all_layer_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, num_layers, dim)
            
            # Combine and reshape for batch processing through MLP
            combined = instruction_pooled_exp + layer_embeds_batch  # (batch, num_layers, dim)
            combined_flat = combined.reshape(batch_size * self.num_layers, -1)  # (batch*num_layers, dim)
            
            # Process through MLP
            encoded_flat = self.context_encoder(combined_flat)  # (batch*num_layers, dim)
            layer_contexts = encoded_flat.reshape(batch_size, self.num_layers, -1)  # (batch, num_layers, dim)
        
        # Apply dropout to all contexts at once
        layer_contexts = self.embedding_dropout(layer_contexts)
        
        # Process each dimension group together for efficiency
        lora_params = [None] * self.num_layers
        
        for layer_name in self.output_head_dim:
            in_dim, out_dim = self.output_head_dim[layer_name]
            layer_indices = self.output_head_groups[layer_name]
            # Get contexts for all layers in this group
            group_contexts = layer_contexts[:, layer_indices, :]  # (batch, group_size, dim)
            
            # Use shared output heads for this dimension group
            lora_down = self.output_heads[layer_name + "_down"](group_contexts)
            lora_up = self.output_heads[layer_name + "_up"](group_contexts)
            
            # Assign to correct positions
            for i, layer_idx in enumerate(layer_indices):
                lora_A = lora_down[:, i, :].view(batch_size, in_dim, self.config.lora_rank)
                lora_B = lora_up[:, i, :].view(batch_size, self.config.lora_rank, out_dim)
                lora_params[layer_idx] = (lora_A, lora_B)
        
        return lora_params


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
                # Check if it's a target layer (MLP/FFN in the language model)
                # Apply LoRA to all linear layers
                # is_target = any(p in full_name.lower() for p in ['mlp', 'ffn', 'gate', 'up_proj', 'down_proj'])
                # is_target = True
                # is_llm = 'llm_backbone' in full_name or 'language_model' in full_name
                # is_llm = True
                # skip original VLM output head
                if "lm_head" in full_name:
                    continue
                
                # if is_target and is_llm:
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
    
    # Print HyperNetwork summary if torchinfo is available
    print("\n" + "="*80)
    print("HN-LoRA HyperNetwork Created")
    print("="*80)
    hypernet.print_summary(batch_size=1, device=str(device) if device is not None else 'cpu')
    
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
    # Track if this has been called
    if not hasattr(base_model, '_hn_lora_call_count'):
        base_model._hn_lora_call_count = 0

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
                hn_ids = None
            if hn_ids is not None:
                if (hasattr(base_model, '_debug_hn_lora') and base_model._debug_hn_lora) or (os.environ.get("OPENVLA_DEBUG_INPUT_IDS", "0") == "1"):
                    print("\033[92m[HN][INSTRUCTION] Using explicit hn_input_ids for HyperNet conditioning\033[0m")
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

            if (hasattr(base_model, '_debug_hn_lora') and base_model._debug_hn_lora) or (os.environ.get("OPENVLA_DEBUG_INPUT_IDS", "0") == "1"):
                print("\033[92m[HN][INSTRUCTION] Using hn_instruction_text for HyperNet conditioning\033[0m")
            return embedding_layer(ids)

        return None

    def hn_lora_forward(*args, **kwargs):
        # Increment call counter
        base_model._hn_lora_call_count += 1

        # Print colorful message on first call
        if base_model._hn_lora_call_count == 1:
            print("\n" + "="*80)
            print("\033[92m[SUCCESS] HN-LoRA Forward is being called!\033[0m")
            print("\033[92m[CONFIRMED] This confirms the fix is working correctly!\033[0m")
            print("="*80 + "\n")
        elif base_model._hn_lora_call_count % 100 == 0:
            print(f"\033[94m[UPDATE] HN-LoRA Forward called {base_model._hn_lora_call_count} times...\033[0m")

        # Extract input_ids - handle both input_ids and inputs_embeds cases
        input_ids = kwargs.get('input_ids', args[0] if args else None)
        inputs_embeds = kwargs.get('inputs_embeds', None)

        # Handle case where inputs_embeds is provided instead of input_ids (happens during generation with cache)
        if input_ids is None and inputs_embeds is not None:
            # During cached generation, we only have the last token's embedding
            # We need to handle this case gracefully
            if hasattr(base_model, '_debug_hn_lora') and base_model._debug_hn_lora:
                print(f"\n[HN-LoRA Debug] Forward pass with inputs_embeds (cached generation):")
                print(f"  - inputs_embeds shape: {inputs_embeds.shape}")
                print(f"  - Using previously generated LoRA params")

            # For cached generation, we don't regenerate LoRA params
            # The params should already be set from the first forward pass
            # Just call original forward
            return original_forward(*args, **kwargs)

        # Debug output
        if hasattr(base_model, '_debug_hn_lora') and base_model._debug_hn_lora:
            print(f"\n[HN-LoRA Debug] Forward pass:")
            print(f"  - input_ids shape: {input_ids.shape if input_ids is not None else 'None'}")
            if input_ids is not None and input_ids.numel() > 0:
                print(f"  - input_ids first 10 tokens: {input_ids[0][:10].tolist() if input_ids.shape[1] >= 10 else input_ids[0].tolist()}")

        # Get instruction embeddings (prefer explicit HN inputs if provided)
        with torch.no_grad():
            instruction_embeds = _get_instruction_embeds_from_kwargs(kwargs)
            if instruction_embeds is None:
                instruction_embeds = embedding_layer(input_ids)
            # Ensure dtype matches the hypernet
            if hasattr(base_model.hn_lora_hypernet, 'dtype'):
                instruction_embeds = instruction_embeds.to(dtype=base_model.hn_lora_hypernet.dtype)
            elif hasattr(base_model.hn_lora_hypernet, 'parameters'):
                # Get dtype from first parameter
                first_param = next(base_model.hn_lora_hypernet.parameters(), None)
                if first_param is not None:
                    instruction_embeds = instruction_embeds.to(dtype=first_param.dtype)

        if hasattr(base_model, '_debug_hn_lora') and base_model._debug_hn_lora:
            print(f"  - instruction_embeds shape: {instruction_embeds.shape}")
            print(f"  - instruction_embeds dtype: {instruction_embeds.dtype}")
            print(f"  - instruction_embeds mean: {instruction_embeds.mean().item():.6f}")
            print(f"  - instruction_embeds std: {instruction_embeds.std().item():.6f}")

        # Generate LoRA parameters
        lora_params = base_model.hn_lora_hypernet(instruction_embeds)

        if hasattr(base_model, '_debug_hn_lora') and base_model._debug_hn_lora:
            print(f"  - Generated {len(lora_params)} LoRA parameter pairs")
            # Check first few LoRA parameters
            for i, (lora_A, lora_B) in enumerate(lora_params[:3]):
                print(f"    Layer {i}: lora_A norm={torch.norm(lora_A).item():.6f}, lora_B norm={torch.norm(lora_B).item():.6f}")

        # Set parameters in layers
        for layer, (lora_A, lora_B) in zip(base_model.hn_lora_layers, lora_params):
            layer.set_lora_params(lora_A, lora_B)

        # Remove HN-only kwargs before calling original forward
        kwargs.pop('hn_input_ids', None)
        kwargs.pop('hn_instruction_text', None)
        # Call original forward
        return original_forward(*args, **kwargs)

    # Replace forward method
    base_model.forward = hn_lora_forward
    print(f"[HN-LoRA] Forward method replaced. Base model type: {type(base_model).__name__}")

    # CRITICAL: Also wrap predict_action method since that's what evaluation actually calls
    if hasattr(base_model, 'predict_action'):
        original_predict_action = base_model.predict_action

        def hn_lora_predict_action(
            input_ids=None,
            unnorm_key=None,
            proprio=None,
            proprio_projector=None,
            action_head=None,
            noisy_action_projector=None,
            use_film=False,
            **kwargs
        ):
            # Increment call counter
            base_model._hn_lora_call_count += 1

            # Print colorful message on first call
            if base_model._hn_lora_call_count == 1:
                print("\n" + "="*80)
                print("\033[92m[SUCCESS] HN-LoRA predict_action is being called!\033[0m")
                print("\033[92m[CONFIRMED] This confirms the fix is working correctly!\033[0m")
                print("\033[92m[ACTIVE] HyperNetwork is now generating LoRA parameters!\033[0m")
                print("="*80 + "\n")

            # Print progress every call with different colors
            if base_model._hn_lora_call_count <= 10:
                # First 10 calls - show in purple
                print(f"\033[95m[CALL] HN-LoRA call #{base_model._hn_lora_call_count}: Processing action prediction...\033[0m")
            elif base_model._hn_lora_call_count % 50 == 0:
                # Every 50 calls - show in blue
                print(f"\033[94m[MILESTONE] HN-LoRA milestone: {base_model._hn_lora_call_count} predictions completed\033[0m")
            elif base_model._hn_lora_call_count % 10 == 0:
                # Every 10 calls - show in cyan
                print(f"\033[96m[ACTIVE] HN-LoRA active: {base_model._hn_lora_call_count} calls\033[0m")

            # Debug output
            if hasattr(base_model, '_debug_hn_lora') and base_model._debug_hn_lora:
                print(f"\n[HN-LoRA Debug] predict_action called:")
                print(f"  - input_ids shape: {input_ids.shape if input_ids is not None else 'None'}")

            # Get instruction embeddings (prefer explicit HN inputs if provided)
            if input_ids is not None or 'hn_input_ids' in kwargs or 'hn_instruction_text' in kwargs:
                with torch.no_grad():
                    instruction_embeds = _get_instruction_embeds_from_kwargs(kwargs)
                    if instruction_embeds is None:
                        instruction_embeds = embedding_layer(input_ids)
                    # Ensure dtype matches the hypernet
                    if hasattr(base_model.hn_lora_hypernet, 'parameters'):
                        first_param = next(base_model.hn_lora_hypernet.parameters(), None)
                        if first_param is not None:
                            instruction_embeds = instruction_embeds.to(dtype=first_param.dtype)

                    if hasattr(base_model, '_debug_hn_lora') and base_model._debug_hn_lora:
                        print(f"  - instruction_embeds shape: {instruction_embeds.shape}")
                        print(f"  - instruction_embeds dtype: {instruction_embeds.dtype}")

                    # Generate LoRA parameters
                    lora_params = base_model.hn_lora_hypernet(instruction_embeds)

                    # Always show LoRA parameter generation info for first few calls
                    if base_model._hn_lora_call_count <= 5:
                        print(f"\033[93m[GENERATED] {len(lora_params)} LoRA parameter pairs for {len(base_model.hn_lora_layers)} layers\033[0m")

                    if hasattr(base_model, '_debug_hn_lora') and base_model._debug_hn_lora:
                        print(f"  - Generated {len(lora_params)} LoRA parameter pairs")

                    # Set parameters in layers
                    for layer, (lora_A, lora_B) in zip(base_model.hn_lora_layers, lora_params):
                        layer.set_lora_params(lora_A, lora_B)

            # Remove HN-only kwargs before calling original predict_action
            kwargs.pop('hn_input_ids', None)
            kwargs.pop('hn_instruction_text', None)
            # Call original predict_action
            return original_predict_action(
                input_ids=input_ids,
                unnorm_key=unnorm_key,
                proprio=proprio,
                proprio_projector=proprio_projector,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector,
                use_film=use_film,
                **kwargs
            )

        # Replace predict_action method
        base_model.predict_action = hn_lora_predict_action
        print(f"[HN-LoRA] predict_action method also wrapped for HN-LoRA")

    print("\033[93m[WAITING] Waiting for first HN-LoRA call to verify it's working...\033[0m")

    # Add method to print trainable parameters
    def print_trainable_parameters():
        trainable = sum(p.numel() for p in base_model.hn_lora_hypernet.parameters() if p.requires_grad)
        total = sum(p.numel() for p in base_model.parameters())
        print(f"HN-LoRA trainable params: {trainable:,} || "
              f"all params: {total:,} || "
              f"trainable%: {100 * trainable / total:.2f}")
        print(f"Note: Only HyperNetwork parameters ({trainable:,}) are trainable")
    
    base_model.print_trainable_parameters = print_trainable_parameters
    
    # Add comprehensive model summary method
    def print_hn_lora_summary(show_lora_layers: bool = False, show_base_network: bool = True, 
                              show_hypernetwork: bool = True, batch_size: int = 1):
        """
        Print comprehensive HN-LoRA model summary showing BOTH networks
        
        Args:
            show_lora_layers: Whether to list individual LoRA layers
            show_base_network: Whether to show base network architecture
            show_hypernetwork: Whether to show hypernetwork architecture
            batch_size: Batch size for summary
        """
        print("\n" + "="*80)
        print("HN-LoRA Complete System Summary")
        print("="*80)
        
        # === PART 1: Configuration ===
        print(f"\n{'='*40} Configuration {'='*40}")
        print(f"  LoRA Rank: {config.lora_rank}")
        print(f"  LoRA Alpha: {config.lora_alpha}")
        print(f"  LoRA Dropout: {config.lora_dropout}")
        print(f"  Context Embedding Dim: {config.context_embedding_dim}")
        print(f"  Encoder Type: {config.context_encoder_type}")
        if config.context_encoder_type == "transformer":
            print(f"  Transformer Layers: {config.context_encoder_layers}")
            print(f"  Transformer Heads: {config.context_encoder_heads}")
        print(f"  MLP Hidden Dim: {config.mlp_hidden_dim}")
        
        # === PART 2: Base Network Summary ===
        if show_base_network:
            print(f"\n{'='*35} BASE NETWORK (with LoRA) {'='*35}")
            
            # Basic stats
            base_params = sum(p.numel() for p in base_model.parameters())
            base_trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
            print(f"\nBase Network Statistics:")
            print(f"  Total parameters: {base_params:,}")
            print(f"  Trainable parameters: {base_trainable:,}")
            print(f"  Frozen parameters: {base_params - base_trainable:,}")
            
            # LoRA Layer info
            print(f"\nLoRA Integration:")
            print(f"  Total LoRA layers: {len(lora_layers)}")
            unique_dims = {}
            for dim in layer_dims:
                if dim not in unique_dims:
                    unique_dims[dim] = 0
                unique_dims[dim] += 1
            print(f"  Unique dimension configurations: {len(unique_dims)}")
            for dim, count in unique_dims.items():
                print(f"    {dim[0]} → {dim[1]}: {count} layers")
            
            if show_lora_layers:
                print("\n  Sample LoRA layers (first 5):")
                for i, layer in enumerate(lora_layers[:5]):
                    print(f"    Layer {i}: {layer}")
            
            # Use torchinfo for base network if available
            if TORCHINFO_AVAILABLE and hasattr(base_model, '__class__'):
                try:
                    print("\nBase Network Architecture (simplified view):")
                    # Create sample input
                    sample_input_ids = torch.randint(0, 32000, (batch_size, 32))
                    if device is not None:
                        sample_input_ids = sample_input_ids.to(device)
                    
                    # Try to get a summary (this might fail for complex models)
                    print("  [Note: Full base network may be too complex for detailed display]")
                    print(f"  Model Type: {base_model.__class__.__name__}")
                    print(f"  LoRA-modified layers: {len(lora_layers)}")
                except Exception as e:
                    print(f"  [Could not generate full base network summary: {type(e).__name__}]")
        
        # === PART 3: HyperNetwork Summary ===
        if show_hypernetwork:
            print(f"\n{'='*35} HYPERNETWORK (LoRA Generator) {'='*35}")
            
            hypernet_params = sum(p.numel() for p in base_model.hn_lora_hypernet.parameters())
            print(f"\nHyperNetwork Statistics:")
            print(f"  Total parameters: {hypernet_params:,}")
            print(f"  All parameters trainable: Yes")
            print(f"  Dimension groups: {len(base_model.hn_lora_hypernet.output_head_dim)}")
            
            # # Show dimension groups
            # print(f"\nShared Output Head Groups:")
            # for (in_dim, out_dim), indices in base_model.hn_lora_hypernet.output_head_dim.items():
            #     print(f"  ({in_dim}, {out_dim}): {len(indices)} layers")
            
            # Detailed HyperNetwork architecture
            if TORCHINFO_AVAILABLE:
                print("\nDetailed HyperNetwork Architecture:")
                base_model.hn_lora_hypernet.print_summary(batch_size=batch_size, 
                                                         device=str(device) if device is not None else 'cpu')
        
        # === PART 4: System Overview ===
        print(f"\n{'='*35} SYSTEM OVERVIEW {'='*35}")
        
        # Total system stats
        total_params = sum(p.numel() for p in base_model.parameters())
        hypernet_params = sum(p.numel() for p in base_model.hn_lora_hypernet.parameters())
        base_frozen = total_params - hypernet_params
        
        print(f"\nComplete HN-LoRA System:")
        print(f"  Base Network (frozen): {base_frozen:,} parameters")
        print(f"  HyperNetwork (trainable): {hypernet_params:,} parameters")
        print(f"  Total: {total_params:,} parameters")
        print(f"  Trainable ratio: {100 * hypernet_params / total_params:.4f}%")
        
        # Memory estimates
        base_memory = base_frozen * 4 / (1024**3)  # GB for float32
        hypernet_memory = hypernet_params * 4 / (1024**2)  # MB for float32
        print(f"\nMemory Requirements (float32):")
        print(f"  Base Network: ~{base_memory:.2f} GB")
        print(f"  HyperNetwork: ~{hypernet_memory:.2f} MB")
        print(f"  Total: ~{base_memory:.2f} GB + {hypernet_memory:.2f} MB")
        
        # How it works
        print(f"\nOperational Flow:")
        print(f"  1. Input instruction → Base Network embeddings")
        print(f"  2. Embeddings → HyperNetwork")
        print(f"  3. HyperNetwork generates {len(lora_layers)} LoRA weight pairs")
        print(f"  4. LoRA weights injected into Base Network")
        print(f"  5. Base Network performs forward pass with task-specific adaptation")
        
        print("="*80)
    
    base_model.print_hn_lora_summary = print_hn_lora_summary
    
    # Add method to print base network structure with LoRA layers highlighted
    def print_base_network_structure(max_depth: int = 3):
        """Print base network structure with LoRA layers highlighted"""
        print("\n" + "="*80)
        print("Base Network Structure with LoRA Layers")
        print("="*80)
        
        def print_module_tree(module, prefix="", depth=0, module_name="model"):
            if depth > max_depth:
                return
                
            # Check if this is a LoRA layer
            is_lora = isinstance(module, HNLoRALinear)
            
            # Print current module
            if is_lora:
                print(f"{prefix}├─ [LoRA] {module_name}: {module}")
            else:
                print(f"{prefix}├─ {module_name}: {module.__class__.__name__}")
            
            # Recursively print children
            children = list(module.named_children())
            for i, (name, child) in enumerate(children):
                is_last = i == len(children) - 1
                child_prefix = prefix + ("    " if is_last else "│   ")
                print_module_tree(child, child_prefix, depth + 1, name)
        
        print_module_tree(base_model)
        
        # Summary statistics
        total_modules = sum(1 for _ in base_model.modules())
        lora_modules = sum(1 for m in base_model.modules() if isinstance(m, HNLoRALinear))
        print(f"\nSummary:")
        print(f"  Total modules: {total_modules}")
        print(f"  LoRA-modified modules: {lora_modules}")
        print(f"  LoRA coverage: {100 * lora_modules / total_modules:.2f}%")
        print("="*80)
    
    base_model.print_base_network_structure = print_base_network_structure
    
    # Override state_dict to only save HyperNetwork
    original_state_dict = base_model.state_dict
    
    def hn_lora_state_dict(*args, **kwargs):
        # Get full state dict
        full_state = original_state_dict(*args, **kwargs)
        # Filter to only HyperNetwork parameters
        hn_state = {k: v for k, v in full_state.items() if 'hn_lora_hypernet' in k}
        return hn_state
    
    base_model.hn_lora_state_dict = hn_lora_state_dict
    
    return base_model


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
        print(f"Warning: HyperNetwork state file not found at {hypernet_state_path}")


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
    vla = apply_hn_lora_to_base_model(vla, hn_config)
    
    load_hn_lora_checkpoint(vla, cfg.pretrained_checkpoint)

    _load_dataset_stats(vla, cfg.pretrained_checkpoint)
    vla.eval()

    # Enable debug mode if requested
    if hasattr(cfg, 'debug_hn_lora') and cfg.debug_hn_lora:
        vla._debug_hn_lora = True
        print("[HN-LoRA] Debug mode enabled")
    else:
        vla._debug_hn_lora = False

    return vla
