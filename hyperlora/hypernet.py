from typing import List, Tuple

import torch
import torch.nn as nn

from .config import HNLoRAConfig


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
    
    def forward(self, instruction_embeds: torch.Tensor, attention_mask: torch.Tensor = None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Generate LoRA parameters for all layers - optimized batch processing

        Args:
            instruction_embeds: [batch_size, seq_len, embed_dim]
            attention_mask: [batch_size, seq_len] - 1 for real tokens, 0 for padding
        """
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

            # 处理 attention mask
            if attention_mask is not None:
                # 扩展 mask 以匹配 concatenated sequence
                # instruction_mask + all ones for layer embeddings (层嵌入始终需要关注)
                layer_mask = torch.ones(batch_size, self.num_layers, device=device, dtype=attention_mask.dtype)
                combined_mask = torch.cat([attention_mask, layer_mask], dim=1)  # (batch, seq_len + num_layers)
                # 转换为 transformer 格式：0->True(mask), 1->False(attend)
                # PyTorch TransformerEncoder expects True for positions to be masked
                key_padding_mask = (combined_mask == 0)
            else:
                key_padding_mask = None

            # Single forward pass through Transformer encoder with mask
            encoded = self.context_encoder(combined, src_key_padding_mask=key_padding_mask)

            # Extract contexts for all layers
            layer_contexts = encoded[:, -self.num_layers:, :]  # (batch, num_layers, dim)
        else:  # MLP
            # For MLP, we can still batch process
            # 如果有 mask，只对有效 tokens 做 pooling
            if attention_mask is not None:
                # 扩展 mask 以匹配 embedding 维度
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(instruction_context)  # (batch, seq_len, dim)
                # 计算有效 tokens 的和
                sum_embeddings = (instruction_context * mask_expanded).sum(dim=1)  # (batch, dim)
                # 计算有效 tokens 的数量
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-7)  # (batch, dim) 避免除零
                # 平均池化
                instruction_pooled = sum_embeddings / sum_mask  # (batch, dim)
            else:
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
