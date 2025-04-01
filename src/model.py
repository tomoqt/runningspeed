# Add old pre-deepseek meta MTP and scale with that

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect

config = {
    "n_embd": 256,
    "n_head": 16,
    "n_layer": 4,
    "n_experts": 32,
    "dropout": 0.2,
    "vocab_size": 65,
    "ctx_len": 2048,
    "init_moe_scaling": 1.25,
    "type": ['mlp', 'moe', 'mlp', 'moe'],
    "device": 'cuda' if torch.cuda.is_available() else 'cpu'
}

# RoPE

class RoPE(nn.Module):
    def __init__(self, d, base=100_000_000_000, device=config['device']):
        super().__init__()

        self.base = base
        self.d = d
        self.device = device
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x):
        if self.cos_cached is not None:
            return

        head_dim = x.shape[-1]

        theta = 1 / (self.base ** (torch.arange(0, head_dim, 2, device=self.device).float() / self.d))
        seq_idx = torch.arange(x.shape[0], device=self.device).float()
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)

        cos_cache = torch.cos(idx_theta)
        sin_cache = torch.sin(idx_theta)

        self.cos_cached = torch.cat([cos_cache, cos_cache], dim=-1).unsqueeze(0).unsqueeze(0)
        self.sin_cached = torch.cat([sin_cache, sin_cache], dim=-1).unsqueeze(0).unsqueeze(0)

    def _neg_half(self, x):
        head_dim = x.shape[-1]
        d_2 = head_dim // 2
        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

    def forward(self, x):
        if self.cos_cached is None or self.cos_cached.shape[2] != x.shape[1]:
            self._build_cache(x)

        x_rope = x.clone()  # VERY IMPORTANT: Create a copy!
        neg_half_x = self._neg_half(x_rope)
        x_out = (x_rope * self.cos_cached[:, :, :x.shape[1], :]) + (neg_half_x * self.sin_cached[:, :, :x.shape[1], :])
        return x_out
#hyperbolic stuff
def precompute_freqs_cis(dim, end, device, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(end, device=device)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(x: torch.Tensor, y: torch.Tensor, freqs_cis) -> tuple[torch.Tensor,torch.Tensor]:
    cos_freqs, sin_freqs = freqs_cis
    seq_len = x.shape[-2]

    cos_seq = cos_freqs[:seq_len]
    sin_seq = sin_freqs[:seq_len]
    cos_seq = cos_seq.unsqueeze(0).unsqueeze(0)
    sin_seq = sin_seq.unsqueeze(0).unsqueeze(0)
    x_real, x_imag = x.chunk(2, dim=-1)
    y_real, y_imag = y.chunk(2, dim=-1)
    x_rotated_real = x_real * cos_seq - x_imag * sin_seq
    x_rotated_imag = x_real * sin_seq + x_imag * cos_seq
    y_rotated_real = y_real * cos_seq - y_imag * sin_seq
    y_rotated_imag = y_real * sin_seq + y_imag * cos_seq
    x_rotated = torch.cat([x_rotated_real, x_rotated_imag], dim=-1)
    y_rotated = torch.cat([y_rotated_real, y_rotated_imag], dim=-1)
    return x_rotated.type_as(x), y_rotated.type_as(y)


# Hyperbolic geometry utility functions
def mobius_addition(x, y, c):
    """Mobius addition in hyperbolic space with curvature c"""
    # Compute norms
    x_norm = torch.norm(x, dim=-1, keepdim=True)
    y_norm = torch.norm(y, dim=-1, keepdim=True)
    # Compute the inner product
    inner_product = torch.sum(x * y, dim=-1, keepdim=True)
    
    # Compute numerator and denominator following the standard formula
    numerator = (1 + 2*c * inner_product + c * (y_norm ** 2)) * x + \
                (1 - c * (x_norm ** 2)) * y
    denominator = 1 + 2*c * inner_product + (c ** 2) * (x_norm ** 2) * (y_norm ** 2)
    
    return numerator / denominator

def scaling_factor(x, c):
    """Compute scaling factor for hyperbolic space with curvature c"""
    x_norm = torch.norm(x, dim=-1, keepdim=True)
    return 2/(1+c*x_norm**2)

def expmap(x, v, c):
    """Exponential map from tangent space to hyperbolic space with curvature c"""
    scaling_factor_x = scaling_factor(x, c)
    v_norm = torch.norm(v, dim=-1, keepdim=True)
    second_term = (1/c**0.5)*torch.tanh((c*scaling_factor_x*v_norm**2/2)**0.5)*v/v_norm
    return mobius_addition(x, second_term, c)

def logmap(x, u, c):
    """Logarithmic map from hyperbolic space to tangent space with curvature c"""
    scaling_factor_x = scaling_factor(x, c)
    mob_addition = mobius_addition(-x, u, c)
    addition_norm = torch.norm(mob_addition, dim=-1, keepdim=True)
    constant_factor = 2 / (scaling_factor_x * c**0.5)
    direction_factor = mob_addition / addition_norm
    arg = torch.clamp((c * addition_norm) ** 0.5, min=-0.999, max=0.999)  # Single-line fix
    return constant_factor * torch.arctanh(arg) * direction_factor

def calculate_reference_point(x):
    """Calculate reference point for hyperbolic operations"""
    B, T, C = x.size()
    
    # If we have a sequence of length 1, we don't have a previous token
    # so use a zero tensor as reference
    if T <= 1:
        return torch.zeros_like(x)
    
    # Otherwise, use the previous token's position as reference
    # This handles causal scenarios well
    ref_point = x[:, :-1, :]
    ref_point = F.pad(ref_point, (0, 0, 1, 0), mode='constant', value=0)
    
    return ref_point


# MLA-NSA hybrid, not hardware optimized, just uses NSA sparsity for better training rn

class Attn(nn.Module):
    """
    Native Sparse Attention with Multi-headed Latent Attention integration.
    Combines MLA's compression techniques with NSA's natural sparsity, also better loss
    """
    def __init__(self, curvature):
        super().__init__()
        self.device = config['device']
        self.n_embd = config['n_embd']
        self.n_head = config['n_head']
        self.dropout = config['dropout']
        self.ctx_len = config['ctx_len']
        self.rms_norm_eps = config.get('rms_norm_eps', 1e-6)
        # Store curvature
        self.c = curvature

        # Original MLA parameters
        self.v_head_dim = 32
        self.kv_lora_rank = 32
        self.q_lora_rank = 3 * self.kv_lora_rank
        self.rope_head_dim = 64
        self.nope_head_dim = 32
        self.value_dim = self.n_head * self.v_head_dim
        self.nope_dim = self.n_head * self.nope_head_dim
        self.rope_dim = self.n_head * self.rope_head_dim

        # NSA-specific parameters
        self.block_size = config.get('block_size', 16)  # Size of token blocks for compression
        self.num_blocks = self.ctx_len // self.block_size
        self.window_size = config.get('window_size', 128)  # Sliding window size
        self.num_tokens_to_keep = config.get('num_tokens_to_keep', self.ctx_len // 4)  # Number of fine-grained tokens to keep

        # === Branch 1: Coarse-grained compression branch (adapted from MLA) ===
        self.compress_q_linear = nn.Linear(self.n_embd, self.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=self.rms_norm_eps)
        self.decompress_q_nope = nn.Linear(self.q_lora_rank, self.nope_dim, bias=False)
        self.decompress_q_rope = nn.Linear(self.q_lora_rank, self.rope_dim, bias=False)

        self.compress_kv_linear = nn.Linear(self.n_embd, self.kv_lora_rank, bias=False)
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=self.rms_norm_eps)
        self.decompress_k_nope = nn.Linear(self.kv_lora_rank, self.nope_dim, bias=False)
        self.decompress_v_linear = nn.Linear(self.kv_lora_rank, self.value_dim, bias=False)
        self.k_rope_linear = nn.Linear(self.n_embd, self.rope_head_dim, bias=False)

        # === Branch 2: Token Selection Branch (NSA) ===
        # Components for importance-based token selection
        self.importance_scorer = nn.Linear(self.n_embd, 1)
        # Independent KV for selected tokens
        self.selection_k = nn.Linear(self.n_embd, self.n_head * (self.rope_head_dim + self.nope_head_dim), bias=False)
        self.selection_v = nn.Linear(self.n_embd, self.value_dim, bias=False)

        # === Branch 3: Sliding Window Branch (NSA) ===
        # Independent KV for sliding window
        self.window_k = nn.Linear(self.n_embd, self.n_head * (self.rope_head_dim + self.nope_head_dim), bias=False)
        self.window_v = nn.Linear(self.n_embd, self.value_dim, bias=False)

        # Token Compression Mechanism (NSA)
        self.block_compressor = nn.Sequential(
            nn.Linear(self.block_size * self.n_embd, 4 * self.n_embd),
            nn.GELU(),
            nn.Linear(4 * self.n_embd, self.n_embd)
        )

        # Intra-block position encoding
        self.intra_block_pos_encoding = nn.Parameter(
            torch.randn(1, self.block_size, self.n_embd)
        )

        # Gated Multi-Branch Integration (NSA)
        self.branch_gate = nn.Linear(self.n_embd, 3)  # 3 gates for 3 branches

        # Output projection
        self.proj = nn.Linear(self.value_dim, self.n_embd, bias=False)
        self.res_dropout = nn.Dropout(p=self.dropout)

        # Caching for inference
        self.k_cache = None
        self.v_cache = None
        self.cache_filled = 0

        # RoPE
        self.rope = RoPE(self.rope_head_dim, device=self.device)
        self.freqs_cis = precompute_freqs_cis(self.rope_head_dim, self.ctx_len, self.device)

    def _compress_tokens(self, x):
        """Token compression mechanism from NSA"""
        B, T, C = x.size()

        # Ensure T is divisible by block_size for simplicity
        padded_len = ((T + self.block_size - 1) // self.block_size) * self.block_size
        if padded_len > T:
            padding = torch.zeros(B, padded_len - T, C, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=1)
        else:
            x_padded = x

        # Add intra-block position encoding
        blocks = x_padded.view(B, -1, self.block_size, C)
        pos_encoded_blocks = blocks + self.intra_block_pos_encoding

        # Reshape for compression
        blocks_flat = pos_encoded_blocks.view(B, -1, self.block_size * C)

        # Apply block compression
        compressed_blocks = self.block_compressor(blocks_flat)

        return compressed_blocks

    def _select_important_tokens(self, x, importance_scores):
        """Select the most important tokens based on scores"""
        B, T, _ = x.size()

        # Get indices of top-k tokens by importance
        _, indices = torch.topk(importance_scores.squeeze(-1),
                                min(self.num_tokens_to_keep, T),
                                dim=1)

        # Sort indices to maintain sequence order (continuity-aware)
        indices, _ = torch.sort(indices, dim=1)

        # Gather selected tokens
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, indices.size(1))
        selected_tokens = x[batch_indices, indices]

        return selected_tokens, indices

    def _get_sliding_window_tokens(self, x, current_pos=None):
        """Extract tokens within the sliding window"""
        if self.training or current_pos is None:
            # During training, we can use the whole sequence with windowed attention
            return x
        else:
            # During inference, get a window centered around the current position
            B, T, _ = x.size()
            window_start = max(0, current_pos - self.window_size // 2)
            window_end = min(T, window_start + self.window_size)
            return x[:, window_start:window_end]

    def forward(self, x):
        B, T, C = x.size()
        reference_point = calculate_reference_point(x)
        x_hyperbolic = logmap(reference_point, x, self.c)
        # === Prepare queries using MLA's approach ===
        compressed_q = self.compress_q_linear(x_hyperbolic)
        norm_q = self.q_norm(compressed_q)
        query_nope = self.decompress_q_nope(norm_q)
        query_rope = self.decompress_q_rope(norm_q)

        # Reshape and transpose queries
        query_nope = query_nope.view(B, T, self.n_head, self.nope_head_dim).transpose(1, 2)
        query_rope = query_rope.view(B, T, self.n_head, self.rope_head_dim).transpose(1, 2)

        # Apply RoPE to query
        q_rope, _ = apply_rope(query_rope, query_rope, self.freqs_cis)  # Corrected

        # Recombine query parts
        q_recombined = torch.empty((B, self.n_head, T, self.rope_head_dim + self.nope_head_dim),
                                  device=x.device, dtype=x.dtype)
        q_recombined[:, :, :, :self.nope_head_dim] = query_nope
        q_recombined[:, :, :, self.nope_head_dim:] = q_rope

        # Compute branch gates for dynamic weighting
        branch_weights = F.softmax(self.branch_gate(x).mean(dim=1), dim=-1)  # [B, 3]

        # === Branch 1: Coarse-grained compression branch (from MLA) ===
        compressed_kv = self.compress_kv_linear(x)
        norm_kv = self.kv_norm(compressed_kv)
        key_nope_1 = self.decompress_k_nope(norm_kv)
        value_1 = self.decompress_v_linear(norm_kv)
        key_rope_1 = self.k_rope_linear(x)

        # Reshape keys and values
        key_nope_1 = key_nope_1.view(B, T, self.n_head, self.nope_head_dim).transpose(1, 2)
        key_rope_1 = key_rope_1.view(B, T, 1, self.rope_head_dim).transpose(1, 2)
        value_1 = value_1.view(B, T, self.n_head, self.v_head_dim).transpose(1, 2)

        # Apply RoPE to keys
        key_rope_1 = key_rope_1 / self.n_head  # Scale like in original code
        _, k_rope_1 = apply_rope(key_rope_1, key_rope_1, self.freqs_cis) # Corrected

        # Recombine key parts for branch 1
        k_recombined_1 = torch.empty((B, self.n_head, T, self.rope_head_dim + self.nope_head_dim),
                                   device=x.device, dtype=x.dtype)
        k_recombined_1[:, :, :, :self.nope_head_dim] = key_nope_1
        k_recombined_1[:, :, :, self.nope_head_dim:] = k_rope_1

        # === Branch 2: Token Selection Branch (NSA) ===
        # Compute importance scores
        importance_scores = self.importance_scorer(x)

        # Select important tokens
        selected_tokens, selected_indices = self._select_important_tokens(x, importance_scores)

        # Get KV for selected tokens
        B, S, _ = selected_tokens.size()  # S is the number of selected tokens
        k_selected = self.selection_k(selected_tokens)
        v_selected = self.selection_v(selected_tokens)

        # Reshape
        k_selected = k_selected.view(B, S, self.n_head, self.rope_head_dim + self.nope_head_dim).transpose(1, 2)
        v_selected = v_selected.view(B, S, self.n_head, self.v_head_dim).transpose(1, 2)

        # Apply RoPE (only to the RoPE portion)
        k_selected_rope = k_selected[:, :, :, self.nope_head_dim:]
        k_selected_nope = k_selected[:, :, :, :self.nope_head_dim]
        # Corrected: pass k_selected_rope for both x and y
        _, k_selected_rope = apply_rope(k_selected_rope, k_selected_rope, self.freqs_cis)


        # Recombine
        k_selected[:, :, :, self.nope_head_dim:] = k_selected_rope
        k_selected[:, :, :, :self.nope_head_dim] = k_selected_nope  # make sure we add the nope back!

        # === Branch 3: Sliding Window Branch (NSA) ===
        window_tokens = self._get_sliding_window_tokens(x)
        B, W, _ = window_tokens.size()  # W is window size

        k_window = self.window_k(window_tokens)
        v_window = self.window_v(window_tokens)

        # Reshape
        k_window = k_window.view(B, W, self.n_head, self.rope_head_dim + self.nope_head_dim).transpose(1, 2)
        v_window = v_window.view(B, W, self.n_head, self.v_head_dim).transpose(1, 2)

        # Apply RoPE (only to the RoPE portion)
        k_window_rope = k_window[:, :, :, self.nope_head_dim:]
        k_window_nope = k_window[:, :, :, :self.nope_head_dim]
         # Corrected: pass k_window_rope for both x and y
        _, k_window_rope = apply_rope(k_window_rope, k_window_rope, self.freqs_cis)


        # Recombine
        k_window[:, :, :, self.nope_head_dim:] = k_window_rope
        k_window[:, :, :, :self.nope_head_dim] = k_window_nope

        # === Compute attention for each branch and blend results ===
        if self.training:
            self.cache_filled = 0

            # Branch 1: Original MLA attention with full sequence
            output_1 = F.scaled_dot_product_attention(
                q_recombined, k_recombined_1, value_1,
                is_causal=True, dropout_p=self.dropout
            )

            # Branch 2: Attention with selected tokens
            # For selected tokens, we need to compute attention differently
            # as they might not be in sequence order
            output_2 = F.scaled_dot_product_attention(
                q_recombined, k_selected, v_selected,
                is_causal=False, dropout_p=self.dropout  # Non-causal for selected tokens
            )

            # Branch 3: Sliding window attention
            output_3 = F.scaled_dot_product_attention(
                q_recombined, k_window, v_window,
                is_causal=True, dropout_p=self.dropout
            )

            # Blend outputs using branch weights
            blended_output = (
                output_1 * branch_weights[:, 0].view(B, 1, 1, 1) +
                output_2 * branch_weights[:, 1].view(B, 1, 1, 1) +
                output_3 * branch_weights[:, 2].view(B, 1, 1, 1)
            )

        else:
            # Inference mode with KV caching
            if self.k_cache is None or self.v_cache is None or self.k_cache.size(0) != B:
                self.k_cache = torch.zeros(
                    B, self.n_head, self.ctx_len, self.rope_head_dim + self.nope_head_dim,
                    device=self.device, dtype=x.dtype
                )
                self.v_cache = torch.zeros(
                    B, self.n_head, self.ctx_len, self.v_head_dim,
                    device=self.device, dtype=x.dtype
                )
                self.cache_filled = 0

            # Update cache with new tokens
            new_cache_filled = min(self.cache_filled + T, self.ctx_len)

            # Branch 1: Update cache
            k_to_cache = k_recombined_1[:, :, :new_cache_filled - self.cache_filled]
            v_to_cache = value_1[:, :, :new_cache_filled - self.cache_filled]

            self.k_cache[:, :, self.cache_filled:new_cache_filled] = k_to_cache
            self.v_cache[:, :, self.cache_filled:new_cache_filled] = v_to_cache
            self.cache_filled = new_cache_filled

            # Get cached KVs
            k1 = self.k_cache[:, :, :self.cache_filled]
            v1 = self.v_cache[:, :, :self.cache_filled]

            # Branch 1: Attention with cached KVs
            output_1 = F.scaled_dot_product_attention(
                q_recombined, k1, v1, is_causal=True, dropout_p=0
            )

            # Branch 2: Attention with selected tokens (from current sequence)
            output_2 = F.scaled_dot_product_attention(
                q_recombined, k_selected, v_selected, is_causal=False, dropout_p=0
            )

            # Branch 3: Sliding window attention
            current_pos = self.cache_filled - 1  # Current position for window centering
            output_3 = F.scaled_dot_product_attention(
                q_recombined, k_window, v_window, is_causal=True, dropout_p=0
            )

            # Blend outputs using branch weights
            blended_output = (
                output_1 * branch_weights[:, 0].view(B, 1, 1, 1) +
                output_2 * branch_weights[:, 1].view(B, 1, 1, 1) +
                output_3 * branch_weights[:, 2].view(B, 1, 1, 1)
            )

        # Final processing
        output = blended_output.transpose(1, 2).contiguous().view(B, T, self.value_dim)
        output = self.proj(output)
        output = self.res_dropout(output)

        return output, reference_point

# Reg MLP 

class MLP(nn.Module):
    def __init__(self, curvature):
        super().__init__()
        n_embd = config['n_embd']
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(config['dropout'])
        # Initialize curvature parameter randomly
        self.c = curvature

    def forward(self, x, reference_point=None):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        
        # Apply expmap to map from tangent space back to hyperbolic space
        if reference_point is None:
            reference_point = calculate_reference_point(x)
        else:
            # Ensure reference point has same batch shape as x
            if reference_point.shape[0] != x.shape[0]:
                # This handles cases where x was reshaped but reference_point wasn't
                reference_point = reference_point.reshape(-1, reference_point.shape[-1])
        
        # Apply exponential map to map back to hyperbolic space
        x = expmap(reference_point, x, self.c)
        
        return x

# DS-MoE Layer

class UnitCenteredNoise(nn.Module):
    def __init__(self, scaling=0.02):
        super(UnitCenteredNoise, self).__init__()
        self.scaling = scaling
        self.base = 1 - (scaling * 0.5)

    def forward(self, x):
        if self.training:
            noise = torch.rand(x.size(), device=x.device, dtype=x.dtype)
            noise_centered = (noise * self.scaling) + self.base
            return x * noise_centered
        else:
            return x

class DSMoE(nn.Module):

    def __init__(self, index, num_exp=4):
        super().__init__()
        self.hidden_dim = config['n_embd'] * 2  # was 4, had to shrink by 1/2
        self.num_experts = config["n_experts"]
        self.num_exp = num_exp
        self.moe_scaling = config["init_moe_scaling"]
        # Get the curvature parameter from parent block
        self.c = nn.Parameter(torch.rand(1))  # Random initialization for curvature
        self.c.requires_grad = True
        
        # Create MLP experts with curvature parameter
        self.experts = nn.ModuleList([MLP(self.c) for _ in range(self.num_experts)])
        self.gate = nn.Sequential(
            nn.Linear(config['n_embd'], self.num_experts - 1),  # exclude shared expert
            UnitCenteredNoise(scaling=0.02),
            nn.Softmax(dim=-1)
        )
        # Initialize expert bias (excluding the shared expert)
        self.expert_bias = nn.Parameter(torch.zeros(self.num_experts - 1), requires_grad=False)


    def forward(self, x, reference_point=None):
        b, t, c = x.shape
        x_flat = x.reshape(b * t, c)
        
        # Reshape reference point to match flattened input
        if reference_point is not None:
            # Reshape reference_point from [b, t, c] to [b*t, c]
            reference_point_flat = reference_point.reshape(b * t, c)
        else:
            reference_point_flat = None

        gate_val_continuous = self.gate(x_flat)

        # Apply expert bias *before* topk
        biased_gate_vals = gate_val_continuous + self.expert_bias

        # get top-(num_exp-1) experts excluding the first one
        gate_vals, gate_val_indices = torch.topk(biased_gate_vals, self.num_exp - 1, dim=-1)
        gate_vals = gate_vals / gate_vals.sum(dim=-1, keepdim=True)  # normalize

        # prepend the shared expert (index 0) - Corrected handling
        shared_expert_weight = torch.ones_like(gate_vals[:, :1]) / self.num_exp
        gate_vals = torch.cat([shared_expert_weight, gate_vals * (self.num_exp - 1) / self.num_exp], dim=-1)
        gate_val_indices = torch.cat([torch.zeros_like(gate_val_indices[:, :1]), gate_val_indices + 1], dim=-1)

        # process all experts once (fully static)
        # Pass flattened reference point to all experts
        expert_outputs = torch.stack([expert(x_flat, reference_point_flat) for expert in self.experts], dim=0)  # [num_experts, b*t, c]

        # create routing weights matrix (one-hot * gate values)
        router_weights = torch.zeros(x_flat.size(0), self.num_experts, device=x.device)
        for i in range(self.num_exp):
            idx = gate_val_indices[:, i:i+1]  # [b*t, 1]
            val = gate_vals[:, i:i+1]  # [b*t, 1]
            router_weights.scatter_add_(1, idx, val)

        # apply routing weights to expert outputs
        weighted_outputs = expert_outputs * router_weights.transpose(0, 1).unsqueeze(-1)  # [num_experts, b*t, c]
        output = weighted_outputs.sum(dim=0)  # [b*t, c]

        # Return both the output and the router_weights
        return output.reshape(b, t, c), router_weights

class Block(nn.Module):
    def __init__(self, index):
        super().__init__()
        n_embd = config['n_embd']
        # Initialize curvature parameter randomly for hyperbolic operations
        self.c = nn.Parameter(torch.rand(1))  # Random initialization for curvature
        self.c.requires_grad = True
        
        self.attn = Attn(self.c)
        self.ffn_type = config['type'][index]

        if self.ffn_type == "mlp":
            self.ffn = MLP(self.c)
        elif self.ffn_type == "moe":
            self.ffn = DSMoE(index)
        else:
            raise ValueError(f"Invalid layer type: {self.ffn_type}")

        self.rm1 = nn.RMSNorm(n_embd)
        self.rm2 = nn.RMSNorm(n_embd)

    def forward(self, x):
        # Run attention and get both output and reference point
        attn_output, reference_point = self.attn(self.rm1(x))
        x = x + attn_output
        
        if self.ffn_type == "moe":
            x_ffn, router_weights = self.ffn(self.rm2(x), reference_point)
            return x + x_ffn, router_weights
            
        else:
            # Pass the reference point to MLP
            x_ffn = self.ffn(self.rm2(x), reference_point)
            return x + x_ffn, None # no MoE, no route weights

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config['vocab_size'], config['n_embd'])
        self.position_embedding_table = nn.Embedding(config['ctx_len'], config['n_embd'])
        self.blocks = nn.Sequential(*[Block(i) for i in range(config['n_layer'])])
        self.rm_f = nn.RMSNorm(config['n_embd'])
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'])
        self.token_embedding_table.weight = self.lm_head.weight
        self.apply(self._init_weights)
        self.total_params = sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx).clone()
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.config['device']))
        x = tok_emb + pos_emb

        all_router_weights = []  # Collect router_weights across MoEs

        for block in self.blocks:
            x, router_weights = block(x)  # Get router_weights from Block
            if router_weights is not None:
                all_router_weights.append(router_weights)

        x = self.rm_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets) 

        return logits, loss, all_router_weights

    def generate(self, idx, max_new_tokens): # fix temp, topk later
        for _ in range(max_new_tokens):
            
            idx_cond = idx[:, -config['ctx_len']:] # crop to ctx len
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] # use final logits
            probs = F.softmax(logits, dim=-1) # discrete with Softmax

            idx_next = torch.multinomial(probs, num_samples=1) # new token
            idx = torch.cat((idx, idx_next), dim=1) # add token 

        total_size_gb = 0 # flex that kv cache

        for block in self.blocks:

            if hasattr(block.attn, 'k_cache') and block.attn.k_cache is not None:
                # k_cache
                size_bytes = block.attn.k_cache.numel() * block.attn.k_cache.element_size()
                size_gb = size_bytes / (1024**3)
                total_size_gb += size_gb
                # v_cache
                size_bytes = block.attn.v_cache.numel() * block.attn.v_cache.element_size()
                size_gb = size_bytes / (1024**3)
                total_size_gb += size_gb

        return idx, total_size_gb


    def configure_optimizers(self, weight_decay, learning_rate, betas, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Separate parameters for different learning rates
        decay_params = []
        nodecay_params = []
        param_groups = []

        for pn, p in param_dict.items():
            if p.dim() >= 2:
                decay_params.append(p)
                param_groups.append({'params': [p], 'weight_decay': weight_decay, 'lr': learning_rate, 'name': pn})
            else:
                nodecay_params.append(p)
                param_groups.append({'params': [p], 'weight_decay': 0.0, 'lr': learning_rate, 'name': pn})

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, betas=betas, **extra_args) # add Muon
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def update_expert_biases(self, all_router_weights, update_rate):

        with torch.no_grad():
            # Iterate through the blocks and find MoE layers

            j = 0 

            for block in self.blocks:
                if isinstance(block.ffn, DSMoE):

                    router_weights = all_router_weights[j]
                    j += 1

                    c_i = router_weights[:, 1:].sum(dim=0)  # Exclude shared expert, calculate expert load
                    total_routed_tokens = c_i.sum()
                    c_i_bar = total_routed_tokens / (block.ffn.num_experts - 1) # avg load
                    e_i = c_i - c_i_bar # Load violation error

                    block.ffn.expert_bias.add_(update_rate * torch.sign(e_i)) # update step

    def estimate_mfu(self, params, fwdbwd_per_iter, dt):
        N = params
        L, H, Q, T = config['n_layer'], config['n_head'], config['n_embd']//config['n_head'], config['ctx_len']
        flops_per_token = 6*N + 12*L*H*Q*T # fix recalc for MoE
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 65e12 # 65 tflops for a t4
        mfu = flops_achieved / flops_promised
        return mfu
