# transformer.py
"""
TransformerPolicy: Decoder-only transformer with KV caching for efficient autoregressive generation.

Key Features:
- Standard transformer architecture with token/positional embeddings
- Actor-critic heads (logits + value output)
- KV caching for efficient sequential inference
- Supports both training (full sequence) and generation (token-by-token) modes

KV Caching: Stores and reuses attention keys/values from previous tokens during generation,
avoiding redundant computation and significantly improving inference speed.

Usage: Initialize with vocab_size, d_model, nhead, layers, max_T parameters.
Forward pass accepts optional cache and returns (logits, value, new_cache).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
import logging
from datetime import datetime

class TransformerPolicy(nn.Module):
    """
    A decoder-only Transformer model with actor-critic heads and KV caching.

    This policy network uses a standard transformer architecture. It supports
    efficient autoregressive generation via a KV cache, which stores the
    key and value tensors from previous attention computations.

    Key Features:
    - Standard token and positional embeddings.
    - Multi-layer transformer decoder architecture.
    - Dual-head output for actor (policy logits) and critic (value function).
    - KV Caching: For fast generation, avoiding re-computation for past tokens.
    - Debug mode for verbose logging of tensor shapes and operations.
    """
    def __init__(self, vocab: int, *, d_model: int = 16, nhead: int = 4, layers: int = 3, max_T: int = 400, debug: bool = False):
        """
        Initializes the TransformerPolicy.

        Args:
            vocab (int): The size of the vocabulary.
            d_model (int): The dimensionality of the model's embeddings and layers.
            nhead (int): The number of attention heads.
            layers (int): The number of transformer encoder layers.
            max_T (int): The maximum sequence length.
            debug (bool): If True, enables verbose logging.
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.layers = layers
        self.debug = debug
        # FIXED: Removed self.logger initialization, as Logger objects are not JIT-compatible.

        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_T, d_model)
        self.register_buffer("pos", torch.arange(max_T))

        # Create transformer layers
        self.encoder_layers = nn.ModuleList()
        for _ in range(layers):
            self.encoder_layers.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(d_model),
                'attn': nn.MultiheadAttention(d_model, nhead, batch_first=True),
                'norm2': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model)
                )
            }))
        
        self.final_norm = nn.LayerNorm(d_model)
        self.actor = nn.Linear(d_model, vocab) # Policy head
        self.value = nn.Linear(d_model, 1)    # Value head

    def forward(self,
                seq: torch.Tensor,
                cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
               ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Performs a forward pass with optional KV caching.

        Args:
            seq: The input token sequence.
                 - Shape for training (no cache): (batch_size, sequence_length)
                 - Shape for generation (with cache): (batch_size, 1)
            cache: A list of tuples (one per layer), where each tuple contains
                   the 'k' and 'v' tensors from previous steps.

        Returns:
            A tuple containing:
            - logits (torch.Tensor): Output logits for the next token prediction.
            - value (torch.Tensor): The value function output.
            - new_cache (List[Tuple[torch.Tensor, torch.Tensor]]): The updated KV cache.
        """
        B, T = seq.size()
        pos_offset = 0
        if cache is not None:
            # Access the key tensor (at index 0) from the first layer's cache tuple
            pos_offset = cache[0][0].size(1)

        # --- Input & Embeddings ---
        if pos_offset + T > self.pos_emb.num_embeddings:
            raise ValueError(f"Sequence length {pos_offset + T} exceeds max_T {self.pos_emb.num_embeddings}")
        
        pos_emb = self.pos_emb(self.pos[pos_offset : pos_offset + T])
        tok_emb = self.tok_emb(seq)
        x = tok_emb + pos_emb

        # --- Attention Mask ---
        # A causal mask is only needed during training (T > 1).
        # We hint to JIT that attn_mask can be None or a Tensor.
        attn_mask: Optional[torch.Tensor] = None
        if T > 1:
            # Manually create the causal mask instead of using the static method.
            attn_mask = torch.triu(torch.full((T, T), float('-inf'), device=seq.device, dtype=x.dtype), diagonal=1)

        # --- Transformer Layers ---
        # Explicitly type the empty list for the JIT compiler.
        new_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, layer in enumerate(self.encoder_layers):
            x_norm = layer['norm1'](x)
            
            # For self-attention, q, k, and v are all derived from the same source
            q = k = v = x_norm
            
            # Prepend cached keys/values if available (for generation)
            if cache is not None:
                k = torch.cat([cache[i][0], k], dim=1) # cache[i][0] is the key
                v = torch.cat([cache[i][1], v], dim=1) # cache[i][1] is the value
            
            # --- Multi-Head Attention ---
            attn_output, _ = layer['attn'](q, k, v, attn_mask=attn_mask, need_weights=False)
            x = x + attn_output # Add & Norm (residual connection)
            
            # --- Feed-Forward Network ---
            x_norm2 = layer['norm2'](x)
            x_ffn = layer['ffn'](x_norm2)
            x = x + x_ffn # Add & Norm (residual connection)
            
            # Store the updated k/v for the next generation step as a tuple
            new_cache.append((k.detach(), v.detach()))

        # --- Final Output ---
        h = self.final_norm(x)
        logits = self.actor(h)
        value = self.value(h).squeeze(-1)
        
        # FIXED: Removed the self.logger call, as it is not JIT-compatible.
        
        return logits, value, new_cache