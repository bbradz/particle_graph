import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from config import Config
from Token2Model.check import NUM_CHECKS

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, seq_len, d_model, device):
    """Apply rotary position embedding to query and key tensors."""
    # Create position indices
    position = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(-1)
    
    # Create frequency tensor
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * 
                        -(math.log(10000.0) / d_model))
    
    # Create sinusoidal embeddings
    pe = torch.zeros(1, seq_len, d_model, device=device)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    
    # Apply rotation to q and k
    cos = pe[..., 1::2].repeat_interleave(2, dim=-1)
    sin = pe[..., ::2].repeat_interleave(2, dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with RoPE applied to queries and keys."""
    
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        # Align with nn.MultiheadAttention interface expectations
        self.batch_first = True
        self._qkv_same_embed_dim = True
        self.in_proj_bias = None  # We use separate linear layers without bias
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query,
        key,
        value,
        attn_mask=None,
        key_padding_mask=None,
        need_weights=False,
        average_attn_weights=False,
        is_causal=False,
    ):
        batch_size, seq_len, d_model = query.size()
        
        # Linear projections
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)
        
        # Reshape for multi-head attention: (B, H, S, Dk)
        q = q.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        
        # Apply RoPE to queries and keys
        q, k = apply_rope(q, k, seq_len, self.d_k, query.device)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply attn_mask if provided (bool True means mask; float additive supported)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            else:
                # additive mask broadcast to (B, H, S, S)
                scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply key padding mask if provided: shape (B, S) with True to mask
        if key_padding_mask is not None:
            # Convert to boolean if needed
            if key_padding_mask.dtype != torch.bool:
                key_padding_mask = key_padding_mask.bool()
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Concatenate heads -> (B, S, D)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Final linear projection
        output = self.w_o(context)
        
        # Match nn.MultiheadAttention return type
        if need_weights:
            if average_attn_weights:
                # average over heads
                avg_weights = attn_weights.mean(dim=1)
                return output, avg_weights
            return output, attn_weights
        return output, None

class RoPETransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Transformer encoder layer with RoPE attention."""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", batch_first=True):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, batch_first)
        # Replace the default self-attention with RoPE attention
        self.self_attn = RoPEMultiheadAttention(d_model, nhead, dropout)

def make_policy(config: Config, tokenizer) -> nn.Module:
    """Factory function to create the policy model based on the configuration."""
    if config.MODEL_TYPE == 'hf':
        return HuggingFacePolicy(config, tokenizer)
    elif config.MODEL_TYPE == 'transformer':
        return SimpleTransformerPolicy(config, tokenizer)
    else:
        raise ValueError(f"Unknown MODEL_TYPE in config: {config.MODEL_TYPE}")

class HuggingFacePolicy(nn.Module):
    """A wrapper for Hugging Face Causal LM models to serve as the policy."""
    def __init__(self, config: Config, tokenizer):
        super().__init__()
        self.config = config

        dtype_map = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32
        }
        torch_dtype = dtype_map.get(config.TORCH_DTYPE)

        quantization_config = None
        if config.USE_QUANTIZATION and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
            )
            print("Using 4-bit quantization.")

        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            config.HF_MODEL_NAME,
            trust_remote_code=config.TRUST_REMOTE_CODE,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            device_map='auto'
        )

        self.model.resize_token_embeddings(len(tokenizer))
        
        hidden_size = self.model.config.hidden_size
        
        # Multi-headed critic: output vector of check scores instead of single scalar
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size // 2, NUM_CHECKS),
            nn.LeakyReLU()
        )
        self.value_head.to(self.model.device, dtype=torch_dtype)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor = None,
        past_key_values: tuple = None
    ) -> CausalLMOutput:
        """Standard forward pass that now accepts and uses the past_key_values cache."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True
        )

    def value(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        """Computes the vector of check scores from the final hidden states."""
        return self.value_head(hidden_states.to(self.value_head[0].weight.device, dtype=self.value_head[0].weight.dtype))


class SimpleTransformerPolicy(nn.Module):
    """A simple PyTorch transformer implementation for the policy."""
    
    def __init__(self, config: Config, tokenizer):
        super().__init__()
        self.config = config
        
        # Update vocab size from tokenizer
        vocab_size = len(tokenizer)
        config.TRANSFORMER_VOCAB_SIZE = vocab_size
        
        # Model parameters
        self.vocab_size = vocab_size
        self.d_model = config.TRANSFORMER_D_MODEL
        self.nhead = config.TRANSFORMER_NHEAD
        self.num_layers = config.TRANSFORMER_NUM_LAYERS
        self.dim_feedforward = config.TRANSFORMER_DIM_FEEDFORWARD
        self.dropout = config.TRANSFORMER_DROPOUT
        self.max_len = config.TRANSFORMER_MAX_LEN
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, self.d_model)
        
        # Transformer layers with RoPE
        encoder_layer = RoPETransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(self.d_model, vocab_size)
        
        # Value head for critic
        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.LeakyReLU(),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.LayerNorm(self.d_model // 2),
            nn.LeakyReLU(),
            nn.Linear(self.d_model // 2, NUM_CHECKS),
            nn.LeakyReLU()
        )
        
        # Initialize weights
        self._init_weights()
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        print(f"Initialized SimpleTransformerPolicy with RoPE, {self.num_layers} layers, {self.nhead} heads, d_model={self.d_model}")
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask.to(self.device)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor = None,
        past_key_values: tuple = None
    ) -> CausalLMOutput:
        """
        Forward pass for the transformer.
        Note: past_key_values is ignored for simplicity in this implementation.
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding(input_ids)  # (batch_size, seq_len, d_model)
        
        # Create causal mask
        causal_mask = self._create_causal_mask(seq_len)
        
        # Prepare attention mask for transformer
        # PyTorch transformer expects src_key_padding_mask where True means "ignore this position"
        src_key_padding_mask = None
        if attention_mask is not None:
            # Convert to boolean first
            attention_mask = attention_mask.bool()
            
            # Ensure attention_mask has the same sequence length as input
            if attention_mask.size(1) != seq_len:
                # If mask is shorter, pad with True (attend to new positions)
                if attention_mask.size(1) < seq_len:
                    padding = torch.ones(batch_size, seq_len - attention_mask.size(1), 
                                        dtype=torch.bool, device=attention_mask.device)
                    attention_mask = torch.cat([attention_mask, padding], dim=1)
                # If mask is longer, truncate
                else:
                    attention_mask = attention_mask[:, :seq_len]
            
            # Invert: True in attention_mask means "attend", 
            # but src_key_padding_mask expects True to mean "ignore"
            src_key_padding_mask = ~attention_mask
        
        # Transformer forward pass with proper mask handling
        hidden_states = self.transformer(
            x, 
            src_key_padding_mask=src_key_padding_mask,
            mask=causal_mask
        )
        
        # Output logits
        logits = self.output_projection(hidden_states)
        
        # Create CausalLMOutput-like object
        class SimpleCausalLMOutput:
            def __init__(self, logits, hidden_states):
                self.logits = logits
                self.hidden_states = [hidden_states]  # List to match HuggingFace format
                self.past_key_values = None  # Not implemented for simplicity
        
        return SimpleCausalLMOutput(logits, hidden_states)
    
    def value(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        """Computes the vector of check scores from the hidden states."""
        return self.value_head(hidden_states)