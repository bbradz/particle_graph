from __future__ import annotations
import os, sys, json, tempfile, argparse, time, re, random, logging, copy
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import psutil  # Add psutil for system memory tracking
import hashlib  # Add hashlib for sequence hashing

# Import token2fr functionality
from token2fr import tokens_to_fr, ModelTester

# Import SM tokens for validation testing
try:
    from old.t import SM
    SM_TOKENS_AVAILABLE = True
except ImportError:
    SM_TOKENS_AVAILABLE = False
    print("Warning: SM tokens not available for validation testing")

plt.style.use('./rose-pine-dawn.mplstyle')

# --------------------------------------------------------------------------------------
# Global flags / seeds
# --------------------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

try:
    torch.set_float32_matmul_precision('high')  # A100/H100 fast GEMM
except AttributeError:
    pass

DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE     = torch.float32  # Changed from float16 to float32 for full precision
ITYPE     = torch.int32
PAD_VALUE = -1
EPS       = 1e-8

INTERACTION_BIT = 0b1000
FIELD_BIT       = 0b0100
PARTICLE_BIT    = 0b0010

# Print initial system memory state
print("=== INITIAL SYSTEM MEMORY STATE ===")
try:
    import psutil
    system_memory = psutil.virtual_memory()
    print(f"System Memory: {system_memory.total / (1024**3):.1f} GB total, {system_memory.used / (1024**3):.1f} GB used ({system_memory.percent:.1f}%)")
    print(f"Available Memory: {system_memory.available / (1024**3):.1f} GB")
    
    process = psutil.Process()
    process_memory_mb = process.memory_info().rss / (1024**2)
    print(f"Process Memory: {process_memory_mb:.1f} MB")
    
    if torch.cuda.is_available():
        gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Memory: {gpu_total_gb:.1f} GB total")
    else:
        print("GPU Memory: Not available (CPU only)")
    print("=" * 40)
except ImportError:
    print("psutil not available - cannot show memory information")
    print("=" * 40)

# Memory tracking utility functions
def get_memory_usage():
    """Get current memory usage information."""
    # System memory
    system_memory = psutil.virtual_memory()
    system_total_gb = system_memory.total / (1024**3)
    system_used_gb = system_memory.used / (1024**3)
    system_available_gb = system_memory.available / (1024**3)
    system_percent_used = system_memory.percent
    
    # Process memory
    process = psutil.Process()
    process_memory_mb = process.memory_info().rss / (1024**2)
    
    # GPU memory (if available)
    gpu_memory_info = {}
    if torch.cuda.is_available():
        gpu_memory_info = {
            'total_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
            'allocated_gb': torch.cuda.memory_allocated(0) / (1024**3),
            'cached_gb': torch.cuda.memory_reserved(0) / (1024**3),
            'free_gb': (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / (1024**3)
        }
    
    return {
        'system': {
            'total_gb': system_total_gb,
            'used_gb': system_used_gb,
            'available_gb': system_available_gb,
            'percent_used': system_percent_used
        },
        'process_mb': process_memory_mb,
        'gpu': gpu_memory_info
    }

def print_memory_usage(logger, stage="Initial"):
    """Print current memory usage information."""
    memory_info = get_memory_usage()
    
    logger.info(f"=== {stage} MEMORY USAGE ===")
    logger.info(f"System Memory:")
    logger.info(f"  Total: {memory_info['system']['total_gb']:.1f} GB")
    logger.info(f"  Used: {memory_info['system']['used_gb']:.1f} GB ({memory_info['system']['percent_used']:.1f}%)")
    logger.info(f"  Available: {memory_info['system']['available_gb']:.1f} GB")
    logger.info(f"Process Memory: {memory_info['process_mb']:.1f} MB")
    
    if memory_info['gpu']:
        logger.info(f"GPU Memory:")
        logger.info(f"  Total: {memory_info['gpu']['total_gb']:.1f} GB")
        logger.info(f"  Allocated: {memory_info['gpu']['allocated_gb']:.1f} GB")
        logger.info(f"  Cached: {memory_info['gpu']['cached_gb']:.1f} GB")
        logger.info(f"  Free: {memory_info['gpu']['free_gb']:.1f} GB")
    else:
        logger.info("GPU Memory: Not available (CPU only)")
    
    logger.info("=" * 40)

# ======================================================================================
# 1. PPO hyper‑parameters  (unchanged – see original file for comments)
# ======================================================================================
@dataclass
class PPOConfig:
    run_name:                           str   = f"ppo_blueprint_{int(time.time())}"
    log_dir:                            str   = "runs"
    total_steps:                        int   = 2_000
    batch_size:                         int   = 128
    seq_len:                            int   = 400

    learning_rate:                      float = 3e-5 
    ppo_epochs:                         int   = 2     
    vf_epochs:                          int   = 32 
    num_minibatches:                    int   = 8
    gamma:                              float = 0.95  
    gae_lambda:                         float = 0.9   
    clip_coef:                          float = 0.2
    ent_coef:                           float = 8.0   
    vf_coef:                            float = 0.8   
    sup_coef:                           float = 0.01  
    max_grad_norm:                      float = 0.5

    alive_r:                            float = 0.10  
    len_target:                         int   = 329
    len_sigma:                          float = 10.0  
    particle_bonus_weight:              float = 0.5
    field_bonus_weight:                 float = 2.0
    terminal_bonus_weight:              float = 10.0  

    # Removed validation bonus related parameters:
    # per_check_alpha:                    float = 0.2
    # per_check_min_weight:               float = 0.1
    # per_check_max_weight:               float = 5.0
    # check_weights:                      Optional[torch.Tensor] = None 

    balanced_coef:                      float = 0.5

    track_diagnostics:                  bool  = True
    kl_target:                          float = 0.01
    kl_coef:                            float = 0.3
    advantage_normalization:            str   = "standard" 
    advantage_clip_threshold:           float = 15.0  
    value_lr_multiplier:                float = 1.0   
    advantage_normalize_only_alive:     bool  = True
    advantage_robust_normalization:     bool  = False
    enable_gradient_clipping:           bool  = True
    gradient_clip_norm:                 float = 15.0
    gradient_clip_adaptive:             bool  = False
    policy_lr:                          float = 3e-4 
    pg_loss_scale:                      float = 2.0   
    min_policy_change_threshold:        float = 1e-4

    # Checkpoint parameters
    checkpoint_mode:                    str   = "new" 
    checkpoint_path:                    str   = ""    
    checkpoint_step:                    int   = 0     
    checkpoint_interval:                int   = 10   

    # Weight regularization parameters for stability
    weight_decay:                       float = 1e-4  
    max_grad_norm:                      float = 1.0   
    gradient_clip_norm:                 float = 5.0

    plot_interval:                      int   = 20
    
    # --- NEW: Novelty Reward Parameters ---
    novelty_reward_weight:              float = 0.5  # Weight for the novelty bonus
    novelty_buffer_size:                int   = 1000 # Number of unique sequences to remember

    minibatch_size:                     int   = field(init=False)
    def __post_init__(self):
        self.minibatch_size = max(1, self.batch_size // self.num_minibatches)

# ======================================================================================
# 2. Tiny Transformer policy – *with working KV‑cache*
# ======================================================================================
class DummyLogger:
    def __init__(self):
        self._logger = logging.getLogger("ppo-blueprint")
        if not self._logger.handlers:
            h = logging.StreamHandler(sys.stdout)
            h.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(h)
            self._logger.setLevel(logging.INFO)
    def info(self, msg):
        self._logger.info(msg)
    def warning(self, msg):
        self._logger.warning(msg)
    def debug(self, msg):
        self._logger.debug(msg)

# ────────────────────────────────────────────────────────────────────────────
# Causal Transformer with KV‑cache support (single‑headed impl for clarity)
# ────────────────────────────────────────────────────────────────────────────
class CausalTransformerLayer(nn.Module):
    """Transformer layer that supports KV‑caching for autoregressive decoding.
    
    Note: This implementation caches layer outputs rather than raw key/value projections
    for memory efficiency reasons. While a standard KV-cache would store the raw K/V
    projections from nn.Linear layers before MultiheadAttention, this approach caches
    the final layer outputs, which may consume more memory but simplifies the implementation.
    """
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        
        # Use more stable attention with proper initialization
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(d_model, eps=1e-5)
        
        # Replace GELU with ReLU for better stability
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff), 
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), 
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(d_model, eps=1e-5)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_uniform_(module.weight, gain=0.5)  # Reduced gain for stability
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                # Initialize LayerNorm properly
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # Initialize embeddings with smaller values
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,              # (B, T, D) – **query** is always *current* tokens
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  # both (B, T_prev, D)
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, D = x.shape

        # --- Build causal mask only for *new* tokens ---
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device, dtype=x.dtype)

        if use_cache and past_key_value and past_key_value[0] is not None:
            # Concatenate along the sequence dimension
            past_k, past_v = past_key_value
            # NOTE: This current implementation of KV-cache stores the *layer outputs* from previous steps.
            # A more memory-efficient/standard KV-cache would store the raw K/V projections
            # (e.g., from an nn.Linear layer before MultiheadAttention).
            # This current approach works but might consume more memory than strictly necessary
            # if the hidden dimension (D_MODEL) is very large.
            k_cat = torch.cat([past_k, x], dim=1)   # (B, T_prev+T, D)
            v_cat = torch.cat([past_v, x], dim=1)
            # The query is only the *new* tokens
            query = x
            key   = k_cat
            value = v_cat
            # Need an expanded mask: allow query to attend to (T_prev+T)
            T_total = k_cat.size(1)
            tgt_len = T
            causal_mask = nn.Transformer.generate_square_subsequent_mask(T_total, device=x.device, dtype=x.dtype)
            causal_mask = causal_mask[-tgt_len:] # keep last rows (for new query positions)
        else:
            query = key = value = x
            past_k = past_v = None

        # Add residual connection with proper scaling
        attn_out, _ = self.self_attn(query, key, value, attn_mask=causal_mask, need_weights=False)
        x = self.self_attn_norm(x + 0.1 * attn_out)  # Scale down attention output for stability
        
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + 0.1 * ffn_out)  # Scale down FFN output for stability

        # Build new cache: concatenate past (if any) with current layer output
        if use_cache:
            new_k = torch.cat([past_k, x], dim=1) if past_k is not None else x
            new_v = torch.cat([past_v, x], dim=1) if past_v is not None else x
        else:
            new_k = new_v = None
        return x, (new_k, new_v)

class CachedTransformerEncoder(nn.Module):
    """Stack of causal transformer layers with KV‑cache handling."""
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            CausalTransformerLayer(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        if cache is None:
            cache = [ (None, None) for _ in range(len(self.layers)) ]
        new_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        out = x
        for layer, past in zip(self.layers, cache):
            out, (k_new, v_new) = layer(out, use_cache=use_cache, past_key_value=past)
            new_cache.append((k_new, v_new))
        return self.norm(out), new_cache

class DummyTransformerPolicy(nn.Module):
    """Transformer policy/value network that now exploits the KV cache."""
    def __init__(self, vocab_size: int, d_model: int, max_T: int, num_layers: int = 4, nhead: int = 16, dropout: float = 0.2):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.vocab_size = vocab_size
        self.d_model = d_model # Add this line to store d_model as an attribute
        
        # Initialize embeddings with smaller values for stability
        self.embed  = nn.Embedding(vocab_size, d_model)
        self.posemb = nn.Embedding(max_T,     d_model)
        
        # Use more stable encoder with proper initialization
        self.encoder = CachedTransformerEncoder(d_model, nhead, num_layers, d_model*4, dropout)
        
        # Add weight regularization to policy head
        self.policy_head = nn.Linear(d_model, vocab_size)
        
        # Use more stable value encoder
        self.value_encoder = CachedTransformerEncoder(d_model, nhead, 1, d_model*4, dropout)
        self.value_head_intermediate = nn.Linear(d_model, vocab_size)
        self.value_head_final = nn.Linear(vocab_size, 1)
        
        # Initialize all weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with stable initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use smaller initialization for stability
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # Initialize embeddings with very small values
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
    
    def get_weight_regularization_loss(self, weight_decay: float = 1e-4):
        """Compute L2 weight regularization loss."""
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)
        return weight_decay * l2_loss

    # ---------- training forward (full sequence) ----------
    def forward(self, tok_ids: torch.Tensor, cache: Optional[Any]=None):
        B, T = tok_ids.shape
        # Changed dtype to torch.long as nn.Embedding expects long
        pos = torch.arange(T, device=tok_ids.device, dtype=torch.long).unsqueeze(0).expand(B, T)
        h = self.embed(tok_ids) + self.posemb(pos)
        h, _ = self.encoder(h, use_cache=False)
        logits = self.policy_head(h)
        
        # Add debugging for inf/nan values
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"WARNING: Found inf/nan in logits in forward pass!")
            print(f"Logits shape: {logits.shape}")
            print(f"Logits stats: min={logits.min()}, max={logits.max()}, mean={logits.mean()}")
            # Replace inf/nan with zeros to prevent downstream issues
            logits = torch.where(
                torch.isnan(logits) | torch.isinf(logits),
                torch.zeros_like(logits),
                logits
            )
        
        v_h, _ = self.value_encoder(h, use_cache=False)
        v_mid = self.value_head_intermediate(v_h)
        values = self.value_head_final(v_mid).squeeze(-1)
        return logits, values, None

    # ---------- generation forward (one step) -------------
    def step(self, tok_id: torch.Tensor, t: int, cache: Optional[List[Any]]):
        B = tok_id.size(0)
        # Changed dtype to torch.long as nn.Embedding expects long
        pos = torch.full((B, 1), t, device=tok_id.device, dtype=torch.long)
        h = self.embed(tok_id.unsqueeze(1)) + self.posemb(pos)
        h, new_cache = self.encoder(h, cache, use_cache=True)
        logits = self.policy_head(h).squeeze(1)
        
        # Add debugging for inf/nan values
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"WARNING: Found inf/nan in logits in step pass at t={t}!")
            print(f"Logits shape: {logits.shape}")
            print(f"Logits stats: min={logits.min()}, max={logits.max()}, mean={logits.mean()}")
            # Replace inf/nan with zeros to prevent downstream issues
            logits = torch.where(
                torch.isnan(logits) | torch.isinf(logits),
                torch.zeros_like(logits),
                logits
            )
        
        v_h, _ = self.value_encoder(h, use_cache=False)
        v_mid = self.value_head_intermediate(v_h).squeeze(1)
        value = self.value_head_final(v_mid).squeeze(-1)
        return logits, value, new_cache

# ======================================================================================
# 4. Grammar constants / token tables  – UNCHANGED from original file
#    (Full block reproduced verbatim so the code is self‑contained.)
# ======================================================================================
# MAX_PARTICLES, MAX_FIELDS, MAX_INTERACTIONS, MAX_P_PER_F, PARTICLE_TYPE_MAP, 
# FIELD_TYPE_MAP, CHIRALITY_MAP, P_FEAT_*, F_FEAT_*, I_FEAT_*, NUM_*_FEATURES

# Helper to generate ID tokens

def generate_id_tokens(prefix: str, count: int) -> List[str]:
    return [f"{prefix}_{i}" for i in range(count)]

PARTICLE_ID_RANGE = 14
INTERACTION_ID_RANGE = 5
FIELD_ID_RANGE = 11
DIM_RANGE = 5
GEN_RANGE = 5

IDS = range(1, FIELD_ID_RANGE)
DIMS = range(1, DIM_RANGE)
GENS = range(1, GEN_RANGE)
ITRACT_IDS, FIELD_IDS, PARTICLE_IDS = (
    generate_id_tokens(p, count) for p, count in [("ITRACT_ID", INTERACTION_ID_RANGE), ("FIELD_ID", FIELD_ID_RANGE), ("PARTICLE_ID", PARTICLE_ID_RANGE)]
)
DIM_TOKENS = [f"DIM_{i}" for i in DIMS]
GEN_TOKENS = [f"GEN_{i}" for i in GENS]

SU3_REP_TOKENS = ["SU3C_REP_1", "SU3C_REP_3", "SU3C_REP_8"]
SU2_REP_TOKENS = ["SU2L_REP_1", "SU2L_REP_2", "SU2L_REP_3"]
U1Y_CHARGE_TOKENS = [f"U1Y_CHARGE_{i}" for i in [-2, -1, 0, 1, 2]]
QN_L_TOKENS = ["QN_L_0", "QN_L_1"]
QN_B_TOKENS = ["QN_B_0", "QN_B_1"]
REP_TOKENS = SU3_REP_TOKENS + SU2_REP_TOKENS + U1Y_CHARGE_TOKENS + QN_L_TOKENS + QN_B_TOKENS
ALL_TOKEN_NAMES: List[str] = (
    ["BOS", "ITRACT"] + ITRACT_IDS + ["TYPE_YUKAWA", "FIELD"] + FIELD_IDS + ["TYPE_complex", "TYPE_real", "TYPE_fermion"] +
    DIM_TOKENS + GEN_TOKENS + ["SELF_CONJ_TRUE", "SELF_CONJ_FALSE", "CHIRALITY_left", "CHIRALITY_right", "CHIRALITY_none"] +
    REP_TOKENS + ["PARTICLE"] + PARTICLE_IDS + ["MASS_1e2", "CHARGE_0", "CHARGE_1", "END_PARTICLE", "END_FIELD", "END_ITRACT", "EOS", "DEAD"]
)

# Grammar rules – identical to original file; reproduced verbatim
TOKENS_MODEL = {"BOS": ["ITRACT"], "END_ITRACT": ["ITRACT", "EOS"], "EOS": ["EOS"], "DEAD": ["DEAD"]}
TOKENS_INTERACTION = {"ITRACT": ITRACT_IDS, "TYPE_YUKAWA": ["FIELD"], "END_FIELD": ["FIELD", "END_ITRACT"]}
for it_id in ITRACT_IDS:
    TOKENS_INTERACTION[it_id] = ["TYPE_YUKAWA"]
TOKENS_FIELD = {"FIELD": FIELD_IDS, "END_PARTICLE": ["PARTICLE", "END_FIELD"]}
for f_id in FIELD_IDS:
    TOKENS_FIELD[f_id] = ["TYPE_fermion", "TYPE_real", "TYPE_complex"]
TOKENS_FIELD["TYPE_fermion"] = DIM_TOKENS
TOKENS_FIELD["TYPE_real"] = DIM_TOKENS
TOKENS_FIELD["TYPE_complex"] = DIM_TOKENS
for dim_tok in DIM_TOKENS:
    TOKENS_FIELD[dim_tok] = GEN_TOKENS
for gen_tok in GEN_TOKENS:
    TOKENS_FIELD[gen_tok] = ["SELF_CONJ_TRUE", "SELF_CONJ_FALSE"]
TOKENS_FIELD["SELF_CONJ_TRUE"] = ["CHIRALITY_left", "CHIRALITY_right", "CHIRALITY_none"]
TOKENS_FIELD["SELF_CONJ_FALSE"] = ["CHIRALITY_left", "CHIRALITY_right", "CHIRALITY_none"]
REP_SEQUENCE_START = SU3_REP_TOKENS
TOKENS_FIELD["CHIRALITY_left"] = REP_SEQUENCE_START
TOKENS_FIELD["CHIRALITY_right"] = REP_SEQUENCE_START
TOKENS_FIELD["CHIRALITY_none"] = REP_SEQUENCE_START
for su3_tok in SU3_REP_TOKENS: TOKENS_FIELD[su3_tok] = SU2_REP_TOKENS
for su2_tok in SU2_REP_TOKENS: TOKENS_FIELD[su2_tok] = U1Y_CHARGE_TOKENS
for u1y_tok in U1Y_CHARGE_TOKENS: TOKENS_FIELD[u1y_tok] = QN_L_TOKENS
for qnl_tok in QN_L_TOKENS: TOKENS_FIELD[qnl_tok] = QN_B_TOKENS
for qnb_tok in QN_B_TOKENS: TOKENS_FIELD[qnb_tok] = ["PARTICLE"]
TOKENS_PARTICLE = {"PARTICLE": PARTICLE_IDS, "MASS_1e2": ["CHARGE_0", "CHARGE_1"], "CHARGE_0": ["END_PARTICLE"], "CHARGE_1": ["END_PARTICLE"]}
for p_id in PARTICLE_IDS: TOKENS_PARTICLE[p_id] = ["TYPE_fermion", "TYPE_real", "TYPE_complex"]
for p_type in ["TYPE_fermion", "TYPE_real", "TYPE_complex"]: TOKENS_PARTICLE[p_type] = ["MASS_1e2"]

# ---- Grammar Stack Embeddings ----------------------------------------------------
EMBED_VAL_MAP: Dict[str, int] = {"ITRACT": 0b1000, "END_ITRACT": -0b1000, "FIELD": 0b0100, "END_FIELD": -0b0100, "PARTICLE": 0b0010, "END_PARTICLE": -0b0010}

# ======================================================================================
# 4.1 TokenSequenceParser – convert *flat token list* → hierarchical model dict
# ======================================================================================

class TokenSequenceParser:
    """Recursive-descent parser that interprets the linear token stream."""
    MASS_MAP   = {"MASS_1e2": 100.0}
    CHARGE_MAP = {"CHARGE_0": 0, "CHARGE_1": 1}

    @staticmethod
    def _after(tok: str, prefix: str) -> str:
        return tok.replace(f"{prefix}_", "")

    @staticmethod
    def _int(tok: str) -> int:
        return int(tok.split("_")[-1])

    def build_model_from_tokens(self, tokens: List[str]) -> Dict[str, Any]:
        """Parse the flat token list → merged model dict (never returns `{}` on duplicates)."""
        particles, fields, interactions = [], [], []
        i = 0

        # ─── 1) Flat parse into lists ───────────────────────────────────────────────
        while i < len(tokens):
            tok = tokens[i]
            if tok == "ITRACT":
                it, i_after, new_fs, new_ps = self._parse_interaction(tokens, i)
                interactions.append(it)
                fields.extend(new_fs)
                particles.extend(new_ps)
                i = i_after
            elif tok == "FIELD":
                f, i_after, ps = self._parse_field(tokens, i)
                fields.append(f)
                particles.extend(ps)
                i = i_after
            elif tok == "PARTICLE":
                p, i_after = self._parse_particle(tokens, i)
                particles.append(p)
                i = i_after
            else:
                i += 1

        # ─── 2) Merge duplicates (keep first‑seen definition) ────────────────────────
        merged_particles = {}
        for p in particles:
            if p["id"] not in merged_particles:
                merged_particles[p["id"]] = p

        merged_fields = {}
        for f in fields:
            if f["id"] not in merged_fields:
                merged_fields[f["id"]] = f

        seen_iids = set()
        unique_interactions = []
        for it in interactions:
            if it["id"] not in seen_iids:
                seen_iids.add(it["id"])
                unique_interactions.append(it)

        # ─── 3) Return the merged model ─────────────────────────────────────────────
        return {
            "particles":    list(merged_particles.values()),
            "fields":       list(merged_fields.values()),
            "interactions": unique_interactions,
        }

    def _parse_interaction(self, seq: List[str], i: int):
        """Parse `ITRACT … END_ITRACT` block."""
        inter = {
            "id":     f"i{self._int(seq[i+1])}",
            "type":   self._after(seq[i+2], "TYPE").lower(),
            "fields": []
        }
        i += 3
        new_fields, new_particles = [], []
        while seq[i] != "END_ITRACT":
            if seq[i] == "FIELD":
                f, i_after, ps = self._parse_field(seq, i)
                new_fields.append(f)
                new_particles.extend(ps)
                inter["fields"].append(f["id"])
                i = i_after
            else:
                i += 1
        return inter, i + 1, new_fields, new_particles

    def _parse_field(self, seq: List[str], i: int):
        """Parse `FIELD ... END_FIELD` block."""
        base = i
        field_type = self._after(seq[base+2], "TYPE")
        field = {
            "id":             f"f{self._int(seq[base+1])}",
            "type":           field_type,
            "dim":            self._int(seq[base+3]),
            "gen":            self._int(seq[base+4]),
            "self_conjugate": seq[base+5] == "SELF_CONJ_TRUE",
        }
        idx = base + 6
        chir_tok = seq[idx]; idx += 1
        field["chirality"] = self._after(chir_tok, "CHIRALITY") if field_type == "fermion" else "none"
        field.update({
            "SU3_rep":   self._int(seq[idx]),
            "SU2_rep":   self._int(seq[idx+1]),
            "U1_charge": self._int(seq[idx+2]),
            "QN_L":      self._int(seq[idx+3]),
            "QN_B":      self._int(seq[idx+4]),
            "particles": []
        })
        idx += 5
        new_particles = []
        while seq[idx] != "END_FIELD":
            if seq[idx] == "PARTICLE":
                p, idx_after = self._parse_particle(seq, idx)
                new_particles.append(p)
                field["particles"].append(p["id"])
                idx = idx_after
            else:
                idx += 1
        return field, idx + 1, new_particles

    def _parse_particle(self, seq: List[str], i: int):
        """Parse `PARTICLE ... END_PARTICLE` block."""
        base = i
        particle_type = self._after(seq[base+2], "TYPE")
        particle = {
            "id":     f"p{self._int(seq[base+1])}",
            "type":   particle_type,
            "mass":   self.MASS_MAP[seq[base+3]],
            "charge": self.CHARGE_MAP[seq[base+4]],
        }
        return particle, base + 5

# ======================================================================================
# 4.2 Model-based scoring functions
# ======================================================================================

@torch.no_grad()
def model_based_scorer(
    sequences: torch.Tensor,
    idx_to_token: Dict[int, str],
    level: str = "all",
    verbose: bool = False,
) -> Tuple[torch.Tensor, int]:
    """
    Score sequences using token2fr's build_model method.
    Returns the numerator of the score (number of checks passed) from model.py.
    """
    B = sequences.shape[0]
    device = sequences.device
    
    # Convert sequences to token lists
    seq_cpu = sequences.cpu().tolist()
    token_lists = []
    
    for i in range(B):
        # Convert indices to tokens, stopping at dead_idx or eos_idx
        tokens = []
        for idx in seq_cpu[i]:
            if idx == -1:  # dead_idx
                break
            token = idx_to_token.get(idx, "UNKNOWN")
            tokens.append(token)
        token_lists.append(tokens)
    
    # Find EOS index for comparison
    eos_idx = None
    for k, v in idx_to_token.items():
        if v == "EOS":
            eos_idx = k
            break
    if eos_idx is None:
        raise ValueError("EOS token not found in idx_to_token mapping")

    # Score each sequence using token2fr's build_model
    scores = torch.zeros(B, device=device)
    total_valid = 0
    
    for i, tokens in enumerate(token_lists):
        # Reject and do not score if the sequence does not end with EOS
        # (ignoring trailing DEAD tokens)
        # Find the last non-DEAD token
        last_non_dead = None
        for j in range(len(seq_cpu[i]) - 1, -1, -1):
            if seq_cpu[i][j] != -1:
                last_non_dead = j
                break
        if last_non_dead is None or seq_cpu[i][last_non_dead] != eos_idx:
            # Sequence does not end with EOS, reject
            scores[i] = 0.0
            continue
        try:
            # Use token2fr's build_model method to get the model dictionary
            from token2fr import ModelTester
            tester = ModelTester()
            model_dict = tester.build_model(tokens)
            
            # Create a temporary JSON file
            import tempfile
            import json
            import os
            
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
                json.dump(model_dict, tf, indent=2)
                json_path = tf.name
            
            try:
                # Use the Model class from json2fr to get the score
                from json2fr.model import Model
                import tempfile
                
                # Create a temporary output directory
                output_dir = tempfile.mkdtemp(prefix="model_score_")
                
                # Create the model and get the score
                model = Model("TestModel", "TestAuthor", json_path, output_dir)
                model._read_model()  # This populates the score
                
                # Extract the numerator from the score (e.g., "15/20" -> 15)
                score_str = model.score
                if '/' in score_str:
                    numerator = int(score_str.split('/')[0])
                    scores[i] = float(numerator)
                    total_valid += 1
                else:
                    scores[i] = 0.0
                
                # Clean up temporary files
                import shutil
                shutil.rmtree(output_dir)
                
            finally:
                # Clean up the temporary JSON file
                os.unlink(json_path)
                
        except Exception:
            # Silently handle errors - assign 0.0 score without printing
            scores[i] = 0.0
    
    return scores, total_valid

@torch.no_grad()
def model_based_scorer_with_diagnostics(
    sequences: torch.Tensor,
    idx_to_token: Dict[int, str],
    level: str = "all",
    verbose: bool = False,
) -> Tuple[torch.Tensor, int, List[str], List[torch.Tensor]]:
    """
    Score sequences using token2fr's build_model method with diagnostic information.
    Returns scores, total valid, check names, and check results.
    """
    scores, total_valid = model_based_scorer(sequences, idx_to_token, level, verbose)
    
    # For diagnostics, we'll create simple check names and results
    # based on whether the model was successfully built
    check_names = ["Model successfully built"]
    check_results = [scores]  # The scores themselves serve as the check results
    
    return scores, total_valid, check_names, check_results

# ======================================================================================
# 4.3 Diagnostic Tracking Class
# ======================================================================================

class DiagnosticTracker:
    """Comprehensive tracking for training dynamics and bottlenecks."""

    def __init__(self, config: PPOConfig, check_names: List[str], max_history_len: int = 200):
        self.config = config
        self.check_names = check_names
        self.num_checks = len(check_names)
        self.max_history_len = max_history_len # New: Max length for history lists

        # Initialize tracking dictionaries using deque for rolling window
        self.per_check_pass_rates = {name: deque(maxlen=max_history_len) for name in check_names}
        # Removed check_weight_evolution as check_weights are no longer dynamic for validation bonus
        self.kl_divergences = deque(maxlen=max_history_len)
        self.gradient_norms = deque(maxlen=max_history_len)
        self.reward_components = {
            'base_rewards': deque(maxlen=max_history_len),
            'particle_rewards': deque(maxlen=max_history_len),
            'field_rewards': deque(maxlen=max_history_len),
            'terminal_rewards': deque(maxlen=max_history_len),
            'novelty_rewards': deque(maxlen=max_history_len) # NEW: Novelty rewards
            # Removed 'validation_bonus'
        }
        self.sequence_length_stats = {
            'mean_length': deque(maxlen=max_history_len),
            'std_length': deque(maxlen=max_history_len),
            'min_length': deque(maxlen=max_history_len),
            'max_length': deque(maxlen=max_history_len)
        }
        self.action_distribution_drift = {
            'END_FIELD': deque(maxlen=max_history_len),
            'END_ITRACT': deque(maxlen=max_history_len),
            'EOS': deque(maxlen=max_history_len),
            'PARTICLE': deque(maxlen=max_history_len),
            'FIELD': deque(maxlen=max_history_len),
            'ITRACT': deque(maxlen=max_history_len)
        }

        # NEW: Additional diagnostic metrics
        self.clipping_fraction = deque(maxlen=max_history_len)
        self.kl_per_epoch = {
            'mean': deque(maxlen=max_history_len),
            'max': deque(maxlen=max_history_len),
            'p95': deque(maxlen=max_history_len)
        }
        self.advantage_stats = {
            'mean': deque(maxlen=max_history_len),
            'std': deque(maxlen=max_history_len),
            'skew': deque(maxlen=max_history_len),
            'kurtosis': deque(maxlen=max_history_len),
            'p1': deque(maxlen=max_history_len), # Added percentiles
            'p5': deque(maxlen=max_history_len),
            'p25': deque(maxlen=max_history_len),
            'p50': deque(maxlen=max_history_len),
            'p75': deque(maxlen=max_history_len),
            'p95': deque(maxlen=max_history_len),
            'p99': deque(maxlen=max_history_len),
            'negative_fraction': deque(maxlen=max_history_len),
            'extreme_positive_fraction': deque(maxlen=max_history_len),
            'extreme_negative_fraction': deque(maxlen=max_history_len),
            'distribution_symmetry': deque(maxlen=max_history_len),
            'mean_median_diff': deque(maxlen=max_history_len),
        }
        self.value_function_error = {
            'mean_value': deque(maxlen=max_history_len),
            'mean_return': deque(maxlen=max_history_len),
            'value_bias': deque(maxlen=max_history_len),
            'value_variance': deque(maxlen=max_history_len)
        }
        self.vf_epochs_tracking = {
            'vf_loss_per_epoch': deque(maxlen=max_history_len), # Note: this will store lists of losses
            'vf_loss_improvement': deque(maxlen=max_history_len),
            'vf_gradient_norms': deque(maxlen=max_history_len) # Note: this will store lists of norms
        }
        self.mask_density = {
            'mean_legal_tokens': deque(maxlen=max_history_len),
            'min_legal_tokens': deque(maxlen=max_history_len),
            'median_legal_tokens': deque(maxlen=max_history_len),
            'p95_legal_tokens': deque(maxlen=max_history_len)
        }
        self.pg_loss_components = {
            'unclipped_loss': deque(maxlen=max_history_len),
            'clipped_loss': deque(maxlen=max_history_len),
            'actual_loss': deque(maxlen=max_history_len)
        }
        self.timing_breakdown = {
            'rollout_time': deque(maxlen=max_history_len),
            'validation_time': deque(maxlen=max_history_len),
            'reward_calc_time': deque(maxlen=max_history_len),
            'ppo_time': deque(maxlen=max_history_len),
            'vf_time': deque(maxlen=max_history_len),
            'logging_time': deque(maxlen=max_history_len)
        }
        
        # NEW: Parser failure tracking
        self.parser_failure_rate = deque(maxlen=max_history_len)

        # For policy update stats (previously called kl_divergences but now more general)
        self.policy_update_stats = {
            'old_logp_mean': deque(maxlen=max_history_len),
            'new_logp_mean': deque(maxlen=max_history_len),
            'logp_diff_mean': deque(maxlen=max_history_len),
            'logp_diff_std': deque(maxlen=max_history_len),
            'policy_change_magnitude': deque(maxlen=max_history_len),
            'full_distribution_kl': deque(maxlen=max_history_len),
            'sampled_action_kl': deque(maxlen=max_history_len)
        }

        # Critical token indices for action distribution tracking
        self.critical_tokens = {
            'END_FIELD': None,
            'END_ITRACT': None,
            'EOS': None,
            'PARTICLE': None,
            'FIELD': None,
            'ITRACT': None
        }

    def set_token_indices(self, token_to_idx: Dict[str, int]):
        """Set the token indices for critical tokens."""
        for token_name in self.critical_tokens:
            if token_name in token_to_idx:
                self.critical_tokens[token_name] = token_to_idx[token_name]
    
    def update_per_check_pass_rates(self, results: List[torch.Tensor]):
        """Update per-check pass rates."""
        for i, (name, result) in enumerate(zip(self.check_names, results)):
            pass_rate = result.float().mean().item()
            self.per_check_pass_rates[name].append(pass_rate)
    
    def update_check_weight_evolution(self, check_weights: torch.Tensor):
        # Removed as check_weights for validation bonus are no longer dynamic
        pass 
    
    def update_kl_divergence(self, old_log_probs: torch.Tensor, new_log_probs: torch.Tensor, is_alive: torch.Tensor):
        """Compute and update sampled-action KL divergence between old and new policies."""
        if self.config.track_diagnostics:
            alive_old_logp = old_log_probs[is_alive]
            alive_new_logp = new_log_probs[is_alive]

            if len(alive_old_logp) > 0 and len(alive_new_logp) > 0:
                log_ratio = alive_old_logp - alive_new_logp
                sampled_kl_div = log_ratio.mean().item()
                sampled_kl_div = max(0.0, sampled_kl_div) # Ensure non-negativity

                # Update policy_update_stats
                self.policy_update_stats['old_logp_mean'].append(alive_old_logp.mean().item())
                self.policy_update_stats['new_logp_mean'].append(alive_new_logp.mean().item())
                self.policy_update_stats['logp_diff_mean'].append(log_ratio.mean().item())
                self.policy_update_stats['logp_diff_std'].append(log_ratio.std().item())
                self.policy_update_stats['policy_change_magnitude'].append(torch.abs(log_ratio).mean().item())
                self.policy_update_stats['sampled_action_kl'].append(sampled_kl_div)
                # No longer appending to self.kl_divergences for sampled KL, use policy_update_stats['sampled_action_kl']
            else:
                # Append 0 if no alive samples
                self.policy_update_stats['old_logp_mean'].append(0.0)
                self.policy_update_stats['new_logp_mean'].append(0.0)
                self.policy_update_stats['logp_diff_mean'].append(0.0)
                self.policy_update_stats['logp_diff_std'].append(0.0)
                self.policy_update_stats['policy_change_magnitude'].append(0.0)
                self.policy_update_stats['sampled_action_kl'].append(0.0)

    def update_full_distribution_kl(self, full_dist_kl: float):
        """Update full-distribution KL divergence tracking."""
        if self.config.track_diagnostics:
            self.policy_update_stats['full_distribution_kl'].append(full_dist_kl)

    def get_latest_kl_divergence(self) -> float:
        """Get the latest sampled-action KL divergence value (for quick reference)."""
        return self.policy_update_stats['sampled_action_kl'][-1] if self.policy_update_stats['sampled_action_kl'] else 0.0
    
    def update_gradient_norms(self, model: nn.Module):
        """Compute and update gradient norms."""
        if self.config.track_diagnostics:
            # Compute gradient norm BEFORE clipping
            total_norm = 0.0
            param_norms = []
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    param_norms.append(param_norm.item())
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            # Store pre-clip gradient norm
            self.gradient_norms.append(total_norm)
            
            # Additional gradient statistics
            if param_norms:
                if 'gradient_stats' not in self.__dict__:
                    self.gradient_stats = {
                        'mean_param_norm': [],
                        'max_param_norm': [],
                        'min_param_norm': [],
                        'std_param_norm': [],
                        'clipping_ratio': []
                    }
                
                # Filter out NaN and infinite values from param_norms
                valid_norms = [norm for norm in param_norms if np.isfinite(norm)]
                if valid_norms:
                    self.gradient_stats['mean_param_norm'].append(np.mean(valid_norms))
                    self.gradient_stats['max_param_norm'].append(np.max(valid_norms))
                    self.gradient_stats['min_param_norm'].append(np.min(valid_norms))
                    self.gradient_stats['std_param_norm'].append(np.std(valid_norms))
                else:
                    # If no valid norms, append zeros
                    self.gradient_stats['mean_param_norm'].append(0.0)
                    self.gradient_stats['max_param_norm'].append(0.0)
                    self.gradient_stats['min_param_norm'].append(0.0)
                    self.gradient_stats['std_param_norm'].append(0.0)
                
                # Compute what fraction of parameters would be clipped
                clip_threshold = self.config.gradient_clip_norm
                would_be_clipped = sum(1 for norm in param_norms if norm > clip_threshold)
                clipping_ratio = would_be_clipped / len(param_norms) if param_norms else 0.0
                self.gradient_stats['clipping_ratio'].append(clipping_ratio)
    
    def update_clipping_fraction(self, ratios: torch.Tensor, is_alive: torch.Tensor):
        """Update clipping fraction statistics."""
        if self.config.track_diagnostics:
            clip_coef = self.config.clip_coef
            clipped = ((ratios > (1 + clip_coef)) | (ratios < (1 - clip_coef)))[is_alive]
            clipping_frac = clipped.float().mean().item()
            self.clipping_fraction.append(clipping_frac)
    
    def update_advantage_stats(self, advantages: torch.Tensor, is_alive: torch.Tensor):
        """Update advantage distribution statistics."""
        if self.config.track_diagnostics:
            alive_adv = advantages[is_alive]
            if len(alive_adv) > 0:
                # Basic statistics
                mean_adv = alive_adv.mean().item()
                std_adv = alive_adv.std().item()
                self.advantage_stats['mean'].append(mean_adv)
                self.advantage_stats['std'].append(std_adv)
                
                # Percentiles for better distribution understanding
                percentiles = [1, 5, 25, 50, 75, 95, 99]
                for p in percentiles:
                    p_val = alive_adv.quantile(p/100).item()
                    if f'p{p}' not in self.advantage_stats:
                        self.advantage_stats[f'p{p}'] = []
                    self.advantage_stats[f'p{p}'].append(p_val)
                
                # Compute skewness and kurtosis
                if std_adv > 1e-8:
                    normalized = (alive_adv - mean_adv) / std_adv
                    skew = (normalized ** 3).mean().item()
                    kurtosis = (normalized ** 4).mean().item() - 3  # excess kurtosis
                    self.advantage_stats['skew'].append(skew)
                    self.advantage_stats['kurtosis'].append(kurtosis)
                else:
                    self.advantage_stats['skew'].append(0.0)
                    self.advantage_stats['kurtosis'].append(0.0)
                
                # Track fraction of negative advantages (important for policy collapse)
                negative_frac = (alive_adv < 0).float().mean().item()
                if 'negative_fraction' not in self.advantage_stats:
                    self.advantage_stats['negative_fraction'] = []
                self.advantage_stats['negative_fraction'].append(negative_frac)
                
                # Track extreme values
                extreme_positive = (alive_adv > self.config.advantage_clip_threshold).float().mean().item()
                extreme_negative = (alive_adv < -self.config.advantage_clip_threshold).float().mean().item()
                if 'extreme_positive_fraction' not in self.advantage_stats:
                    self.advantage_stats['extreme_positive_fraction'] = []
                    self.advantage_stats['extreme_negative_fraction'] = []
                self.advantage_stats['extreme_positive_fraction'].append(extreme_positive)
                self.advantage_stats['extreme_negative_fraction'].append(extreme_negative)
                
                # NEW: Track distribution symmetry metrics
                if 'distribution_symmetry' not in self.advantage_stats:
                    self.advantage_stats['distribution_symmetry'] = []
                
                # Compute symmetry metric: how close is negative fraction to 0.5?
                symmetry_error = abs(negative_frac - 0.5)
                self.advantage_stats['distribution_symmetry'].append(symmetry_error)
                
                # Track median vs mean to detect skewness
                median_adv = alive_adv.median().item()
                mean_median_diff = abs(mean_adv - median_adv)
                if 'mean_median_diff' not in self.advantage_stats:
                    self.advantage_stats['mean_median_diff'] = []
                self.advantage_stats['mean_median_diff'].append(mean_median_diff)
    
    def update_value_function_error(self, values: torch.Tensor, returns: torch.Tensor, is_alive: torch.Tensor):
        """Update value function error statistics."""
        if self.config.track_diagnostics:
            alive_values = values[is_alive]
            alive_returns = returns[is_alive]
            
            if len(alive_values) > 0:
                mean_value = alive_values.mean().item()
                mean_return = alive_returns.mean().item()
                value_bias = mean_value - mean_return
                value_variance = alive_values.var().item()
                
                self.value_function_error['mean_value'].append(mean_value)
                self.value_function_error['mean_return'].append(mean_return)
                self.value_function_error['value_bias'].append(value_bias)
                self.value_function_error['value_variance'].append(value_variance)
    
    def update_vf_epochs_tracking(self, vf_losses: List[float], vf_gradient_norms: List[float]):
        """Update value function epochs tracking statistics."""
        if self.config.track_diagnostics and vf_losses:
            # Store the value function losses per epoch
            self.vf_epochs_tracking['vf_loss_per_epoch'].append(vf_losses)
            
            # Calculate improvement from first to last epoch
            if len(vf_losses) > 1:
                improvement = vf_losses[0] - vf_losses[-1]
                self.vf_epochs_tracking['vf_loss_improvement'].append(improvement)
            else:
                self.vf_epochs_tracking['vf_loss_improvement'].append(0.0)
            
            # Store gradient norms for value function training
            if vf_gradient_norms:
                self.vf_epochs_tracking['vf_gradient_norms'].append(vf_gradient_norms)
    
    def update_mask_density(self, masks: torch.Tensor, is_alive: torch.Tensor):
        """Update mask density statistics."""
        if self.config.track_diagnostics:
            # Count legal tokens per position
            legal_token_counts = masks[is_alive].sum(dim=-1).float()
            
            if len(legal_token_counts) > 0:
                self.mask_density['mean_legal_tokens'].append(legal_token_counts.mean().item())
                self.mask_density['min_legal_tokens'].append(legal_token_counts.min().item())
                self.mask_density['median_legal_tokens'].append(legal_token_counts.median().item())
                self.mask_density['p95_legal_tokens'].append(legal_token_counts.quantile(0.95).item())
    
    def update_pg_loss_components(self, advantages: torch.Tensor, ratios: torch.Tensor, 
                                 clipped_ratios: torch.Tensor, is_alive: torch.Tensor):
        """Update policy gradient loss components."""
        if self.config.track_diagnostics:
            alive_adv = advantages[is_alive]
            alive_ratios = ratios[is_alive]
            alive_clipped = clipped_ratios[is_alive]
            
            if len(alive_adv) > 0:
                unclipped_loss = -(alive_adv * alive_ratios).mean().item()
                clipped_loss = -(alive_adv * alive_clipped).mean().item()
                actual_loss = -torch.min(alive_adv * alive_ratios, alive_adv * alive_clipped).mean().item()
                
                self.pg_loss_components['unclipped_loss'].append(unclipped_loss)
                self.pg_loss_components['clipped_loss'].append(clipped_loss)
                self.pg_loss_components['actual_loss'].append(actual_loss)
    
    def update_timing_breakdown(self, timing_dict: Dict[str, float]):
        """Update timing breakdown statistics."""
        if self.config.track_diagnostics:
            for key in self.timing_breakdown:
                if key in timing_dict:
                    self.timing_breakdown[key].append(timing_dict[key])
    
    def update_reward_components(self, base_rewards: torch.Tensor, terminal_rewards: torch.Tensor, novelty_rewards: torch.Tensor): # MODIFIED: Added novelty_rewards
        """Update reward component breakdown. Removed validation_bonus."""
        is_alive = base_rewards != 0  # Simple heuristic for alive positions
        
        self.reward_components['base_rewards'].append(base_rewards[is_alive].mean().item())
        self.reward_components['terminal_rewards'].append(terminal_rewards[is_alive].mean().item())
        self.reward_components['novelty_rewards'].append(novelty_rewards.mean().item()) # NEW: Track mean novelty reward
        # Removed: self.reward_components['validation_bonus'].append(validation_bonus.mean().item())
    
    def update_sequence_length_stats(self, sequences: torch.Tensor, dead_idx: int):
        """Update sequence length distribution statistics."""
        is_alive = sequences != dead_idx
        lengths = is_alive.sum(dim=1).float()
        
        self.sequence_length_stats['mean_length'].append(lengths.mean().item())
        self.sequence_length_stats['std_length'].append(lengths.std().item())
        self.sequence_length_stats['min_length'].append(lengths.min().item())
        self.sequence_length_stats['max_length'].append(lengths.max().item())
    
    def update_action_distribution_drift(self, logits: torch.Tensor, masks: torch.Tensor, is_alive: torch.Tensor):
        """Update action distribution drift for critical tokens."""
        if not self.config.track_diagnostics:
            return
        
        # Compute probabilities
        probs = F.softmax(logits.masked_fill(~masks, -1e4), dim=-1)
        
        # Track marginal probabilities for critical tokens
        for token_name, token_idx in self.critical_tokens.items():
            if token_idx is not None:
                # Average probability across all positions and batch
                token_probs = probs[:, :, token_idx]
                avg_prob = token_probs[is_alive].mean().item()
                self.action_distribution_drift[token_name].append(avg_prob)
    
    def log_diagnostic_summary(self, logger, step: int):
        """Log a summary of current diagnostic metrics."""
        if not self.config.track_diagnostics:
            return
        
        logger.info(f"=== DIAGNOSTIC SUMMARY (Step {step}) ===")
        
        # KL divergence (prefer full-distribution KL if available)
        latest_kl = 0.0
        if hasattr(self, 'policy_update_stats'):
            full_list = self.policy_update_stats.get('full_distribution_kl', [])
            if full_list:
                latest_kl = full_list[-1]
        elif self.kl_divergences:
            # fall back to sampled-action KL
            latest_kl = self.kl_divergences[-1]
        logger.info(f"KL Divergence (full dist): {latest_kl:.6f} (target: {self.config.kl_target})")
        
        # Additional policy update diagnostics
        if hasattr(self, 'policy_update_stats') and self.policy_update_stats['policy_change_magnitude']:
            policy_change = self.policy_update_stats['policy_change_magnitude'][-1]
            logp_diff_mean = self.policy_update_stats['logp_diff_mean'][-1]
            logp_diff_std = self.policy_update_stats['logp_diff_std'][-1]
            
            logger.info(f"  Policy change magnitude: {policy_change:.6f} (should be > 1e-4)")
            logger.info(f"  Log prob diff - mean: {logp_diff_mean:.6f}, std: {logp_diff_std:.6f}")
            
            # Show both KL divergence types if available
            if self.policy_update_stats['full_distribution_kl']:
                full_dist_kl = self.policy_update_stats['full_distribution_kl'][-1]
                sampled_kl = self.policy_update_stats['sampled_action_kl'][-1]
                logger.info(f"  Full-distribution KL: {full_dist_kl:.6f} (measures policy difference over entire action space)")
                # logger.info(f"  Sampled-action KL: {sampled_kl:.6f} (measures policy difference for actions actually taken)")
                
                # Explain the difference
                if full_dist_kl > 0.001 and sampled_kl < 0.001:
                    # logger.info(f"  NOTE: Full-distribution KL > 0 but sampled-action KL ≈ 0 is NORMAL and HEALTHY!")
                    # logger.info(f"  This means the policy is changing, but the actions taken had similar probabilities.")
                    # logger.info(f"  This is expected behavior in PPO when policy updates are conservative.")
                    pass
                elif full_dist_kl < 0.001 and sampled_kl < 0.001:
                    logger.info(f"  WARNING: Both KL divergences are near zero - policy may not be updating!")
                    logger.info(f"  Check: learning rate, advantages, gradient flow, loss weights")
            
            # Warnings for policy update issues
            if policy_change < 1e-4:
                logger.info(f"  WARNING: Policy is not updating! Change magnitude too small.")
                logger.info(f"  Consider: increasing learning_rate, checking advantages, reducing other loss weights")
            
            if abs(logp_diff_mean) < 1e-6 and logp_diff_std < 1e-6:
                logger.info(f"  WARNING: Log probabilities are identical - policy gradient not working!")
                logger.info(f"  Check: advantages, loss weights, gradient flow")
        
        # Gradient norm
        if self.gradient_norms:
            latest_grad_norm = self.gradient_norms[-1]
            logger.info(f"Gradient Norm: {latest_grad_norm:.6f} (clipped at: {self.config.gradient_clip_norm})")
            
            # Additional gradient statistics
            if hasattr(self, 'gradient_stats') and self.gradient_stats['clipping_ratio']:
                clipping_ratio = self.gradient_stats['clipping_ratio'][-1]
                mean_param_norm = self.gradient_stats['mean_param_norm'][-1]
                max_param_norm = self.gradient_stats['max_param_norm'][-1]
                logger.info(f"  Clipping ratio: {clipping_ratio:.3f} (high >0.1 indicates aggressive clipping)")
                logger.info(f"  Param norms - mean: {mean_param_norm:.4f}, max: {max_param_norm:.4f}")
                
                # Warning if gradients are being heavily clipped
                if clipping_ratio > 0.1:
                    logger.info(f"  WARNING: {clipping_ratio*100:.1f}% of parameters are being clipped!")
                    logger.info(f"  Consider increasing gradient_clip_norm or reducing learning_rate")
        
        # Clipping fraction
        if self.clipping_fraction:
            latest_clip_frac = self.clipping_fraction[-1]
            logger.info(f"Clipping Fraction: {latest_clip_frac:.3f} (high >0.3, low <0.05)")
        
        # Advantage stats
        if self.advantage_stats['mean']:
            latest_adv_mean = self.advantage_stats['mean'][-1]
            latest_adv_std = self.advantage_stats['std'][-1]
            logger.info(f"Advantage: {latest_adv_mean:.4f} ± {latest_adv_std:.4f}")
            
            # Additional advantage diagnostics
            if 'negative_fraction' in self.advantage_stats and self.advantage_stats['negative_fraction']:
                neg_frac = self.advantage_stats['negative_fraction'][-1]
                logger.info(f"  Negative fraction: {neg_frac:.3f} (should be ~0.5 for standard normalization)")
                
                # NEW: Show symmetry error
                if 'distribution_symmetry' in self.advantage_stats and self.advantage_stats['distribution_symmetry']:
                    symmetry_error = self.advantage_stats['distribution_symmetry'][-1]
                    logger.info(f"  Distribution symmetry error: {symmetry_error:.3f} (0.0 = perfect symmetry)")
                    
                    # Provide guidance based on symmetry error
                    if symmetry_error > 0.1:
                        # logger.info(f"  WARNING: Large symmetry error! Check advantage normalization.")
                        # logger.info(f"  Consider: using robust normalization, checking clipping, or increasing batch size")
                        pass
                    elif symmetry_error > 0.05:
                        # logger.info(f"  NOTE: Moderate symmetry error. This is normal with finite batch sizes.")
                        pass
                    else:
                        # logger.info(f"  ✓ Good symmetry! Distribution is well-centered.")
                        pass
            
            if 'extreme_positive_fraction' in self.advantage_stats and self.advantage_stats['extreme_positive_fraction']:
                extreme_pos = self.advantage_stats['extreme_positive_fraction'][-1]
                extreme_neg = self.advantage_stats['extreme_negative_fraction'][-1]
                logger.info(f"  Extreme values: +{extreme_pos:.3f}, -{extreme_neg:.3f} (clipped at ±{self.config.advantage_clip_threshold})")
            
            if 'p5' in self.advantage_stats and self.advantage_stats['p5']:
                p5 = self.advantage_stats['p5'][-1]
                p95 = self.advantage_stats['p95'][-1]
                logger.info(f"  Percentiles (5th, 95th): {p5:.3f}, {p95:.3f}")
            
            if 'skew' in self.advantage_stats and self.advantage_stats['skew']:
                skew = self.advantage_stats['skew'][-1]
                kurtosis = self.advantage_stats['kurtosis'][-1]
                logger.info(f"  Distribution shape: skew={skew:.3f}, kurtosis={kurtosis:.3f}")
            
            # NEW: Show mean-median difference
            if 'mean_median_diff' in self.advantage_stats and self.advantage_stats['mean_median_diff']:
                mean_median_diff = self.advantage_stats['mean_median_diff'][-1]
                logger.info(f"  Mean-median difference: {mean_median_diff:.4f} (large values indicate skewness)")
        
        # Value function error
        if self.value_function_error['value_bias']:
            latest_bias = self.value_function_error['value_bias'][-1]
            logger.info(f"Value Bias: {latest_bias:.4f} (should be near 0)")
        
        # Mask density
        if self.mask_density['mean_legal_tokens']:
            latest_mask_density = self.mask_density['mean_legal_tokens'][-1]
            logger.info(f"Mask Density: {latest_mask_density:.1f} legal tokens/position")
        
        # Sequence length stats
        if self.sequence_length_stats['mean_length']:
            mean_len = self.sequence_length_stats['mean_length'][-1]
            std_len = self.sequence_length_stats['std_length'][-1]
            logger.info(f"Sequence Length: {mean_len:.1f} ± {std_len:.1f} (target: {self.config.len_target})")
        
        # Action distribution drift
        if self.action_distribution_drift['END_FIELD']:
            end_field_prob = self.action_distribution_drift['END_FIELD'][-1]
            end_itr_prob = self.action_distribution_drift['END_ITRACT'][-1]
            eos_prob = self.action_distribution_drift['EOS'][-1]
            logger.info(f"Critical Token Probs - END_FIELD: {end_field_prob:.4f}, END_ITRACT: {end_itr_prob:.4f}, EOS: {eos_prob:.4f}")
        
        # NEW: Parser failure rate
        if self.parser_failure_rate:
            latest_parser_failure_rate = self.parser_failure_rate[-1]
            logger.info(f"Parser Failure Rate: {latest_parser_failure_rate:.3f} ({latest_parser_failure_rate*100:.1f}%)")
            if latest_parser_failure_rate > 0.1:
                logger.info(f"  WARNING: High parser failure rate! Many sequences cannot be converted to FeynRules.")
                logger.info(f"  This indicates the model is generating malformed token sequences.")

        # NEW: Novelty Rewards
        if 'novelty_rewards' in self.reward_components and self.reward_components['novelty_rewards']:
            avg_novelty_reward = self.reward_components['novelty_rewards'][-1]
            logger.info(f"Novelty Reward (Avg): {avg_novelty_reward:.4f} (Weight: {self.config.novelty_reward_weight})")
        
        # logger.info("=" * 50)

    def get_latest_gradient_norm(self) -> float:
        """Get the latest gradient norm value."""
        return self.gradient_norms[-1] if self.gradient_norms else 0.0

    def update_kl_per_epoch(self, epoch_kl_values: List[float]):
        """Update KL divergence statistics per epoch."""
        if self.config.track_diagnostics and epoch_kl_values:
            self.kl_per_epoch['mean'].append(np.mean(epoch_kl_values))
            self.kl_per_epoch['max'].append(np.max(epoch_kl_values))
            self.kl_per_epoch['p95'].append(np.percentile(epoch_kl_values, 95))
    
    def update_parser_failure_rate(self, failure_rate: float):
        """Update parser failure rate tracking."""
        if self.config.track_diagnostics:
            self.parser_failure_rate.append(failure_rate)

    def save_state(self, filepath: str):
        """Save diagnostic tracker state to file, handling deque serialization."""
        # Convert deques to lists for serialization
        state_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                # Handle nested dictionaries (like advantage_stats, etc.)
                nested_dict = {}
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, deque):
                        nested_dict[nested_key] = list(nested_value)
                    else:
                        nested_dict[nested_key] = nested_value
                state_dict[key] = nested_dict
            elif isinstance(value, deque):
                state_dict[key] = list(value)
            else:
                state_dict[key] = value
        
        torch.save(state_dict, filepath)

    def load_state(self, filepath: str):
        """Load diagnostic tracker state from file, reconstructing deques."""
        state_dict = torch.load(filepath, map_location='cpu')
        
        for key, value in state_dict.items():
            if key == 'config' or key == 'check_names' or key == 'num_checks' or key == 'max_history_len':
                # These are not deques, just set directly
                setattr(self, key, value)
            elif isinstance(value, dict):
                # Handle nested dictionaries
                if hasattr(self, key) and isinstance(getattr(self, key), dict):
                    current_dict = getattr(self, key)
                    for nested_key, nested_value in value.items():
                        if nested_key in current_dict and isinstance(current_dict[nested_key], deque):
                            # Reconstruct deque with maxlen
                            maxlen = current_dict[nested_key].maxlen
                            current_dict[nested_key] = deque(nested_value, maxlen=maxlen)
                        else:
                            current_dict[nested_key] = nested_value
            elif isinstance(value, list) and hasattr(self, key):
                # Reconstruct deque from list
                current_attr = getattr(self, key)
                if isinstance(current_attr, deque):
                    maxlen = current_attr.maxlen
                    setattr(self, key, deque(value, maxlen=maxlen))
                else:
                    setattr(self, key, value)
            else:
                setattr(self, key, value)

# ======================================================================================
# 5. GrammarModelTrainer  – now with **incremental legality‑mask logic**
# ======================================================================================

class SequenceGenerator(nn.Module):
    """
    Sequence generator that uses KV-caching for efficient autoregressive generation.
    
    This class implements incremental sequence generation with proper KV-cache management
    to avoid recomputing attention keys and values for previously seen tokens.
    
    STATE UPDATE RULES:
    ===================
    
    1. GRAMMAR STATE TRACKING (gstate):
       - Uses bit flags to track current grammar context
       - INTERACTION_BIT (0b1000): Currently parsing an interaction block
       - FIELD_BIT (0b0100): Currently parsing a field block  
       - PARTICLE_BIT (0b0010): Currently parsing a particle block
       - States are updated via embed_val tensor: gstate += embed_val[next_token]
       - embed_val contains push/pop values: +8 for ITRACT, -8 for END_ITRACT, etc.
    
    2. FIELD MULTIPLICITY TRACKING:
       - field_counter: Counts particles within current field (resets on new field)
       - field_target: Expected number of particles = dim * gen (set when GEN token seen)
       - field_parsing_state: Tracks parsing progress within field:
         * 0: Initial state
         * 1: After FIELD token
         * 2: After FIELD_ID token  
         * 3: After TYPE token
         * 4: After DIM token
         * 5: After GEN token (sets field_target = dim * gen)
       - field_dim_cache: Stores DIM value for field_target calculation
       - field_gen_cache: Stores GEN value for field_target calculation
    
    3. INTERACTION/FIELD COUNTERS:
       - end_itr: Counts END_ITRACT tokens (must reach 2 before EOS allowed)
       - end_fld: Counts END_FIELD tokens (for validation)
    
    4. LEGALITY MASK CONSTRUCTION:
       - Uses adjacency stack (4xVxV) for different grammar contexts
       - Context 0: Model level (BOS, EOS, ITRACT)
       - Context 1: Interaction level (TYPE_YUKAWA, FIELD, END_ITRACT)  
       - Context 2: Field level (TYPE_*, DIM_*, GEN_*, PARTICLE, END_FIELD)
       - Context 3: Particle level (TYPE_*, MASS_*, CHARGE_*, END_PARTICLE)
       - Additional constraints:
         * EOS only allowed after 2 interactions (end_itr >= 2)
         * END_FIELD only allowed when field_counter >= field_target
         * DEAD token used as fallback when no legal tokens available
    """
    def __init__(self, model: DummyTransformerPolicy, max_len: int, bos_idx: int, dead_idx: int,
                 adj_stack: torch.Tensor, embed_val: torch.Tensor,
                 token_to_idx: Dict[str, int], end_itr_idx: int, end_field_idx: int, eos_idx: int):
        super().__init__()
        self.model = model
        self.max_len = max_len
        self.bos_idx = bos_idx
        self.dead_idx = dead_idx
        self.adj_stack = adj_stack  # Grammar adjacency matrices (4xVxV) for different contexts
        self.embed_val = embed_val  # Stack push/pop values for each token (V,)
        self.token_to_idx = token_to_idx
        self.end_itr_idx = end_itr_idx
        self.end_field_idx = end_field_idx
        self.eos_idx = eos_idx

        # Pre-compute token indices for vectorized operations
        self.field_idx = torch.tensor(self.token_to_idx['FIELD'], device=DEVICE)
        self.particle_idx = torch.tensor(self.token_to_idx['PARTICLE'], device=DEVICE)

        # Token indices for field parsing state updates
        self.dim_token_indices = torch.tensor([self.token_to_idx[f"DIM_{i}"] for i in range(1, DIM_RANGE)], device=DEVICE, dtype=torch.long)
        self.gen_token_indices = torch.tensor([self.token_to_idx[f"GEN_{i}"] for i in range(1, GEN_RANGE)], device=DEVICE, dtype=torch.long)
        self.field_id_indices = torch.tensor([self.token_to_idx[f"FIELD_ID_{i}"] for i in range(1, FIELD_ID_RANGE)], device=DEVICE, dtype=torch.long)
        self.type_indices = torch.tensor([self.token_to_idx[type_name] for type_name in ["TYPE_fermion", "TYPE_real", "TYPE_complex"]], device=DEVICE, dtype=torch.long)
        
        self.num_encoder_layers = len(self.model.encoder.layers)
        
        # Pre-compute dimension and generation token values as tensors for field_target calculation
        self.dim_token_values = torch.arange(1, DIM_RANGE, device=DEVICE, dtype=ITYPE)
        self.gen_token_values = torch.arange(1, GEN_RANGE, device=DEVICE, dtype=ITYPE)

    def forward(self, B: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate B sequences using grammar-constrained autoregressive generation.
        
        STATE VARIABLES TRACKED:
        ========================
        - gstate: Grammar state bit flags (INTERACTION_BIT | FIELD_BIT | PARTICLE_BIT)
        - end_itr: Number of END_ITRACT tokens seen (must reach 2 before EOS)
        - end_fld: Number of END_FIELD tokens seen (for validation)
        - field_counter: Particles counted in current field (resets on new field)
        - field_target: Expected particles = dim * gen (set when GEN token seen)
        - field_parsing_state: Field parsing progress (0-5, see rules above)
        - field_dim_cache: Cached DIM value for field_target calculation
        - field_gen_cache: Cached GEN value for field_target calculation
        - finished: Whether sequence has reached EOS or DEAD state
        """
        T = self.max_len
        V = self.model.vocab_size
        D_MODEL = self.model.d_model

        # Initialize output buffers
        seq_buf = torch.full((B, T), self.dead_idx, device=DEVICE, dtype=torch.long)
        logit_buf = torch.empty(B, T, V, device=DEVICE)
        val_buf = torch.empty(B, T, device=DEVICE)
        mask_buf = torch.empty(B, T, V, device=DEVICE, dtype=torch.bool)

        # Step 0 initialization - use cached step for BOS token
        seq_buf[:, 0] = self.bos_idx
        
        # Initialize KV cache for all layers to avoid recomputing attention
        cache = []
        for _ in range(self.num_encoder_layers):
            cache.append((
                torch.empty(B, 0, D_MODEL, device=DEVICE, dtype=DTYPE),
                torch.empty(B, 0, D_MODEL, device=DEVICE, dtype=DTYPE)
            ))
        
        # Use cached step for the first token
        step0_logits, step0_values, cache = self.model.step(seq_buf[:, 0], 0, cache)
        logit_buf[:, 0] = step0_logits
        val_buf[:, 0] = step0_values
        mask_buf[:, 0].fill_(True)

        # Initialize state tracking variables for all sequences
        t_init = torch.tensor(1, dtype=torch.int32, device=DEVICE)
        gstate_init = torch.zeros(B, dtype=torch.int32, device=DEVICE)  # Grammar state bit flags
        end_itr_init = torch.zeros(B, dtype=torch.int32, device=DEVICE)  # END_ITRACT counter
        end_fld_init = torch.zeros(B, dtype=torch.int32, device=DEVICE)  # END_FIELD counter
        finished_init = torch.zeros(B, dtype=torch.bool, device=DEVICE)  # Sequence completion flag
        
        # Field multiplicity tracking variables
        field_counter_init = torch.zeros(B, dtype=torch.int32, device=DEVICE)  # Particles in current field
        field_target_init = torch.zeros(B, dtype=torch.int32, device=DEVICE)   # Expected particles = dim * gen
        field_parsing_state_init = torch.zeros(B, dtype=torch.int32, device=DEVICE)  # Field parsing progress (0-5)
        field_dim_cache_init = torch.zeros(B, dtype=torch.int32, device=DEVICE)  # Cached DIM value
        field_gen_cache_init = torch.zeros(B, dtype=torch.int32, device=DEVICE)  # Cached GEN value

        prev_tok_init = seq_buf[:, 0]

        # Replace torch.while_loop with regular Python while loop to avoid compilation issues
        t = t_init
        seq_buf_var = seq_buf
        logit_buf_var = logit_buf
        val_buf_var = val_buf
        mask_buf_var = mask_buf
        gstate_var = gstate_init
        end_itr_var = end_itr_init
        end_fld_var = end_fld_init
        finished_var = finished_init
        cache_var = cache  # Use the initialized cache
        field_counter_var = field_counter_init
        field_target_var = field_target_init
        field_parsing_state_var = field_parsing_state_init
        field_dim_cache_var = field_dim_cache_init
        field_gen_cache_var = field_gen_cache_init
        prev_tok_var = prev_tok_init

        while t < T and not finished_var.all():
            # --- Build legality mask for step t (vectorized) ---
            # Determine grammar context for each sequence in batch
            context_idx = torch.zeros(B, dtype=torch.int32, device=DEVICE)
            mask_particle = (gstate_var & PARTICLE_BIT) != 0  # Currently parsing particle
            mask_field = (~mask_particle) & ((gstate_var & FIELD_BIT) != 0)  # Currently parsing field
            mask_itr = (~mask_particle & ~mask_field) & ((gstate_var & INTERACTION_BIT) != 0)  # Currently parsing interaction
            context_idx[mask_itr] = 1      # Interaction context
            context_idx[mask_field] = 2    # Field context  
            context_idx[mask_particle] = 3 # Particle context
            # context_idx[mask_model] = 0   # Model context (default)

            # Get legality mask from adjacency stack based on current context and previous token
            current_step_mask = self.adj_stack[context_idx, prev_tok_var]

            # Apply stack depth constraint: ensure grammar state remains valid
            current_step_mask = current_step_mask & ((gstate_var.unsqueeze(1) + self.embed_val) >= 0)
            
            # CONSTRAINT 1: EOS only allowed after 2 interactions
            eos_token_mask_v = (torch.arange(V, device=DEVICE) == self.eos_idx) # (V,)
            eos_token_mask_b_v = eos_token_mask_v.unsqueeze(0).expand(B, V) # (B,V)
            need_2_itrs = end_itr_var < 2 # (B,) - sequences that haven't seen 2 interactions yet
            current_step_mask = torch.where(need_2_itrs.unsqueeze(1), current_step_mask & (~eos_token_mask_b_v), current_step_mask)

            # CONSTRAINT 2: END_FIELD only allowed when field has enough particles
            in_field_context = (context_idx == 2)  # Currently parsing a field
            need_particles = field_counter_var < field_target_var  # Field needs more particles
            end_field_token_mask_v = (torch.arange(V, device=DEVICE) == self.end_field_idx) # (V,)
            end_field_token_mask_b_v = end_field_token_mask_v.unsqueeze(0).expand(B, V) # (B,V)
            forbid_end_field = in_field_context & need_particles  # In field context but need more particles
            current_step_mask = torch.where(forbid_end_field.unsqueeze(1), current_step_mask & (~end_field_token_mask_b_v), current_step_mask)

            # CONSTRAINT 3: Use DEAD token as fallback when no legal tokens available
            no_legal = ~current_step_mask.any(dim=1)  # Sequences with no legal tokens
            dead_token_mask_v = (torch.arange(V, device=DEVICE) == self.dead_idx) # (V,)
            dead_token_mask_b_v = dead_token_mask_v.unsqueeze(0).expand(B, V) # (B,V)
            current_step_mask = torch.where(no_legal.unsqueeze(1), current_step_mask | dead_token_mask_b_v, current_step_mask)

            mask_buf_var[:, t] = current_step_mask

            # --- Policy step (cached) ---
            step_logits, step_values, next_cache = self.model.step(prev_tok_var, t, cache_var)
            
            # Additional debugging for inf/nan values
            if torch.isnan(step_logits).any() or torch.isinf(step_logits).any():
                # Replace inf/nan with zeros
                step_logits = torch.where(
                    torch.isnan(step_logits) | torch.isinf(step_logits),
                    torch.zeros_like(step_logits),
                    step_logits
                )
            
            logit_buf_var[:, t] = step_logits
            val_buf_var[:, t] = step_values

            # --- Sample next token & update state ---
            # Apply legality mask to logits for sampling
            masked_logits = step_logits.masked_fill(~current_step_mask, -1e4)
            
            # Check for inf/nan values and handle them
            if torch.isnan(masked_logits).any() or torch.isinf(masked_logits).any():
                print(f"WARNING: Found inf/nan in masked_logits at t={t}!")
                # Replace inf/nan with large negative values
                masked_logits = torch.where(
                    torch.isnan(masked_logits) | torch.isinf(masked_logits),
                    torch.tensor(-1e4, device=masked_logits.device, dtype=masked_logits.dtype),
                    masked_logits
                )
            
            # Apply log_softmax for numerical stability, then exp
            log_probs = torch.log_softmax(masked_logits, dim=-1)
            probs = torch.exp(log_probs)
            
            # Ensure probabilities are valid (no inf/nan/negative)
            probs = torch.clamp(probs, min=1e-10, max=1.0)
            
            # Normalize to ensure sum = 1
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)
            
            # Final check for invalid probabilities before multinomial
            if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                print(f"ERROR: Invalid probabilities detected at t={t}!")
                print(f"Probs stats: min={probs.min()}, max={probs.max()}, mean={probs.mean()}")
                # Fallback to uniform distribution over legal tokens
                legal_probs = current_step_mask.float()
                legal_probs = legal_probs / (legal_probs.sum(dim=-1, keepdim=True) + 1e-10)
                probs = legal_probs
            
            next_tok = torch.multinomial(probs, 1).squeeze(1)
            
            # Don't sample new tokens for already finished sequences
            next_tok = torch.where(finished_var, torch.zeros_like(next_tok) + self.dead_idx, next_tok)
            
            seq_buf_var[:, t] = next_tok

            # Update finished flag: sequence ends on EOS or DEAD
            next_finished = finished_var | (next_tok == self.eos_idx) | (next_tok == self.dead_idx)

            # --- Update all states in tensor form (vectorized) ---
            # Initialize next state variables
            next_field_counter = field_counter_var
            next_field_target = field_target_var
            next_field_parsing_state = field_parsing_state_var
            next_field_dim_cache = field_dim_cache_var
            next_field_gen_cache = field_gen_cache_var
            next_gstate = gstate_var
            next_end_itr = end_itr_var
            next_end_fld = end_fld_var

            # RULE 1: Reset field tracking on new FIELD token
            is_new_field = next_tok == self.field_idx
            next_field_counter = torch.where(is_new_field, torch.zeros_like(next_field_counter), next_field_counter)
            next_field_target = torch.where(is_new_field, torch.zeros_like(next_field_target), next_field_target)
            next_field_parsing_state = torch.where(is_new_field, torch.ones_like(next_field_parsing_state), next_field_parsing_state)

            # RULE 2: Increment particle counter when PARTICLE token seen in field context
            is_particle = next_tok == self.particle_idx
            in_field_context_for_update = ((gstate_var & FIELD_BIT) != 0) & ~((gstate_var & PARTICLE_BIT) != 0)  # In field but not in particle
            field_particle = is_particle & in_field_context_for_update
            next_field_counter = torch.where(field_particle, next_field_counter + 1, next_field_counter)

            # RULE 3: Update field parsing state and cache DIM tokens
            for i in range(self.dim_token_indices.numel()):
                is_dim_token = next_tok == self.dim_token_indices[i]
                dim_token_val = self.dim_token_values[i].expand_as(next_field_dim_cache)
                next_field_dim_cache = torch.where(is_dim_token, dim_token_val, next_field_dim_cache)
                next_field_parsing_state = torch.where(is_dim_token, torch.zeros_like(next_field_parsing_state) + 4, next_field_parsing_state)

            # RULE 4: Update field parsing state, cache GEN tokens, and set field_target
            for i in range(self.gen_token_indices.numel()):
                is_gen_token = next_tok == self.gen_token_indices[i]
                gen_token_val = self.gen_token_values[i].expand_as(next_field_gen_cache)
                next_field_gen_cache = torch.where(is_gen_token, gen_token_val, next_field_gen_cache)
                next_field_parsing_state = torch.where(is_gen_token, torch.zeros_like(next_field_parsing_state) + 5, next_field_parsing_state)
                # Set field_target = dim * gen when GEN token is seen
                next_field_target = torch.where(is_gen_token, next_field_dim_cache * next_field_gen_cache, next_field_target)

            # RULE 5: Update field parsing state for FIELD_ID tokens
            for field_id_idx_val in self.field_id_indices:
                is_field_id = next_tok == field_id_idx_val
                next_field_parsing_state = torch.where(is_field_id, torch.zeros_like(next_field_parsing_state) + 2, next_field_parsing_state)

            # RULE 6: Update field parsing state for TYPE tokens
            for type_idx_val in self.type_indices:
                is_type = next_tok == type_idx_val
                next_field_parsing_state = torch.where(is_type, torch.zeros_like(next_field_parsing_state) + 3, next_field_parsing_state)

            # RULE 7: Update grammar state using embed_val (stack push/pop)
            next_gstate = gstate_var + self.embed_val[next_tok]
            
            # RULE 8: Update interaction and field counters
            next_end_itr = end_itr_var + (next_tok == self.end_itr_idx).int()
            next_end_fld = end_fld_var + (next_tok == self.end_field_idx).int()

            # RULE 9: Reset field tracking when exiting field (END_FIELD token)
            exited_field = next_tok == self.end_field_idx
            next_field_counter = torch.where(exited_field, torch.zeros_like(next_field_counter), next_field_counter)
            next_field_target = torch.where(exited_field, torch.zeros_like(next_field_target), next_field_target)
            next_field_parsing_state = torch.where(exited_field, torch.zeros_like(next_field_parsing_state), next_field_parsing_state)
            next_field_dim_cache = torch.where(exited_field, torch.zeros_like(next_field_dim_cache), next_field_dim_cache)
            next_field_gen_cache = torch.where(exited_field, torch.zeros_like(next_field_gen_cache), next_field_gen_cache)
            
            # Update loop variables for next iteration
            t = t + 1
            gstate_var = next_gstate
            end_itr_var = next_end_itr
            end_fld_var = next_end_fld
            finished_var = next_finished
            cache_var = next_cache  # Update cache for next iteration
            field_counter_var = next_field_counter
            field_target_var = next_field_target
            field_parsing_state_var = next_field_parsing_state
            field_dim_cache_var = next_field_dim_cache
            field_gen_cache_var = next_field_gen_cache
            prev_tok_var = next_tok
        
        # Return the final results
        return seq_buf_var, logit_buf_var, val_buf_var, mask_buf_var


class GrammarModelTrainer:
    def __init__(self, config: PPOConfig, logger, model: DummyTransformerPolicy): # Pass model to init
        self.config, self.logger = config, logger
        self.model = model # Store the raw model
        # Token ↔ index
        self.token_to_idx = {tok:i for i,tok in enumerate(ALL_TOKEN_NAMES)}
        self.idx_to_token = {i:tok for tok,i in self.token_to_idx.items()}
        self.vocab_size   = len(ALL_TOKEN_NAMES)

        # Special indices
        self.bos_idx  = self.token_to_idx['BOS']
        self.eos_idx  = self.token_to_idx['EOS']
        self.dead_idx = self.token_to_idx['DEAD']
        self.end_part_idx  = self.token_to_idx['END_PARTICLE']
        self.end_field_idx = self.token_to_idx['END_FIELD']
        self.end_itr_idx   = self.token_to_idx['END_ITRACT']

        # Grammar adjacency → (4, V, V) stacked tensor [MODEL, ITR, FIELD, PART]
        self.adj_stack = self._build_adjacency_stack()

        # Stack push/pop embedding values (V,)
        self.embed_val = self._build_embed_val_tensor()

        # Parser from previous file  
        self.parser = TokenSequenceParser()

        self.sequence_generator = SequenceGenerator(
            self.model,
            max_len=self.config.seq_len,
            bos_idx=self.bos_idx,
            dead_idx=self.dead_idx,
            adj_stack=self.adj_stack,
            embed_val=self.embed_val,
            token_to_idx=self.token_to_idx,
            end_itr_idx=self.end_itr_idx,
            end_field_idx=self.end_field_idx,
            eos_idx=self.eos_idx
        )
        
        # NEW: Novelty buffer for diversity reward
        self.novelty_buffer = deque(maxlen=self.config.novelty_buffer_size)

    # ------------------------------------------------------------------
    # Adjacency‑stack builder
    # ------------------------------------------------------------------
    def _build_adjacency_stack(self):
        # Ensure every token has DEAD fallback
        for token in ALL_TOKEN_NAMES:
            for ruleset in [TOKENS_MODEL, TOKENS_INTERACTION, TOKENS_FIELD, TOKENS_PARTICLE]:
                ruleset.setdefault(token, ['DEAD'])
        V = self.vocab_size
        def build_mat(rules):
            m = torch.zeros(V, V, dtype=torch.bool)
            for prev, nxt in rules.items():
                p = self.token_to_idx[prev]
                m[p, [self.token_to_idx[n] for n in nxt]] = True
            return m
        mats = [build_mat(r) for r in [TOKENS_MODEL, TOKENS_INTERACTION, TOKENS_FIELD, TOKENS_PARTICLE]]
        return torch.stack(mats, dim=0).to(DEVICE)   # (4,V,V)

    def _build_embed_val_tensor(self):
        t = torch.zeros(self.vocab_size, dtype=torch.int8)  # Changed from torch.long - values are small
        for tok,val in EMBED_VAL_MAP.items():
            t[self.token_to_idx[tok]] = val
        return t.to(DEVICE)
    
    @torch.no_grad()
    def calculate_novelty_bonus(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Calculates a bonus for sequences that are novel (not seen before in the buffer).
        Returns a tensor of 0s and 1s indicating novelty.
        """
        B = sequences.shape[0]
        device = sequences.device
        novelty_bonus = torch.zeros(B, device=device, dtype=torch.float32)

        # Convert sequences to hashable tuples, excluding DEAD tokens
        # Use CPU for hashing to avoid device transfer overhead per sequence
        seq_cpu_list = sequences.cpu().tolist() 
        current_hashes = []
        for b in range(B):
            # Extract alive tokens and convert to a tuple of integers
            # Important: Filter out PAD_VALUE (-1) or dead_idx if they are part of the raw sequence data
            # The dead_idx is used by the generator to fill unused portions, so we filter it here.
            alive_tokens = [idx for idx in seq_cpu_list[b] if idx != self.dead_idx]
            if len(alive_tokens) > 0:
                # Use str(tuple(...)) for hashing as hashlib requires bytes
                # Convert the tuple of token IDs to a string representation to hash
                seq_hash = hashlib.sha256(str(tuple(alive_tokens)).encode('utf-8')).hexdigest()
                current_hashes.append(seq_hash)
            else:
                current_hashes.append("") # Empty hash for empty/dead sequences

        # Check for novelty and update buffer
        for i, seq_hash in enumerate(current_hashes):
            if seq_hash and seq_hash not in self.novelty_buffer:
                novelty_bonus[i] = 1.0 # Assign bonus
                self.novelty_buffer.append(seq_hash) # Add to buffer
        return novelty_bonus

    def calculate_base_rewards(self, sequences: torch.Tensor) -> torch.Tensor:
        """Calculate simple rewards: small bonus for being alive, bigger bonus for good length."""
        B, T = sequences.shape
        is_alive = (sequences != self.dead_idx)
        rewards = torch.zeros_like(sequences, dtype=torch.float32)
        
        # Add alive reward with small penalty for excessive steps to discourage runaway length
        rewards += is_alive.float() * (self.config.alive_r - 0.001)  # net alive reward slightly smaller
        lengths = is_alive.sum(dim=1)
        
        # Gaussian bonus for being near the target length
        len_bonus = torch.exp(-0.5 * ((lengths - self.config.len_target) / self.config.len_sigma)**2)
        
        # Add length bonus to the reward at the last *alive* timestep
        last_alive_idx = lengths.clamp(min=1) - 1 # Ensure index is at least 0 for length 0 sequences
        rewards[torch.arange(B), last_alive_idx] += len_bonus
        
        # NEW: Add novelty bonus
        novelty_bonus = self.calculate_novelty_bonus(sequences) # This returns (B,) tensor
        rewards[torch.arange(B), last_alive_idx] += novelty_bonus * self.config.novelty_reward_weight
        
        return rewards

    @torch.no_grad()
    def calculate_terminal_reward(self, sequences: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Calculate terminal rewards for sequences ending with EOS using model-based scoring."""
        B, T = sequences.shape
        device = sequences.device

        eos_idx = self.eos_idx
        dead_idx = self.dead_idx

        seq_cpu = sequences.cpu().tolist()
        alive = sequences != dead_idx
        last_alive = alive.sum(dim=1).clamp(min=1) - 1
        last_list  = last_alive.tolist()

        idxs = torch.arange(B, device=device)
        eos_mask = sequences[idxs, last_alive] == eos_idx

        terminal_rewards = torch.zeros(B, T, device=device)
        term_idxs = torch.nonzero(eos_mask, as_tuple=False).squeeze(1)
        num_terminal = term_idxs.numel()

        if num_terminal > 0:
            # Score sequences ending with EOS using model-based scorer
            scores, total = model_based_scorer(sequences[term_idxs], self.idx_to_token, level='all')
            if total > 0:
                bs = term_idxs.to(device)
                ts = last_alive[term_idxs]
                # Use the raw scores (number of checks passed) as terminal rewards
                terminal_rewards[bs, ts] = scores

        return terminal_rewards, num_terminal

    def generate_deterministic_sequence(self, model: DummyTransformerPolicy) -> torch.Tensor:
        """Generate a single sequence using argmax decoding (greedy) with KV-caching."""
        model.eval()
        T = self.config.seq_len
        sequence = torch.full((1, T), self.dead_idx, dtype=torch.long, device=DEVICE)
        sequence[:, 0] = self.bos_idx
        
        # Initialize state tracking (same as in generate_sequences)
        gstate = torch.zeros(1, dtype=torch.int32, device=DEVICE)
        end_itr = torch.zeros(1, dtype=torch.int32, device=DEVICE)
        end_fld = torch.zeros(1, dtype=torch.int32, device=DEVICE)
        finished = torch.zeros(1, dtype=torch.bool, device=DEVICE)
        
        # Initialize KV cache for all layers
        cache = []
        for _ in range(len(model.encoder.layers)):
            cache.append((
                torch.empty(1, 0, model.d_model, device=DEVICE, dtype=DTYPE),
                torch.empty(1, 0, model.d_model, device=DEVICE, dtype=DTYPE)
            ))
        
        # Field multiplicity tracking (same as generate_sequences)
        field_counter = torch.zeros(1, dtype=torch.int32, device=DEVICE)
        field_target = torch.zeros(1, dtype=torch.int32, device=DEVICE)
        field_parsing_state = torch.zeros(1, dtype=torch.int32, device=DEVICE)
        field_dim_cache = torch.zeros(1, dtype=torch.int32, device=DEVICE)
        field_gen_cache = torch.zeros(1, dtype=torch.int32, device=DEVICE)
        
        # Pre-compute token indices for vectorized operations
        field_idx = self.token_to_idx['FIELD']
        particle_idx = self.token_to_idx['PARTICLE']
        end_field_idx = self.end_field_idx
        
        # Pre-compute dim and gen token indices for vectorized parsing
        dim_token_indices = [self.token_to_idx[f"DIM_{i}"] for i in range(1, DIM_RANGE)]
        gen_token_indices = [self.token_to_idx[f"GEN_{i}"] for i in range(1, GEN_RANGE)]
        field_id_indices = [self.token_to_idx[f"FIELD_ID_{i}"] for i in range(1, FIELD_ID_RANGE)]
        type_indices = [self.token_to_idx[type_name] for type_name in ["TYPE_fermion", "TYPE_real", "TYPE_complex"]]
        
        with torch.no_grad():
            prev_tok = sequence[:, 0]
            for t in range(1, T):
                # Build legality mask (same logic as generate_sequences)
                context_idx = torch.zeros(1, dtype=torch.int32, device=DEVICE)
                mask_particle = (gstate & PARTICLE_BIT) != 0
                mask_field = (~mask_particle) & ((gstate & FIELD_BIT) != 0)
                mask_itr = (~mask_particle & ~mask_field) & ((gstate & INTERACTION_BIT) != 0)
                context_idx[mask_itr] = 1
                context_idx[mask_field] = 2
                context_idx[mask_particle] = 3
                
                step_mask = self.adj_stack[context_idx, prev_tok]
                step_mask &= (gstate.unsqueeze(1) + self.embed_val) >= 0
                
                need_2_itrs = end_itr < 2
                if need_2_itrs.any():
                    step_mask[need_2_itrs, self.eos_idx] = False
                
                # Field multiplicity constraint
                in_field_context = (context_idx == 2)
                need_particles = field_counter < field_target
                step_mask[need_particles & in_field_context, end_field_idx] = False
                
                no_legal = ~step_mask.any(dim=1)
                step_mask[no_legal, self.dead_idx] = True
                
                # Policy step with caching
                logits, _, cache = model.step(prev_tok, t, cache)
                
                # Apply mask and argmax
                masked_logits = logits.masked_fill(~step_mask, -1e4)
                next_token = torch.argmax(masked_logits, dim=-1)
                sequence[:, t] = next_token
                
                # Field multiplicity tracking updates (vectorized)
                is_new_field = next_token == field_idx
                field_counter = torch.where(is_new_field, torch.zeros_like(field_counter), field_counter)
                field_target = torch.where(is_new_field, torch.zeros_like(field_target), field_target)
                field_parsing_state = torch.where(is_new_field, torch.ones_like(field_parsing_state), field_parsing_state)
                
                is_particle = next_token == particle_idx
                field_particle = is_particle & in_field_context
                field_counter = torch.where(field_particle, field_counter + 1, field_counter)
                
                # Vectorized field parsing state machine
                # Parse dim tokens (DIM_1, DIM_2, etc.)
                is_dim = torch.zeros(1, dtype=torch.bool, device=DEVICE)
                for i, dim_token_idx in enumerate(dim_token_indices, 1):
                    is_dim_token = next_token == dim_token_idx
                    is_dim |= is_dim_token
                    field_dim_cache = torch.where(is_dim_token, torch.zeros_like(field_dim_cache) + i, field_dim_cache)
                    field_parsing_state = torch.where(is_dim_token, torch.zeros_like(field_parsing_state) + 4, field_parsing_state)
                
                # Parse gen tokens (GEN_1, GEN_2, etc.)
                is_gen = torch.zeros(1, dtype=torch.bool, device=DEVICE)
                for i, gen_token_idx in enumerate(gen_token_indices, 1):
                    is_gen_token = next_token == gen_token_idx
                    is_gen |= is_gen_token
                    field_gen_cache = torch.where(is_gen_token, torch.zeros_like(field_gen_cache) + i, field_gen_cache)
                    field_parsing_state = torch.where(is_gen_token, torch.zeros_like(field_parsing_state) + 5, field_parsing_state)
                    
                    # Set the field target now that we have both dim and gen
                    field_target = torch.where(is_gen_token, field_dim_cache * field_gen_cache, field_target)
                
                # Update parsing state for other field tokens
                for field_id_idx in field_id_indices:
                    is_field_id = next_token == field_id_idx
                    field_parsing_state = torch.where(is_field_id, torch.zeros_like(field_parsing_state) + 2, field_parsing_state)
                
                # Update parsing state for type tokens
                for type_idx in type_indices:
                    is_type = next_token == type_idx
                    field_parsing_state = torch.where(is_type, torch.zeros_like(field_parsing_state) + 3, field_parsing_state)
                
                # Update state
                finished |= (next_token == self.eos_idx) | (next_token == self.dead_idx)
                gstate += self.embed_val[next_token]
                end_itr += (next_token == self.end_itr_idx)
                end_fld += (next_token == end_field_idx)
                
                # Reset field tracking when we exit a field
                exited_field = next_token == end_field_idx
                field_counter = torch.where(exited_field, torch.zeros_like(field_counter), field_counter)
                field_target = torch.where(exited_field, torch.zeros_like(field_target), field_target)
                field_parsing_state = torch.where(exited_field, torch.zeros_like(field_parsing_state), field_parsing_state)
                field_dim_cache = torch.where(exited_field, torch.zeros_like(field_dim_cache), field_dim_cache)
                field_gen_cache = torch.where(exited_field, torch.zeros_like(field_gen_cache), field_gen_cache)
                
                prev_tok = next_token
                
                if finished.all():
                    break
                    
        return sequence

    def compute_diversity_metrics(self, sequences: torch.Tensor, logits: torch.Tensor, masks: torch.Tensor) -> Dict[str, float]:
        """Compute various diversity metrics for the generated sequences."""
        B, T = sequences.shape  # sequences is (B, T), not (B, T, V)
        device = sequences.device
        
        # 1. Policy entropy (average entropy across all valid tokens)
        is_alive = sequences != self.dead_idx
        valid_positions = is_alive.sum().item()
        
        if valid_positions > 0:
            # Compute entropy for all positions, then average over valid ones
            probs = F.softmax(logits.masked_fill(~masks, -1e4), dim=-1)
            log_probs = F.log_softmax(logits.masked_fill(~masks, -1e4), dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)  # (B, T)
            avg_entropy = entropy[is_alive].mean().item()
        else:
            avg_entropy = 0.0
        
        # 2. Sequence uniqueness (how many unique sequences in the batch)
        # Convert sequences to strings for comparison, ignoring DEAD tokens
        unique_sequences = set()
        for b in range(B):
            seq_len = is_alive[b].sum().item()
            if seq_len > 0:
                # Convert to tuple of token indices (excluding DEAD)
                seq_tuple = tuple(sequences[b, :seq_len].tolist())
                unique_sequences.add(seq_tuple)
        
        uniqueness_ratio = len(unique_sequences) / B if B > 0 else 0.0
        
        # 3. Token diversity (how many different tokens are used)
        all_tokens = sequences[is_alive].unique()
        token_diversity = len(all_tokens) / self.vocab_size
        
        # 4. Structural diversity (diversity in sequence structure)
        structural_patterns = set()
        for b in range(B):
            seq_len = is_alive[b].sum().item()
            if seq_len > 0:
                tokens = [self.idx_to_token[i.item()] for i in sequences[b, :seq_len]]
                # Count structural elements
                num_particles = tokens.count('PARTICLE')
                num_fields = tokens.count('FIELD')
                num_interactions = tokens.count('ITRACT')
                pattern = (num_particles, num_fields, num_interactions)
                structural_patterns.add(pattern)
        
        structural_diversity = len(structural_patterns) / B if B > 0 else 0.0
        
        # 5. Length diversity (variance in sequence lengths)
        lengths = is_alive.sum(dim=1).float()
        length_diversity = lengths.std().item() / (lengths.mean().item() + 1e-8)
        
        # 6. Top-k token diversity (how concentrated are the most likely tokens)
        if valid_positions > 0:
            # Get top-5 probabilities for each position
            probs = F.softmax(logits.masked_fill(~masks, -1e4), dim=-1)
            top_k = 5
            top_probs, _ = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)
            top_k_concentration = top_probs.sum(dim=-1)[is_alive].mean().item()
        else:
            top_k_concentration = 0.0
        
        return {
            'avg_entropy': avg_entropy,
            'uniqueness_ratio': uniqueness_ratio,
            'token_diversity': token_diversity,
            'structural_diversity': structural_diversity,
            'length_diversity': length_diversity,
            'top_k_concentration': top_k_concentration,
            'num_unique_sequences': len(unique_sequences),
            'num_structural_patterns': len(structural_patterns)
        }

    def _generate_comprehensive_progress_plots(self, step, training_steps, all_checks_passed_history, 
                                             diversity_history, loss_history, sequence_length_history, 
                                             diagnostic_tracker, config, logger):
        """Generate comprehensive progress plots showing all available metrics."""
        if not all_checks_passed_history:
            return
            
        # Ensure log directory exists
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Create a comprehensive dashboard with multiple subplots - expanded to 16 plots
        fig = plt.figure(figsize=(24, 32))
        
        # Plot 1: Validation Performance (top left)
        plt.subplot(4, 4, 1)
        plt.plot(training_steps, all_checks_passed_history, 'b-', linewidth=2, label='All Checks Passed %')
        plt.xlabel('Training Step')
        plt.ylabel('Validation Pass Rate (%)')
        plt.title('Validation Performance')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if len(all_checks_passed_history) > 1:
            final_pct = all_checks_passed_history[-1]
            max_pct = max(all_checks_passed_history)
            avg_pct = np.mean(all_checks_passed_history)
            plt.text(0.02, 0.98, f'Final: {final_pct:.1f}%\nMax: {max_pct:.1f}%\nAvg: {avg_pct:.1f}%', 
                     transform=plt.gca().transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 2: Training Losses (top center)
        plt.subplot(4, 4, 2)
        if loss_history['pg_loss']:
            # Plot all losses
            plt.plot(training_steps, loss_history['pg_loss'], 'blue', linewidth=2, label='Policy Loss')
            plt.plot(training_steps, loss_history['v_loss'], 'red', linewidth=2, label='Value Loss')
            plt.plot(training_steps, loss_history['ent_loss'], 'green', linewidth=2, label='Entropy Loss')
            plt.plot(training_steps, loss_history['sup_loss'], 'orange', linewidth=2, label='Supervision Loss')
            # Plot average reward
            if loss_history['reward']:
                plt.plot(training_steps, loss_history['reward'], 'black', linewidth=2, label='Avg Reward')
            plt.xlabel('Training Step')
            plt.ylabel('Loss / Reward Value')
            plt.title('Training Losses and Reward')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.yscale('log')  # Use log scale for better visualization
        
        # Plot 3: Sequence Length (top right)
        plt.subplot(4, 4, 3)
        if sequence_length_history['mean_length']:
            plt.plot(training_steps, sequence_length_history['mean_length'], 'blue', linewidth=2, label='Mean Length')
            plt.fill_between(training_steps, 
                           [m - s for m, s in zip(sequence_length_history['mean_length'], sequence_length_history['std_length'])],
                           [m + s for m, s in zip(sequence_length_history['mean_length'], sequence_length_history['std_length'])],
                           alpha=0.3, color='blue', label='±1 Std Dev')
            plt.plot(training_steps, sequence_length_history['min_length'], 'green', linewidth=1, alpha=0.7, label='Min Length')
            plt.plot(training_steps, sequence_length_history['max_length'], 'red', linewidth=1, alpha=0.7, label='Max Length')
            plt.axhline(y=config.len_target, color='orange', linestyle='--', linewidth=2, label=f'Target Length ({config.len_target})')
            plt.xlabel('Training Step')
            plt.ylabel('Sequence Length')
            plt.title('Sequence Length Over Time')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add statistics text
            if len(sequence_length_history['mean_length']) > 1:
                final_avg_len = sequence_length_history['mean_length'][-1]
                final_std_len = sequence_length_history['std_length'][-1]
                min_len = min(sequence_length_history['min_length'])
                max_len = max(sequence_length_history['max_length'])
                plt.text(0.02, 0.98, f'Final: {final_avg_len:.1f} ± {final_std_len:.1f}\nMin: {min_len:.0f}\nMax: {max_len:.0f}', 
                         transform=plt.gca().transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Plot 4: Reward Components (top right)
        plt.subplot(4, 4, 4)
        if diagnostic_tracker is not None and diagnostic_tracker.reward_components['base_rewards']:
            base_rewards = list(diagnostic_tracker.reward_components['base_rewards'])
            # particle_rewards = list(diagnostic_tracker.reward_components['particle_rewards']) # Not in use, removed from plotting
            # field_rewards = list(diagnostic_tracker.reward_components['field_rewards']) # Not in use, removed from plotting
            terminal_rewards = list(diagnostic_tracker.reward_components['terminal_rewards'])
            novelty_rewards = list(diagnostic_tracker.reward_components['novelty_rewards']) # NEW: Novelty rewards
            
            reward_steps = list(range(1, len(base_rewards) + 1))
            
            plt.plot(reward_steps, base_rewards, 'blue', linewidth=2, label='Base Rewards')
            # Only plot particle_rewards if it has data
            # if particle_rewards:
            #     plt.plot(reward_steps, particle_rewards, 'red', linewidth=2, label='Particle Rewards')
            # Only plot field_rewards if it has data
            # if field_rewards:
            #     plt.plot(reward_steps, field_rewards, 'green', linewidth=2, label='Field Rewards')
            plt.plot(reward_steps, terminal_rewards, 'orange', linewidth=2, label='Terminal Rewards')
            plt.plot(reward_steps, novelty_rewards, 'red', linewidth=2, label='Novelty Rewards') # NEW: Plot novelty rewards
            plt.xlabel('Minibatch Update')
            plt.ylabel('Reward Value')
            plt.title('Reward Components Breakdown')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # Plot 5: KL Divergence (second row left)
        plt.subplot(4, 4, 5)
        if diagnostic_tracker is not None and hasattr(diagnostic_tracker, 'policy_update_stats'):
            full_dist_kl = diagnostic_tracker.policy_update_stats.get('full_distribution_kl', [])
            sampled_kl = diagnostic_tracker.policy_update_stats.get('sampled_action_kl', [])
            
            if full_dist_kl and sampled_kl:
                # Convert deques to lists for slicing
                full_dist_kl = list(full_dist_kl)
                sampled_kl = list(sampled_kl)
                
                # Ensure both arrays have the same length by truncating to the shorter one
                min_length = min(len(full_dist_kl), len(sampled_kl))
                full_dist_kl = full_dist_kl[:min_length]
                sampled_kl = sampled_kl[:min_length]
                
                # Create step indices for KL plotting (one per minibatch update)
                kl_steps = list(range(1, min_length + 1))
                
                plt.plot(kl_steps, full_dist_kl, 'blue', linewidth=2, label='Full-distribution KL')
                plt.plot(kl_steps, sampled_kl, 'red', linewidth=2, label='Sampled-action KL')
                plt.axhline(y=config.kl_target, color='green', linestyle='--', alpha=0.7, label=f'KL Target ({config.kl_target})')
                plt.xlabel('Minibatch Update')
                plt.ylabel('KL Divergence')
                plt.title('KL Divergence Over Time')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.yscale('log')  # Use log scale for better visualization
            elif full_dist_kl:
                # Only full-distribution KL available
                full_dist_kl = list(full_dist_kl)
                kl_steps = list(range(1, len(full_dist_kl) + 1))
                plt.plot(kl_steps, full_dist_kl, 'blue', linewidth=2, label='Full-distribution KL')
                plt.axhline(y=config.kl_target, color='green', linestyle='--', alpha=0.7, label=f'KL Target ({config.kl_target})')
                plt.xlabel('Minibatch Update')
                plt.ylabel('KL Divergence')
                plt.title('KL Divergence (Full-distribution only)')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.yscale('log')
            elif sampled_kl:
                # Only sampled-action KL available
                sampled_kl = list(sampled_kl)
                kl_steps = list(range(1, len(sampled_kl) + 1))
                plt.plot(kl_steps, sampled_kl, 'red', linewidth=2, label='Sampled-action KL')
                plt.axhline(y=config.kl_target, color='green', linestyle='--', alpha=0.7, label=f'KL Target ({config.kl_target})')
                plt.xlabel('Minibatch Update')
                plt.ylabel('KL Divergence')
                plt.title('KL Divergence (Sampled-action only)')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.yscale('log')
            else:
                plt.text(0.5, 0.5, 'KL divergence data\nnot available yet', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=plt.gca().transAxes, fontsize=12)
                plt.title('KL Divergence (No data)')
                plt.grid(True, alpha=0.3)
        
        # Plot 6: Gradient Norms (second row center-left)
        plt.subplot(4, 4, 6)
        if diagnostic_tracker is not None and diagnostic_tracker.gradient_norms:
            grad_norms = list(diagnostic_tracker.gradient_norms)
            grad_steps = list(range(1, len(grad_norms) + 1))
            plt.plot(grad_steps, grad_norms, 'purple', linewidth=2, label='Gradient Norm')
            plt.axhline(y=config.gradient_clip_norm, color='red', linestyle='--', alpha=0.7, label=f'Clip Threshold ({config.gradient_clip_norm})')
            plt.xlabel('Minibatch Update')
            plt.ylabel('Gradient Norm')
            plt.title('Gradient Norms Over Time')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.yscale('log')
        
        # Plot 7: Advantage Statistics (second row center-right)
        plt.subplot(4, 4, 7)
        if diagnostic_tracker is not None and diagnostic_tracker.advantage_stats['mean']:
            adv_mean = list(diagnostic_tracker.advantage_stats['mean'])
            adv_std = list(diagnostic_tracker.advantage_stats['std'])
            adv_steps = list(range(1, len(adv_mean) + 1))
            
            plt.plot(adv_steps, adv_mean, 'blue', linewidth=2, label='Mean Advantage')
            plt.fill_between(adv_steps, 
                           [m - s for m, s in zip(adv_mean, adv_std)],
                           [m + s for m, s in zip(adv_mean, adv_std)],
                           alpha=0.3, color='blue', label='±1 Std Dev')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Zero Line')
            plt.xlabel('Minibatch Update')
            plt.ylabel('Advantage Value')
            plt.title('Advantage Distribution Statistics')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # Plot 8: Policy Entropy and Diversity (second row right)
        plt.subplot(4, 4, 8)
        if diversity_history['avg_entropy']:
            plt.plot(training_steps, diversity_history['avg_entropy'], 'g-', linewidth=2, label='Avg Policy Entropy')
            plt.plot(training_steps, diversity_history['token_diversity'], 'r-', linewidth=2, label='Token Diversity')
            plt.xlabel('Training Step')
            plt.ylabel('Diversity Metrics')
            plt.title('Policy Entropy & Token Diversity')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # Plot 9: Sequence Diversity Metrics (third row left)
        plt.subplot(4, 4, 9)
        if diversity_history['uniqueness_ratio']:
            plt.plot(training_steps, diversity_history['uniqueness_ratio'], 'blue', linewidth=2, label='Sequence Uniqueness')
            plt.plot(training_steps, diversity_history['structural_diversity'], 'red', linewidth=2, label='Structural Diversity')
            plt.plot(training_steps, diversity_history['token_diversity'], 'green', linewidth=2, label='Token Diversity')
            plt.plot(training_steps, diversity_history['length_diversity'], 'orange', linewidth=2, label='Length Diversity')
            plt.xlabel('Training Step')
            plt.ylabel('Diversity Ratio')
            plt.title('Sequence Diversity Metrics')
            plt.grid(True, alpha=0.3)
            plt.legend()
            # Add warning threshold for uniqueness
            plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Low Diversity Warning (50%)')
        
        # Plot 10: Value Function Error (third row center-left)
        plt.subplot(4, 4, 10)
        if diagnostic_tracker is not None and diagnostic_tracker.value_function_error['value_bias']:
            value_bias = list(diagnostic_tracker.value_function_error['value_bias'])
            value_steps = list(range(1, len(value_bias) + 1))
            plt.plot(value_steps, value_bias, 'orange', linewidth=2, label='Value Bias')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Zero Line')
            plt.xlabel('Minibatch Update')
            plt.ylabel('Value Bias')
            plt.title('Value Function Error')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # Plot 11: Clipping Fraction (third row center-right)
        plt.subplot(4, 4, 11)
        if diagnostic_tracker is not None and diagnostic_tracker.clipping_fraction:
            clip_frac = list(diagnostic_tracker.clipping_fraction)
            clip_steps = list(range(1, len(clip_frac) + 1))
            plt.plot(clip_steps, clip_frac, 'brown', linewidth=2, label='Clipping Fraction')
            plt.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='High Clipping Warning (30%)')
            plt.axhline(y=0.05, color='green', linestyle='--', alpha=0.7, label='Low Clipping Warning (5%)')
            plt.xlabel('Minibatch Update')
            plt.ylabel('Clipping Fraction')
            plt.title('Policy Clipping Fraction')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # Plot 12: Mask Density (third row right)
        plt.subplot(4, 4, 12)
        if diagnostic_tracker is not None and diagnostic_tracker.mask_density['mean_legal_tokens']:
            mask_density = list(diagnostic_tracker.mask_density['mean_legal_tokens'])
            mask_steps = list(range(1, len(mask_density) + 1))
            plt.plot(mask_steps, mask_density, 'teal', linewidth=2, label='Mean Legal Tokens')
            plt.xlabel('Minibatch Update')
            plt.ylabel('Legal Tokens per Position')
            plt.title('Mask Density')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # Plot 14: Timing Breakdown (fourth row center-left)
        plt.subplot(4, 4, 14)
        if diagnostic_tracker is not None and diagnostic_tracker.timing_breakdown['rollout_time']:
            timing_data = {
                'Rollout': list(diagnostic_tracker.timing_breakdown['rollout_time']),
                'Validation': list(diagnostic_tracker.timing_breakdown['validation_time']),
                'Reward Calc': list(diagnostic_tracker.timing_breakdown['reward_calc_time']),
                'PPO': list(diagnostic_tracker.timing_breakdown['ppo_time']),
                'VF': list(diagnostic_tracker.timing_breakdown['vf_time']),
                'Logging': list(diagnostic_tracker.timing_breakdown['logging_time'])
            }
            
            timing_steps = list(range(1, len(timing_data['Rollout']) + 1))
            
            for name, data in timing_data.items():
                if data:
                    plt.plot(timing_steps, data, linewidth=2, label=name)
            
            plt.xlabel('Minibatch Update')
            plt.ylabel('Time (seconds)')
            plt.title('Timing Breakdown')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.yscale('log')
        
        # Plot 15: Action Distribution Drift (fourth row center-right)
        plt.subplot(4, 4, 15)
        if diagnostic_tracker is not None and diagnostic_tracker.action_distribution_drift['PARTICLE']:
            action_data = {
                'PARTICLE': list(diagnostic_tracker.action_distribution_drift['PARTICLE']),
                'FIELD': list(diagnostic_tracker.action_distribution_drift['FIELD']),
                'ITRACT': list(diagnostic_tracker.action_distribution_drift['ITRACT']),
                'END_FIELD': list(diagnostic_tracker.action_distribution_drift['END_FIELD']),
                'END_ITRACT': list(diagnostic_tracker.action_distribution_drift['END_ITRACT']),
                'EOS': list(diagnostic_tracker.action_distribution_drift['EOS'])
            }
            
            action_steps = list(range(1, len(action_data['PARTICLE']) + 1))
            
            for name, data in action_data.items():
                if data:
                    plt.plot(action_steps, data, linewidth=2, label=name)
            
            plt.xlabel('Minibatch Update')
            plt.ylabel('Action Probability')
            plt.title('Action Distribution Drift')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # Plot 16: PG Loss Components (fourth row right)
        plt.subplot(4, 4, 16)
        if diagnostic_tracker is not None and diagnostic_tracker.pg_loss_components['unclipped_loss']:
            unclipped_loss = list(diagnostic_tracker.pg_loss_components['unclipped_loss'])
            clipped_loss = list(diagnostic_tracker.pg_loss_components['clipped_loss'])
            actual_loss = list(diagnostic_tracker.pg_loss_components['actual_loss'])
            
            pg_steps = list(range(1, len(unclipped_loss) + 1))
            
            plt.plot(pg_steps, unclipped_loss, 'blue', linewidth=2, label='Unclipped Loss')
            plt.plot(pg_steps, clipped_loss, 'red', linewidth=2, label='Clipped Loss')
            plt.plot(pg_steps, actual_loss, 'green', linewidth=2, label='Actual Loss')
            plt.xlabel('Minibatch Update')
            plt.ylabel('Loss Value')
            plt.title('PG Loss Components')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.yscale('log')
        
        # Add overall title
        plt.suptitle(f'Comprehensive Training Progress Dashboard - Step {step}', fontsize=16, y=0.98)
        
        # Adjust layout and save
        plt.tight_layout()
        progress_plot_path = os.path.join(config.log_dir, f"progress_step_{step}_{config.run_name}.png")
        plt.savefig(progress_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Comprehensive progress dashboard saved as: {progress_plot_path}")
        
        # Also save individual plots for easier viewing
        self._save_individual_plots(step, training_steps, all_checks_passed_history, 
                                  diversity_history, loss_history, sequence_length_history, 
                                  diagnostic_tracker, config, logger)

    def _save_individual_plots(self, step, training_steps, all_checks_passed_history, 
                             diversity_history, loss_history, sequence_length_history, 
                             diagnostic_tracker, config, logger):
        """Save individual plots for easier viewing."""
        
        # Plot 1: Validation pass rates
        plt.figure(figsize=(12, 8))
        plt.plot(training_steps, all_checks_passed_history, 'b-', linewidth=2, label='All Checks Passed %')
        plt.xlabel('Training Step')
        plt.ylabel('Percentage of Sequences Passing All Checks (%)')
        plt.title(f'Validation Performance Over Time (Step {step})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if len(all_checks_passed_history) > 1:
            final_pct = all_checks_passed_history[-1]
            max_pct = max(all_checks_passed_history)
            avg_pct = np.mean(all_checks_passed_history)
            plt.text(0.02, 0.98, f'Final: {final_pct:.1f}%\nMax: {max_pct:.1f}%\nAvg: {avg_pct:.1f}%', 
                     transform=plt.gca().transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        validation_plot_path = os.path.join(config.log_dir, f"validation_performance_step_{step}_{config.run_name}.png")
        plt.savefig(validation_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Validation performance plot saved as: {validation_plot_path}")
        
        # Plot 2: Training losses
        if loss_history['pg_loss']:
            plt.figure(figsize=(12, 8))

            # Plot all losses
            plt.plot(training_steps, loss_history['pg_loss'], 'blue', linewidth=2, label='Policy Loss')
            plt.plot(training_steps, loss_history['v_loss'], 'red', linewidth=2, label='Value Loss')
            plt.plot(training_steps, loss_history['ent_loss'], 'green', linewidth=2, label='Entropy Loss')
            plt.plot(training_steps, loss_history['sup_loss'], 'orange', linewidth=2, label='Supervision Loss')
            # Plot average reward
            if loss_history['reward']:
                plt.plot(training_steps, loss_history['reward'], 'black', linewidth=2, label='Avg Reward')
            plt.xlabel('Training Step')
            plt.ylabel('Loss / Reward Value')
            plt.title(f'Training Losses and Reward Over Time (Step {step})')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.yscale('log')  # Use log scale for better visualization
            
            losses_plot_path = os.path.join(config.log_dir, f"training_losses_step_{step}_{config.run_name}.png")
            plt.savefig(losses_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"✓ Training losses plot saved as: {losses_plot_path}")
        
        # Plot 3: Sequence length over time
        if sequence_length_history['mean_length']:
            plt.figure(figsize=(12, 8))
            plt.plot(training_steps, sequence_length_history['mean_length'], 'blue', linewidth=2, label='Mean Length')
            plt.fill_between(training_steps, 
                           [m - s for m, s in zip(sequence_length_history['mean_length'], sequence_length_history['std_length'])],
                           [m + s for m, s in zip(sequence_length_history['mean_length'], sequence_length_history['std_length'])],
                           alpha=0.3, color='blue', label='±1 Std Dev')
            plt.plot(training_steps, sequence_length_history['min_length'], 'green', linewidth=1, alpha=0.7, label='Min Length')
            plt.plot(training_steps, sequence_length_history['max_length'], 'red', linewidth=1, alpha=0.7, label='Max Length')
            plt.axhline(y=config.len_target, color='orange', linestyle='--', linewidth=2, label=f'Target Length ({config.len_target})')
            plt.xlabel('Training Step')
            plt.ylabel('Sequence Length')
            plt.title(f'Sequence Length Over Time (Step {step})')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add statistics text
            if len(sequence_length_history['mean_length']) > 1:
                final_avg_len = sequence_length_history['mean_length'][-1]
                final_std_len = sequence_length_history['std_length'][-1]
                min_len = min(sequence_length_history['min_length'])
                max_len = max(sequence_length_history['max_length'])
                plt.text(0.02, 0.98, f'Final: {final_avg_len:.1f} ± {final_std_len:.1f}\nMin: {min_len:.0f}\nMax: {max_len:.0f}', 
                         transform=plt.gca().transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            sequence_length_plot_path = os.path.join(config.log_dir, f"sequence_length_step_{step}_{config.run_name}.png")
            plt.savefig(sequence_length_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"✓ Sequence length plot saved as: {sequence_length_plot_path}")

# ======================================================================================
# 6. GAE
# ======================================================================================
@torch.jit.script
def compute_gae_jit(rewards: torch.Tensor, values: torch.Tensor, is_alive: torch.Tensor, gamma: float, gae_lambda: float):
    """Compute Generalized Advantage Estimation (GAE) for the given rewards and values."""
    B, T = rewards.shape
    adv  = torch.zeros_like(rewards)
    last = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
    is_alive_f = is_alive.float()
    
    if torch.isnan(rewards).any() or torch.isinf(rewards).any():
        # Replace NaN/Inf rewards with zeros
        rewards = torch.where(
            torch.isnan(rewards) | torch.isinf(rewards),
            torch.zeros_like(rewards),
            rewards
        )
    if torch.isnan(values).any() or torch.isinf(values).any():
        # Replace NaN/Inf values with zeros
        values = torch.where(
            torch.isnan(values) | torch.isinf(values),
            torch.zeros_like(values),
            values
        )
    
    for t in range(T-2, -1, -1):
        next_alive = is_alive_f[:,t+1]
        next_val   = values[:,t+1] * next_alive
        delta = rewards[:,t] + gamma*next_val - values[:,t]
        
        # CRITICAL FIX: Check for NaN/Inf in delta before updating last
        if torch.isnan(delta).any() or torch.isinf(delta).any():
            # Replace NaN/Inf delta with zeros
            delta = torch.where(
                torch.isnan(delta) | torch.isinf(delta),
                torch.zeros_like(delta),
                delta
            )
        
        last  = delta + gamma*gae_lambda*last*next_alive
        
        # CRITICAL FIX: Check for NaN/Inf in last before assigning to adv
        if torch.isnan(last).any() or torch.isinf(last).any():
            # Replace NaN/Inf last with zeros
            last = torch.where(
                torch.isnan(last) | torch.isinf(last),
                torch.zeros_like(last),
                last
            )
        
        adv[:,t] = last
    
    return adv

def save_checkpoint(config: PPOConfig, step: int, model: nn.Module, optimizer: torch.optim.Optimizer, 
                   scaler: torch.amp.GradScaler, 
                   diagnostic_tracker, training_steps: List[int], all_checks_passed_history: List[float],
                   diversity_history: Dict[str, List[float]], loss_history: Dict[str, List[float]], 
                   sequence_length_history: Dict[str, List[float]],
                   logger, checkpoint_dir: str = None):
    """Save complete training state to a checkpoint file."""
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(config.log_dir, "checkpoints")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model state (use raw model if compiled)
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
    
    checkpoint = {
        'step': step,
        'config': config,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'training_steps': training_steps,
        'all_checks_passed_history': all_checks_passed_history,
        'diversity_history': diversity_history,
        'loss_history': loss_history,
        'sequence_length_history': sequence_length_history,
        # Removed ema_pass_rates as it's for validation bonus
    }
    
    # Save diagnostic tracker if it exists
    if diagnostic_tracker is not None:
        diagnostic_path = os.path.join(checkpoint_dir, f"diagnostic_tracker_step_{step}.pt")
        diagnostic_tracker.save_state(diagnostic_path)
        checkpoint['diagnostic_tracker_path'] = diagnostic_path
    
    # Save main checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    logger.info(f"✓ Checkpoint saved to {checkpoint_path}")
    if diagnostic_tracker is not None:
        logger.info(f"✓ Diagnostic tracker saved to {diagnostic_path}")
    
    return checkpoint_path

def load_checkpoint(config: PPOConfig, model: nn.Module, optimizer: torch.optim.Optimizer, 
                   scaler: torch.amp.GradScaler, 
                   logger, checkpoint_path: str):
    """Load complete training state from a checkpoint file."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Load model state
    model_to_load = model._orig_mod if hasattr(model, '_orig_mod') else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scaler state
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    # Load diagnostic tracker if it exists
    diagnostic_tracker = None
    if 'diagnostic_tracker_path' in checkpoint:
        diagnostic_path = checkpoint['diagnostic_tracker_path']
        try:
            # We'll need to reconstruct the diagnostic tracker in the main function
            # since we need the check_names which aren't saved in the state
            logger.info(f"Diagnostic tracker state found at {diagnostic_path}")
            # The actual loading will be done in main() after creating the tracker
        except Exception as e:
            logger.warning(f"Failed to load diagnostic tracker state: {e}")
    
    # Extract training state
    step = checkpoint['step']
    training_steps = checkpoint['training_steps']
    all_checks_passed_history = checkpoint['all_checks_passed_history']
    diversity_history = checkpoint['diversity_history']
    loss_history = checkpoint['loss_history']
    sequence_length_history = checkpoint['sequence_length_history']
    # Removed ema_pass_rates
    
    logger.info(f"✓ Checkpoint loaded successfully from step {step}")
    
    return (step, training_steps, all_checks_passed_history, diversity_history, 
            loss_history, sequence_length_history, diagnostic_tracker) # Removed ema_pass_rates from return

# ======================================================================================
# 7. main() – orchestrates everything 
# ======================================================================================

def print_step_table(logger, table_title, headers, rows):
    """Print a formatted table with headers and rows."""
    logger.info(f"\n{table_title}")
    logger.info("=" * 80)
    
    # Print headers
    header_str = " | ".join(f"{h:>15}" for h in headers)
    logger.info(header_str)
    logger.info("-" * len(header_str))
    
    # Print rows
    for row in rows:
        row_str = " | ".join(f"{str(r):>15}" for r in row)
        logger.info(row_str)
    logger.info("=" * 80)

def test_sm_integration(logger):
    """
    Test the integration between bufffr.py and token2fr.py using SM tokens.
    This validates that the model-based scorer works correctly with the updated token2fr.py.
    """
    if not SM_TOKENS_AVAILABLE:
        logger.warning("SM tokens not available - skipping integration test")
        return False
    
    logger.info("=" * 80)
    logger.info("TESTING SM INTEGRATION WITH TOKEN2FR")
    logger.info("=" * 80)
    
    try:
        # Test 1: Direct token2fr conversion (this already handles legacy tokens)
        logger.info("Test 1: Direct token2fr conversion...")
        tester = ModelTester()
        model_dict = tester.build_model(SM)
        logger.info(f"✅ Model built successfully from SM tokens!")
        logger.info(f"   • Gauge groups: {len(model_dict.get('GaugeGroups', []))}")
        logger.info(f"   • Particles: {len(model_dict.get('particles', []))}")
        logger.info(f"   • Fields: {len(model_dict.get('fields', []))}")
        logger.info(f"   • Interactions: {len(model_dict.get('interactions', []))}")
        
        # Test 2: Convert SM tokens to new format and use model-based scorer
        logger.info("\nTest 2: Converting SM tokens to new format and using model-based scorer...")
        
        # Get the token vocabulary from bufffr.py
        token_to_idx = {tok: i for i, tok in enumerate(ALL_TOKEN_NAMES)}
        
        # Convert legacy SM tokens to new format tokens
        sm_new_format = []
        conversion_map = {
            # Type tokens
            'TYPE_FERMION': 'TYPE_fermion',
            'TYPE_COMPLEX': 'TYPE_complex',
            'TYPE_real': 'TYPE_real',
            
            # Representation tokens
            'REP_SU3C_singlet': 'SU3C_REP_1',
            'REP_SU3C_fnd': 'SU3C_REP_3',
            'REP_SU2L_singlet': 'SU2L_REP_1',
            'REP_SU2L_fnd': 'SU2L_REP_2',
            'REP_U1Y_-2': 'U1Y_CHARGE_-2',
            'REP_U1Y_-1': 'U1Y_CHARGE_-1',
            'REP_U1Y_0.3333333': 'U1Y_CHARGE_0',  # Approximate
            'REP_U1Y_1': 'U1Y_CHARGE_1',
            'REP_U1Y_1.3333333': 'U1Y_CHARGE_2',  # Approximate
            'REP_U1Y_-0.6666667': 'U1Y_CHARGE_-1',  # Approximate
            
            # Quantum number tokens
            'QN_LeptonNumber_0': 'QN_L_0',
            'QN_BaryonNumber_0': 'QN_B_0',
            
            # Charge tokens
            'CHARGE_-1': 'CHARGE_0',  # Map to available charge
            
            # Chirality tokens
            'CHIRALITY_na': 'CHIRALITY_none',
            
            # Mass tokens
            'MASS_1e0': 'MASS_1e2',  # Map to available mass
        }
        
        # Convert SM tokens to new format
        for token in SM:
            if token in conversion_map:
                new_token = conversion_map[token]
                sm_new_format.append(new_token)
            elif token in token_to_idx:
                # Token already in new format
                sm_new_format.append(token)
            else:
                # For particle names and other tokens not in vocabulary, use a placeholder
                # that won't break the sequence structure
                if token in ['Phi', 'c', 'vt', 'uR', 'tau', 'mu', 've', 'phi_p', 'phi_0', 'b', 'd', 'dR', 'e', 'eR', 's', 't', 'u', 'vm', 'LL', 'QL', 'Yukawa_d', 'Yukawa_e', 'Yukawa_u']:
                    # Use a BOS placeholder for unknown tokens that could break parsing if not handled
                    sm_new_format.append('BOS')
                else:
                    sm_new_format.append(token)
        
        # Convert to indices
        sm_indices = []
        missing_tokens = []
        
        for token in sm_new_format:
            if token in token_to_idx:
                sm_indices.append(token_to_idx[token])
            else:
                missing_tokens.append(token)
                sm_indices.append(0)  # placeholder
        
        if missing_tokens:
            logger.warning(f"⚠️  Still missing tokens after conversion: {len(missing_tokens)}")
            logger.warning(f"   First few missing: {missing_tokens[:5]}")
        
        # Convert to tensor and score
        sm_sequence = torch.tensor([sm_indices], device='cpu')
        
        # Score using model-based scorer
        scores, num_passed = model_based_scorer(
            sequences=sm_sequence,
            idx_to_token={i: tok for tok, i in token_to_idx.items()},
            level="all",
            verbose=True
        )
        
        score_value = scores[0].item()
        logger.info(f"✅ Model-based scoring completed!")
        logger.info(f"   • Score: {score_value}")
        logger.info(f"   • Checks passed: {num_passed}")
        logger.info(f"   • Original sequence length: {len(SM)}")
        logger.info(f"   • Converted sequence length: {len(sm_new_format)}")
        
        # Test 3: Verify expected score (44/51 from test_sm_validation.py)
        expected_score = 44.0  # Expected score from test_sm_validation.py
        expected_total = 51.0  # Total checks from test_sm_validation.py
        
        logger.info(f"\nTest 3: Score validation...")
        logger.info(f"   • Expected score: {expected_score}/{expected_total} ({expected_score/expected_total*100:.1f}%)")
        logger.info(f"   • Actual score: {score_value}")
        
        if abs(score_value - expected_score) < 5.0:  # Allow larger tolerance due to token conversion
            logger.info(f"✅ Score validation PASSED! Integration working correctly.")
            return True
        else:
            logger.warning(f"⚠️  Score validation FAILED! Expected ~{expected_score}, got {score_value}")
            logger.warning(f"   This may indicate an issue with the integration or token conversion.")
            return False
            
    except Exception as e:
        logger.error(f"❌ SM integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # ------------------------------------------------------------------
    # 0. Set-up – instantiate config, logger, trainer, model, optimiser
    # 
    # This implementation uses KV-caching for efficient autoregressive generation,
    # which significantly reduces computation time during sequence generation by
    # avoiding recomputation of attention keys and values for previously seen tokens.
    # ------------------------------------------------------------------
    config = PPOConfig()
    
    # Check if we should load from a checkpoint
    if config.checkpoint_mode == "old" and config.checkpoint_path:
        logger = DummyLogger()
        logger.info(f"Loading from checkpoint: {config.checkpoint_path}")
        
        if not os.path.exists(config.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {config.checkpoint_path}")
        
        checkpoint = torch.load(config.checkpoint_path, map_location=DEVICE)
        loaded_config = checkpoint['config']
        loaded_step = checkpoint['step']
        
        # Update our config with the loaded values
        config.run_name = loaded_config.run_name
        config.checkpoint_step = loaded_step
        
        # NEW: Load novelty_reward_weight and novelty_buffer_size from loaded config
        config.novelty_reward_weight = loaded_config.novelty_reward_weight
        config.novelty_buffer_size = loaded_config.novelty_buffer_size

        logger.info(f"Will resume training from step {loaded_step}")
    
    logger = DummyLogger()
    logger.info(f"Starting optimized blueprint run '{config.run_name}' on device {DEVICE}")

    # Run SM integration test before training starts
    logger.info("Running SM integration test...")
    test_passed = test_sm_integration(logger)
    if not test_passed:
        logger.warning("SM integration test failed, but continuing with training...")
    logger.info("SM integration test completed.")

    # Print initial memory usage before training starts
    print_memory_usage(logger, "BEFORE TRAINING")

    # Create the raw model first
    raw_model = DummyTransformerPolicy(
        len(ALL_TOKEN_NAMES), d_model=256, max_T=config.seq_len, nhead=8
    ).to(DEVICE)
    
    # Pass the raw_model to the trainer init
    trainer = GrammarModelTrainer(config, logger, raw_model)

    # Take a snapshot of the raw model BEFORE compilation (now `old_raw_model` will be used for old policy in PPO)
    old_raw_model = DummyTransformerPolicy(
        len(ALL_TOKEN_NAMES), d_model=256, max_T=config.seq_len, nhead=8
    ).to(DEVICE)
    old_raw_model.load_state_dict(raw_model.state_dict())  # Efficient parameter copying
    old_raw_model.eval()
    
    # No need to compile `model` here, `trainer.sequence_generator` is compiled internally.
    # The `model` variable can just be `raw_model` since it's passed into the trainer.
    model = raw_model

    # Log model architecture details
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model architecture: vocab_size={trainer.vocab_size}, d_model=256, nhead=8, num_layers=8")
    logger.info(f"Total parameters: {total_params:,}, Trainable parameters: {trainable_params:,}")

    # Separate policy vs value learning rates to prevent value function blow-up
    policy_params = [p for n, p in raw_model.named_parameters() if "value_head" not in n]
    value_params = [p for n, p in raw_model.named_parameters() if "value_head" in n]
    
    optimizer = optim.AdamW([
        {"params": policy_params, "lr": config.learning_rate},
        {"params": value_params, "lr": config.learning_rate * config.value_lr_multiplier},
    ])
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    logger.info(f"Optimizer set up with separate learning rates:")
    logger.info(f"  Policy parameters: {config.learning_rate}")
    logger.info(f"  Value parameters: {config.learning_rate * config.value_lr_multiplier}")
    logger.info(f"  Policy params: {sum(p.numel() for p in policy_params):,}")
    logger.info(f"  Value params: {sum(p.numel() for p in value_params):,}")

    # Initialize diagnostic tracker (will be set up after first validation)
    diagnostic_tracker = None

    # Tracking variables for plotting and EMA
    all_checks_passed_history = []
    training_steps = []
    # Removed ema_pass_rates as it's for validation bonus
    
    # Loss tracking variables
    loss_history = {
        'pg_loss': [],
        'v_loss': [],
        'ent_loss': [],
        'sup_loss': [],
        'her_loss': [],
        'div_loss': [],
        'reward': []  # Track average reward per sequence
    }
    
    # Diversity tracking variables
    diversity_history = {
        'avg_entropy': [],
        'uniqueness_ratio': [],
        'token_diversity': [],
        'structural_diversity': [],
        'length_diversity': [],
        'top_k_concentration': [],
        'num_unique_sequences': [],
        'num_structural_patterns': []
    }
    
    # Sequence length tracking variables
    sequence_length_history = {
        'mean_length': [],
        'std_length': [],
        'min_length': [],
        'max_length': []
    }

    # ------------------------------------------------------------------
    # 1. Outer training loop – iterate over *rollout → optimisation* cycles
    # ------------------------------------------------------------------
    
    # Load checkpoint if specified
    checkpoint = None
    if config.checkpoint_mode == "old" and config.checkpoint_path:
        logger.info(f"Loading from checkpoint: {config.checkpoint_path}")
        try:
            (loaded_step, training_steps, all_checks_passed_history, diversity_history, 
             loss_history, sequence_length_history, loaded_diagnostic_tracker) = load_checkpoint( # Removed ema_pass_rates
                config=config,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                logger=logger,
                checkpoint_path=config.checkpoint_path
            )
            
            # Update diagnostic tracker if it was loaded
            if loaded_diagnostic_tracker is not None:
                diagnostic_tracker = loaded_diagnostic_tracker
            
            # Update the starting step for the training loop
            start_step = loaded_step + 1
            logger.info(f"Resuming training from step {start_step}")
            
            # Load the checkpoint to check for diagnostic tracker path
            checkpoint = torch.load(config.checkpoint_path, map_location=DEVICE)
            
            # NEW: Restore novelty buffer from checkpoint
            if 'novelty_buffer' in checkpoint:
                trainer.novelty_buffer.extend(checkpoint['novelty_buffer'])
                logger.info(f"✓ Novelty buffer restored with {len(trainer.novelty_buffer)} unique sequences.")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Starting fresh training instead")
            start_step = 1
    else:
        start_step = 1
    
    # If we loaded a checkpoint, we need to reconstruct the diagnostic tracker
    # after the first validation step since we need the check_names
    diagnostic_tracker_loaded = False
    if config.checkpoint_mode == "old" and config.checkpoint_path and checkpoint and 'diagnostic_tracker_path' in checkpoint:
        diagnostic_tracker_loaded = True
    
    for step in range(start_step, config.total_steps + 1):
        step_start_time = time.time()
        logger.info(f"\n=== Step {step}/{config.total_steps} ===")

        # ----------------------------------------------------------
        # 1.1 Generate trajectories (now with compiled SequenceGenerator!)
        # ----------------------------------------------------------
        rollout_start = time.time()
        model.eval() # Model should be in eval mode for generation
        with torch.no_grad():
            (
                sequences,        # (B, T)
                rollout_logits,   # (B, T, V)
                rollout_values,   # (B, T)
                legality_masks,   # (B, T, V) – bool (now computed incrementally!)
            ) = trainer.sequence_generator(config.batch_size) # Call the compiled generator with KV-caching
        rollout_time = time.time() - rollout_start
        logger.info(f"1.1 Rollout generation: {rollout_time:.3f}s")

        # CRITICAL FIX: Take model snapshot AFTER rollout generation but BEFORE PPO updates
        # This ensures we're comparing the policy that generated the rollout vs the updated policy
        old_raw_model.load_state_dict(raw_model.state_dict())
        old_raw_model.eval()
        
        # Verify the snapshot worked correctly
        with torch.no_grad():
            test_seq = sequences[:2]  # Use first 2 sequences for testing
            test_logits_old, _, _ = old_raw_model(test_seq)
            test_logits_new, _, _ = raw_model(test_seq)
            max_diff = torch.abs(test_logits_old - test_logits_new).max().item()
            if max_diff > 1e-6:
                logger.warning(f"Model snapshot verification failed! Max diff: {max_diff:.8f}")
            # else:
            #     logger.info(f"Model snapshot verified (max diff: {max_diff:.8f})")
            
            # Also check if compiled model vs raw model have different outputs
            # This check is now less relevant as `model` is always `raw_model` here.
            # However, if `trainer.sequence_generator.model` *itself* were a compiled version,
            # this check would still be relevant to ensure consistency.
            if trainer.sequence_generator.model != raw_model:  # Only if we're using compiled model (unlikely after changes)
                test_logits_compiled, _, _ = trainer.sequence_generator.model(test_seq)
                compiled_raw_diff = torch.abs(test_logits_compiled - test_logits_new).max().item()
                # logger.info(f"Compiled vs raw model diff: {compiled_raw_diff:.8f}")
            
            # Check parameter differences between old and current models
            param_diff_sum = 0.0
            param_count = 0
            for (name_old, param_old), (name_new, param_new) in zip(old_raw_model.named_parameters(), raw_model.named_parameters()):
                if name_old == name_new:
                    param_diff = torch.abs(param_old - param_new).sum().item()
                    param_diff_sum += param_diff
                    param_count += 1
            avg_param_diff = param_diff_sum / param_count if param_count > 0 else 0.0
            # logger.info(f"Average parameter difference between old and current models: {avg_param_diff:.8f}")

        # --- compute and log diversity metrics ---
        diversity_start = time.time()
        diversity_metrics = trainer.compute_diversity_metrics(sequences, rollout_logits, legality_masks)
        diversity_time = time.time() - diversity_start
        logger.info(f"1.2 Diversity metrics computation: {diversity_time:.3f}s")

        # --- compute and log full-model check rates ---
        validation_start = time.time()
        batch_models = []
        parser_failures = []  # Track which sequences fail strict parser
        
        for seq_idx, seq in enumerate(sequences):
            seq_len = (seq != trainer.dead_idx).sum().item()
            if seq_len == 0:
                batch_models.append({})
                parser_failures.append(False)  # Empty sequences don't count as parser failures
                continue
            tokens = [trainer.idx_to_token[i.item()] for i in seq[:seq_len] if i.item() != trainer.dead_idx]
            try:
                # Use the same parser as token2fr.py for consistency
                from token2fr import ModelTester
                tester = ModelTester()
                model_dict = tester.build_model(tokens)
                batch_models.append(model_dict)
                parser_failures.append(False)  # Parser succeeded
            except (IndexError, KeyError, ValueError) as e:
                # If the strict parser fails, the sequence should NOT get max score
                logger.debug(f"Strict parser failed for sequence {seq_idx}: {e}")
                batch_models.append({})
                parser_failures.append(True)  # Parser failed
        
        # ingest_batch call removed - no longer needed with model-based scoring
        _, _, names, results = model_based_scorer_with_diagnostics(sequences, trainer.idx_to_token, level="all", verbose=False)
        validation_time = time.time() - validation_start
        logger.info(f"1.3 Model parsing & validation: {validation_time:.3f}s")
        
        # Log parser failure statistics
        num_parser_failures = sum(parser_failures)
        if num_parser_failures > 0:
            logger.warning(f"⚠️  {num_parser_failures}/{len(sequences)} sequences failed strict parser validation")
        
        # Initialize diagnostic tracker after first validation
        if diagnostic_tracker is None and config.track_diagnostics:
            diagnostic_tracker = DiagnosticTracker(config, names)
            diagnostic_tracker.set_token_indices(trainer.token_to_idx)
            logger.info(f"Diagnostic tracker initialized with {len(names)} checks")
            
            # If we loaded a checkpoint, reconstruct the diagnostic tracker state
            if diagnostic_tracker_loaded and checkpoint and 'diagnostic_tracker_path' in checkpoint:
                try:
                    diagnostic_path = checkpoint['diagnostic_tracker_path']
                    diagnostic_state = torch.load(diagnostic_path, map_location=DEVICE)
                    
                    # Restore the diagnostic tracker state
                    for key, value in diagnostic_state.items():
                        if hasattr(diagnostic_tracker, key):
                            setattr(diagnostic_tracker, key, value)
                    
                    logger.info(f"✓ Diagnostic tracker state restored from {diagnostic_path}")
                except Exception as e:
                    logger.warning(f"Failed to restore diagnostic tracker state: {e}")
                    logger.info("Continuing with fresh diagnostic tracker")
        
        # Log validation results
        for i, (name, passed) in enumerate(zip(names, results)):
            pct = passed.float().mean().item() * 100
            logger.info(f"     {name:<55} {pct:5.1f}%")

        # Update diagnostic tracker with validation results
        if diagnostic_tracker is not None:
            diagnostic_tracker.update_per_check_pass_rates(results)
            # Removed: diagnostic_tracker.update_check_weight_evolution(config.check_weights)
            diagnostic_tracker.update_sequence_length_stats(sequences, trainer.dead_idx)
            diagnostic_tracker.update_action_distribution_drift(rollout_logits, legality_masks, sequences != trainer.dead_idx)

        # --- Dynamic reward weight scaling based on validation pass rates --- (REMOVED)
        reward_calc_start = time.time()
        if len(results) > 0:
            # Removed: config.check_weights, ema_pass_rates and their update logic
            
            batch_results = torch.stack([res.float() for res in results], dim=1)
            # Removed: validation_bonus = (batch_results * config.check_weights).sum(dim=1)

            # Removed: Check for validation bonus explosion
            
            raw_scores = batch_results.sum(dim=1)
            
            # NEW: Save sequences with maximum check score (17) into FeynRules files
            max_score = 17  # Maximum possible check score
            max_score_indices = (raw_scores == max_score).nonzero(as_tuple=True)[0]
            
            if len(max_score_indices) > 0:
                logger.info(f"🎉 Found {len(max_score_indices)} sequence(s) with maximum check score ({max_score})!")
                
                # Create directory for FeynRules files if it doesn't exist
                fr_output_dir = os.path.join(config.log_dir, "feynrules_models")
                os.makedirs(fr_output_dir, exist_ok=True)
                
                for idx in max_score_indices:
                    idx = idx.item()
                    seq = sequences[idx]
                    
                    # Convert sequence to tokens
                    seq_len = (seq != trainer.dead_idx).sum().item()
                    if seq_len == 0:
                        logger.warning(f"❌ Sequence {idx} has zero length (all dead tokens)")
                        continue
                    
                    tokens = [trainer.idx_to_token[i.item()] for i in seq[:seq_len] if i.item() != trainer.dead_idx]
                    
                    # Pre-validate with strict parser before attempting FeynRules conversion
                    try:
                        from token2fr import ModelTester
                        tester = ModelTester()
                        test_model = tester.build_model(tokens)
                        # If we get here, the strict parser succeeded
                        strict_parser_ok = True
                    except Exception as e:
                        logger.warning(f"❌ Sequence {idx} failed strict parser validation: {e}")
                        logger.warning(f"   This sequence should NOT have gotten max score!")
                        strict_parser_ok = False
                    
                    if not strict_parser_ok:
                        continue
                    
                    # Generate unique filename based on step and sequence index
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_name = f"MaxScoreModel_Step{step}_Seq{idx}_{timestamp}"
                    
                    # Save as FeynRules file
                    fr_file_path = os.path.join(fr_output_dir, f"{model_name}.fr")
                    try:
                        from token2fr import ModelTester
                        tester = ModelTester()
                        tester.save_model(tokens, fr_file_path)
                        logger.info(f"✅ Saved FeynRules model: {fr_file_path}")
                    except Exception as e:
                        logger.warning(f"❌ Failed to save FeynRules model for sequence {idx}: {e}")
            
            # Print periodic memory usage only at checkpoint intervals
            if step % config.checkpoint_interval == 0: # Throttle based on checkpoint interval
                print_memory_usage(logger, f"STEP {step} (Periodic)")

        # ----------------------------------------------------------
        # 1.4 Compute *on-policy* rewards + advantages
        # ----------------------------------------------------------
        num_terminal = 0  # Initialize num_terminal variable
        with torch.no_grad():
            is_alive = sequences != trainer.dead_idx

            base_rewards = trainer.calculate_base_rewards(sequences) # This now includes length and novelty
            terminal_rewards, num_terminal = trainer.calculate_terminal_reward(sequences)
            
            # NEW: Extract novelty bonus component for diagnostics.
            # This requires re-calculating it, as calculate_base_rewards only returns total.
            # It's okay to re-calculate here as it's @torch.no_grad and not on the critical path.
            novelty_rewards_component_for_logging = trainer.calculate_novelty_bonus(sequences)
            
            total_rewards = base_rewards + terminal_rewards
            
            # NEW: Apply parser failure penalty
            # Convert parser_failures list to tensor and apply -1 reward penalty
            parser_failure_tensor = torch.tensor(parser_failures, dtype=torch.float, device=DEVICE)
            # Apply penalty at the end of the sequence for simplicity
            last_alive_idx = (sequences != trainer.dead_idx).sum(dim=1).clamp(min=1) - 1
            total_rewards[torch.arange(config.batch_size), last_alive_idx] += -1.0 * parser_failure_tensor
            
            # NEW: Apply parser success bonus for sequences that pass strict validation
            parser_success_bonus = 0.1  # Small positive reward for passing strict parser
            # Apply bonus at the end of the sequence for simplicity
            total_rewards[torch.arange(config.batch_size), last_alive_idx] += parser_success_bonus * (1.0 - parser_failure_tensor)
            
            # Log parser penalty statistics
            if parser_failure_tensor.sum() > 0:
                logger.info(f"🔴 Applied parser failure penalty to {parser_failure_tensor.sum().item()} sequences")
            
            # Log parser success statistics
            parser_success_count = (1.0 - parser_failure_tensor).sum().item()
            if parser_success_count > 0:
                logger.info(f"✅ Applied parser success bonus to {parser_success_count} sequences")
            
            # CRITICAL FIX: Clip rewards to prevent extreme values that cause numerical instability
            reward_clip_threshold = 100.0  # Reasonable upper bound for rewards
            total_rewards = torch.clamp(total_rewards, -reward_clip_threshold, reward_clip_threshold)
            
            # Removed: Add validation bonus if available
            # if len(results) > 0 and config.check_weights is not None:
            #     batch_results = torch.stack([res.float() for res in results], dim=1)  # (B, num_checks)
            #     validation_bonus = (batch_results * config.check_weights).sum(dim=1)  # (B,)
            #     B, T = sequences.shape
            #     last_alive = (sequences != trainer.dead_idx).sum(dim=1).clamp(min=1) - 1
            #     total_rewards[torch.arange(B, device=DEVICE), last_alive] += validation_bonus

            # CRITICAL FIX: Clip rollout values to prevent extreme estimates that cause numerical instability
            value_clip_threshold = 100.0  # Reasonable upper bound for value estimates
            rollout_values = torch.clamp(rollout_values, -value_clip_threshold, value_clip_threshold)

            advantages = compute_gae_jit(
                total_rewards, rollout_values, is_alive, config.gamma, config.gae_lambda
            )
            
            # CRITICAL FIX: Check for NaN/Inf in raw advantages before normalization
            if torch.isnan(advantages).any() or torch.isinf(advantages).any():
                logger.warning(f"WARNING: Raw advantages contain NaN/Inf values at step {step}!")
                logger.warning(f"  Raw advantage stats: min={advantages.min().item()}, max={advantages.max().item()}, mean={advantages.mean().item()}")
                logger.warning(f"  Total rewards stats: min={total_rewards.min().item()}, max={total_rewards.max().item()}, mean={total_rewards.mean().item()}")
                logger.warning(f"  Rollout values stats: min={rollout_values.min().item()}, max={rollout_values.max().item()}, mean={rollout_values.mean().item()}")
                
                # Replace NaN/Inf with zeros to prevent training collapse
                advantages = torch.where(
                    torch.isnan(advantages) | torch.isinf(advantages),
                    torch.zeros_like(advantages),
                    advantages
                )
            
            # FIXED: Apply normalization BEFORE clipping to ensure proper centering
            if config.advantage_normalization == "standard":
                # Standard normalization: subtract mean, divide by std
                # Only normalize over alive positions to avoid bias from dead positions
                if config.advantage_normalize_only_alive:
                    alive_advantages = advantages[is_alive]
                    if len(alive_advantages) > 0:
                        adv_mean = alive_advantages.mean()
                        adv_std = alive_advantages.std()
                        # CRITICAL FIX: Prevent division by zero or very small values
                        if adv_std < 1e-8:
                            logger.warning(f"WARNING: Very small advantage std ({adv_std.item()}) at step {step}, using identity normalization")
                            advantages = advantages - adv_mean  # Only center, don't scale
                        else:
                            advantages = (advantages - adv_mean) / adv_std
                else:
                    adv_mean = advantages.mean()
                    adv_std = advantages.std()
                    # CRITICAL FIX: Prevent division by zero or very small values
                    if adv_std < 1e-8:
                        logger.warning(f"WARNING: Very small advantage std ({adv_std.item()}) at step {step}, using identity normalization")
                        advantages = advantages - adv_mean  # Only center, don't scale
                    else:
                        advantages = (advantages - adv_mean) / adv_std
            elif config.advantage_normalization == "robust":
                # Robust normalization: use median and MAD instead of mean and std
                if config.advantage_normalize_only_alive:
                    alive_advantages = advantages[is_alive]
                    if len(alive_advantages) > 0:
                        median_adv = alive_advantages.median()
                        mad_adv = torch.median(torch.abs(alive_advantages - median_adv))
                        # CRITICAL FIX: Prevent division by zero or very small values
                        if mad_adv < 1e-8:
                            logger.warning(f"WARNING: Very small advantage MAD ({mad_adv.item()}) at step {step}, using identity normalization")
                            advantages = advantages - median_adv  # Only center, don't scale
                        else:
                            advantages = (advantages - median_adv) / mad_adv
                else:
                    median_adv = advantages.median()
                    mad_adv = torch.median(torch.abs(advantages - median_adv))
                    # CRITICAL FIX: Prevent division by zero or very small values
                    if mad_adv < 1e-8:
                        logger.warning(f"WARNING: Very small advantage MAD ({mad_adv.item()}) at step {step}, using identity normalization")
                        advantages = advantages - median_adv  # Only center, don't scale
                    else:
                        advantages = (advantages - median_adv) / mad_adv
            elif config.advantage_normalization == "none":
                # No normalization - use raw advantages
                pass
            else:
                # Default to standard normalization
                if config.advantage_normalize_only_alive:
                    alive_advantages = advantages[is_alive]
                    if len(alive_advantages) > 0:
                        adv_mean = alive_advantages.mean()
                        adv_std = alive_advantages.std()
                        # CRITICAL FIX: Prevent division by zero or very small values
                        if adv_std < 1e-8:
                            logger.warning(f"WARNING: Very small advantage std ({adv_std.item()}) at step {step}, using identity normalization")
                            advantages = advantages - adv_mean  # Only center, don't scale
                        else:
                            advantages = (advantages - adv_mean) / adv_std
                else:
                    adv_mean = advantages.mean()
                    adv_std = advantages.std()
                    # CRITICAL FIX: Prevent division by zero or very small values
                    if adv_std < 1e-8:
                        logger.warning(f"WARNING: Very small advantage std ({adv_std.item()}) at step {step}, using identity normalization")
                        advantages = advantages - adv_mean  # Only center, don't scale
                    else:
                        advantages = (advantages - adv_mean) / adv_std
            
            # Apply clipping AFTER normalization to preserve the centered distribution
            advantages = torch.clamp(advantages, -config.advantage_clip_threshold, config.advantage_clip_threshold)
            
            # FIXED: Rescale advantages to prevent clipping to flat regions (Fix #2)
            # If advantages are too small compared to the clip threshold, they get clipped to flat regions
            # Rescale them to have a mean absolute value of ~1.0 before clipping
            if config.advantage_normalize_only_alive:
                alive_advantages = advantages[is_alive]
                if len(alive_advantages) > 0:
                    adv_abs_mean = alive_advantages.abs().mean()
                    # CRITICAL FIX: Prevent division by zero or very small values
                    if adv_abs_mean > 1e-8:
                        advantages = advantages / adv_abs_mean.detach()
                    else:
                        logger.warning(f"WARNING: Very small advantage abs mean ({adv_abs_mean.item()}) at step {step}, skipping rescaling")
            else:
                adv_abs_mean = advantages.abs().mean()
                # CRITICAL FIX: Prevent division by zero or very small values
                if adv_abs_mean > 1e-8:
                    advantages = advantages / adv_abs_mean.detach()
                else:
                    logger.warning(f"WARNING: Very small advantage abs mean ({adv_abs_mean.item()}) at step {step}, skipping rescaling")
            
            # Re-apply clipping after rescaling
            advantages = torch.clamp(advantages, -config.advantage_clip_threshold, config.advantage_clip_threshold)
            
            # FINAL CHECK: Ensure no NaN/Inf values after all processing
            if torch.isnan(advantages).any() or torch.isinf(advantages).any():
                logger.warning(f"CRITICAL ERROR: Advantages still contain NaN/Inf values after processing at step {step}!")
                logger.warning(f"  Final advantage stats: min={advantages.min().item()}, max={advantages.max().item()}, mean={advantages.mean().item()}")
                # Replace with zeros as last resort
                advantages = torch.where(
                    torch.isnan(advantages) | torch.isinf(advantages),
                    torch.zeros_like(advantages),
                    advantages
                )
            
            # FIXED: Update advantage statistics AFTER normalization so negative fraction is computed correctly
            if diagnostic_tracker is not None:
                diagnostic_tracker.update_advantage_stats(advantages, is_alive)
                
                # NEW: Track parser failure statistics
                parser_failure_rate = parser_failure_tensor.float().mean().item()
                diagnostic_tracker.update_parser_failure_rate(parser_failure_rate)
            
            # Check for advantage explosion
            if torch.isnan(advantages).any() or torch.isinf(advantages).any():
                logger.warning(f"WARNING: Advantages contain NaN or Inf values at step {step}!")
                logger.warning(f"  Advantage stats: min={advantages.min().item()}, max={advantages.max().item()}, mean={advantages.mean().item()}")
                logger.warning(f"  Total rewards stats: min={total_rewards.min().item()}, max={total_rewards.max().item()}, mean={total_rewards.mean().item()}")
                logger.warning(f"  Rollout values stats: min={rollout_values.min().item()}, max={rollout_values.max().item()}, mean={rollout_values.mean().item()}")
            
            returns = advantages + rollout_values
            
            # Update value function error statistics
            if diagnostic_tracker is not None:
                diagnostic_tracker.update_value_function_error(rollout_values, returns, is_alive)
                diagnostic_tracker.update_mask_density(legality_masks, is_alive)
            
            # Update diagnostic tracker with reward components
            if diagnostic_tracker is not None:
                # Removed: validation_bonus_tensor and its assignment
                diagnostic_tracker.update_reward_components(
                    base_rewards, terminal_rewards, novelty_rewards_component_for_logging # MODIFIED: added novelty_rewards_component_for_logging
                )

        # Log novelty buffer size after reward calculation
        logger.info(f"Novelty buffer size: {len(trainer.novelty_buffer)}")

        reward_calc_time = time.time() - reward_calc_start
        logger.info(f"1.4 Reward calculation & advantage estimation: {reward_calc_time:.3f}s")

        # ----------------------------------------------------------
        # 1.5 PPO optimisation (multiple epochs / minibatches)
        # ----------------------------------------------------------
        ppo_start = time.time()
        model.train()

        # Track KL values per epoch for detailed analysis
        epoch_kl_values = []

        # Ensure rollout_log_probs is computed using the 'old' model BEFORE any updates
        with torch.no_grad():
            # This ensures old_logits are truly from the policy that generated the rollout
            old_logits_for_rollout, _, _ = old_raw_model(sequences)
            rollout_log_probs = (
                F.log_softmax(old_logits_for_rollout.masked_fill(~legality_masks, -1e4), dim=-1)
                .gather(2, sequences.unsqueeze(-1))
                .squeeze(-1)
            )

        # Initialize loss variables for tracking
        pg_loss = torch.tensor(0.0, device=DEVICE)
        v_loss = torch.tensor(0.0, device=DEVICE)
        ent_loss = torch.tensor(0.0, device=DEVICE)
        sup_loss = torch.tensor(0.0, device=DEVICE)
        # div_loss = torch.tensor(0.0, device=DEVICE)  # Removed - not used

        for ppo_epoch in range(config.ppo_epochs):
            perm = torch.randperm(config.batch_size, device=DEVICE)
            for mb_start in range(0, config.batch_size, config.minibatch_size):
                mb_idx = perm[mb_start:mb_start+config.minibatch_size]
                mb_seq = sequences[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_ret = returns[mb_idx]
                mb_old_logp = rollout_log_probs[mb_idx]
                mb_masks = legality_masks[mb_idx]
                mb_is_alive = is_alive[mb_idx]

                with torch.amp.autocast(device_type=DEVICE.type, enabled=False):  # Disabled for full precision
                    # CRITICAL FIX: Use raw model for new logits to match old logits
                    # This ensures we're comparing the same model type (raw vs raw)
                    new_logits, new_values, _ = raw_model(mb_seq)

                    masked_new_logp_dist = F.log_softmax(
                        new_logits.masked_fill(~mb_masks, -1e4), dim=-1
                    )

                    new_logp = masked_new_logp_dist.gather(2, mb_seq.unsqueeze(-1)).squeeze(-1)
                    
                    # Compute KL divergence between old and new policies using the full action distribution
                    # Both old and new logits now come from raw models (consistent)
                    with torch.no_grad():
                        old_logits, _, _ = old_raw_model(mb_seq)
                        
                        # Compute full action distributions for both old and new policies
                        old_logp_dist = F.log_softmax(old_logits.masked_fill(~mb_masks, -1e4), dim=-1)
                        new_logp_dist = F.log_softmax(new_logits.masked_fill(~mb_masks, -1e4), dim=-1)
                        
                        # Convert old log probabilities to probabilities for KL computation
                        old_p_dist = old_logp_dist.exp()
                        
                        # Compute KL divergence over the full action distribution: D_KL(old || new)
                        # KL = sum_a P_old(a) * (log P_old(a) - log P_new(a))
                        kl_per_step = (old_p_dist * (old_logp_dist - new_logp_dist)).sum(dim=-1)  # (B, T)
                        
                        # Average over all alive positions
                        kl_div = kl_per_step[mb_is_alive].mean().item()
                        
                        # Ensure non-negativity (should be true mathematically, but numerical issues can occur)
                        kl_div = max(0.0, kl_div)
                    
                    if diagnostic_tracker is not None:
                        # For diagnostic tracking, compute both KL divergences consistently
                        # CRITICAL FIX: Both KL divergences should be computed over the full distribution
                        # The issue was that sampled-action KL was only looking at taken actions
                        
                        # Compute sampled-action KL using the same full-distribution approach
                        # but only for the actions that were actually taken
                        old_logp_sampled = old_logp_dist.gather(2, mb_seq.unsqueeze(-1)).squeeze(-1)
                        new_logp_sampled = masked_new_logp_dist.gather(2, mb_seq.unsqueeze(-1)).squeeze(-1)
                        
                        # For sampled-action KL, we compute the average log ratio for taken actions
                        # This gives us a measure of how much the policy changed for the actions that were taken
                        log_ratio_sampled = old_logp_sampled - new_logp_sampled
                        sampled_action_kl = log_ratio_sampled[mb_is_alive].mean().item()
                        sampled_action_kl = max(0.0, sampled_action_kl)  # Ensure non-negativity
                        
                        # Update diagnostic tracker with both KL divergences
                        diagnostic_tracker.update_kl_divergence(old_logp_sampled, new_logp_sampled, mb_is_alive)
                        diagnostic_tracker.update_full_distribution_kl(kl_div)
                        
                        # Store the sampled-action KL separately for comparison
                        if 'policy_update_stats' not in diagnostic_tracker.__dict__:
                            diagnostic_tracker.policy_update_stats = {
                                'old_logp_mean': [],
                                'new_logp_mean': [],
                                'logp_diff_mean': [],
                                'logp_diff_std': [],
                                'policy_change_magnitude': [],
                                'full_distribution_kl': [],
                                'sampled_action_kl': []
                            }
                        diagnostic_tracker.policy_update_stats['sampled_action_kl'].append(sampled_action_kl)
                    
                    epoch_kl_values.append(kl_div)
                    
                    ratio    = torch.exp(new_logp - mb_old_logp)
                    ratio_clipped = torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)

                    # Update clipping fraction tracking
                    if diagnostic_tracker is not None:
                        diagnostic_tracker.update_clipping_fraction(ratio, mb_is_alive)
                        diagnostic_tracker.update_pg_loss_components(mb_adv, ratio, ratio_clipped, mb_is_alive)

                    pg_loss = -torch.min(mb_adv * ratio, mb_adv * ratio_clipped)[mb_is_alive].mean()

                    # Check for policy loss explosion
                    if torch.isnan(pg_loss) or torch.isinf(pg_loss):
                        logger.warning(f"WARNING: Policy loss is NaN or Inf at step {step}, epoch {ppo_epoch}, mb_start {mb_start}!")
                        logger.warning(f"  Policy loss value: {pg_loss.item()}")
                        logger.warning(f"  Minibatch advantage stats: min={mb_adv.min().item()}, max={mb_adv.max().item()}, mean={mb_adv.mean().item()}")
                        logger.warning(f"  Ratio stats: min={ratio.min().item()}, max={ratio.max().item()}, mean={ratio.mean().item()}")
                        logger.warning(f"  Clipped ratio stats: min={ratio_clipped.min().item()}, max={ratio_clipped.max().item()}, mean={ratio_clipped.mean().item()}")

                    v_loss = F.mse_loss(new_values[mb_is_alive], mb_ret[mb_is_alive])

                    # Check for value loss explosion
                    if torch.isnan(v_loss) or torch.isinf(v_loss):
                        logger.warning(f"WARNING: Value loss is NaN or Inf at step {step}, epoch {ppo_epoch}, mb_start {mb_start}!")
                        logger.warning(f"  Value loss value: {v_loss.item()}")
                        logger.warning(f"  New values stats: min={new_values.min().item()}, max={new_values.max().item()}, mean={new_values.mean().item()}")
                        logger.warning(f"  Returns stats: min={mb_ret.min().item()}, max={mb_ret.max().item()}, mean={mb_ret.mean().item()}")

                    masked_logits = new_logits.masked_fill(~mb_masks, -1e4)
                    # Calculate entropy directly over legal tokens only
                    legal_probs = F.softmax(masked_logits, dim=-1)
                    legal_log_probs = torch.log(legal_probs + 1e-8)
                    ent_loss = -(legal_probs * legal_log_probs).sum(dim=-1)
                    ent_loss = ent_loss[mb_is_alive].mean()

                    sup_loss = (
                        F.softmax(new_logits, dim=-1).masked_fill(mb_masks, 0.0).sum(dim=-1)
                    )[mb_is_alive].mean()

                    # Add diversity loss
                    # div_loss = torch.tensor(0.0, device=DEVICE)  # Removed - not used

                    # Update the tracking variables with current minibatch values (keep gradients)
                    # Note: These variables are used for tracking but don't affect the computation graph

                    total_loss = (
                        config.pg_loss_scale * pg_loss
                        + config.vf_coef * v_loss
                        + config.ent_coef * ent_loss
                        + config.sup_coef * sup_loss
                    )

                    # Add weight regularization for stability
                    if config.weight_decay > 0:
                        reg_loss = raw_model.get_weight_regularization_loss(config.weight_decay)
                        total_loss += reg_loss

                    # Check for total loss explosion
                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        logger.warning(f"WARNING: Total loss is NaN or Inf at step {step}, epoch {ppo_epoch}, mb_start {mb_start}!")
                        logger.warning(f"  Total loss value: {total_loss.item()}")
                        logger.warning(f"  Component losses: pg={pg_loss.item()}, v={v_loss.item()}, ent={ent_loss.item()}, sup={sup_loss.item()}")
                        logger.warning(f"  Coefficients: pg_scale={config.pg_loss_scale}, vf_coef={config.vf_coef}, ent_coef={config.ent_coef}, sup_coef={config.sup_coef}")

                    if config.balanced_coef > 0 and len(results) > 0:
                        mb_batch_results = torch.stack([res[mb_idx].float() for res in results], dim=1)
                        mb_pass_rates = mb_batch_results.mean(dim=0)
                        mb_balance_penalty = ((mb_pass_rates - mb_pass_rates.mean())**2).mean()
                        total_loss += config.balanced_coef * mb_balance_penalty

                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                
                # Update gradient norm tracking BEFORE clipping
                if diagnostic_tracker is not None:
                    diagnostic_tracker.update_gradient_norms(raw_model)  # Use raw_model instead of model
                
                # Apply gradient clipping with new configuration
                if config.enable_gradient_clipping:
                    if config.gradient_clip_adaptive and diagnostic_tracker is not None:
                        # Adaptive clipping based on recent gradient statistics
                        if diagnostic_tracker.gradient_norms:
                            recent_norms = diagnostic_tracker.gradient_norms[-10:]  # last 10 steps
                            # Filter out NaN and infinite values
                            valid_norms = [norm for norm in recent_norms if np.isfinite(norm)]
                            if valid_norms:
                                adaptive_threshold = min(config.gradient_clip_norm, np.percentile(valid_norms, 90))
                            else:
                                adaptive_threshold = config.gradient_clip_norm
                        else:
                            adaptive_threshold = config.gradient_clip_norm
                        nn.utils.clip_grad_norm_(raw_model.parameters(), adaptive_threshold)  # Use raw_model
                    else:
                        nn.utils.clip_grad_norm_(raw_model.parameters(), config.gradient_clip_norm)  # Use raw_model
                
                scaler.step(optimizer)
                scaler.update()
                
                # Check if model parameters actually changed (for debugging KL divergence issues)
                if diagnostic_tracker is not None and ppo_epoch == 0 and mb_start == 0:
                    # After the first update, check if parameters changed
                    with torch.no_grad():
                        test_logits_after_update, _, _ = raw_model(test_seq)  # Use raw_model
                        after_update_diff = torch.abs(test_logits_after_update - test_logits_new).max().item()
                        # logger.info(f"Model output change after first update: {after_update_diff:.8f}")
                        
                        # Also check parameter changes
                        param_change_sum = 0.0
                        param_count = 0
                        for (name_old, param_old), (name_new, param_new) in zip(old_raw_model.named_parameters(), raw_model.named_parameters()):
                            if name_old == name_new:
                                param_change = torch.abs(param_old - param_new).sum().item()
                                param_change_sum += param_change
                                param_count += 1
                        avg_param_change = param_change_sum / param_count if param_count > 0 else 0.0
                        # logger.info(f"Average parameter change after first update: {avg_param_change:.8f}")
                        
                        if avg_param_change < 1e-8:
                            logger.warning("WARNING: Model parameters are not changing! This will cause KL divergence to be zero.")
                            logger.warning("Check: learning rate, gradient clipping, loss computation, optimizer setup")
        
        # Update KL per epoch statistics
        if diagnostic_tracker is not None:
            diagnostic_tracker.update_kl_per_epoch(epoch_kl_values)
            
        ppo_time = time.time() - ppo_start
        logger.info(f"1.5 PPO optimization ({config.ppo_epochs} epochs): {ppo_time:.3f}s")

        # ----------------------------------------------------------
        # 1.5.5 Value Function Epochs - Additional backward passes for value function
        # ----------------------------------------------------------
        vf_start = time.time()
        vf_loss_value = 0.0
        
        # Track value function training for diagnostics
        vf_losses = []
        vf_gradient_norms = []
        
        # Run additional epochs specifically for value function training
        if config.vf_epochs > 0:
            raw_model.train()
            for vf_epoch in range(config.vf_epochs):
                epoch_vf_losses = []
                epoch_vf_grad_norms = []
                
                # Use the same permutation as the main PPO loop for consistency
                perm = torch.randperm(config.batch_size, device=DEVICE)
                for mb_start in range(0, config.batch_size, config.minibatch_size):
                    mb_idx = perm[mb_start:mb_start+config.minibatch_size]
                    mb_seq = sequences[mb_idx]
                    mb_ret = returns[mb_idx]
                    mb_is_alive = is_alive[mb_idx]

                    with torch.amp.autocast(device_type=DEVICE.type, enabled=False):  # Disabled for full precision
                        # Forward pass - only compute values (ignore logits for efficiency)
                        _, new_values, _ = raw_model(mb_seq)
                        
                        # Value function loss only
                        vf_loss = F.mse_loss(new_values[mb_is_alive], mb_ret[mb_is_alive])
                        
                        # Add weight regularization for value function training
                        if config.weight_decay > 0:
                            reg_loss = raw_model.get_weight_regularization_loss(config.weight_decay)
                            vf_loss += reg_loss
                        
                        vf_loss_value = vf_loss.item()
                        epoch_vf_losses.append(vf_loss_value)

                        # Check for value function loss explosion
                        if torch.isnan(vf_loss) or torch.isinf(vf_loss):
                            logger.warning(f"WARNING: Value function loss is NaN or Inf at step {step}, vf_epoch {vf_epoch}, mb_start {mb_start}!")
                            logger.warning(f"  VF loss value: {vf_loss.item()}")
                            logger.warning(f"  New values stats: min={new_values.min().item()}, max={new_values.max().item()}, mean={new_values.mean().item()}")
                            logger.warning(f"  Returns stats: min={mb_ret.min().item()}, max={mb_ret.max().item()}, mean={mb_ret.mean().item()}")

                    # Value function optimization
                    optimizer.zero_grad()
                    scaler.scale(vf_loss).backward()
                    scaler.unscale_(optimizer)
                    
                    # Track gradient norm for value function training
                    if diagnostic_tracker is not None:
                        total_norm = 0.0
                        for p in raw_model.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** (1. / 2)
                        epoch_vf_grad_norms.append(total_norm)
                    
                    # Apply gradient clipping for value function training
                    if config.enable_gradient_clipping:
                        nn.utils.clip_grad_norm_(raw_model.parameters(), config.gradient_clip_norm)
                    
                    scaler.step(optimizer)
                    scaler.update()
                
                # Store epoch statistics
                if epoch_vf_losses:
                    vf_losses.append(np.mean(epoch_vf_losses))
                if epoch_vf_grad_norms:
                    vf_gradient_norms.append(np.mean(epoch_vf_grad_norms))
        
        # Update diagnostic tracker with value function epochs data
        if diagnostic_tracker is not None:
            diagnostic_tracker.update_vf_epochs_tracking(vf_losses, vf_gradient_norms)
        
        vf_time = time.time() - vf_start
        logger.info(f"1.5.5 Value function epochs ({config.vf_epochs} epochs): {vf_time:.3f}s | final VF loss: {vf_loss_value:.4f}")
        
        # Log value function improvement if available
        if vf_losses and len(vf_losses) > 1:
            improvement = vf_losses[0] - vf_losses[-1]
            logger.info(f"  └─ VF loss improvement: {improvement:.6f} (from {vf_losses[0]:.6f} to {vf_losses[-1]:.6f})")

        # Track losses for plotting
        loss_history['pg_loss'].append(pg_loss.item())
        # Use the final value function loss after VF epochs instead of the main PPO loop loss
        loss_history['v_loss'].append(vf_loss_value if config.vf_epochs > 0 else v_loss.item())
        loss_history['ent_loss'].append(ent_loss.item())
        loss_history['sup_loss'].append(sup_loss.item())
        # loss_history['div_loss'].append(div_loss.item())  # Removed - div_loss not computed
        # Note: avg_reward will be appended later after it's calculated

        # ----------------------------------------------------------
        # 1.7 Logging
        # ----------------------------------------------------------
        logging_start = time.time()
        avg_reward = total_rewards[is_alive].mean().item()
        
        # Now append avg_reward to loss_history
        loss_history['reward'].append(avg_reward)

        batch_results = torch.stack([res.float() for res in results], dim=1) if results else None

        if batch_results is not None:
            particle_pass_counts = batch_results[:, 0:2].sum(dim=1)
            field_pass_counts    = batch_results[:, 2:11].sum(dim=1)
            interact_pass_counts = batch_results[:, 11:].sum(dim=1)
            
            avg_particle_checks    = particle_pass_counts.mean().item()
            avg_field_checks       = field_pass_counts.mean().item()
            avg_interact_checks    = interact_pass_counts.mean().item()
            
            # Calculate total average checks passed
            total_checks_passed = batch_results.sum(dim=1)
            avg_total_checks = total_checks_passed.mean().item()
            
            all_checks_passed = (batch_results == 1.0).all(dim=1)
            pct_all_passed = all_checks_passed.float().mean().item() * 100.0
        else:
            avg_particle_checks = avg_field_checks = avg_interact_checks = 0.0
            avg_total_checks = 0.0
            pct_all_passed = 0.0

        loss_log = (
            f"PG Loss: {pg_loss.item():.3f} | V Loss: {vf_loss_value if config.vf_epochs > 0 else v_loss.item():.3f} | "
            f"E Loss: {ent_loss.item():.3f} | Sup Loss: {sup_loss.item():.3f} | "
        )
        reward_log = (
            f"Avg Checks Passed: {avg_total_checks:.1f} | All Checks Passed: {pct_all_passed:.1f} | Terminal Sequences: {num_terminal}"
        )
        
        # Add diversity metrics to logging
        diversity_log = (
            f"Entropy: {diversity_metrics['avg_entropy']:.3f} | "
            f"Uniqueness: {diversity_metrics['uniqueness_ratio']:.3f} | "
            f"Token Diversity: {diversity_metrics['token_diversity']:.3f} | "
            f"Structural Diversity: {diversity_metrics['structural_diversity']:.3f} | "
            f"Length Diversity: {diversity_metrics['length_diversity']:.3f} | "
            f"Top-K Concentration: {diversity_metrics['top_k_concentration']:.3f}"
        )
        
        # Add sequence length metrics to logging
        is_alive = sequences != trainer.dead_idx
        lengths = is_alive.sum(dim=1).float()
        avg_seq_len = lengths.mean().item()
        std_seq_len = lengths.std().item()
        min_seq_len = lengths.min().item()
        max_seq_len = lengths.max().item()
        sequence_length_log = (
            f"Avg Length: {avg_seq_len:.1f} ± {std_seq_len:.1f} | "
            f"Min: {min_seq_len:.0f} | Max: {max_seq_len:.0f} | "
            f"Target: {config.len_target}"
        )
        
        # --- Build step summary table ---
        # Add a summary line for key metrics
        logger.info(f"📊 Step {step} Summary: Avg Checks={avg_total_checks:.1f}, All Passed={pct_all_passed:.1f}%, Terminal={num_terminal}")
        
        table_title = f"Step {step}/{config.total_steps} | Time: {time.time() - step_start_time:.2f}s | Avg Reward/Seq: {avg_reward:.4f}"
        headers = [
            "Metric", "Value"
        ]
        rows = []
        # Validation
        rows.append(("Validation", reward_log))
        # Sequence Length
        rows.append(("Seq Length", sequence_length_log))
        # Diversity
        rows.append(("Diversity", diversity_log))
        # Losses
        rows.append(("Losses", loss_log))
        # Add any other key stats you want in the table here
        print_step_table(logger, table_title, headers, rows)

        # Log diagnostic summary
        if diagnostic_tracker is not None:
            diagnostic_tracker.log_diagnostic_summary(logger, step)
        
        logging_time = time.time() - logging_start
        logger.info(f"1.7 Logging & statistics: {logging_time:.3f}s")

        # Update timing breakdown
        if diagnostic_tracker is not None:
            timing_dict = {
                'rollout_time': rollout_time,
                'validation_time': validation_time,
                'reward_calc_time': reward_calc_time,
                'ppo_time': ppo_time,
                'vf_time': vf_time,
                'logging_time': logging_time
            }
            diagnostic_tracker.update_timing_breakdown(timing_dict)

        all_checks_passed_history.append(pct_all_passed)
        training_steps.append(step)
        
        # Track diversity metrics over time
        for key in diversity_history:
            diversity_history[key].append(diversity_metrics[key])
        
        # Track sequence length statistics
        is_alive = sequences != trainer.dead_idx
        lengths = is_alive.sum(dim=1).float()
        sequence_length_history['mean_length'].append(lengths.mean().item())
        sequence_length_history['std_length'].append(lengths.std().item())
        sequence_length_history['min_length'].append(lengths.min().item())
        sequence_length_history['max_length'].append(lengths.max().item())

        # Create progress plots every 100 steps
        if step % config.plot_interval == 0 and step > 0:
            # Generate comprehensive progress plots
            trainer._generate_comprehensive_progress_plots(
                step, training_steps, all_checks_passed_history, 
                diversity_history, loss_history, sequence_length_history, 
                diagnostic_tracker, config, logger
            )

        total_step_time = time.time() - step_start_time
        logger.info(f"=== Total step time: {total_step_time:.3f}s ===\n")

        # Print portion of sequences ending with EOS
        eos_idx = trainer.eos_idx
        dead_idx = trainer.dead_idx
        # Find the last non-dead token for each sequence
        is_alive = sequences != dead_idx
        last_alive = is_alive.sum(dim=1).clamp(min=1) - 1
        idxs = torch.arange(sequences.size(0), device=sequences.device)
        eos_mask = sequences[idxs, last_alive] == eos_idx
        portion_eos = eos_mask.float().mean().item()
        logger.info(f"Portion of sequences ending with EOS: {portion_eos:.3f} ({eos_mask.sum().item()}/{sequences.size(0)})")

    # ------------------------------------------------------------------
    # 2. Generate and evaluate final sequence
    # ------------------------------------------------------------------
    logger.info("Generating and evaluating final sequence...")
    final_sequence_tensor = trainer.generate_deterministic_sequence(raw_model)  # Use raw_model instead of model
    token_indices = final_sequence_tensor.squeeze().tolist()
    tokens = [trainer.idx_to_token[i] for i in token_indices if i != trainer.dead_idx]
    
    print("\n--- Generated Sequence ---")
    indent_level = 0
    for token in tokens:
        if token in ["END_PARTICLE", "END_FIELD", "END_ITRACT", "EOS"]:
            indent_level = max(0, indent_level - 1)
        print("  " * indent_level + token)
        if token in ["ITRACT", "FIELD", "PARTICLE"]:
            indent_level += 1

    # ------------------------------------------------------------------
    # 3. Generate and save individual training progress plots
    # ------------------------------------------------------------------
    if all_checks_passed_history:
        # Ensure log directory exists
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Plot 1: Validation pass rates
        plt.figure(figsize=(12, 8))
        plt.plot(training_steps, all_checks_passed_history, 'b-', linewidth=2, label='All Checks Passed %')
        plt.xlabel('Training Step')
        plt.ylabel('Percentage of Sequences Passing All Checks (%)')
        plt.title('Validation Performance Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if len(all_checks_passed_history) > 1:
            final_pct = all_checks_passed_history[-1]
            max_pct = max(all_checks_passed_history)
            avg_pct = np.mean(all_checks_passed_history)
            plt.text(0.02, 0.98, f'Final: {final_pct:.1f}%\nMax: {max_pct:.1f}%\nAvg: {avg_pct:.1f}%', 
                     transform=plt.gca().transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        validation_plot_path = os.path.join(config.log_dir, f"validation_performance_{config.run_name}.png")
        plt.savefig(validation_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Validation performance plot saved as: {validation_plot_path}")
        
        # Plot 2: Policy entropy and token diversity
        if diversity_history['avg_entropy']:
            plt.figure(figsize=(12, 8))
            plt.plot(training_steps, diversity_history['avg_entropy'], 'g-', linewidth=2, label='Avg Policy Entropy')
            plt.plot(training_steps, diversity_history['token_diversity'], 'r-', linewidth=2, label='Token Diversity')
            # config.min_entropy_threshold is not defined in PPOConfig, so commenting this out or adding it if necessary
            # plt.axhline(y=config.min_entropy_threshold, color='g', linestyle='--', alpha=0.7, label=f'Min Entropy Threshold ({config.min_entropy_threshold})')
            plt.xlabel('Training Step')
            plt.ylabel('Diversity Metrics')
            plt.title('Policy Entropy & Token Diversity Over Time')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            entropy_plot_path = os.path.join(config.log_dir, f"policy_entropy_diversity_{config.run_name}.png")
            plt.savefig(entropy_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Policy entropy and diversity plot saved as: {entropy_plot_path}")
        
        # Plot 3: Training losses
        if loss_history['pg_loss']:
            plt.figure(figsize=(12, 8))
            
            # Plot all losses
            plt.plot(training_steps, loss_history['pg_loss'], 'blue', linewidth=2, label='Policy Loss')
            plt.plot(training_steps, loss_history['v_loss'], 'red', linewidth=2, label='Value Loss')
            plt.plot(training_steps, loss_history['ent_loss'], 'green', linewidth=2, label='Entropy Loss')
            plt.plot(training_steps, loss_history['sup_loss'], 'orange', linewidth=2, label='Supervision Loss')
            # Plot average reward
            if loss_history['reward']:
                plt.plot(training_steps, loss_history['reward'], 'black', linewidth=2, label='Avg Reward')
            plt.xlabel('Training Step')
            plt.ylabel('Loss / Reward Value')
            plt.title('Training Losses and Reward Over Time')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.yscale('log')  # Use log scale for better visualization
            
            losses_plot_path = os.path.join(config.log_dir, f"training_losses_{config.run_name}.png")
            plt.savefig(losses_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Training losses plot saved as: {losses_plot_path}")
        
        # Plot 4: Sequence uniqueness and structural diversity
        if diversity_history['uniqueness_ratio']:
            plt.figure(figsize=(12, 8))
            plt.plot(training_steps, diversity_history['uniqueness_ratio'], 'blue', linewidth=2, label='Sequence Uniqueness')
            plt.plot(training_steps, diversity_history['structural_diversity'], 'red', linewidth=2, label='Structural Diversity')
            plt.plot(training_steps, diversity_history['token_diversity'], 'green', linewidth=2, label='Token Diversity')
            plt.plot(training_steps, diversity_history['length_diversity'], 'orange', linewidth=2, label='Length Diversity')
            plt.xlabel('Training Step')
            plt.ylabel('Diversity Ratio')
            plt.title('Sequence Diversity Metrics Over Time')
            plt.grid(True, alpha=0.3)
            plt.legend()
            # Add warning threshold for uniqueness
            plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Low Diversity Warning (50%)')
            
            sequence_diversity_plot_path = os.path.join(config.log_dir, f"sequence_diversity_{config.run_name}.png")
            plt.savefig(sequence_diversity_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Sequence diversity plot saved as: {sequence_diversity_plot_path}")
        
        # Plot 5: Sequence length over time
        if sequence_length_history['mean_length']:
            plt.figure(figsize=(12, 8))
            plt.plot(training_steps, sequence_length_history['mean_length'], 'blue', linewidth=2, label='Mean Length')
            plt.fill_between(training_steps, 
                           [m - s for m, s in zip(sequence_length_history['mean_length'], sequence_length_history['std_length'])],
                           [m + s for m, s in zip(sequence_length_history['mean_length'], sequence_length_history['std_length'])],
                           alpha=0.3, color='blue', label='±1 Std Dev')
            plt.plot(training_steps, sequence_length_history['min_length'], 'green', linewidth=1, alpha=0.7, label='Min Length')
            plt.plot(training_steps, sequence_length_history['max_length'], 'red', linewidth=1, alpha=0.7, label='Max Length')
            plt.axhline(y=config.len_target, color='orange', linestyle='--', linewidth=2, label=f'Target Length ({config.len_target})')
            plt.xlabel('Training Step')
            plt.ylabel('Sequence Length')
            plt.title(f'Sequence Length Over Time (Step {step})')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add statistics text
            if len(sequence_length_history['mean_length']) > 1:
                final_avg_len = sequence_length_history['mean_length'][-1]
                final_std_len = sequence_length_history['std_length'][-1]
                min_len = min(sequence_length_history['min_length'])
                max_len = max(sequence_length_history['max_length'])
                plt.text(0.02, 0.98, f'Final: {final_avg_len:.1f} ± {final_std_len:.1f}\nMin: {min_len:.0f}\nMax: {max_len:.0f}', 
                         transform=plt.gca().transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            sequence_length_plot_path = os.path.join(config.log_dir, f"sequence_length_step_{step}_{config.run_name}.png")
            plt.savefig(sequence_length_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"✓ Sequence length plot saved as: {sequence_length_plot_path}")

        # Plot 6: KL divergence over time (if available)
        if diagnostic_tracker is not None and hasattr(diagnostic_tracker, 'policy_update_stats'):
            plt.figure(figsize=(12, 8))
            
            full_dist_kl = diagnostic_tracker.policy_update_stats.get('full_distribution_kl', [])
            sampled_kl = diagnostic_tracker.policy_update_stats.get('sampled_action_kl', [])
            
            if full_dist_kl and sampled_kl:
                # Convert deques to lists for slicing
                full_dist_kl = list(full_dist_kl)
                sampled_kl = list(sampled_kl)
                
                # Ensure both arrays have the same length by truncating to the shorter one
                min_length = min(len(full_dist_kl), len(sampled_kl))
                full_dist_kl = full_dist_kl[:min_length]
                sampled_kl = sampled_kl[:min_length]
                
                # Create step indices for KL plotting (one per minibatch update)
                kl_steps = list(range(1, min_length + 1))
                
                plt.plot(kl_steps, full_dist_kl, 'blue', linewidth=2, label='Full-distribution KL')
                plt.plot(kl_steps, sampled_kl, 'red', linewidth=2, label='Sampled-action KL')
                plt.axhline(y=config.kl_target, color='green', linestyle='--', alpha=0.7, label=f'KL Target ({config.kl_target})')
                plt.xlabel('Minibatch Update')
                plt.ylabel('KL Divergence')
                plt.title('KL Divergence Over Time')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.yscale('log')  # Use log scale for better visualization
            elif full_dist_kl:
                # Only full-distribution KL available
                full_dist_kl = list(full_dist_kl)
                kl_steps = list(range(1, len(full_dist_kl) + 1))
                plt.plot(kl_steps, full_dist_kl, 'blue', linewidth=2, label='Full-distribution KL')
                plt.axhline(y=config.kl_target, color='green', linestyle='--', alpha=0.7, label=f'KL Target ({config.kl_target})')
                plt.xlabel('Minibatch Update')
                plt.ylabel('KL Divergence')
                plt.title('KL Divergence (Full-distribution only)')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.yscale('log')
            elif sampled_kl:
                # Only sampled-action KL available
                sampled_kl = list(sampled_kl)
                kl_steps = list(range(1, len(sampled_kl) + 1))
                plt.plot(kl_steps, sampled_kl, 'red', linewidth=2, label='Sampled-action KL')
                plt.axhline(y=config.kl_target, color='green', linestyle='--', alpha=0.7, label=f'KL Target ({config.kl_target})')
                plt.xlabel('Minibatch Update')
                plt.ylabel('KL Divergence')
                plt.title('KL Divergence (Sampled-action only)')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.yscale('log')
            else:
                plt.text(0.5, 0.5, 'KL divergence data\nnot available yet', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=plt.gca().transAxes, fontsize=12)
                plt.title('KL Divergence (No data)')
                plt.grid(True, alpha=0.3)
            
            kl_divergence_plot_path = os.path.join(config.log_dir, f"kl_divergence_{config.run_name}.png")
            plt.savefig(kl_divergence_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"KL divergence plot saved as: {kl_divergence_plot_path}")

# ======================================================================================
# Main training function
# ======================================================================================

if __name__ == "__main__":

    
    # Run the main training
    main()