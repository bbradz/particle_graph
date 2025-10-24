import csv
import random
from tqdm import tqdm
import time
from typing import List, Tuple, Optional
import signal
import os
from collections import Counter, deque
import multiprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Categorical
import numpy as np
from difflib import SequenceMatcher
from random import randint, sample
from functools import lru_cache
from test_sm_with_mapping import get_all_possible_check_names, io_sequence_test

# ----------------------------------------------------------------------
# Vocabulary and Tokenization (No changes needed here)
# ----------------------------------------------------------------------

SEQ_0  = ["BOS", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"]
SEQ_1  = ["ITRACT", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"]
SEQ_2  = ["ITRACT_ID_1", "ITRACT_ID_2", "ITRACT_ID_3", "ITRACT_ID_4", "ITRACT_ID_5", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"]
SEQ_3  = ["TYPE_YUKAWA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"]
SEQ_4  = ["FIELD", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"]
SEQ_5  = ["FIELD_ID_1", "FIELD_ID_2", "FIELD_ID_3", "FIELD_ID_4", "FIELD_ID_5", "FIELD_ID_6", "FIELD_ID_7", "FIELD_ID_8", "FIELD_ID_9", "FIELD_ID_10", "FIELD_ID_11", "FIELD_ID_12", "FIELD_ID_13", "FIELD_ID_14"]
SEQ_6  = ["TYPE_fermion", "TYPE_real", "TYPE_complex", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"]
SEQ_7  = ["DIM_1", "DIM_2", "DIM_3", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"]
SEQ_8  = ["GEN_1", "GEN_2", "GEN_3", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"]
SEQ_9  = ["SELF_CONJ_TRUE", "SELF_CONJ_FALSE", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"]
SEQ_10 = ["CHIRALITY_left", "CHIRALITY_right", "CHIRALITY_none", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"]
SEQ_11 = ["SU3C_REP_1", "SU3C_REP_2", "SU3C_REP_3", "NA", "NA", "NA", "NA", "NA","NA", "NA", "NA", "NA", "NA", "NA"]
SEQ_12 = ["SU2L_REP_1", "SU2L_REP_2", "SU2L_REP_3", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"]
SEQ_13 = ["U1Y_CHARGE_-1", "U1Y_CHARGE_0", "U1Y_CHARGE_1", "U1Y_CHARGE_2", "U1Y_CHARGE_3", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"]
SEQ_14 = ["QN_L_0", "QN_L_1", "QN_L_2", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"]
SEQ_15 = ["QN_B_0", "QN_B_1", "QN_B_2", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"]
SEQ_16 = ["PARTICLE", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"]
SEQ_17 = ["PARTICLE_ID_1", "PARTICLE_ID_2", "PARTICLE_ID_3", "PARTICLE_ID_4", "PARTICLE_ID_5", "PARTICLE_ID_6", "PARTICLE_ID_7", "PARTICLE_ID_8", "PARTICLE_ID_9", "PARTICLE_ID_10", "PARTICLE_ID_11", "PARTICLE_ID_12", "PARTICLE_ID_13", "PARTICLE_ID_14"]
SEQ_18 = ["TYPE_fermion", "TYPE_real", "TYPE_complex", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"]
SEQ_19 = ["MASS_1e-9", "MASS_1e-4", "MASS_1e-3", "MASS_1e-2", "MASS_1e-1", "MASS_1e0", "MASS_1e1", "MASS_1e2", "NA", "NA", "NA", "NA", "NA", "NA"]
SEQ_20 = ["CHARGE_-1", "CHARGE_0", "CHARGE_1", "CHARGE_2", "CHARGE_3", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"]
SEQ_21 = ["END_PARTICLE", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"]
SEQ_22 = ["END_FIELD", "PARTICLE", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"]
SEQ_23 = ["END_ITRACT", "FIELD", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"]
SEQ_24 = ["EOS", "ITRACT", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"]
PAD = "PAD"
seq_list = [
    SEQ_0, SEQ_1, SEQ_2, SEQ_3, SEQ_4, SEQ_5, SEQ_6, SEQ_7,
    SEQ_8, SEQ_9, SEQ_10, SEQ_11, SEQ_12, SEQ_13, SEQ_14, SEQ_15,
    SEQ_16, SEQ_17, SEQ_18, SEQ_19, SEQ_20, SEQ_21, SEQ_22, SEQ_23, SEQ_24
]
ALL_TOKENS = sorted(list(set(tok for seq in seq_list for tok in seq) | {PAD}))
TOKEN_TO_IDX = {token: i for i, token in enumerate(ALL_TOKENS)}
IDX_TO_TOKEN = {i: token for token, i in TOKEN_TO_IDX.items()}
VOCAB_SIZE = len(ALL_TOKENS)

track_back = {"BOS": 1, "PARTICLE": 17, "FIELD": 5, "ITRACT": 2, "END_PARTICLE": 22, "END_FIELD": 23, "END_ITRACT": 24}

# ----------------------------------------------------------------------
# Environment and Reward (No changes needed here)
# ----------------------------------------------------------------------

CHECK_NAMES = sorted(list(get_all_possible_check_names()))
TARGET_CHECKS = CHECK_NAMES
NUM_CHECKS = len(CHECK_NAMES)

@lru_cache(maxsize=10000)
def _get_reward_vector_cached(seq_tuple: Tuple[str, ...]) -> torch.Tensor:
    seq = list(seq_tuple)
    if "NA" in seq:
        return torch.zeros(NUM_CHECKS)
    try:
        def timeout_handler(signum, frame): raise TimeoutError("score_model timed out")
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)
        try:
            res = io_sequence_test(tokens=seq, keep_files=False)
            signal.alarm(0)
            if not res:
                return torch.zeros(NUM_CHECKS)
            checks_passed = res.get('checks_passed', {})
            reward_vector = torch.zeros(NUM_CHECKS)
            for i, check_name in enumerate(CHECK_NAMES):
                reward_vector[i] = checks_passed.get(check_name, 0.0)
            yukawa_idx = CHECK_NAMES.index("_yukawa_mass") if "_yukawa_mass" in CHECK_NAMES else None
            if yukawa_idx is not None and reward_vector[yukawa_idx] == 0.0:
                try:
                    dense_score = _approx_yukawa_alignment_score(seq)
                    reward_vector[yukawa_idx] = max(reward_vector[yukawa_idx], dense_score)
                except Exception:
                    pass
            return reward_vector
        except (TimeoutError, Exception):
            signal.alarm(0)
            return torch.zeros(NUM_CHECKS)
        finally:
            signal.alarm(0)
    except (ImportError, Exception):
        return torch.zeros(NUM_CHECKS)

def get_reward_vector(seq: List[str]) -> torch.Tensor:
    seq_tuple = tuple(seq)
    return _get_reward_vector_cached(seq_tuple)

# <<< FIX 1: Add 'in_particle' to the state to distinguish between FIELD and PARTICLE contexts.
def _fresh_field_state() -> dict: 
    return dict(
        in_field=False,
        in_particle=False, # <<< FIX: New state flag
        field_type=None, 
        dim=None, 
        gen=None, 
        self_conj_true=False, 
        particle_count=0, 
        non_abelian_dims=[],
        itract_type=None,
        field_count=0,
        chirality=None,
        interaction_fields_props=[],
        fermion_dims=[],
        charges_per_dim={},
        non_zero_charge_seen=False,
        charges_per_row={}
    )

INTERACTION_REQUIREMENTS = {"YUKAWA": 3}
YUKAWA_FIELD_REQUIREMENTS = [
    {'type': 'TYPE_fermion', 'chirality': 'left'},
    {'type': 'TYPE_fermion', 'chirality': 'right'},
    {'type': 'TYPE_complex', 'chirality': 'none'},
]

# <<< FIX 2: Add guards to FIELD-specific rules to prevent them from applying inside a PARTICLE block.
def _masked_seq(seq_idx: int, st: dict, current_len: int, max_len: int) -> List[str]:
    base = seq_list[seq_idx][:] 

    # --- Grammar Rules ---

    # <<< FIX: This rule should only apply to FIELD definitions, not PARTICLEs.
    if seq_idx == 7 and st["in_field"] and not st.get("in_particle", False):
        if st['itract_type'] == 'YUKAWA' and st['field_type'] in ['TYPE_complex', 'TYPE_real']:
            valid_dims = {f"DIM_{d}" for d in st['fermion_dims']}
            if valid_dims:
                base = [tok if tok in valid_dims else "NA" for tok in base]
        else:
            dims = sorted(list(set(st["non_abelian_dims"])))
            if dims:
                valid_dims = {f"DIM_{d}" for d in dims}
                base = [tok if tok in valid_dims else "NA" for tok in base]
            else:
                base = [tok if tok == "DIM_1" else "NA" for tok in base]

    # <<< FIX: This rule should only apply to FIELD definitions, not PARTICLEs.
    if seq_idx == 8 and st["in_field"] and not st.get("in_particle", False) and st["field_type"] != "TYPE_fermion":
        base = [tok if tok == "GEN_1" else "NA" for tok in base]

    if seq_idx == 8 and st["in_field"] and not st.get("in_particle", False) and st["field_type"] == "TYPE_fermion":
        base = [tok if tok in {"GEN_1", "GEN_2", "GEN_3"} else "NA" for tok in base]

    if seq_idx == 16 and st["in_field"]:
        if st["dim"] is None or st["gen"] is None:
            base = ["NA" if tok == "PARTICLE" else tok for tok in base]
        else:
            need = st["dim"] * st["gen"]
            have = st["particle_count"]
            if have >= need:
                base = ["NA" if tok == "PARTICLE" else tok for tok in base]

    # <<< FIX: This rule should only apply to FIELD definitions, not PARTICLEs.
    if seq_idx == 9 and st["in_field"] and not st.get("in_particle", False) and st["field_type"] == "TYPE_fermion":
        base = [tok if tok == "SELF_CONJ_FALSE" else "NA" if tok == "SELF_CONJ_TRUE" else tok for tok in base]

    if seq_idx == 9 and st["in_field"] and not st.get("in_particle", False) and st.get("non_zero_charge_seen", False):
        base = ["NA" if tok == "SELF_CONJ_TRUE" else tok for tok in base]

    # <<< FIX: This rule should only apply to FIELD definitions, not PARTICLEs.
    if seq_idx == 10 and st["in_field"] and not st.get("in_particle", False) and st["field_type"] == "TYPE_fermion":
        base = ["NA" if tok == "CHIRALITY_none" else tok for tok in base]
    
    if st['in_field'] and st['itract_type'] == 'YUKAWA':
        reqs_fulfilled = {'left_fermion': 0, 'right_fermion': 0, 'scalar': 0}
        for props in st.get('interaction_fields_props', []):
            if props.get('type') == 'TYPE_fermion' and props.get('chirality') == 'left':
                reqs_fulfilled['left_fermion'] += 1
            elif props.get('type') == 'TYPE_fermion' and props.get('chirality') == 'right':
                reqs_fulfilled['right_fermion'] += 1
            elif props.get('type') in ['TYPE_complex', 'TYPE_real'] and props.get('chirality') == 'none':
                reqs_fulfilled['scalar'] += 1
        
        if seq_idx == 6 and reqs_fulfilled['scalar'] >= 1:
            base = [tok if tok not in ['TYPE_complex', 'TYPE_real'] else 'NA' for tok in base]
        
        if seq_idx == 6:
            can_be_fermion = reqs_fulfilled['left_fermion'] < 1 or reqs_fulfilled['right_fermion'] < 1
            can_be_scalar = reqs_fulfilled['scalar'] < 1
            if reqs_fulfilled['left_fermion'] < 1 or reqs_fulfilled['right_fermion'] < 1:
                can_be_scalar = False
            if not can_be_fermion:
                base = [tok if tok != 'TYPE_fermion' else 'NA' for tok in base]
            if not can_be_scalar:
                base = [tok if tok not in ['TYPE_complex', 'TYPE_real'] else 'NA' for tok in base]

        if seq_idx == 10:
            if st['field_type'] == 'TYPE_fermion':
                if reqs_fulfilled['left_fermion'] >= 1:
                    base = [tok if tok != 'CHIRALITY_left' else 'NA' for tok in base]
                if reqs_fulfilled['right_fermion'] >= 1:
                    base = [tok if tok != 'CHIRALITY_right' else 'NA' for tok in base]
            elif st['field_type'] in ['TYPE_complex', 'TYPE_real']:
                base = [tok if tok == 'CHIRALITY_none' else 'NA' for tok in base]

    # <<< FIX: This rule should only apply to FIELD definitions, not PARTICLEs.
    if seq_idx == 12 and st["in_field"] and not st.get("in_particle", False):
        if 3 in st["non_abelian_dims"]:
            base = [tok if tok == "SU2L_REP_2" else "NA" for tok in base]
    
    if seq_idx == 18 and st["in_field"] and st["field_type"]:
        base = [tok if (tok == st["field_type"]) else "NA" for tok in base]

    if seq_idx == 20 and st["in_field"]:
        if st.get("dim"):
            col_idx = (st["particle_count"] - 1) % st["dim"] if st["particle_count"] > 0 else 0
            if col_idx in st.get("charges_per_dim", {}):
                fixed_tok = st["charges_per_dim"][col_idx]
                base = [tok if tok == fixed_tok else "NA" for tok in base]
            row_idx = (st["particle_count"] - 1) // st["dim"] if st["particle_count"] > 0 else 0
            if row_idx in st.get("charges_per_row", {}):
                fixed_row = st["charges_per_row"][row_idx]
                base = [tok if tok == fixed_row else "NA" for tok in base]
        if st["self_conj_true"]:
            base = [tok if (tok == "CHARGE_0") else "NA" for tok in base]

    if seq_idx == 22 and st["in_field"]:
        if st["dim"] is not None and st["gen"] is not None:
            need = st["dim"] * st["gen"]
            have = st["particle_count"]
            if have == need:
                base = ["NA" if tok != "END_FIELD" else tok for tok in base]
            else:
                base = ["NA" if tok == "END_FIELD" else tok for tok in base]

    if seq_idx == 23:
        required_fields = INTERACTION_REQUIREMENTS.get(st['itract_type'])
        if required_fields is not None:
            if st['field_count'] < required_fields:
                base = ["NA" if tok == "END_ITRACT" else tok for tok in base]
            elif st['field_count'] >= required_fields:
                base = ["NA" if tok == "FIELD" else tok for tok in base]

    if "ITRACT" in base and current_len >= max_len - 100:
        base = ["NA" if tok == "ITRACT" else tok for tok in base]
    if "FIELD" in base and current_len >= max_len - 50:
        base = ["NA" if tok == "FIELD" else tok for tok in base]
    if "PARTICLE" in base and current_len >= max_len - 10:
        base = ["NA" if tok == "PARTICLE" else tok for tok in base]
    
    if (seq_idx == 17 and st.get("itract_type") == "YUKAWA"
            and st["in_field"] and st.get("field_type") == "TYPE_fermion"
            and st.get("chirality") == "right"):
        left_ids = st.get("yukawa_left_ids", set())
        if left_ids:
            valid_set = {f"PARTICLE_ID_{id_}" for id_ in left_ids}
            base = [tok if tok in valid_set else "NA" for tok in base]
    
    return base

# ----------------------------------------------------------------------
# Actor-Critic Transformer (No changes needed here)
# ----------------------------------------------------------------------
# (Code for CausalTransformerLayer, CachedTransformerEncoder, SequenceTransformer remains the same)
class CausalTransformerLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == self.d_model, "d_model must be divisible by nhead"
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.ReLU(),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, attn_mask: Optional[torch.Tensor] = None):
        B, T, D = x.shape
        x_norm = self.norm1(x)
        q, k, v = self.qkv_proj(x_norm).split(self.d_model, dim=-1)
        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        present_key_value = (k, v)
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p if self.training else 0.0)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        attn_output = self.o_proj(attn_output)
        h = x + self.dropout(attn_output)
        h_norm = self.norm2(h)
        ffn_output = self.ffn(h_norm)
        out = h + self.dropout(ffn_output)
        return out, present_key_value
class CachedTransformerEncoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, layers: int, dim_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            CausalTransformerLayer(d_model, nhead, dim_ff, dropout)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x: torch.Tensor, cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, attn_mask: Optional[torch.Tensor] = None):
        if cache is None:
            cache = [None] * len(self.layers)
        new_cache = []
        for layer, past_kv in zip(self.layers, cache):
            x, new_kv = layer(x, past_key_value=past_kv, attn_mask=attn_mask)
            new_cache.append(new_kv)
        return self.norm(x), new_cache
class SequenceTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_T: int, num_checks: int,
                 layers: int = 4, nhead: int = 16, dropout: float = 0.2):
        super().__init__()
        if d_model % nhead != 0: raise ValueError("d_model must be divisible by nhead")
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_T, d_model)
        self.encoder = CachedTransformerEncoder(d_model, nhead, layers, d_model * 4, dropout)
        self.policy_head = nn.Linear(d_model, vocab_size)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, num_checks)
        )
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
    def forward(self, tok_ids: torch.Tensor, cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T = tok_ids.shape
        device = tok_ids.device
        if cache is not None and T == 1:
            past_len = cache[0][0].shape[-2]
            pos = torch.tensor([[past_len]], device=device, dtype=torch.long)
        else:
            pos = torch.arange(0, T, device=device).unsqueeze(0).expand(B, T)
        causal_mask = None
        if T > 1:
            causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        tok_emb = self.tok_embed(tok_ids)
        pos_emb = self.pos_embed(pos)
        h, new_cache = self.encoder(tok_emb + pos_emb, cache=cache, attn_mask=causal_mask)
        logits = self.policy_head(h)
        values = self.value_head(h)
        return logits, values, new_cache

# ----------------------------------------------------------------------
# Champions Experience Replay Buffer (No changes needed here)
# ----------------------------------------------------------------------
# (Code for ChampionsBuffer remains the same)
class ChampionsBuffer:
    def __init__(
        self,
        all_check_names: List[str],
        target_checks_raw: List[str],
        top_k: int = 50,
    ):
        self.target_checks: List[str] = []
        for name in target_checks_raw:
            if name in all_check_names:
                self.target_checks.append(name)
            elif ("_" + name) in all_check_names:
                self.target_checks.append("_" + name)
            else:
                print(f"[ChampionsBuffer] ⚠️  Target check '{name}' not found in the checklist – it will be ignored.")
        self.check_name_to_idx = {name: all_check_names.index(name) for name in self.target_checks}
        self.buffer: dict[str, List[Tuple[float, float, List[int]]]] = {name: [] for name in self.target_checks}
        self.seq_in_buffer: dict[str, set[Tuple[int, ...]]] = {name: set() for name in self.target_checks}
        self.top_k = top_k
    def _add_sequence(self, check_name: str, score: float, surprise: float, seq: List[int]):
        seq_tuple = tuple(seq)
        if seq_tuple in self.seq_in_buffer[check_name]:
            return
        if len(self.buffer[check_name]) == self.top_k:
            worst_score, worst_surprise, _ = self.buffer[check_name][-1]
            if score < worst_score or (score == worst_score and surprise <= worst_surprise):
                return
        self.buffer[check_name].append((score, surprise, seq))
        self.buffer[check_name].sort(key=lambda x: (x[0], x[1]), reverse=True)
        self.buffer[check_name] = self.buffer[check_name][: self.top_k]
        self.seq_in_buffer[check_name] = {tuple(s) for _, _, s in self.buffer[check_name]}
    def recalculate_surprises(self, surprise_calculator: callable, check_names_in_order: List[str]):
        all_seqs_with_info = []
        for check_name, champions in self.buffer.items():
            for score, _, seq in champions:
                all_seqs_with_info.append((check_name, score, tuple(seq)))
        if not all_seqs_with_info:
            return
        unique_seq_tuples = sorted(list(set(s for _, _, s in all_seqs_with_info)))
        unique_seqs_list = [list(s) for s in unique_seq_tuples]
        surprise_vectors = surprise_calculator(unique_seqs_list)
        seq_to_surprise_vec = {seq: surprise for seq, surprise in zip(unique_seq_tuples, surprise_vectors)}
        self.buffer = {name: [] for name in self.target_checks}
        self.seq_in_buffer = {name: set() for name in self.target_checks}
        check_name_to_idx = {name: i for i, name in enumerate(check_names_in_order)}
        for check_name, score, seq_tuple in all_seqs_with_info:
            surprise_vec = seq_to_surprise_vec[seq_tuple]
            check_idx = check_name_to_idx.get(check_name)
            if check_idx is None: continue
            new_surprise = surprise_vec[check_idx].item()
            self._add_sequence(check_name, score, new_surprise, list(seq_tuple))
    def update(self, sequences: List[List[int]], rewards: torch.Tensor, surprises: torch.Tensor):
        for i in range(rewards.shape[0]):
            seq = sequences[i]
            for check_name, check_idx in self.check_name_to_idx.items():
                score = rewards[i, check_idx].item()
                surprise = surprises[i, check_idx].item()
                if score > 0:
                    self._add_sequence(check_name, score, surprise, seq)
    def get_champions(self) -> List[List[int]]:
        unique: set[Tuple[int, ...]] = set()
        for seq_list in self.buffer.values():
            for _, _, seq in seq_list:
                unique.add(tuple(seq))
        return [list(seq) for seq in unique]
    def get_top_scores(self) -> dict:
        out = {}
        for name, seqs in self.buffer.items():
            out[name] = seqs[0][0] if seqs else -1.0
        return out
    def get_buffer_counts(self) -> dict[str, int]:
        return {name: len(seqs) for name, seqs in self.buffer.items()}

# ----------------------------------------------------------------------
# Helper function for finding unclosed blocks
# ----------------------------------------------------------------------

def _find_outermost_unclosed_block_start(sequence: List[int]) -> Optional[int]:
    """
    Finds the starting index of the outermost block (ITRACT > FIELD > PARTICLE)
    that was opened but never closed.
    """
    open_stacks = {"ITRACT": [], "FIELD": [], "PARTICLE": []}
    
    openers = {"ITRACT", "FIELD", "PARTICLE"}
    closers = {"END_ITRACT", "END_FIELD", "END_PARTICLE"}
    block_map = {"END_ITRACT": "ITRACT", "END_FIELD": "FIELD", "END_PARTICLE": "PARTICLE"}

    for i, token_id in enumerate(sequence):
        token_str = IDX_TO_TOKEN.get(token_id)
        if not token_str: 
            continue

        if token_str in openers:
            open_stacks[token_str].append(i)
        elif token_str in closers:
            block_type = block_map.get(token_str)
            if block_type and open_stacks[block_type]:
                open_stacks[block_type].pop()

    # Check for unclosed blocks in order of hierarchy (outermost first)
    if open_stacks["ITRACT"]:
        return open_stacks["ITRACT"][0]
    elif open_stacks["FIELD"]:
        return open_stacks["FIELD"][0]
    elif open_stacks["PARTICLE"]:
        return open_stacks["PARTICLE"][0]
    
    return None

# ----------------------------------------------------------------------
# PPO Trainer
# ----------------------------------------------------------------------
# (PPO_Trainer class and its methods need to be updated to use the new state logic)

class PPO_Trainer:
    def __init__(self, model: SequenceTransformer, optimizer: torch.optim.Optimizer,
                 scaler: torch.amp.GradScaler, champions_buffer: ChampionsBuffer,
                 batch_size: int, ppo_epochs: int, clip_epsilon: float,
                 imitation_loss_coeff: float, value_loss_coeff: float, max_len: int,
                 supervised_loss_coeff: float, device: torch.device, training_iterations: int,
                 imitation_lr_multiplier: float, csv_writer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.champions_buffer = champions_buffer
        self.batch_size = batch_size
        self.ppo_epochs = ppo_epochs
        self.clip_epsilon = clip_epsilon
        self.imitation_loss_coeff = imitation_loss_coeff
        self.value_loss_coeff = value_loss_coeff
        self.supervised_loss_coeff = supervised_loss_coeff
        self.max_len = max_len
        self.device = device
        self.check_pass_rate_history = deque(maxlen=100)
        self.training_iterations = training_iterations
        self.imitation_lr_multiplier = imitation_lr_multiplier
        self.csv_writer = csv_writer
        self._logged_sequences: set[tuple[int, ...]] = set()
        self.reward_weights = torch.full((NUM_CHECKS,), 1.0, device=self.device)
        high_reward_checks = {
            '_assign_colors': 5.0, '_check_mass_type': 5.0, '_particles_check': 5.0,
            '_assign_mass_type': 25.0, '_yukawa_mass': 25.0, '_sort_field': 25.0,
            '_all_field_pass_checks': 25.0, '_ptcl_check': 25.0,
        }
        for check_name, weight in high_reward_checks.items():
            if check_name in CHECK_NAMES:
                self.reward_weights[CHECK_NAMES.index(check_name)] = weight
        self.medium_checks = {'_assign_colors', '_check_mass_type', '_particles_check'} & set(CHECK_NAMES)
        self.hard_checks = {'_assign_mass_type', '_yukawa_mass', '_sort_field', '_all_field_pass_checks', '_ptcl_check'} & set(CHECK_NAMES)
    
    # (Helper methods like _is_one_tok_diff, _get_surprise_for_sequences, _log_interesting_sequences remain the same)
    @staticmethod
    def _is_one_tok_diff(a: list[int], b: list[int]) -> bool:
        if abs(len(a) - len(b)) > 1: return False
        if len(a) > len(b): a, b = b, a
        i = j = edits = 0
        while i < len(a) and j < len(b):
            if a[i] == b[j]: i += 1; j += 1
            else:
                edits += 1
                if edits > 1: return False
                if len(a) == len(b): i += 1; j += 1
                else: j += 1
        edits += (len(b) - j)
        return edits == 1
    def _get_surprise_for_sequences(self, sequences: List[List[int]]) -> torch.Tensor:
        if not sequences: return torch.empty(0, NUM_CHECKS, device=self.device)
        rewards = self._get_rewards_in_parallel(sequences)
        weighted_rewards = rewards * self.reward_weights.unsqueeze(0)
        seq_tensor = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(s, dtype=torch.long) for s in sequences],
            batch_first=True, padding_value=TOKEN_TO_IDX[PAD]
        ).to(self.device)
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == 'cuda'):
            _, v_pred, _ = self.model(seq_tensor)
            seq_lens = [len(s) for s in sequences]
            v_last = v_pred[torch.arange(v_pred.size(0)), [l - 1 for l in seq_lens], :]
        return (v_last - weighted_rewards).pow(2)
    def _log_interesting_sequences(self, sequences: list[list[int]], rewards: torch.Tensor):
        for idx, seq in enumerate(sequences):
            seq_tuple = tuple(seq)
            if seq_tuple in self._logged_sequences: continue
            passed_checks = {CHECK_NAMES[i] for i, r in enumerate(rewards[idx]) if r >= 1.0}
            target_pass = passed_checks & (self.medium_checks | self.hard_checks)
            should_log = bool(target_pass)
            if not should_log:
                for check_set in (self.medium_checks | self.hard_checks):
                    c_idx = CHECK_NAMES.index(check_set)
                    if rewards[idx, c_idx] > 0: continue
                    for _, _, champ_seq in self.champions_buffer.buffer.get(check_set, []):
                        if self._is_one_tok_diff(seq, champ_seq):
                            should_log = True; break
                    if should_log: break
            if should_log:
                token_strs = [IDX_TO_TOKEN[t] for t in seq]
                reward_list = rewards[idx].tolist()
                self.csv_writer.writerow([" ".join(token_strs), reward_list])
                self._logged_sequences.add(seq_tuple)

    def _generate_trajectories(self) -> dict:
        self.model.eval()
        B = self.batch_size
        
        # <<< FIX: Explicitly start with BOS token in the tracking list
        current_sequences_toks = [["BOS"] for _ in range(B)]
        
        log_probs_list = [[] for _ in range(B)]
        values_list = [[] for _ in range(B)]
        invalid_masks_list = [[] for _ in range(B)]
        seq_indices = [1] * B
        
        # <<< FIX: Use the corrected state initialization function
        field_states = [_fresh_field_state() for _ in range(B)]
        
        active_mask = torch.ones(B, dtype=torch.bool, device=self.device)
        cache = None
        masked_seq_total_time = 0.0
        
        # <<< FIX: Initial input to the model is the BOS token
        last_tok_indices = torch.full((B, 1), TOKEN_TO_IDX["BOS"], device=self.device, dtype=torch.long)

        for t in range(self.max_len - 1):
            if not active_mask.any():
                break

            with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == 'cuda'):
                logits, value_preds, cache = self.model(last_tok_indices, cache=cache)
            
            last_logits = logits[:, -1, :]
            last_values = value_preds[:, -1, :]
            mask = torch.full_like(last_logits, -float('inf'))
            invalid_mask_step_batch = torch.ones_like(last_logits, dtype=torch.bool)
            active_indices = torch.where(active_mask)[0]
            
            # <<< FIX 3: Force the first generated token (after BOS) to be ITRACT
            if t == 0:
                for i in active_indices:
                    mask[i, TOKEN_TO_IDX["ITRACT"]] = 0
                    invalid_mask_step_batch[i, TOKEN_TO_IDX["ITRACT"]] = False
            else:
                t_mask_start = time.time()
                for i in active_indices:
                    possible_tokens = _masked_seq(seq_indices[i], field_states[i], t + 1, self.max_len)
                    valid_token_indices = [TOKEN_TO_IDX[tok] for tok in possible_tokens if tok != "NA"]
                    if not valid_token_indices:
                        current_sequences_toks[i].append("NA")
                        log_probs_list[i].append(torch.tensor(0.0, device=self.device))
                        values_list[i].append(last_values[i])
                        invalid_masks_list[i].append(torch.ones_like(mask[i]))
                        active_mask[i] = False
                        continue
                    
                    mask[i, valid_token_indices] = 0
                    invalid_mask_step_batch[i, valid_token_indices] = False
                masked_seq_total_time += time.time() - t_mask_start

            mask[~active_mask, TOKEN_TO_IDX[PAD]] = 0.0
            dist = Categorical(logits=last_logits + mask)
            actions = dist.sample()

            for i in active_indices:
                if not active_mask[i]: continue

                action = actions[i]
                chosen_token = IDX_TO_TOKEN[action.item()]

                log_probs_list[i].append(dist.log_prob(action)[i])
                values_list[i].append(last_values[i])
                invalid_masks_list[i].append(invalid_mask_step_batch[i])
                
                # <<< FIX: Append the *generated* token. BOS is already in the list.
                current_sequences_toks[i].append(chosen_token)

                # --- State Update Logic ---
                # <<< FIX 4: Update state with the new 'in_particle' flag
                if chosen_token == "PARTICLE":
                    field_states[i]['particle_count'] += 1
                    field_states[i]['in_particle'] = True # <<< FIX: Enter particle context
                elif chosen_token == "END_PARTICLE":
                    field_states[i]['in_particle'] = False # <<< FIX: Exit particle context
                elif chosen_token == "ITRACT":
                    field_states[i]['itract_type'] = None
                    field_states[i]['field_count'] = 0
                    field_states[i]['interaction_fields_props'] = []
                    field_states[i]['fermion_dims'] = []
                elif chosen_token.startswith("TYPE_"):
                    if field_states[i]['in_field']:
                        field_states[i]['field_type'] = chosen_token
                    else:
                        field_states[i]['itract_type'] = chosen_token.replace("TYPE_", "")
                elif chosen_token == "FIELD":
                    itract_context = {
                        'itract_type': field_states[i]['itract_type'],
                        'field_count': field_states[i]['field_count'] + 1,
                        'interaction_fields_props': field_states[i]['interaction_fields_props'],
                        'fermion_dims': field_states[i]['fermion_dims']
                    }
                    field_states[i] = _fresh_field_state()
                    field_states[i].update(itract_context)
                    field_states[i]['in_field'] = True
                elif chosen_token == "END_FIELD":
                    props = {
                        "type": field_states[i]['field_type'],
                        "chirality": field_states[i]['chirality']
                    }
                    itract_type = field_states[i]['itract_type']
                    field_count = field_states[i]['field_count']
                    interaction_fields_props = field_states[i]['interaction_fields_props'] + [props]
                    fermion_dims = field_states[i]['fermion_dims']
                    if field_states[i]['field_type'] == 'TYPE_fermion':
                        fermion_dims = fermion_dims + [field_states[i]['dim']]
                    field_states[i] = _fresh_field_state()
                    field_states[i]['itract_type'] = itract_type
                    field_states[i]['field_count'] = field_count
                    field_states[i]['interaction_fields_props'] = interaction_fields_props
                    field_states[i]['fermion_dims'] = fermion_dims
                elif chosen_token.startswith("DIM_"):
                    field_states[i]['dim'] = int(chosen_token.split('_')[1])
                elif chosen_token.startswith("GEN_"):
                    field_states[i]['gen'] = int(chosen_token.split('_')[1])
                elif chosen_token == "SELF_CONJ_TRUE":
                    field_states[i]['self_conj_true'] = True
                elif chosen_token.startswith("CHIRALITY_"):
                    field_states[i]['chirality'] = chosen_token.replace("CHIRALITY_", "")
                elif chosen_token.startswith("PARTICLE_ID_"):
                    pid_val = chosen_token.replace("PARTICLE_ID_", "")
                    if (field_states[i].get('itract_type') == 'YUKAWA' and
                            field_states[i].get('field_type') == 'TYPE_fermion' and
                            field_states[i].get('chirality') == 'left'):
                        field_states[i].setdefault('yukawa_left_ids', set()).add(pid_val)
                elif chosen_token.startswith("SU2L_REP_"):
                    dim = int(chosen_token.split('_')[-1])
                    if dim > 1:
                        field_states[i]['non_abelian_dims'].append(dim)
                elif chosen_token.startswith("SU3C_REP_"):
                    rep_num = int(chosen_token.split('_')[-1])
                    if rep_num == 2:
                        field_states[i]['non_abelian_dims'].append(3)
                    elif rep_num == 3:
                        field_states[i]['non_abelian_dims'].append(8)
                elif chosen_token.startswith("CHARGE_"):
                    if field_states[i]['in_field'] and field_states[i]['dim']:
                        col_idx = (field_states[i]['particle_count'] - 1) % field_states[i]['dim']
                        row_idx = (field_states[i]['particle_count'] - 1) // field_states[i]['dim']
                        field_states[i]['charges_per_dim'].setdefault(col_idx, chosen_token)
                        field_states[i]['charges_per_row'].setdefault(row_idx, chosen_token)
                        if chosen_token != "CHARGE_0":
                            field_states[i]['non_zero_charge_seen'] = True
                
                if chosen_token == "EOS" or (t == self.max_len - 2):
                    active_mask[i] = False
                
                if chosen_token in track_back: seq_indices[i] = track_back[chosen_token]
                else: seq_indices[i] += 1
                
                if seq_indices[i] >= len(seq_list): active_mask[i] = False
            
            last_tok_indices = actions.unsqueeze(1)
        
        num_processes = min(B, os.cpu_count() or 1)
        start_method = 'fork' if 'fork' in multiprocessing.get_all_start_methods() else None
        
        t_reward_start = time.time()
        with multiprocessing.get_context(start_method).Pool(processes=num_processes) as pool:
            reward_vectors = pool.map(get_reward_vector, current_sequences_toks)
        reward_calc_time = time.time() - t_reward_start
        
        trajectories = {
            "sequences": [[TOKEN_TO_IDX[tok] for tok in seq] for seq in current_sequences_toks],
            "log_probs": [torch.stack(lp) for lp in log_probs_list if lp],
            "values": [torch.stack(v) for v in values_list if v],
            "rewards": reward_vectors,
            "invalid_masks": [torch.stack(im, dim=0) for im in invalid_masks_list if im]
        }
        self.model.train()
        return trajectories, {'masked_seq_time': masked_seq_total_time, 'reward_calc_time': reward_calc_time}

    # (The rest of the PPO_Trainer class, main function, and other helper functions can remain as they are)
    def _get_rewards_in_parallel(self, sequences_toks: List[List[int]]) -> torch.Tensor:
        if not sequences_toks: return torch.empty(0, NUM_CHECKS, device=self.device)
        sequences_str = [[IDX_TO_TOKEN[i] for i in seq] for seq in sequences_toks]
        num_processes = min(len(sequences_str), os.cpu_count() or 1)
        start_method = 'fork' if 'fork' in multiprocessing.get_all_start_methods() else None
        with multiprocessing.get_context(start_method).Pool(processes=num_processes) as pool:
            reward_vectors = pool.map(get_reward_vector, sequences_str)
        return torch.stack(reward_vectors).to(self.device)
    def train(self) -> dict:
        diagnostics = {}
        trajectories = {}
        t0 = time.time()
        trajectories, generation_diagnostics = self._generate_trajectories()
        if self.device.type == 'cuda': torch.cuda.synchronize()
        diagnostics['trajectory_generation_total'] = time.time() - t0
        diagnostics.update(generation_diagnostics)
        sequences_toks = trajectories["sequences"]
        values_list = trajectories["values"]
        old_log_probs = torch.nn.utils.rnn.pad_sequence(trajectories["log_probs"], batch_first=True, padding_value=0.).to(self.device)
        sequences = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(s) for s in sequences_toks], batch_first=True, padding_value=TOKEN_TO_IDX[PAD]
        ).to(self.device)
        rewards = torch.stack(trajectories["rewards"]).to(self.device)
        invalid_masks_list = trajectories["invalid_masks"]
        invalid_masks = torch.nn.utils.rnn.pad_sequence(
            invalid_masks_list, batch_first=True, padding_value=True
        ).to(self.device)
        del trajectories, invalid_masks_list
        self.check_pass_rate_history.append(rewards.mean(dim=0).cpu().numpy())
        total_correct_itracts, total_correct_fields, total_correct_particles, total_correct_field_particle_num = 0, 0, 0, 0
        for seq in sequences_toks:
            counts = _extract_blocks(seq)
            total_correct_itracts += counts.get("itract", 0)
            total_correct_fields += counts.get("field", 0)
            total_correct_particles += counts.get("particle", 0)
            total_correct_field_particle_num += counts.get("correct_particle_number", 0)
        avg_seq_len = sum(len(s) for s in sequences_toks) / self.batch_size
        eos_count = sum(1 for s in sequences_toks if TOKEN_TO_IDX["EOS"] in s)
        percent_to_reach_eos = eos_count / self.batch_size
        avg_num_itrct = total_correct_itracts / self.batch_size
        avg_num_fld = total_correct_fields / self.batch_size
        avg_num_prtcl = total_correct_particles / self.batch_size
        frac_correct_field_particles = (total_correct_field_particle_num / max(total_correct_fields, 1))
        with torch.no_grad():
            final_values_pred = torch.stack([v[-1] for v in values_list]).to(self.device)
            weighted_rewards = rewards * self.reward_weights.unsqueeze(0)
            surprises = (final_values_pred - weighted_rewards).pow(2)
        self.champions_buffer.recalculate_surprises(self._get_surprise_for_sequences, CHECK_NAMES)
        self.champions_buffer.update(sequences_toks, rewards, surprises)
        champion_sequences = self.champions_buffer.get_champions()
        precomputed_champ_invalid_masks = None
        champ_rewards = None
        champ_seq_tensor = None
        if champion_sequences:
            champ_seq_tensor = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(s) for s in champion_sequences], batch_first=True, padding_value=TOKEN_TO_IDX[PAD]
            ).to(self.device)
            champ_rewards = self._get_rewards_in_parallel(champion_sequences)
            dummy_probs_shape = (champ_seq_tensor.shape[0], champ_seq_tensor.shape[1] - 1, VOCAB_SIZE)
            precomputed_champ_invalid_masks = torch.ones(dummy_probs_shape, dtype=torch.bool, device=self.device)
            for i, seq in enumerate(champion_sequences):
                seq_str = [IDX_TO_TOKEN[t] for t in seq]
                field_st = _fresh_field_state()
                seq_idx = 1
                for t in range(len(seq) - 1):
                    if t >= precomputed_champ_invalid_masks.shape[1]: break
                    possible_tokens = _masked_seq(seq_idx, field_st, t + 1, self.max_len)
                    valid_indices = [TOKEN_TO_IDX[tok] for tok in possible_tokens if tok != "NA"]
                    if valid_indices:
                        precomputed_champ_invalid_masks[i, t, valid_indices] = False
                    chosen_token = seq_str[t + 1]
                    if chosen_token == "ITRACT":
                        field_st['itract_type'] = None; field_st['field_count'] = 0; field_st['interaction_fields_props'] = []; field_st['fermion_dims'] = []
                    elif chosen_token.startswith("TYPE_"):
                        if field_st['in_field']: field_st['field_type'] = chosen_token
                        else: field_st['itract_type'] = chosen_token.replace("TYPE_", "")
                    elif chosen_token == "FIELD":
                        itract_context = {'itract_type': field_st['itract_type'], 'field_count': field_st['field_count'] + 1, 'interaction_fields_props': field_st['interaction_fields_props'], 'fermion_dims': field_st['fermion_dims']}
                        field_st = _fresh_field_state(); field_st.update(itract_context); field_st['in_field'] = True
                    elif chosen_token == "END_FIELD":
                        props = {"type": field_st['field_type'], "chirality": field_st['chirality']}
                        itract_type = field_st['itract_type']; field_count = field_st['field_count']; interaction_fields_props = field_st['interaction_fields_props'] + [props]; fermion_dims = field_st['fermion_dims']
                        if field_st['field_type'] == 'TYPE_fermion': fermion_dims = fermion_dims + [field_st['dim']]
                        field_st = _fresh_field_state(); field_st['itract_type'] = itract_type; field_st['field_count'] = field_count; field_st['interaction_fields_props'] = interaction_fields_props; field_st['fermion_dims'] = fermion_dims
                    elif chosen_token.startswith("DIM_"): field_st['dim'] = int(chosen_token.split('_')[1])
                    elif chosen_token.startswith("GEN_"): field_st['gen'] = int(chosen_token.split('_')[1])
                    elif chosen_token == "SELF_CONJ_TRUE": field_st['self_conj_true'] = True
                    elif chosen_token.startswith("CHIRALITY_"): field_st['chirality'] = chosen_token.replace("CHIRALITY_", "")
                    elif chosen_token.startswith("PARTICLE_ID_"):
                        pid_val = chosen_token.replace("PARTICLE_ID_", "")
                        if (field_st.get('itract_type') == 'YUKAWA' and field_st.get('field_type') == 'TYPE_fermion' and field_st.get('chirality') == 'left'):
                            field_st.setdefault('yukawa_left_ids', set()).add(pid_val)
                    elif chosen_token == "PARTICLE": field_st['particle_count'] += 1
                    elif chosen_token.startswith("SU2L_REP_"):
                        dim = int(chosen_token.split('_')[-1])
                        if dim > 1: field_st['non_abelian_dims'].append(dim)
                    elif chosen_token.startswith("SU3C_REP_"):
                        rep_num = int(chosen_token.split('_')[-1])
                        if rep_num == 2: field_st['non_abelian_dims'].append(3)
                        elif rep_num == 3: field_st['non_abelian_dims'].append(8)
                    elif chosen_token.startswith("CHARGE_"):
                        if field_st['in_field'] and field_st['dim']:
                            col_idx = (field_st['particle_count'] - 1) % field_st['dim']; row_idx = (field_st['particle_count'] - 1) // field_st['dim']
                            field_st['charges_per_dim'].setdefault(col_idx, chosen_token); field_st['charges_per_row'].setdefault(row_idx, chosen_token)
                            if chosen_token != "CHARGE_0": field_st['non_zero_charge_seen'] = True
                    if chosen_token == "EOS": break
                    if chosen_token in track_back: seq_idx = track_back[chosen_token]
                    else: seq_idx += 1
                    if seq_idx >= len(seq_list): break
        ppo_loop_diagnostics = {'model_forward': 0.0, 'imitation_loss': 0.0, 'backward_and_step': 0.0}
        surprise_on_champions_per_check = torch.zeros(NUM_CHECKS, device=self.device)
        reward_weights = self.reward_weights
        ppo_loss = torch.tensor(0.0, device=self.device); policy_loss = torch.tensor(0.0, device=self.device); value_loss = torch.tensor(0.0, device=self.device)
        kl_divergence = torch.tensor(0.0, device=self.device); supervised_loss = torch.tensor(0.0, device=self.device)
        imitation_loss = torch.tensor(0.0, device=self.device); imitation_supervised_loss = torch.tensor(0.0, device=self.device); total_loss = torch.tensor(0.0, device=self.device)
        for ppo_epoch in range(self.ppo_epochs):
            self.optimizer.zero_grad(); t1_ppo = time.time()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, values, _ = self.model(sequences); actions = sequences[:, 1:]; values_pred = values[:, :-1, :]; logits_pred = logits[:, :-1, :]
                action_mask = (actions != TOKEN_TO_IDX[PAD]); weighted_rewards = rewards * reward_weights.unsqueeze(0)
                value_loss_targets = weighted_rewards.unsqueeze(1).expand_as(values_pred); value_loss = F.mse_loss(values_pred[action_mask], value_loss_targets[action_mask])
                advantages = (weighted_rewards.unsqueeze(1) - values_pred.detach())
                
                # --- START: Reward Shaping for Truncation/EOS ---
                for i in range(self.batch_size):
                    seq_len = len(sequences_toks[i])
                    # Check if the original sequence was truncated (no EOS)
                    if TOKEN_TO_IDX["EOS"] not in sequences_toks[i]:
                        start_idx = _find_outermost_unclosed_block_start(sequences_toks[i])
                        if start_idx is not None:
                            # Apply penalty from the start of the failed block onward
                            # The penalty is a simple scalar added to the advantage of each check
                            penalty = -0.05  # INCOMPLETE_BLOCK_PENALTY_PER_TOKEN
                            advantages[i, start_idx:, :] += penalty
                    else:
                        # Find the EOS position and apply a bonus
                        try:
                            eos_idx = sequences_toks[i].index(TOKEN_TO_IDX["EOS"])
                            # Apply bonus to the advantage at the EOS step
                            bonus = 0.1  # EOS_BONUS
                            if eos_idx > 0:  # Safety check
                                advantages[i, eos_idx - 1, :] += bonus
                        except ValueError:
                            pass  # Should not happen if EOS is in list
                # --- END: Reward Shaping ---
                
                dist = Categorical(logits=logits_pred); new_log_probs = dist.log_prob(actions)
                ratios = torch.exp(new_log_probs[action_mask] - old_log_probs[action_mask]); valid_advantages = advantages[action_mask]
                surr1 = ratios.unsqueeze(-1) * valid_advantages; surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon).unsqueeze(-1) * valid_advantages
                policy_loss = -torch.min(surr1, surr2).mean(); ppo_loss = policy_loss + self.value_loss_coeff * value_loss; kl_divergence = (ratios - 1 - torch.log(ratios)).mean()
                probs = F.softmax(logits_pred, dim=-1); min_len = min(probs.shape[1], invalid_masks.shape[1]); aligned_probs = probs[:, :min_len, :]
                aligned_masks = invalid_masks[:, :min_len, :].float(); aligned_action_mask = action_mask[:, :min_len]
                masked_invalid_prob_mass = (aligned_probs * aligned_masks).sum(dim=-1)[aligned_action_mask]; supervised_loss = F.relu(masked_invalid_prob_mass - 0.1).mean()
                trajectory_loss = ppo_loss + self.supervised_loss_coeff * supervised_loss
            self.scaler.scale(trajectory_loss).backward(); self.scaler.step(self.optimizer); self.scaler.update()
            if self.device.type == 'cuda': torch.cuda.synchronize()
            ppo_loop_diagnostics['model_forward'] += time.time() - t1_ppo
        t1_imitation = time.time()
        if champion_sequences:
            CHAMPION_EPOCHS = 20; CHAMPION_BATCH_SIZE = 32
            champion_dataset = TensorDataset(champ_seq_tensor, champ_rewards, precomputed_champ_invalid_masks)
            champion_dataloader = DataLoader(champion_dataset, batch_size=CHAMPION_BATCH_SIZE, shuffle=True, drop_last=True)
            if len(champion_dataloader) == 0: champion_dataloader = DataLoader(champion_dataset, batch_size=len(champion_dataset), shuffle=True)
            for _ in range(CHAMPION_EPOCHS):
                for batch in champion_dataloader:
                    top_seq_tensor_b, top_rewards_b, top_invalid_masks_b = batch; self.optimizer.zero_grad()
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        champ_logits, champ_values, _ = self.model(top_seq_tensor_b); weighted_champ_rewards = top_rewards_b * self.reward_weights.unsqueeze(0)
                        champ_values_pred = champ_values[:, :-1, :]; champ_action_mask = (top_seq_tensor_b[:, 1:] != TOKEN_TO_IDX[PAD])
                        champ_value_targets = weighted_champ_rewards.unsqueeze(1).expand_as(champ_values_pred); imitation_value_loss = F.mse_loss(champ_values_pred[champ_action_mask], champ_value_targets[champ_action_mask])
                        targets = top_seq_tensor_b[:, 1:]; logits_for_ce = champ_logits[:, :-1, :].contiguous()
                        imitation_ce_loss = F.cross_entropy(logits_for_ce.view(-1, logits_for_ce.size(-1)), targets.contiguous().view(-1), ignore_index=TOKEN_TO_IDX[PAD])
                        imitation_loss = (imitation_ce_loss + self.value_loss_coeff * imitation_value_loss) * self.imitation_lr_multiplier
                        champ_logits_for_loss = champ_logits[:, :-1, :]; max_len_is = min(champ_logits_for_loss.shape[1], top_invalid_masks_b.shape[1])
                        champ_probs = F.softmax(champ_logits_for_loss[:, :max_len_is, :], dim=-1); champ_invalid_masks_for_loss = top_invalid_masks_b[:, :max_len_is, :]
                        aligned_champ_action_mask = champ_action_mask[:, :max_len_is]; masked_invalid_prob_mass_is = (champ_probs * champ_invalid_masks_for_loss.float()).sum(dim=-1)[aligned_champ_action_mask]
                        imitation_supervised_loss = F.relu(masked_invalid_prob_mass_is - 0.1).mean() * self.imitation_lr_multiplier
                        if torch.isnan(imitation_supervised_loss): imitation_supervised_loss = torch.tensor(0.0, device=self.device)
                        champion_total_loss = imitation_loss + imitation_supervised_loss
                    self.scaler.scale(champion_total_loss).backward(); self.scaler.step(self.optimizer); self.scaler.update()
            with torch.no_grad():
                champ_logits, champ_values, _ = self.model(champ_seq_tensor); champ_seq_lens = [len(s) for s in champion_sequences]
                final_champ_values_pred = champ_values[torch.arange(champ_values.size(0)), [l - 1 for l in champ_seq_lens], :]
                weighted_rewards_all = champ_rewards * self.reward_weights.unsqueeze(0)
                surprise_tensor = (final_champ_values_pred - weighted_rewards_all).pow(2)
                surprise_on_champions_per_check = surprise_tensor.mean(dim=0)
        if self.device.type == 'cuda': torch.cuda.synchronize()
        ppo_loop_diagnostics['imitation_loss'] += time.time() - t1_imitation
        total_loss = ppo_loss + supervised_loss + imitation_loss + imitation_supervised_loss
        for k, v in ppo_loop_diagnostics.items(): diagnostics[f'ppo_loop_{k}'] = v
        avg_pass_rate_per_check = rewards.mean(dim=0); avg_surprise_per_check = surprises.mean(dim=0)
        champions_scores = self.champions_buffer.get_top_scores(); buffer_counts = self.champions_buffer.get_buffer_counts()
        weighted_pass_rate_per_check = avg_pass_rate_per_check * self.reward_weights
        self._log_interesting_sequences(sequences_toks, rewards)
        return {"total_loss": total_loss.item(), "ppo_loss": ppo_loss.item(), "policy_loss": policy_loss.item(), "value_loss": value_loss.item(),
            "imitation_loss": imitation_loss.item(), "supervised_loss": supervised_loss.item(), "imitation_supervised_loss": imitation_supervised_loss.item(),
            "entropy": dist.entropy()[action_mask].mean().item(), "kl_divergence": kl_divergence.item(), "avg_seq_len": avg_seq_len,
            "percent_to_reach_eos": percent_to_reach_eos, "avg_pass_rate_per_check": avg_pass_rate_per_check.cpu().numpy(),
            "weighted_pass_rate_per_check": weighted_pass_rate_per_check.cpu().numpy(), "avg_surprise_per_check": avg_surprise_per_check.cpu().numpy(),
            "surprise_on_champions_per_check": surprise_on_champions_per_check.cpu().numpy(), "champions_scores": champions_scores,
            "buffer_counts": buffer_counts, "avg_num_itrct": avg_num_itrct, "avg_num_fld": avg_num_fld, "avg_num_prtcl": avg_num_prtcl,
            "frac_correct_field_particles": frac_correct_field_particles, "diagnostics": diagnostics,
        }
# (The rest of the file remains the same)
def main():
    torch._dynamo.config.cache_size_limit = 64; D_MODEL = 64; N_HEAD = 8; N_LAYERS = 4; DROPOUT = 0.1; MAX_LEN = 400; BATCH_SIZE = 256
    LEARNING_RATE = 3e-5; PPO_EPOCHS = 12; CLIP_EPSILON = 0.2; VALUE_LOSS_COEFF = 0.5; IMITATION_LOSS_COEFF = 1.0; SUPERVISED_LOSS_COEFF = 100
    TRAINING_ITERATIONS = 1000; IMITATION_LR_MULTIPLIER = 0.03
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Using device: {device}")
    print(f"PPO Epochs: {PPO_EPOCHS}, Imitation LR Multiplier: {IMITATION_LR_MULTIPLIER}x")
    model = SequenceTransformer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, max_T=MAX_LEN, num_checks=NUM_CHECKS, layers=N_LAYERS, nhead=N_HEAD, dropout=DROPOUT).to(device)
    model = torch.compile(model, fullgraph=True); scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available()); champions_buffer = ChampionsBuffer(CHECK_NAMES, TARGET_CHECKS)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE); CSV_LOG_PATH = "interesting_sequences.csv"; csv_already_exists = os.path.isfile(CSV_LOG_PATH)
    csv_log_handle = open(CSV_LOG_PATH, "a", newline=""); csv_writer = csv.writer(csv_log_handle)
    if not csv_already_exists: csv_writer.writerow(["tokens", "check_results"])
    trainer = PPO_Trainer(model=model, optimizer=optimizer, scaler=scaler, champions_buffer=champions_buffer, batch_size=BATCH_SIZE, ppo_epochs=PPO_EPOCHS,
        clip_epsilon=CLIP_EPSILON, imitation_loss_coeff=IMITATION_LOSS_COEFF, value_loss_coeff=VALUE_LOSS_COEFF, supervised_loss_coeff=SUPERVISED_LOSS_COEFF,
        max_len=MAX_LEN, device=device, training_iterations=TRAINING_ITERATIONS, imitation_lr_multiplier=IMITATION_LR_MULTIPLIER, csv_writer=csv_writer
    ); reward_history = deque(maxlen=100)
    minimal_yukawa_sequence = ['BOS', 'ITRACT', 'ITRACT_ID_1', 'TYPE_YUKAWA', 'FIELD', 'FIELD_ID_1', 'TYPE_fermion', 'DIM_2', 'GEN_3', 'SELF_CONJ_FALSE', 'CHIRALITY_left',
        'SU3C_REP_1', 'SU2L_REP_2', 'U1Y_CHARGE_-1', 'QN_L_1', 'QN_B_0', 'PARTICLE', 'PARTICLE_ID_1', 'TYPE_fermion', 'MASS_1e-9', 'CHARGE_0', 'END_PARTICLE',
        'PARTICLE', 'PARTICLE_ID_2', 'TYPE_fermion', 'MASS_1e-4', 'CHARGE_-1', 'END_PARTICLE', 'PARTICLE', 'PARTICLE_ID_3', 'TYPE_fermion', 'MASS_1e-9', 'CHARGE_0', 'END_PARTICLE',
        'PARTICLE', 'PARTICLE_ID_4', 'TYPE_fermion', 'MASS_1e-1', 'CHARGE_-1', 'END_PARTICLE', 'PARTICLE', 'PARTICLE_ID_5', 'TYPE_fermion', 'MASS_1e-9', 'CHARGE_0', 'END_PARTICLE',
        'PARTICLE', 'PARTICLE_ID_6', 'TYPE_fermion', 'MASS_1e0', 'CHARGE_-1', 'END_PARTICLE', 'END_FIELD', 'FIELD', 'FIELD_ID_2', 'TYPE_fermion', 'DIM_1', 'GEN_3', 'SELF_CONJ_FALSE', 'CHIRALITY_right',
        'SU3C_REP_1', 'SU2L_REP_1', 'U1Y_CHARGE_-1', 'QN_L_1', 'QN_B_0', 'PARTICLE', 'PARTICLE_ID_2', 'TYPE_fermion', 'MASS_1e-4', 'CHARGE_-1', 'END_PARTICLE',
        'PARTICLE', 'PARTICLE_ID_4', 'TYPE_fermion', 'MASS_1e-1', 'CHARGE_-1', 'END_PARTICLE', 'PARTICLE', 'PARTICLE_ID_6', 'TYPE_fermion', 'MASS_1e0', 'CHARGE_-1', 'END_PARTICLE', 'END_FIELD',
        'FIELD', 'FIELD_ID_3', 'TYPE_complex', 'DIM_2', 'GEN_1', 'SELF_CONJ_FALSE', 'CHIRALITY_none', 'SU3C_REP_1', 'SU2L_REP_2', 'U1Y_CHARGE_1', 'QN_L_0', 'QN_B_0',
        'PARTICLE', 'PARTICLE_ID_7', 'TYPE_complex', 'MASS_1e2', 'CHARGE_1', 'END_PARTICLE', 'PARTICLE', 'PARTICLE_ID_8', 'TYPE_complex', 'MASS_1e2', 'CHARGE_0', 'END_PARTICLE', 'END_FIELD', 'END_ITRACT', 'EOS']
    reward_vector = get_reward_vector(minimal_yukawa_sequence); yukawa_mass_check_name = '_yukawa_mass'; seq_indices = [TOKEN_TO_IDX[tok] for tok in minimal_yukawa_sequence]
    if yukawa_mass_check_name in CHECK_NAMES:
        score = reward_vector[CHECK_NAMES.index(yukawa_mass_check_name)].item(); print(f"Yukawa mass check score for the minimal sequence: {score:.1f}")
        initial_surprise = torch.zeros_like(reward_vector).unsqueeze(0); champions_buffer.update([seq_indices], reward_vector.unsqueeze(0), initial_surprise)
        if champions_buffer.get_top_scores().get(yukawa_mass_check_name, -1.0) > 0: print("Successfully added minimal Yukawa sequence to Champions Buffer.")
        else: print("Minimal sequence did not pass the Yukawa check, will not be added as a champion.")
    else: print(f"Check '{yukawa_mass_check_name}' not found in CHECK_NAMES.")
    SUPERVISED_WARMUP_STEPS = 0; champ_seq_tensor = torch.tensor([seq_indices], device=device)
    for step in tqdm(range(SUPERVISED_WARMUP_STEPS), desc="Supervised warm-up"):
        optimizer.zero_grad(); logits, _, _ = model(champ_seq_tensor); logits_for_loss = logits[:, :-1, :].contiguous(); targets = champ_seq_tensor[:, 1:].contiguous()
        ce_loss = F.cross_entropy(logits_for_loss.view(-1, logits_for_loss.size(-1)), targets.view(-1)); scaler.scale(ce_loss).backward(); scaler.step(optimizer); scaler.update()
    with torch.no_grad():
        logits, _, _ = model(champ_seq_tensor); logits_eval = logits[:, :-1, :]; targets_eval = champ_seq_tensor[:, 1:]
        ce_loss_final = F.cross_entropy(logits_eval.contiguous().view(-1, logits_eval.size(-1)), targets_eval.contiguous().view(-1)).item()
        preds = logits_eval.argmax(dim=-1); accuracy_final = (preds == targets_eval).float().mean().item()
    print(f"Supervised warm-up complete. Token accuracy: {accuracy_final:.3f}, CE loss: {ce_loss_final:.4f}\nBeginning PPO training…")
    for i in tqdm(range(TRAINING_ITERATIONS), desc="PPO Training"):
        log = trainer.train()
        if i % 5 == 0:
            print(f"\n{'='*40} Iteration {i} {'='*40}"); print(f"  {'General Stats':<25} | {'Value':<15}"); print(f"  {'-'*25} | {'-'*15}")
            print(f"  {'Total Loss':<25} | {log['total_loss']:.4f}"); print(f"  {'PPO Loss':<25} | {log['ppo_loss']:.4f}"); print(f"  {'Policy Loss':<25} | {log['policy_loss']:.4f}")
            print(f"  {'Value Loss':<25} | {log['value_loss']:.4f}"); print(f"  {'Imitation Loss':<25} | {log['imitation_loss']:.4f}"); print(f"  {'Supervised Loss':<25} | {log['supervised_loss']:.4f}")
            print(f"  {'Imitation Supervised Loss':<25} | {log['imitation_supervised_loss']:.4f}"); print(f"  {'Entropy':<25} | {log['entropy']:.4f}"); print(f"  {'KL Divergence':<25} | {log['kl_divergence']:.4f}"); print("-" * 60)
            if 'diagnostics' in log:
                print(f"  {'Timing Diagnostics (s)':<35} | {'Duration':<15}"); print(f"  {'-'*35} | {'-'*15}"); total_time = sum(log['diagnostics'].values())
                for name, duration in log['diagnostics'].items():
                    percent = (duration / total_time) * 100 if total_time > 0 else 0; print(f"  {name.replace('_', ' ').title():<35} | {duration:<15.4f} ({percent:.1f}%)")
                print(f"  {'-'*35} | {'-'*15}"); print(f"  {'Total':<35} | {total_time:<15.4f}"); print("-" * 60)
            print(f"  {'Sequence Stats':<25} | {'Value':<15}"); print(f"  {'-'*25} | {'-'*15}"); print(f"  {'Avg Length':<25} | {log['avg_seq_len']:.2f}"); print(f"  {'% Reaching EOS':<25} | {log['percent_to_reach_eos']:.2%}")
            print(f"  {'Avg #Correct Interactions':<25} | {log['avg_num_itrct']:.2f}"); print(f"  {'Avg #Correct Fields':<25} | {log['avg_num_fld']:.2f}"); print(f"  {'Avg #Correct Particles':<25} | {log['avg_num_prtcl']:.2f}")
            print(f"  {'% Fields w/Correct #Prtcls':<25} | {log['frac_correct_field_particles']:.2%}"); print("-" * 60)
            print(f"  {'Per-Check Stats':<30} | {'Pass Rate':<15} | {'W. Pass Rate':<15} | {'Top Score':<15} | {'# Champions':<15} | {'Surprise (MSE)':<15} | {'Champ Surprise':<15}")
            print(f"  {'-'*30} | {'-'*15} | {'-'*15} | {'-'*15} | {'-'*15} | {'-'*15} | {'-'*15}"); champions_scores = log['champions_scores']; buffer_counts = log['buffer_counts']
            for idx, name in enumerate(CHECK_NAMES):
                pass_rate = log['avg_pass_rate_per_check'][idx]; weighted_pass_rate = log['weighted_pass_rate_per_check'][idx]; surprise = log['avg_surprise_per_check'][idx]
                champ_surprise = log['surprise_on_champions_per_check'][idx]; top_score = champions_scores.get(name, -1.0); num_champs = buffer_counts.get(name, 0)
                if (name in champions_scores) or (top_score < 1.0) or (pass_rate < 1.0):
                    print(f"  {name:<30} | {pass_rate:<15.3f} | {weighted_pass_rate:<15.3f} | {top_score:<15.3f} | {num_champs:<15} | {surprise:<15.4f} | {champ_surprise:<15.4f}")
            avg_overall_pass_rate = log['avg_pass_rate_per_check'].mean(); avg_weighted_pass_rate = log['weighted_pass_rate_per_check'].mean()
            avg_overall_surprise = log['avg_surprise_per_check'].mean(); avg_champ_surprise = log['surprise_on_champions_per_check'].mean()
            print(f"  {'-'*30} | {'-'*15} | {'-'*15} | {'-'*15} | {'-'*15} | {'-'*15} | {'-'*15}")
            print(f"  {'AVERAGE':<30} | {avg_overall_pass_rate:<15.3f} | {avg_weighted_pass_rate:<15.3f} | {'-':<15} | {'-':<15} | {avg_overall_surprise:<15.4f} | {avg_champ_surprise:<15.4f}")
            print("=" * (127 + 18))
def _extract_blocks(seq_int: List[int]) -> dict:
    counts = {"itract": 0, "field": 0, "particle": 0, "correct_particle_number": 0}; i = 0
    while i < len(seq_int):
        if IDX_TO_TOKEN[seq_int[i]] == "ITRACT":
            end_itract_idx = _find_matching_end(seq_int, i, "END_ITRACT")
            if end_itract_idx != -1:
                counts["itract"] += 1; j = i + 1; dim_val, gen_val = None, None; n_particles_in_field = 0
                while j < end_itract_idx:
                    if IDX_TO_TOKEN[seq_int[j]] == "FIELD":
                        end_field_idx = _find_matching_end(seq_int, j, "END_FIELD")
                        if end_field_idx != -1 and end_field_idx < end_itract_idx:
                            counts["field"] += 1; first_particle_idx = next((k for k in range(j + 1, end_field_idx) if IDX_TO_TOKEN[seq_int[k]] == "PARTICLE"), end_field_idx)
                            dim_val, gen_val = None, None; n_particles_in_field = 0
                            header_tokens = [IDX_TO_TOKEN[t] for t in seq_int[j:first_particle_idx]]
                            for tok in header_tokens:
                                if tok.startswith("DIM_"):
                                    try: dim_val = int(tok.split("_")[1])
                                    except ValueError: dim_val = None
                                elif tok.startswith("GEN_"):
                                    try: gen_val = int(tok.split("_")[1])
                                    except ValueError: gen_val = None
                            k = j + 1
                            while k < end_field_idx:
                                if IDX_TO_TOKEN[seq_int[k]] == "PARTICLE":
                                    end_particle_idx = _find_matching_end(seq_int, k, "END_PARTICLE")
                                    if end_particle_idx != -1 and end_particle_idx < end_field_idx:
                                        counts["particle"] += 1; n_particles_in_field += 1; k = end_particle_idx
                                k += 1
                            if dim_val is not None and gen_val is not None:
                                if n_particles_in_field > 0 and n_particles_in_field == (dim_val * gen_val):
                                    counts["correct_particle_number"] += 1
                            j = end_field_idx
                    j += 1
                i = end_itract_idx
        i += 1
    return counts
def _find_matching_end(seq_int: List[int], start_idx: int, end_tok_name: str) -> int:
    for i in range(start_idx + 1, len(seq_int)):
        if IDX_TO_TOKEN[seq_int[i]] == end_tok_name: return i
    return -1
def _approx_yukawa_alignment_score(seq_tokens: list[str]) -> float:
    in_itract = False; current_field = None; fields: list[dict] = []; idx = 0
    while idx < len(seq_tokens):
        tok = seq_tokens[idx]
        if tok == "ITRACT": in_itract = True; itract_type = None; fields = []
        elif in_itract and tok.startswith("TYPE_") and itract_type is None:
            itract_type = tok.replace("TYPE_", "");
            if itract_type != "YUKAWA": in_itract = False
        elif in_itract and tok == "FIELD": current_field = {"chirality": None, "ids": set(), "dim": None, "gen": None, "type": None,}; fields.append(current_field)
        elif current_field is not None:
            if tok.startswith("CHIRALITY_"): current_field["chirality"] = tok.replace("CHIRALITY_", "")
            elif tok.startswith("PARTICLE_ID_"): current_field["ids"].add(tok.replace("PARTICLE_ID_", ""))
            elif tok.startswith("DIM_"):
                try: current_field["dim"] = int(tok.split("_")[1])
                except ValueError: pass
            elif tok.startswith("GEN_"):
                try: current_field["gen"] = int(tok.split("_")[1])
                except ValueError: pass
            elif tok == "END_ITRACT": break
        idx += 1
    left_field = next((f for f in fields if f["chirality"] == "left"), None); right_field = next((f for f in fields if f["chirality"] == "right"), None)
    if not (left_field and right_field): return 0.0
    dim = left_field["dim"] or 1; gen = left_field["gen"] or 1; total_needed = dim * gen
    if total_needed == 0: return 0.0
    overlap = len(left_field["ids"] & right_field["ids"]); dense_score = min(overlap / total_needed, 0.99)
    return dense_score
    
if __name__ == "__main__":
    main()