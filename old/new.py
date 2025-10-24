# full_ppo_blueprint_with_type_dependent_grammar.py
# Final version with a corrected grammar that allows for meaningful validation failures.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
from dataclasses import dataclass, field
import numpy as np
import time
import os
import sys
import logging
from typing import List, Dict, Optional, Any, Tuple

# ======================================================================================
# Phase 0: Configuration and Global Constants
# ======================================================================================

# --- Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
ITYPE = torch.int64

@dataclass
class PPOConfig:
    run_name: str = f"ppo_blueprint_{int(time.time())}"
    log_dir: str = "runs"
    total_steps: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-4
    ppo_epochs: int = 4
    num_minibatches: int = 2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    sup_coef: float = 1.0
    max_grad_norm: float = 0.5
    seq_len: int = 200
    alive_r: float = 0.01
    len_target: int = 180
    len_sigma: float = 30.0
    particle_bonus_weight: float = 0.5
    field_bonus_weight: float = 2.0
    terminal_bonus_weight: float = 20.0
    minibatch_size: int = field(init=False)
    def __post_init__(self): self.minibatch_size = self.batch_size // self.num_minibatches

# --- Dummy Logger and Model for Self-Containment ---
class DummyLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def debug(self, msg): pass

class DummyTransformerPolicy(nn.Module):
    def __init__(self, vocab_size, d_model, max_T):
        super().__init__(); self.vocab_size = vocab_size; self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_T, d_model); self.transformer = nn.TransformerEncoderLayer(d_model=d_model, nhead=2, dim_feedforward=d_model*2, batch_first=True)
        self.policy_head = nn.Linear(d_model, vocab_size); self.value_head = nn.Linear(d_model, 1)
    def forward(self, x: torch.Tensor, cache: Optional[Any] = None) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        B, T = x.shape; positions = torch.arange(0, T, device=x.device).unsqueeze(0).expand(B, T)
        tok_emb = self.embedding(x); pos_emb = self.pos_embedding(positions); x = tok_emb + pos_emb
        out = self.transformer(x); logits = self.policy_head(out); values = self.value_head(out).squeeze(-1)
        return logits, values, None

# ======================================================================================
# Part 1: score_calculator.py Logic
# ======================================================================================
MAX_PARTICLES, MAX_FIELDS, MAX_INTERACTIONS, MAX_PARTICLES_PER_FIELD = 50, 50, 50, 50
PAD_VALUE = -1
PARTICLE_TYPE_MAP = {'fermion': 0, 'real': 1, 'complex': 2, 'vector': 3}
FIELD_TYPE_MAP = {'fermion': 0, 'real': 1, 'complex': 2, 'vector': 3, 'scalar': 4}
CHIRALITY_MAP = {'none': 0, 'left': 1, 'right': 2}
P_FEAT_ID, P_FEAT_TYPE, P_FEAT_MASS, P_FEAT_CHARGE = range(4); NUM_PARTICLE_FEATURES = 4
(F_FEAT_ID, F_FEAT_TYPE, F_FEAT_DIM, F_FEAT_GEN, F_FEAT_SELF_CONJ, F_FEAT_CHIRALITY,
 F_FEAT_SU3_REP, F_FEAT_SU2_REP, F_FEAT_U1_CHARGE, F_FEAT_QN_L, F_FEAT_QN_B) = range(11); NUM_FIELD_FEATURES = 11
I_FEAT_ID, I_FEAT_TYPE, I_FEAT_F0, I_FEAT_F1, I_FEAT_F2 = range(5); NUM_INTERACTION_FEATURES = 5

def ingest_batch(batch_of_models: list) -> dict:
    batch_size = len(batch_of_models)
    np_particles = np.full((batch_size, MAX_PARTICLES, NUM_PARTICLE_FEATURES), PAD_VALUE, dtype=np.float32)
    np_fields = np.full((batch_size, MAX_FIELDS, NUM_FIELD_FEATURES), PAD_VALUE, dtype=np.float32)
    np_interactions = np.full((batch_size, MAX_INTERACTIONS, NUM_INTERACTION_FEATURES), PAD_VALUE, dtype=np.int64)
    np_field_particle_map = np.full((batch_size, MAX_FIELDS, MAX_PARTICLES_PER_FIELD), PAD_VALUE, dtype=np.int64)
    for i, model in enumerate(batch_of_models):
        if not model: continue
        p_map = {p_data['id']: j for j, p_data in enumerate(model.get('particles', []))}
        f_map = {f_data['id']: j for j, f_data in enumerate(model.get('fields', []))}
        for j, p_data in enumerate(model.get('particles', [])):
            if j >= MAX_PARTICLES: continue
            np_particles[i, j, P_FEAT_ID] = p_map.get(p_data['id'], -1); np_particles[i, j, P_FEAT_TYPE] = PARTICLE_TYPE_MAP.get(p_data.get('type'), -1)
            np_particles[i, j, P_FEAT_MASS] = p_data.get('mass', 0.0); np_particles[i, j, P_FEAT_CHARGE] = p_data.get('charge', 0)
        for j, f_data in enumerate(model.get('fields', [])):
            if j >= MAX_FIELDS: continue
            np_fields[i, j, F_FEAT_ID] = f_map.get(f_data['id'], -1); np_fields[i, j, F_FEAT_TYPE] = FIELD_TYPE_MAP.get(f_data.get('type', ''), -1)
            np_fields[i, j, F_FEAT_DIM] = f_data.get('dim', 1); np_fields[i, j, F_FEAT_GEN] = f_data.get('gen', 1)
            np_fields[i, j, F_FEAT_SELF_CONJ] = 1.0 if f_data.get('self_conjugate', False) else 0.0
            np_fields[i, j, F_FEAT_CHIRALITY] = CHIRALITY_MAP.get(f_data.get('chirality', 'none'), 0)
            np_fields[i, j, F_FEAT_SU3_REP] = f_data.get('SU3_rep', 1); np_fields[i, j, F_FEAT_SU2_REP] = f_data.get('SU2_rep', 1)
            np_fields[i, j, F_FEAT_U1_CHARGE] = f_data.get('U1_charge', 0); np_fields[i, j, F_FEAT_QN_L] = f_data.get('QN_L', 0)
            np_fields[i, j, F_FEAT_QN_B] = f_data.get('QN_B', 0)
            for k, p_id in enumerate(f_data.get('particles', [])):
                if k >= MAX_PARTICLES_PER_FIELD: break
                if p_id in p_map: np_field_particle_map[i, j, k] = p_map[p_id]
        for j, itr_data in enumerate(model.get('interactions', [])):
            if j >= MAX_INTERACTIONS: continue
            np_interactions[i, j, I_FEAT_ID] = j; np_interactions[i, j, I_FEAT_TYPE] = 0 # Assume Yukawa
            field_ids = [f_map.get(f_id, -1) for f_id in itr_data.get('fields', [])]
            if len(field_ids) >= 3:
                np_interactions[i, j, I_FEAT_F0] = field_ids[0]; np_interactions[i, j, I_FEAT_F1] = field_ids[1]; np_interactions[i, j, I_FEAT_F2] = field_ids[2]
    return {'particles': torch.from_numpy(np_particles).to(DEVICE), 'fields': torch.from_numpy(np_fields).to(DEVICE),'interactions': torch.from_numpy(np_interactions).to(DEVICE, dtype=ITYPE),'field_particle_map': torch.from_numpy(np_field_particle_map).to(DEVICE, dtype=ITYPE)}

def parallel_validator_and_scorer(tensors: dict, level: str = 'all') -> Tuple[torch.Tensor, int]:
    if 'particles' not in tensors or tensors['particles'].shape[0] == 0: return torch.tensor([], device=DEVICE), 0
    batch_size = tensors['particles'].shape[0]; scores = torch.zeros(batch_size, device=DEVICE); total_checks = 0
    particles = tensors['particles']; valid_particles = particles[:, :, P_FEAT_ID] != PAD_VALUE
    if level in ['particle', 'field', 'all']:
        scores += ((particles[:, :, P_FEAT_MASS] >= 0) | ~valid_particles).all(dim=1); total_checks += 1
        scores += ((particles[:, :, P_FEAT_TYPE] != -1) | ~valid_particles).all(dim=1); total_checks += 1
    if level in ['field', 'all']:
        fields = tensors['fields']; field_particle_map = tensors['field_particle_map']; valid_fields = fields[:, :, F_FEAT_ID] != PAD_VALUE
        is_fermion_field = (fields[:, :, F_FEAT_TYPE] == FIELD_TYPE_MAP['fermion'])
        scores += ((fields[:, :, F_FEAT_TYPE] != -1) | ~valid_fields).all(dim=1); total_checks += 1
        scores += ((fields[:, :, F_FEAT_DIM] > 0) | ~valid_fields).all(dim=1); total_checks += 1
        scores += ((fields[:, :, F_FEAT_GEN] > 0) | ~valid_fields).all(dim=1); total_checks += 1
        num_particles_in_field = (field_particle_map != PAD_VALUE).sum(dim=2); expected_particles = fields[:, :, F_FEAT_DIM] * fields[:, :, F_FEAT_GEN]
        scores += ((num_particles_in_field == expected_particles) | ~valid_fields).all(dim=1); total_checks += 1
        scores += (((fields[:, :, F_FEAT_CHIRALITY] != CHIRALITY_MAP['none']) | ~is_fermion_field) | ~valid_fields).all(dim=1); total_checks += 1
        B, MAX_F, MAX_P_PER_F = field_particle_map.shape; batch_idx_3d = torch.arange(B, device=DEVICE).view(B, 1, 1); f_p_map_clamped = field_particle_map.clamp(min=0)
        p_types_in_fields = particles[batch_idx_3d, f_p_map_clamped, P_FEAT_TYPE]; f_types_expanded = fields[:, :, F_FEAT_TYPE].unsqueeze(2).expand(-1, -1, MAX_P_PER_F)
        is_scalar_field_exp = (f_types_expanded == FIELD_TYPE_MAP['real']) | (f_types_expanded == FIELD_TYPE_MAP['complex']) | (f_types_expanded == FIELD_TYPE_MAP['scalar'])
        is_scalar_particle = (p_types_in_fields == PARTICLE_TYPE_MAP['real']) | (p_types_in_fields == PARTICLE_TYPE_MAP['complex'])
        type_match_ok = ((f_types_expanded == FIELD_TYPE_MAP['fermion']) & (p_types_in_fields == PARTICLE_TYPE_MAP['fermion'])) | (is_scalar_field_exp & is_scalar_particle)
        scores += ((type_match_ok | ~(field_particle_map != PAD_VALUE)).all(dim=2) | ~valid_fields).all(dim=1); total_checks += 1
        is_self_conj = (fields[:, :, F_FEAT_SELF_CONJ] == 1.0); p_charges_in_fields = particles[batch_idx_3d, f_p_map_clamped, P_FEAT_CHARGE]
        scores += (((p_charges_in_fields == 0).all(dim=2) | ~is_self_conj) | ~valid_fields).all(dim=1); total_checks += 1
        scores += ((((fields[:, :, F_FEAT_GEN] == 1) | is_fermion_field)) | ~valid_fields).all(dim=1); total_checks += 1
        field_dims = fields[:, :, F_FEAT_DIM]; su2_reps = fields[:, :, F_FEAT_SU2_REP]; su3_reps = fields[:, :, F_FEAT_SU3_REP]
        is_singlet = (field_dims == 1); su2_consistent = (field_dims == su2_reps) & (su3_reps == 1); su3_consistent = (field_dims == su3_reps) & (su2_reps == 1)
        scores += ((is_singlet | su2_consistent | su3_consistent) | ~valid_fields).all(dim=1); total_checks += 1
    if level == 'all':
        fields = tensors['fields']; interactions = tensors['interactions']; B, MAX_I, _ = interactions.shape; batch_idx = torch.arange(B, device=DEVICE).view(B, 1)
        valid_fields = fields[:, :, F_FEAT_ID] != PAD_VALUE
        valid_yukawa = (interactions[:, :, I_FEAT_TYPE] == 0) & (interactions[:, :, I_FEAT_F0] != PAD_VALUE) & (interactions[:, :, I_FEAT_F1] != PAD_VALUE) & (interactions[:, :, I_FEAT_F2] != PAD_VALUE)
        scores += valid_yukawa.all(dim=1); total_checks += 1
        f0_ids, f1_ids, f2_ids = interactions[:,:,I_FEAT_F0].clamp(min=0), interactions[:,:,I_FEAT_F1].clamp(min=0), interactions[:,:,I_FEAT_F2].clamp(min=0)
        f0_props, f1_props, f2_props = fields[batch_idx, f0_ids], fields[batch_idx, f1_ids], fields[batch_idx, f2_ids]
        def get_role_masks(props):
            is_fermion = props[:, :, F_FEAT_TYPE] == FIELD_TYPE_MAP['fermion']
            is_scalar = (props[:, :, F_FEAT_TYPE] == FIELD_TYPE_MAP['real']) | (props[:, :, F_FEAT_TYPE] == FIELD_TYPE_MAP['complex']) | (props[:, :, F_FEAT_TYPE] == FIELD_TYPE_MAP['scalar'])
            is_L = is_fermion & (props[:, :, F_FEAT_CHIRALITY] == CHIRALITY_MAP['left']); is_R = is_fermion & (props[:, :, F_FEAT_CHIRALITY] == CHIRALITY_MAP['right'])
            return is_L, is_R, is_scalar
        f0_is_L, f0_is_R, f0_is_S = get_role_masks(f0_props); f1_is_L, f1_is_R, f1_is_S = get_role_masks(f1_props); f2_is_L, f2_is_R, f2_is_S = get_role_masks(f2_props)
        num_L = f0_is_L.int() + f1_is_L.int() + f2_is_L.int(); num_R = f0_is_R.int() + f1_is_R.int() + f2_is_R.int(); num_S = f0_is_S.int() + f1_is_S.int() + f2_is_S.int()
        type_check_ok = (num_L == 1) & (num_R == 1) & (num_S == 1)
        scores += (type_check_ok | ~valid_yukawa).all(dim=1); total_checks += 1
        L_gen = f0_props[:, :, F_FEAT_GEN] * f0_is_L + f1_props[:, :, F_FEAT_GEN] * f1_is_L + f2_props[:, :, F_FEAT_GEN] * f2_is_L
        R_gen = f0_props[:, :, F_FEAT_GEN] * f0_is_R + f1_props[:, :, F_FEAT_GEN] * f1_is_R + f2_props[:, :, F_FEAT_GEN] * f2_is_R
        scores += (((L_gen == R_gen) | ~type_check_ok) | ~valid_yukawa).all(dim=1); total_checks += 1
        L_dim = f0_props[:, :, F_FEAT_DIM] * f0_is_L + f1_props[:, :, F_FEAT_DIM] * f1_is_L + f2_props[:, :, F_FEAT_DIM] * f2_is_L
        R_dim = f0_props[:, :, F_FEAT_DIM] * f0_is_R + f1_props[:, :, F_FEAT_DIM] * f1_is_R + f2_props[:, :, F_FEAT_DIM] * f2_is_R
        S_dim = f0_props[:, :, F_FEAT_DIM] * f0_is_S + f1_props[:, :, F_FEAT_DIM] * f1_is_S + f2_props[:, :, F_FEAT_DIM] * f2_is_S
        scores += ((((S_dim == L_dim) | (S_dim == R_dim)) | ~type_check_ok) | ~valid_yukawa).all(dim=1); total_checks += 1
        p_masses_in_fields = particles[torch.arange(B).view(B,1,1), field_particle_map.clamp(min=0), P_FEAT_MASS]
        field_is_massive = (p_masses_in_fields > 0).any(dim=2)
        massive_fermion_field = (fields[:, :, F_FEAT_TYPE] == FIELD_TYPE_MAP['fermion']) & field_is_massive & valid_fields
        field_in_yukawa_mask = torch.zeros_like(fields[:, :, F_FEAT_ID], dtype=torch.bool)
        if valid_yukawa.any():
            batch_yukawa_idx, _ = valid_yukawa.nonzero(as_tuple=True); valid_interaction_tensors = interactions[valid_yukawa]
            yukawa_f0_indices = valid_interaction_tensors[:, I_FEAT_F0]; yukawa_f1_indices = valid_interaction_tensors[:, I_FEAT_F1]
            field_in_yukawa_mask[batch_yukawa_idx, yukawa_f0_indices] = True; field_in_yukawa_mask[batch_yukawa_idx, yukawa_f1_indices] = True
        scores += (~massive_fermion_field | field_in_yukawa_mask).all(dim=1); total_checks += 1
        is_fermion_field = (fields[:, :, F_FEAT_TYPE] == FIELD_TYPE_MAP['fermion']) & valid_fields
        u1_charges = fields[:, :, F_FEAT_U1_CHARGE]; dims = fields[:, :, F_FEAT_DIM]; gens = fields[:, :, F_FEAT_GEN]
        total_u1_charge = (u1_charges * dims * gens * is_fermion_field.float()).sum(dim=1)
        scores += (torch.abs(total_u1_charge) < 1e-6); total_checks += 1
    return scores, total_checks

# ======================================================================================
# Part 2: grammar_generator.py Logic
# ======================================================================================
def generate_id_tokens(prefix: str, count: int) -> List[str]: return [f"{prefix}_{i}" for i in range(count)]
N_RANGE = 11; IDS = range(1, N_RANGE); DIMS = range(1, N_RANGE); GENS = range(1, N_RANGE)
ITRACT_IDS, FIELD_IDS, PARTICLE_IDS = (generate_id_tokens(p, N_RANGE) for p in ["ITRACT_ID", "FIELD_ID", "PARTICLE_ID"])
DIM_TOKENS = [f"DIM_{i}" for i in DIMS]; GEN_TOKENS = [f"GEN_{i}" for i in GENS]
SU3_REP_TOKENS = ["SU3C_REP_1", "SU3C_REP_3", "SU3C_REP_8"]; SU2_REP_TOKENS = ["SU2L_REP_1", "SU2L_REP_2", "SU2L_REP_3"]
U1Y_CHARGE_TOKENS = [f"U1Y_CHARGE_{i}" for i in [-2, -1, 0, 1, 2]]; QN_L_TOKENS = ["QN_L_0", "QN_L_1"]; QN_B_TOKENS = ["QN_B_0", "QN_B_1"]
REP_TOKENS = SU3_REP_TOKENS + SU2_REP_TOKENS + U1Y_CHARGE_TOKENS + QN_L_TOKENS + QN_B_TOKENS
ALL_TOKEN_NAMES: List[str] = (["BOS", "ITRACT"] + ITRACT_IDS + ["TYPE_YUKAWA", "FIELD"] + FIELD_IDS + ["TYPE_complex", "TYPE_real", "TYPE_fermion"] + DIM_TOKENS + GEN_TOKENS + ["SELF_CONJ_TRUE", "SELF_CONJ_FALSE", "CHIRALITY_left", "CHIRALITY_right", "CHIRALITY_none"] + REP_TOKENS + ["PARTICLE"] + PARTICLE_IDS + ["MASS_1e2", "CHARGE_0", "CHARGE_1", "END_PARTICLE", "END_FIELD", "END_ITRACT", "EOS", "DEAD"])
TOKENS_MODEL = {"BOS": ["ITRACT"], "END_ITRACT": ["ITRACT", "EOS"], "EOS": ["EOS"], "DEAD": ["DEAD"]}
TOKENS_INTERACTION = {"ITRACT": ITRACT_IDS, "TYPE_YUKAWA": ["FIELD"], "END_FIELD": ["FIELD", "END_ITRACT"]}
for it_id in ITRACT_IDS: TOKENS_INTERACTION[it_id] = ["TYPE_YUKAWA"]
# --- GRAMMAR FIX: Create type-dependent branches ---
TOKENS_FIELD = {"FIELD": FIELD_IDS, "END_PARTICLE": ["PARTICLE", "END_FIELD"]}
for f_id in FIELD_IDS: TOKENS_FIELD[f_id] = ["TYPE_fermion", "TYPE_real", "TYPE_complex"]
# Fermion Branch
TOKENS_FIELD["TYPE_fermion"] = DIM_TOKENS
# Scalar Branch (for real and complex)
TOKENS_FIELD["TYPE_real"] = DIM_TOKENS
TOKENS_FIELD["TYPE_complex"] = DIM_TOKENS
for dim_tok in DIM_TOKENS: TOKENS_FIELD[dim_tok] = GEN_TOKENS
for gen_tok in GEN_TOKENS: TOKENS_FIELD[gen_tok] = ["SELF_CONJ_TRUE", "SELF_CONJ_FALSE"]
# After SELF_CONJ, fermion path requires CHIRALITY, scalar path does not
TOKENS_FIELD["SELF_CONJ_TRUE"] = ["CHIRALITY_left", "CHIRALITY_right", "CHIRALITY_none"]
TOKENS_FIELD["SELF_CONJ_FALSE"] = ["CHIRALITY_left", "CHIRALITY_right", "CHIRALITY_none"]
# Path forward from Chirality / No Chirality
REP_SEQUENCE_START = SU3_REP_TOKENS
TOKENS_FIELD["CHIRALITY_left"] = REP_SEQUENCE_START
TOKENS_FIELD["CHIRALITY_right"] = REP_SEQUENCE_START
TOKENS_FIELD["CHIRALITY_none"] = REP_SEQUENCE_START
# Common path for Reps and QNs
for su3_tok in SU3_REP_TOKENS: TOKENS_FIELD[su3_tok] = SU2_REP_TOKENS
for su2_tok in SU2_REP_TOKENS: TOKENS_FIELD[su2_tok] = U1Y_CHARGE_TOKENS
for u1y_tok in U1Y_CHARGE_TOKENS: TOKENS_FIELD[u1y_tok] = QN_L_TOKENS
for qnl_tok in QN_L_TOKENS: TOKENS_FIELD[qnl_tok] = QN_B_TOKENS
for qnb_tok in QN_B_TOKENS: TOKENS_FIELD[qnb_tok] = ["PARTICLE"]
TOKENS_PARTICLE = {"PARTICLE": PARTICLE_IDS, "MASS_1e2": ["CHARGE_0", "CHARGE_1"], "CHARGE_0": ["END_PARTICLE"], "CHARGE_1": ["END_PARTICLE"]}
for p_id in PARTICLE_IDS: TOKENS_PARTICLE[p_id] = ["TYPE_fermion", "TYPE_real", "TYPE_complex"]
for p_type in ["TYPE_fermion", "TYPE_real", "TYPE_complex"]: TOKENS_PARTICLE[p_type] = ["MASS_1e2"]
EMBED_VAL_MAP: Dict[str, int] = {"ITRACT": 0b1000, "END_ITRACT": -0b1000, "FIELD": 0b0100, "END_FIELD": -0b0100, "PARTICLE": 0b0010, "END_PARTICLE": -0b0010}
INTERACTION_BIT, FIELD_BIT, PARTICLE_BIT = 0b1000, 0b0100, 0b0010

class TokenSequenceParser:
    MASS_MAP = {"MASS_1e2": 100.0}; CHARGE_MAP = {"CHARGE_0": 0, "CHARGE_1": 1}
    @staticmethod
    def _after(tok: str, prefix: str) -> str: return tok.replace(f"{prefix}_", "")
    @staticmethod
    def _int(tok: str) -> int: return int(tok.split("_")[-1])
    def build_model_from_tokens(self, tokens: List[str]) -> Dict[str, Any]:
        particles, fields, interactions = [], [], []
        i = 0
        try:
            while i < len(tokens):
                if tokens[i] == "ITRACT": it, i_after, new_fs, new_ps = self._parse_interaction(tokens, i); interactions.append(it); fields.extend(new_fs); particles.extend(new_ps); i = i_after
                elif tokens[i] == "FIELD": f, i_after, ps = self._parse_field(tokens, i); fields.append(f); particles.extend(ps); i = i_after
                elif tokens[i] == "PARTICLE": p, i_after = self._parse_particle(tokens, i); particles.append(p); i = i_after
                else: i += 1
            particles = list({p['id']: p for p in particles}.values()); fields = list({f['id']: f for f in fields}.values())
            return {"particles": particles, "fields": fields, "interactions": interactions}
        except (IndexError, ValueError, KeyError): return {}
    def _parse_interaction(self, seq, i):
        inter = {"id": f"i{self._int(seq[i+1])}", "type": self._after(seq[i+2], "TYPE").lower(), "fields": []}; i += 3; fields_in_inter, particles_in_inter = [], []
        while i < len(seq) and seq[i] != "END_ITRACT":
            if seq[i] == "FIELD": f, i_after_f, ps = self._parse_field(seq, i); fields_in_inter.append(f); particles_in_inter.extend(ps); inter["fields"].append(f["id"]); i = i_after_f
            else: i += 1
        return inter, i + 1, fields_in_inter, particles_in_inter
    def _parse_field(self, seq, i):
        # UPDATE: Handle optional chirality in parser
        field_type = self._after(seq[i+2], "TYPE")
        base_idx = i
        field = {"id": f"f{self._int(seq[base_idx+1])}","type": field_type,"dim": self._int(seq[base_idx+3]),"gen": self._int(seq[base_idx+4]),"self_conjugate": seq[base_idx+5] == "SELF_CONJ_TRUE"}
        
        current_idx = base_idx + 6
        if field_type == 'fermion':
            field["chirality"] = self._after(seq[current_idx], "CHIRALITY")
            current_idx += 1
        else: # Scalars don't have chirality
            field["chirality"] = 'none'

        field.update({"SU3_rep": self._int(seq[current_idx]), "SU2_rep": self._int(seq[current_idx+1]), "U1_charge": self._int(seq[current_idx+2]), "QN_L": self._int(seq[current_idx+3]), "QN_B": self._int(seq[current_idx+4]), "particles": []})
        current_idx += 5
        
        i = current_idx
        particles_in_field = []
        while i < len(seq) and seq[i] != "END_FIELD":
            if seq[i] == "PARTICLE": p, i_after_p = self._parse_particle(seq, i); particles_in_field.append(p); field["particles"].append(p["id"]); i = i_after_p
            else: i += 1
        return field, i + 1, particles_in_field
    def _parse_particle(self, seq, i):
        particle = {"id": f"p{self._int(seq[i+1])}","type": self._after(seq[i+2], "TYPE"),"mass": self.MASS_MAP.get(seq[i+3], -1.0),"charge": self.CHARGE_MAP.get(seq[i+4], 99)}; i += 5
        while i < len(seq) and seq[i] != "END_PARTICLE": i += 1
        return particle, i + 1

class GrammarModelTrainer:
    def __init__(self, config: PPOConfig, logger: DummyLogger):
        self.config = config; self.logger = logger; self.device = DEVICE
        self.token_to_idx: Dict[str, int] = {token: i for i, token in enumerate(ALL_TOKEN_NAMES)}; self.idx_to_token: Dict[int, str] = {i: token for token, i in self.token_to_idx.items()}
        self.vocab_size = len(ALL_TOKEN_NAMES)
        self.bos_token_idx = self.token_to_idx["BOS"]; self.eos_token_idx = self.token_to_idx["EOS"]; self.dead_token_idx = self.token_to_idx["DEAD"]
        self.end_particle_idx = self.token_to_idx["END_PARTICLE"]; self.end_field_idx = self.token_to_idx["END_FIELD"]
        self.parser = TokenSequenceParser(); self.legality_tensors = self._build_legality_tensors(); self.embed_val_tensor = self._build_embed_val_tensor()
    def _build_legality_tensors(self) -> Dict[str, torch.Tensor]:
        tensors = {}
        for token in ALL_TOKEN_NAMES:
            for ruleset in [TOKENS_MODEL, TOKENS_INTERACTION, TOKENS_FIELD, TOKENS_PARTICLE]: ruleset.setdefault(token, ["DEAD"])
        for name, rules in {"MODEL": TOKENS_MODEL, "INTERACTION": TOKENS_INTERACTION, "FIELD": TOKENS_FIELD, "PARTICLE": TOKENS_PARTICLE}.items():
            matrix = torch.zeros((self.vocab_size, self.vocab_size), dtype=torch.bool);
            for prev_tok, next_toks in rules.items():
                if prev_tok in self.token_to_idx:
                    prev_idx = self.token_to_idx[prev_tok]; next_indices = [self.token_to_idx[n] for n in next_toks if n in self.token_to_idx]
                    if next_indices: matrix[prev_idx, next_indices] = True
            tensors[name] = matrix.to(self.device)
        return tensors
    def _build_embed_val_tensor(self) -> torch.Tensor:
        embed_vals = torch.zeros(self.vocab_size, dtype=torch.long)
        for token, value in EMBED_VAL_MAP.items(): embed_vals[self.token_to_idx[token]] = value
        return embed_vals.to(self.device)
    def get_legality_masks_vectorized(self, sequences: torch.Tensor) -> torch.Tensor:
        B, T = sequences.shape; V = self.vocab_size; bos = torch.full((B, 1), self.bos_token_idx, dtype=torch.long, device=self.device)
        prev_tokens = torch.cat([bos, sequences[:, :-1]], dim=1); embed_history = self.embed_val_tensor[sequences]
        gstates_after_token = torch.cumsum(embed_history, dim=1); zero_state = torch.zeros((B, 1), dtype=gstates_after_token.dtype, device=self.device)
        gstates_before_token = torch.cat([zero_state, gstates_after_token[:, :-1]], dim=1)
        is_particle = (gstates_before_token & PARTICLE_BIT) != 0; is_field = (~is_particle) & ((gstates_before_token & FIELD_BIT) != 0)
        is_interaction = (~is_particle & ~is_field) & ((gstates_before_token & INTERACTION_BIT) != 0); is_model = (~is_particle & ~is_field & ~is_interaction)
        flat_prev_tokens = prev_tokens.flatten()
        grammar_legality = (self.legality_tensors["PARTICLE"][flat_prev_tokens] * is_particle.flatten().unsqueeze(1) + self.legality_tensors["FIELD"][flat_prev_tokens] * is_field.flatten().unsqueeze(1) + self.legality_tensors["INTERACTION"][flat_prev_tokens] * is_interaction.flatten().unsqueeze(1) + self.legality_tensors["MODEL"][flat_prev_tokens] * is_model.flatten().unsqueeze(1)).bool().view(B, T, V)
        stack_ok = (gstates_before_token.unsqueeze(2) + self.embed_val_tensor.view(1, 1, V)) >= 0
        return grammar_legality & stack_ok
    def generate_sequences(self, model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = self.config.batch_size, self.config.seq_len; sequences = torch.full((B, T), self.dead_token_idx, dtype=torch.long, device=self.device)
        sequences[:, 0] = self.bos_token_idx; all_logits = torch.zeros(B, T, self.vocab_size, device=self.device); all_values = torch.zeros(B, T, device=self.device)
        is_finished = torch.zeros(B, dtype=torch.bool, device=self.device)
        for t in range(1, T):
            if is_finished.all(): break
            logits, values, _ = model(sequences[:, :t], None); step_logits, step_values = logits[:, -1], values[:, -1]
            step_legality_mask = self.get_legality_masks_vectorized(sequences)[:, t]; no_legal_tokens = ~step_legality_mask.any(dim=1)
            step_legality_mask[no_legal_tokens, self.dead_token_idx] = True; masked_logits = step_logits.masked_fill(~step_legality_mask, -torch.inf)
            probs = torch.softmax(masked_logits, dim=-1); next_tokens = torch.multinomial(probs.nan_to_num(0.0), 1).squeeze(1)
            sequences[:, t] = torch.where(is_finished, self.dead_token_idx, next_tokens); all_logits[:, t], all_values[:, t] = step_logits, step_values
            is_finished |= (next_tokens == self.eos_token_idx) | (next_tokens == self.dead_token_idx)
        return sequences, all_logits, all_values, self.get_legality_masks_vectorized(sequences)
    def calculate_base_rewards(self, sequences: torch.Tensor) -> torch.Tensor:
        B, T = sequences.shape; is_alive = (sequences != self.dead_token_idx); rewards = torch.zeros_like(sequences, dtype=torch.float32)
        rewards += is_alive.float() * self.config.alive_r; lengths = is_alive.sum(dim=1)
        len_bonus = torch.exp(-0.5 * ((lengths - self.config.len_target) / self.config.len_sigma)**2)
        rewards[torch.arange(B), lengths.clamp(min=1) - 1] += len_bonus
        return rewards
    def calculate_intermediate_rewards(self, sequences: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        B, T = sequences.shape; particle_rewards = torch.zeros_like(sequences, dtype=torch.float32); field_rewards = torch.zeros_like(sequences, dtype=torch.float32)
        num_particles, num_fields = 0, 0
        end_particle_indices = (sequences == self.end_particle_idx).nonzero(as_tuple=False)
        if end_particle_indices.numel() > 0:
            sub_seqs, owners = [], []
            for b, t in end_particle_indices:
                for start_t in range(t, -1, -1):
                    if self.idx_to_token[sequences[b, start_t].item()].startswith("PARTICLE"): sub_seqs.append([self.idx_to_token[i.item()] for i in sequences[b, start_t:t+1]]); owners.append((b, t)); break
            if sub_seqs:
                num_particles = len(sub_seqs)
                scores, total_checks = parallel_validator_and_scorer(ingest_batch([self.parser.build_model_from_tokens(s) for s in sub_seqs]), level='particle')
                if total_checks > 0 and scores.numel() > 0:
                    for i, (b, t) in enumerate(owners): particle_rewards[b, t] += (scores[i] / total_checks) * self.config.particle_bonus_weight
        end_field_indices = (sequences == self.end_field_idx).nonzero(as_tuple=False)
        if end_field_indices.numel() > 0:
            sub_seqs, owners = [], []
            for b, t in end_field_indices:
                for start_t in range(t, -1, -1):
                    if self.idx_to_token[sequences[b, start_t].item()].startswith("FIELD"): sub_seqs.append([self.idx_to_token[i.item()] for i in sequences[b, start_t:t+1]]); owners.append((b, t)); break
            if sub_seqs:
                num_fields = len(sub_seqs)
                scores, total_checks = parallel_validator_and_scorer(ingest_batch([self.parser.build_model_from_tokens(s) for s in sub_seqs]), level='field')
                if total_checks > 0 and scores.numel() > 0:
                    for i, (b, t) in enumerate(owners): field_rewards[b, t] += (scores[i] / total_checks) * self.config.field_bonus_weight
        return particle_rewards, field_rewards, num_particles, num_fields
    def calculate_terminal_reward(self, sequences: torch.Tensor) -> Tuple[torch.Tensor, int]:
        B, T = sequences.shape; terminal_rewards = torch.zeros_like(sequences, dtype=torch.float32); num_terminal_models = 0
        last_alive_indices = (sequences != self.dead_token_idx).sum(dim=1).clamp(min=1) - 1
        last_tokens = sequences[torch.arange(B), last_alive_indices]; eos_mask = (last_tokens == self.eos_token_idx)
        if eos_mask.any():
            eos_indices = eos_mask.nonzero(as_tuple=True)[0]; num_terminal_models = len(eos_indices)
            token_seqs_to_parse = [[self.idx_to_token[i.item()] for i in sequences[idx, :(last_alive_indices[idx]+1)]] for idx in eos_indices]
            scores, total_checks = parallel_validator_and_scorer(ingest_batch([self.parser.build_model_from_tokens(s) for s in token_seqs_to_parse]), level='all')
            if total_checks > 0 and scores.numel() > 0:
                for i, master_idx in enumerate(eos_indices): terminal_rewards[master_idx, last_alive_indices[master_idx]] += (scores[i] / total_checks) * self.config.terminal_bonus_weight
        return terminal_rewards, num_terminal_models

# ======================================================================================
# Part 3: ppo_trainer.py Logic (The Main Loop)
# ======================================================================================
@torch.jit.script
def compute_gae_jit(rewards: torch.Tensor, values: torch.Tensor, is_alive: torch.Tensor, gamma: float, gae_lambda: float) -> torch.Tensor:
    B, T = rewards.shape; advantages = torch.zeros_like(rewards); last_gae = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
    is_alive_float = is_alive.float()
    for t in range(T - 2, -1, -1):
        next_alive_mask = is_alive_float[:, t + 1]; next_val = values[:, t + 1] * next_alive_mask
        delta = rewards[:, t] + gamma * next_val - values[:, t]; last_gae = delta + gamma * gae_lambda * last_gae * next_alive_mask
        advantages[:, t] = last_gae
    return advantages

def main():
    config = PPOConfig(); logger = DummyLogger()
    logger.info(f"Starting blueprint run '{config.run_name}' on device {DEVICE}")
    trainer = GrammarModelTrainer(config, logger)
    model = DummyTransformerPolicy(trainer.vocab_size, d_model=64, max_T=config.seq_len).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=(DEVICE.type == 'cuda'))
    for step in range(1, config.total_steps + 1):
        step_start_time = time.time()
        model.eval()
        with torch.no_grad():
            sequences, rollout_logits, rollout_values, legality_masks = trainer.generate_sequences(model)
        with torch.no_grad():
            is_alive = (sequences != trainer.dead_token_idx); base_rewards = trainer.calculate_base_rewards(sequences)
            particle_rewards, field_rewards, num_particles, num_fields = trainer.calculate_intermediate_rewards(sequences)
            terminal_rewards, num_interactions = trainer.calculate_terminal_reward(sequences)
            total_rewards = base_rewards + particle_rewards + field_rewards + terminal_rewards
            advantages = compute_gae_jit(total_rewards, rollout_values, is_alive, config.gamma, config.gae_lambda); returns = advantages + rollout_values
        model.train()
        rollout_log_probs = F.log_softmax(rollout_logits, dim=-1).gather(2, sequences.unsqueeze(-1)).squeeze(-1)
        idxs = np.arange(config.batch_size)
        for ppo_epoch in range(config.ppo_epochs):
            np.random.shuffle(idxs)
            for mb_start in range(0, config.batch_size, config.minibatch_size):
                mb_end = mb_start + config.minibatch_size; mb_idx = idxs[mb_start:mb_end]
                mb_seq, mb_adv, mb_ret, mb_old_logp = sequences[mb_idx], advantages[mb_idx], returns[mb_idx], rollout_log_probs[mb_idx]
                mb_masks, mb_is_alive = legality_masks[mb_idx], is_alive[mb_idx]
                with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type=='cuda')):
                    new_logits, new_values, _ = model(mb_seq)
                    masked_new_logp_dist = F.log_softmax(new_logits.masked_fill(~mb_masks, -torch.inf), dim=-1)
                    new_logp = masked_new_logp_dist.gather(2, mb_seq.unsqueeze(-1)).squeeze(-1); ratio = (new_logp - mb_old_logp).exp()
                    pg1, pg2 = -mb_adv * ratio, -mb_adv * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                    pg_loss = torch.max(pg1, pg2)[mb_is_alive].mean(); v_loss = F.mse_loss(new_values[mb_is_alive], mb_ret[mb_is_alive])
                    ent_loss = -Categorical(logits=new_logits.masked_fill(~mb_masks, -torch.inf)).entropy()[mb_is_alive].mean()
                    sup_loss = F.softmax(new_logits, dim=-1).masked_fill(mb_masks, 0.0).sum(dim=-1)[mb_is_alive].mean()
                    total_loss = (pg_loss + config.vf_coef * v_loss + config.ent_coef * ent_loss + config.sup_coef * sup_loss)
                optimizer.zero_grad(); scaler.scale(total_loss).backward(); scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer); scaler.update()
        avg_reward = total_rewards[is_alive].mean().item()
        avg_particle_bonus = (particle_rewards.sum().item() / num_particles) if num_particles > 0 else 0.0
        avg_field_bonus = (field_rewards.sum().item() / num_fields) if num_fields > 0 else 0.0
        avg_interaction_bonus = (terminal_rewards.sum().item() / num_interactions) if num_interactions > 0 else 0.0
        loss_log = (f"PG Loss: {pg_loss.item():.3f} | V Loss: {v_loss.item():.3f} | E Loss: {ent_loss.item():.3f} | Sup Loss: {sup_loss.item():.3f}")
        reward_log = (f"Particle Bonus: {avg_particle_bonus:.3f} ({num_particles}) | Field Bonus: {avg_field_bonus:.3f} ({num_fields}) | Interaction Bonus: {avg_interaction_bonus:.3f} ({num_interactions})")
        logger.info(f"Step {step}/{config.total_steps} | Time: {time.time()-step_start_time:.2f}s | Avg Reward/Seq: {avg_reward:.4f}")
        logger.info(f"  └─ Rewards (Per-Object Avg & Count): {reward_log}")
        logger.info(f"  └─ Losses: {loss_log}")

    logger.info("Blueprint training loop finished successfully.")

if __name__ == "__main__":
    main()