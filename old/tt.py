"""
PARTICLE PHYSICS MODEL GENERATION WITH REINFORCEMENT LEARNING
============================================================

This file implements a reinforcement learning system for generating particle physics models
using a grammar-constrained transformer. The code is organized into the following major sections:

1. IMPORTS AND CONFIGURATION
   - Standard library imports (json, time, warnings, logging, etc.)
   - Scientific computing imports (numpy, torch, sympy)
   - Configuration constants for model validation and scoring

2. SCORER LOGIC
   - ModelSerializer: Converts JSON model representations to tensor format
   - BatchValidator: Validates batches of models using GPU acceleration
   - GPUValidator: Performs validation checks on gauge groups, fields, and interactions
   - Scoring system that evaluates physical consistency of generated models

3. GRAMMAR DEFINITIONS
   - Token generation for particle physics vocabulary
   - Grammar rules for model, interaction, field, and particle structures
   - Embed value mapping for state tracking during generation
   - GrammarModelTrainer: Manages grammar-constrained sequence generation

4. CORE CLASSES AND DATA STRUCTURES
   - TrajectoryData: Stores training trajectory information
   - SILBuffer: Self-Imitation Learning buffer for storing high-reward sequences
   - RunningStat: Exponential moving average statistics tracker
   - Logger: Experiment logging with TensorBoard integration
   - TorchGrammar: Immutable grammar container
   - TorchGrammarBatchParser: Incremental batch parser for grammar validation

5. MODEL TESTING
   - ModelTester: Deserializes token streams and scores resulting models
   - Parsing functions for particles, fields, and interactions
   - Model building from token sequences

6. PPO REINFORCEMENT LEARNING ALGORITHM
   - TransformerPolicy: Neural network architecture with KV caching
   - HParams: Hyperparameter configuration
   - ExperienceBuffer: Manages training data with GAE advantage calculation
   - train_reinforce(): Main training loop with PPO updates
   - sample(): Generation function for creating new models

7. UTILITY FUNCTIONS
   - Helper functions for directory creation, gradient clipping, etc.
   - Warmup cosine learning rate scheduling
   - AdamW optimizer with fused operations

The system generates particle physics models by:
1. Using a transformer to predict next tokens in a grammar-constrained sequence
2. Validating generated models against physical consistency rules
3. Using PPO reinforcement learning to improve generation quality
4. Maintaining a buffer of high-quality examples for self-imitation learning
"""

import json
import time
import warnings
import logging
import csv
import math
import sys
import os
import random
import queue
import threading
import re
from pathlib import Path
from dataclasses import dataclass, replace
from collections import deque
from typing import Dict, Any, List, Tuple, Union, Iterable, Optional, Callable, NamedTuple
from datetime import datetime

import numpy as np
import torch
from sympy import Matrix, sympify
from fractions import Fraction
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.distributions import Categorical
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import trange

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from torch.profiler import profile, record_function, ProfilerActivity

try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# SCORER LOGIC
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "MAX_GROUPS": 5, "MAX_PARTICLES": 20, "MAX_FIELDS": 10, "MAX_PARTICLES_PER_FIELD": 10,
    "MAX_INTERACTIONS": 5, "PADDING_VALUE": -1, "GROUP_TYPE_MAP": {"U": 1, "SU": 2},
    "FIELD_TYPE_MAP": {"fermion": 1, "real": 2, "complex": 3, "vector": 4},
    "REP_MAP": {"singlet": 1, "fnd": 2, "adj": 3}, "CHIRALITY_MAP": {"left": 1, "right": 2, "none": 0},
    "GROUP_PROPS": {"TYPE": 0, "DIM": 1, "IS_COLOR": 2},
    "PARTICLE_PROPS": {"TYPE": 0, "CHARGE": 1, "MASS": 2},
    "FIELD_PROPS": {"TYPE": 0, "DIM": 1, "GEN": 2, "CHIRALITY": 3},
    "INTERACTION_PROPS": {"FIELD_1": 0, "FIELD_2": 1, "FIELD_3": 2, "HIGGS_LOC_IDX": 3},
    "REP_DIM_LUT": np.array([
        [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]],
        [[0,0,0,0], [0,1,1,1], [0,0,0,0], [0,0,0,0]],
        [[0,0,0,0], [0,0,0,0], [0,1,2,3], [0,1,3,8]],
    ], dtype=np.int32),
    "SCORE_NORMALIZATION": {
        "PER_GAUGE_GROUP": 3, "PER_INTERACTION": 4, "PER_SCALAR_FIELD": 7, "PER_FERMION_FIELD": 10,
    }
}

class ModelSerializer:
    def __init__(self, model_dict: Dict[str, Any]):
        self.model = model_dict
        self.padding = CONFIG['PADDING_VALUE']
        self.group_id_to_idx = {g['id']: i for i, g in enumerate(self.model.get('GaugeGroups', []))}
        self.particle_id_to_idx = {p['id']: i for i, p in enumerate(self.model.get('particles', []))}
        self.field_id_to_idx = {f['id']: i for i, f in enumerate(self.model.get('fields', []))}
        self.field_id_to_obj = {f['id']: f for f in self.model.get('fields', [])}
        self.serialized_data = {
            "groups": np.full((CONFIG['MAX_GROUPS'], 3), self.padding, dtype=np.int32),
            "particles": np.full((CONFIG['MAX_PARTICLES'], 3), self.padding, dtype=np.float32),
            "fields": np.full((CONFIG['MAX_FIELDS'], 4), self.padding, dtype=np.int32),
            "field_reps": np.full((CONFIG['MAX_FIELDS'], CONFIG['MAX_GROUPS']), self.padding, dtype=np.int32),
            "field_to_particles": np.full((CONFIG['MAX_FIELDS'], CONFIG['MAX_PARTICLES_PER_FIELD']), self.padding, dtype=np.int32),
            "interactions": np.full((CONFIG['MAX_INTERACTIONS'], 4), self.padding, dtype=np.int32),
        }
    def serialize(self) -> Dict[str, np.ndarray]:
        self._serialize_groups(); self._serialize_particles(); self._serialize_fields(); self._serialize_interactions()
        return self.serialized_data

    def _validate_and_derive_yukawa(self, interaction: Dict[str, Any]) -> int:
        try:
            field_ids, f_left, f_right, scalar = interaction['fields'], self.field_id_to_obj[interaction['fields'][0]], self.field_id_to_obj[interaction['fields'][1]], self.field_id_to_obj[interaction['fields'][2]]
            if f_left.get('gen',1) != f_right.get('gen',1):
                print(f"(Note: CPU-side gen mismatch in {interaction['id']}) ", end="")
                return self.padding
            left_p = np.array(f_left['particles']).reshape(f_left.get('gen',1), f_left['dim']).transpose()
            right_p = np.array(f_right['particles']).reshape(f_right.get('gen',1), f_right['dim'])
            bilinear = Matrix(left_p) * Matrix(right_p)
            loc = -1
            for i in range(scalar['dim']):
                if str(sympify(str(bilinear[i]).replace(" ", ""))).count("**2") == f_left.get('gen',1): loc = i; break
            return loc if loc != -1 else self.padding
        except Exception:
            return self.padding

    def _serialize_interactions(self):
        for i, itr in enumerate(self.model.get('interactions', [])):
            self.serialized_data['interactions'][i, 0], self.serialized_data['interactions'][i, 1], self.serialized_data['interactions'][i, 2] = [self.field_id_to_idx.get(fid, self.padding) for fid in itr['fields']]
            self.serialized_data['interactions'][i, 3] = self._validate_and_derive_yukawa(itr)

    def _serialize_groups(self):
        for i, g in enumerate(self.model.get('GaugeGroups', [])):
            g_type, g_dim = g['group'].split('_'); self.serialized_data['groups'][i, 0], self.serialized_data['groups'][i, 1], self.serialized_data['groups'][i, 2] = CONFIG['GROUP_TYPE_MAP'].get(g_type, 0), int(g_dim), 1 if g.get('name') == 'SU3C' else 0
    def _serialize_particles(self):
        for i, p in enumerate(self.model.get('particles', [])): self.serialized_data['particles'][i, 0], self.serialized_data['particles'][i, 1], self.serialized_data['particles'][i, 2] = CONFIG['FIELD_TYPE_MAP'].get(p['type'], 0), p['charge'], p['mass']

    def _serialize_fields(self):
        for i, f in enumerate(self.model.get('fields', [])):
            chirality_val = CONFIG['CHIRALITY_MAP'].get(f.get('chirality'), 0)

            self.serialized_data['fields'][i, 0], self.serialized_data['fields'][i, 1], self.serialized_data['fields'][i, 2], self.serialized_data['fields'][i, 3] = \
                CONFIG['FIELD_TYPE_MAP'].get(f['type'], 0), f['dim'], f.get('gen', 1), chirality_val

            for gid, rname in f['reps'].items():
                gidx = self.group_id_to_idx.get(gid)
                if gidx is not None: self.serialized_data['field_reps'][i, gidx] = CONFIG['REP_MAP']['singlet'] if isinstance(rname, (float,int)) else CONFIG['REP_MAP'].get(rname, 0)
            for j, pid in enumerate(f.get('particles', [])):
                if j < CONFIG['MAX_PARTICLES_PER_FIELD']: self.serialized_data['field_to_particles'][i, j] = self.particle_id_to_idx.get(pid, self.padding)

class BatchValidator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"; print(f"BatchValidator initialized. Target device: {self.device.upper()}")
    def validate_batch(self, batch_of_models: List[Dict[str, Any]]) -> np.ndarray:
        print(f"\nProcessing batch of {len(batch_of_models)} models..."); serialized_models = []
        for i, m in enumerate(batch_of_models):
            try:
                print(f"  - Serializing model {i+1}... ", end="")
                s = ModelSerializer(m); d = s.serialize(); serialized_models.append(d); print("OK.")
            except Exception as e:
                # This will now only catch critical programming/key errors, not validation failures
                print(f"FATAL SERIALIZATION ERROR: {e}")
                return np.full(len(batch_of_models), -2, dtype=np.int32) # Return a different error code

        print(f"\nSerialization complete for all models. Collating for GPU...");
        batch_data = {k: np.stack([d[k] for d in serialized_models]) for k in serialized_models[0].keys()}

        validator = GPUValidator(batch_data); gpu_scores = validator.validate().cpu().numpy()
        return gpu_scores

class GPUValidator:
    def __init__(self, batch_data: Dict[str, np.ndarray]):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tensors = {name: torch.from_numpy(arr).to(self.device) for name, arr in batch_data.items()}
        self.rep_dim_lut = torch.from_numpy(CONFIG['REP_DIM_LUT']).to(self.device)
        self.batch_size = self.tensors['groups'].shape[0]
        self.padding = CONFIG['PADDING_VALUE']
        self.scores = CONFIG['SCORE_NORMALIZATION']

    def validate(self) -> torch.Tensor:
        group_scores = self.check_gauge_groups()
        field_scores = self.check_fields()
        interaction_scores = self.check_interactions()
        total_score = torch.sum(group_scores, dim=-1) + torch.sum(field_scores, dim=-1) + torch.sum(interaction_scores, dim=-1)
        return total_score

    def check_gauge_groups(self) -> torch.Tensor:
        g, mask = self.tensors['groups'], (self.tensors['groups'][:, :, 0] != self.padding)
        t, d = g[:, :, CONFIG['GROUP_PROPS']['TYPE']], g[:, :, CONFIG['GROUP_PROPS']['DIM']]
        is_valid = ( (t == 1) | (t == 2) ) & ( (t != 1) | (d == 1) ) & ( (t != 2) | ((d == 2) | (d == 3)) )
        return (is_valid & mask).int() * self.scores['PER_GAUGE_GROUP']

    def check_fields(self) -> torch.Tensor:
        f, p, fp, fr, g = self.tensors['fields'], self.tensors['particles'], self.tensors['field_to_particles'], self.tensors['field_reps'], self.tensors['groups']
        mask = (f[:, :, 0] != self.padding)
        field_type = f[:, :, CONFIG['FIELD_PROPS']['TYPE']]
        type_check = field_type > 0
        count_check = torch.sum(fp != self.padding, dim=-1) == f[:, :, CONFIG['FIELD_PROPS']['DIM']] * f[:, :, CONFIG['FIELD_PROPS']['GEN']]
        b_idx, p_idx = torch.arange(self.batch_size, device=self.device).view(-1, 1, 1), torch.clamp(fp, min=0)
        p_types = p[b_idx, p_idx, CONFIG['PARTICLE_PROPS']['TYPE']]
        match = (p_types == field_type.unsqueeze(2).expand_as(fp)); match[fp == self.padding] = True
        particle_type_check = torch.all(match, dim=-1)
        is_fermion_mask = (field_type == CONFIG['FIELD_TYPE_MAP']['fermion'])
        chi = f[:, :, CONFIG['FIELD_PROPS']['CHIRALITY']]
        chirality_check = ((chi == 1) | (chi == 2)) | ~is_fermion_mask
        gt, gd = g[:, :, CONFIG['GROUP_PROPS']['TYPE']].unsqueeze(1), g[:, :, CONFIG['GROUP_PROPS']['DIM']].unsqueeze(1)
        rep_dims = self.rep_dim_lut[gt, gd, fr]; is_na_ns = (gt == 2) & (fr != 1); relevant_rep_dims = rep_dims * is_na_ns
        field_dim_exp = f[:, :, CONFIG['FIELD_PROPS']['DIM']].unsqueeze(-1)
        is_singlet_everywhere = torch.sum(relevant_rep_dims, dim=-1) == 0
        is_dim_one = f[:, :, CONFIG['FIELD_PROPS']['DIM']] == 1
        dim_matches_any_rep = torch.any((field_dim_exp == relevant_rep_dims) & (relevant_rep_dims != 0), dim=-1)
        dim_consistency_check = is_singlet_everywhere | is_dim_one | dim_matches_any_rep
        is_valid = type_check & count_check & particle_type_check & chirality_check & dim_consistency_check
        is_valid = is_valid & mask
        base_score = is_valid.int() * self.scores['PER_SCALAR_FIELD']
        fermion_bonus = is_valid.int() * (self.scores['PER_FERMION_FIELD'] - self.scores['PER_SCALAR_FIELD'])
        return base_score + (fermion_bonus * is_fermion_mask)

    def check_interactions(self) -> torch.Tensor:
        i, f = self.tensors['interactions'], self.tensors['fields']
        mask = (i[:, :, 0] != self.padding)
        if not torch.any(mask): return torch.zeros_like(i[:,:,0], dtype=torch.int32)
        b = torch.arange(self.batch_size, device=self.device).view(-1, 1)
        i1, i2, i3 = torch.clamp(i[:, :, 0], min=0), torch.clamp(i[:, :, 1], min=0), torch.clamp(i[:, :, 2], min=0)
        p1, p2, p3 = f[b, i1], f[b, i2], f[b, i3]
        type_check_ok = (p1[..., 0] == 1) & (p1[..., 3] == 1) & (p2[..., 0] == 1) & (p2[..., 3] == 2) & (p3[..., 0] == 3)
        gen_check_ok = (p1[..., 2] == p2[..., 2])
        symbolic_check_ok = (i[..., 3] != self.padding)
        score = (type_check_ok.int() * 1) + (gen_check_ok.int() * 1) + (symbolic_check_ok.int() * 2)
        return score * mask.int()

# ─────────────────────────────────────────────────────────────────────────────
# GRAMMAR DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

# Helper to create token strings with numeric suffixes
def generate_id_tokens(prefix: str, count: int) -> List[str]:
    return [f"{prefix}_{i}" for i in range(count)]  # Returns a list like ["PREFIX_0", "PREFIX_1", ...]

NUM_IDS = 10  # Number of unique IDs to create for each category
ITRACT_IDS = generate_id_tokens("ITRACT_ID", NUM_IDS)  # Interaction identifiers
FIELD_IDS = generate_id_tokens("FIELD_ID", NUM_IDS)    # Field identifiers
PARTICLE_IDS = generate_id_tokens("PARTICLE_ID", NUM_IDS) # Particle identifiers

# Base token list without the IDs that are now generated programmatically
BASE_TOKEN_NAMES: List[str] = [  # Pre-defined tokens that describe the grammar structure
    "BOS", "ITRACT", "TYPE_DC", "TYPE_YUKAWA", "TYPE_VLF", "TYPE_PHI4",
    "TYPE_FF", "TYPE_FFDUAL", "FIELD", "TYPE_complex", "TYPE_real",
    "TYPE_fermion", "TYPE_vector", "DIM_1", "DIM_2", "DIM_3", "GEN_1",
    "GEN_2", "GEN_3", "SELF_CONJ_TRUE", "SELF_CONJ_FALSE", "CHIRALITY_left",
    "CHIRALITY_right", "CHIRALITY_na", "REP_SU3C_singlet", "REP_SU3C_fnd",
    "REP_SU3C_adj", "REP_SU2L_singlet", "REP_SU2L_fnd", "REP_SU2L_adj",
    "REP_U1Y_minus1", "REP_U1Y_0", "REP_U1Y_1", "QN_LeptonNumber_-1",
    "QN_LeptonNumber_0", "QN_LeptonNumber_1", "QN_BaryonNumber_-1",
    "QN_BaryonNumber_0", "QN_BaryonNumber_1", "PARTICLE", "MASS_1e0",
    "MASS_1e1", "MASS_1e2", "MASS_1e3", "CHARGE_-1", "CHARGE_0",
    "CHARGE_1", "END_PARTICLE", "END_FIELD", "END_ITRACT", "EOS", "DEAD"
]

# Combine base tokens with generated ID tokens
ALL_TOKEN_NAMES: List[str] = (
    ["BOS", "ITRACT"] + ITRACT_IDS +  # BOS plus interaction IDs
    [
        "TYPE_DC", "TYPE_YUKAWA", "TYPE_VLF", "TYPE_PHI4", "TYPE_FF", "TYPE_FFDUAL",
        "FIELD"
    ] + FIELD_IDS +  # Field IDs appended
    [
        "TYPE_complex", "TYPE_real", "TYPE_fermion", "TYPE_vector", "DIM_1", "DIM_2",
        "DIM_3", "GEN_1", "GEN_2", "GEN_3", "SELF_CONJ_TRUE", "SELF_CONJ_FALSE",
        "CHIRALITY_left", "CHIRALITY_right", "CHIRALITY_na", "REP_SU3C_singlet",
        "REP_SU3C_fnd", "REP_SU3C_adj", "REP_SU2L_singlet", "REP_SU2L_fnd",
        "REP_SU2L_adj", "REP_U1Y_minus1", "REP_U1Y_0", "REP_U1Y_1",
        "QN_LeptonNumber_-1", "QN_LeptonNumber_0", "QN_LeptonNumber_1",
        "QN_BaryonNumber_-1", "QN_BaryonNumber_0", "QN_BaryonNumber_1", "PARTICLE"
    ] + PARTICLE_IDS +  # Particle IDs appended
    [
        "MASS_1e0", "MASS_1e1", "MASS_1e2", "MASS_1e3", "CHARGE_-1", "CHARGE_0",
        "CHARGE_1", "END_PARTICLE", "END_FIELD", "END_ITRACT", "EOS", "DEAD"
    ]
)

# --- Model-level Grammar Rules ---
TOKENS_MODEL = {  # Defines legal successors in the global model context
    "BOS": ["ITRACT"],  # Start must be followed by an interaction
    "END_ITRACT": ["ITRACT", "EOS"],  # After an interaction you can start another or end sequence
    "EOS": ["EOS"],  # Once EOS is reached, stay there
    "DEAD": ["DEAD"]  # Invalid state loops forever
}

# --- Interaction Grammar ---
TOKENS_INTERACTION = {
    "ITRACT": ITRACT_IDS,  # Interaction token must be followed by its ID
    "TYPE_DC": ["FIELD"], "TYPE_YUKAWA": ["FIELD"], "TYPE_VLF": ["FIELD"],
    "TYPE_PHI4": ["FIELD"], "TYPE_FF": ["FIELD"], "TYPE_FFDUAL": ["FIELD"],
    "END_FIELD": ["FIELD", "END_ITRACT"],  # Either another field or close interaction
    "DEAD": ["DEAD"]  # Error state
}
# Add rules for all new ITRACT_IDs
interaction_types = ["TYPE_DC", "TYPE_YUKAWA", "TYPE_VLF", "TYPE_PHI4", "TYPE_FF", "TYPE_FFDUAL"]
for itract_id in ITRACT_IDS:
    TOKENS_INTERACTION[itract_id] = interaction_types  # Each ID can start any of these types

# --- Field Grammar ---
TOKENS_FIELD = {
    "FIELD": FIELD_IDS,  # Field token must be followed by its ID
    "TYPE_complex": ["DIM_1", "DIM_2", "DIM_3"], "TYPE_real": ["DIM_1", "DIM_2", "DIM_3"],
    "TYPE_fermion": ["DIM_1", "DIM_2", "DIM_3"], "TYPE_vector": ["DIM_1", "DIM_2", "DIM_3"],
    "DIM_1": ["GEN_1", "GEN_2", "GEN_3"], "DIM_2": ["GEN_1", "GEN_2", "GEN_3"],
    "DIM_3": ["GEN_1", "GEN_2", "GEN_3"], "GEN_1": ["SELF_CONJ_TRUE", "SELF_CONJ_FALSE"],
    "GEN_2": ["SELF_CONJ_TRUE", "SELF_CONJ_FALSE"], "GEN_3": ["SELF_CONJ_TRUE", "SELF_CONJ_FALSE"],
    "SELF_CONJ_TRUE": ["CHIRALITY_left", "CHIRALITY_right", "CHIRALITY_na"],
    "SELF_CONJ_FALSE": ["CHIRALITY_left", "CHIRALITY_right", "CHIRALITY_na"],
    "CHIRALITY_left": ["REP_SU3C_singlet", "REP_SU3C_fnd", "REP_SU3C_adj"],
    "CHIRALITY_right": ["REP_SU3C_singlet", "REP_SU3C_fnd", "REP_SU3C_adj"],
    "CHIRALITY_na": ["REP_SU3C_singlet", "REP_SU3C_fnd", "REP_SU3C_adj"],
    "REP_SU3C_singlet": ["REP_SU2L_singlet", "REP_SU2L_fnd", "REP_SU2L_adj"],
    "REP_SU3C_fnd": ["REP_SU2L_singlet", "REP_SU2L_fnd", "REP_SU2L_adj"],
    "REP_SU3C_adj": ["REP_SU2L_singlet", "REP_SU2L_fnd", "REP_SU2L_adj"],
    "REP_SU2L_singlet": ["REP_U1Y_minus1", "REP_U1Y_0", "REP_U1Y_1"],
    "REP_SU2L_fnd": ["REP_U1Y_minus1", "REP_U1Y_0", "REP_U1Y_1"],
    "REP_SU2L_adj": ["REP_U1Y_minus1", "REP_U1Y_0", "REP_U1Y_1"],
    "REP_U1Y_minus1": ["QN_LeptonNumber_-1", "QN_LeptonNumber_0", "QN_LeptonNumber_1"],
    "REP_U1Y_0": ["QN_LeptonNumber_-1", "QN_LeptonNumber_0", "QN_LeptonNumber_1"],
    "REP_U1Y_1": ["QN_LeptonNumber_-1", "QN_LeptonNumber_0", "QN_LeptonNumber_1"],
    "QN_LeptonNumber_-1": ["QN_BaryonNumber_-1", "QN_BaryonNumber_0", "QN_BaryonNumber_1"],
    "QN_LeptonNumber_0": ["QN_BaryonNumber_-1", "QN_BaryonNumber_0", "QN_BaryonNumber_1"],
    "QN_LeptonNumber_1": ["QN_BaryonNumber_-1", "QN_BaryonNumber_0", "QN_BaryonNumber_1"],
    "QN_BaryonNumber_-1": ["PARTICLE"], "QN_BaryonNumber_0": ["PARTICLE"],
    "QN_BaryonNumber_1": ["PARTICLE"], "END_PARTICLE": ["PARTICLE", "END_FIELD"], "DEAD": ["DEAD"]
}
# Add rules for all new FIELD_IDs
field_types = ["TYPE_complex", "TYPE_real", "TYPE_fermion", "TYPE_vector"]  # Valid field types
for field_id in FIELD_IDS:
    TOKENS_FIELD[field_id] = field_types  # Each field ID maps to possible type tokens

# --- Particle Grammar ---
TOKENS_PARTICLE = {
    "PARTICLE": PARTICLE_IDS,  # Particle token must be followed by its ID
    "TYPE_complex": ["MASS_1e0", "MASS_1e1", "MASS_1e2", "MASS_1e3"],
    "TYPE_real": ["MASS_1e0", "MASS_1e1", "MASS_1e2", "MASS_1e3"],
    "TYPE_fermion": ["MASS_1e0", "MASS_1e1", "MASS_1e2", "MASS_1e3"],
    "TYPE_vector": ["MASS_1e0", "MASS_1e1", "MASS_1e2", "MASS_1e3"],
    "MASS_1e0": ["CHARGE_-1", "CHARGE_0", "CHARGE_1"],
    "MASS_1e1": ["CHARGE_-1", "CHARGE_0", "CHARGE_1"],
    "MASS_1e2": ["CHARGE_-1", "CHARGE_0", "CHARGE_1"],
    "MASS_1e3": ["CHARGE_-1", "CHARGE_0", "CHARGE_1"],
    "CHARGE_-1": ["END_PARTICLE"], "CHARGE_0": ["END_PARTICLE"], "CHARGE_1": ["END_PARTICLE"],
    "DEAD": ["DEAD"]
}
# Add rules for all new PARTICLE_IDs
for particle_id in PARTICLE_IDS:
    TOKENS_PARTICLE[particle_id] = field_types  # Particle IDs mirror field type options

# --- Embed value map for state tracking ---
EMBED_VAL_MAP: Dict[str, int] = {
    "ITRACT": 0b1000, "END_ITRACT": -0b1000,
    "FIELD": 0b0100, "END_FIELD": -0b0100,
    "PARTICLE": 0b0010, "END_PARTICLE": -0b0010,
}
# Bit masks for fast property checks
INTERACTION_BIT = 0b1000  # Bit flag representing interaction context
FIELD_BIT = 0b0100        # Flag for field context
PARTICLE_BIT = 0b0010       # Flag for particle context

class GrammarModelTrainer:
    """Manages grammar-constrained sequence generation and reward calculation."""
    def __init__(self, device: torch.device):
        self.device = device  # Target device (CPU or GPU)
        self.token_to_idx: Dict[str, int] = {token: i for i, token in enumerate(ALL_TOKEN_NAMES)}  # Token→index
        self.idx_to_token: Dict[int, str] = {i: token for token, i in self.token_to_idx.items()}  # Index→token
        self.vocab_size = len(ALL_TOKEN_NAMES)  # Total vocabulary size
        # Cache common token indices for speed
        self.bos_token_idx, self.eos_token_idx, self.dead_token_idx, self.itract_token_idx = (
            self.token_to_idx[k] for k in ["BOS", "EOS", "DEAD", "ITRACT"])
        self.legality_tensors = self._build_legality_tensors()  # Precompute legality matrices
        self.embed_val_tensor = self._build_embed_val_tensor()  # Precompute embed value lookup

    def _build_legality_tensors(self) -> Dict[str, torch.Tensor]:
        """Pre-compiles grammar dictionaries into boolean matrices for O(1) legality checks."""
        tensors = {}
        # Ensure every token appears in each rule set (fallback to DEAD)
        for token in ALL_TOKEN_NAMES:
            TOKENS_MODEL.setdefault(token, ["DEAD"])
            TOKENS_INTERACTION.setdefault(token, ["DEAD"])
            TOKENS_FIELD.setdefault(token, ["DEAD"])
            TOKENS_PARTICLE.setdefault(token, ["DEAD"])
        rule_sets = {"MODEL": TOKENS_MODEL, "INTERACTION": TOKENS_INTERACTION,
                       "FIELD": TOKENS_FIELD, "PARTICLE": TOKENS_PARTICLE}
        for name, rules in rule_sets.items():
            matrix = torch.zeros((self.vocab_size, self.vocab_size), dtype=torch.bool)  # Init legality matrix
            for prev_tok, next_toks in rules.items():
                if prev_tok not in self.token_to_idx:
                    continue  # Skip tokens unknown to the global vocab
                prev_idx = self.token_to_idx[prev_tok]  # Row index
                next_indices = [self.token_to_idx[n] for n in next_toks if n in self.token_to_idx]  # Valid cols
                if next_indices:
                    matrix[prev_idx, next_indices] = True  # Mark legal transitions
            tensors[name] = matrix.to(self.device)  # Move to target device
        return tensors  # Return dict of legality tensors

    def _build_embed_val_tensor(self) -> torch.Tensor:
        """Convert EMBED_VAL_MAP into a vector for fast arithmetic during generation."""
        embed_vals = torch.zeros(self.vocab_size, dtype=torch.long)  # Default 0 for tokens without bits
        for token, value in EMBED_VAL_MAP.items():
            embed_vals[self.token_to_idx[token]] = value  # Assign bit value
        return embed_vals.to(self.device)  # Move to device

    def get_active_grammar_mask(self, embed_vals: torch.Tensor, prev_tokens: torch.Tensor) -> torch.Tensor:
        """Determines which tokens are legal next steps given current embed state."""
        # Prepare empty mask [batch, vocab]
        mask = torch.zeros((embed_vals.shape[0], self.vocab_size), dtype=torch.bool, device=self.device)
        # Determine which grammar applies via bit tests
        is_particle = (embed_vals & PARTICLE_BIT) != 0  # In particle context
        is_field = (~is_particle) & ((embed_vals & FIELD_BIT) != 0)  # Field but not particle context
        is_interaction = (~is_particle & ~is_field) & ((embed_vals & INTERACTION_BIT) != 0)  # Interaction
        is_model = (~is_particle & ~is_field & ~is_interaction)  # Top-level model context
        # Apply appropriate legality tensor based on context
        mask[is_particle] = self.legality_tensors["PARTICLE"][prev_tokens[is_particle]]
        mask[is_field] = self.legality_tensors["FIELD"][prev_tokens[is_field]]
        mask[is_interaction] = self.legality_tensors["INTERACTION"][prev_tokens[is_interaction]]
        mask[is_model] = self.legality_tensors["MODEL"][prev_tokens[is_model]]
        return mask  # Boolean mask of legal next tokens

    def generate_sequences(self, model: nn.Module, batch_size: int, seq_len: int) -> torch.Tensor:
        """Autoregressively generates token sequences under the grammar constraints."""
        sequences = torch.full((batch_size, seq_len), self.dead_token_idx, dtype=torch.long, device=self.device)  # Pre-allocate with DEAD
        sequences[:, 0] = self.bos_token_idx  # First token is always BOS

        embed_vals = torch.zeros(batch_size, dtype=torch.long, device=self.device)  # Track embed bits per sample
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)  # Track finished sequences
        itract_counts = torch.zeros(batch_size, dtype=torch.long, device=self.device)  # Count interactions per sample
        cache = None  # Placeholder for KV cache (for real models)
        current_tokens = sequences[:, 0]  # Current input token for model

        # Sequential loop across sequence positions (can't be vectorized easily)
        for i in range(1, seq_len):
            if is_finished.all():
                break  # Early exit if every sequence finished

            logits, cache = model(current_tokens.unsqueeze(1), cache)  # Forward pass with cache (B,1)->(B,1,V)
            logits = logits.squeeze(1)  # Remove seq_len dim -> (B,V)

            legality_mask = self.get_active_grammar_mask(embed_vals, current_tokens)  # Mask illegal tokens
            legality_mask[itract_counts < 2, self.eos_token_idx] = False  # Disallow EOS until ≥2 interactions
            logits.masked_fill_(~legality_mask, -float('inf'))  # Suppress illegal logits

            probs = torch.softmax(logits, dim=-1).nan_to_num_(0.0)  # Safe softmax
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)  # Sample next token

            is_valid = (embed_vals + self.embed_val_tensor[next_tokens] >= 0) & (~is_finished)  # Check balance bits

            current_tokens = torch.where(is_valid, next_tokens, self.dead_token_idx)  # Replace invalid with DEAD
            sequences[:, i] = current_tokens  # Store generated token

            embed_vals += self.embed_val_tensor[current_tokens]  # Update embed bit state
            itract_counts += (current_tokens == self.itract_token_idx).long()  # Increment interaction count
            is_finished |= (current_tokens == self.dead_token_idx) | (current_tokens == self.eos_token_idx)  # Update finished mask

        return sequences  # Generated batch of sequences

    def calculate_rewards(self, sequences: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
        """Compute discounted survival rewards fully vectorized."""
        is_alive = (sequences != self.dead_token_idx).float()  # 1 where token is not DEAD
        is_alive_flipped = torch.flip(is_alive, dims=[1])  # Reverse temporal order
        gammas = torch.full_like(is_alive_flipped, gamma)  # Discount factors matrix
        discounts = torch.cumprod(torch.cat([torch.ones_like(gammas[:, :1]), gammas[:, :-1]], dim=1), dim=1)  # Cumulative discount
        rewards_flipped = torch.cumsum(torch.flip(is_alive_flipped * discounts, dims=[1]), dim=1)  # Future rewards sum
        return torch.flip(rewards_flipped, dims=[1]) / discounts * is_alive  # Normalize and zero after DEAD

    def display_results(self, sequences: torch.Tensor, rewards: torch.Tensor):
        """Pretty-print sequences along with embed state and rewards for debugging."""
        sequences_cpu = sequences.cpu().numpy()  # Move to CPU for printing
        rewards_cpu = rewards.cpu().numpy()  # Same for rewards

        for i in range(sequences_cpu.shape[0]):
            print(f"--- Sequence {i:02d} ---")  # Header per sequence
            header = "Step | Token                | Embed | Reward"
            print(header, "\n" + "-" * len(header), sep="")

            embed_val = 0  # Reset embed bits
            for j in range(sequences_cpu.shape[1]):
                token_idx = sequences_cpu[i, j]  # Index of current token
                token_str = self.idx_to_token[token_idx]  # String form
                current_embed_str = f'{embed_val:04b}'  # Binary embed bits
                reward_val = rewards_cpu[i, j]  # Reward at this step
                print(f"{j:<4d} | {token_str:<18} | {current_embed_str} | {reward_val:>6.2f}")  # Log line
                embed_val += self.embed_val_tensor[token_idx].item()  # Update embed bits for next step
                if token_idx in (self.dead_token_idx, self.eos_token_idx):
                    break  # Stop printing after termination token


# ─────────────────────────────────────────────────────────────────────────────
# CORE CLASSES AND DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO,
                      format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)
Token    = Union[int, Tuple[int, ...], str, Dict[str, Any]]
Predicate = Callable[[int], bool]

@dataclass
class TrajectoryData: # Renamed from Trajectory to avoid conflict with NamedTuple
    states: torch.Tensor # (T_i + 1,) token IDs incl. BOS
    actions: torch.Tensor # (T_i,) token IDs (step-wise)
    ret: float # scalar episode return

class SILBuffer:
    """Top-K buffer for Self-Imitation Learning (SIL)."""
    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self._data: deque[TrajectoryData] = deque(maxlen=capacity)

    def add(self, trajs: List[TrajectoryData]) -> None:
        """Insert & keep only the *capacity* highest-return trajectories."""
        self._data.extend(trajs)
        self._data = deque(sorted(self._data, key=lambda x: x.ret, reverse=True)[:self.capacity], maxlen=self.capacity)

    def sample(self, batch: int) -> List[TrajectoryData]:
        idx = torch.randint(0, len(self._data), (batch,))
        return [self._data[i] for i in idx]

class RunningStat:
    """Exponential moving average (EMA) style running mean / variance tracker."""
    def __init__(self, alpha: float = 0.1) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self._alpha = alpha
        self._count: int = 0
        self._mean: float = 0.0
        self._var: float = 0.0  # Running variance using EMA
        self._most_recent: float = 0.0

    @property
    def count(self) -> int:
        return self._count
    @property
    def mean(self) -> float:
        return self._mean
    @property
    def variance(self) -> float:
        return self._var
    @property
    def std(self) -> float:
        return math.sqrt(max(0.0, self._var))
    @property
    def most_recent(self) -> float:
        return self._most_recent

    def update(self, x: float) -> None:
        if not math.isfinite(x):
            return
        self._most_recent = x
        self._count += 1
        if self._count == 1:
            self._mean = x
            self._var = 0.0
            return
        delta = x - self._mean
        self._mean += self._alpha * delta
        self._var = (1 - self._alpha) * self._var + self._alpha * delta * delta


class Logger:
    """Generic experiment logger."""
    CLI_CSV, STATE_CSV = "cli.csv", "state.csv"
    WEIGHTS_DIR, TBOARD_DIR = "weights", "tensorboard"

    def __init__(self, path: str | os.PathLike, *, cli_interval: int = 1_000, state_interval: int = 1, weights_interval: int = 100_000):
        self.root = Path(path); self.root.mkdir(parents=True, exist_ok=True)
        self.cli_interval, self.state_interval = cli_interval, state_interval
        self.weights_interval = weights_interval
        self.weights_dir = self.root / self.WEIGHTS_DIR; self.weights_dir.mkdir(exist_ok=True)
        self.tboard_dir  = self.root / self.TBOARD_DIR ; self.tboard_dir.mkdir(exist_ok=True)
        self._cli_csv    = (self.root / self.CLI_CSV  ).open("w", newline="", buffering=1)
        self._state_csv = (self.root / self.STATE_CSV).open("w", newline="", buffering=1)
        self.cli_writer, self.state_writer = csv.writer(self._cli_csv), csv.writer(self._state_csv)
        self.tb = SummaryWriter(str(self.tboard_dir), flush_secs=1)
        self.timestep, self.seconds = 0, 0.0
        self._t0 = time.time()
        self.stats  : dict[str, RunningStat] = {}
        self.scalars : dict[str, float]       = {}
        self.add_metric("fps", kind="stat")
        self.state, self.action, self.logits, self.model = None, None, None, None
        self._cli_header_written  = False
        self._state_header_written= False

    def __del__(self):
        if hasattr(self, '_cli_csv'): self._cli_csv.close()
        if hasattr(self, '_state_csv'): self._state_csv.close()
        if hasattr(self, 'tb'): self.tb.close()

    def add_metric(self, name: str, *, kind: str = "stat") -> None:
        if kind not in {"stat", "scalar"}: raise ValueError("kind must be 'stat' or 'scalar'")
        if name in self.stats or name in self.scalars: raise ValueError(f"Metric '{name}' already registered")
        if kind == "stat": self.stats[name] = RunningStat()
        else: self.scalars[name] = 0.0

    def log(self, values: Dict[str, Any]) -> None:
        for k, v in values.items():
            if k in self.stats: self.stats[k].update(float(v))
            elif k in self.scalars: self.scalars[k] = float(v)
            elif k in {"state", "action", "logits", "weights"}: setattr(self, k if k != "weights" else "model", v)
            else: raise KeyError(f"'{k}' not registered – call add_metric() first")

    def state_step(self) -> None:
        self.timestep += 1
        self.seconds = time.time() - self._t0
        self.stats["fps"].update(self.timestep / max(self.seconds, 1e-9))
        if not self._state_header_written: self._write_state_header(); self._state_header_written = True
        if self.timestep % self.state_interval == 0:
            self.state_writer.writerow([self.timestep, self.seconds, self.state, self.stats.get("reward", RunningStat()).most_recent, self.action, self.logits])
            self.tb.add_scalar("state/reward", self.stats.get("reward", RunningStat()).most_recent, self.timestep)
            self.tb.add_scalar("state/fps",    self.stats["fps"].mean, self.timestep)
        if self.model is not None and self.timestep % self.weights_interval == 0:
            torch.save(self.model.state_dict(), self.weights_dir / f"weights_{self.timestep}.pt")

    def cli_step(self) -> None:
        if not self._cli_header_written: self._write_cli_header(); self._cli_header_written = True
        row = [self.timestep, self.seconds, self.stats["fps"].mean, self.stats["fps"].std]
        for name in sorted(self.scalars): row.append(self.scalars[name])
        for name in sorted(self.stats):
            if name == "fps": continue
            st = self.stats[name]
            row.extend([st.most_recent, st.mean, st.std])
        self.cli_writer.writerow(row)
        self._cli_csv.flush()
        for name, st in self.stats.items():
            self.tb.add_scalar(f"train/{name}_mean", st.mean, self.timestep)
            self.tb.add_scalar(f"train/{name}_std",  st.std,  self.timestep)
            self.tb.add_scalar(f"train/{name}_mr",   st.most_recent, self.timestep)
        for name, val in self.scalars.items(): self.tb.add_scalar(f"train/{name}", val, self.timestep)
        self.tb.flush()

    def _write_state_header(self) -> None: self.state_writer.writerow(["timestep", "seconds", "state", "reward_most_recent", "action", "logits"])
    def _write_cli_header(self) -> None:
        hdr = ["timestep", "seconds", "fps_mean", "fps_std"]
        hdr.extend(sorted(self.scalars))
        for name in sorted(self.stats):
            if name == "fps": continue
            hdr.extend([f"{name}_most_recent", f"{name}_mean", f"{name}_std"])
        self.cli_writer.writerow(hdr)


class TorchGrammar:
    """*Immutable* container mapping *symbol → rule-list*."""
    def __init__(self) -> None:
        self.raw_rules: Dict[str, List[Token]] = {}
        self.symbol_to_index: Dict[str, int] = {}
        self.index_to_symbol: Dict[int, str] = {}

    def add_object(self, name: str, tokens: List[Token]) -> None:
        """Register **one** grammar object."""
        self.raw_rules[name] = tokens
        if name not in self.symbol_to_index:
            idx = len(self.symbol_to_index)
            self.symbol_to_index[name] = idx
            self.index_to_symbol[idx] = name

    @staticmethod
    def expand_literal(tok: Token) -> List[int]:
        """Expand a *literal spec* → concrete list of **token IDs**."""
        if isinstance(tok, int): return [tok]
        if isinstance(tok, tuple): return list(range(tok[0], tok[1] + 1)) if len(tok) == 2 else list(tok)
        raise ValueError()

    @staticmethod
    def wrap_pred(raw: Any) -> Predicate:
        """Return predicate *raw* unchanged if callable, else ``cnt >= raw``."""
        return raw if callable(raw) else (lambda cnt, m=raw: cnt >= m)


class TorchGrammarBatchParser:
    """Incremental **batch** parser operating entirely in *Torch* tensors."""
    DEFAULT_MAX_DEPTH = 16
    DEFAULT_MAX_LEN = 128
    DEFAULT_MAX_NEXT = 32

    def __init__(self, grammar: TorchGrammar, batch_size: int, *, device: str | torch.device = "cuda", max_depth: int | None = None, max_len: int | None = None, max_next: int | None = None):
        self.g = grammar
        self.bs = batch_size
        self.dev = torch.device(device)
        self.max_depth = max_depth if max_depth is not None else self.DEFAULT_MAX_DEPTH
        self.max_len = max_len if max_len is not None else self.DEFAULT_MAX_LEN
        self.max_next = max_next if max_next is not None else self.DEFAULT_MAX_NEXT
        self.stack = torch.full((self.bs, self.max_depth, 3), -1, dtype=torch.long, device=self.dev)
        self.ptr = torch.zeros(self.bs, dtype=torch.long, device=self.dev)
        self.seqs = torch.full((self.bs, self.max_len), -1, dtype=torch.long, device=self.dev)
        self.lens = torch.zeros(self.bs, dtype=torch.long, device=self.dev)
        self.next_cache = torch.full((self.bs, self.max_next), -1, dtype=torch.long, device=self.dev)
        self.next_lens = torch.zeros(self.bs, dtype=torch.long, device=self.dev)

    def set_batch(self, roots: List[str]) -> None:
        """Reset parser state with new *root objects* (length == ``batch_size``)."""
        for i, name in enumerate(roots):
            idx = self.g.symbol_to_index[name]
            self.stack[i, 0] = torch.tensor([idx, 0, 0], device=self.dev)
            self.ptr[i] = 1
        self._refresh_next()

    def fast_is_valid(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(self.bs, dtype=torch.bool, device=self.dev)
        for i in range(self.bs):
            if self.ptr[i] == 0: continue
            tok = tokens[i].item()
            n_cached = self.next_lens[i].item()
            if n_cached and (self.next_cache[i, :n_cached] == tok).any():
                mask[i] = True
                pos = self.lens[i].item()
                if pos < self.max_len:
                    self.seqs[i, pos] = tok
                    self.lens[i] += 1
                o_idx, o_ptr, o_rep = self.stack[i, self.ptr[i] - 1].tolist()
                obj_name = self.g.index_to_symbol[o_idx]
                curr_tok = self.g.raw_rules[obj_name][o_ptr]
                self._apply_token(i, curr_tok, tok, o_rep)
        self._refresh_next()
        return mask

    def next_token_mask(self, vocab_size: int, *, fill_value: float = float("-inf")) -> torch.Tensor:
        mask = torch.full((self.bs, vocab_size), fill_value, device=self.dev)
        for i in range(self.bs):
            k = self.next_lens[i].item()
            if k: mask[i, self.next_cache[i, :k]] = 0.0
        return mask

    def get_sequences(self) -> List[List[int]]:
        return [self.seqs[i, : self.lens[i]].tolist() for i in range(self.bs)]

    def decode_sequences(self, id2token: Dict[int, str]) -> List[List[str]]:
        return [[id2token[t] for t in seq] for seq in self.get_sequences()]

    def _resolve_to_literals(self, symbol: Token, visited: Optional[set] = None) -> List[int]:
        """Helper to recursively resolve a grammar symbol to a list of terminal token IDs."""
        if visited is None:
            visited = set()

        # Base case: The symbol is already a literal that can be expanded.
        if isinstance(symbol, (int, tuple)):
            return self.g.expand_literal(symbol)

        # If the symbol is not a string, it's not a non-terminal we can look up.
        if not isinstance(symbol, str):
            return []

        # Avoid infinite loops in cyclic grammars.
        if symbol in visited:
            return []
        visited.add(symbol)

        # It's a non-terminal string. Get its rule from the grammar.
        if symbol not in self.g.raw_rules:
            return [] 

        all_literals = []
        # For each possible next symbol in the rule...
        for next_symbol in self.g.raw_rules[symbol]:
            # ...recursively find all the literals it can start with.
            all_literals.extend(self._resolve_to_literals(next_symbol, visited.copy()))

        return all_literals

    def _compute_valid(self, tok: Token, rep_cnt: int, tokens: List[Token], idx: int) -> List[int]:
        """Compute *valid next literals* by recursively resolving grammar rules."""
        # Handle repetition rules (dict tokens)
        if isinstance(tok, dict):
            sub, raw_pred = next(iter(tok.items()))
            pred = self.g.wrap_pred(raw_pred)

            # Case 1 – Below repetition threshold → MUST stay inside the sub-object.
            if not pred(rep_cnt):
                return list(set(self._resolve_to_literals(sub)))

            # Case 2 – Threshold met → Exit OR Repeat is allowed.
            valid_literals: List[int] = []

            # 2a) Add literals for the "exit" path (the token after the repetition group).
            if idx + 1 < len(tokens):
                next_in_seq = tokens[idx + 1]
                valid_literals.extend(self._resolve_to_literals(next_in_seq))

            # 2b) Add literals for the "repeat" path (the start of the sub-object again).
            valid_literals.extend(self._resolve_to_literals(sub))
            return list(set(valid_literals))

        # Handle non-dict tokens (standard non-terminals or terminals).
        return list(set(self._resolve_to_literals(tok)))

    def _apply_token(self, i: int, curr: Token, tok: int, rep_cnt: int) -> None:
        if isinstance(curr, dict):
            sub, _ = next(iter(curr.items()))
            first_lits = set(self.g.expand_literal(self.g.raw_rules[sub][0]))
            if tok not in first_lits:
                self._pop_and_advance(i)
                return
            depth = self.ptr[i].item()
            sid = self.g.symbol_to_index[sub]
            self.stack[i, depth] = torch.tensor([sid, 1, 0], device=self.dev)
            self.ptr[i] += 1
            self.stack[i, depth - 1, 2] += 1
        elif isinstance(curr, str):
            depth = self.ptr[i].item()
            sid = self.g.symbol_to_index[curr]
            self.stack[i, depth] = torch.tensor([sid, 1, 0], device=self.dev)
            self.ptr[i] += 1
        else:
            self.stack[i, self.ptr[i] - 1, 1] += 1
            self.stack[i, self.ptr[i] - 1, 2] = 0

    def _pop_and_advance(self, i: int) -> None:
        self.ptr[i] -= 1
        if self.ptr[i] == 0: return
        o_idx, o_ptr, _ = self.stack[i, self.ptr[i] - 1].tolist()
        token = self.g.raw_rules[self.g.index_to_symbol[o_idx]][o_ptr]
        if not isinstance(token, dict):
            self.stack[i, self.ptr[i] - 1, 1] = o_ptr + 1
            self.stack[i, self.ptr[i] - 1, 2] = 0

    def _refresh_next(self) -> None:
        self.next_lens.zero_()
        for i in range(self.bs):
            while self.ptr[i] > 0:
                o_idx, o_ptr, o_rep = self.stack[i, self.ptr[i] - 1].tolist()
                obj_name = self.g.index_to_symbol[o_idx]
                tokens = self.g.raw_rules[obj_name]
                if o_ptr >= len(tokens):
                    self._pop_and_advance(i)
                    continue
                curr = tokens[o_ptr]
                valid = self._compute_valid(curr, o_rep, tokens, o_ptr)
                if valid:
                    k = len(valid)
                    self.next_cache[i, :k] = torch.tensor(valid, device=self.dev)
                    self.next_lens[i] = k
                    break
                self.ptr[i] -= 1


class ModelTester:
    """Deserializes token streams and scores the resulting model."""
    MASS_MAP = {"MASS_1e0": 1, "MASS_1e1": 10, "MASS_1e2": 100, "MASS_1e3": 1000}
    CHARGE_MAP = {"CHARGE_-1": -1, "CHARGE_0": 0, "CHARGE_1": 1}
    REP_U1Y_MAP = {"REP_U1Y_minus1": -1, "REP_U1Y_0": 0, "REP_U1Y_1": 1}

    def __init__(self):
        self.validator = BatchValidator()

    @staticmethod
    def _after(tok: str, prefix: str) -> str: return tok.replace(f"{prefix}_", "")
    @staticmethod
    def _int(tok: str) -> int: return int(tok.split("_")[-1])

    def parse_particle(self, seq: List[str], i: int) -> Tuple[Dict[str, Any], int]:
        particle = {
            "id": f"p{self._int(seq[i + 1]) + 1}", "name": seq[i + 1],
            "type": self._after(seq[i + 2], "TYPE"), "mass": self.MASS_MAP.get(seq[i + 3], 0),
            "charge": self.CHARGE_MAP.get(seq[i + 4], 0),
        }
        i += 5
        while i < len(seq) and seq[i] != "END_PARTICLE": i += 1
        return particle, i + 1

    def parse_field(self, seq: List[str], i: int) -> Tuple[Dict[str, Any], int, List[Dict[str, Any]]]:
        field = {
            "id": f"m{self._int(seq[i + 1]) + 1}", "name": seq[i + 1],
            "type": self._after(seq[i + 2], "TYPE"), "dim": self._int(seq[i + 3]),
            "gen": self._int(seq[i + 4]), "self_conjugate": seq[i + 5] == "SELF_CONJ_TRUE",
            "chirality": None if seq[i + 6] == "CHIRALITY_na" else self._after(seq[i + 6], "CHIRALITY"),
            "reps": { "g1": self.REP_U1Y_MAP.get(seq[i + 9], 0), "g2": self._after(seq[i + 8], "REP_SU2L"), "g3": self._after(seq[i + 7], "REP_SU3C")},
            "QuantumNumber": {"LeptonNumber": self._int(seq[i + 10]), "BaryonNumber": self._int(seq[i + 11])},
            "particles": [],
        }
        i += 12
        particles_in_field = []
        while i < len(seq) and seq[i] != "END_FIELD":
            if seq[i] == "PARTICLE":
                p, i_after_p = self.parse_particle(seq, i)
                particles_in_field.append(p)
                field["particles"].append(p["id"])
                i = i_after_p
            else: i += 1
        return field, i + 1, particles_in_field

    def parse_interaction(self, seq: List[str], i: int) -> Tuple[Dict[str, Any], int, List[Dict[str, Any]], List[Dict[str, Any]]]:
        inter = { "id": f"i{self._int(seq[i + 1]) + 1}", "type": self._after(seq[i + 2], "TYPE").lower(), "fields": []}
        i += 3
        fields_in_inter, particles_in_inter = [], []
        while i < len(seq) and seq[i] != "END_ITRACT":
            if seq[i] == "FIELD":
                f, i_after_f, ps = self.parse_field(seq, i)
                fields_in_inter.append(f)
                particles_in_inter.extend(ps)
                inter["fields"].append(f["id"])
                i = i_after_f
            else: i += 1
        return inter, i + 1, fields_in_inter, particles_in_inter

    def build_model(self, tokens: List[str]) -> Dict[str, Any]:
        particles, fields, inters, i = [], [], [], 0
        while i < len(tokens):
            if tokens[i] == "PARTICLE": p, i = self.parse_particle(tokens, i); particles.append(p)
            elif tokens[i] == "FIELD": f, i, new_ps = self.parse_field(tokens, i); fields.append(f); particles.extend(new_ps)
            elif tokens[i] == "ITRACT": it, i, new_fs, new_ps = self.parse_interaction(tokens, i); inters.append(it); fields.extend(new_fs); particles.extend(new_ps)
            else: i += 1
        particles = list({p["id"]: p for p in particles}.values())
        fields = list({f["id"]: f for f in fields}.values())
        inters = list({it["id"]: it for it in inters}.values())
        return {
            "GaugeGroups": [
                {"id": "g1", "name": "U1Y", "charge": "Y", "group": "U_1", "coupling": "g1", "boson": "B"},
                {"id": "g2", "name": "SU2L", "charge": "I", "group": "SU_2", "coupling": "gw", "boson": "W"},
                {"id": "g3", "name": "SU3C", "charge": "C", "group": "SU_3", "coupling": "gs", "boson": "G"},
            ],
            "vevs": [{"id": "v1", "name": "vev", "vacuum": [0, 1], "value": 246.22}],
            "particles": particles, "fields": fields, "interactions": inters,
            "links": {"vevs": [{"source": "v1", "target": fields[0]["id"] if fields else ""}]},
        }

    def test_model(self, tokens: List[str]) -> Tuple[float, str]:
        """Builds and scores the model using the BatchValidator."""
        try:
            model_json_data = self.build_model(tokens)
        except Exception as e:
            logger.debug(f"Parsing error in build_model for sequence. Error: {e}")
            return 0.0, "<parsing_error>"
        try:
            # Use the integrated BatchValidator to score the model
            scores = self.validator.validate_batch([model_json_data])
            score_val = scores[0] if scores.size > 0 else 0.0
            return float(score_val), "<in-memory-model>"
        except Exception as e:
            logger.error(f"Error during model validation: {e}")
            # logger.debug(f"Problematic JSON: {json.dumps(model_json_data, indent=2)}")
            return 0.0, f"<error_scoring: {str(e)[:50]}>"


# ─────────────────────────────────────────────────────────────────────────────
# PPO REINFORCEMENT LEARNING ALGORITHM
# ─────────────────────────────────────────────────────────────────────────────

warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True")
original_stderr = sys.stderr
class CppWarningFilter(logging.Filter):
    def filter(self, record):
        suppress_phrases = ["Unable to register cuFFT factory", "Unable to register cuDNN factory", "Unable to register cuBLAS factory", "computation placer already registered", "This TensorFlow binary is optimized"]
        return not any(phrase in record.getMessage() for phrase in suppress_phrases)

cpp_logger = logging.getLogger("cpp_warnings_filter")
cpp_logger.propagate = False
handler = logging.StreamHandler(original_stderr)
handler.addFilter(CppWarningFilter())
cpp_logger.addHandler(handler)

class StderrRedirector:
    def __init__(self, logger):
        self._logger = logger
        self._local = threading.local()
    def write(self, message):
        if getattr(self._local, 'is_writing', False): return
        try:
            self._local.is_writing = True
            if message.rstrip(): self._logger.warning(message.rstrip())
        finally: self._local.is_writing = False
    def flush(self):
        for h in self._logger.handlers: h.flush()

sys.stderr = StderrRedirector(cpp_logger)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SEED = 42
EPS = 1e-9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED); torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
_AMP_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
SCALER = amp.GradScaler(enabled=(DEVICE.type == "cuda"))
CAN_COMPILE = hasattr(torch, "compile")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
main_logger = logging.getLogger(__name__)


class TransformerPolicy(nn.Module):
    def __init__(self, vocab: int, *, d_model: int = 16, nhead: int = 4, layers: int = 3, max_T: int = 512):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.layers = layers
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_T, d_model)
        self.register_buffer("pos", torch.arange(max_T))
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
        self.actor = nn.Linear(d_model, vocab)
        self.value = nn.Linear(d_model, 1)

    def forward(self, 
                seq: torch.Tensor, 
                cache: Optional[List[Dict[str, torch.Tensor]]] = None
               ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Forward pass with KV Caching.

        Args:
            seq: Input token sequence.
                 - During training/no cache: (batch, seq_len)
                 - During generation/with cache: (batch, 1)
            cache: A list of dictionaries, one for each layer, containing 'k' and 'v' tensors.

        Returns:
            A tuple of (logits, value, new_cache).
        """
        B, T = seq.size()
        pos_offset = cache[0]['k'].size(1) if cache is not None else 0
        if pos_offset + T > self.pos_emb.num_embeddings:
            raise ValueError(f"Sequence length {pos_offset + T} exceeds max_T {self.pos_emb.num_embeddings}")
        pos_emb = self.pos_emb(self.pos[pos_offset : pos_offset + T])
        x = self.tok_emb(seq) + pos_emb
        new_cache = []
        attn_mask = None
        if T > 1:
             attn_mask = nn.Transformer.generate_square_subsequent_mask(T, device=seq.device)

        for i, layer in enumerate(self.encoder_layers):
            x_norm = layer['norm1'](x)
            
            # --- KV Cache Logic ---
            if cache is not None:
                qkv = torch.cat((
                    x_norm, # Query is from the new token
                    cache[i]['k'], # Key is from the cache
                    cache[i]['v'] # Value is from the cache
                ), dim=1)
                q = x_norm
                k = qkv[:, 1:1+pos_offset+T]
                v = qkv[:, 1+pos_offset+T:]
            else:
                q, k, v = x_norm, x_norm, x_norm
            attn_output, _ = layer['attn'](q, k, v, attn_mask=attn_mask, need_weights=False)
            x = x + attn_output
            x_ffn = layer['ffn'](layer['norm2'](x))
            x = x + x_ffn
            new_cache.append({'k': k.detach(), 'v': v.detach()})
        h = self.final_norm(x)
        return self.actor(h), self.value(h).squeeze(-1), new_cache

@dataclass(frozen=True)
class HParams:
    epochs: int = 500
    batch_size: int = 1
    seq_len: int = 512
    lr: float = 3e-4
    gamma: float = 0.997
    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    gae_lambda: float = 0.95
    sup_coef: float = 0.05
    value_coef: float = 0.5
    ent_coef: float = 0.01
    grad_clip_norm: float = 1.0
    ckpt_every: int = 10
    buffer_size: int = 32
    min_buffer_size: int = 4
    rank_k: float = 0.5
    enable_logging: bool = True
    debug: bool = True

class Trajectory(NamedTuple):
    """Stores raw data for a trajectory, detached from any computation graph."""
    seq: torch.Tensor
    rewards: torch.Tensor
    logps: torch.Tensor
    values: torch.Tensor
    ep_len: int
    total_reward: float

class ExperienceBuffer:
    def __init__(self, hp: HParams):
        self.buffer = deque(maxlen=hp.buffer_size)
        self.hp = hp

    def __len__(self) -> int: return len(self.buffer)

    def add_batch(self, generated_seqs: torch.Tensor, batch_data: Dict[str, torch.Tensor]) -> None:
        rews, logps, values = batch_data['rewards'], batch_data['logps'], batch_data['values']
        disc_ret_rank = torch.zeros_like(rews)
        R_rank = torch.zeros(rews.size(0), device=DEVICE)
        for rev_t in range(rews.size(1) - 1, -1, -1):
            R_rank = rews[:, rev_t] + self.hp.gamma * R_rank
            disc_ret_rank[:, rev_t] = R_rank
        for i in range(rews.size(0)):
            ep_len = batch_data['ep_len'][i].item()
            if ep_len > 0:
                traj = Trajectory(
                    seq=generated_seqs[i, :ep_len + 1].clone().cpu(),
                    rewards=rews[i, :ep_len].clone().cpu(),
                    logps=logps[i, :ep_len].clone().cpu(),
                    values=values[i, :ep_len].clone().cpu(),
                    ep_len=ep_len,
                    total_reward=disc_ret_rank[i, 0].item()
                )
                if self.hp.debug: print(f"  [DEBUG] Adding trajectory to buffer. Length: {traj.ep_len}, Total Reward: {traj.total_reward:.4f}")
                self.buffer.append(traj)

    def sample(self) -> Dict[str, torch.Tensor]:
        buffer_rewards = torch.tensor([t.total_reward for t in self.buffer], device=DEVICE)
        D_size = len(self.buffer)
        ranks = buffer_rewards.argsort(descending=True).argsort().float() + 1.0
        weights_unnorm = (self.hp.rank_k * D_size + ranks).pow(-1)
        weights = weights_unnorm / weights_unnorm.sum()
        indices = torch.multinomial(weights, self.hp.batch_size, replacement=True)
        samples = [self.buffer[i] for i in indices]
        max_len = max(s.ep_len for s in samples)
        collated_seqs = torch.full((self.hp.batch_size, max_len + 1), 0, device=DEVICE, dtype=torch.long) # Filled with BOS_TOKEN later
        collated_rewards = torch.zeros((self.hp.batch_size, max_len), device=DEVICE)
        collated_logps = torch.zeros((self.hp.batch_size, max_len), device=DEVICE)
        collated_values = torch.zeros((self.hp.batch_size, max_len), device=DEVICE)
        collated_ep_len = torch.tensor([s.ep_len for s in samples], device=DEVICE, dtype=torch.long)
        for i, s in enumerate(samples):
            len_s = s.ep_len
            collated_seqs[i, :len_s + 1] = s.seq.to(DEVICE)
            collated_rewards[i, :len_s] = s.rewards.to(DEVICE)
            collated_logps[i, :len_s] = s.logps.to(DEVICE)
            collated_values[i, :len_s] = s.values.to(DEVICE)
        advantages = torch.zeros_like(collated_rewards)
        last_gae_lam = 0
        for t in reversed(range(max_len)):
            is_not_last_step = (t < collated_ep_len - 1).float()
            next_values = collated_values[:, t + 1] if t < max_len - 1 else torch.zeros_like(collated_values[:, t])
            delta = collated_rewards[:, t] + self.hp.gamma * next_values * is_not_last_step - collated_values[:, t]
            advantages[:, t] = last_gae_lam = delta + self.hp.gamma * self.hp.gae_lambda * last_gae_lam * is_not_last_step
        returns = advantages + collated_values
        return {'seq': collated_seqs, 'logps': collated_logps, 'values': collated_values, 'returns': returns, 'advantages': advantages, 'ep_len': collated_ep_len}

def ensure_dirs(paths: Union[str, Path, Iterable[Union[str, Path]]]) -> None:
    if isinstance(paths, (str, Path)): paths = [paths]
    for p in paths: Path(p).mkdir(parents=True, exist_ok=True)
def safe_item(x: Union[float, torch.Tensor]) -> float:
    v = float(x.item() if isinstance(x, torch.Tensor) else x)
    return v if math.isfinite(v) else 0.0
def warmup_cosine(opt: torch.optim.Optimizer, warm: int, total: int) -> LambdaLR:
    def f(step: int) -> float:
        if total == 0: return 1.0
        if step < warm: return step / max(1, warm)
        p = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * p))
    return LambdaLR(opt, f, -1)
def _build_adamw(params, lr):
    try: return torch.optim.AdamW(params, lr=lr, fused=True)
    except Exception:
        warnings.warn("Fused AdamW unavailable → fallback."); return torch.optim.AdamW(params, lr=lr)

def train_reinforce(model: nn.Module, opt, hp: HParams, grammar: TorchGrammar, global_vars: dict) -> Path:
    # Unpack global variables
    TOKEN2ID, ID2TOK, VOCAB_SIZE = global_vars['TOKEN2ID'], global_vars['ID2TOK'], global_vars['VOCAB_SIZE']
    BOS_TOKEN, EOS_TOKEN = global_vars['BOS_TOKEN'], global_vars['EOS_TOKEN']
    ROOTS, START_TOKS, END_TOKS = global_vars['ROOTS'], global_vars['START_TOKS'], global_vars['END_TOKS']

    # --- This initial setup section is unchanged ---
    run_id = datetime.now().strftime("comb_%Y%m%d_%H%M%S")
    root_dir = Path.cwd()
    logdir = root_dir / "particle" / "runs" / run_id
    ckptdir, prof_dir = logdir / "checkpoints", logdir / "profiler"
    ensure_dirs([logdir, ckptdir, prof_dir])

    logger: Optional[Logger] = None
    if hp.enable_logging:
        logger = Logger(logdir, cli_interval=1, state_interval=1, weights_interval=hp.ckpt_every)
        logger.model = model
        for name in ["reward", "seq_len", "pl", "vl", "el", "gl", "tl", "buffer_size", "n_particles", "n_fields", "n_itracts", "l1_norm", "l2_norm", "grad_l1_norm", "grad_l2_norm", "logit_entropy", "approx_kl"]:
            logger.add_metric(name, kind="stat")
        for name in ["n_updates", "n_rollouts", "n_samples", "learning_rate"]:
            logger.add_metric(name, kind="scalar")

    if CAN_COMPILE:
        print("Compiling model …"); model = torch.compile(model, mode="max-autotune"); print("Compilation complete.")

    sched = warmup_cosine(opt, int(hp.epochs * 0.10), hp.epochs)
    tester = ModelTester()

    prof_sched = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=10)
    trace_handler = torch.profiler.tensorboard_trace_handler(str(prof_dir))
    experience_buffer = ExperienceBuffer(hp)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], schedule=prof_sched, on_trace_ready=trace_handler, record_shapes=True, profile_memory=True, with_stack=True, with_flops=True, with_modules=True) as prof:
        for ep in trange(1, hp.epochs + 1, desc="epochs", leave=False, disable=not hp.debug):
            prof.step()
            model.eval()
            B = hp.batch_size


            with record_function("rollout_generation"):
                parser = TorchGrammarBatchParser(grammar, B, device=DEVICE, max_len=hp.seq_len+1)
                parser.set_batch(random.choices(ROOTS, k=B))
                
                # --- Store the full generated sequences here
                generated_seqs = torch.full((B, hp.seq_len + 1), 0, device=DEVICE, dtype=torch.long)
                generated_seqs[:, 0] = BOS_TOKEN
                
                # --- Start with only the BOS token to feed into the model
                current_tokens = torch.full((B, 1), BOS_TOKEN, device=DEVICE, dtype=torch.long)
                
                # --- Initialize the KV cache as empty
                cache = None
                
                ep_len, alive = torch.zeros(B, dtype=torch.long, device=DEVICE), torch.ones(B, dtype=torch.bool, device=DEVICE)
                rewards_history, logps_history, values_history = [], [], []
                history = [[BOS_TOKEN] for _ in range(B)]
                n_particles, n_fields, n_itracts = torch.zeros(B, device=DEVICE), torch.zeros(B, device=DEVICE), torch.zeros(B, device=DEVICE)

                for t in range(hp.seq_len):
                    if not alive.any(): # --- Early exit if all sequences in the batch are done
                        break

                    ep_len[alive] += 1
                    with torch.no_grad():
                        # --- Efficiently call model with only the current token and the cache
                        logits_t, values_t, cache = model(current_tokens, cache)
                    
                    logits, mask = logits_t.float(), parser.next_token_mask(VOCAB_SIZE)
                    
                    eligible = alive & (~parser.next_lens.eq(0))
                    if not eligible.any(): break
                        
                    probs = F.softmax(logits + mask, dim=-1)
                    probs[~eligible] = 0.0; probs[~eligible, EOS_TOKEN] = 1.0
                    if torch.isnan(probs).any():
                        nan_rows = torch.isnan(probs).any(dim=-1)
                        probs[nan_rows] = 0.0; probs[nan_rows, EOS_TOKEN] = 1.0
                    
                    dist = Categorical(probs=probs)
                    acts = dist.sample()
                    logps_val = dist.log_prob(acts)
                    
                    rew, valid_row = torch.zeros(B, device=DEVICE), parser.fast_is_valid(acts)
                    n_particles[valid_row] += (acts[valid_row] == TOKEN2ID["PARTICLE"]).float()
                    n_fields[valid_row] += (acts[valid_row] == TOKEN2ID["FIELD"]).float()
                    n_itracts[valid_row] += (acts[valid_row] == TOKEN2ID["ITRACT"]).float()

                    sequences_to_score_info: List[Dict[str, Any]] = []
                    for i in range(B):
                        if not eligible[i]: continue
                        tok_id = acts[i].item(); history[i].append(tok_id)
                        if tok_id in END_TOKS:
                            obj_type, start_tok_id = END_TOKS[tok_id], next((tid for tid, s in START_TOKS.items() if s == END_TOKS[tok_id]), None)
                            if start_tok_id is not None:
                                rev_idx = next((k for k in range(len(history[i]) - 2, -1, -1) if history[i][k] == start_tok_id), -1)
                                if rev_idx != -1: sequences_to_score_info.append({'batch_idx': i, 'bonus_type': 'object', 'tokens': [ID2TOK[x] for x in history[i][rev_idx:]]})
                        if tok_id == EOS_TOKEN: sequences_to_score_info.append({'batch_idx': i, 'bonus_type': 'eos', 'tokens': [ID2TOK[x] for x in history[i][1:]]})

                    if sequences_to_score_info:
                        try:
                            models_to_score = [tester.build_model(info['tokens']) for info in sequences_to_score_info]
                            scored_vals = tester.validator.validate_batch(models_to_score)
                            for info, s_val in zip(sequences_to_score_info, scored_vals):
                                rew[info['batch_idx']] += s_val if info['bonus_type'] == 'object' else 1000 * s_val
                        except Exception as e: main_logger.error("Scoring error", exc_info=e)
                    
                    rewards_history.append(rew); logps_history.append(logps_val.clone()); values_history.append(values_t.clone())
                    
                    # --- Update the full sequence history and set the next token for the model
                    generated_seqs[:, t + 1] = acts
                    current_tokens = acts.unsqueeze(1)
                    alive = alive & valid_row & (acts != EOS_TOKEN)

            if rewards_history:
                batch_dict = {'rewards': torch.stack(rewards_history, dim=1), 'logps': torch.stack(logps_history, dim=1), 'values': torch.stack(values_history, dim=1), 'ep_len': ep_len}
                # --- Pass the fully generated sequences to the experience buffer
                experience_buffer.add_batch(generated_seqs, batch_dict)
                if hp.enable_logging and logger:
                    logger.log({"reward": batch_dict['rewards'].sum(1).mean().item(), "seq_len": ep_len.float().mean().item(), "buffer_size": len(experience_buffer), "n_particles": n_particles.mean().item(), "n_fields": n_fields.mean().item(), "n_itracts": n_itracts.mean().item()})

            # --- This PPO update section is unchanged ---
            with record_function("ppo_update_phase"):
                model.train()
                if len(experience_buffer) < hp.min_buffer_size:
                    if hp.debug: print(f"[DEBUG] Ep {ep} skip train (buffer {len(experience_buffer)} < {hp.min_buffer_size})")
                    sched.step();
                    if hp.enable_logging and logger: logger.cli_step()
                    continue
                for _ in range(hp.ppo_epochs):
                    tbatch = experience_buffer.sample()
                    b_seq, b_actions, max_len = tbatch['seq'], tbatch['seq'][:, 1:].clone(), tbatch['returns'].size(1)
                    
                    # --- The model call here processes the full sequence, which is correct for training
                    logits, vals, _ = model(b_seq[:, :-1]) # Ignore the returned cache
                    
                    dist = Categorical(logits=logits)
                    new_logps = dist.log_prob(b_actions)
                    ents = dist.entropy()
                    ts_grid = torch.arange(max_len, device=DEVICE).unsqueeze(0)
                    valid_mask = ts_grid < tbatch['ep_len'].unsqueeze(1)
                    N_valid = valid_mask.sum().clamp_min(1.0)
                    adv = tbatch['advantages']
                    if N_valid > 1: flat = adv[valid_mask]; adv = (adv - flat.mean()) / (flat.std() + EPS)
                    log_ratio = new_logps - tbatch['logps']
                    ratio = log_ratio.exp()
                    surr1, surr2 = ratio * adv, torch.clamp(ratio, 1 - hp.clip_epsilon, 1 + hp.clip_epsilon) * adv
                    pg_loss = -torch.min(surr1, surr2)[valid_mask].sum() / N_valid
                    v_loss = F.mse_loss(vals[valid_mask], tbatch['returns'][valid_mask])
                    e_loss = -ents[valid_mask].sum() / N_valid

                    train_parser = TorchGrammarBatchParser(grammar, hp.batch_size, device=DEVICE, max_len=hp.seq_len+1)
                    train_parser.set_batch(random.choices(ROOTS, k=hp.batch_size))
                    g_penalties = []
                    for t in range(max_len):
                        mask_t, logits_t = train_parser.next_token_mask(VOCAB_SIZE), logits[:, t]
                        banned = (mask_t != 0.0).float()
                        pen_row = F.relu(logits_t * banned).sum(-1)
                        g_penalties.append(pen_row / banned.sum(-1).clamp_min(1.0))
                        train_parser.fast_is_valid(b_actions[:, t])
                    g_loss = torch.stack(g_penalties, dim=1)[valid_mask].sum() / N_valid
                    total_loss = pg_loss + hp.value_coef * v_loss + hp.ent_coef * e_loss + hp.sup_coef * g_loss

                    # Print detailed loss breakdown
                    print(f"Epoch {ep} - Loss Components:")
                    print(f"  Policy Loss (PG): {pg_loss.item():.6f}")
                    print(f"  Value Loss (VL): {v_loss.item():.6f} (coef: {hp.value_coef})")
                    print(f"  Entropy Loss (EL): {e_loss.item():.6f} (coef: {hp.ent_coef})")
                    print(f"  Grammar Loss (GL): {g_loss.item():.6f} (coef: {hp.sup_coef})")
                    print(f"  Total Loss: {total_loss.item():.6f}")
                    print(f"  Valid tokens: {N_valid.item()}")
                    print(f"  Approx KL: {log_ratio[valid_mask].mean().item():.6f}")
                    print("-" * 50)

                    opt.zero_grad(set_to_none=True)
                    SCALER.scale(total_loss).backward()
                    SCALER.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip_norm)
                    SCALER.step(opt); SCALER.update()
                    if hp.enable_logging and logger: logger.state_step()
            
            # --- This final logging/checkpointing section is unchanged ---
            sched.step()
            if hp.enable_logging and logger:
                scale = SCALER.get_scale()
                grads_exist = any(p.grad is not None for p in model.parameters())
                g_l1 = sum((p.grad / scale).abs().sum().item() for p in model.parameters() if p.grad is not None) if grads_exist else 0.0
                g_l2 = math.sqrt(sum(((p.grad / scale) ** 2).sum().item() for p in model.parameters() if p.grad is not None)) if grads_exist else 0.0
                logger.log({ "n_updates": hp.ppo_epochs, "n_rollouts": 1, "n_samples": B, "learning_rate": sched.get_last_lr()[0], "l1_norm": sum(p.abs().sum().item() for p in model.parameters()), "l2_norm": math.sqrt(sum((p ** 2).sum().item() for p in model.parameters())), "grad_l1_norm": g_l1, "grad_l2_norm": g_l2 })
                logger.cli_step()

            if ep % hp.ckpt_every == 0 or ep == hp.epochs:
                ckpt_path = ckptdir / f"ep{ep}.pt"
                model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save(model_to_save.state_dict(), ckpt_path)
                main_logger.info("✓ saved %s", ckpt_path)
                
                # Sample from the model to show progress
                print(f"\n=== Sampling from model at epoch {ep} ===")
                sample(model, hp, grammar, global_vars, n=3, temp=0.7)
                print(f"=== End of epoch {ep} samples ===\n")
    
    # --- Profiler output is unchanged ---
    print("\n================ PyTorch profiler (top-20 CUDA) ================")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
    print("\n================ PyTorch profiler (top-20 CPU) ================")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
    print(f"\n🎛  Inspect interactive traces with:\n    tensorboard --logdir={prof_dir}\n")
    prof.export_stacks(str(logdir / "profiler_cpu.stacks"), metric="self_cpu_time_total")
    prof.export_stacks(str(logdir / "profiler_cuda.stacks"), metric="self_cuda_time_total")

    final_ckpt = ckptdir / f"ep{hp.epochs}.pt"
    if hp.epochs % hp.ckpt_every != 0:
        model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
        torch.save(model_to_save.state_dict(), final_ckpt)

    return final_ckpt

@torch.no_grad()
def sample(model_to_sample: nn.Module, hp: HParams, grammar: TorchGrammar, global_vars: dict, *, n: int = 4, temp: float = 1.0) -> None:
    # Unpack global variables
    TOKEN2ID, ID2TOK, VOCAB_SIZE = global_vars['TOKEN2ID'], global_vars['ID2TOK'], global_vars['VOCAB_SIZE']
    BOS_TOKEN, EOS_TOKEN = global_vars['BOS_TOKEN'], global_vars['EOS_TOKEN']
    ROOTS = global_vars['ROOTS']

    model_to_sample.eval()
    starts = {TOKEN2ID[s] for s in ["PARTICLE", "FIELD", "ITRACT"]}
    ends = {TOKEN2ID[s] for s in ["END_PARTICLE", "END_FIELD", "END_ITRACT"]}
    for i in range(n):
        parser = TorchGrammarBatchParser(grammar, 1, device=DEVICE, max_len=hp.seq_len+1)
        parser.set_batch([random.choice(ROOTS)])
        seq_ids, indent = [BOS_TOKEN], 0
        print(f"\nSample {i+1}:\n", end="")
        for token_index_in_sample in range(hp.seq_len):
            logits_all, _ = model_to_sample(torch.tensor([seq_ids], device=DEVICE))
            logits_tensor, mask_tensor = logits_all[:, -1].squeeze(0), parser.next_token_mask(VOCAB_SIZE)[0]
            logits_tensor = logits_tensor + mask_tensor
            if temp <= EPS: tok_id = int(torch.argmax(logits_tensor).item())
            else:
                probs_tensor = F.softmax(logits_tensor / max(temp, 1e-5), 0)
                if not torch.isfinite(probs_tensor).any():
                    valid_ids_tensor = (mask_tensor == 0).nonzero(as_tuple=True)[0]
                    tok_id = int(valid_ids_tensor[random.randrange(len(valid_ids_tensor))].item()) if len(valid_ids_tensor) > 0 else EOS_TOKEN
                else: tok_id = int(Categorical(probs_tensor).sample().item())
            token_str = ID2TOK[tok_id]
            if token_index_in_sample == 0:
                if tok_id in ends: indent = max(0, indent - 1); print(f"{'  ' * indent}{token_str}", end="")
                elif tok_id in starts: print(f"{'  ' * indent}{token_str}", end=""); indent += 1
                else: print(f"{token_str}", end="")
            else:
                if tok_id in ends: indent = max(0, indent - 1); print(f",\n{'  ' * indent}{token_str}", end="")
                elif tok_id in starts: print(f",\n{'  ' * indent}{token_str}", end=""); indent += 1
                else: print(f", {token_str}", end="")
            seq_ids.append(tok_id)
            is_valid_token = parser.fast_is_valid(torch.tensor([tok_id], device=DEVICE)).item()
            if not is_valid_token: print("\n...(Invalid by grammar; terminating sample)]", end=""); break
            if tok_id == EOS_TOKEN or parser.next_lens[0] == 0: print("\n(Grammar complete)", end=""); break
    print()
    print("-" * 30)
    model_to_sample.train()


if __name__ == "__main__":
    # --- Hyperparameters ---
    hp = HParams()
    # hp = replace(hp, debug=True, epochs=20, enable_logging=False) # Example for debugging

    if hp.min_buffer_size > hp.buffer_size:
        warnings.warn(f"min_buffer_size ({hp.min_buffer_size}) > buffer_size ({hp.buffer_size}). Setting min_buffer_size = buffer_size.")
        hp = replace(hp, min_buffer_size=hp.buffer_size)

    # --- Grammar and Tokenizer Setup ---
    TOKEN2ID = {t: i for i, t in enumerate(ALL_TOKEN_NAMES)}
    ID2TOK = {i: t for t, i in TOKEN2ID.items()}
    VOCAB_SIZE = len(ALL_TOKEN_NAMES)
    BOS_TOKEN, EOS_TOKEN = TOKEN2ID["BOS"], TOKEN2ID["EOS"]

    # Define Start/End token mappings for scoring logic
    START_TOKS = {TOKEN2ID[k]: v for k, v in {"PARTICLE": "particle", "FIELD": "field", "ITRACT": "itract"}.items()}
    END_TOKS = {TOKEN2ID[k]: v for k, v in {"END_PARTICLE": "particle", "END_FIELD": "field", "END_ITRACT": "itract"}.items()}

    # Combine all grammar rule dictionaries
    full_grammar_rules = {**TOKENS_MODEL, **TOKENS_INTERACTION, **TOKENS_FIELD, **TOKENS_PARTICLE}

    # Instantiate and populate the grammar object
    grammar = TorchGrammar()
    for name, tokens in full_grammar_rules.items():
        grammar.add_object(name, tokens)

    ROOTS = ["BOS"]

    # Package global vars to pass to functions
    global_vars = {
        "TOKEN2ID": TOKEN2ID, "ID2TOK": ID2TOK, "VOCAB_SIZE": VOCAB_SIZE,
        "BOS_TOKEN": BOS_TOKEN, "EOS_TOKEN": EOS_TOKEN, "ROOTS": ROOTS,
        "START_TOKS": START_TOKS, "END_TOKS": END_TOKS,
    }

    # --- Main Execution ---
    main_logger.info(f"Device: {DEVICE}, AMP: {_AMP_DTYPE if DEVICE.type=='cuda' else 'off'}")
    main_logger.info(f"Algorithm: PPO with GAE. Buffer: size={hp.buffer_size}, k={hp.rank_k}")
    main_logger.info(f"Logging {'enabled' if hp.enable_logging else 'disabled'}")
    main_logger.info(f"Debug mode: {'ON' if hp.debug else 'OFF'}")

    transformer_model = TransformerPolicy(VOCAB_SIZE, max_T=hp.seq_len + 1).to(DEVICE)
    optimizer = _build_adamw(transformer_model.parameters(), hp.lr)

    final_checkpoint = train_reinforce(transformer_model, optimizer, hp, grammar, global_vars)
    main_logger.info("Training complete – model saved to %s", final_checkpoint)

    # Reload for sampling
    model_to_sample = TransformerPolicy(VOCAB_SIZE, max_T=hp.seq_len + 1).to(DEVICE)
    model_to_sample.load_state_dict(torch.load(final_checkpoint, map_location=DEVICE))

    print("\n=== Greedy samples (temp=0.0) ===")
    sample(model_to_sample, hp, grammar, global_vars, n=5, temp=0.0)

    print("\n=== Temperature-0.7 samples ===")
    sample(model_to_sample, hp, grammar, global_vars, n=5, temp=0.7)

    print("\nDone.")