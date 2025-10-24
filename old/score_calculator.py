# score_calculator.py
"""
Batch-Validates Particle Physics Models

A standalone utility for scoring particle physics models based on a predefined 
set of 15 validation checks. Features:

- Ingests batches of models defined in a dictionary format.
- Vectorizes model data into tensors for efficient, parallel processing on a GPU or CPU.
- Performs particle, group, field, and interaction-level checks.
- Returns a numerical score for each model in the batch, representing the number of passed checks.
"""

import torch
import numpy as np
import logging

# Configure a basic logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================================================================================
# Phase 0: Schema, Constants, and Configuration
# ======================================================================================

# --- Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
ITYPE = torch.int64

# --- Fixed-Size Schema (for padding and creating tensors)
MAX_PARTICLES = 50
MAX_FIELDS = 50
MAX_GROUPS = 10
MAX_INTERACTIONS = 50
MAX_PARTICLES_PER_FIELD = 50
PAD_VALUE = -1

# --- Mapping Dictionaries (Enums for converting strings to integers)
PARTICLE_TYPE_MAP = {'fermion': 0, 'real': 1, 'complex': 2, 'vector': 3}
FIELD_TYPE_MAP = {'fermion': 0, 'real': 1, 'complex': 2, 'vector': 3, 'scalar': 4}
GROUP_TYPE_MAP = {'U_1': 0, 'SU_2': 1, 'SU_3': 2}
CHIRALITY_MAP = {'none': 0, 'left': 1, 'right': 2}
INTERACTION_TYPE_MAP = {'yukawa': 0}

# --- Feature Indices (for easier slicing of tensors)
# Particle features
P_FEAT_ID, P_FEAT_TYPE, P_FEAT_MASS, P_FEAT_CHARGE, P_FEAT_SPIN = range(5)
NUM_PARTICLE_FEATURES = 5
# Field features
F_FEAT_ID, F_FEAT_TYPE, F_FEAT_DIM, F_FEAT_GEN, F_FEAT_SELF_CONJ, F_FEAT_CHIRALITY = range(6)
NUM_FIELD_FEATURES = 6
# Group features
G_FEAT_ID, G_FEAT_TYPE, G_FEAT_DIM = range(3)
NUM_GROUP_FEATURES = 3
# Interaction features
I_FEAT_ID, I_FEAT_TYPE, I_FEAT_F0, I_FEAT_F1, I_FEAT_F2 = range(5)
NUM_INTERACTION_FEATURES = 5

# ======================================================================================
# Phase 1: Data Ingestion and Vectorization (for the validator)
# ======================================================================================
def ingest_batch(batch_of_models: list, debug: bool = False) -> dict:
    """Converts a list of model dictionaries into a dictionary of batched tensors."""
    if debug:
        logger.debug(f"[Ingest] Starting ingestion for a batch of {len(batch_of_models)} models.")
    
    batch_size = len(batch_of_models)

    # Initialize numpy arrays
    np_particles = np.full((batch_size, MAX_PARTICLES, NUM_PARTICLE_FEATURES), PAD_VALUE, dtype=np.float32)
    np_fields = np.full((batch_size, MAX_FIELDS, NUM_FIELD_FEATURES), PAD_VALUE, dtype=np.float32)
    np_groups = np.full((batch_size, MAX_GROUPS, NUM_GROUP_FEATURES), PAD_VALUE, dtype=np.float32)
    np_interactions = np.full((batch_size, MAX_INTERACTIONS, NUM_INTERACTION_FEATURES), PAD_VALUE, dtype=np.int64)
    np_field_particle_map = np.full((batch_size, MAX_FIELDS, MAX_PARTICLES_PER_FIELD), PAD_VALUE, dtype=np.int64)

    for i, model in enumerate(batch_of_models):
        if not model: # Handle cases where parsing failed
            continue

        p_map = {p_data['id']: j for j, p_data in enumerate(model.get('particles', []))}
        f_map = {f_data['id']: j for j, f_data in enumerate(model.get('fields', []))}
        g_map = {g_data['id']: j for j, g_data in enumerate(model.get('GaugeGroups', []))}

        for j, g_data in enumerate(model.get('GaugeGroups', [])):
            if j >= MAX_GROUPS: continue
            np_groups[i, j, G_FEAT_ID] = g_map.get(g_data['id'], -1)
            group_str = g_data.get('group', '')
            np_groups[i, j, G_FEAT_TYPE] = GROUP_TYPE_MAP.get(group_str, -1)
            try:
                if '_' in group_str:
                    N = int(group_str.split('_')[-1])
                    np_groups[i, j, G_FEAT_DIM] = N
                else: # Handle groups like U_1
                    np_groups[i, j, G_FEAT_DIM] = 1
            except (ValueError, IndexError):
                np_groups[i, j, G_FEAT_DIM] = -1


        for j, p_data in enumerate(model.get('particles', [])):
            if j >= MAX_PARTICLES: continue
            np_particles[i, j, P_FEAT_ID] = p_map.get(p_data['id'],-1)
            p_type = p_data.get('type')
            if isinstance(p_type, str):
                p_type_val = PARTICLE_TYPE_MAP.get(p_type, -1)
            else:
                p_type_val = p_type if p_type in PARTICLE_TYPE_MAP.values() else -1
            np_particles[i, j, P_FEAT_TYPE] = p_type_val
            np_particles[i, j, P_FEAT_MASS] = p_data.get('mass', 0.0)
            np_particles[i, j, P_FEAT_CHARGE] = p_data.get('charge', 0)
            np_particles[i, j, P_FEAT_SPIN] = p_data.get('spin', 1)

        for j, f_data in enumerate(model.get('fields', [])):
            if j >= MAX_FIELDS: continue
            np_fields[i, j, F_FEAT_ID] = f_map.get(f_data['id'], -1)
            np_fields[i, j, F_FEAT_TYPE] = FIELD_TYPE_MAP.get(f_data.get('type', ''), -1)
            np_fields[i, j, F_FEAT_DIM] = f_data.get('dim', 1)
            np_fields[i, j, F_FEAT_GEN] = f_data.get('gen', 1)
            np_fields[i, j, F_FEAT_SELF_CONJ] = 1.0 if f_data.get('self_conjugate', False) else 0.0
            np_fields[i, j, F_FEAT_CHIRALITY] = CHIRALITY_MAP.get(f_data.get('chirality', 'none'), 0)
            for k, p_id_str in enumerate(f_data.get('particles', [])):
                if k >= MAX_PARTICLES_PER_FIELD: break
                if p_id_str in p_map: np_field_particle_map[i, j, k] = p_map[p_id_str]

        for j, itr_data in enumerate(model.get('interactions', [])):
            if j >= MAX_INTERACTIONS: continue
            np_interactions[i, j, I_FEAT_ID] = j
            np_interactions[i, j, I_FEAT_TYPE] = INTERACTION_TYPE_MAP.get(itr_data.get('type',''), -1)
            field_ids = [f_map.get(f_id_str, -1) for f_id_str in itr_data.get('fields', [])]
            if len(field_ids) >= 3:
                np_interactions[i, j, I_FEAT_F0] = field_ids[0]
                np_interactions[i, j, I_FEAT_F1] = field_ids[1]
                np_interactions[i, j, I_FEAT_F2] = field_ids[2]
    
    if debug:
        logger.debug("[Ingest] Finished creating tensors from model dictionaries.")

    return {
        'particles': torch.from_numpy(np_particles).to(DEVICE),
        'fields': torch.from_numpy(np_fields).to(DEVICE),
        'groups': torch.from_numpy(np_groups).to(DEVICE),
        'interactions': torch.from_numpy(np_interactions).to(DEVICE, dtype=ITYPE),
        'field_particle_map': torch.from_numpy(np_field_particle_map).to(DEVICE, dtype=ITYPE),
    }

# ======================================================================================
# Phase 2 & 3: Parallel Validator & Score Aggregator
# ======================================================================================
def tensor_popcount(tensor: torch.Tensor) -> torch.Tensor:
    """A compatible popcount implementation."""
    tensor = tensor.to(torch.int64)
    c = (tensor & 0x5555555555555555) + ((tensor >> 1) & 0x5555555555555555)
    c = (c & 0x3333333333333333) + ((c >> 2) & 0x3333333333333333)
    c = (c & 0x0F0F0F0F0F0F0F0F) + ((c >> 4) & 0x0F0F0F0F0F0F0F0F)
    c = (c & 0x00FF00FF00FF00FF) + ((c >> 8) & 0x00FF00FF00FF00FF)
    c = (c & 0x0000FFFF0000FFFF) + ((c >> 16) & 0x0000FFFF0000FFFF)
    c = (c & 0x00000000FFFFFFFF) + ((c >> 32) & 0x00000000FFFFFFFF)
    return c

def parallel_validator_and_scorer(tensors: dict, debug: bool = False) -> tuple[torch.Tensor, int]:
    """Takes a dictionary of tensors, performs all validation checks, and returns scores."""
    batch_size = tensors['particles'].shape[0]
    if debug:
        logger.debug(f"[Validator] Starting parallel validation for batch_size={batch_size}.")
        for name, tensor in tensors.items():
            logger.debug(f"[Validator]  - Input tensor '{name}' shape: {tensor.shape}")

    results_mask = torch.zeros((batch_size, 1), dtype=torch.int64, device=DEVICE)
    check_bit = 0

    def add_check_result(result_tensor: torch.Tensor):
        nonlocal check_bit, results_mask
        if check_bit >= 63: raise ValueError("Exceeded 63 checks.")
        result_tensor = result_tensor.to(device=DEVICE, dtype=torch.bool)
        bit_to_set = (1 << check_bit)
        update_tensor = result_tensor.to(torch.int64) * bit_to_set
        results_mask[:, 0] |= update_tensor
        check_bit += 1

    particles, fields, groups, interactions = tensors['particles'], tensors['fields'], tensors['groups'], tensors['interactions']
    field_particle_map = tensors['field_particle_map']

    valid_particles = particles[:, :, P_FEAT_ID] != PAD_VALUE
    valid_fields = fields[:, :, F_FEAT_ID] != PAD_VALUE
    valid_groups = groups[:, :, G_FEAT_ID] != PAD_VALUE
    valid_interactions = interactions[:, :, I_FEAT_ID] != PAD_VALUE

    add_check_result(((particles[:, :, P_FEAT_MASS] >= 0) | ~valid_particles).all(dim=1))
    add_check_result(((particles[:, :, P_FEAT_TYPE] != -1) | ~valid_particles).all(dim=1))
    add_check_result(((groups[:, :, G_FEAT_TYPE] != -1) | ~valid_groups).all(dim=1))
    is_su_group = (groups[:, :, G_FEAT_TYPE] == GROUP_TYPE_MAP['SU_2']) | (groups[:, :, G_FEAT_TYPE] == GROUP_TYPE_MAP['SU_3'])
    add_check_result((((groups[:, :, G_FEAT_DIM] == 2) | (groups[:, :, G_FEAT_DIM] == 3)) | ~is_su_group | ~valid_groups).all(dim=1))
    field_types = fields[:, :, F_FEAT_TYPE]
    add_check_result(((field_types != -1) | ~valid_fields).all(dim=1))
    field_dims = fields[:, :, F_FEAT_DIM]
    add_check_result(((field_dims > 0) | ~valid_fields).all(dim=1))
    field_gens = fields[:, :, F_FEAT_GEN]
    add_check_result(((field_gens > 0) | ~valid_fields).all(dim=1))
    add_check_result((((field_particle_map != PAD_VALUE).sum(dim=2) == field_dims * field_gens) | ~valid_fields).all(dim=1))
    is_fermion_field = (field_types == FIELD_TYPE_MAP['fermion'])
    chirality = fields[:, :, F_FEAT_CHIRALITY]
    add_check_result((((chirality == CHIRALITY_MAP['left']) | (chirality == CHIRALITY_MAP['right'])) | ~is_fermion_field | ~valid_fields).all(dim=1))
    add_check_result(((field_gens == 1) | is_fermion_field | ~valid_fields).all(dim=1))
    
    yukawa_mask = (interactions[:, :, I_FEAT_TYPE] == INTERACTION_TYPE_MAP['yukawa']) & valid_interactions
    f0_ids = interactions[:, :, I_FEAT_F0].clamp(min=0)
    f1_ids = interactions[:, :, I_FEAT_F1].clamp(min=0)
    f2_ids = interactions[:, :, I_FEAT_F2].clamp(min=0)
    batch_idx = torch.arange(batch_size, device=DEVICE).view(-1, 1)
    gens_f0, gens_f1 = field_gens[batch_idx, f0_ids], field_gens[batch_idx, f1_ids]
    add_check_result(((gens_f0 == gens_f1) | ~yukawa_mask).all(dim=1))
    dims_f0, dims_f1, dims_f2 = field_dims[batch_idx, f0_ids], field_dims[batch_idx, f1_ids], field_dims[batch_idx, f2_ids]
    add_check_result((((dims_f2 == dims_f0) | (dims_f2 == dims_f1)) | ~yukawa_mask).all(dim=1))
    
    B, MAX_F, MAX_P_PER_F = field_particle_map.shape
    batch_idx_3d = batch_idx.view(B, 1, 1)
    f_p_map_clamped = field_particle_map.clamp(min=0)
    p_types_in_fields = particles[batch_idx_3d, f_p_map_clamped, P_FEAT_TYPE]
    f_types_expanded = field_types.unsqueeze(2).expand(B, MAX_F, MAX_P_PER_F)
    is_scalar_field = (f_types_expanded == FIELD_TYPE_MAP['real']) | (f_types_expanded == FIELD_TYPE_MAP['complex']) | (f_types_expanded == FIELD_TYPE_MAP['scalar'])
    is_scalar_particle = (p_types_in_fields == PARTICLE_TYPE_MAP['real']) | (p_types_in_fields == PARTICLE_TYPE_MAP['complex'])
    type_match_ok = (is_scalar_field & is_scalar_particle) | ((f_types_expanded == FIELD_TYPE_MAP['fermion']) & (p_types_in_fields == PARTICLE_TYPE_MAP['fermion']))
    add_check_result(((type_match_ok | ~(field_particle_map != PAD_VALUE)).all(dim=2) | ~valid_fields).all(dim=1))

    is_self_conj = (fields[:, :, F_FEAT_SELF_CONJ] == 1.0)
    p_charges_in_fields = particles[batch_idx_3d, f_p_map_clamped, P_FEAT_CHARGE]
    add_check_result((((p_charges_in_fields == 0) | ~(field_particle_map != PAD_VALUE)).all(dim=2) | ~is_self_conj | ~valid_fields).all(dim=1))

    field_is_massive = (particles[batch_idx_3d, f_p_map_clamped, P_FEAT_MASS] > 0).any(dim=2)
    massive_fermion_field = is_fermion_field & field_is_massive & valid_fields
    yukawa_fermion_mask = torch.zeros_like(valid_fields)
    yukawa_f0_indices, yukawa_f1_indices = f0_ids[yukawa_mask], f1_ids[yukawa_mask]
    batch_yukawa_idx = batch_idx.expand_as(interactions[:,:,I_FEAT_ID])[yukawa_mask]
    if batch_yukawa_idx.numel() > 0:
        yukawa_fermion_mask[batch_yukawa_idx, yukawa_f0_indices] = True
        yukawa_fermion_mask[batch_yukawa_idx, yukawa_f1_indices] = True
    add_check_result((~massive_fermion_field | yukawa_fermion_mask | ~valid_fields).all(dim=1))

    final_scores = tensor_popcount(results_mask[:, 0])
    if debug:
        logger.debug(f"[Validator] Validation finished. Total checks: {check_bit}. Final scores shape: {final_scores.shape}")
    return final_scores, check_bit