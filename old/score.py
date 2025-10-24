# gpu_validator_final_robust.py
# Description: This final version fixes a crash during serialization and correctly
# replicates the partial scoring mechanism of the non-GPU script.

import json
import time
from typing import Dict, Any, List
import numpy as np
import torch
from sympy import Matrix, sympify
from fractions import Fraction

# ====================================================================
# 📜 MODEL DEFINITIONS & CONFIG (No changes here)
# ====================================================================
sm_model_data = {
    "GaugeGroups": [
        {"id": "g1", "name": "U1Y", "charge": "Y", "group": "U_1", "coupling": "g1", "boson": "B"},
        {"id": "g2", "name": "SU2L", "charge": "I", "group": "SU_2", "coupling": "gw", "boson": "W"},
        {"id": "g3", "name": "SU3C", "charge": "C", "group": "SU_3", "coupling": "gs", "boson": "G"}
    ],
    "particles": [
        {"id": "e", "type": "fermion", "name": "e", "mass": 0.000511, "charge": -3},{"id": "mu", "type": "fermion", "name": "mu", "mass": 0.1057, "charge": -3},
        {"id": "tau", "type": "fermion", "name": "tau", "mass": 1.777, "charge": -3},{"id": "ve", "type": "fermion", "name": "ve", "mass": 0.0, "charge": 0},
        {"id": "vm", "type": "fermion", "name": "vm", "mass": 0.0, "charge": 0},{"id": "vt", "type": "fermion", "name": "vt", "mass": 0.0, "charge": 0},
        {"id": "u", "type": "fermion", "name": "u", "mass": 0.0023, "charge": 2},{"id": "d", "type": "fermion", "name": "d", "mass": 0.0047, "charge": -1},
        {"id": "c", "type": "fermion", "name": "c", "mass": 1.27, "charge": 2},{"id": "s", "type": "fermion", "name": "s", "mass": 0.095, "charge": -1},
        {"id": "t", "type": "fermion", "name": "t", "mass": 172.76, "charge": 2},{"id": "b", "type": "fermion", "name": "b", "mass": 4.18, "charge": -1},
        {"id": "phi_p", "type": "complex", "name": "phip", "mass": 125.0, "charge": 3},{"id": "phi_0", "type": "complex", "name": "phi0", "mass": 125.0, "charge": 0}
    ], "fields": [
        {"id": "LL", "name": "LL", "type": "fermion", "dim": 2, "gen": 3, "chirality": "left", "reps": {"g1": -1, "g2": "fnd", "g3": "singlet"}, "particles": ["ve", "e", "vm", "mu", "vt", "tau"]},
        {"id": "eR", "name": "eR", "type": "fermion", "dim": 1, "gen": 3, "chirality": "right", "reps": {"g1": -2, "g2": "singlet", "g3": "singlet"}, "particles": ["e", "mu", "tau"]},
        {"id": "QL", "name": "QL", "type": "fermion", "dim": 2, "gen": 3, "chirality": "left", "reps": {"g1": 0.3333333, "g2": "fnd", "g3": "fnd"}, "particles": ["u", "d", "c", "s", "t", "b"]},
        {"id": "uR", "name": "uR", "type": "fermion", "dim": 1, "gen": 3, "chirality": "right", "reps": {"g1": 1.3333333, "g2": "singlet", "g3": "fnd"}, "particles": ["u", "c", "t"]},
        {"id": "dR", "name": "dR", "type": "fermion", "dim": 1, "gen": 3, "chirality": "right", "reps": {"g1": -0.6666667, "g2": "singlet", "g3": "fnd"}, "particles": ["d", "s", "b"]},
        {"id": "Phi", "name": "Phi", "type": "complex", "dim": 2, "gen": 1, "chirality": "none", "reps": {"g1": 1, "g2": "fnd", "g3": "singlet"}, "particles": ["phi_p", "phi_0"]}
    ], "interactions": [
        {"id": "Yukawa_e", "type": "yukawa", "fields": ["LL", "eR", "Phi"]},{"id": "Yukawa_u", "type": "yukawa", "fields": ["QL", "uR", "Phi"]},
        {"id": "Yukawa_d", "type": "yukawa", "fields": ["QL", "dR", "Phi"]}
    ]
}
failing_model_data = {
    "GaugeGroups": [{"id": "g1", "name": "U1X", "charge": "X", "group": "U_1", "coupling": "gX", "boson": "X"}],
    "particles": [
        {"id": "psi1", "type": "fermion", "name": "psi1", "mass": 100.0, "charge": 0},{"id": "psi2", "type": "fermion", "name": "psi2", "mass": 110.0, "charge": 0},
        {"id": "chi", "type": "fermion", "name": "chi", "mass": 105.0, "charge": 0},{"id": "phi", "type": "complex", "name": "phi", "mass": 200.0, "charge": 0},
    ], "fields": [
        {"id": "PsiL", "name": "PsiL", "type": "fermion", "dim": 1, "gen": 2, "chirality": "left", "reps": {"g1": 1}, "particles": ["psi1", "psi2"]},
        {"id": "ChiR", "name": "ChiR", "type": "fermion", "dim": 1, "gen": 1, "chirality": "right", "reps": {"g1": 2}, "particles": ["chi"]},
        {"id": "Phi", "name": "Phi", "type": "complex", "dim": 1, "gen": 1, "reps": {"g1": -1}, "particles": ["phi"]}, # This field has no chirality key
    ], "interactions": [{"id": "Yukawa_fail", "type": "yukawa", "fields": ["PsiL", "ChiR", "Phi"]}],
}
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

# ====================================================================
# SERIALIZER & BATCH VALIDATOR
# ====================================================================
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
            ### FINAL FIX ###
            # The default value for the chirality lookup must be an integer (0), not a string ("none").
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

# ====================================================================
# GPUValidator - WITH FINAL CORRECTED PARTIAL SCORING
# ====================================================================
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

# ====================================================================
# MAIN EXECUTION BLOCK
# ====================================================================
if __name__ == '__main__':
    print("="*65)
    print("GPU Validator with Truly Identical Partial Scoring (Robust)")
    print("="*65)
    
    EXPECTED_SM_SCORE = 78
    EXPECTED_FAILING_SCORE = 31

    model_batch = [sm_model_data, failing_model_data, sm_model_data]
    model_names = ["Standard Model (Valid)", "Failing Model (Gen Mismatch)", "Standard Model Copy (Valid)"]

    batch_validator = BatchValidator()
    score_vector = batch_validator.validate_batch(model_batch)

    print("\n" + "="*65)
    print("Final Batch Validation Results (Identical Scoring)")
    print("="*65)
    print(f"Target score for Standard Model: {EXPECTED_SM_SCORE}")
    print(f"Target score for Failing Model: {EXPECTED_FAILING_SCORE}")
    print("-" * 35)
    for name, score in zip(model_names, score_vector):
        is_pass = (name.startswith("Standard") and score == EXPECTED_SM_SCORE) or \
                  (name.startswith("Failing") and score == EXPECTED_FAILING_SCORE)
        
        status = "✅ PASSED" if is_pass else "❌ FAILED"
        
        print(f"Model: {name:<30} | Score: {score:<5} | Status: {status}")
    print("="*65)