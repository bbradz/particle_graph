"""
Implements the multi-target value system with discounted future return and 
supervised error loss signal separation.
"""
import torch
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Set
import re
import time

from .environment import EnvOutcome
from config import Config
from Token2Model.check import CHECK_TO_IDX, NUM_CHECKS, IDX_TO_CHECK

@dataclass
class ShapedRewards:
    # Target for the value network: discounted cumulative reward for each check
    per_token_value_targets: torch.FloatTensor  # Shape: (T, C)
    # The raw, single-step reward assigned to each token
    per_token_instantaneous_rewards: torch.FloatTensor # Shape: (T, C)
    # Binary mask indicating where to apply the supervised zero-push loss
    error_signal_mask: torch.BoolTensor          # Shape: (T, C)
    # Total scalar reward (for PPO advantage calculation)
    scalar_total_reward: float
    # Diagnostics for logging
    diagnostics: Dict[str, Any]


class RewardShaper:
    def __init__(self, cfg: Config, policy_dtype: torch.dtype):
        self.cfg = cfg
        self.policy_dtype = policy_dtype
        # Pre-process mapping for parser compatibility
        self.path_prefix_map = {
            'particles': 'particle', 'fields': 'field',
            'interactions': 'interaction', 'global': 'global'
        }

    def _parse_attr_path(self, var_path: str) -> Optional[Tuple[str, str, str]]:
        """Maps checklist paths like 'fields.m1.dim' to a token_map key."""
        match = re.match(r'(particles|fields|interactions)\.([a-zA-Z0-9_]+)\.([\w_]+(?:\.[\w_]+)*)', var_path)
        if match:
            obj_type_prefix, obj_id, attr_path = match.groups()
            obj_type = self.path_prefix_map.get(obj_type_prefix, None)
            if obj_type:
                return (obj_type, obj_id, attr_path)
        return None

    def _aggregate_check_data(self, checklist: Dict[str, Dict[str, Any]], token_map: Dict[Tuple, List[int]], seq_len: int) -> Tuple[torch.FloatTensor, Dict[int, Set[int]], Dict[int, Set[int]], Dict[int, Set[int]]]:
        """
        Aggregates check results, calculates R_Success, and creates token index sets.
        """
        raw_scores: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(NUM_CHECKS)}
        sets_good: Dict[int, Set[int]] = {i: set() for i in range(NUM_CHECKS)}
        sets_error: Dict[int, Set[int]] = {i: set() for i in range(NUM_CHECKS)}
        sets_block: Dict[int, Set[int]] = {i: set() for i in range(NUM_CHECKS)}

        for obj_id, checks in checklist.items():
            for check_name, result in checks.items():
                if check_name not in CHECK_TO_IDX: continue
                check_idx = CHECK_TO_IDX[check_name]
                
                raw_scores[check_idx].append((result.get('score', 0.0), result.get('max_score', 1.0)))
                
                if result.get('message') not in ["Skipped", "CRASHED"]:
                    for var_path in result.get('error_var', []) or []:
                        parsed = self._parse_attr_path(var_path)
                        if parsed:
                            sets_error[check_idx].update(token_map.get(parsed, []))
                        
                    for var_path in result.get('good_var', []) or []:
                        parsed = self._parse_attr_path(var_path)
                        if parsed:
                            sets_good[check_idx].update(token_map.get(parsed, []))

                obj_type = self.path_prefix_map.get(obj_id[0], 'global') if obj_id and obj_id[0] in self.path_prefix_map else 'global'
                if obj_type != 'global':
                    block_key = (obj_type, obj_id, 'block_span')
                    sets_block[check_idx].update(token_map.get(block_key, []))
                else:
                    sets_block[check_idx].update(range(seq_len))

        r_success_tensor = torch.zeros(NUM_CHECKS, dtype=torch.float)
        for check_idx, scores_list in raw_scores.items():
            if scores_list:
                total_score = sum(s for s, m in scores_list)
                total_max_score = sum(m for s, m in scores_list)
                if total_max_score > 0:
                    r_success_tensor[check_idx] = total_score / total_max_score

        return r_success_tensor, sets_good, sets_error, sets_block


    def calculate_rewards(self, outcome: EnvOutcome, sequence_tensor: torch.Tensor) -> ShapedRewards:
        start_total = time.perf_counter()
        device = sequence_tensor.device
        seq_len = sequence_tensor.size(0)
        reward_timing: Dict[str, float] = {}

        # [TIME] Reward Shaping: Aggregate Check Data
        start_aggregate = time.perf_counter()
        r_success_tensor, sets_good, sets_error, sets_block = self._aggregate_check_data(
            outcome.checklist, outcome.token_map, seq_len
        )
        reward_timing['aggregate_data_time'] = time.perf_counter() - start_aggregate
        r_success_tensor = r_success_tensor.to(device, dtype=self.policy_dtype)

        # 2. Calculate Instantaneous Rewards
        start_instantaneous = time.perf_counter()
        r_instantaneous_matrix = torch.zeros((seq_len, NUM_CHECKS), device=device, dtype=self.policy_dtype)
        
        for check_idx in range(NUM_CHECKS):
            r_success = r_success_tensor[check_idx].item()
            if r_success <= 0: continue
            
            T_credit = sets_good[check_idx] if sets_good[check_idx] else sets_block[check_idx]
            N_credit = len(T_credit)
            
            if N_credit > 0:
                r_base_c = r_success / N_credit
                token_indices = torch.tensor(list(T_credit), device=device, dtype=torch.long)
                if token_indices.numel() > 0:
                    valid_indices = token_indices[token_indices < seq_len]
                    source_tensor = torch.full((len(valid_indices),), r_base_c, dtype=self.policy_dtype, device=device)
                    update_matrix = torch.zeros_like(r_instantaneous_matrix)
                    update_matrix[valid_indices, check_idx] = source_tensor
                    r_instantaneous_matrix += update_matrix
        reward_timing['instantaneous_reward_time'] = time.perf_counter() - start_instantaneous

        # [TIME] Reward Shaping: Discounted Return Loop
        # 3. Discounted Return main loop
        start_discounted = time.perf_counter()
        gamma = self.cfg.GAMMA
        # Vectorized discounted cumulative sum over time dimension using reverse-cumsum trick
        # R[t] = r[t] + gamma * R[t+1]
        # Let d[t] = gamma^t. Then R = flip(cumsum(flip(r * d), dim=0), dim=0) / d
        time_idx = torch.arange(seq_len, device=device, dtype=r_instantaneous_matrix.dtype)
        discount_powers = torch.pow(torch.as_tensor(gamma, dtype=r_instantaneous_matrix.dtype, device=device), time_idx)
        weighted = r_instantaneous_matrix * discount_powers.view(-1, 1)
        reversed_weighted = torch.flip(weighted, dims=[0])
        reversed_cumsum = torch.cumsum(reversed_weighted, dim=0)
        R_target_matrix = torch.flip(reversed_cumsum, dims=[0]) / discount_powers.view(-1, 1)
        reward_timing['discounted_return_loop_time'] = time.perf_counter() - start_discounted

        # [TIME] Reward Shaping: Misc Scalar/Mask Calculation
        # 4. Misc scalar and mask calculation
        start_misc_calc = time.perf_counter()
        scalar_total_reward = float(R_target_matrix[0].sum().item())
        
        diagnostics: Dict[str, Any] = {"checks": outcome.checklist, "structural_penalty": 0.0, "eos_bonus": 0.0, "r_success_tensor": r_success_tensor}
        
        if outcome.unclosed_block_info:
            _, depth = outcome.unclosed_block_info
            penalty = self.cfg.UNCLOSED_BLOCK_PENALTY_PER_DEPTH * depth
            diagnostics['structural_penalty'] = penalty
            scalar_total_reward += penalty

        if (sequence_tensor == self.cfg.EOS_TOKEN_ID).any():
            diagnostics['eos_bonus'] = self.cfg.EOS_BONUS
            scalar_total_reward += self.cfg.EOS_BONUS
            
        error_signal_mask = torch.zeros((seq_len, NUM_CHECKS), dtype=torch.bool, device=device)
        for check_idx in range(NUM_CHECKS):
            if sets_error[check_idx]:
                token_indices = torch.tensor(list(sets_error[check_idx]), device=device, dtype=torch.long)
                if token_indices.numel() > 0:
                    valid_indices = token_indices[token_indices < seq_len]
                    error_signal_mask[valid_indices, check_idx] = True
        
        R_target_matrix.clamp_(min=0.0, max=1.0)
        reward_timing['misc_scalar_mask_time'] = time.perf_counter() - start_misc_calc
        reward_timing['reward_total_time'] = time.perf_counter() - start_total
        diagnostics['timing'] = reward_timing

        return ShapedRewards(
            per_token_value_targets=R_target_matrix,
            per_token_instantaneous_rewards=r_instantaneous_matrix,
            error_signal_mask=error_signal_mask,
            scalar_total_reward=scalar_total_reward,
            diagnostics=diagnostics
        )