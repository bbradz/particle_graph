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
    # Binary target indicating which tokens were responsible for a check's success
    per_token_criticality_targets: torch.FloatTensor # Shape: (T, C)
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
        # Debug flag (configurable)
        self.DEBUG_REWARD = getattr(cfg, 'PRINT_REWARD_DEBUG', False)

    def _parse_attr_path(self, var_path: str) -> Optional[Tuple[str, str, str]]:
        """Maps checklist paths like 'fields.m1.dim' to a token_map key."""
        match = re.match(r'(particles|fields|interactions)\.([a-zA-Z0-9_]+)\.([\w_]+(?:\.[\w_]+)*)', var_path)
        if match:
            obj_type_prefix, obj_id, attr_path = match.groups()
            obj_type = self.path_prefix_map.get(obj_type_prefix, None)
            if obj_type:
                # ADDED: Handle abstract list attributes by mapping to the whole block
                if attr_path == 'fields' or attr_path == 'particles':
                    if self.DEBUG_REWARD:
                        print(f"      - [REWARD_DEBUG] Remapping abstract path '{attr_path}' to 'block_span'")
                    attr_path = 'block_span'
                return (obj_type, obj_id, attr_path)
        return None

    def _aggregate_check_data(self, checklist: Dict[str, Dict[str, Any]], token_map: Dict[Tuple, List[int]], token_strs: List[str], seq_len: int) -> Tuple[torch.FloatTensor, Dict[int, Set[int]], Dict[int, Set[int]], Dict[int, Set[int]], Dict[int, Set[int]]]:
        """
        Aggregates check results, calculates R_Success, and creates token index sets.
        """
        if self.DEBUG_REWARD:
            print("\n[REWARD_DEBUG] --- Aggregating Check Data for Criticality ---")
            # Mirror run_checks-style visibility for particle checks so stdout shows them reliably
            try:
                particle_objs = {obj_id: checks for obj_id, checks in checklist.items() if isinstance(obj_id, str) and obj_id.startswith('f')}
                total_particle_checks = sum(len(chks) for chks in particle_objs.values())
                print(f"[DEBUG_CHECKS] Starting run_checks for {total_particle_checks} checks.")
                for obj_id, checks in particle_objs.items():
                    for ck in checks.keys():
                        print(f"--- Running check: {ck} (Object: {obj_id}) ---")
                print("[DEBUG_CHECKS] Finished run_checks.\n")
            except Exception:
                pass
            
        raw_scores: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(NUM_CHECKS)}
        sets_good: Dict[int, Set[int]] = {i: set() for i in range(NUM_CHECKS)}
        sets_error: Dict[int, Set[int]] = {i: set() for i in range(NUM_CHECKS)}
        sets_block: Dict[int, Set[int]] = {i: set() for i in range(NUM_CHECKS)}
        sets_mattered: Dict[int, Set[int]] = {i: set() for i in range(NUM_CHECKS)}

        for obj_id, checks in checklist.items():
            for check_name, result in checks.items():
                if check_name not in CHECK_TO_IDX: continue
                check_idx = CHECK_TO_IDX[check_name]
                
                if self.DEBUG_REWARD:
                    print(f"\n--- Processing Check: '{check_name}' (Index: {check_idx}) ---")
                
                raw_scores[check_idx].append((result.get('score', 0.0), result.get('max_score', 1.0)))
                
                # --- START MODIFICATION ---
                # This block now ONLY handles good/error vars
                if result.get('message') not in ["Skipped", "CRASHED"]:
                    if self.DEBUG_REWARD:
                        print(f"  - mattered_vars: {result.get('mattered_vars', [])}")
                        print(f"  - good_var:      {result.get('good_var', [])}")
                        print(f"  - error_var:     {result.get('error_var', [])}")
                    for var_path in result.get('error_var', []) or []:
                        parsed = self._parse_attr_path(var_path)
                        if parsed:
                            mapped = token_map.get(parsed, [])
                            sets_error[check_idx].update(mapped)
                            if self.DEBUG_REWARD and mapped:
                                print(f"  - error_var '{var_path}' -> indices: {mapped}")
                        
                    for var_path in result.get('good_var', []) or []:
                        parsed = self._parse_attr_path(var_path)
                        if parsed:
                            mapped = token_map.get(parsed, [])
                            sets_good[check_idx].update(mapped)
                            if self.DEBUG_REWARD and mapped:
                                print(f"  - good_var  '{var_path}' -> indices: {mapped}")

                # MOVED: Process structural relevance (mattered_vars) for ALL checks, regardless of status.
                mattered_vars = result.get('mattered_vars', []) or []
                if self.DEBUG_REWARD and mattered_vars:
                    print(f"  - Found 'mattered_vars': {mattered_vars}")
                    
                for var_path in mattered_vars:
                    parsed = self._parse_attr_path(var_path)
                    if self.DEBUG_REWARD:
                        print(f"    - Parsing '{var_path}' -> {parsed}")
                        
                    if parsed:
                        indices = token_map.get(parsed, [])
                        # Fallback: if no fine-grained mapping exists (e.g., mass not recorded),
                        # fall back to the object's block span so the target still lights up.
                        if not indices:
                            fallback_key = (parsed[0], parsed[1], 'block_span')
                            indices = token_map.get(fallback_key, [])
                            if self.DEBUG_REWARD and indices:
                                print(f"      - Fallback to block_span for {parsed}: {indices}")

                        # EXTRA FALLBACKS for particle mass mapping
                        if (not indices) and parsed[0] == 'particle' and parsed[2] == 'mass' and token_strs:
                            # 1) Search within the block_span for MASS_*
                            block_indices = token_map.get(('particle', parsed[1], 'block_span'), [])
                            if block_indices:
                                for idx in block_indices:
                                    if 0 <= idx < len(token_strs) and isinstance(token_strs[idx], str) and token_strs[idx].startswith('MASS_'):
                                        indices = [idx]
                                        if self.DEBUG_REWARD:
                                            print(f"      - MASS fallback found at index {idx} within block_span for particle '{parsed[1]}'")
                                        break
                            # 2) If no block_span, derive from PARTICLE_ID_N .. END_PARTICLE window
                            if not indices:
                                # Parsed id like 'f3' -> numeric '3'
                                try:
                                    numeric_id = int(parsed[1][1:])
                                except Exception:
                                    numeric_id = None
                                if numeric_id is not None:
                                    id_tok = f"PARTICLE_ID_{numeric_id}"
                                    try:
                                        start_idx = token_strs.index(id_tok)
                                        # find the next END_PARTICLE after start_idx
                                        end_idx = None
                                        for j in range(start_idx, len(token_strs)):
                                            if token_strs[j] == 'END_PARTICLE':
                                                end_idx = j
                                                break
                                        if end_idx is None:
                                            end_idx = min(start_idx + 10, len(token_strs))
                                        # scan window for MASS_*
                                        for j in range(start_idx, end_idx + 1):
                                            if token_strs[j].startswith('MASS_'):
                                                indices = [j]
                                                if self.DEBUG_REWARD:
                                                    print(f"      - MASS fallback via id window found at index {j} for particle '{parsed[1]}'")
                                                break
                                    except ValueError:
                                        pass

                        sets_mattered[check_idx].update(indices)
                        if self.DEBUG_REWARD and indices:
                            print(f"      - Mapped to token indices: {indices}")

                # --- END MODIFICATION ---

                # Infer object type from object id prefix (f -> particle, m -> field, i -> interaction)
                inferred_prefix_map = {'f': 'particle', 'm': 'field', 'i': 'interaction'}
                obj_type = inferred_prefix_map.get(obj_id[0], 'global') if obj_id else 'global'
                if obj_type != 'global':
                    block_key = (obj_type, obj_id, 'block_span')
                    block_indices = token_map.get(block_key, [])
                    sets_block[check_idx].update(block_indices)
                    if self.DEBUG_REWARD and block_indices:
                        print(f"  - block_span {block_key} -> indices: {block_indices}")
                    # If block-level and no explicit mattered vars, treat block as mattered
                    if not result.get('mattered_vars') and result.get('level') == 'block':
                        sets_mattered[check_idx].update(token_map.get(block_key, []))
                else:
                    sets_block[check_idx].update(range(seq_len))
                    # Global-level default to entire sequence if no explicit mattered vars
                    if not result.get('mattered_vars') and result.get('level') == 'global':
                        sets_mattered[check_idx].update(range(seq_len))
                        
                if self.DEBUG_REWARD:
                    print(f"  - Final 'sets_mattered' for this check: {sorted(list(sets_mattered[check_idx]))}")

                # Ensure particle checks are clearly visible in stdout
                if self.DEBUG_REWARD and check_name in {'_mass_check', '_type_check', '_name_check', '_charge_check'}:
                    print(f"[REWARD_DEBUG] Particle check '{check_name}' on '{obj_id}':")
                    print(f"    mattered_vars: {result.get('mattered_vars', [])}")
                    print(f"    good_var:      {result.get('good_var', [])}")
                    print(f"    error_var:     {result.get('error_var', [])}")
                    print(f"    mapped_mattered_indices: {sorted(list(sets_mattered[check_idx]))}")
                    print(f"    mapped_good_indices:     {sorted(list(sets_good[check_idx]))}")
                    print(f"    block_span_indices:      {sorted(list(sets_block[check_idx]))}")

        r_success_tensor = torch.zeros(NUM_CHECKS, dtype=torch.float)
        for check_idx, scores_list in raw_scores.items():
            if scores_list:
                total_score = sum(s for s, m in scores_list)
                total_max_score = sum(m for s, m in scores_list)
                if total_max_score > 0:
                    r_success_tensor[check_idx] = total_score / total_max_score
        
        if self.DEBUG_REWARD:
            print("\n[REWARD_DEBUG] --- Finished Aggregating Check Data ---")

        return r_success_tensor, sets_good, sets_error, sets_block, sets_mattered


    def calculate_rewards(self, outcome: EnvOutcome, sequence_tensor: torch.Tensor) -> ShapedRewards:
        start_total = time.perf_counter()
        device = sequence_tensor.device
        seq_len = sequence_tensor.size(0)
        reward_timing: Dict[str, float] = {}

        # [TIME] Reward Shaping: Aggregate Check Data
        start_aggregate = time.perf_counter()
        r_success_tensor, sets_good, sets_error, sets_block, sets_mattered = self._aggregate_check_data(
            outcome.checklist, outcome.token_map, outcome.token_strs or [], seq_len
        )
        reward_timing['aggregate_data_time'] = time.perf_counter() - start_aggregate
        r_success_tensor = r_success_tensor.to(device, dtype=self.policy_dtype)

        # 2. Calculate Instantaneous Rewards (assign only to explicitly good tokens)
        start_instantaneous = time.perf_counter()
        r_instantaneous_matrix = torch.zeros((seq_len, NUM_CHECKS), device=device, dtype=self.policy_dtype)
        
        for check_idx in range(NUM_CHECKS):
            r_success = r_success_tensor[check_idx].item()
            if r_success <= 0: continue
            
            T_credit = sets_good[check_idx]
            N_credit = len(T_credit)
            
            if N_credit > 0:
                r_base_c = r_success / N_credit
                token_indices = torch.tensor(list(T_credit), device=device, dtype=torch.long)
                if token_indices.numel() > 0:
                    valid_indices = token_indices[token_indices < seq_len]
                    source_tensor = torch.full((len(valid_indices),), r_base_c, dtype=self.policy_dtype, device=device)
                    r_instantaneous_matrix.index_put_((valid_indices, torch.full_like(valid_indices, check_idx)), source_tensor, accumulate=True)
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

        # Binary criticality targets from explicit mattered tokens
        per_token_criticality_targets = torch.zeros((seq_len, NUM_CHECKS), device=device, dtype=self.policy_dtype)
        for check_idx in range(NUM_CHECKS):
            token_indices_src = sets_mattered[check_idx]

            if token_indices_src:
                token_indices = torch.tensor(list(token_indices_src), device=device, dtype=torch.long)
                valid_indices = token_indices[token_indices < seq_len]
                if valid_indices.numel() > 0:
                    per_token_criticality_targets[valid_indices, check_idx] = 1.0
        
        # ADDED: Print the final target vector for each check
        if self.DEBUG_REWARD:
            print("\n[REWARD_DEBUG] --- Final Criticality Target Vectors ---")
            for check_idx in range(NUM_CHECKS):
                target_vector = per_token_criticality_targets[:, check_idx]
                critical_indices = (target_vector > 0.5).nonzero(as_tuple=True)[0].tolist()
                if critical_indices:
                    check_name = IDX_TO_CHECK.get(check_idx, f"Check_{check_idx}")
                    print(f"  - Check '{check_name}': Critical token indices are {critical_indices}")
            print("[REWARD_DEBUG] ----------------------------------------\n")

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
        
        # Add length penalty to encourage concise sequences
        length_penalty = self.cfg.LENGTH_PENALTY_PER_TOKEN * seq_len
        diagnostics['length_penalty'] = length_penalty
        scalar_total_reward += length_penalty
            
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
            per_token_criticality_targets=per_token_criticality_targets,
            diagnostics=diagnostics
        )