import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Dict, Tuple, Optional
import time 
import gc 
import psutil 
import os 
import numpy as np
import random
import multiprocessing as mp

from config import Config, get_config
from rl.network import HuggingFacePolicy, SimpleTransformerPolicy
from rl.grammar import GrammarMasker
from rl.environment import ParticlePhysicsEnvironment, EnvOutcome
from rl.reward import RewardShaper, ShapedRewards
from rl.curriculum import CurriculumManager
from rl.logging import log_batch_summary
from collections import defaultdict
from rl.replay_buffer import ReplayBuffer
from rl.plotting import TrainingPlotter
from Token2Model.check import NUM_CHECKS, IDX_TO_CHECK, CHECK_TO_IDX

DEBUG = get_config().DEBUG_PRINTS

def cpu_worker_process(task_queue, result_queue, config, idx_to_token):
    """
    A long-running process that consumes a sequence, performs all environment
    and reward calculations, and produces a dictionary of simple, pickle-able results.
    """
    # Each worker initializes its own environment and reward shaper.
    # This is the core of the solution.
    env = ParticlePhysicsEnvironment(config)
    reward_shaper = RewardShaper(config, torch.float32)

    while True:
        # Get a task from the queue
        i, sequence_list = task_queue.get()

        if sequence_list is None:  # Sentinel for termination
            break

        # Collect timing information
        import time
        timing_info = {}
        
        # 1. Environment processing timing
        start_env = time.perf_counter()
        episode_outcome = env.run_episode(sequence_list, idx_to_token)
        timing_info['env_total'] = time.perf_counter() - start_env
        
        # Extract individual environment timings if available
        if hasattr(episode_outcome, 'timing_info'):
            timing_info.update(episode_outcome.timing_info)
        
        # 2. Reward calculation timing
        start_reward = time.perf_counter()
        sequence_tensor = torch.tensor(sequence_list, dtype=torch.long)
        reward_object = reward_shaper.calculate_rewards(episode_outcome, sequence_tensor)
        timing_info['reward_total'] = time.perf_counter() - start_reward
        
        # Extract individual reward timings if available
        if hasattr(reward_object, 'timing_info'):
            timing_info.update(reward_object.timing_info)
        
        # 2. Extract only the pickle-able results into a dictionary
        result_payload = {
            # Tensors needed for PPO update
            "per_token_value_targets": reward_object.per_token_value_targets.cpu(),
            "per_token_criticality_targets": reward_object.per_token_criticality_targets.cpu(),
            "error_signal_mask": reward_object.error_signal_mask.cpu(),
            "scalar_total_reward": reward_object.scalar_total_reward,
            "per_token_instantaneous_rewards": reward_object.per_token_instantaneous_rewards.cpu(),
            
            # Simplified outcome data for logging
            "outcome": {
                "checklist": episode_outcome.checklist,
                "unclosed_block_info": episode_outcome.unclosed_block_info,
                "token_strs": episode_outcome.token_strs,
                "success": episode_outcome.success,
                "meta": episode_outcome.meta
            },
            
            # Timing information
            "timing_info": timing_info,
            
            # Other diagnostics from reward shaper if needed
            "diagnostics": {
                "r_success_tensor": reward_object.diagnostics.get('r_success_tensor', torch.zeros(NUM_CHECKS)).cpu(),
                "checks": reward_object.diagnostics.get('checks', {})
            }
        }

        # 3. Put the simple dictionary back into the result queue
        result_queue.put((i, result_payload))

class RLTrainer:
    def __init__(
        self,
        cfg: Config,
        policy,
        tokenizer,
        token_maps: Tuple[Dict[str, int], Dict[int, str], int],
    ):
        self.cfg = cfg
        self.policy = policy
        self.tokenizer = tokenizer
        self.token_to_idx, self.idx_to_token, self.vocab_size = token_maps[0], token_maps[1], token_maps[2]
        
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.LEARNING_RATE)
        
        if hasattr(policy, 'model'):
            self.device = policy.model.device
        else:
            self.device = policy.device
        
        self.policy_dtype = next(policy.parameters()).dtype

        self.grammar = GrammarMasker(cfg, self.token_to_idx, self.idx_to_token, self.vocab_size)
        # self.env = ParticlePhysicsEnvironment(cfg) # <-- REMOVE this line from init
        self.curriculum = CurriculumManager(cfg)
        
        # self.reward_shaper = RewardShaper(cfg, self.policy_dtype) # <-- REMOVE this line from init
        
        self.step_count = 0
        self.replay_buffer = ReplayBuffer(max_size=500, policy_dtype=self.policy_dtype)


        # Use a reasonable number of workers based on SLURM allocation
        # SLURM allocates 16 cores (-c 16), so use 14 workers to leave cores for main process
        num_workers = min(14, max(1, mp.cpu_count() - 2))
        print(f"Spawning {num_workers} CPU workers for environment processing.")
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.workers = []
        
        for _ in range(num_workers):
            # Pass only the necessary data to the worker
            p = mp.Process(target=cpu_worker_process, 
                           args=(self.task_queue, self.result_queue, self.cfg, self.idx_to_token))
            p.daemon = True # Allows main process to exit even if workers are running
            p.start()
            self.workers.append(p)
        
        # Initialize plotting functionality
        plot_output_dir = cfg.PLOT_OUTPUT_DIR
        self.plotter = TrainingPlotter(plot_output_dir)
        
        # Initialize history dictionary for collecting metrics
        self.history = {
            'step': [],
            'ppo_loss_start': [],
            'ppo_loss_end': [],
            'ppo_loss_reduction': [],
            'avg_reward': [],
            'policy_loss': [],
            'value_loss': [],
            'reward_loss': [],
            'criticality_loss': [],
            'consistency_loss': [],
            'eos_percentage': [],
            'avg_jaccard_dist': [],
            'bigram_entropy': [],
            'replay_original_loss': [],
            'replay_current_loss': [],
            'replay_improvement': []
        }
        
        # Initialize per-check metrics
        for check_idx in range(NUM_CHECKS):
            check_name = IDX_TO_CHECK.get(check_idx, f"check_{check_idx}")
            self.history[f'check_loss_{check_name}'] = []
            self.history[f'check_score_{check_name}'] = []
            self.history[f'value_mean_{check_name}'] = []
            self.history[f'value_var_{check_name}'] = []
        self.replay_buffer.policy_device = self.device # Set the device for sampling

    def _collect_env_rewards_async(self, sequences: torch.Tensor, attn_masks: torch.Tensor) -> Tuple[list, list, torch.Tensor, torch.Tensor, torch.Tensor, list, dict]:
        """
        Offloads environment processing and returns pre-stacked tensors, logging info, and timing data.
        """
        B = sequences.size(0)
        
        # Send tasks to workers
        for i in range(B):
            effective_len = attn_masks[i].sum().item()
            seq_to_process = sequences[i, :effective_len].cpu().tolist()
            self.task_queue.put((i, seq_to_process))
            
        # Collect results and prepare lists for stacking
        results_in_order = [None] * B
        for _ in range(B):
            i, result_payload = self.result_queue.get()
            results_in_order[i] = result_payload

        # Unpack the results and stack them into batch tensors
        batch_value_targets = torch.stack([r['per_token_value_targets'] for r in results_in_order]).to(self.device)
        batch_criticality_targets = torch.stack([r['per_token_criticality_targets'] for r in results_in_order]).to(self.device)
        batch_error_masks = torch.stack([r['error_signal_mask'] for r in results_in_order]).to(self.device)
        
        # NEW: Stack the instantaneous rewards as well
        batch_instantaneous_rewards = torch.stack([r['per_token_instantaneous_rewards'] for r in results_in_order]).to(self.device)
        
        scalar_rewards_list = [r['scalar_total_reward'] for r in results_in_order]
        
        # Prepare data needed for logging
        log_outcomes = [r['outcome'] for r in results_in_order]
        log_rewards_info = results_in_order # The payload itself is good for logging
        
        # Aggregate timing information from all workers
        aggregated_timing = {}
        if results_in_order and 'timing_info' in results_in_order[0]:
            # Collect all timing keys
            all_timing_keys = set()
            for result in results_in_order:
                if 'timing_info' in result:
                    all_timing_keys.update(result['timing_info'].keys())
            
            # Calculate averages for each timing metric
            for key in all_timing_keys:
                values = [result['timing_info'].get(key, 0.0) for result in results_in_order if 'timing_info' in result]
                if values:
                    aggregated_timing[f'avg_{key}'] = sum(values) / len(values)
                    aggregated_timing[f'total_{key}'] = sum(values)

        return log_outcomes, log_rewards_info, batch_value_targets, batch_criticality_targets, batch_error_masks, scalar_rewards_list, aggregated_timing, batch_instantaneous_rewards

    def shutdown_workers(self):
        """Signals all worker processes to terminate."""
        for _ in self.workers:
            self.task_queue.put((None, None)) # Send sentinel value
        for p in self.workers:
            p.join(timeout=5) # Wait briefly for them to exit
            if p.is_alive():
                p.terminate() # Force terminate if they don't exit gracefully
        print("CPU workers shut down.")

    def _calculate_exploration_metrics(self, sequences: torch.Tensor, attn_masks: torch.Tensor) -> dict:
        """Calculate exploration metrics like Jaccard distance and bigram entropy."""
        import numpy as np
        from collections import Counter
        
        # Convert sequences to lists of tokens for each valid sequence
        valid_sequences = []
        for i in range(sequences.size(0)):
            effective_len = attn_masks[i].sum().item()
            if effective_len > 0:
                seq_tokens = sequences[i, :effective_len].cpu().tolist()
                valid_sequences.append(seq_tokens)
        
        if len(valid_sequences) < 2:
            return {'avg_jaccard_dist': 0.0, 'bigram_entropy': 0.0}
        
        # Calculate pairwise Jaccard distances
        jaccard_distances = []
        for i in range(len(valid_sequences)):
            for j in range(i + 1, len(valid_sequences)):
                set1 = set(valid_sequences[i])
                set2 = set(valid_sequences[j])
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                jaccard_dist = 1.0 - (intersection / union) if union > 0 else 0.0
                jaccard_distances.append(jaccard_dist)
        
        avg_jaccard_dist = np.mean(jaccard_distances) if jaccard_distances else 0.0
        
        # Calculate bigram entropy
        all_bigrams = []
        for seq in valid_sequences:
            for k in range(len(seq) - 1):
                all_bigrams.append((seq[k], seq[k + 1]))
        
        if all_bigrams:
            bigram_counts = Counter(all_bigrams)
            total_bigrams = len(all_bigrams)
            bigram_probs = [count / total_bigrams for count in bigram_counts.values()]
            bigram_entropy = -sum(p * np.log2(p) for p in bigram_probs if p > 0)
        else:
            bigram_entropy = 0.0
        
        return {
            'avg_jaccard_dist': avg_jaccard_dist,
            'bigram_entropy': bigram_entropy
        }

    def _collect_metrics(self, step_result: dict, rewards_objects: list, sequences: torch.Tensor, attn_masks: torch.Tensor):
        """Collect metrics from training step for plotting."""
        step = self.step_count
        
        # Basic metrics
        self.history['step'].append(step)
        
        # PPO loss metrics
        ppo_info = step_result.get('ppo_loss_info', {})
        self.history['ppo_loss_start'].append(ppo_info.get('ppo_loss_start', 0.0))
        self.history['ppo_loss_end'].append(ppo_info.get('ppo_loss_end', 0.0))
        self.history['ppo_loss_reduction'].append(ppo_info.get('ppo_loss_diff', 0.0))
        
        # Replay buffer metrics
        replay_info = step_result.get('replay_loss_info', {})
        if replay_info:
            self.history['replay_original_loss'].append(replay_info.get('original_replay_loss', 0.0))
            self.history['replay_current_loss'].append(replay_info.get('current_replay_loss', 0.0))
            self.history['replay_improvement'].append(
                replay_info.get('original_replay_loss', 0.0) - replay_info.get('current_replay_loss', 0.0)
            )
            # Add replay-specific losses for combined losses plot
            if 'replay_value_loss' not in self.history:
                self.history['replay_value_loss'] = []
            if 'replay_reward_loss' not in self.history:
                self.history['replay_reward_loss'] = []
            self.history['replay_value_loss'].append(replay_info.get('replay_value_loss', 0.0))
            self.history['replay_reward_loss'].append(replay_info.get('replay_reward_loss', 0.0))
        
        # Calculate average reward across batch
        total_reward = sum(r['scalar_total_reward'] for r in rewards_objects)
        avg_reward = total_reward / len(rewards_objects) if rewards_objects else 0.0
        self.history['avg_reward'].append(avg_reward)
        
        # Calculate EOS percentage
        eos_count = step_result.get('eos_count', 0)
        eos_percentage = (eos_count / len(rewards_objects)) * 100.0 if rewards_objects else 0.0
        self.history['eos_percentage'].append(eos_percentage)
        
        # Get actual value predictions from the policy for this batch
        with torch.no_grad():
            self.policy.eval()
            outputs = self.policy(sequences, attention_mask=attn_masks.to(torch.long))
            predicted_value_scores = self.policy.value(outputs.hidden_states[-1])  # Shape: (B, T, C)
            self.policy.train()
        
        # Collect per-check metrics
        for check_idx in range(NUM_CHECKS):
            check_name = IDX_TO_CHECK.get(check_idx, f"check_{check_idx}")
            
            # Calculate average loss and score for this check across the batch
            check_losses = []
            check_scores = []
            value_means = []
            value_vars = []
            
            for i, reward_obj in enumerate(rewards_objects):
                # Get per-token criticality targets for this check
                if 'per_token_criticality_targets' in reward_obj:
                    targets = reward_obj['per_token_criticality_targets']
                    if targets.size(1) > check_idx:
                        # Calculate actual L1 loss between predicted and target criticality
                        check_targets = targets[:, check_idx].to(self.device)  # Move to GPU
                        
                        # Get corresponding predictions for this sequence and check
                        if i < predicted_value_scores.size(0) and check_idx < predicted_value_scores.size(2):
                            seq_predictions = predicted_value_scores[i, :, check_idx]  # Shape: (T,)
                            valid_mask = attn_masks[i]  # Only consider valid tokens
                            
                            if valid_mask.sum() > 0:
                                valid_targets = check_targets[valid_mask]
                                valid_predictions = seq_predictions[valid_mask]
                                
                                # Calculate L1 loss between predictions and targets
                                l1_loss = torch.abs(valid_predictions - valid_targets).mean().item()
                                check_losses.append(l1_loss)
                            else:
                                check_losses.append(1.0)  # Fallback for empty sequences
                        else:
                            check_losses.append(1.0)  # Fallback for missing data
                
                # Get success score for this check
                if 'diagnostics' in reward_obj and 'r_success_tensor' in reward_obj['diagnostics']:
                    success_tensor = reward_obj['diagnostics']['r_success_tensor']
                    if success_tensor.size(0) > check_idx:
                        check_scores.append(success_tensor[check_idx].item())
                
                # Get actual value predictions for this check
                if i < predicted_value_scores.size(0) and check_idx < predicted_value_scores.size(2):
                    # Get predictions for this sequence and check, only for valid tokens
                    seq_predictions = predicted_value_scores[i, :, check_idx]  # Shape: (T,)
                    valid_predictions = seq_predictions[attn_masks[i]]  # Only valid tokens
                    
                    if valid_predictions.numel() > 0:
                        value_means.append(valid_predictions.mean().item())
                        value_vars.append(valid_predictions.var().item())
                    else:
                        value_means.append(0.0)
                        value_vars.append(0.0)
                else:
                    value_means.append(0.0)
                    value_vars.append(0.0)
            
            # Store averages
            self.history[f'check_loss_{check_name}'].append(
                np.mean(check_losses) if check_losses else 0.0
            )
            self.history[f'check_score_{check_name}'].append(
                np.mean(check_scores) if check_scores else 0.0
            )
            self.history[f'value_mean_{check_name}'].append(
                np.mean(value_means) if value_means else 0.0
            )
            self.history[f'value_var_{check_name}'].append(
                np.mean(value_vars) if value_vars else 0.0
            )
        
        # Extract actual loss metrics from step_result
        loss_info = step_result.get('loss_info', {})
        self.history['policy_loss'].append(loss_info.get('policy_loss', 0.0))
        self.history['value_loss'].append(loss_info.get('value_loss', 0.0))
        self.history['reward_loss'].append(loss_info.get('reward_loss', 0.0))
        self.history['criticality_loss'].append(loss_info.get('criticality_loss', 0.0))
        self.history['consistency_loss'].append(loss_info.get('consistency_loss', 0.0))
        
        # Extract exploration metrics from step_result
        exploration_info = step_result.get('exploration_info', {})
        self.history['avg_jaccard_dist'].append(exploration_info.get('avg_jaccard_dist', 0.0))
        self.history['bigram_entropy'].append(exploration_info.get('bigram_entropy', 0.0))

    def _generate_trajectories(self, active_checks: list = None, check_mask: torch.Tensor = None) -> Tuple[torch.LongTensor, torch.BoolTensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Generates trajectories starting with a fixed prefix and then continuing with policy sampling.
        """
        self.policy.eval()
        B, T = self.cfg.BATCH_SIZE, self.cfg.MAX_LEN
        
        sequences = torch.full((B, T), self.cfg.PAD_TOKEN_ID, dtype=torch.long, device=self.device)
        attn_mask = torch.zeros((B, T), dtype=torch.bool, device=self.device)
        
        # Tensor to hold per-token exploration bonuses
        per_token_exploration_bonuses = torch.zeros((B, T + 1, NUM_CHECKS), device=self.device, dtype=self.policy_dtype)
        # Tensor to hold raw exploration head scores (before criticality weighting)
        per_token_exploration_head_scores = torch.zeros((B, T + 1, NUM_CHECKS), device=self.device, dtype=self.policy_dtype)

        prefix_tokens_str = [
            'ITRACT', 'ITRACT_ID_1', 'TYPE_YUKAWA',
            'FIELD', 'FIELD_ID_1', 'TYPE_FIELD_fermion', 'DIM_2', 'GEN_3', 'SELF_CONJ_FALSE', 'CHIRALITY_left', 'SU3C_REP_1', 'SU2L_REP_2', 'U1Y_CHARGE_-1', 'QN_L_1', 'QN_B_0',
            'PARTICLE', 'PARTICLE_ID_1', 'TYPE_PARTICLE_fermion', 'MASS_1e-9', 'CHARGE_-1', 'END_PARTICLE',
            'PARTICLE', 'PARTICLE_ID_2', 'TYPE_PARTICLE_fermion', 'MASS_1e-4', 'CHARGE_-1', 'END_PARTICLE',
            'PARTICLE', 'PARTICLE_ID_3', 'TYPE_PARTICLE_fermion', 'MASS_1e-9', 'CHARGE_-1', 'END_PARTICLE',
            'PARTICLE', 'PARTICLE_ID_4', 'TYPE_PARTICLE_fermion', 'MASS_1e-1', 'CHARGE_-1', 'END_PARTICLE',
            'PARTICLE', 'PARTICLE_ID_5', 'TYPE_PARTICLE_fermion', 'MASS_1e0', 'CHARGE_-1', 'END_PARTICLE',
            'PARTICLE', 'PARTICLE_ID_6', 'TYPE_PARTICLE_fermion', 'MASS_1e0', 'CHARGE_-1', 'END_PARTICLE',
            'END_FIELD',
            'FIELD', 'FIELD_ID_2', 'TYPE_FIELD_fermion', 'DIM_1', 'GEN_3', 'SELF_CONJ_FALSE', 'CHIRALITY_right', 'SU3C_REP_1', 'SU2L_REP_1', 'U1Y_CHARGE_-1', 'QN_L_1', 'QN_B_0',
            'PARTICLE', 'PARTICLE_ID_2', 'TYPE_PARTICLE_fermion', 'MASS_1e-4', 'CHARGE_-1', 'END_PARTICLE',
            'PARTICLE', 'PARTICLE_ID_4', 'TYPE_PARTICLE_fermion', 'MASS_1e-1', 'CHARGE_-1', 'END_PARTICLE',
            'PARTICLE', 'PARTICLE_ID_6', 'TYPE_PARTICLE_fermion', 'MASS_1e0', 'CHARGE_-1', 'END_PARTICLE',
            'END_FIELD',
            'FIELD', 'FIELD_ID_3', 'TYPE_FIELD_complex', 'DIM_2', 'GEN_1', 'SELF_CONJ_FALSE', 'CHIRALITY_none', 'SU3C_REP_1', 'SU2L_REP_2', 'U1Y_CHARGE_1', 'QN_L_0', 'QN_B_0',
            'PARTICLE', 'PARTICLE_ID_7', 'TYPE_PARTICLE_complex', 'MASS_1e2', 'CHARGE_1', 'END_PARTICLE',
            'PARTICLE', 'PARTICLE_ID_8', 'TYPE_PARTICLE_complex', 'MASS_1e2', 'CHARGE_1', 'END_PARTICLE',
            'END_FIELD',
            'END_ITRACT',
            'ITRACT', 'ITRACT_ID_2'
        ]
        prefix_tokens_ids = [self.token_to_idx[tok] for tok in prefix_tokens_str]
        prefix_len = len(prefix_tokens_ids)
        
        # Prepend BOS to sequences for environment compatibility
        bos_id = self.token_to_idx['BOS']
        sequences_with_bos = torch.zeros((B, T + 1), dtype=torch.long, device=self.device)
        sequences_with_bos[:, 0] = bos_id  # Set BOS at position 0
        sequences_with_bos[:, 1:prefix_len+1] = torch.tensor(prefix_tokens_ids, dtype=torch.long, device=self.device).expand(B, -1)
        
        # Update attention mask to include BOS
        attn_mask_with_bos = torch.zeros((B, T + 1), dtype=torch.bool, device=self.device)
        attn_mask_with_bos[:, 0] = True  # BOS is always active
        attn_mask_with_bos[:, 1:prefix_len+1] = True
        
        # Use sequences with BOS for the rest of the function
        sequences = sequences_with_bos
        attn_mask = attn_mask_with_bos
        prefix_len = prefix_len + 1  # Account for BOS
        
        grammar_states = [self.grammar.initial_state() for _ in range(B)]
        # Step grammar state with BOS first
        for i in range(B):
            grammar_states[i] = self.grammar.step(grammar_states[i], bos_id)
        # Then step with prefix tokens
        for t, tok_id in enumerate(prefix_tokens_ids):
            for i in range(B):
                grammar_states[i] = self.grammar.step(grammar_states[i], tok_id)

        # --- NEW: Forward pass on the prefix to get initial hidden states and populate critic/exploration values ---
        with torch.no_grad():
            prefix_outputs = self.policy(sequences[:, :prefix_len], attention_mask=attn_mask[:, :prefix_len].to(torch.long))
            prefix_hidden_states = prefix_outputs.hidden_states[-1] # Shape: (B, prefix_len, H)
            
            # Populate critic and exploration head scores for the prefix part of the sequence
            crit_logits_prefix = self.policy.criticality(prefix_hidden_states) # (B, prefix_len, C)
            crit_probs_prefix = torch.sigmoid(crit_logits_prefix)

            if hasattr(self.policy, 'exploration_q_values'):
                q_logits_matrix_prefix = self.policy.exploration_q_values(prefix_hidden_states) # (B, prefix_len, V, C)
                q_scores_prefix = torch.sigmoid(q_logits_matrix_prefix)
                
                # Get max Q-scores over vocabulary for the bonus calculation
                max_q_scores_prefix = q_scores_prefix.max(dim=2).values # (B, prefix_len, C)
                
                # Calculate and store the bonus and head scores for the prefix
                per_check_bonus_prefix = crit_probs_prefix * max_q_scores_prefix
                per_token_exploration_bonuses[:, :prefix_len, :] = per_check_bonus_prefix * self.cfg.EXPLORATION_BETA
                per_token_exploration_head_scores[:, :prefix_len, :] = max_q_scores_prefix

        # Set up for autoregressive generation
        current_tokens = sequences[:, prefix_len - 1].unsqueeze(-1)
        past_key_values = prefix_outputs.past_key_values # Use the cache from the prefix forward pass
        
        single_token_inference_time, grammar_mask_time = 0, 0
        
        for t_gen in range(prefix_len - 1, T):
            # [TIME] Trajectory Generation: Autoregressive Single-Token Inference
            with torch.no_grad():
                start_inference = time.perf_counter()
                if self.cfg.MODEL_TYPE == 'hf':
                    outputs = self.policy(current_tokens, past_key_values=past_key_values)
                else:
                    outputs = self.policy(sequences[:, :t_gen+1], attention_mask=attn_mask[:, :t_gen+1].to(torch.long))
                if torch.cuda.is_available(): torch.cuda.synchronize()
                single_token_inference_time += time.perf_counter() - start_inference
            
            logits, hidden_states, past_key_values = outputs.logits, outputs.hidden_states[-1], outputs.past_key_values
            
            # [TIME] Grammar Masking: Apply grammar-valid action mask to logits
            start_grammar_mask = time.perf_counter()
            valid_mask = torch.stack([self.grammar.get_valid_actions(s) for s in grammar_states]).to(self.device)
            masked_logits = logits[:, -1, :].masked_fill(~valid_mask, float("-inf"))
            grammar_mask_time += time.perf_counter() - start_grammar_mask
            
            # Criticality-driven exploration: weighted sum of per-check contributions
            last_hidden = hidden_states[:, -1:, :]  # (B,1,H)
            # 1) Criticality weights per check (logits -> probs)
            crit_logits_last = self.policy.criticality(last_hidden)  # (B,1,C)
            crit_probs_last = torch.sigmoid(crit_logits_last)       # (B,1,C)
            
            # Apply curriculum-aware masking to criticality probabilities
            if check_mask is not None:
                crit_probs_last = crit_probs_last * check_mask.view(1, 1, -1)

            # 2) Q-head per-token, per-check contribution scores at this step
            if hasattr(self.policy, 'exploration_q_values'):
                q_logits_matrix = self.policy.exploration_q_values(last_hidden)  # (B,1,V,C)
                q_scores = torch.sigmoid(q_logits_matrix)                        # (B,1,V,C)

                # 3) Weighted sum across checks using criticality as weights
                weighted_scores = q_scores * crit_probs_last.unsqueeze(2)        # (B,1,V,C)
                exploration_scores = weighted_scores.sum(dim=-1).squeeze(1)     # (B,V)

                # Track per-check exploration bonuses
                max_q_scores = q_scores.max(dim=2).values  # (B,1,C)
                per_check_bonus = crit_probs_last * max_q_scores  # (B,1,C)
                per_token_exploration_bonuses[:, t_gen + 1, :] = per_check_bonus.squeeze(1) * self.cfg.EXPLORATION_BETA
                
                # Store raw exploration head scores (max across vocabulary for each check)
                per_token_exploration_head_scores[:, t_gen + 1, :] = max_q_scores.squeeze(1)

                # Respect current grammar: mask invalid actions prior to blending
                exploration_scores = exploration_scores.masked_fill(~valid_mask, 0.0)

                # 4) Add exploration bonus to policy logits
                blended_logits = masked_logits + self.cfg.EXPLORATION_BETA * exploration_scores
            else:
                blended_logits = masked_logits

            dist = Categorical(logits=blended_logits / self.cfg.TEMPERATURE)
            next_toks = dist.sample()
            
            sequences[:, t_gen + 1] = next_toks
            attn_mask[:, t_gen + 1] = True
            
            # Update grammar states
            for i in range(B):
                # The state is updated with the new token
                grammar_states[i] = self.grammar.step(grammar_states[i], next_toks[i].item())

            current_tokens = next_toks.unsqueeze(-1)
        
        # --- Final Forward Pass for Critic/Value/LogProbs with fine-grained timing ---
        # [TIME] Trajectory Generation: Full-sequence Forward + Value Head
        start_final_forward = time.perf_counter()
        with torch.no_grad():
            full_outputs = self.policy(
                sequences, 
                attention_mask=attn_mask.to(torch.long)
            )
            if torch.cuda.is_available(): torch.cuda.synchronize()
            full_forward_time = time.perf_counter() - start_final_forward
            
            full_logits, full_hidden_states = full_outputs.logits, full_outputs.hidden_states[-1]

            # [TIME] Trajectory Generation: Value Head Calculation
            start_value_head = time.perf_counter()
            value_vectors = self.policy.value(full_hidden_states)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            value_head_time = time.perf_counter() - start_value_head
            
            full_dist = Categorical(logits=full_logits)
            log_probs = full_dist.log_prob(sequences)

        total_model_time = single_token_inference_time + full_forward_time + value_head_time

        return (
            sequences,
            attn_mask,
            log_probs.detach(),
            value_vectors.detach(),
            per_token_exploration_bonuses.detach(),
            per_token_exploration_head_scores.detach(),
            {
                'single_token_inference_time': single_token_inference_time,
                'grammar_mask_time': grammar_mask_time,
                'full_forward_time': full_forward_time,
                'value_head_time': value_head_time,
                'total_model_forward_time': total_model_time,
            },
        )


    def _training_step(self, is_update_step: bool) -> dict:
        self.policy.train()
        step_timings = {}
        replay_loss_info = {}

        # Update curriculum and get active checks
        self.curriculum.update_step(self.step_count)
        active_checks = self.curriculum.get_active_checks()
        active_check_indices = torch.tensor([CHECK_TO_IDX[name] for name in active_checks if name in CHECK_TO_IDX], device=self.device, dtype=torch.long)
        check_mask = torch.zeros(NUM_CHECKS, device=self.device, dtype=self.policy_dtype)
        if active_check_indices.numel() > 0:
            check_mask[active_check_indices] = 1.0

        # [TIME] Trajectory Generation: Total
        start_time = time.perf_counter()
        sequences, attn_masks, log_probs_gen, value_vectors_old, per_token_exploration_bonuses, per_token_exploration_head_scores, gen_timings = self._generate_trajectories(active_checks, check_mask)
        step_timings['generate_trajectories_total'] = time.perf_counter() - start_time
        # Use the newly calculated total model time for the "Model" part, and keep the detailed breakdown
        step_timings['total_model_forward_time'] = gen_timings.pop('total_model_forward_time')
        step_timings.update(gen_timings)
        if torch.cuda.is_available(): torch.cuda.synchronize()

        # [TIME] Environment & Reward: Total
        start_env_reward_time = time.perf_counter()
        
        # This one call now handles all CPU work and returns ready-to-use tensors
        (log_outcomes, 
         log_rewards_payloads, 
         batch_value_targets, 
         batch_criticality_targets, 
         batch_error_masks, 
         scalar_rewards_list,
         worker_timing_info,
         batch_instantaneous_rewards) = self._collect_env_rewards_async(sequences, attn_masks)

        step_timings['env_and_reward_total'] = time.perf_counter() - start_env_reward_time
        
        # Add detailed timing information from workers
        step_timings.update(worker_timing_info)
        
        # Debug: Print timing information
        if worker_timing_info:
            print(f"DEBUG: Worker timing info: {worker_timing_info}")

        # Apply curriculum-aware reward masking to the pre-stacked tensors
        batch_value_targets *= check_mask.view(1, -1)
        
        # Recalculate scalar rewards with curriculum masking
        for i in range(len(scalar_rewards_list)):
            scalar_rewards_list[i] = float(batch_value_targets[i, 0].sum().item())
            # Add other reward components if needed
            if 'diagnostics' in log_rewards_payloads[i]:
                diagnostics = log_rewards_payloads[i]['diagnostics']
                scalar_rewards_list[i] += diagnostics.get('structural_penalty', 0.0)
                scalar_rewards_list[i] += diagnostics.get('eos_bonus', 0.0)
                scalar_rewards_list[i] += diagnostics.get('length_penalty', 0.0)

        # Calculate the initial per-sequence MSE Loss (for replay buffer)
        initial_mse_losses = torch.zeros(self.cfg.BATCH_SIZE, device=self.device, dtype=self.policy_dtype)
        for i in range(self.cfg.BATCH_SIZE):
            # Mask to filter out padding in time and padding checks
            active_tokens_mask = attn_masks[i]
            predicted_scores_old_i = value_vectors_old[i, active_tokens_mask]
            target_i = batch_value_targets[i, active_tokens_mask]

            # Calculate MSE only on active tokens
            if target_i.numel() > 0:
                initial_mse_losses[i] = F.mse_loss(predicted_scores_old_i, target_i)

        # Store the trajectory to the replay buffer every 10 steps
        if self.step_count % 10 == 0 and self.step_count > 0:
            self.replay_buffer.store(sequences, attn_masks, batch_value_targets, initial_mse_losses)

        for ppo_epoch in range(self.cfg.PPO_EPOCHS):
            # Recompute against current policy parameters each PPO epoch
            outputs = self.policy(
                sequences, 
                attention_mask=attn_masks.to(torch.long)
            )
            logits, hidden_states = outputs.logits, outputs.hidden_states[-1]
            predicted_score_vectors = self.policy.value(hidden_states)
            # Criticality predictions
            predicted_criticality_scores = self.policy.criticality(hidden_states)
            # RND intrinsic reward prediction features
            if hasattr(self.policy, 'rnd_predictor_network') and hasattr(self.policy, 'rnd_target_network'):
                rnd_target = self.policy.rnd_target_network(hidden_states.detach())
                rnd_pred = self.policy.rnd_predictor_network(hidden_states)
                rnd_loss = F.mse_loss(rnd_pred[attn_masks], rnd_target[attn_masks])
            else:
                rnd_loss = torch.tensor(0.0, device=self.device, dtype=self.policy_dtype)

            value_loss = F.mse_loss(
                predicted_score_vectors[attn_masks],
                batch_value_targets[attn_masks]
            )

            # Binary cross-entropy loss for criticality targets (masked to active tokens)
            criticality_loss = F.binary_cross_entropy_with_logits(
                predicted_criticality_scores[attn_masks],
                batch_criticality_targets[attn_masks]
            )
            
            # Corrected Exploration Loss for the Q-Head
            exploration_loss = torch.tensor(0.0, device=self.device)
            if hasattr(self.policy, 'exploration_q_values'):
                # The goal is to train Q(s_t, a_{t+1}, c) to predict r_{t+1}(c).
                # This requires careful alignment of states, actions, rewards, and masks.
                
                # 1. States (s_t): Hidden states from t=0 to T-2.
                # These are the states *before* taking an action.
                states_for_q_pred = outputs.hidden_states[-1][:, :-1, :] # Shape: (B, T-1, H)
                
                # 2. Actions (a_{t+1}): Actions taken from t=1 to T-1.
                # This is the action taken *from* the corresponding state.
                actions_taken = sequences[:, 1:] # Shape: (B, T-1)
                
                # 3. Targets (r_{t+1}): Instantaneous rewards received for taking those actions.
                q_targets = batch_instantaneous_rewards[:, 1:, :] # Shape: (B, T-1, C)

                # 4. Mask: The loss only applies if the *state* (s_t) was critical.
                # We use the criticality targets for the states from t=0 to T-2.
                criticality_mask = (batch_criticality_targets[:, :-1, :] > 0.5) # Shape: (B, T-1, C)

                # Only proceed if there's at least one critical state in the batch
                if criticality_mask.any():
                    # Get Q-head predictions for all actions from the relevant states
                    q_preds_all_actions = self.policy.exploration_q_values(states_for_q_pred) # Shape: (B, T-1, V, C)
                    
                    # Prepare action indices for gathering
                    action_indices_for_gather = actions_taken.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, NUM_CHECKS)
                    
                    # Gather the specific Q-predictions for the actions that were actually taken
                    q_preds_for_actions_taken = q_preds_all_actions.gather(2, action_indices_for_gather).squeeze(2) # Shape: (B, T-1, C)
                    
                    # Compute MSE loss, but only on the elements selected by the criticality_mask
                    exploration_loss = F.mse_loss(
                        q_preds_for_actions_taken[criticality_mask],
                        q_targets[criticality_mask]
                    )

            error_value_predictions = predicted_score_vectors[batch_error_masks]
            supervised_loss = F.mse_loss(error_value_predictions, torch.zeros_like(error_value_predictions)) if error_value_predictions.numel() > 0 else torch.tensor(0.0, device=self.device)

            advantages = torch.tensor(scalar_rewards_list, device=self.device, dtype=self.policy_dtype) - value_vectors_old[:, 0, :].sum(dim=-1)
            
            dist = Categorical(logits=logits.view(-1, self.vocab_size))
            new_log_probs = dist.log_prob(sequences.view(-1))
            
            ratios = torch.exp(new_log_probs - log_probs_gen.view(-1))
            expanded_advantages = advantages.repeat_interleave(sequences.size(1))
            surr1 = ratios * expanded_advantages
            surr2 = torch.clamp(ratios, 1.0 - self.cfg.CLIP_RANGE, 1.0 + self.cfg.CLIP_RANGE) * expanded_advantages
            
            active_tokens_mask = attn_masks.view(-1)
            policy_loss = -torch.min(surr1[active_tokens_mask], surr2[active_tokens_mask]).mean()
            entropy_loss = -dist.entropy()[active_tokens_mask].mean()

            total_loss = (
                policy_loss
                + self.cfg.VALUE_COEF * value_loss
                + self.cfg.ENTROPY_COEF * entropy_loss
                + self.cfg.SUPERVISED_LOSS_COEF * supervised_loss
                + self.cfg.CRITICALITY_COEF * criticality_loss
                + self.cfg.RND_COEF * rnd_loss
                + self.cfg.EXPLORATION_LOSS_COEF * exploration_loss
            )

            # Capture PPO loss before/after per-epoch update
            if ppo_epoch == 0:
                ppo_loss_start = total_loss.item()
            if ppo_epoch == self.cfg.PPO_EPOCHS - 1:
                ppo_loss_end = total_loss.item()

            # Per-epoch update: clear grads, backprop, clip, and step
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.MAX_GRAD_NORM)
            self.optimizer.step()

        # Evaluate on the replay buffer after PPO updates (using new weights)
        if self.replay_buffer.is_ready(4):
            with torch.no_grad():
                replay_seqs, replay_masks, replay_targets, replay_original_losses = self.replay_buffer.sample(4)

                if replay_seqs is not None:
                    # For replay buffer, we don't have hierarchical embeddings, so we pass None
                    # The models should handle None values gracefully
                    replay_outputs = self.policy(
                        replay_seqs, 
                        attention_mask=replay_masks.to(torch.long)
                    )
                    replay_hidden_states = replay_outputs.hidden_states[-1]
                    replay_predicted_scores = self.policy.value(replay_hidden_states)
                    
                    masked_predicted = replay_predicted_scores[replay_masks]
                    masked_targets = replay_targets[replay_masks]
                    
                    if masked_predicted.numel() > 0:
                        current_replay_loss = F.mse_loss(
                            masked_predicted,
                            masked_targets
                        ).item()
                    else:
                        current_replay_loss = 0.0
                    
                    replay_loss_info = {
                        'current_replay_loss': current_replay_loss,
                        'original_replay_loss': replay_original_losses.mean().item(),
                        'replay_value_loss': current_replay_loss,  # Using same as current for now
                        'replay_reward_loss': 0.0  # Not computed in this training step
                    }
        
        eos_count = (sequences == self.cfg.EOS_TOKEN_ID).any(dim=1).sum().item()
        ppo_loss_info = {
            'ppo_loss_start': ppo_loss_start,
            'ppo_loss_end': ppo_loss_end,
            'ppo_loss_diff': ppo_loss_start - ppo_loss_end # Calculate difference for easy tracking
        }
        
        # Calculate exploration metrics
        exploration_info = self._calculate_exploration_metrics(sequences, attn_masks)
        
        step_result = {
            'eos_count': eos_count, 
            'replay_loss_info': replay_loss_info, 
            'step_timings': step_timings, 
            'ppo_loss_info': ppo_loss_info,
                'loss_info': {
                    'policy_loss': policy_loss.item() if 'policy_loss' in locals() else 0.0,
                    'value_loss': value_loss.item() if 'value_loss' in locals() else 0.0,
                    'criticality_loss': criticality_loss.item() if 'criticality_loss' in locals() else 0.0,
                    'supervised_loss': supervised_loss.item() if 'supervised_loss' in locals() else 0.0,
                    'rnd_loss': rnd_loss.item() if 'rnd_loss' in locals() else 0.0,
                    'entropy_loss': entropy_loss.item() if 'entropy_loss' in locals() else 0.0,
                    'exploration_loss': exploration_loss.item() if 'exploration_loss' in locals() else 0.0,
                    'total_loss': total_loss.item() if 'total_loss' in locals() else 0.0,
                    'reward_loss': 0.0,  # Not computed in this training step
                    'consistency_loss': 0.0  # Not computed in this training step
                },
            'exploration_info': exploration_info
        }
        
        if self.cfg.PRINT_GENERATED_SEQUENCES:
            num_to_print = self.cfg.NUM_SEQUENCES_TO_PRINT if self.cfg.NUM_SEQUENCES_TO_PRINT > 0 else self.cfg.BATCH_SIZE
            for i in range(min(num_to_print, self.cfg.BATCH_SIZE)):
                effective_len = attn_masks[i].sum().item()
                if effective_len > 0:
                    should_print_detailed = (
                        self.cfg.PRINT_DETAILED_CHECKLIST and
                        ((self.step_count + 1) % self.cfg.EVAL_EVERY_N_STEPS == 0)
                    )
                    log_batch_summary(
                        batch_idx=i,
                        outcome=log_outcomes[i],  # Use the collected outcome for logging
                        sequence_tokens=[self.idx_to_token.get(tok.item(), "UNK") for tok in sequences[i, :effective_len]],
                        rewards_info=log_rewards_payloads[i], # Pass the full payload for rich logging
                        predicted_score_vectors=predicted_score_vectors[i, :effective_len].detach(),
                        per_token_instantaneous_rewards=log_rewards_payloads[i]['per_token_instantaneous_rewards'][:effective_len].detach(),
                        per_token_target_scores=batch_value_targets[i, :effective_len].detach(),
                        averaged_check_type_scores=log_rewards_payloads[i]['diagnostics']['r_success_tensor'].cpu(),
                        active_checks=active_checks,
                        per_token_exploration_bonus=per_token_exploration_bonuses[i, :effective_len].detach(),
                        criticality_scores=predicted_criticality_scores[i, :effective_len].detach(),
                        exploration_head_scores=per_token_exploration_head_scores[i, :effective_len].detach(),
                        print_sequence=self.cfg.PRINT_SEQUENCE,
                        print_detailed_checklist=should_print_detailed,
                        print_per_check_rewards_detail=self.cfg.PRINT_PER_CHECK_REWARDS_DETAIL,
                        step_number=self.step_count + 1,
                        timings=step_timings
                    )
        
        if self.step_count % 10 == 0:
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Collect metrics for plotting
        self._collect_metrics(step_result, log_rewards_payloads, sequences, attn_masks)
        
        # Return replay loss info
        return step_result


    def train(self):
        for epoch in range(self.cfg.EPOCHS):
            print(f"\n--- Starting Epoch {epoch + 1}/{self.cfg.EPOCHS} ---")
            
            for step in range(self.cfg.STEPS_PER_EPOCH):
                self.step_count = epoch * self.cfg.STEPS_PER_EPOCH + step
                # Maintain step_count and run a full training step (updates occur inside)
                is_update_step = True
                
                step_result = self._training_step(is_update_step)

                # Print PPO Loss Metrics every step (updates happen each step now)
                ppo_info = step_result['ppo_loss_info']
                print(f"\n--- PPO Loss Summary (Step {self.step_count+1}) ---")
                print(f"  PPO Epochs: {self.cfg.PPO_EPOCHS}")
                print(f"  Loss at Start: {ppo_info['ppo_loss_start']:.6f}")
                print(f"  Loss at End:   {ppo_info['ppo_loss_end']:.6f}")
                print(f"  Loss Reduction: {ppo_info['ppo_loss_diff']:.6f}")
                
                # Print individual head losses
                loss_info = step_result.get('loss_info', {})
                print(f"\n--- Individual Head Losses (Step {self.step_count+1}) ---")
                print(f"  Policy Loss:     {loss_info.get('policy_loss', 0.0):.6f}")
                print(f"  Value Loss:      {loss_info.get('value_loss', 0.0):.6f}")
                print(f"  Criticality Loss: {loss_info.get('criticality_loss', 0.0):.6f}")
                print(f"  Supervised Loss: {loss_info.get('supervised_loss', 0.0):.6f}")
                print(f"  RND Loss:        {loss_info.get('rnd_loss', 0.0):.6f}")
                print(f"  Exploration Loss: {loss_info.get('exploration_loss', 0.0):.6f}")
                print(f"  Entropy Loss:    {loss_info.get('entropy_loss', 0.0):.6f}")
                print(f"  Total Loss:      {loss_info.get('total_loss', 0.0):.6f}")
                
                # Print loss coefficients for reference
                print(f"\n--- Loss Coefficients ---")
                print(f"  Value Coef:      {self.cfg.VALUE_COEF:.3f}")
                print(f"  Criticality Coef: {self.cfg.CRITICALITY_COEF:.3f}")
                print(f"  Supervised Coef: {self.cfg.SUPERVISED_LOSS_COEF:.3f}")
                print(f"  RND Coef:        {self.cfg.RND_COEF:.3f}")
                print(f"  Exploration Coef: {self.cfg.EXPLORATION_LOSS_COEF:.3f}")
                print(f"  Entropy Coef:    {self.cfg.ENTROPY_COEF:.3f}")
                if step_result.get('replay_loss_info', {}):
                    rl_info = step_result['replay_loss_info']
                    print(f"\n--- Replay Buffer Summary (Step {self.step_count+1}) ---")
                    print(f"  Buffer Size: {len(self.replay_buffer.sequences)}")
                    print(f"  Original Loss (Mean MSE on sampled batch): {rl_info['original_replay_loss']:.6f}")
                    print(f"  Current Loss (Mean MSE on sampled batch): {rl_info['current_replay_loss']:.6f}")
                    print(f"  Loss Improvement: {rl_info['original_replay_loss'] - rl_info['current_replay_loss']:.6f}")

                
                if (self.step_count + 1) % self.cfg.LOG_EVERY_N_STEPS == 0:
                    print(f"\nEpoch {epoch+1}, Step {self.step_count+1} completed.")
                    process = psutil.Process(os.getpid())
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    print(f"Memory usage: {memory_mb:.1f} MB")
                    if torch.cuda.is_available():
                        cuda_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                        cuda_cached_mb = torch.cuda.memory_reserved() / 1024 / 1024
                        print(f"CUDA memory: {cuda_memory_mb:.1f} MB allocated, {cuda_cached_mb:.1f} MB cached")
                
                # Generate plots periodically (every 50 steps or at the end of each epoch)
                if (self.step_count + 1) % 50 == 0 or (step + 1) == self.cfg.STEPS_PER_EPOCH:
                    print(f"\n--- Generating Training Plots (Step {self.step_count+1}) ---")
                    try:
                        self.plotter.generate_plots(self.history)
                        print("Training plots generated successfully.")
                    except Exception as e:
                        print(f"Warning: Failed to generate plots: {e}")