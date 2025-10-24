# ==== rl/agent.py ====
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

from config import Config, get_config
from rl.network import HuggingFacePolicy, SimpleTransformerPolicy
from rl.grammar import GrammarMasker
from rl.environment import ParticlePhysicsEnvironment, EnvOutcome
from rl.reward import RewardShaper, ShapedRewards
from rl.curriculum import Curriculum
from rl.logging import log_batch_summary
from collections import defaultdict
from rl.replay_buffer import ReplayBuffer
from Token2Model.check import NUM_CHECKS

DEBUG = get_config().DEBUG_PRINTS

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
        self.env = ParticlePhysicsEnvironment(cfg)
        self.curriculum = Curriculum()
        
        self.reward_shaper = RewardShaper(cfg, self.policy_dtype)
        
        self.step_count = 0
        self.replay_buffer = ReplayBuffer(max_size=500, policy_dtype=self.policy_dtype)
        self.replay_buffer.policy_device = self.device # Set the device for sampling


    def _generate_trajectories(self) -> Tuple[torch.LongTensor, torch.BoolTensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Generates trajectories starting with a fixed prefix and then continuing with policy sampling.
        """
        self.policy.eval()
        B, T = self.cfg.BATCH_SIZE, self.cfg.MAX_LEN
        
        sequences = torch.full((B, T), self.cfg.PAD_TOKEN_ID, dtype=torch.long, device=self.device)
        attn_mask = torch.zeros((B, T), dtype=torch.bool, device=self.device)
        
        prefix_tokens_str = [
            'BOS',
                'ITRACT', 'ITRACT_ID_1', 'TYPE_YUKAWA',
                    'FIELD', 'FIELD_ID_1', 'TYPE_FIELD_fermion', 'DIM_2', 'GEN_3', 'SELF_CONJ_FALSE', 'CHIRALITY_left', 'SU3C_REP_1', 'SU2L_REP_2', 'U1Y_CHARGE_-1', 'QN_L_1', 'QN_B_0',
                        'PARTICLE', 'PARTICLE_ID_1', 'TYPE_PARTICLE_fermion', 'MASS_1e-9', 'CHARGE_0', 'END_PARTICLE',
                        'PARTICLE', 'PARTICLE_ID_2', 'TYPE_PARTICLE_fermion', 'MASS_1e-4', 'CHARGE_-1', 'END_PARTICLE',
                        'PARTICLE', 'PARTICLE_ID_3', 'TYPE_PARTICLE_fermion', 'MASS_1e-9', 'CHARGE_0', 'END_PARTICLE',
                        'PARTICLE', 'PARTICLE_ID_4', 'TYPE_PARTICLE_fermion', 'MASS_1e-1', 'CHARGE_-1', 'END_PARTICLE',
                        'PARTICLE', 'PARTICLE_ID_5', 'TYPE_PARTICLE_fermion', 'MASS_1e-9', 'CHARGE_0', 'END_PARTICLE',
                        'PARTICLE', 'PARTICLE_ID_6', 'TYPE_PARTICLE_fermion', 'MASS_1e0', 'CHARGE_-1', 'END_PARTICLE',
                    'END_FIELD',
                    'FIELD', 'FIELD_ID_2', 'TYPE_FIELD_fermion', 'DIM_1', 'GEN_3', 'SELF_CONJ_FALSE', 'CHIRALITY_right', 'SU3C_REP_1', 'SU2L_REP_1', 'U1Y_CHARGE_-1', 'QN_L_1', 'QN_B_0',
                        'PARTICLE', 'PARTICLE_ID_2', 'TYPE_PARTICLE_fermion', 'MASS_1e-4', 'CHARGE_-1', 'END_PARTICLE',
                        'PARTICLE', 'PARTICLE_ID_4', 'TYPE_PARTICLE_fermion', 'MASS_1e-1', 'CHARGE_-1', 'END_PARTICLE',
                        'PARTICLE', 'PARTICLE_ID_6', 'TYPE_PARTICLE_fermion', 'MASS_1e0', 'CHARGE_-1', 'END_PARTICLE',
                    'END_FIELD',
                    'FIELD', 'FIELD_ID_3', 'TYPE_FIELD_complex', 'DIM_2', 'GEN_1', 'SELF_CONJ_FALSE', 'CHIRALITY_none', 'SU3C_REP_1', 'SU2L_REP_2', 'U1Y_CHARGE_1', 'QN_L_0', 'QN_B_0',
                        'PARTICLE', 'PARTICLE_ID_7', 'TYPE_PARTICLE_complex', 'MASS_1e2', 'CHARGE_1', 'END_PARTICLE',
                        'PARTICLE', 'PARTICLE_ID_8', 'TYPE_PARTICLE_complex', 'MASS_1e2', 'CHARGE_0', 'END_PARTICLE',
                    'END_FIELD',
                'END_ITRACT',
            'ITRACT', 'ITRACT_ID_2'
        ]
        prefix_tokens_ids = [self.token_to_idx[tok] for tok in prefix_tokens_str]
        prefix_len = len(prefix_tokens_ids)

        sequences[:, :prefix_len] = torch.tensor(prefix_tokens_ids, dtype=torch.long, device=self.device).expand(B, -1)
        attn_mask[:, :prefix_len] = True
        
        grammar_states = [self.grammar.initial_state() for _ in range(B)]
        for tok_id in prefix_tokens_ids:
            for i in range(B):
                grammar_states[i] = self.grammar.step(grammar_states[i], tok_id)

        current_tokens = sequences[:, prefix_len - 1].unsqueeze(-1)
        past_key_values = None
        if self.cfg.MODEL_TYPE == 'hf':
            with torch.no_grad():
                outputs = self.policy(sequences[:, :prefix_len - 1], attention_mask=attn_mask[:, :prefix_len - 1].to(torch.long))
                past_key_values = outputs.past_key_values
        
        single_token_inference_time, grammar_mask_time = 0, 0
        
        for t_gen in range(prefix_len - 1, T - 1):
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
            
            dist = Categorical(logits=masked_logits / self.cfg.TEMPERATURE)
            next_toks = dist.sample()
            
            sequences[:, t_gen + 1] = next_toks
            attn_mask[:, t_gen + 1] = True
            
            for i in range(B):
                grammar_states[i] = self.grammar.step(grammar_states[i], next_toks[i].item())

            current_tokens = next_toks.unsqueeze(-1)
        
        # --- Final Forward Pass for Critic/Value/LogProbs with fine-grained timing ---
        # [TIME] Trajectory Generation: Full-sequence Forward + Value Head
        start_final_forward = time.perf_counter()
        with torch.no_grad():
            full_outputs = self.policy(sequences, attention_mask=attn_mask.to(torch.long))
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

        # [TIME] Trajectory Generation: Total
        start_time = time.perf_counter()
        sequences, attn_masks, log_probs_gen, value_vectors_old, gen_timings = self._generate_trajectories()
        step_timings['generate_trajectories_total'] = time.perf_counter() - start_time
        # Use the newly calculated total model time for the "Model" part, and keep the detailed breakdown
        step_timings['total_model_forward_time'] = gen_timings.pop('total_model_forward_time')
        step_timings.update(gen_timings)
        if torch.cuda.is_available(): torch.cuda.synchronize()

        # Initialize aggregators for batch timings
        env_timing_sum = defaultdict(float)
        reward_timing_sum = defaultdict(float)
        
        # [TIME] Environment & Reward: Total
        start_env_reward_time = time.perf_counter()
        rewards_objects = []
        for i in range(self.cfg.BATCH_SIZE):
            episode_outcome = self.env.run_episode(sequences[i, attn_masks[i]].tolist(), self.idx_to_token)
            reward_object = self.reward_shaper.calculate_rewards(episode_outcome, sequences[i])
            rewards_objects.append(reward_object)
            
            for k, v in episode_outcome.meta.get('timing', {}).items():
                env_timing_sum[k] += v
            for k, v in reward_object.diagnostics.get('timing', {}).items():
                reward_timing_sum[k] += v
        step_timings['env_and_reward_total'] = time.perf_counter() - start_env_reward_time

        # Calculate averages for batch timings and add to step_timings
        B = self.cfg.BATCH_SIZE
        for k, v in env_timing_sum.items():
            step_timings[f'avg_env_timing_{k}'] = v / B
        for k, v in reward_timing_sum.items():
            step_timings[f'avg_reward_timing_{k}'] = v / B

        batch_value_targets = torch.stack([r.per_token_value_targets for r in rewards_objects])
        batch_error_masks = torch.stack([r.error_signal_mask for r in rewards_objects])
        scalar_rewards_list = [r.scalar_total_reward for r in rewards_objects]

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
            outputs = self.policy(sequences, attention_mask=attn_masks.to(torch.long))
            logits, hidden_states = outputs.logits, outputs.hidden_states[-1]
            predicted_score_vectors = self.policy.value(hidden_states)

            value_loss = F.mse_loss(
                predicted_score_vectors[attn_masks],
                batch_value_targets[attn_masks]
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
                    replay_outputs = self.policy(replay_seqs, attention_mask=replay_masks.to(torch.long))
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
                        'original_replay_loss': replay_original_losses.mean().item()
                    }
        
        eos_count = (sequences == self.cfg.EOS_TOKEN_ID).any(dim=1).sum().item()
        ppo_loss_info = {
            'ppo_loss_start': ppo_loss_start,
            'ppo_loss_end': ppo_loss_end,
            'ppo_loss_diff': ppo_loss_start - ppo_loss_end # Calculate difference for easy tracking
        }
        step_result = {'eos_count': eos_count, 'replay_loss_info': replay_loss_info, 'step_timings': step_timings, 'ppo_loss_info': ppo_loss_info}
        
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
                        outcome=self.env.run_episode(sequences[i, attn_masks[i]].tolist(), self.idx_to_token),
                        sequence_tokens=[self.idx_to_token.get(tok.item(), "UNK") for tok in sequences[i, :effective_len]],
                        rewards_info=rewards_objects[i],
                        predicted_score_vectors=predicted_score_vectors[i, :effective_len].detach(),
                        per_token_instantaneous_rewards=rewards_objects[i].per_token_instantaneous_rewards[:effective_len].detach(),
                        per_token_target_scores=batch_value_targets[i, :effective_len].detach(),
                        averaged_check_type_scores=rewards_objects[i].diagnostics.get('r_success_tensor', torch.zeros(NUM_CHECKS)).cpu(),
                        print_sequence=self.cfg.PRINT_SEQUENCE,
                        print_detailed_checklist=should_print_detailed,
                        print_per_check_rewards_detail=self.cfg.PRINT_PER_CHECK_REWARDS_DETAIL,
                        step_number=self.step_count + 1,
                        timings=step_timings
                    )
        
        if self.step_count % 10 == 0:
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

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
                
                # Print Replay Buffer Metrics in the Step Summary (if available)
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