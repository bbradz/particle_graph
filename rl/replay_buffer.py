import torch
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from config import Config
from .dependencies import HIERARCHICAL_DEPS

class ReplayBuffer:
    """
    Stores a fixed size of past, successfully processed sequences and their
    rewards/targets for hindsight tracking of model convergence.
    """
    def __init__(self, max_size: int = 500, policy_dtype: torch.dtype = torch.float32):
        self.max_size = max_size
        self.policy_dtype = policy_dtype
        self.sequences = []
        self.attn_masks = []
        self.value_targets = []
        self.original_losses = [] # Store the initial MSE loss for this sequence
        self.storage_idx = 0
        self.policy_device = torch.device('cpu') # Placeholder, set by RLTrainer

    def store(self, seqs: torch.LongTensor, masks: torch.BoolTensor, targets: torch.FloatTensor, losses: torch.FloatTensor):
        """
        Stores all sequences from the provided batch. The calling agent is responsible
        for pre-selecting which sequences to store (e.g., only the best ones).
        """
        # Iterate over the provided batch and store each sequence
        for i in range(seqs.size(0)):
            # Ensure data is moved to CPU and detached before storage
            seq_to_store = seqs[i].detach().clone().cpu()
            mask_to_store = masks[i].detach().clone().cpu()
            target_to_store = targets[i].detach().clone().cpu()
            loss_to_store = losses[i].item()

            if len(self.sequences) < self.max_size:
                self.sequences.append(seq_to_store)
                self.attn_masks.append(mask_to_store)
                self.value_targets.append(target_to_store)
                self.original_losses.append(loss_to_store)
            else:
                # Overwrite the oldest element (circular buffer behavior)
                idx_to_replace = self.storage_idx % self.max_size
                self.sequences[idx_to_replace] = seq_to_store
                self.attn_masks[idx_to_replace] = mask_to_store
                self.value_targets[idx_to_replace] = target_to_store
                self.original_losses[idx_to_replace] = loss_to_store

            self.storage_idx += 1

    def is_ready(self, min_size: int) -> bool:
        """Checks if the buffer contains at least the minimum number of samples."""
        return len(self.sequences) >= min_size

    def sample(self, batch_size: int) -> Tuple[torch.LongTensor, torch.BoolTensor, torch.FloatTensor, torch.FloatTensor]:
        """Samples a batch and returns the tensors on the policy's device."""
        if not self.is_ready(batch_size):
            return None, None, None, None
            
        indices = random.sample(range(len(self.sequences)), batch_size)
        
        sampled_seqs = torch.stack([self.sequences[i] for i in indices]).to(self.policy_device)
        sampled_masks = torch.stack([self.attn_masks[i] for i in indices]).to(self.policy_device)
        sampled_targets = torch.stack([self.value_targets[i] for i in indices]).to(self.policy_device)
        sampled_losses = torch.tensor([self.original_losses[i] for i in indices], 
                                      dtype=self.policy_dtype, 
                                      device=self.policy_device)
        
        return sampled_seqs, sampled_masks, sampled_targets, sampled_losses

class PrioritizedReplayBuffer:
    """
    Curriculum-aware prioritized experience replay buffer.
    Stores experiences based on a curriculum-aware "surprise" score.
    """
    
    def __init__(self, config: Config, max_size: int = 1000, policy_dtype: torch.dtype = torch.float32):
        self.config = config
        self.max_size = max_size
        self.policy_dtype = policy_dtype
        self.alpha = config.REPLAY_ALPHA  # Prioritization exponent
        self.beta = config.REPLAY_BETA   # Importance sampling exponent
        self.beta_anneal_steps = config.REPLAY_BETA_ANNEAL_STEPS
        self.epsilon = config.REPLAY_EPSILON
        
        # Storage
        self.sequences = []
        self.attn_masks = []
        self.value_targets = []
        self.original_losses = []
        self.priorities = []
        self.curriculum_scores = []
        self.storage_idx = 0
        self.policy_device = torch.device('cpu')
        
        # Curriculum tracking
        self.current_phase = 1
        self.step_count = 0
    
    def update_curriculum_info(self, phase: int, step: int):
        """Update curriculum information for scoring."""
        self.current_phase = phase
        self.step_count = step
    
    def _calculate_surprise_score(self, targets: torch.FloatTensor, predicted: torch.FloatTensor, 
                                active_checks: List[str], success_scores: torch.FloatTensor = None) -> float:
        """Calculate curriculum-aware surprise score for prioritization with success bonus."""
        from Token2Model.check import CHECK_TO_IDX
        
        # Get indices of active checks
        active_indices = []
        for check_name in active_checks:
            if check_name in CHECK_TO_IDX:
                active_indices.append(CHECK_TO_IDX[check_name])
        
        if not active_indices:
            return 0.0
        
        # Calculate surprise only for active checks
        active_targets = targets[active_indices]
        active_predicted = predicted[active_indices]
        
        # Temporal difference error (surprise)
        td_error = torch.abs(active_targets - active_predicted).mean().item()
        
        # Curriculum weighting: higher weight for checks that are newly introduced
        curriculum_weight = 1.0
        for check_name in active_checks:
            if check_name in CHECK_TO_IDX:
                check_idx = CHECK_TO_IDX[check_name]
                level = HIERARCHICAL_DEPS.get_check_hierarchical_level(check_name)
                if level == self.current_phase:
                    curriculum_weight += 0.5  # Boost for newly introduced checks
        
        base_score = td_error * curriculum_weight
        
        # Apply success bonus if success scores are provided
        if success_scores is not None:
            # Calculate average success score across active checks
            active_success_scores = success_scores[active_indices]
            mean_success_score = active_success_scores.mean().item()
            
            # If the sequence shows any sign of success, add a large bonus
            if mean_success_score > 0.05:  # Small threshold to avoid rewarding near-zero scores
                base_score += self.config.PER_SUCCESS_BONUS
        
        return base_score
    
    def store(self, seqs: torch.LongTensor, masks: torch.BoolTensor, targets: torch.FloatTensor, 
              losses: torch.FloatTensor, predicted: torch.FloatTensor = None, active_checks: List[str] = None,
              success_scores: torch.FloatTensor = None):
        """
        Store experiences with curriculum-aware prioritization and success bonus.
        """
        if not self.config.PRIORITIZED_REPLAY_ENABLED:
            # Fallback to regular replay buffer behavior
            for i in range(seqs.size(0)):
                self._store_single(seqs[i], masks[i], targets[i], losses[i], 1.0, 0.0)
            return
        
        # Calculate surprise scores for prioritization
        for i in range(seqs.size(0)):
            if predicted is not None and active_checks is not None:
                # Pass success scores if available
                seq_success_scores = success_scores[i] if success_scores is not None else None
                surprise_score = self._calculate_surprise_score(
                    targets[i], predicted[i], active_checks, seq_success_scores
                )
            else:
                surprise_score = losses[i].item()
            
            # Convert to priority
            priority = (surprise_score + self.epsilon) ** self.alpha
            
            self._store_single(seqs[i], masks[i], targets[i], losses[i], priority, surprise_score)
    
    def _store_single(self, seq: torch.Tensor, mask: torch.Tensor, target: torch.Tensor, 
                     loss: torch.Tensor, priority: float, curriculum_score: float):
        """Store a single experience."""
        # Move to CPU and detach
        seq_to_store = seq.detach().clone().cpu()
        mask_to_store = mask.detach().clone().cpu()
        target_to_store = target.detach().clone().cpu()
        loss_to_store = loss.item()
        
        if len(self.sequences) < self.max_size:
            self.sequences.append(seq_to_store)
            self.attn_masks.append(mask_to_store)
            self.value_targets.append(target_to_store)
            self.original_losses.append(loss_to_store)
            self.priorities.append(priority)
            self.curriculum_scores.append(curriculum_score)
        else:
            # Overwrite oldest element
            idx_to_replace = self.storage_idx % self.max_size
            self.sequences[idx_to_replace] = seq_to_store
            self.attn_masks[idx_to_replace] = mask_to_store
            self.value_targets[idx_to_replace] = target_to_store
            self.original_losses[idx_to_replace] = loss_to_store
            self.priorities[idx_to_replace] = priority
            self.curriculum_scores[idx_to_replace] = curriculum_score
        
        self.storage_idx += 1
    
    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples."""
        return len(self.sequences) >= min_size
    
    def sample(self, batch_size: int) -> Tuple[torch.LongTensor, torch.BoolTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Sample a batch using prioritized sampling.
        Returns (sequences, masks, targets, losses, importance_weights)
        """
        if not self.is_ready(batch_size):
            return None, None, None, None, None
        
        if not self.config.PRIORITIZED_REPLAY_ENABLED:
            # Uniform sampling
            indices = random.sample(range(len(self.sequences)), batch_size)
            importance_weights = torch.ones(batch_size, device=self.policy_device)
        else:
            # Prioritized sampling
            priorities = np.array(self.priorities)
            probabilities = priorities / priorities.sum()
            
            indices = np.random.choice(len(self.sequences), batch_size, p=probabilities)
            
            # Calculate importance sampling weights
            beta = min(1.0, self.beta + (1.0 - self.beta) * (self.step_count / self.beta_anneal_steps))
            importance_weights = (len(self.sequences) * probabilities[indices]) ** (-beta)
            importance_weights = torch.tensor(importance_weights, device=self.policy_device, dtype=self.policy_dtype)
            importance_weights = importance_weights / importance_weights.max()  # Normalize
        
        # Sample the data
        sampled_seqs = torch.stack([self.sequences[i] for i in indices]).to(self.policy_device)
        sampled_masks = torch.stack([self.attn_masks[i] for i in indices]).to(self.policy_device)
        sampled_targets = torch.stack([self.value_targets[i] for i in indices]).to(self.policy_device)
        sampled_losses = torch.tensor([self.original_losses[i] for i in indices], 
                                      dtype=self.policy_dtype, device=self.policy_device)
        
        return sampled_seqs, sampled_masks, sampled_targets, sampled_losses, importance_weights
    
    def update_priorities(self, indices: List[int], new_priorities: List[float]):
        """Update priorities for sampled experiences."""
        if not self.config.PRIORITIZED_REPLAY_ENABLED:
            return
        
        for idx, priority in zip(indices, new_priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = (priority + self.epsilon) ** self.alpha
    
    def get_buffer_stats(self) -> Dict[str, float]:
        """Get statistics about the replay buffer."""
        if not self.sequences:
            return {}
        
        priorities = np.array(self.priorities)
        curriculum_scores = np.array(self.curriculum_scores)
        
        return {
            'buffer_size': len(self.sequences),
            'max_size': self.max_size,
            'priority_mean': float(priorities.mean()),
            'priority_std': float(priorities.std()),
            'curriculum_score_mean': float(curriculum_scores.mean()),
            'curriculum_score_std': float(curriculum_scores.std()),
            'current_phase': self.current_phase,
            'step_count': self.step_count
        }