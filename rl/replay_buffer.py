import torch
import random
from typing import List, Dict, Tuple, Optional

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
        """Stores one randomly selected sequence from the batch."""
        B = seqs.size(0)
        # Select a random sequence index to store
        idx = random.randint(0, B - 1)
        
        # Ensure data is moved to CPU and detached before storage
        seq_to_store = seqs[idx].detach().clone().cpu()
        mask_to_store = masks[idx].detach().clone().cpu()
        target_to_store = targets[idx].detach().clone().cpu()
        loss_to_store = losses[idx].item()
        
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