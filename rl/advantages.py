import torch
import numpy as np

def compute_advantages(rewards: torch.FloatTensor, values: torch.FloatTensor, gamma: float, gae_lambda: float) -> torch.FloatTensor:
    """Computes General Advantage Estimation (GAE)."""
    advantages = torch.zeros_like(rewards)
    last_advantage = 0.0
    
    for t in reversed(range(rewards.size(0))):
        next_value = values[t + 1] if t + 1 < values.size(0) else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        
        last_advantage = delta + gamma * gae_lambda * last_advantage
        advantages[t] = last_advantage
        
    return advantages