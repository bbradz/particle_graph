# ppo_trainer.py (main script)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
from dataclasses import dataclass, field
import numpy as np
import time
import os
import sys
import logging
from datetime import datetime
import torch.amp

# ensure project modules are on path
# Assuming grammar_generator, transformer, logging_funcs, and score_calculator are in the same directory
# For a self-contained example, these would need to be defined or stubbed.
# Since the error is in the main script logic, we can proceed.
dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, dir_path)
from grammar_generator import GrammarModelTrainer
from transformer import TransformerPolicy
from score_calculator import DEVICE
torch.cuda.empty_cache()


# Dummy logger for self-containment if logging_funcs.py is not available
class Logger:
    def __init__(self, debug=False):
        self.metrics = {}
        self.step_data = {}
        self.tb_writers = {} # Dummy
    def metric(self, name, kind, dest, tb):
        # This will store the configuration for each metric
        self.metrics[name] = {'kind': kind, 'dest': dest, 'tb': tb, 'values': []}
    def log(self, name, value):
        if name in self.metrics:
            self.step_data[name] = value
    def step(self, csv_log_path):
        # In a real logger, this would write to CSV and TB
        for name, value in self.step_data.items():
             if name in self.metrics:
                self.metrics[name]['values'].append(value)
        self.step_data = {} # Clear for next step
    def print_metrics(self, *names):
        print("--- Step Metrics ---")
        output = []
        for name in names:
            if name in self.metrics and self.metrics[name]['values']:
                # For stat metrics, show the last value logged
                last_val = self.metrics[name]['values'][-1]
                output.append(f"{name}: {last_val:.4f}")
        print(" | ".join(output))

@dataclass
class PPOConfig:
    """Hyperparameters for PPO training."""
    debug: bool = False
    run_name: str = f"ppo_transformer_{int(time.time())}"
    log_dir: str = "runs"
    model_save_dir: str = "models"
    total_steps: int = 40_000   # train long enough to polish
    batch_size: int = 1024      # larger batches → stabler KL
    learning_rate: float = 1e-4   # reduced learning rate for stability
    ppo_epochs: int = 32          # less passes per batch
    num_minibatches: int = 8
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip_coef: float = 0.20 # lets policy move farther once advantages get big
    ent_coef: float = 0.005 # lower entropy = surer decisions
    vf_coef: float = 1.0
    pg_coef: float = 0.5
    sup_coef: float = 1.0      # stronger illegal-mass penalty
    max_grad_norm: float = 0.1  # much stricter gradient clipping
    seq_len: int = 400
    validation_bonus: float = 1000.0
    len_target: int = 329
    len_sigma: float = 10.0
    alive_r: float = 0.00
    d_model: int = 128
    nhead: int = 4
    layers: int = 4
    log_interval: int = 1
    save_interval: int = 100
    minibatch_size: int = field(init=False)
    max_ratio: float = 20.0 # maximum allowed ratio for policy loss
    advantage_clip: float = 10.0  # clip advantages to prevent extreme values

    def __post_init__(self):
        self.minibatch_size = self.batch_size // self.num_minibatches


def calculate_norms(model: nn.Module, grad: bool = False) -> tuple[float, float]:
    """Calculates L1 and L2 norms for model parameters or gradients."""
    l1_norm = 0.0
    l2_norm = 0.0
    for p in model.parameters():
        val = p.grad if grad and p.grad is not None else p.data
        if val is not None:
            l1_norm += torch.abs(val).sum().item()
            l2_norm += (val ** 2).sum().item()
    return l1_norm, l2_norm**0.5


def sequence_to_string(sequence: torch.Tensor, trainer) -> str:
    """Convert a token sequence to a readable string."""
    tokens = []
    for token_idx in sequence:
        if token_idx == trainer.dead_token_idx:
            break
        token_str = trainer.idx_to_token[token_idx.item()]
        tokens.append(token_str)
    return " ".join(tokens)

@torch.jit.script
def compute_gae_jit(rewards: torch.Tensor, values: torch.Tensor, is_alive: torch.Tensor, gamma: float, gae_lambda: float) -> torch.Tensor:
    """
    Computes Generalized Advantage Estimation (GAE) in a JIT-compiled function for performance.
    Correctly handles batch processing and resets advantages for terminated sequences.
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    # last_gae must be a per-batch item tensor
    last_gae = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
    
    # is_alive is bool, needs to be float for multiplication
    is_alive_float = is_alive.float()

    for t in range(T - 2, -1, -1):
        # The value of the next state is 0 if the state is terminal
        next_alive_mask = is_alive_float[:, t + 1]
        next_val = values[:, t + 1] * next_alive_mask
        
        # Calculate delta
        delta = rewards[:, t] + gamma * next_val - values[:, t]
        
        # Update GAE. If the next state is not alive, the recursive part of GAE is 0.
        last_gae = delta + gamma * gae_lambda * last_gae * next_alive_mask
        advantages[:, t] = last_gae
        
    return advantages

def main():
    # 1) Load config
    config = PPOConfig()

    # 2) Setup Python logging
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.DEBUG if config.debug else logging.INFO)
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    datefmt = '%H:%M:%S'
    formatter = logging.Formatter(fmt, datefmt=datefmt)
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)
    # File handler
    os.makedirs(config.log_dir, exist_ok=True)
    debug_log_path = os.path.join(config.log_dir, f"{config.run_name}.debug.log")
    fh = logging.FileHandler(debug_log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)

    logger = logging.getLogger(__name__)
    logger.info(f"Starting PPO run '{config.run_name}' (debug={config.debug})")
    logger.debug(f"Debug log file: {debug_log_path}")

    # 3) Prepare output dirs
    os.makedirs(config.log_dir, exist_ok=True)
    ckpt_dir = os.path.join(config.model_save_dir, config.run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    csv_log_path = os.path.join(config.log_dir, f"{config.run_name}.csv")
    model_ckpt_template = os.path.join(ckpt_dir, f"{config.run_name}_step_{{}}.pt")

    # 4) Setup CSV & TB logger
    logger_obj = Logger(debug=config.debug)
    metrics = [
        ("avg_reward", "stat"), ("avg_seq_len", "stat"),
        ("pg_loss", "stat"), ("v_loss", "stat"), ("e_loss", "stat"), ("sup_loss", "stat"),
        ("total_loss", "stat"), ("learning_rate", "scalar"),
        ("l1_norm", "scalar"), ("l2_norm", "scalar"),
        ("grad_l1_norm", "scalar"), ("grad_l2_norm", "scalar"),
        ("avg_val", "stat"), ("percent_dead", "stat"),
        ("rollout_time", "stat"), ("gae_time", "stat"),
        ("optim_time", "stat"), ("total_step_time", "stat")
    ]
    for name, kind in metrics:
        logger_obj.metric(name, kind, dest=csv_log_path, tb=True)

    # 5) Init trainer, model, optimizer, scaler
    trainer = GrammarModelTrainer(DEVICE, validation_bonus=config.validation_bonus, debug=config.debug)
    trainer.max_len_schedule = lambda step: config.seq_len
    model = TransformerPolicy(
        vocab=trainer.vocab_size,
        d_model=256,
        nhead=8,
        layers=6,
        max_T=config.seq_len, debug=config.debug
    ).to(DEVICE)
    
    # Initialize weights for stability
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
    
    model.apply(init_weights)

    # OPTIMIZATION: Use torch.compile for end-to-end model optimization (PyTorch 2.0+).
    # This fuses kernels and significantly reduces Python overhead.
    if hasattr(torch, 'compile'):
        logger.info("Using torch.compile() on the model.")
        model = torch.compile(model)
    else:
        logger.info("torch.compile() not available. Running with eager mode model.")

    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, eps=1e-5)
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))
    
    # Add learning rate scheduler for stability
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, 
        min_lr=1e-6
    )

    start_time = time.time()
    for step in range(1, config.total_steps + 1):
        step_start = time.time()

        # Rollout
        t0 = time.time()
        model.eval()
        with torch.no_grad(), torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type=='cuda')):
            curr_T = trainer.max_len_schedule(step)
            sequences = trainer.generate_sequences(model, config.batch_size, curr_T)
            rewards = trainer.calculate_rewards_with_validation(
                sequences,
                gamma=config.gamma,
                alive_r=config.alive_r,
                len_target=config.len_target,
                len_sigma=config.len_sigma
            )
            logits, values, _ = model(sequences)
        rollout_time = time.time() - t0

        # Terminal processing
        with torch.no_grad():
            B, T = sequences.shape
            term_mask = (sequences == trainer.eos_token_idx) | (sequences == trainer.dead_token_idx)
            term_mask[:, -1] = True
            first_term = torch.argmax(term_mask.int(), dim=1)
            is_alive = torch.arange(T, device=DEVICE).unsqueeze(0) <= first_term.unsqueeze(1)
            last_tokens = sequences[torch.arange(B), first_term]
            successful = last_tokens == trainer.eos_token_idx
            final_rewards = rewards[torch.arange(B), first_term]
            bonus = (final_rewards - 1.0).clamp(min=0) / config.validation_bonus
            validation_scores = torch.zeros_like(bonus)
            validation_scores[successful] = bonus[successful]

        orig_avg_reward = rewards.mean().item()

        # GAE
        t2 = time.time()
        # OPTIMIZATION: Use the JIT-compiled GAE function. This is much faster than the python loop.
        advantages = compute_gae_jit(rewards, values, is_alive, config.gamma, config.gae_lambda)
        returns = advantages + values
        gae_time = time.time() - t2

        # OPTIMIZATION: Pre-compute legality masks for the entire batch ONCE before PPO epochs.
        # This avoids massive redundant computation inside the loops.
        with torch.no_grad():
            bos = torch.full((config.batch_size, 1), trainer.bos_token_idx, dtype=torch.long, device=DEVICE)
            prev_seq = torch.cat([bos, sequences[:, :-1]], dim=1)
            
            # Reconstruct the history of grammar states (embed_vals) at the start of each step
            embed_history = trainer.embed_val_tensor[sequences]
            gstates_after_token = torch.cumsum(embed_history, dim=1)
            zero_state = torch.zeros((config.batch_size, 1), dtype=gstates_after_token.dtype, device=DEVICE)
            gstates_before_token = torch.cat([zero_state, gstates_after_token[:, :-1]], dim=1)

            # Now compute the full [B, T, V] mask tensor for the entire rollout
            grammar_ok_full = trainer.get_active_grammar_mask_vectorized(gstates_before_token, prev_seq)
            stack_ok_full = (gstates_before_token.unsqueeze(2) + trainer.embed_val_tensor) >= 0
            legality_masks_full = grammar_ok_full & stack_ok_full

        # Optimization
        t3 = time.time()
        # OPTIMIZATION: Aggregate losses across all minibatches for more stable logging.
        all_pg_losses, all_v_losses, all_e_losses, all_sup_losses, all_total_losses = [], [], [], [], []
        
        idxs = np.arange(config.batch_size)
        model.train()
        for ppo_epoch in range(config.ppo_epochs):
            np.random.shuffle(idxs)
            for mb_id in range(config.num_minibatches):
                mb_idx = idxs[mb_id * config.minibatch_size : (mb_id+1) * config.minibatch_size]
                mb_seq = sequences[mb_idx]

                mb_old_lp = F.log_softmax(logits[mb_idx].float(), dim=-1).gather(2, mb_seq.unsqueeze(-1)).squeeze(-1)
                
                mb_adv = advantages[mb_idx]
                mb_ret = returns[mb_idx]
                mb_alive = is_alive[mb_idx] & (mb_seq != trainer.dead_token_idx)

                if not mb_alive.any():
                    continue

                # OPTIMIZATION: Instead of re-computing masks, just slice the pre-computed tensor.
                masks_prev = legality_masks_full[mb_idx]

                # Forward & loss terms
                with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type=='cuda')):
                    mb_logits, mb_vals, _ = model(mb_seq)

                    raw_probs = F.softmax(mb_logits.float(), dim=-1)
                    illegal_mass = raw_probs.masked_fill(masks_prev, 0.0).sum(dim=-1)
                    sup_loss = illegal_mass[mb_alive].mean() if mb_alive.any() else torch.tensor(0.0, device=DEVICE)

                    neg_inf = torch.finfo(mb_logits.dtype).min
                    mb_logits = mb_logits.masked_fill(~masks_prev, neg_inf)
                    mb_logits = torch.nan_to_num(mb_logits, nan=neg_inf, posinf=0.0, neginf=neg_inf)

                    adv_alive = mb_adv[mb_alive]
                    if adv_alive.numel() > 0:
                        adv_std = adv_alive.std(unbiased=False)
                        if adv_std < 1e-6:
                            adv_norm = torch.zeros_like(adv_alive)
                        else:
                            adv_norm = (adv_alive - adv_alive.mean()) / (adv_std + 1e-8)
                            adv_norm = torch.clamp(adv_norm, -config.advantage_clip, config.advantage_clip)
                    else:
                        adv_norm = torch.zeros_like(adv_alive)

                    new_lp_full = F.log_softmax(mb_logits.float(), dim=-1)
                    new_lp = new_lp_full.gather(2, mb_seq.unsqueeze(-1)).squeeze(-1)[mb_alive]
                    old_lp_alive = mb_old_lp[mb_alive]
                    
                    log_ratio = new_lp - old_lp_alive
                    log_ratio = torch.clamp(log_ratio, -torch.log(torch.tensor(config.max_ratio, device=DEVICE)), torch.log(torch.tensor(config.max_ratio, device=DEVICE)))
                    ratio = log_ratio.exp()
                    
                    pg1 = -adv_norm * ratio
                    pg2 = -adv_norm * torch.clamp(ratio, 1-config.clip_coef, 1+config.clip_coef)
                    pg_loss = torch.max(pg1, pg2).mean()

                    v_loss = F.mse_loss(mb_vals[mb_alive], mb_ret[mb_alive])

                    if mb_alive.any():
                        safe_logits = mb_logits[mb_alive].clone()
                        safe_logits = torch.where(safe_logits == neg_inf, torch.tensor(-1e6, device=DEVICE), safe_logits)
                        safe_logits = torch.nan_to_num(safe_logits, nan=-1e6, posinf=0.0, neginf=-1e6)
                        ent = Categorical(logits=safe_logits).entropy().mean()
                    else:
                        ent = torch.tensor(0.0, device=DEVICE)
                    e_loss = -ent

                    sup_coef = 1.0

                    total_loss = (
                        config.pg_coef * pg_loss +
                        config.vf_coef * v_loss +
                        config.ent_coef * e_loss +
                        sup_coef * sup_loss
                    )
                    total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=1e6, neginf=-1e6)

                if not torch.isfinite(total_loss):
                    logger.warning(f"Non-finite loss at step={step}, epoch={ppo_epoch}, mb={mb_id}; skipping")
                    continue

                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)

                has_nan_grad = False
                for param in model.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    logger.warning(f"NaN gradients detected at step={step}, epoch={ppo_epoch}, mb={mb_id}; skipping")
                    optimizer.zero_grad()
                    # scaler.update() # Not needed here, will be updated below
                    continue
                
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

                # OPTIMIZATION: Collect all loss values for averaging later.
                all_pg_losses.append(pg_loss.detach())
                all_v_losses.append(v_loss.detach())
                all_e_losses.append(e_loss.detach())
                all_sup_losses.append(sup_loss.detach())
                all_total_losses.append(total_loss.detach())

        optim_time = time.time() - t3
        
        # Logging & examples
        if step % config.log_interval == 0:
            l1, l2   = calculate_norms(model)
            gl1, gl2 = calculate_norms(model, grad=True)
            total_t  = time.time() - step_start

            # OPTIMIZATION: Log the average of the collected losses for stable metrics.
            avg_pg_loss = torch.stack(all_pg_losses).mean().item() if all_pg_losses else 0.0
            avg_v_loss = torch.stack(all_v_losses).mean().item() if all_v_losses else 0.0
            avg_e_loss = torch.stack(all_e_losses).mean().item() if all_e_losses else 0.0
            avg_sup_loss = torch.stack(all_sup_losses).mean().item() if all_sup_losses else 0.0
            avg_total_loss = torch.stack(all_total_losses).mean().item() if all_total_losses else 0.0

            for name, val in [
                ("rollout_time", rollout_time), ("gae_time", gae_time),
                ("optim_time", optim_time), ("total_step_time", total_t),
                ("avg_reward", orig_avg_reward), ("avg_seq_len", is_alive.sum(dim=1).float().mean().item()),
                ("pg_loss", avg_pg_loss), ("v_loss", avg_v_loss),
                ("e_loss", avg_e_loss), ("sup_loss", avg_sup_loss),
                ("total_loss", avg_total_loss), ("learning_rate", optimizer.param_groups[0]['lr']),
                ("l1_norm", l1), ("l2_norm", l2), ("grad_l1_norm", gl1), ("grad_l2_norm", gl2),
                ("avg_val", validation_scores.mean().item()), ("percent_dead", (~successful).float().mean().item()*100)
            ]:
                logger_obj.log(name, val)
            logger_obj.step(csv_log_path)
            logger_obj.print_metrics(
                "total_loss", "pg_loss", "v_loss", "e_loss",
                "sup_loss", "avg_reward", "avg_seq_len",
                "avg_val", "percent_dead"
            )
            
            # Update learning rate scheduler
            scheduler.step(avg_total_loss)

        # Checkpoint
        if step % config.save_interval == 0:
            ckpt = model_ckpt_template.format(step)
            # When using torch.compile, we save the original model's state dict
            model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(model_to_save.state_dict(), ckpt)
            logger.info(f"Checkpoint saved: {ckpt}")

    # Cleanup
    for w in logger_obj.tb_writers.values():
        w.close()
    elapsed_min = (time.time() - start_time) / 60
    logger.info(f"Training complete in {elapsed_min:.2f} min.")
    # Explicitly clean up the trainer to close the multiprocessing pool
    del trainer


if __name__ == "__main__":
    main()