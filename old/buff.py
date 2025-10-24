# ppo_trainer.py (this is the complete, modified main script)
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
import copy

# ensure project modules are on path
dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, dir_path)
from grammar_generator import GrammarModelTrainer
from transformer import TransformerPolicy
from score_calculator import DEVICE
torch.cuda.empty_cache()


# Dummy logger for self-containment
class Logger:
    def __init__(self, debug=False):
        self.metrics = {}
        self.step_data = {}
        self.tb_writers = {} # Dummy
    def metric(self, name, kind, dest, tb):
        self.metrics[name] = {'kind': kind, 'dest': dest, 'tb': tb, 'values': []}
    def log(self, name, value):
        if name in self.metrics:
            self.step_data[name] = value
    def step(self, csv_log_path):
        for name, value in self.step_data.items():
             if name in self.metrics:
                self.metrics[name]['values'].append(value)
        self.step_data = {} # Clear for next step
    def print_metrics(self, *names):
        print("--- Step Metrics ---")
        # Always print all loss parts and key stats
        default_names = [
            "total_loss", "pg_loss", "v_loss", "her_loss", "her_v_loss", "her_pg_loss", "kl_loss", "db_loss",
            "avg_reward", "avg_seq_len", "avg_val", "percent_dead"
        ]
        output = []
        for name in default_names:
            if name in self.metrics and self.metrics[name]['values']:
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
    total_steps: int = 40_000
    batch_size: int = 1024
    learning_rate: float = 1e-4
    ppo_epochs: int = 4
    num_minibatches: int = 8
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip_coef: float = 0.20
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    pg_coef: float = 1.0
    sup_coef: float = 0.1
    max_grad_norm: float = 0.5
    seq_len: int = 160
    validation_bonus: float = 1000.0
    len_target: int = 160
    len_sigma: float = 10.0
    alive_r: float = 0.00
    d_model: int = 128
    nhead: int = 4
    layers: int = 4
    log_interval: int = 1
    save_interval: int = 100
    minibatch_size: int = field(init=False)
    
    her_capacity: int = 4096
    her_batch_size: int = 128
    her_coef: float = 0.5
    kl_coef: float = 0.02

    # Switch to Detailed Balance loss by default
    use_detailed_balance: bool = True

    def __post_init__(self):
        self.minibatch_size = self.batch_size // self.num_minibatches

@dataclass
class HERBuffer:
    """Hindsight Experience Replay buffer for successful sequences."""
    capacity: int
    batch_size: int
    device: torch.device
    sequences: list[torch.Tensor] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    def __len__(self):
        return len(self.sequences)

    def is_ready(self):
        return len(self) >= self.batch_size

    def add(self, new_sequences: torch.Tensor, new_scores: torch.Tensor):
        if new_sequences.shape[0] == 0:
            return
        new_sequences, new_scores = new_sequences.cpu(), new_scores.cpu()
        self.sequences.extend(list(new_sequences))
        self.scores.extend(list(new_scores.numpy()))

        sorted_experiences = sorted(zip(self.scores, self.sequences), key=lambda x: x[0], reverse=True)
        self.scores = [exp[0] for exp in sorted_experiences[:self.capacity]]
        self.sequences = [exp[1] for exp in sorted_experiences[:self.capacity]]

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_ready():
            raise ValueError("Not enough samples in HER buffer.")
        
        ranks = np.arange(len(self))
        p = 1 / (ranks + 1)
        p = p / p.sum()
        
        idxs = np.random.choice(len(self), self.batch_size, replace=False, p=p)
        
        sampled_seqs = torch.stack([self.sequences[i] for i in idxs]).to(self.device)
        sampled_scores = torch.tensor([self.scores[i] for i in idxs], dtype=torch.float32, device=self.device)
        return sampled_seqs, sampled_scores

def calculate_norms(model: nn.Module, grad: bool = False) -> tuple[float, float]:
    l1_norm, l2_norm = 0.0, 0.0
    for p in model.parameters():
        val = p.grad if grad and p.grad is not None else p.data
        if val is not None:
            l1_norm += torch.abs(val).sum().item()
            l2_norm += (val ** 2).sum().item()
    return l1_norm, l2_norm**0.5


def sequence_to_string(sequence: torch.Tensor, trainer) -> str:
    tokens = [trainer.idx_to_token[token_idx.item()] for token_idx in sequence if token_idx != trainer.dead_token_idx]
    return " ".join(tokens)


@torch.jit.script
def compute_gae_jit(rewards: torch.Tensor, values: torch.Tensor, is_alive: torch.Tensor, gamma: float, gae_lambda: float) -> torch.Tensor:
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
    is_alive_float = is_alive.float()
    for t in range(T - 2, -1, -1):
        next_alive_mask = is_alive_float[:, t + 1]
        next_val = values[:, t + 1] * next_alive_mask
        delta = rewards[:, t] + gamma * next_val - values[:, t]
        last_gae = delta + gamma * gae_lambda * last_gae * next_alive_mask
        advantages[:, t] = last_gae
    return advantages

def main():
    config = PPOConfig()
    
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.DEBUG if config.debug else logging.INFO)
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    datefmt = '%H:%M:%S'
    formatter = logging.Formatter(fmt, datefmt=datefmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)
    os.makedirs(config.log_dir, exist_ok=True)
    debug_log_path = os.path.join(config.log_dir, f"{config.run_name}.debug.log")
    fh = logging.FileHandler(debug_log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting PPO run '{config.run_name}' (debug={config.debug})")

    os.makedirs(config.log_dir, exist_ok=True)
    ckpt_dir = os.path.join(config.model_save_dir, config.run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    csv_log_path = os.path.join(config.log_dir, f"{config.run_name}.csv")
    model_ckpt_template = os.path.join(ckpt_dir, f"{config.run_name}_step_{{}}.pt")

    logger_obj = Logger(debug=config.debug)
    metrics = [
        ("avg_reward", "stat"), ("avg_seq_len", "stat"),
        ("pg_loss", "stat"), ("v_loss", "stat"), ("e_loss", "stat"), ("sup_loss", "stat"),
        ("total_loss", "stat"), ("learning_rate", "scalar"),
        ("avg_val", "stat"), ("percent_dead", "stat"),
        ("rollout_time", "stat"), ("optim_time", "stat"), ("total_step_time", "stat"), ("gae_time", "stat"),
        ("her_loss", "stat"), ("her_v_loss", "stat"), ("her_pg_loss", "stat"),
        ("kl_loss", "stat"), ("db_loss", "stat"),
    ]
    for name, kind in metrics:
        logger_obj.metric(name, kind, dest=csv_log_path, tb=True)

    trainer = GrammarModelTrainer(DEVICE, validation_bonus=config.validation_bonus, debug=config.debug)
    
    forward_policy = TransformerPolicy(
        vocab=trainer.vocab_size, d_model=256, nhead=8, layers=6, max_T=config.seq_len, debug=config.debug
    ).to(DEVICE)
    
    backward_policy = TransformerPolicy(
        vocab=trainer.vocab_size, d_model=256, nhead=8, layers=6, max_T=config.seq_len, debug=config.debug
    ).to(DEVICE)
    
    pretrained_policy = TransformerPolicy(
        vocab=trainer.vocab_size, d_model=256, nhead=8, layers=6, max_T=config.seq_len, debug=config.debug
    ).to(DEVICE)

    her_buffer = HERBuffer(
        capacity=config.her_capacity, batch_size=config.her_batch_size, device=DEVICE
    )

    logger.info("Applying initial weights (simulating pre-training)...")
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    forward_policy.apply(init_weights)
    backward_policy.apply(init_weights)
    
    pretrained_policy.load_state_dict(copy.deepcopy(forward_policy.state_dict()))
    pretrained_policy.eval()
    for param in pretrained_policy.parameters():
        param.requires_grad = False
    
    if hasattr(torch, 'compile'):
        logger.info("Using torch.compile() on the model.")
        forward_policy = torch.compile(forward_policy)
        backward_policy = torch.compile(backward_policy)

    optimizer_forward = optim.AdamW(forward_policy.parameters(), lr=config.learning_rate, eps=1e-5)
    optimizer_backward = optim.AdamW(backward_policy.parameters(), lr=config.learning_rate, eps=1e-5)

    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_forward, mode='min', factor=0.5, patience=50, min_lr=1e-6)

    start_time = time.time()
    for step in range(1, config.total_steps + 1):
        step_start = time.time()

        t0 = time.time()
        forward_policy.eval()
        with torch.no_grad(), torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type=='cuda')):
            sequences = trainer.generate_sequences(forward_policy, config.batch_size, config.seq_len)
            rewards = trainer.calculate_rewards_with_validation(
                sequences, gamma=config.gamma, alive_r=config.alive_r,
                len_target=config.len_target, len_sigma=config.len_sigma
            )
            on_policy_logits, on_policy_values, _ = forward_policy(sequences)
        rollout_time = time.time() - t0
        
        with torch.no_grad():
            B, T = sequences.shape
            term_mask = (sequences == trainer.eos_token_idx) | (sequences == trainer.dead_token_idx)
            term_mask[:, -1] = True
            first_term = torch.argmax(term_mask.int(), dim=1)
            is_alive = torch.arange(T, device=DEVICE).unsqueeze(0) <= first_term.unsqueeze(1)
            
            bonus = (rewards[torch.arange(B), first_term] - 1.0).clamp(min=0) / config.validation_bonus
            validation_scores = torch.zeros_like(bonus)
            last_tokens = sequences[torch.arange(B), first_term]
            successful = (last_tokens == trainer.eos_token_idx)
            validation_scores[successful] = bonus[successful]

            if successful.any() and config.her_coef > 0:
                her_buffer.add(sequences[successful], validation_scores[successful])
        
        orig_avg_reward = rewards.mean().item()
        
        t2 = time.time()
        advantages, returns = None, None
        if not config.use_detailed_balance:
            advantages = compute_gae_jit(rewards, on_policy_values, is_alive, config.gamma, config.gae_lambda)
            returns = advantages + on_policy_values
        gae_time = time.time() - t2

        # Restore legality_masks_full computation for KL loss
        with torch.no_grad():
            bos = torch.full((config.batch_size, 1), trainer.bos_token_idx, dtype=torch.long, device=DEVICE)
            prev_seq = torch.cat([bos, sequences[:, :-1]], dim=1)
            embed_history = trainer.embed_val_tensor[sequences]
            gstates_after_token = torch.cumsum(embed_history, dim=1)
            zero_state = torch.zeros((config.batch_size, 1), dtype=gstates_after_token.dtype, device=DEVICE)
            gstates_before_token = torch.cat([zero_state, gstates_after_token[:, :-1]], dim=1)
            grammar_ok_full = trainer.get_active_grammar_mask_vectorized(gstates_before_token, prev_seq)
            stack_ok_full = (gstates_before_token.unsqueeze(2) + trainer.embed_val_tensor) >= 0
            legality_masks_full = grammar_ok_full & stack_ok_full

        t3 = time.time()
        all_losses = {k: [] for k in ["pg", "v", "e", "sup", "total", "her", "her_v", "her_pg", "kl", "db"]}

        idxs = np.arange(config.batch_size)
        forward_policy.train()
        backward_policy.train()

        for ppo_epoch in range(config.ppo_epochs):
            np.random.shuffle(idxs)
            for mb_id in range(config.num_minibatches):
                mb_idx = idxs[mb_id * config.minibatch_size : (mb_id+1) * config.minibatch_size]
                
                optimizer_forward.zero_grad()
                optimizer_backward.zero_grad()
                
                with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type=='cuda')):
                    mb_seq = sequences[mb_idx]
                    mb_alive = is_alive[mb_idx] & (mb_seq != trainer.dead_token_idx)
                    if not mb_alive.any(): continue
                    
                    # --- Detailed Balance Loss ---
                    if config.use_detailed_balance:
                        mb_logits, mb_log_flows, _ = forward_policy(mb_seq)
                        mb_log_pf = F.log_softmax(mb_logits, dim=-1)
                        
                        rev_mb_seq = torch.flip(mb_seq, dims=[1])
                        rev_logits, _, _ = backward_policy(rev_mb_seq)
                        # The backward policy needs log_probs for its actions on the reversed sequence
                        rev_log_probs_dist = F.log_softmax(rev_logits, dim=-1)

                        db_losses = []
                        # Iterate over all valid transitions in the minibatch
                        for i in range(len(mb_idx)):
                            for t in range(config.seq_len - 1):
                                if not mb_alive[i, t+1]: break

                                s_t = mb_seq[i, t]
                                s_t_plus_1 = mb_seq[i, t+1]
                                
                                # log F(s_t) from forward policy
                                log_flow_t = mb_log_flows[i, t]
                                # log F(s_{t+1}) from forward policy
                                log_flow_t_plus_1 = mb_log_flows[i, t+1]
                                
                                # log P_F(s_{t+1} | s_t) from forward policy
                                log_pf_t = mb_log_pf[i, t, s_t_plus_1]
                                
                                # For P_B(s_t | s_{t+1}), state is s_{t+1}, parent is s_t
                                # In reversed sequence, state s_{t+1} is at index T-1-(t+1)
                                rev_t_idx = config.seq_len - 1 - (t+1)
                                log_pb_t = rev_log_probs_dist[i, rev_t_idx, s_t]
                                
                                db_error = log_flow_t + log_pf_t - log_flow_t_plus_1 - log_pb_t
                                db_losses.append(db_error**2)
                        
                        db_loss = torch.stack(db_losses).mean()
                        pg_loss = db_loss
                        v_loss = torch.tensor(0.0) # GAE-based value loss is not used in DB
                        all_losses["db"].append(db_loss.detach())
                    
                    # --- PPO Loss ---
                    else:
                        mb_logits, mb_vals, _ = forward_policy(mb_seq)
                        v_loss = F.mse_loss(mb_vals[mb_alive], returns[mb_idx][mb_alive])
                        mb_old_lp = F.log_softmax(on_policy_logits[mb_idx].float(), dim=-1).gather(2, mb_seq.unsqueeze(-1)).squeeze(-1)
                        mb_adv = advantages[mb_idx]
                        adv_alive = mb_adv[mb_alive]
                        adv_norm = (adv_alive - adv_alive.mean()) / (adv_alive.std() + 1e-8)
                        
                        new_lp = F.log_softmax(mb_logits.float(), dim=-1).gather(2, mb_seq.unsqueeze(-1)).squeeze(-1)[mb_alive]
                        old_lp_alive = mb_old_lp[mb_alive]
                        
                        ratio = (new_lp - old_lp_alive).exp()
                        pg1 = -adv_norm * ratio
                        pg2 = -adv_norm * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                        pg_loss = torch.max(pg1, pg2).mean()

                    # --- SHARED LOSSES (for both DB and PPO) ---
                    ent_dist = Categorical(logits=mb_logits[mb_alive])
                    e_loss = -ent_dist.entropy().mean()

                    her_loss_term = torch.tensor(0.0, device=DEVICE)
                    if her_buffer.is_ready() and config.her_coef > 0:
                        her_seqs, her_final_scores_norm = her_buffer.sample()
                        B_her, T_her = her_seqs.shape
                        her_logits, her_vals, _ = forward_policy(her_seqs)

                        her_term_mask = (her_seqs == trainer.eos_token_idx); her_term_mask[:,-1] = True
                        her_first_term = torch.argmax(her_term_mask.int(), dim=1)
                        her_is_alive = torch.arange(T_her, device=DEVICE).unsqueeze(0) <= her_first_term.unsqueeze(1)
                        
                        her_returns_norm = torch.zeros_like(her_vals)
                        for i in range(B_her):
                            end_idx = her_first_term[i]
                            her_returns_norm[i, :end_idx+1] = her_final_scores_norm[i]
                        
                        her_v_loss = F.mse_loss(her_vals[her_is_alive], her_returns_norm[her_is_alive])
                        
                        her_log_probs = F.log_softmax(her_logits.float(), dim=-1)
                        her_action_log_probs = her_log_probs.gather(2, her_seqs.unsqueeze(-1)).squeeze(-1)
                        her_pg_loss = -her_action_log_probs[her_is_alive].mean()
                        
                        her_loss_term = her_v_loss + her_pg_loss
                        all_losses["her_v"].append(her_v_loss.detach())
                        all_losses["her_pg"].append(her_pg_loss.detach())
                        all_losses["her"].append(her_loss_term.detach())

                    kl_loss_term = torch.tensor(0.0, device=DEVICE)
                    if config.kl_coef > 0:
                        with torch.no_grad():
                            pretrained_logits, _, _ = pretrained_policy(mb_seq)
                        
                        masks_prev = legality_masks_full[mb_idx]
                        neg_inf = torch.finfo(mb_logits.dtype).min
                        mb_logits_masked = mb_logits.masked_fill(~masks_prev, neg_inf)

                        kl_log_probs = F.log_softmax(mb_logits_masked.float(), dim=-1)
                        kl_pretrained_probs = F.softmax(pretrained_logits.float(), dim=-1)
                        
                        kl_loss = F.kl_div(kl_log_probs, kl_pretrained_probs, reduction='none', log_target=False).sum(dim=-1)
                        kl_loss_term = kl_loss[mb_alive].mean()
                        all_losses["kl"].append(kl_loss_term.detach())

                    total_loss = (
                        config.pg_coef * pg_loss +
                        config.vf_coef * v_loss +
                        config.ent_coef * e_loss +
                        config.her_coef * her_loss_term +
                        config.kl_coef * kl_loss_term
                    )
                    total_loss = torch.nan_to_num(total_loss, nan=0.0)

                scaler.scale(total_loss).backward()
                
                scaler.unscale_(optimizer_forward)
                nn.utils.clip_grad_norm_(forward_policy.parameters(), config.max_grad_norm)
                scaler.step(optimizer_forward)

                scaler.unscale_(optimizer_backward)
                nn.utils.clip_grad_norm_(backward_policy.parameters(), config.max_grad_norm)
                scaler.step(optimizer_backward)
                
                scaler.update()

                all_losses["pg"].append(pg_loss.detach())
                all_losses["v"].append(v_loss.detach())
                all_losses["e"].append(e_loss.detach())
                all_losses["total"].append(total_loss.detach())
        
        optim_time = time.time() - t3
        
        if step % config.log_interval == 0:
            l1, l2   = calculate_norms(forward_policy)
            gl1, gl2 = calculate_norms(forward_policy, grad=True)
            total_t  = time.time() - step_start

            avg_losses = {k: torch.stack(v).mean().item() if v else 0.0 for k, v in all_losses.items()}
            
            log_data = {
                "rollout_time": rollout_time, "gae_time": gae_time,
                "optim_time": optim_time, "total_step_time": total_t,
                "avg_reward": orig_avg_reward, "avg_seq_len": is_alive.sum(dim=1).float().mean().item(),
                "pg_loss": avg_losses["pg"], "v_loss": avg_losses["v"],
                "e_loss": avg_losses["e"], "total_loss": avg_losses["total"], 
                "learning_rate": optimizer_forward.param_groups[0]['lr'],
                "l1_norm": l1, "l2_norm": l2, "grad_l1_norm": gl1, "grad_l2_norm": gl2,
                "avg_val": validation_scores.mean().item(),
                "percent_dead": (~successful).float().mean().item()*100,
                "her_loss": avg_losses["her"], "her_v_loss": avg_losses["her_v"], "her_pg_loss": avg_losses["her_pg"],
                "kl_loss": avg_losses["kl"], "db_loss": avg_losses["db"],
            }
            for name, val in log_data.items():
                logger_obj.log(f"{name}_loss" if "loss" in name else name, val)

            logger_obj.step(csv_log_path)
            logger_obj.print_metrics()
            scheduler.step(avg_losses['total'])

        if step % config.save_interval == 0:
            ckpt_path = model_ckpt_template.format(step)
            model_to_save = forward_policy._orig_mod if hasattr(forward_policy, '_orig_mod') else forward_policy
            torch.save(model_to_save.state_dict(), ckpt_path)
            logger.info(f"Checkpoint saved: {ckpt_path}")

    for w in logger_obj.tb_writers.values():
        w.close()
    elapsed_min = (time.time() - start_time) / 60
    logger.info(f"Training complete in {elapsed_min:.2f} min.")
    del trainer


if __name__ == "__main__":
    main()