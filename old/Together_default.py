from __future__ import annotations

# ───────────────────── Imports & Global Setup ───────────────────────
import csv
import math
import os
import random
import threading
import time
import queue
import contextlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union
import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.distributions import Categorical
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import trange
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from particle_graph.training.old_classes import TorchGrammar, TorchGrammarBatchParser, ModelTester, Logger

# ─────────────────────────── Fast defaults ──────────────────────────
SEED, EPS = 42, 1e-9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported():
    _AMP_DTYPE = torch.bfloat16
else:
    _AMP_DTYPE = torch.float16
SCALER = amp.GradScaler(enabled=(DEVICE.type == "cuda"))
CAN_COMPILE = hasattr(torch, "compile")

# ───────────────────── Domain‑specific tokens ───────────────────────
TOKENS: List[str] = (
    ["BOS", "EOS"]
    + ["ITRACT", "END_ITRACT"]
    + [f"ITRACT_ID_{i}" for i in range(3)]
    + ["FIELD", "END_FIELD"]
    + [f"FIELD_ID_{i}" for i in range(3)]
    + [f"USE_FIELD_{i}" for i in range(3)]
    + ["PARTICLE", "END_PARTICLE"]
    + [f"PARTICLE_ID_{i}" for i in range(3)]
    + [f"USE_PARTICLE_{i}" for i in range(3)]
    + [
        "TYPE_DC", "TYPE_YUKAWA", "TYPE_VLF", "TYPE_PHI4", "TYPE_FF", "TYPE_FFDUAL",
        "TYPE_complex", "TYPE_real", "TYPE_fermion", "TYPE_vector",
        "MASS_1e0", "MASS_1e1", "MASS_1e2", "MASS_1e3",
        "CHARGE_-1", "CHARGE_0", "CHARGE_1",
        "DIM_1", "DIM_2", "DIM_3",
        "GEN_1", "GEN_2", "GEN_3",
        "SELF_CONJ_TRUE", "SELF_CONJ_FALSE",
        "CHIRALITY_left", "CHIRALITY_right", "CHIRALITY_na",
        "REP_SU3C_singlet", "REP_SU3C_fnd", "REP_SU3C_adj",
        "REP_SU2L_singlet", "REP_SU2L_fnd", "REP_SU2L_adj",
        "REP_U1Y_minus1", "REP_U1Y_0", "REP_U1Y_1",
        "QN_LeptonNumber_-1", "QN_LeptonNumber_0", "QN_LeptonNumber_1",
        "QN_BaryonNumber_-1", "QN_BaryonNumber_0", "QN_BaryonNumber_1",
    ]
)
TOKEN2ID = {t: i for i, t in enumerate(TOKENS)}
ID2TOK   = {i: t for t, i in TOKEN2ID.items()}
VOCAB_SIZE, BOS_TOKEN = len(TOKENS), TOKEN2ID["BOS"]

# Define constants for specific end tokens for reward signal
END_PARTICLE_TOKEN = TOKEN2ID["END_PARTICLE"]
END_FIELD_TOKEN = TOKEN2ID["END_FIELD"]
END_ITRACT_TOKEN = TOKEN2ID["END_ITRACT"]

grammar = TorchGrammar()
tr = lambda s, e: (TOKEN2ID[s], TOKEN2ID[e])  # tuple‐range helper

grammar.add_object(
    "Particle",
    [
        TOKEN2ID["PARTICLE"],
        tr("PARTICLE_ID_0", "PARTICLE_ID_2"),
        tr("TYPE_complex", "TYPE_vector"),
        tr("MASS_1e0", "MASS_1e3"),
        tr("CHARGE_-1", "CHARGE_1"),
        TOKEN2ID["END_PARTICLE"],
    ],
)
grammar.add_object(
    "Field",
    [
        TOKEN2ID["FIELD"],
        tr("FIELD_ID_0", "FIELD_ID_2"),
        tr("TYPE_complex", "TYPE_vector"),
        tr("DIM_1", "DIM_3"),
        tr("GEN_1", "GEN_3"),
        tr("SELF_CONJ_TRUE", "SELF_CONJ_FALSE"),
        tr("CHIRALITY_left", "CHIRALITY_na"),
        tr("REP_SU3C_singlet", "REP_SU3C_adj"),
        tr("REP_SU2L_singlet", "REP_SU2L_adj"),
        tr("REP_U1Y_minus1", "REP_U1Y_1"),
        tr("QN_LeptonNumber_-1", "QN_LeptonNumber_1"),
        tr("QN_BaryonNumber_-1", "QN_BaryonNumber_1"),
        {"Particle": lambda c: c > 0}, # "embed only Particle until lambda returns True, after you are able to continue or stop"
        TOKEN2ID["END_FIELD"],
    ],
)
grammar.add_object(
    "Interaction",
    [
        TOKEN2ID["ITRACT"],
        tr("ITRACT_ID_0", "ITRACT_ID_2"),
        tr("TYPE_DC", "TYPE_FFDUAL"),
        {"Field": lambda c: c > 0},
        TOKEN2ID["END_ITRACT"],
    ],
)

grammar.add_object(
    "Model",
    [
        TOKEN2ID["BOS"],
        {"Interaction": lambda c: c > 1},
        TOKEN2ID["EOS"],
    ],
)

ROOTS = ["Model"] # <- training selects among these randomly for seq, for each batch


# ─────────────────── Hyper‑parameters & Paths ───────────────────────
@dataclass(frozen=True)
class HParams:
    epochs: int = 200               # Total training updates for the combined objective
    batch_size: int = 64            # Vectorized batch size for rollouts
    gamma: float = 1.0              # Discount factor for RL
    value_coef: float = 0.05        # Coefficient for the value loss
    ent_coef: float = 0.02          # Coefficient for the entropy bonus
    grammar_loss_weight: float = 0.00 # Weight for the grammar adherence supervised signal
    lr: float = 4e-4                # Learning rate
    seq_len: int = 64               # Maximum sequence length for rollouts
    ckpt_every: int = 700           # Checkpoint saving frequency (in epochs)

hp = HParams()
dir_root = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
LOG_DIR      = dir_root / "particle" / "runs" # Changed log directory name to include runs subdirectory
CSV_NAME     = "cli_stats_combined.csv"
GRAPHICS_DIR = LOG_DIR / "graphics"


# ────────────────────── Utility helpers ─────────────────────────────
def ensure_dirs(paths: Iterable[Union[str, Path]]) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def safe_item(x: Union[float, torch.Tensor]) -> float:
    v = float(x.item() if isinstance(x, torch.Tensor) else x)
    return v if not (math.isnan(v) or math.isinf(v)) else -1.0


def load_state_dict_compat(
    model: nn.Module,
    path: Union[str, Path],
    *,
    device: torch.device = DEVICE
) -> None:
    sd = torch.load(path, map_location=device)
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)

# ─────────────── LR scheduler (warm‑cosine) ──────────────────────────
def warmup_cosine(opt: torch.optim.Optimizer, warm: int, total: int) -> LambdaLR:
    def f(step: int) -> float:
        if total == 0: return 1.0
        if step < warm: return step / max(1, warm)
        prog = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * prog))
    return LambdaLR(opt, f, last_epoch=-1)

# ─────────────────── Model definition ───────────────────
class TransformerPolicy(nn.Module):
    """Transformer‑encoder policy with actor & critic heads."""
    def __init__(
        self, vocab: int, *,
        d_model: int = 256,
        nhead: int = 4,
        layers: int = 3,
        max_T: int = 64,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_T, d_model)
        self.register_buffer("pos", torch.arange(max_T, dtype=torch.long))
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, 0.1,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.actor = nn.Linear(d_model, vocab)
        self.value = nn.Linear(d_model, 1)

    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """seq (B,T) → logits (B,T,V), values (B,T)"""
        B, T = seq.shape
        x = self.tok_emb(seq) + self.pos_emb(self.pos[:T].to(seq.device))
        h = self.encoder(x)
        return self.actor(h), self.value(h).squeeze(-1)


# ───────────── CSV logger ─────────────
class CSVThread(threading.Thread):
    def __init__(self, path: Path, header: List[str]):
        super().__init__(daemon=True)
        self.path = path
        self.header = header
        self.q = queue.Queue()

    def run(self) -> None:
        ensure_dirs([self.path.parent])
        with self.path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self.header)
            f.flush()
            while True:
                row = self.q.get()
                if row is None:
                    self.q.task_done()
                    break
                w.writerow([safe_item(x) for x in row])
                f.flush()
                self.q.task_done()


# ─────────────────── Optimizer Builder ───────────────────
def _build_adamw(params, lr):
    """Fused AdamW when available, else fallback."""
    try:
        return torch.optim.AdamW(params, lr=lr, fused=True)
    except (TypeError, AttributeError): # older torch or no CUDA/fused
        warnings.warn("Fused AdamW not available, falling back to non-fused AdamW.")
        return torch.optim.AdamW(params, lr=lr)

# ───────────────── Combined Objective Training ───────────────
def train_combined_objective(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    hp: HParams,
    *,
    T_max_seq: int | None = None,  # Max sequence length for rollouts
    debug: bool = False,  # Flag to control debug print statements
) -> Path:
    """Train model using combined PPO objective."""
    if T_max_seq is None:
        T_max_seq = hp.seq_len

    run_name  = datetime.now().strftime("comb_%Y%m%d_%H%M%S")
    log_dir   = LOG_DIR / run_name
    ckpt_dir  = log_dir / "checkpoints"
    ensure_dirs([log_dir, ckpt_dir, GRAPHICS_DIR])

    logger = Logger(
        log_dir,
        cli_interval    = 1,
        state_interval  = 1,
        weights_interval= hp.ckpt_every
    )
    logger.model = model

    model.train()
    if DEVICE.type == "cuda":
        dummy_input = torch.full((hp.batch_size, T_max_seq),
                                 BOS_TOKEN, dtype=torch.long, device=DEVICE)
        opt.zero_grad(set_to_none=True)
        with amp.autocast(device_type="cuda", dtype=_AMP_DTYPE):
            logits_d, vals_d = model(dummy_input)
            loss_d = (logits_d.sum() + vals_d.sum()) * 0.0
        SCALER.scale(loss_d).backward()
        SCALER.step(opt); SCALER.update(); opt.zero_grad(set_to_none=True)

    if CAN_COMPILE:
        print("Compiling model…")
        model = torch.compile(model)
        print("Compilation complete.")

    scheduler = warmup_cosine(opt, int(hp.epochs * 0.1), hp.epochs)
    NEG_TIMEOUT = -1.0

    for epoch in range(1, hp.epochs + 1):
        BS      = hp.batch_size
        parser  = TorchGrammarBatchParser(grammar, BS, device=DEVICE)
        parser.set_batch(random.choices(ROOTS, k=BS)) # determines maximum object level for each batch

        seqs    = torch.full((BS, T_max_seq + 1), BOS_TOKEN,
                             dtype=torch.long, device=DEVICE)
        alive   = torch.ones(BS, dtype=torch.bool, device=DEVICE)
        ep_len  = torch.zeros(BS, dtype=torch.long, device=DEVICE)

        logP_list, val_list, ent_list, rew_list, gram_list = [], [], [], [], []
        n_particles = torch.zeros(BS, device=DEVICE)
        n_fields    = torch.zeros(BS, device=DEVICE)
        n_itracts   = torch.zeros(BS, device=DEVICE)

        for t_step in range(T_max_seq):
            ep_len[alive] += 1
            input_prefix = seqs[:, : t_step + 1].clone()

            with amp.autocast(device_type="cuda", dtype=_AMP_DTYPE,
                              enabled=(DEVICE.type == "cuda")):
                logits_t, values_t = model(input_prefix)

            raw_logits = logits_t[:, -1].float()
            values     = values_t[:, -1].float()
            mask_l     = parser.next_token_mask(VOCAB_SIZE)

            no_valid     = parser.next_lens.eq(0)
            eligible     = alive & (~no_valid)
            if not eligible.any():
                break

            banned_indic = (mask_l[eligible] != 0.0).float()
            pen_per_row  = F.relu(raw_logits[eligible] * banned_indic).sum(-1)
            denom        = banned_indic.sum(-1).clamp_min(1.0)
            gram_loss_t  = torch.zeros(BS, device=DEVICE)
            gram_loss_t[eligible] = pen_per_row / denom
            gram_list.append(gram_loss_t)

            final_logits         = raw_logits + mask_l
            final_logits[~eligible] = 0.0

            actions       = torch.full((BS,), BOS_TOKEN, device=DEVICE, dtype=torch.long)
            logprob_batch = torch.zeros(BS, device=DEVICE)
            entropy_batch = torch.zeros(BS, device=DEVICE)

            idx_elig = eligible.nonzero(as_tuple=False).squeeze(1)
            if idx_elig.numel():
                dist      = Categorical(logits=final_logits[idx_elig])
                a_sel     = dist.sample()
                lp_sel    = dist.log_prob(a_sel)
                ent_sel   = dist.entropy()

                actions[idx_elig]       = a_sel
                logprob_batch[idx_elig] = lp_sel
                entropy_batch[idx_elig] = ent_sel

                n_particles[idx_elig] += (a_sel == TOKEN2ID["PARTICLE"]).float()
                n_fields[idx_elig]    += (a_sel == TOKEN2ID["FIELD"]).float()
                n_itracts[idx_elig]   += (a_sel == TOKEN2ID["ITRACT"]).float()

            valid_mask = parser.fast_is_valid(actions)
            step_rew   = torch.zeros(BS, device=DEVICE)

            if idx_elig.numel():
                valid_e   = valid_mask[idx_elig]
                idx_valid = idx_elig[valid_e]
                if idx_valid.numel():
                    chosen = actions[idx_valid]
                    rvals  = torch.full_like(chosen, 0.2, dtype=torch.float)

                    rvals[chosen == TOKEN2ID["PARTICLE"]] = 1.0
                    rvals[chosen == TOKEN2ID["FIELD"]]    = 2.0
                    rvals[chosen == TOKEN2ID["ITRACT"]]   = 3.0

                    completed_mask = (parser.next_lens[idx_valid] == 0)
                    rvals[completed_mask] += 1.0

                    if t_step == T_max_seq - 1:
                        incomplete_mask = (parser.next_lens[idx_valid] > 0)
                        rvals[incomplete_mask] += NEG_TIMEOUT

                    step_rew[idx_valid] = rvals

            rew_list.append(step_rew)
            logP_list.append(logprob_batch)
            val_list.append(values)
            ent_list.append(entropy_batch)
            seqs[:, t_step + 1] = actions

            next_alive = torch.zeros_like(alive)
            if idx_elig.numel():
                still_valid = valid_mask[idx_elig]
                next_alive[idx_elig[still_valid]] = True
            alive = next_alive
            if not alive.any():
                break

            logger.log({
                "state": f"epoch_{epoch}_step_{t_step}",
                "reward": step_rew.mean().item(),
                "action": actions.tolist(),
                "logits": raw_logits.tolist(),
            })
            logger.state_step()

        if not rew_list:
            lr_now = scheduler.get_last_lr()[0]
            scale = SCALER.get_scale()
            g_l1 = sum((p.grad / scale).abs().sum().item() for p in model.parameters() if p.grad is not None) if any(p.grad is not None for p in model.parameters()) else 0.0
            g_l2 = math.sqrt(sum(((p.grad / scale) ** 2).sum().item() for p in model.parameters() if p.grad is not None)) if any(p.grad is not None for p in model.parameters()) else 0.0
            
            logger.log({
                "learning_rate": lr_now,
                "n_updates": 0,
                "n_rollouts": 1,
                "n_samples": BS,
                "seq_len": 0,
                "reward": 0.0,
                "l1_norm": sum(p.abs().sum().item() for p in model.parameters()),
                "l2_norm": math.sqrt(sum((p ** 2).sum().item() for p in model.parameters())),
                "grad_l1_norm": g_l1,
                "grad_l2_norm": g_l2,
            })
            logger.cli_step()
            scheduler.step()
            continue

        stacked_logP  = torch.stack(logP_list, dim=1)
        stacked_vals  = torch.stack(val_list, dim=1)
        stacked_ents  = torch.stack(ent_list, dim=1)
        stacked_rews  = torch.stack(rew_list, dim=1)
        stacked_grams = (torch.stack(gram_list, dim=1)
                         if gram_list else torch.zeros_like(stacked_rews))

        disc_ret = torch.zeros_like(stacked_rews)
        R         = torch.zeros(BS, device=DEVICE)
        for rev in range(stacked_rews.size(1) - 1, -1, -1):
            R = stacked_rews[:, rev] + hp.gamma * R
            disc_ret[:, rev] = R

        advantages = disc_ret - stacked_vals

        ts_mat  = torch.arange(stacked_rews.size(1), device=DEVICE).unsqueeze(0).expand(BS, -1)
        valid_m = ts_mat < ep_len.unsqueeze(1)
        N_valid = valid_m.sum().clamp_min(1.0)

        if valid_m.sum() > 1:
            adv_flat = advantages[valid_m]
            adv_mean = adv_flat.mean()
            adv_std  = adv_flat.std().clamp_min(EPS)
            advantages = (advantages - adv_mean) / adv_std
        else:
            advantages = advantages - advantages[valid_m].mean()

        p_loss = -(stacked_logP[valid_m] * advantages.detach()[valid_m]).sum() / N_valid
        v_loss = F.mse_loss(stacked_vals[valid_m], disc_ret.detach()[valid_m])
        e_loss = -stacked_ents[valid_m].sum() / N_valid
        g_loss = stacked_grams[valid_m].sum() / N_valid

        total_loss = (
            p_loss
            + hp.value_coef * v_loss
            + hp.ent_coef   * e_loss
            + hp.grammar_loss_weight * g_loss
        )

        opt.zero_grad(set_to_none=True)
        
        if debug:
            print(f"\nDebug - Loss components:")
            print(f"p_loss: {safe_item(p_loss)}")
            print(f"v_loss: {safe_item(v_loss)}")
            print(f"e_loss: {safe_item(e_loss)}")
            print(f"g_loss: {safe_item(g_loss)}")
            print(f"total_loss before scaling: {safe_item(total_loss)}")
        
        SCALER.scale(total_loss).backward()
        
        if debug:
            print("\nDebug - Gradient stats:")
            temp_scale = SCALER.get_scale()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_scaled_norm = (param.grad / temp_scale if temp_scale > 0 else param.grad).norm().item()
                    display_name = name.removeprefix("_orig_mod.") if name.startswith("_orig_mod.") else name
                    print(f"{display_name}: grad_norm={grad_norm:.3e}, scaled_norm={grad_scaled_norm:.3e}")
        
        scale = SCALER.get_scale()
        g_l1 = sum((p.grad / scale).abs().sum().item() for p in model.parameters() if p.grad is not None)
        g_l2 = math.sqrt(sum(((p.grad / scale) ** 2).sum().item() for p in model.parameters() if p.grad is not None))
        
        if debug:
            print(f"\nDebug - Final gradient norms (scaled):")
            print(f"g_l1: {g_l1:.3e}")
            print(f"g_l2: {g_l2:.3e}")
        
        SCALER.step(opt); SCALER.update(); scheduler.step()

        logger.log({
            "n_updates":   1,
            "n_rollouts":  1,
            "n_samples":   BS,
            "seq_len":     ep_len.float().mean().item(),
            "pl":          safe_item(p_loss),
            "vl":          safe_item(v_loss),
            "el":          safe_item(-e_loss),
            "gl":          safe_item(g_loss),
            "tl":          safe_item(total_loss),
            "n_particles": n_particles.mean().item(),
            "n_fields":    n_fields.mean().item(),
            "n_itracts":   n_itracts.mean().item(),
            "reward":      disc_ret[:, 0].mean().item(),
            "learning_rate": scheduler.get_last_lr()[0],
            "state":       f"epoch_{epoch}",
            "l1_norm":     sum(p.abs().sum().item() for p in model.parameters()),
            "l2_norm":     math.sqrt(sum((p ** 2).sum().item() for p in model.parameters())),
            "grad_l1_norm": g_l1,
            "grad_l2_norm": g_l2,
            "logit_entropy": -torch.sum(
                F.softmax(raw_logits, -1) * F.log_softmax(raw_logits, -1)
            ).mean().item(),
        })
        logger.cli_step()

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[Combined] Epoch {epoch:>4} | "
                f"Reward={logger.reward.most_recent:.3f} (μ={logger.reward.mean:.3f}, σ={logger.reward.std:.3f}) | "
                f"AvgLen={logger.seq_len.most_recent:.2f} | "
                f"PL={logger.pl.most_recent:.3f} | "
                f"VL={logger.vl.most_recent:.3f} | "
                f"Ent={logger.el.most_recent:.3f} | "
                f"GramL={logger.gl.most_recent:.3f} | "
                f"TotL={logger.tl.most_recent:.3f} | "
                f"LR={logger.learning_rate:.2e}"
            )

        if epoch % hp.ckpt_every == 0 or epoch == hp.epochs:
            ckpt_path = ckpt_dir / f"ep{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"✅ Saved combined-objective checkpoint: {ckpt_path}")

    final_ckpt = ckpt_dir / f"ep{hp.epochs}.pt"
    if hp.epochs % hp.ckpt_every:
        torch.save(model.state_dict(), final_ckpt)
    print(f"Training complete. Final checkpoint: {final_ckpt}")
    return final_ckpt




# ───────────────────── Sampling ──────────────────
@torch.no_grad()
def sample(model: nn.Module, T: int, n: int = 4, temp: float = 1.0) -> None:
    model.eval()
    starts = {TOKEN2ID[s] for s in ["PARTICLE", "FIELD", "ITRACT"]}
    ends   = {TOKEN2ID[s] for s in ["END_PARTICLE", "END_FIELD", "END_ITRACT"]}

    for i in range(n):
        parser = TorchGrammarBatchParser(grammar, 1, device=DEVICE)
        parser.set_batch([random.choice(ROOTS)])
        seq = [BOS_TOKEN] 
        indent = 0
        print(f"\nSample {i+1}:\n", end="") 

        for token_index_in_sample in range(T):
            with (amp.autocast(device_type="cuda", dtype=_AMP_DTYPE) if DEVICE.type == "cuda"
                  else contextlib.nullcontext()):
                logits_all, _ = model(torch.tensor([seq], device=DEVICE))
            
            raw_logits = logits_all[0, -1].float()
            grammar_mask = parser.next_token_mask(VOCAB_SIZE)[0] 
            logits = raw_logits + grammar_mask

            if temp <= EPS: 
                tok = int(torch.argmax(logits).item())
            else: 
                probs = F.softmax(logits / temp, dim=0)
                if not torch.any(probs > 0) or torch.isinf(probs).all() or torch.isnan(probs).all(): # Check for bad distribution
                    allowed_indices = (grammar_mask == 0.0).nonzero(as_tuple=True)[0]
                    if allowed_indices.numel() > 0:
                        tok = int(allowed_indices[torch.randint(0, allowed_indices.numel(), (1,))].item())
                    else:
                         print(" ... (No grammatically allowed tokens by parser or bad distribution; terminating sample)]", end="")
                         break
                else:
                    tok = int(torch.multinomial(probs, 1).item())

            token_str = ID2TOK[tok]

            if token_index_in_sample == 0: 
                if tok in ends:
                    indent = max(0, indent - 1) 
                    print(f"{'  ' * indent}{token_str}", end="")
                elif tok in starts:
                    print(f"{'  ' * indent}{token_str}", end="") 
                    indent += 1
                else: 
                    print(f"{token_str}", end="")
            else: 
                if tok in ends:
                    indent = max(0, indent - 1)
                    print(f",\n{'  ' * indent}{token_str}", end="")
                elif tok in starts:
                    print(f",\n{'  ' * indent}{token_str}", end="")
                    indent += 1
                else:
                    print(f", {token_str}", end="")

            seq.append(tok)
            if not parser.fast_is_valid(torch.tensor([tok], device=DEVICE)).item():
                print("\n...(Invalid by grammar; terminating sample)]", end="")
                break
            if parser.next_lens[0] == 0: 
                print("\n(Grammar complete)", end="")
                break
        print() 
    print("-" * 30)
    model.train()


# ───────────────────────────── Main ────────────────────────────────
if __name__ == "__main__":
    ensure_dirs([LOG_DIR, GRAPHICS_DIR])

    print(f"Using device: {DEVICE}, AMP dtype: {_AMP_DTYPE if DEVICE.type == 'cuda' else 'N/A'}")
    print(f"Hyperparameters: {hp}")

    model = TransformerPolicy(VOCAB_SIZE, max_T=hp.seq_len).to(DEVICE)
    optimizer = _build_adamw(model.parameters(), hp.lr)
    
    print("\n=== Training with Combined Objective ===")
    final_checkpoint = train_combined_objective(model, optimizer, hp, T_max_seq=hp.seq_len, debug=False)
    
    print(f"\nLoading model from last checkpoint for sampling: {final_checkpoint}")
    load_state_dict_compat(model, final_checkpoint, device=DEVICE) 

    print("\nGenerating samples after combined training (temp=0.0 for greedy):")
    sample(model, hp.seq_len, n=5, temp=0.0)
    print("\nGenerating samples after combined training (temp=0.7):")
    sample(model, hp.seq_len, n=5, temp=0.7)

    print("Done.")