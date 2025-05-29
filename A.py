from __future__ import annotations

# ───────────────────────────── Imports ──────────────────────────────
# Standard library
import os
import re
import csv
import math
import time
import queue
import random
import warnings
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

# Third-party
import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange
import torch
from torch import amp
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from scipy.sparse.linalg import LinearOperator, eigsh

try:
    import matplotlib.pyplot as plt
    import matplotlib.style as style
except ModuleNotFoundError:
    plt = None 
    style = None 

try:
    from torch.linalg import lobpcg
except ImportError:
    from torch import lobpcg

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except (ModuleNotFoundError, ImportError):
    SDPBackend = None
    sdpa_kernel = None

warnings.filterwarnings("ignore")

# ───────────────────────────── Config ────────────────────────────
SEED = 42
EPS = 1e-9
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
RL_EPOCHS: int = 400
RL_BATCHES: int = 64
RL_GAMMA: float = 1.0
CKPT_EVERY_EPOCHS: int = 25
VALUE_LOSS_COEF: float = 0.5
ENT_COEFF: float = 0.02
LR: float = 4e-4
SEQ_LEN: int = 64

# Logging / I/O
DIR_NAME = Path(__file__).resolve().parent
LOG_DIR = DIR_NAME / "particle"
CSV_NAME = "cli_stats.csv"
STATE_CSV_NAME = "state_variables.csv"
GRAPHICS_DIR = "graphics"

# ────────────────────────── RNG & CUDA set-up ───────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if _DEVICE.type == "cuda":
    torch.cuda.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
if hasattr(torch, "_nested_tensor") and hasattr(torch._nested_tensor, "enable_nested_tensor"):
    torch._nested_tensor.enable_nested_tensor(False)  # type: ignore[attr-defined]
_SCALER = amp.GradScaler(enabled=_DEVICE.type == "cuda")

# ─────────────────────── Updated Vocab / Grammar ───────────────────
@dataclass(frozen=True)
class Constraint:
    trigger_token: int       # e.g. CREATE_MULTIPLET (34)
    trigger_ctx: str         # e.g. "multiplet"
    required_token: Optional[int]  # e.g. any finished-particle token (27..33)
    required_ctx: str        # context in which to count the requirement
    min_count: int           # how many times B must appear before A is allowed

TokenId = int
# Vocabulary map
token_map: Dict[TokenId, str] = {
    0:  "CREATE_PARTICLE", 
    1:  "PARTICLE_ID",      
    2:  "PARTICLE_TYPE",
    3:  "PARTICLE_FULL_NAME",
    4:  "PARTICLE_NAME", 
    5:  "PARTICLE_MASS E-10",
    6:  "PARTICLE_MASS E-9",  
    7:  "PARTICLE_MASS E-8", 
    8:  "PARTICLE_MASS E-7",
    9:  "PARTICLE_MASS E-6", 
    10: "PARTICLE_MASS E-5", 
    11: "PARTICLE_MASS E-4",
    12: "PARTICLE_MASS E-3", 
    13: "PARTICLE_MASS E-2", 
    14: "PARTICLE_MASS E-1",
    15: "PARTICLE_MASS E-0", 
    16: "PARTICLE_MASS E+1", 
    17: "PARTICLE_MASS E+2",
    18: "PARTICLE_MASS E+3", 
    19: "PARTICLE_MASS E+4", 
    20: "PARTICLE_MASS E+5",
    21: "PARTICLE_MASS E+6", 
    22: "PARTICLE_MASS E+7", 
    23: "PARTICLE_MASS E+8",
    24: "PARTICLE_MASS E+9", 
    25: "PARTICLE_MASS E+10",
    26: "PARTICLE_SELF_CONJUGATE",
    27:"QN -3",28:"QN -2",29:"QN -1",30:"QN 0",31:"QN 1",32:"QN 2",33:"QN 3",
    34: "CREATE_MULTIPLET",
    35:"MULTIPLET_ID",
    36:"MULTIPLET_NAME",
    37:"MULTIPLET_DIMENSION",
    38:"MULTIPLET_GENERATION",
    39:"MULTIPLET_SELF_CONJUGATE",
    40:"MULTIPLET_REP_1",
    41:"MULTIPLET_REP_2",
    42:"MULTIPLET_REP_3",
    43:"BOS",
}
VOCAB_SIZE: int = len(token_map)
BOS_TOKEN: int = 43

# Contexts
contexts = ["particle", "multiplet"]
context2idx = {c: i for i, c in enumerate(contexts)}

# Grammar: transitions
token_grammar: Dict[int, Dict[str, List[int]]] = {
    0: {"particle": [1]},
    1: {"particle": [2]},
    2: {"particle": [3]},
    3: {"particle": [4]},
    4: {"particle": list(range(5, 26))},
    **{m: {"particle": [26]} for m in range(5, 26)}, # mass terms
    26:{"particle": list(range(27, 34))},
    **{q: {"particle": [0,34], "multiplet": [0,34]} for q in range(27,34)}, # quantum number
    34:{"multiplet": [35]},
    35:{"multiplet": [36]},
    36:{"multiplet": [37]},
    37:{"multiplet": [38]},
    38:{"multiplet": [39]},
    39:{"multiplet": [40,41,42]},
    40:{"multiplet": list(range(27,34))},
    41:{"multiplet": list(range(27,34))},
    42:{"multiplet": list(range(27,34))},
    43:{"particle":[0],"multiplet":[34]},
}

constraints: List[Constraint] = [
    # “Before emitting 34 in multiplet, require at least 1 of [27..33] in multiplet”
    Constraint(trigger_token=34, trigger_ctx="multiplet",
               required_token=None, required_ctx="particle", min_count=1),
]

# ───────────────────────── Utility Helpers ─────────────────────────
class CSVLoggerThread(threading.Thread):
    """Background thread that pulls rows off a queue and writes to CSV."""
    def __init__(self, file_path: Union[str, Path], header: List[str]):
        super().__init__(daemon=True)
        self.file_path = file_path
        self.header    = header
        self.queue     = queue.Queue()

    def run(self):
        with open(self.file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.header)
            f.flush()
            while True:
                row = self.queue.get()
                if row is None:
                    break
                writer.writerow(row)
                f.flush()

def make_policy(seq_len: int, *, compile_ok: bool = True) -> TransformerPolicy:
    model = TransformerPolicy(VOCAB_SIZE, len(contexts), max_seq_len=seq_len)
    if compile_ok and hasattr(torch, "compile"):
        model = TransformerPolicy.compile(model)
    return model.to(_DEVICE)

def _unwrap(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)

def ensure_dirs(paths: Iterable[Union[str, Path]]) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def save_state_dict(model: nn.Module, path: Union[str, Path]) -> None:
    path = Path(path)
    ensure_dirs([path.parent])
    torch.save(_unwrap(model).state_dict(), path)

def load_state_dict_compat(model: nn.Module, path: Union[str, Path], *, device: torch.device = _DEVICE) -> None:
    sd = torch.load(Path(path), map_location=device)
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)

def amp_autocast() -> amp.autocast:
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.amp.autocast("cuda", dtype=dtype)

def load_csv_column(csv_name: str, column: str, log_dir: Union[str, Path]) -> pd.Series:
    df = pd.read_csv(Path(log_dir) / csv_name)
    return df[column]

def closest_index(series: pd.Series, target: int) -> int:
    return int(np.abs(series.index.to_numpy() - target).argmin())

# ─────────────────── Grammar Helpers & Context ─────────────────────
def update_counters(
    counters: Dict[int, torch.Tensor],
    constraints: List[Constraint],
    action: torch.Tensor,
    ctx_idx: torch.Tensor,
) -> None:
    """
    In‑place reset & increment via pure arithmetic (no boolean‑indexing assignment).
    """
    for ci, con in enumerate(constraints):
        # mask for resets
        reset_mask = (action == con.trigger_token) & (ctx_idx == context2idx[con.trigger_ctx])
        reset_int  = reset_mask.int()

        # mask for requirement sightings
        if con.required_token is None:
            saw_req = (
                (ctx_idx == context2idx[con.required_ctx])
                & (action >= 27)
                & (action <= 33)
            )
        else:
            saw_req = action == con.required_token
        saw_int = saw_req.int()

        # apply: zero-out where reset_mask, then add 1 where saw_req
        # counters[ci] is an int tensor of shape [B]
        c = counters[ci]
        c.mul_(1 - reset_int)   # zeros out entries where reset_mask==1
        c.add_(   saw_int   )   # increments entries where saw_req==1

        # store back (in case we shadowed)
        counters[ci] = c

def update_need_particle(
    need_particle: torch.Tensor,
    ctx_idx: torch.Tensor,
    action: torch.Tensor,
) -> None:
    """
    Vectorised update of `need_particle` flag.

    Rules:
      • If action == 34 (CREATE_MULTIPLET) ⇒ need_particle ← True
      • If ctx == "particle" AND 27 ≤ action ≤ 33 ⇒ need_particle ← False
    """
    need_particle |= (action == 34)
    mask_qn = (ctx_idx == context2idx["particle"]) & (action >= 27) & (action <= 33)
    need_particle[mask_qn] = False

def get_next_mask(
    last_tok: torch.Tensor,
    ctx_idx: torch.Tensor,
    need_particle: torch.Tensor,
    counters: Dict[int, torch.Tensor],
) -> torch.Tensor:
    B = last_tok.size(0)
    mask = torch.zeros(B, VOCAB_SIZE, device=_DEVICE)

    # (1) grammar-based transitions
    for ctx_i, ctx in enumerate(contexts):
        rows = (ctx_idx == ctx_i).nonzero(as_tuple=True)[0]
        for r in rows:
            tok = int(last_tok[r].item())
            allowed = token_grammar.get(tok, {}).get(ctx, [])
            mask[r, allowed] = 1.0

    # (2) block 34 when need_particle is True
    if need_particle.any():
        mask[need_particle, 34] = 0

    # (3) constraint-driven blocks
    for ci, con in enumerate(constraints):
        trg_ctx_i = context2idx[con.trigger_ctx]
        rows = (need_particle & (ctx_idx == trg_ctx_i)).nonzero(as_tuple=True)[0] if con.trigger_token == 34 \
             else (ctx_idx == trg_ctx_i).nonzero(as_tuple=True)[0]
        too_soon = rows[counters[ci][rows] < con.min_count]
        if too_soon.numel():
            mask[too_soon, con.trigger_token] = 0

    return mask

def next_context(current_ctx: int, token_id: int) -> int:
    """Finite-state update for context."""
    if token_id == 0:   return context2idx["particle"]
    if token_id == 34:  return context2idx["multiplet"]
    return current_ctx

def build_batch(batch_size: int, seq_len: int):
    """
    Vectorised sampling + one‑shot context update + pre‑created buffers.
    """
    ctx_idx       = torch.randint(len(contexts), (batch_size,), device=_DEVICE)
    need_particle = torch.zeros(batch_size, dtype=torch.bool, device=_DEVICE)
    counters      = {ci: torch.zeros(batch_size, dtype=torch.int, device=_DEVICE)
                     for ci in range(len(constraints))}

    teacher    = torch.full((batch_size, seq_len), BOS_TOKEN, device=_DEVICE, dtype=torch.long)
    batch_mask = torch.zeros(batch_size, seq_len, VOCAB_SIZE, device=_DEVICE)
    ctx_seq    = torch.zeros(batch_size, seq_len, dtype=torch.long, device=_DEVICE)
    last_tok   = torch.full((batch_size,), BOS_TOKEN, device=_DEVICE, dtype=torch.long)

    for t in range(seq_len):
        ctx_seq[:, t] = ctx_idx

        # 1) grammar mask
        step_mask = get_next_mask(last_tok, ctx_idx, need_particle, counters)
        batch_mask[:, t] = step_mask

        # 2) vectorised sampling
        probs  = step_mask / step_mask.sum(-1, keepdim=True)
        action = torch.multinomial(probs, 1).squeeze(1)
        teacher[:, t] = action

        # 3) flag & counter updates
        update_need_particle(need_particle, ctx_idx, action)
        update_counters(      counters,     constraints, action, ctx_idx)

        # 4) one‑shot context update (use token 0=CREATE_PARTICLE, 34=CREATE_MULTIPLET)
        is_cp = action == 0
        is_cm = action == 34
        new_ctx = ctx_idx.clone()
        new_ctx[is_cp] = context2idx["particle"]
        new_ctx[is_cm] = context2idx["multiplet"]
        ctx_idx = new_ctx

        last_tok = action

    return ctx_seq, batch_mask, teacher

# ───────────────────────── LR Scheduler ────────────────────────────
def build_warmup_cosine_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    if warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be < total_steps")
    def lr_lambda(step: int) -> float:
        if step < 0: return 0.0
        if step < warmup_steps: return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * prog)))
    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)

def closest_lr(series: pd.Series, step: int) -> float:
    idx = closest_index(series, step)
    return series.iloc[idx]

# ─────────────────────────── Models ──────────────────────────
class TransformerPolicy(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_ctx: int,
        *,
        d_model: int = 64,
        nhead: int = 2,
        num_layers: int = 2,
        max_seq_len: int = 64
    ) -> None:
        super().__init__()
        self.d_model     = d_model
        self.max_seq_len = max_seq_len

        # token & context embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.ctx_emb = nn.Embedding(num_ctx, d_model)

        # positional embedding + pre‑created index buffer
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.register_buffer("pos_idx", torch.arange(max_seq_len, dtype=torch.long))

        # encoder with gradient checkpointing
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

        # actor / value heads
        self.actor_head = nn.Linear(d_model, vocab_size)
        self.value_head = nn.Linear(d_model, 1)

        # turn on memory‑efficient attention if on CUDA
        if _DEVICE.type == "cuda":
            torch.backends.cuda.enable_mem_efficient_sdp(True)
    @staticmethod
    def script(model: "TransformerPolicy") -> torch.jit.ScriptModule:
        return torch.jit.script(model.eval()).train()
    @staticmethod
    def compile(model: "TransformerPolicy", **kwargs: Any) -> "TransformerPolicy":
        if hasattr(torch, "compile"): return torch.compile(model, dynamic=True, **kwargs)  # type: ignore
        return model
    def forward(
        self,
        seq: torch.Tensor,
        ctx_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = seq.shape
        # token embedding
        tok_e = self.tok_emb(seq)                   # (B, T, d)

        # positional embedding via pre‑registered buffer
        pos_ids = self.pos_idx[:T]                  # (T,)
        pos_e   = self.pos_emb(pos_ids)             # (T, d)
        pos_e   = pos_e.unsqueeze(0).expand(B, T, -1)

        # context embedding
        ctx_e = self.ctx_emb(ctx_seq)               # (B, T, d)

        # encode & heads
        h = self.encoder(tok_e + pos_e + ctx_e)     # (B, T, d)
        return self.actor_head(h), self.value_head(h).squeeze(-1)


class RNNPolicy(nn.Module):
    def __init__(self, vocab_size: int, num_ctx: int, d_model: int = 64, hidden: int = 128, num_layers: int = 1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.ctx_emb = nn.Embedding(num_ctx, d_model)
        self.rnn = nn.GRU(d_model, hidden, num_layers, batch_first=True)
        self.actor = nn.Linear(hidden, vocab_size)
        self.critic = nn.Linear(hidden, 1)
        if hasattr(torch, "compile"): self.forward = torch.compile(self.forward, dynamic=True)  # type: ignore
        else: torch.jit.script(self)
    def forward(self, seq: torch.Tensor, ctx_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = seq.shape
        assert ctx_seq.shape == (B, T), "ctx_seq must match seq dimensions"
        x = self.tok_emb(seq) + self.ctx_emb(ctx_seq)
        h, _ = self.rnn(x)
        return self.actor(h), self.critic(h).squeeze(-1)

# ───────────────────────── Logging Helper ─────────────────────────
def log_and_record(
    writer: SummaryWriter,
    csv_path: Path,
    step: int,
    seq_len: int,
    ep_rew: float,
    fps: float,
    elapsed: float,
    total_ts: int,
    entropy_loss: float,
    logit_entropy: float,
    explained_var: float,
    updates: int,
    td_err_mean: float,
    td_err_var: float,
    return_diff_mean: float,
    policy_loss: float,
    policy_entropy: float,
    l1_norm: float,
    l2_norm: float,
    learning_rate: float,
) -> int:
    """Pretty console block + CSV + TensorBoard."""
    block = [
        "------------------------------------------------",
        f"| rollout/ep_len_mean      | {seq_len:10d}",
        f"| rollout/ep_rew_mean      | {ep_rew:10.4f}",
        f"| time/fps                 | {fps:10.2f}",
        f"| time/time_elapsed        | {elapsed:10.2f}",
        f"| time/total_timesteps     | {total_ts:10d}",
        f"| train/entropy_loss       | {entropy_loss:10.4f}",
        f"| train/logit_entropy      | {logit_entropy:10.4f}",
        f"| train/explained_variance | {explained_var:10.4f}",
        f"| train/n_updates          | {updates:10d}",
        f"| train/td_error_mean      | {td_err_mean:10.4f}",
        f"| train/td_error_variance  | {td_err_var:10.4f}",
        f"| train/return_diff_mean   | {return_diff_mean:10.4f}",
        f"| train/policy_loss        | {policy_loss:10.4f}",
        f"| train/policy_entropy     | {policy_entropy:10.4f}",
        f"| train/l1_norm            | {l1_norm:10.4f}",
        f"| train/l2_norm            | {l2_norm:10.4f}",
        f"| train/learning_rate      | {learning_rate:10.6f}",
        "------------------------------------------------",
    ]
    print("\n".join(block))

    with csv_path.open("a", newline="") as f:
        csv.writer(f).writerow([
            time.time(), seq_len, ep_rew, fps, elapsed, total_ts, entropy_loss, logit_entropy, explained_var, updates, td_err_mean, td_err_var, return_diff_mean, policy_loss, policy_entropy, l1_norm, l2_norm, learning_rate,
        ])

    tb_metrics = [
        ("rollout/ep_len_mean", seq_len),
        ("rollout/ep_rew_mean", ep_rew),
        ("time/fps", fps),
        ("time/time_elapsed", elapsed),
        ("time/total_timesteps", total_ts),
        ("train/entropy_loss", entropy_loss),
        ("train/logit_entropy", logit_entropy),
        ("train/explained_variance", explained_var),
        ("train/n_updates", updates),
        ("train/td_error_mean", td_err_mean),
        ("train/td_error_variance", td_err_var),
        ("train/return_diff_mean", return_diff_mean),
        ("train/policy_loss", policy_loss),
        ("train/policy_entropy", policy_entropy),
        ("train/l1_norm", l1_norm),
        ("train/l2_norm", l2_norm),
        ("train/learning_rate", learning_rate),
    ]
    for tag, val in tb_metrics:
        writer.add_scalar(tag, val, step)

    return len(block)

# ──────────────────────── Supervised Training ───────────────────────
def train_supervised(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    seq_len: int,
    *,
    epochs: int = RL_EPOCHS,
    batch_size: int = RL_BATCHES,
    warmup_frac: float = 0.10
) -> str:
    run_id    = datetime.now().strftime("%Y%m%d_%H%M%S")
    vars_dir  = LOG_DIR / "log_vars" / f"supervised_{run_id}"
    tb_dir    = LOG_DIR / "tensorboard" / f"supervised_{run_id}"
    ckpt_dir  = LOG_DIR / "checkpoints"  / f"supervised_{run_id}"
    ensure_dirs([vars_dir, tb_dir, ckpt_dir])

    # set up async CSV logger
    csv_path = vars_dir / CSV_NAME
    header   = ["epoch", "time_elapsed", "loss", "lr"]
    csv_logger = CSVLoggerThread(csv_path, header)
    csv_logger.start()

    # tensorboard
    writer = SummaryWriter(tb_dir)

    # scheduler: step exactly once per epoch
    total_updates = epochs
    warmup_steps  = int(total_updates * warmup_frac)
    scheduler     = build_warmup_cosine_scheduler(optimizer, warmup_steps, total_updates)

    start = time.time()
    model.train()

    for ep in range(epochs):
        ctx_seq, mask, teacher = build_batch(batch_size, seq_len)
        inp = torch.cat([
            torch.full((batch_size, 1), BOS_TOKEN, device=_DEVICE),
            teacher[:, :-1]
        ], dim=1)

        with amp_autocast():
            logits, _ = model(inp, ctx_seq)
            allowed   = (F.softmax(logits, dim=-1) * mask).sum(-1).clamp_min(EPS)
            loss      = -torch.log(allowed).mean()

        # step optimizer & scheduler
        _SCALER.scale(loss).backward()
        _SCALER.step(optimizer)
        _SCALER.update()
        optimizer.zero_grad(set_to_none=True)

        # record metrics
        elapsed    = time.time() - start
        current_lr = scheduler.get_last_lr()[0]
        csv_logger.queue.put([ep, elapsed, loss.item(), current_lr])
        writer.add_scalar("train/loss", loss.item(), ep)
        writer.add_scalar("train/learning_rate", current_lr, ep)

        # scheduler stepped once per epoch
        scheduler.step()

        if ep % 50 == 0:
            print(f"[Supervised] Epoch {ep:>3} | loss={loss.item():.4f} | lr={current_lr:.3e}")
        if ep % CKPT_EVERY_EPOCHS == 0:
            save_state_dict(model, ckpt_dir / f"supervised_{(ep+1)*batch_size*seq_len}.pt")

    # finalize
    csv_logger.queue.put(None)
    csv_logger.join()
    writer.close()

    save_state_dict(model, ckpt_dir / "supervised_final.pt")
    return run_id

# ───────────────────────── REINFORCE Training ─────────────────────
def train_reinforce(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    seq_len: int,
    *,
    epochs: int = RL_EPOCHS,
    gamma: float = RL_GAMMA,
    batch_size: int = RL_BATCHES,
    warmup_frac: float = 0.10,
    ckpt_every: int = CKPT_EVERY_EPOCHS,
) -> str:
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    vars_dir = LOG_DIR / "log_vars" / f"rl_{run_id}"
    tb_dir   = LOG_DIR / "tensorboard" / f"rl_{run_id}"
    ckpt_dir = LOG_DIR / "checkpoints"  / f"rl_{run_id}"
    ensure_dirs([vars_dir, tb_dir, ckpt_dir])

    # async CSV loggers
    csv_path    = vars_dir / CSV_NAME
    state_csv   = vars_dir / STATE_CSV_NAME
    main_logger = CSVLoggerThread(csv_path, [
        "timestamp","rollout/ep_len_mean","rollout/ep_rew_mean",
        "time/fps","time/time_elapsed","time/total_timesteps",
        "train/entropy_loss","train/logit_entropy","train/explained_variance",
        "train/n_updates","train/td_error_mean","train/td_error_variance",
        "train/return_diff_mean","train/policy_loss","train/policy_entropy",
        "train/l1_norm","train/l2_norm","train/learning_rate"
    ])
    state_logger = CSVLoggerThread(state_csv, ["state","reward","action","logits","entropy"])
    main_logger.start()
    state_logger.start()

    writer       = SummaryWriter(tb_dir)
    warmup_steps = int(epochs * warmup_frac)
    scheduler    = build_warmup_cosine_scheduler(opt, warmup_steps, epochs)

    total_ts, updates, start_time, last_lines = 0, 0, time.time(), 0

    for ep in range(1, epochs + 1):
        if last_lines:
            print(f"\033[{last_lines}A\033[2K", end="")

        # ── initialize rollout ───────────────────────────────
        ctx_idx       = torch.randint(len(contexts), (batch_size,), device=_DEVICE)
        need_particle = torch.zeros(batch_size, dtype=torch.bool, device=_DEVICE)
        counters      = {
            ci: torch.zeros(batch_size, dtype=torch.int, device=_DEVICE)
            for ci in range(len(constraints))
        }

        seq        = torch.full((batch_size, 1), BOS_TOKEN, device=_DEVICE, dtype=torch.long)
        ctx_seq    = ctx_idx.unsqueeze(1).clone()
        last_tok   = seq[:, -1].clone()

        rewards     = torch.zeros(batch_size, device=_DEVICE)
        logp_sums   = torch.zeros(batch_size, device=_DEVICE)
        entropy_lst = []
        value_lst   = []
        reward_steps: List[torch.Tensor] = []
        prefix_valid = torch.ones(batch_size, dtype=torch.bool, device=_DEVICE)

        # ── generate trajectory ───────────────────────────────
        for t in range(seq_len):
            step_mask = get_next_mask(last_tok, ctx_idx, need_particle, counters)
            with amp_autocast():
                logits_seq, values_seq = model(seq, ctx_seq)
            logits, values = logits_seq[:, -1], values_seq[:, -1]

            dist   = Categorical(logits=logits)
            action = dist.sample()
            logp   = dist.log_prob(action)
            ent    = dist.entropy()

            allowed_now = step_mask.gather(1, action.unsqueeze(1)).squeeze(1).bool()
            valid       = allowed_now & prefix_valid
            prefix_valid &= allowed_now

            # accumulate reward & bookkeeping
            rewards      += 0.2 * valid.float()
            logp_sums    += logp
            entropy_lst.append(ent)
            value_lst.append(values)
            reward_steps.append(valid.float())

            # async-state log
            state_logger.queue.put([
                seq.cpu().tolist(),
                valid.cpu().tolist(),
                action.cpu().tolist(),
                logits.detach().cpu().tolist(),
                ent.cpu().tolist(),
            ])

            # append token & ctx
            seq     = torch.cat([seq, action.unsqueeze(1)], dim=1)
            ctx_seq = torch.cat([ctx_seq, ctx_idx.unsqueeze(1)], dim=1)

            update_need_particle(need_particle, ctx_idx, action)
            update_counters(counters, constraints, action, ctx_idx)

            # one‑shot context update
            is_cp = action == 0
            is_cm = action == 34
            new_ctx = ctx_idx.clone()
            new_ctx[is_cp] = context2idx["particle"]
            new_ctx[is_cm] = context2idx["multiplet"]
            ctx_idx = new_ctx

            last_tok = action.clone()
            total_ts += batch_size

        # final reward if still valid
        rewards += prefix_valid.float()

        # ── compute returns & advantage ───────────────────────
        R = torch.zeros(batch_size, device=_DEVICE)
        returns_steps: List[torch.Tensor] = []
        for r in reversed(reward_steps):
            R = r + gamma * R
            returns_steps.insert(0, R)
        returns = torch.stack(returns_steps)    # shape (T, B)
        values_t = torch.stack(value_lst)       # shape (T, B)
        adv      = returns - values_t

        # ── loss + backward ───────────────────────────────────
        with amp_autocast():
            policy_loss = -(logp_sums * adv.mean(0).detach()).mean()
            value_loss  = F.mse_loss(values_t, returns)
            entropy_reg = -ENT_COEFF * torch.cat(entropy_lst).mean()
            loss        = policy_loss + VALUE_LOSS_COEF * value_loss + entropy_reg

        _SCALER.scale(loss).backward()
        _SCALER.step(opt)
        _SCALER.update()
        opt.zero_grad(set_to_none=True)
        scheduler.step()
        updates += 1

        # ── log metrics ───────────────────────────────────────
        td         = adv.detach().cpu().numpy().mean()
        var_err    = (rewards - values_t[0]).var(unbiased=False)
        var_return = rewards.var(unbiased=False)
        denom      = var_return if var_return.item() > 1e-3 else torch.tensor(1e-3, device=_DEVICE)
        explained  = (1.0 - (var_err / denom)).item()

        elapsed    = time.time() - start_time
        fps        = total_ts / max(elapsed, 1e-8)
        logit_ent  = torch.cat(entropy_lst).mean().item()
        current_lr = scheduler.get_last_lr()[0]

        last_lines = log_and_record(
            writer, csv_path, updates, seq_len,
            rewards.mean().item(), fps, elapsed, total_ts,
            entropy_reg.item(), logit_ent, explained, updates,
            td, ((rewards - values_t[0]).var().item()),  # td_err, td_var
            0.0,                                            # return_diff_mean
            policy_loss.item(), logit_ent,
            sum(p.abs().sum().item() for p in model.parameters()),
            math.sqrt(sum((p**2).sum().item() for p in model.parameters())),
            current_lr,
        )

        if ep % ckpt_every == 0:
            save_state_dict(model, ckpt_dir / f"step_{total_ts}.pt")

    # tear down
    main_logger.queue.put(None);  main_logger.join()
    state_logger.queue.put(None); state_logger.join()
    writer.close()
    save_state_dict(model, ckpt_dir / "rl_final.pt")
    return run_id

# ───────────────────────── Sampling & Plotting ─────────────────────
@torch.no_grad()
def generate_samples(model: nn.Module, seq_len: int, n: int = 4) -> List[List[int]]:
    model.eval()
    samples: List[List[int]] = []

    for i in range(n):
        ctx          = torch.randint(len(contexts), (1,), device=_DEVICE).item()
        need_particle = torch.zeros(1, dtype=torch.bool, device=_DEVICE)
        seq          = [BOS_TOKEN]
        ctx_seq      = [ctx]
        last_tok     = BOS_TOKEN

        print(f"- Sample {i + 1}  Context='{contexts[ctx]}' → [{last_tok}", end="")

        for _ in range(seq_len):
            inp        = torch.tensor([seq], device=_DEVICE)
            ctx_tensor = torch.tensor([ctx_seq], device=_DEVICE)
            with amp_autocast():
                logits, _ = model(inp, ctx_tensor)
            nxt = int(logits[0, -1].argmax().item())

            seq.append(nxt)
            ctx_seq.append(ctx)
            print(f", {nxt}", end="")

            update_need_particle(need_particle, torch.tensor([ctx], device=_DEVICE), torch.tensor([nxt], device=_DEVICE))
            ctx      = next_context(ctx, nxt)
            last_tok = nxt

        print("]")
        samples.append(seq)

    print("")
    model.train()
    return samples

# ───────────────────────────── Plotting ─────────────────────────────

def run_and_save_plot(plot_fn: Callable[..., None], save_dir: Union[str, Path], filename: str, **kwargs: Any) -> None:
    if plt is None:
        return
    save_dir = Path(save_dir)
    save_path = save_dir / GRAPHICS_DIR / filename
    ensure_dirs([save_path.parent])
    kwargs.setdefault("show", False)
    plot_fn(**kwargs)
    plt.gcf().savefig(save_path)
    plt.close()


def plot_reward_per_episode(csv_name: str = CSV_NAME, column: str = "rollout/ep_rew_mean", log_dir: Optional[Union[str, Path]] = None, *, show: bool = True) -> None:
    if plt is None or log_dir is None:
        return
    series = load_csv_column(csv_name, column, log_dir)
    eps = range(1, len(series) + 1)
    try:
        style.use(["seaborn-v0_8-dark", "seaborn-v0_8"])
    except Exception:
        pass
    plt.figure()
    plt.plot(eps, series)
    plt.xlabel("Episode")
    plt.ylabel(column)
    plt.title("Reward per Episode")
    plt.tight_layout()
    if show:
        plt.show()


def plot_reward_vs_time(csv_name: str = CSV_NAME, log_dir: Optional[Union[str, Path]] = None, *, time_column: str = "time/time_elapsed", value_column: str = "rollout/ep_rew_mean", title: str = "Reward vs Time", show: bool = True) -> None:
    if plt is None or log_dir is None:
        return
    times = load_csv_column(csv_name, time_column, log_dir)
    vals = load_csv_column(csv_name, value_column, log_dir)
    try:
        style.use(["seaborn-v0_8-dark", "seaborn-v0_8"])
    except Exception:
        pass
    plt.figure()
    plt.plot(times, vals)
    plt.xlabel("Time (s)")
    plt.ylabel(value_column)
    plt.title(title)
    plt.tight_layout()
    if show:
        plt.show()

# ───────────────────── Hessian spectrum utilities ───────────────────
def compute_hessian_over_checkpoints(
    model: nn.Module,
    checkpoint_dir: Union[str, Path],
    seq_len: int,
    *,
    batch_size: int = 64,
    iters: int = 5,
) -> Tuple[List[int], List[float]]:
    checkpoint_dir = Path(checkpoint_dir)
    step_rgx = re.compile(r"_(\d+)\.pt$")
    ckpts = [
        (int(m.group(1)), checkpoint_dir / f)
        for f in os.listdir(checkpoint_dir)
        if (m := step_rgx.search(f))
    ]
    if not ckpts:
        print(f"⚠️  No numeric checkpoints in {checkpoint_dir}")
        return [], []
    ckpts.sort(key=lambda x: x[0])

    # Hessian-vector product
    def _hvp(loss: torch.Tensor, params: List[nn.Parameter], vec: torch.Tensor) -> torch.Tensor:
        grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
        flat_grads = torch.cat([
            (g if g is not None else torch.zeros_like(p)).view(-1)
            for p, g in zip(params, grads)
        ])
        hv = torch.autograd.grad((flat_grads * vec).sum(), params, retain_graph=True, allow_unused=True)
        return torch.cat([
            (h if h is not None else torch.zeros_like(p)).view(-1)
            for p, h in zip(params, hv)
        ])

    # Power‐iteration to approximate max eigenvalue
    def _max_eig(loss: torch.Tensor, params: List[nn.Parameter]) -> float:
        vec = torch.randn(sum(p.numel() for p in params), device=_DEVICE)
        vec = vec / (vec.norm() + EPS)
        for _ in range(iters):
            hv = _hvp(loss, params, vec)
            vec = hv / (hv.norm() + EPS)
        return torch.dot(vec, _hvp(loss, params, vec)).abs().item()

    steps: List[int] = []
    eigs: List[float] = []
    ctx = sdpa_kernel(SDPBackend.MATH) if sdpa_kernel else nullcontext()
    with ctx:
        for step, fname in ckpts:
            load_state_dict_compat(model, fname)
            ctx_seq, mask, teacher = build_batch(batch_size, seq_len)
            inp = torch.cat([
                torch.full((batch_size, 1), BOS_TOKEN, device=_DEVICE),
                teacher[:, :-1]
            ], dim=1)

            logits, _ = model(inp, ctx_seq)
            loss = -torch.log((F.softmax(logits, dim=-1) * mask).sum(-1).clamp_min(EPS)).mean()

            eigs.append(_max_eig(loss, [p for p in model.parameters() if p.requires_grad]))
            steps.append(step)

    return steps, eigs



def plot_hessian_eigenvalues(
    steps: List[int],
    eigs: List[float],
    *,
    lrs: Optional[List[float]] = None,
    save_dir: Union[str, Path] = LOG_DIR,
    filename: str = "hessian_eigs.png",
    show: bool = True,
) -> None:
    if plt is None:
        return
    if not steps:
        print("⚠️  No Hessian data – nothing to plot.")
        return
    if lrs is not None and len(lrs) != len(steps):
        raise ValueError("`lrs` must match length of `steps`.")

    order = np.argsort(steps)
    steps_s = np.array(steps)[order]
    eigs_s = np.array(eigs)[order]
    lrs_s = np.array(lrs)[order] if lrs is not None else None

    try:
        style.use(["seaborn-v0_8-dark", "seaborn-v0_8"])
    except Exception:
        pass

    plt.figure()
    plt.plot(steps_s, eigs_s, marker="o" if len(steps_s) == 1 else None, label="max Hessian eig")
    if lrs_s is not None:
        plt.plot(steps_s, 2.0 / lrs_s, linestyle="--", marker="x" if len(steps_s) == 1 else None, label="2 / learning-rate")
    plt.yscale("log")
    plt.xlabel("Training Steps")
    plt.ylabel("Value (log)")
    plt.title("Hessian Spectrum vs 2/LR")
    plt.legend()
    plt.tight_layout()

    save_path = Path(save_dir) / GRAPHICS_DIR / filename
    ensure_dirs([save_path.parent])
    plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()
    print(f"✅  Hessian plot written to {save_path}")

def _math_sdp_ctx(): return sdpa_kernel(SDPBackend.MATH) if sdpa_kernel else nullcontext()

@torch.no_grad()
def lanczos_eigs(hvp_np: Callable[[np.ndarray], np.ndarray], dim: int, k: int, maxiter: int = 100, tol: float = 1e-5) -> np.ndarray:
    H_linop = LinearOperator((dim, dim), matvec=hvp_np, dtype=np.float32)
    eigvals, _ = eigsh(A=H_linop, k=k, which="LA", maxiter=maxiter, tol=tol)
    return eigvals

def hvp_factory(loss: torch.Tensor, params: List[nn.Parameter]) -> Tuple[Callable[[np.ndarray], np.ndarray], int]:
    with _math_sdp_ctx(): grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    flat_grads = torch.cat([(g if g is not None else torch.zeros_like(p)).view(-1) for p, g in zip(params, grads)])
    dim, device = flat_grads.numel(), flat_grads.device
    def hvp(vec_np: np.ndarray) -> np.ndarray:
        vec = torch.from_numpy(vec_np).to(device).view(-1)
        with _math_sdp_ctx(): grad2 = torch.autograd.grad((flat_grads * vec).sum(), params, retain_graph=True, allow_unused=True)
        flat_hv = torch.cat([(g if g is not None else torch.zeros_like(p)).view(-1) for p, g in zip(params, grad2)])
        return flat_hv.detach().cpu().numpy()
    return hvp, dim

def plot_hessian_density_slq(
    loss: torch.Tensor,
    params: List[nn.Parameter],
    save_dir: Union[str, Path],
    step: int,
    *,
    num_probes: int = 10,
    lanczos_order: int = 50,
    bins: int = 50,
    show: bool = False,
) -> None:
    hvp, dim = hvp_factory(loss, params)
    H = LinearOperator((dim, dim), matvec=hvp, dtype=np.float32)
    k = max(1, min(lanczos_order // 2, dim - 1))
    spectra: List[np.ndarray] = []
    for _ in trange(num_probes, desc="SLQ probes", unit="probe"):
        eig_hi, _ = eigsh(H, k=k, which="LA")
        eig_lo, _ = eigsh(H, k=k, which="SA")
        spectra += [eig_hi, eig_lo]
    eigs = np.concatenate(spectra)
    if plt is None:
        print("⚠️ matplotlib not available – skipping plot.")
        return
    out_dir = Path(save_dir) / GRAPHICS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.hist(eigs, bins=bins)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Eigenvalue"); plt.ylabel("Frequency")
    plt.title(f"Hessian SLQ Spectrum @ step {step}")
    plt.tight_layout()
    out = out_dir / f"hessian_slq_spectrum_step{step}.png"
    plt.savefig(out)
    if show: plt.show()
    plt.close()
    print(f"✅  Saved SLQ spectrum →  {out}")

if __name__ == "__main__":
    SEQ_LENS = [32]
    supervised_runs, pretrained_runs, scratch_runs = {}, {}, {}

    for seq_len in SEQ_LENS:
        print(f"\n================  SEQ_LEN = {seq_len}  ================\n")

        # 1) SUPERVISED PRE-TRAINING
        sup_model = make_policy(seq_len, compile_ok=True)
        sup_opt   = torch.optim.AdamW(sup_model.parameters(), lr=LR)
        sup_run_id = train_supervised(sup_model, sup_opt, seq_len)
        sup_ckpt_dir = LOG_DIR / "checkpoints" / f"supervised_{sup_run_id}"
        sup_run_dir  = LOG_DIR / "log_vars"    / f"supervised_{sup_run_id}"
        supervised_runs[seq_len] = str(sup_run_dir)

        generate_samples(sup_model, seq_len)

        # light Hessian on supervised checkpoints
        sup_steps, sup_eigs = compute_hessian_over_checkpoints(
            make_policy(seq_len, compile_ok=False),
            sup_ckpt_dir,
            seq_len,
        )
        lr_series_sup   = load_csv_column(CSV_NAME, "lr", sup_run_dir)
        ts_sup          = np.arange(1, len(lr_series_sup) + 1) * RL_BATCHES * seq_len
        lr_by_step_sup  = pd.Series(lr_series_sup.values, index=ts_sup).sort_index()
        lr_for_ckpts_sup = [closest_lr(lr_by_step_sup, s) for s in sup_steps]
        plot_hessian_eigenvalues(
            sup_steps,
            sup_eigs,
            lrs=lr_for_ckpts_sup,
            filename=f"hessian_eigs_supervised{seq_len}.png",
            show=False,
        )

        # 2) RL – initialized from supervised checkpoint
        rl_pre_model = make_policy(seq_len, compile_ok=False)
        load_state_dict_compat(rl_pre_model, sup_ckpt_dir / "supervised_final.pt")
        if hasattr(torch, "compile"):
            rl_pre_model = TransformerPolicy.compile(rl_pre_model)
        rl_pre_opt = torch.optim.AdamW(rl_pre_model.parameters(), lr=LR)
        rl_pre_run_id = train_reinforce(rl_pre_model, rl_pre_opt, seq_len)
        rl_pre_run_dir = LOG_DIR / "log_vars" / f"rl_{rl_pre_run_id}"
        pretrained_runs[seq_len] = str(rl_pre_run_dir)

        generate_samples(rl_pre_model, seq_len)

        # # light Hessian on RL‐pretrained checkpoints
        # rl_pre_ckpt_dir = LOG_DIR / "checkpoints" / f"rl_{rl_pre_run_id}"
        # steps, eigs = compute_hessian_over_checkpoints(
        #     make_policy(seq_len, compile_ok=False),
        #     rl_pre_ckpt_dir,
        #     seq_len,
        # )
        # ts_series = load_csv_column(CSV_NAME, "time/total_timesteps", rl_pre_run_dir)
        # lr_series = load_csv_column(CSV_NAME, "train/learning_rate",  rl_pre_run_dir)
        # lr_by_ts  = pd.Series(lr_series.values, index=ts_series.values).sort_index()
        # lr_for_ckpts = [closest_lr(lr_by_ts, s) for s in steps]
        # plot_hessian_eigenvalues(
        #     steps,
        #     eigs,
        #     lrs=lr_for_ckpts,
        #     filename=f"hessian_eigs_pretrained{seq_len}.png",
        #     show=False,
        # )

        # # ───── New: FULL Hessian spectra ─────
        # print("\n>>> SLQ‐estimating full Hessian spectrum…")
        # spectrum_model = make_policy(seq_len, compile_ok=False)

        # # reuse your checkpoint discovery:
        # step_rgx   = re.compile(r"_(\d+)\.pt$")
        # ckpt_files = sorted(
        #     [(int(m.group(1)), fname)
        #     for fname in os.listdir(rl_pre_ckpt_dir)
        #     if (m := step_rgx.search(fname))],
        #     key=lambda x: x[0]
        # )

        # for step, fname in ckpt_files:
        #     load_state_dict_compat(spectrum_model, rl_pre_ckpt_dir / fname)

        #     ctx_idx, mask, teacher = build_batch(64, seq_len)
        #     inp = torch.cat(
        #         [torch.full((64, 1), BOS_TOKEN, device=_DEVICE), teacher[:, :-1]], 1
        #     )

        #     # >>> MATH SDP FORWARD <<<  ← crucial!
        #     with (_math_sdp_ctx()):
        #         logits, _ = spectrum_model(inp, ctx_idx)
        #         probs = F.softmax(logits, dim=-1)
        #         loss  = -torch.log((probs * mask).sum(-1).clamp_min(EPS)).mean()

        #     # SLQ spectrum
        #     plot_hessian_density_slq(
        #         loss,
        #         [p for p in spectrum_model.parameters() if p.requires_grad],
        #         save_dir=LOG_DIR,
        #         step=step,
        #         num_probes=10,
        #         lanczos_order=50,
        #         bins=100,
        #         show=False,
        #     )

    #     # clean up VRAM
    #     del sup_model, rl_pre_model
    #     if _DEVICE.type == "cuda":
    #         torch.cuda.empty_cache()

    #     # 3) RL from scratch
    #     rl_scratch_model = make_policy(seq_len, compile_ok=True)
    #     rl_scratch_opt   = torch.optim.AdamW(rl_scratch_model.parameters(), lr=LR)
    #     rl_scratch_run_id = train_reinforce(rl_scratch_model, rl_scratch_opt, seq_len)
    #     rl_scratch_run_dir = LOG_DIR / "log_vars" / f"rl_{rl_scratch_run_id}"
    #     scratch_runs[seq_len] = str(rl_scratch_run_dir)

    #     generate_samples(rl_scratch_model, seq_len)
    #     run_and_save_plot(
    #         plot_reward_vs_time,
    #         rl_scratch_run_dir,
    #         f"scratch_reward_time_seq{seq_len}.png",
    #         csv_name=CSV_NAME,
    #         log_dir=rl_scratch_run_dir,
    #     )

    #     # light Hessian on scratch checkpoints
    #     rl_scratch_ckpt_dir = LOG_DIR / "checkpoints" / f"rl_{rl_scratch_run_id}"
    #     steps, eigs = compute_hessian_over_checkpoints(
    #         make_policy(seq_len, compile_ok=False),
    #         rl_scratch_ckpt_dir,
    #         seq_len,
    #     )
    #     plot_hessian_eigenvalues(
    #         steps,
    #         eigs,
    #         filename=f"hessian_eigs_scratch{seq_len}.png",
    #         show=False,
    #     )

    #     del rl_scratch_model
    #     if _DEVICE.type == "cuda":
    #         torch.cuda.empty_cache()

    # # Final summary plots (optional)
    # if plt:
    #     # Supervised loss vs time
    #     plt.figure()
    #     for s, run_dir in supervised_runs.items():
    #         t = load_csv_column(CSV_NAME, "time_elapsed", run_dir)
    #         l = load_csv_column(CSV_NAME, "loss",         run_dir)
    #         plt.plot(t - t.iloc[0], l, label=f"SEQ{s}")
    #     plt.xlabel("Elapsed Time (s)")
    #     plt.ylabel("Loss")
    #     plt.title("Supervised Loss vs Time")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(LOG_DIR / "super_loss_vs_time.png")

        # Pre-trained RL reward vs time
        plt.figure()
        for s, run_dir in pretrained_runs.items():
            t = load_csv_column(CSV_NAME, "time/time_elapsed",  run_dir)
            r = load_csv_column(CSV_NAME, "rollout/ep_rew_mean", run_dir)
            plt.plot(t - t.iloc[0], r, label=f"SEQ{s}")
        plt.xlabel("Elapsed Time (s)")
        plt.ylabel("Reward")
        plt.title("RL Reward vs Time (pre-trained)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(LOG_DIR / "[B]pre_reward_vs_time.png")

    #     plt.close("all")

