"""
reinforce_train.py (renamed to Together.py for your context)
– Grammar-constrained REINFORCE trainer with supervised mask-MSE regulariser.
  Keeps CSV / TensorBoard logging and periodic checkpoints via the `Logger`
  utility defined in *classes.py*.

Reward scheme
─────────────
• +1   each time the chosen token is **allowed** by the grammar.
• +S   when an object closes – S ≔ test_model(object-tokens).
• +1000·T when EOS closes the full model – T ≔ test_model(full-sequence).

Loss =  ⟨−log π(a) · (G − b)⟩   +   λ · MSE(logits⊘mask, 0)
        policy gradient          supervised mask-MSE
"""

# ==== Start: Add this block to suppress C++ backend warnings ====
from __future__ import annotations
import sys
import logging
import os
import threading
import warnings

# Filter out the nested tensor warning
warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True")

# --- This block should be at the absolute top of your script ---

# 1. Keep a reference to the original stderr stream.
original_stderr = sys.stderr

# 2. Create a custom filter to identify and block the specific warning messages.
class CppWarningFilter(logging.Filter):
    def filter(self, record):
        suppress_phrases = [
            "Unable to register cuFFT factory",
            "Unable to register cuDNN factory",
            "Unable to register cuBLAS factory",
            "computation placer already registered",
            "This TensorFlow binary is optimized"
        ]
        # Return False to suppress the message, True to let it through.
        return not any(phrase in record.getMessage() for phrase in suppress_phrases)

# 3. Set up a dedicated logger.
#    This logger will have one handler, which prints to the *original* stderr.
#    We will attach our custom filter to this handler.
cpp_logger = logging.getLogger("cpp_warnings_filter")
cpp_logger.propagate = False # Prevent messages from being passed to the root logger.

# Create a handler that writes to the original stderr we saved earlier.
handler = logging.StreamHandler(original_stderr)
handler.addFilter(CppWarningFilter()) # Attach our filter to this handler.
cpp_logger.addHandler(handler)

# 4. Create a thread-safe and recursion-proof class to redirect stderr.
class StderrRedirector:
    def __init__(self, logger):
        self._logger = logger
        # Use thread-local storage for the re-entrancy guard to make it thread-safe.
        self._local = threading.local()

    def write(self, message):
        # Prevent recursive writes. If we are already writing, do nothing.
        if getattr(self._local, 'is_writing', False):
            return
        try:
            self._local.is_writing = True
            # Pass the message to our filtered logger. The logger's handler will
            # decide whether to print it based on the filter.
            if message.rstrip():
                self._logger.warning(message.rstrip())
        finally:
            # Ensure the guard is always released.
            self._local.is_writing = False

    def flush(self):
        # The handler flushes automatically, but we can be explicit.
        for h in self._logger.handlers:
            h.flush()

# 5. Perform the redirection of sys.stderr.
sys.stderr = StderrRedirector(cpp_logger)

# 6. Set the TensorFlow environment variable as a complementary first line of defense.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 3 = FATAL errors only
# ===================================================================
# ==== End: C++ Warning Suppression Block ====
# ===================================================================


# ─────────────────────────────────────────────────────────────────────────────
# 1. Imports
# ─────────────────────────────────────────────────────────────────────────────
import csv, math, os, random, time, queue, warnings, contextlib, logging, threading
import sys # Added for sys.path manipulation in worker_init_fn
from pathlib import Path
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union, Iterable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.distributions import Categorical
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import trange
try: import matplotlib.pyplot as plt
except ModuleNotFoundError: plt = None
from .classes import Logger, TorchGrammar, TorchGrammarBatchParser, ModelTester # Corrected to relative
import multiprocessing
try: multiprocessing.set_start_method('spawn', force=True)
except RuntimeError: pass
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# 2. CUDA backend / global setup
# ─────────────────────────────────────────────────────────────────────────────
SEED, EPS = 42, 1e-9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED); torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
_AMP_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
SCALER = amp.GradScaler(enabled=(DEVICE.type == "cuda"))
CAN_COMPILE = hasattr(torch, "compile")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.INFO,
                  format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
main_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Domain-specific tokens & grammar (Model needs >1 Interaction)
# ─────────────────────────────────────────────────────────────────────────────
TOKENS: List[str] = (
    ["BOS", "EOS"]
    + ["ITRACT", "END_ITRACT"] + [f"ITRACT_ID_{i}" for i in range(3)]
    + ["FIELD", "END_FIELD"] + [f"FIELD_ID_{i}" for i in range(3)]
    + ["PARTICLE", "END_PARTICLE"] + [f"PARTICLE_ID_{i}" for i in range(3)]
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
ID2TOK = {i: t for t, i in TOKEN2ID.items()}
VOCAB_SIZE, BOS_TOKEN, EOS_TOKEN = len(TOKENS), TOKEN2ID["BOS"], TOKEN2ID["EOS"]

grammar = TorchGrammar()
tr = lambda s, e: (TOKEN2ID[s], TOKEN2ID[e])

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
        {"Particle": lambda c: c > 0},
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
        {"Interaction": lambda c: c >= 2},
        TOKEN2ID["EOS"],
    ],
)
ROOTS = ["Model"]

START_TOKS = {
    TOKEN2ID["PARTICLE"]: "PARTICLE",
    TOKEN2ID["FIELD"]: "FIELD",
    TOKEN2ID["ITRACT"]: "ITRACT",
}
END_TOKS = {
    TOKEN2ID["END_PARTICLE"]: "PARTICLE",
    TOKEN2ID["END_FIELD"]: "FIELD",
    TOKEN2ID["END_ITRACT"]: "ITRACT",
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. Transformer (actor-critic) - RESTORED
# ─────────────────────────────────────────────────────────────────────────────
class TransformerPolicy(nn.Module):
    def __init__(self, vocab: int, *, d_model: int = 256, nhead: int = 4,
                 layers: int = 3, max_T: int = 128):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_T, d_model)
        self.register_buffer("pos", torch.arange(max_T))
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, 0.1,
                                      batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.actor = nn.Linear(d_model, vocab)
        self.value = nn.Linear(d_model, 1) # <-- CRITIC HEAD RESTORED

    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: # <-- RETURN TYPE RESTORED
        x = self.tok_emb(seq) + self.pos_emb(self.pos[: seq.size(1)].to(seq.device))
        h = self.encoder(x)
        return self.actor(h), self.value(h).squeeze(-1) # <-- RETURN BOTH LOGITS AND VALUES

# ─────────────────────────────────────────────────────────────────────────────
# 5. Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class HParams:
    epochs: int = 200
    batch_size: int = 64
    seq_len: int = 64
    lr: float = 3e-4
    gamma: float = 1.0
    sup_coef: float = 0.01
    value_coef: float = 0.5
    ent_coef: float = 0.02
    grad_clip_norm: float = 1.0 
    ckpt_every: int = 500
    num_score_processes: Optional[int] = None

hp = HParams()
if hp.num_score_processes is None:
    hp = replace(hp, num_score_processes=os.cpu_count() or 4)

# ─────────────────────────────────────────────────────────────────────────────
# 6. CSV helper thread
# ─────────────────────────────────────────────────────────────────────────────
class CSVThread(threading.Thread):
    def __init__(self, path: Path, header: List[str]):
        super().__init__(daemon=True)
        self.path, self.header = path, header
        self.q: "queue.Queue[List[Any]|None]" = queue.Queue()

    def run(self) -> None:
        p = self.path; p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="") as f:
            w = csv.writer(f); w.writerow(self.header); f.flush()
            while True:
                row = self.q.get()
                if row is None:
                    self.q.task_done(); break
                w.writerow(row); f.flush(); self.q.task_done()

# ─────────────────────────────────────────────────────────────────────────────
# utility helpers
# ─────────────────────────────────────────────────────────────────────────────
def ensure_dirs(paths: Union[str, Path, Iterable[Union[str, Path]]]) -> None:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def safe_item(x: Union[float, torch.Tensor]) -> float:
    v = float(x.item() if isinstance(x, torch.Tensor) else x)
    return v if math.isfinite(v) else 0.0

def warmup_cosine(opt: torch.optim.Optimizer, warm: int, total: int) -> LambdaLR:
    def f(step: int) -> float:
        if total == 0: return 1.0
        if step < warm: return step / max(1, warm)
        p = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * p))
    return LambdaLR(opt, f, -1)

def _build_adamw(params, lr):
    try:
        return torch.optim.AdamW(params, lr=lr, fused=True)
    except Exception:
        warnings.warn("Fused AdamW unavailable → fallback.")
        return torch.optim.AdamW(params, lr=lr)

# Global ModelTester instance for each process in the pool
_global_model_tester_instance: Optional[ModelTester] = None

def _get_model_tester_instance():
    global _global_model_tester_instance
    if _global_model_tester_instance is None:
        _global_model_tester_instance = ModelTester()
    return _global_model_tester_instance

def worker_init_fn():
    """Initializer for multiprocessing Pool workers."""
    main_launch_dir_str = os.environ.get("MAIN_PROCESS_LAUNCH_DIR")
    if main_launch_dir_str:
        main_launch_dir = Path(main_launch_dir_str)
        project_root_for_imports = main_launch_dir.parent
        if str(project_root_for_imports) not in sys.path:
            sys.path.insert(0, str(project_root_for_imports))
            main_logger.debug(f"Worker sys.path updated with: {project_root_for_imports}")
    else:
        try:
            script_dir = Path(__file__).resolve().parent
            particle_graph_dir = script_dir.parent
            project_root_for_imports = particle_graph_dir.parent
            if str(project_root_for_imports) not in sys.path:
                sys.path.insert(0, str(project_root_for_imports))
                main_logger.debug(f"Worker sys.path updated (fallback) with: {project_root_for_imports}")
        except NameError:
            main_logger.warning("Worker_init_fn: Could not determine project root from __file__.")
    _get_model_tester_instance()

def _score_single_sequence(tokens: List[str]) -> float:
    tester = _get_model_tester_instance()
    try:
        score_val, _ = tester.test_model(tokens)
        return float(score_val) if score_val is not None else 0.0
    except Exception as e:
        main_logger.error(f"Error scoring sequence in worker: {e}. Sequence: {' '.join(tokens)}", exc_info=True)
        return 0.0

# ─────────────────────────────────────────────────────────────────────────────
# 7. REINFORCE training loop - MODIFIED TO ACTOR-CRITIC
# ─────────────────────────────────────────────────────────────────────────────
def train_reinforce(model: nn.Module, opt, hp: HParams) -> Path:
    run_id = datetime.now().strftime("rein_%Y%m%d_%H%M%S")
    script_root_dir = Path(__file__).resolve().parent
    logdir = script_root_dir / "particle" / "runs" / run_id
    ckptdir = logdir / "ckpt"
    ensure_dirs(logdir)
    ensure_dirs(ckptdir)

    logger = Logger(
        logdir,
        cli_interval=1,
        state_interval=1,
        weights_interval=hp.ckpt_every
    )
    logger.model = model
    for name in ["reward", "seq_len", "pl", "vl", "el", "gl", "tl", "l1_norm", "l2_norm"]:
        logger.add_metric(name, kind="stat")
    for name in ["n_updates", "n_rollouts", "n_samples", "learning_rate"]:
        logger.add_metric(name, kind="scalar")

    if CAN_COMPILE:
        model = torch.compile(model)
    sched = warmup_cosine(opt, int(hp.epochs*0.1), hp.epochs)

    os.environ["MAIN_PROCESS_LAUNCH_DIR"] = os.getcwd()

    score_pool = multiprocessing.Pool(
        processes=hp.num_score_processes,
        initializer=worker_init_fn
    )

    for ep in trange(1, hp.epochs+1, desc="epochs", leave=False, disable=True):
        B = hp.batch_size
        parser = TorchGrammarBatchParser(grammar, B, device=DEVICE)
        parser.set_batch(random.choices(ROOTS, k=B))
        seq = torch.full((B, hp.seq_len+1), BOS_TOKEN, device=DEVICE, dtype=torch.long)
        ep_len = torch.zeros(B, dtype=torch.long, device=DEVICE)
        alive = torch.ones(B, dtype=torch.bool, device=DEVICE)

        logps, rewards, sup_losses, val_list, ent_list = [], [], [], [], []
        history = [[BOS_TOKEN] for _ in range(B)]

        for t in range(hp.seq_len):
            ep_len[alive] += 1
            input_seq = seq[:, :t+1].clone()
            with amp.autocast(device_type="cuda", dtype=_AMP_DTYPE, enabled=(DEVICE.type == "cuda")):
                logits_t, values_t = model(input_seq)

            logits, values = logits_t[:, -1].float(), values_t[:, -1].float()
            mask = parser.next_token_mask(VOCAB_SIZE)
            masked = logits + mask

            no_valid = parser.next_lens.eq(0)
            eligible = alive & (~no_valid)
            if not eligible.any():
                break

            masked[~eligible] = 0.0

            probs = F.softmax(masked, -1)
            dist = Categorical(probs=probs)
            acts = dist.sample()
            logp = dist.log_prob(acts)
            entropy = dist.entropy()

            gram_loss_t = torch.zeros(B, device=DEVICE)
            if eligible.any():
                banned_indic = (mask[eligible] != 0.0).float()
                pen_per_row = F.relu(logits[eligible] * banned_indic).sum(-1)
                denom = banned_indic.sum(-1).clamp_min(1.0)
                gram_loss_t[eligible] = pen_per_row / denom
            sup_losses.append(gram_loss_t)

            logps.append(logp)
            val_list.append(values)
            ent_list.append(entropy)

            rew = torch.zeros(B, device=DEVICE)
            valid_row = parser.fast_is_valid(acts)
            rew += valid_row.float()

            sequences_to_score_info: List[Dict[str, Any]] = []
            for i in range(B):
                if not eligible[i]:
                    continue
                current_tok_id = acts[i].item()
                history[i].append(current_tok_id)
                if current_tok_id in END_TOKS:
                    obj_type_str = END_TOKS[current_tok_id]
                    start_tok_id = next(
                        (tok_id for tok_id, s_type in START_TOKS.items() if s_type == obj_type_str),
                        None
                    )
                    if start_tok_id is not None:
                        rev_idx = -1
                        for k in range(len(history[i]) - 2, -1, -1):
                            if history[i][k] == start_tok_id:
                                rev_idx = k
                                break
                        if rev_idx != -1:
                            sub_seq_tokens = [ID2TOK[x] for x in history[i][rev_idx:]]
                            sequences_to_score_info.append({
                                'batch_idx': i, 'bonus_type': 'object', 'tokens': sub_seq_tokens
                            })
                if current_tok_id == EOS_TOKEN:
                    full_seq_tokens = [ID2TOK[x] for x in history[i][1:]] # Exclude BOS
                    sequences_to_score_info.append({
                        'batch_idx': i, 'bonus_type': 'eos', 'tokens': full_seq_tokens
                    })

            if sequences_to_score_info:
                token_lists_for_pool = [item['tokens'] for item in sequences_to_score_info]
                try:
                    scored_results = score_pool.map(_score_single_sequence, token_lists_for_pool)
                    for score_idx, score_val in enumerate(scored_results):
                        info = sequences_to_score_info[score_idx]
                        idx_update = info['batch_idx']
                        if info['bonus_type'] == 'object': rew[idx_update] += score_val
                        elif info['bonus_type'] == 'eos': rew[idx_update] += 1000 * score_val
                except Exception as pool_exc:
                    main_logger.error(f"Error during score_pool.map: {pool_exc}", exc_info=True)

            rewards.append(rew)
            seq[:, t+1] = acts
            alive = alive & valid_row & (acts != EOS_TOKEN)
            logger.log({"state": f"ep{ep}_t{t}", "reward": rew.mean().item()})
            logger.state_step()

        if not rewards: continue

        stacked_logP = torch.stack(logps, dim=1)
        stacked_vals = torch.stack(val_list, dim=1)
        stacked_ents = torch.stack(ent_list, dim=1)
        stacked_rews = torch.stack(rewards, dim=1)
        stacked_grams = torch.stack(sup_losses, dim=1)

        disc_ret = torch.zeros_like(stacked_rews)
        R = torch.zeros(B, device=DEVICE)
        for rev_t in range(stacked_rews.size(1) - 1, -1, -1):
            R = stacked_rews[:, rev_t] + hp.gamma * R
            disc_ret[:, rev_t] = R

        # RESTORED ADVANTAGE CALCULATION
        advantages = disc_ret - stacked_vals

        ts_mat = torch.arange(stacked_rews.size(1), device=DEVICE).unsqueeze(0).expand(B, -1)
        valid_m = ts_mat < ep_len.unsqueeze(1)
        N_valid = valid_m.sum().clamp_min(1.0)

        # Advantage normalization for stability
        if N_valid > 1:
            adv_flat = advantages[valid_m]
            advantages = (advantages - adv_flat.mean()) / (adv_flat.std() + EPS)

        pg_loss = -(stacked_logP[valid_m] * advantages.detach()[valid_m]).sum() / N_valid
        v_loss = F.mse_loss(stacked_vals[valid_m], disc_ret[valid_m].detach())
        e_loss = -stacked_ents[valid_m].sum() / N_valid
        g_loss = stacked_grams[valid_m].sum() / N_valid

        total_loss = pg_loss + hp.value_coef * v_loss + hp.ent_coef * e_loss + hp.sup_coef * g_loss

        opt.zero_grad(set_to_none=True)
        SCALER.scale(total_loss).backward()
        
        # Gradient Clipping for stability
        SCALER.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=hp.grad_clip_norm)

        SCALER.step(opt); SCALER.update(); sched.step()

        logger.log({
            "pl": pg_loss.item(), "vl": v_loss.item(), "el": e_loss.item(),
            "gl": g_loss.item(), "tl": total_loss.item(),
            "reward": disc_ret[:,0].mean().item(),
            "learning_rate": sched.get_last_lr()[0],
            "seq_len": ep_len.float().mean().item()
        })
        logger.cli_step()

        if ep % 10 == 0 or ep == 1:
            print(
                f"[Combined] Epoch {ep:>4} | "
                f"Reward={logger.stats['reward'].most_recent:.3f} "
                f"(μ={logger.stats['reward'].mean:.3f}, σ={logger.stats['reward'].std:.3f}) | "
                f"AvgLen={logger.stats['seq_len'].most_recent:.2f} | "
                f"PL={logger.stats['pl'].most_recent:.3f} | "
                f"VL={logger.stats['vl'].most_recent:.3f} | "
                f"Ent={logger.stats['el'].most_recent:.3f} | "
                f"GramL={logger.stats['gl'].most_recent:.3f} | "
                f"TotL={logger.stats['tl'].most_recent:.3f} | "
                f"LR={logger.scalars['learning_rate']:.2e}"
            )

        if ep % hp.ckpt_every == 0 or ep == hp.epochs:
            path = ckptdir / f"ep{ep}.pt"
            model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(model_to_save.state_dict(), path)
            main_logger.info("✓ saved %s", path)

    final_ckpt_path = ckptdir / f"ep{hp.epochs}.pt"
    if hp.epochs % hp.ckpt_every != 0 :
        model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
        torch.save(model_to_save.state_dict(), final_ckpt_path)

    score_pool.close()
    score_pool.join()
    main_logger.info("Multiprocessing pool closed.")
    del os.environ["MAIN_PROCESS_LAUNCH_DIR"] # Clean up env var

    return final_ckpt_path

# ─────────────────────────────────────────────────────────────────────────────
# 8. Sampling helper
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def sample(model_to_sample: nn.Module, T: int, *, n: int = 4, temp: float = 1.0) -> None:
    model_to_sample.eval()
    starts = {TOKEN2ID[s] for s in ["PARTICLE", "FIELD", "ITRACT"]}
    ends = {TOKEN2ID[s] for s in ["END_PARTICLE", "END_FIELD", "END_ITRACT"]}

    for i in range(n):
        parser = TorchGrammarBatchParser(grammar, 1, device=DEVICE)
        parser.set_batch([random.choice(ROOTS)])
        seq_ids = [BOS_TOKEN]
        indent = 0
        print(f"\nSample {i+1}:\n", end="")
        
        for token_index_in_sample in range(T):
            # MODIFIED: Get both logits and values, but only use logits here
            logits_all, _ = model_to_sample(torch.tensor([seq_ids], device=DEVICE))
            
            logits_tensor = logits_all[:, -1].squeeze(0)
            mask_tensor = parser.next_token_mask(VOCAB_SIZE)[0]
            logits_tensor = logits_tensor + mask_tensor

            if temp <= EPS:
                tok_id = int(torch.argmax(logits_tensor).item())
            else:
                probs_tensor = F.softmax(logits_tensor / max(temp, 1e-5), 0)
                if not torch.isfinite(probs_tensor).any():
                    valid_ids_tensor = (mask_tensor == 0).nonzero(as_tuple=True)[0]
                    tok_id = int(valid_ids_tensor[random.randrange(len(valid_ids_tensor))].item()) if len(valid_ids_tensor) > 0 else EOS_TOKEN
                else:
                    tok_id = int(Categorical(probs_tensor).sample().item())

            token_str = ID2TOK[tok_id]

            if token_index_in_sample == 0:
                if tok_id in ends:
                    indent = max(0, indent - 1)
                    print(f"{'  ' * indent}{token_str}", end="")
                elif tok_id in starts:
                    print(f"{'  ' * indent}{token_str}", end="")
                    indent += 1
                else:
                    print(f"{token_str}", end="")
            else:
                if tok_id in ends:
                    indent = max(0, indent - 1)
                    print(f",\n{'  ' * indent}{token_str}", end="")
                elif tok_id in starts:
                    print(f",\n{'  ' * indent}{token_str}", end="")
                    indent += 1
                else:
                    print(f", {token_str}", end="")

            seq_ids.append(tok_id)
            is_valid_token = parser.fast_is_valid(torch.tensor([tok_id], device=DEVICE)).item()
            if not is_valid_token:
                print("\n...(Invalid by grammar; terminating sample)]", end="")
                break
            if tok_id == EOS_TOKEN or parser.next_lens[0] == 0:
                print("\n(Grammar complete)", end="")
                break
        print()
    print("-" * 30)
    model_to_sample.train()

# ─────────────────────────────────────────────────────────────────────────────
# 9. Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main_logger.info(f"Device: {DEVICE}, AMP: {_AMP_DTYPE if DEVICE.type=='cuda' else 'off'}")
    main_logger.info(f"Using {hp.num_score_processes} processes for score calculation.")
    main_logger.info(f"Main process CWD: {os.getcwd()}")

    transformer_model = TransformerPolicy(VOCAB_SIZE, max_T=hp.seq_len).to(DEVICE)
    optimizer = _build_adamw(transformer_model.parameters(), hp.lr)

    final_checkpoint = train_reinforce(transformer_model, optimizer, hp)
    main_logger.info("Training complete – model saved to %s", final_checkpoint)

    # Reload for sampling
    model_to_sample = TransformerPolicy(VOCAB_SIZE, max_T=hp.seq_len).to(DEVICE)
    model_to_sample.load_state_dict(torch.load(final_checkpoint, map_location=DEVICE))

    print("\n=== Greedy samples (temp=0.0) ===")
    sample(model_to_sample, hp.seq_len, n=5, temp=0.0)

    print("\n=== Temperature-0.7 samples ===")
    sample(model_to_sample, hp.seq_len, n=5, temp=0.7)

    print("\nDone.")