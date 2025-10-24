# ─────────────────────────────────────────────────────────────────────────────
# 1. Imports
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import csv, math, sys, os, random, time, queue, warnings, contextlib, logging, threading, multiprocessing
import sys # Added for sys.path manipulation in worker_init_fn
from pathlib import Path
from dataclasses import dataclass, replace
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union, Iterable, Optional, NamedTuple
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
from .classes import Logger, TorchGrammar, TorchGrammarBatchParser, ModelTester
from torch.profiler import profile, record_function, ProfilerActivity
try: multiprocessing.set_start_method('spawn', force=True)
except RuntimeError: pass
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except ImportError: pass
import json

# ─────────────────────────────────────────────────────────────────────────────
# 2. CUDA backend / global setup
# ─────────────────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True")
original_stderr = sys.stderr
class CppWarningFilter(logging.Filter):
    def filter(self, record):
        suppress_phrases = [
            "Unable to register cuFFT factory",
            "Unable to register cuDNN factory",
            "Unable to register cuBLAS factory",
            "computation placer already registered",
            "This TensorFlow binary is optimized"
        ]
        return not any(phrase in record.getMessage() for phrase in suppress_phrases)

cpp_logger = logging.getLogger("cpp_warnings_filter")
cpp_logger.propagate = False
handler = logging.StreamHandler(original_stderr)
handler.addFilter(CppWarningFilter())
cpp_logger.addHandler(handler)

class StderrRedirector:
    def __init__(self, logger):
        self._logger = logger
        self._local = threading.local()
    def write(self, message):
        if getattr(self._local, 'is_writing', False): return
        try:
            self._local.is_writing = True
            if message.rstrip(): self._logger.warning(message.rstrip())
        finally: self._local.is_writing = False
    def flush(self):
        for h in self._logger.handlers: h.flush()

sys.stderr = StderrRedirector(cpp_logger)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 3 = FATAL errors only

SEED = 42
EPS = 1e-9
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
    TOKEN2ID["PARTICLE"]: "PARTICL   E",
    TOKEN2ID["FIELD"]: "FIELD",
    TOKEN2ID["ITRACT"]: "ITRACT",
}
END_TOKS = {
    TOKEN2ID["END_PARTICLE"]: "PARTICLE",
    TOKEN2ID["END_FIELD"]: "FIELD",
    TOKEN2ID["END_ITRACT"]: "ITRACT",
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. Transformer (actor-critic)
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
        self.value = nn.Linear(d_model, 1)

    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.tok_emb(seq) + self.pos_emb(self.pos[: seq.size(1)].to(seq.device))
        h = self.encoder(x)
        return self.actor(h), self.value(h).squeeze(-1)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class HParams:
    epochs: int = 500
    batch_size: int = 4
    seq_len: int = 512
    lr: float = 3e-4
    gamma: float = 0.997
    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    gae_lambda: float = 0.95
    sup_coef: float = 0.05
    value_coef: float = 0.5
    ent_coef: float = 0.01
    grad_clip_norm: float = 1.0
    ckpt_every: int = 10
    num_score_processes: Optional[int] = None
    buffer_size: int = 2048
    min_buffer_size: int = 256
    rank_k: float = 0.5
    enable_logging: bool = True
    debug: bool = False

hp = HParams()
if hp.num_score_processes is None:
    hp = replace(hp, num_score_processes=os.cpu_count() or 4)
if hp.min_buffer_size > hp.buffer_size:
    warnings.warn(f"min_buffer_size ({hp.min_buffer_size}) > buffer_size ({hp.buffer_size}). Setting min_buffer_size = buffer_size.")
    hp = replace(hp, min_buffer_size=hp.buffer_size)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Rank-Based Experience Buffer
# ─────────────────────────────────────────────────────────────────────────────
class Trajectory(NamedTuple):
    """Stores raw data for a trajectory, detached from any computation graph."""
    seq: torch.Tensor
    rewards: torch.Tensor
    logps: torch.Tensor
    values: torch.Tensor
    ep_len: int
    total_reward: float

class ExperienceBuffer:
    def __init__(self, hp: HParams):
        self.buffer = deque(maxlen=hp.buffer_size)
        self.hp = hp

    def __len__(self) -> int:
        return len(self.buffer)

    def add_batch(self, generated_seqs: torch.Tensor, batch_data: Dict[str, torch.Tensor]) -> None:
        """Adds a batch of generated trajectories to the buffer."""
        rews = batch_data['rewards']
        logps = batch_data['logps']
        values = batch_data['values']
        
        disc_ret_rank = torch.zeros_like(rews)
        R_rank = torch.zeros(rews.size(0), device=DEVICE)
        for rev_t in range(rews.size(1) - 1, -1, -1):
            R_rank = rews[:, rev_t] + self.hp.gamma * R_rank
            disc_ret_rank[:, rev_t] = R_rank

        for i in range(rews.size(0)):
            ep_len = batch_data['ep_len'][i].item()
            if ep_len > 0:
                traj = Trajectory(
                    seq=generated_seqs[i, :ep_len + 1].clone().cpu(),
                    rewards=rews[i, :ep_len].clone().cpu(),
                    logps=logps[i, :ep_len].clone().cpu(),
                    values=values[i, :ep_len].clone().cpu(),
                    ep_len=ep_len,
                    total_reward=disc_ret_rank[i, 0].item()
                )
                if self.hp.debug:
                    print(f"  [DEBUG] Adding trajectory to buffer. Length: {traj.ep_len}, Total Reward: {traj.total_reward:.4f}")
                self.buffer.append(traj)

    def sample(self) -> Dict[str, torch.Tensor]:
        """Ranks trajectories, samples a batch, and computes GAE + returns."""
        buffer_rewards = torch.tensor([t.total_reward for t in self.buffer], device=DEVICE)
        D_size = len(self.buffer)
        ranks = buffer_rewards.argsort(descending=True).argsort().float() + 1.0
        weights_unnorm = (self.hp.rank_k * D_size + ranks).pow(-1)
        weights = weights_unnorm / weights_unnorm.sum()
        indices = torch.multinomial(weights, self.hp.batch_size, replacement=True)
        
        samples = [self.buffer[i] for i in indices]
        max_len = max(s.ep_len for s in samples)
        
        collated_seqs = torch.full((self.hp.batch_size, max_len + 1), BOS_TOKEN, device=DEVICE, dtype=torch.long)
        collated_rewards = torch.zeros((self.hp.batch_size, max_len), device=DEVICE)
        collated_logps = torch.zeros((self.hp.batch_size, max_len), device=DEVICE)
        collated_values = torch.zeros((self.hp.batch_size, max_len), device=DEVICE)
        collated_ep_len = torch.tensor([s.ep_len for s in samples], device=DEVICE, dtype=torch.long)
        
        for i, s in enumerate(samples):
            len_s = s.ep_len
            collated_seqs[i, :len_s + 1] = s.seq.to(DEVICE)
            collated_rewards[i, :len_s] = s.rewards.to(DEVICE)
            collated_logps[i, :len_s] = s.logps.to(DEVICE)
            collated_values[i, :len_s] = s.values.to(DEVICE)

        advantages = torch.zeros_like(collated_rewards)
        last_gae_lam = 0
        for t in reversed(range(max_len)):
            is_not_last_step = (t < collated_ep_len - 1).float()
            if t == max_len - 1:
                next_values = torch.zeros_like(collated_values[:, t])
            else:
                next_values = collated_values[:, t + 1]
            delta = collated_rewards[:, t] + self.hp.gamma * next_values * is_not_last_step - collated_values[:, t]
            advantages[:, t] = last_gae_lam = delta + self.hp.gamma * self.hp.gae_lambda * last_gae_lam * is_not_last_step
        
        returns = advantages + collated_values

        return {
            'seq': collated_seqs,
            'logps': collated_logps,
            'values': collated_values,
            'returns': returns,
            'advantages': advantages,
            'ep_len': collated_ep_len
        }

def ensure_dirs(paths: Union[str, Path, Iterable[Union[str, Path]]]) -> None:
    if isinstance(paths, (str, Path)): paths = [paths]
    for p in paths: Path(p).mkdir(parents=True, exist_ok=True)
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
    try: return torch.optim.AdamW(params, lr=lr, fused=True)
    except Exception:
        warnings.warn("Fused AdamW unavailable → fallback."); return torch.optim.AdamW(params, lr=lr)
_global_model_tester_instance: Optional[ModelTester] = None
def _get_model_tester_instance():
    global _global_model_tester_instance
    if _global_model_tester_instance is None: _global_model_tester_instance = ModelTester()
    return _global_model_tester_instance
def worker_init_fn():
    main_launch_dir_str = os.environ.get("MAIN_PROCESS_LAUNCH_DIR")
    if main_launch_dir_str:
        project_root_for_imports = Path(main_launch_dir_str).parent
        if str(project_root_for_imports) not in sys.path: sys.path.insert(0, str(project_root_for_imports))
    _get_model_tester_instance()
def _score_single_sequence(tokens: List[str]) -> float:
    tester = _get_model_tester_instance()
    try:
        score_val, _ = tester.test_model(tokens); return float(score_val) if score_val is not None else 0.0
    except Exception as e:
        print(f"WORKER ERROR: scoring failed for sequence {' '.join(tokens)} -> {e}", file=sys.stderr)
        return 0.0

# ─────────────────────────────────────────────────────────────────────────────
# 7. PPO training loop  ← full, self-contained implementation (with Steps 1 & 2)
# ─────────────────────────────────────────────────────────────────────────────
def train_reinforce(model: nn.Module, opt, hp: HParams) -> Path:
    # ────────────── bookkeeping & dirs ──────────────
    run_id         = datetime.now().strftime("comb_%Y%m%d_%H%M%S")
    root_dir       = Path(__file__).resolve().parent          # …/training
    logdir         = root_dir / "particle" / "runs" / run_id
    ckptdir        = logdir / "checkpoints"
    prof_dir       = logdir / "profiler"                      # tensorboard traces
    ensure_dirs([logdir, ckptdir, prof_dir])

    # ────────────── logger (unchanged) ──────────────
    logger: Optional[Logger] = None
    if hp.enable_logging:
        logger = Logger(logdir, cli_interval=1, state_interval=1,
                        weights_interval=hp.ckpt_every)
        logger.model = model
        for name in [
            "reward", "seq_len", "pl", "vl", "el", "gl", "tl", "buffer_size",
            "n_particles", "n_fields", "n_itracts", "l1_norm", "l2_norm",
            "grad_l1_norm", "grad_l2_norm", "logit_entropy", "approx_kl"
        ]:
            logger.add_metric(name, kind="stat")
        for name in ["n_updates", "n_rollouts", "n_samples", "learning_rate"]:
            logger.add_metric(name, kind="scalar")
        if CAN_COMPILE:
            print("Compiling model …")
            model = torch.compile(model, mode="max-autotune")
            print("Compilation complete.")

    # ────────────── schedulers & pools ──────────────
    sched          = warmup_cosine(opt, int(hp.epochs * 0.10), hp.epochs)
    os.environ["MAIN_PROCESS_LAUNCH_DIR"] = os.getcwd()
    score_pool     = multiprocessing.Pool(
        processes=hp.num_score_processes, initializer=worker_init_fn
    )

    # profiler schedule: wait-0 warmup-1 active-1 repeat-10
    prof_sched     = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=10)
    trace_handler  = torch.profiler.tensorboard_trace_handler(str(prof_dir))
    optimal_settings_path = logdir / "optimal_settings.json"

    experience_buffer = ExperienceBuffer(hp)

    # ───────────────────────── main profile context ─────────────────────────
    with profile(
        activities      = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule        = prof_sched,
        on_trace_ready  = trace_handler,
        record_shapes   = True,
        profile_memory  = True,
        with_stack      = True,
        with_flops      = True,
        with_modules    = True
    ) as prof:

        # ════════════════════════════════════════════════════════════════════
        #                              EPOCH LOOP
        # ════════════════════════════════════════════════════════════════════
        for ep in trange(1, hp.epochs + 1, desc="epochs",
                         leave=False, disable=not hp.debug):
            prof.step()                         # advance profiler schedule
            model.eval()
            B = hp.batch_size

            # ───────────────── rollout generation ─────────────────
            with record_function("rollout_generation"):
                parser      = TorchGrammarBatchParser(grammar, B, device=DEVICE)
                parser.set_batch(random.choices(ROOTS, k=B))

                seq         = torch.full((B, hp.seq_len + 1), BOS_TOKEN,
                                          device=DEVICE, dtype=torch.long)
                ep_len      = torch.zeros(B, dtype=torch.long, device=DEVICE)
                alive       = torch.ones(B, dtype=torch.bool, device=DEVICE)

                rewards_history, logps_history, values_history = [], [], []
                history      = [[BOS_TOKEN] for _ in range(B)]
                n_particles  = torch.zeros(B, device=DEVICE)
                n_fields     = torch.zeros(B, device=DEVICE)
                n_itracts    = torch.zeros(B, device=DEVICE)

                for t in range(hp.seq_len):
                    ep_len[alive] += 1
                    with torch.no_grad():
                        logits_t, values_t = model(seq[:, : t + 1].clone())
                    logits  = logits_t[:, -1].float()
                    mask    = parser.next_token_mask(VOCAB_SIZE)
                    eligible = alive & (~parser.next_lens.eq(0))
                    if not eligible.any():
                        break

                    probs = F.softmax(logits + mask, dim=-1)
                    probs[~eligible] = 0.0
                    probs[~eligible, EOS_TOKEN] = 1.0
                    if torch.isnan(probs).any():
                        nan_rows = torch.isnan(probs).any(dim=-1)
                        probs[nan_rows] = 0.0
                        probs[nan_rows, EOS_TOKEN] = 1.0

                    dist      = Categorical(probs=probs)
                    acts      = dist.sample()
                    logps_val = dist.log_prob(acts)

                    # debug print
                    if hp.debug:
                        idx0_tok = ID2TOK[acts[0].item()] if B > 0 else 'N/A'
                        print(f"[DEBUG] Ep {ep} step {t} » sample {idx0_tok}, "
                              f"alive {alive.sum().item()}/{B}")

                    # rewards & bookkeeping
                    rew        = torch.zeros(B, device=DEVICE)
                    valid_row  = parser.fast_is_valid(acts)
                    n_particles[valid_row] += (acts[valid_row] == TOKEN2ID["PARTICLE"]).float()
                    n_fields   [valid_row] += (acts[valid_row] == TOKEN2ID["FIELD"]).float()
                    n_itracts  [valid_row] += (acts[valid_row] == TOKEN2ID["ITRACT"]).float()

                    # optional external scoring
                    sequences_to_score_info: List[Dict[str, Any]] = []
                    for i in range(B):
                        if not eligible[i]:
                            continue
                        tok_id = acts[i].item()
                        history[i].append(tok_id)

                        if tok_id in END_TOKS:          # object closed
                            obj_type = END_TOKS[tok_id]
                            start_tok_id = next(
                                (tid for tid, s in START_TOKS.items() if s == obj_type),
                                None
                            )
                            if start_tok_id is not None:
                                rev_idx = next(
                                    (k for k in range(len(history[i]) - 2, -1, -1)
                                     if history[i][k] == start_tok_id),
                                    -1
                                )
                                if rev_idx != -1:
                                    sub_tokens = [ID2TOK[x] for x in history[i][rev_idx:]]
                                    sequences_to_score_info.append(
                                        {'batch_idx': i, 'bonus_type': 'object',
                                         'tokens': sub_tokens}
                                    )
                        if tok_id == EOS_TOKEN:         # model closed
                            full_tokens = [ID2TOK[x] for x in history[i][1:]]
                            sequences_to_score_info.append(
                                {'batch_idx': i, 'bonus_type': 'eos',
                                 'tokens': full_tokens}
                            )

                    if sequences_to_score_info:
                        try:
                            scored_vals = score_pool.map(
                                _score_single_sequence,
                                [info['tokens'] for info in sequences_to_score_info]
                            )
                            for info, s_val in zip(sequences_to_score_info, scored_vals):
                                if info['bonus_type'] == 'object':
                                    rew[info['batch_idx']] += s_val
                                else:                                    # "eos"
                                    rew[info['batch_idx']] += 1000 * s_val
                        except Exception as e:
                            main_logger.error("score_pool error", exc_info=e)

                    # history stacks
                    rewards_history.append(rew)
                    logps_history.append(logps_val.clone())
                    values_history.append(values_t[:, -1].clone())

                    seq[:, t + 1] = acts
                    alive = alive & valid_row & (acts != EOS_TOKEN)

                # push into buffer
                if rewards_history:
                    batch_dict = {
                        'rewards': torch.stack(rewards_history, dim=1),
                        'logps':   torch.stack(logps_history,   dim=1),
                        'values':  torch.stack(values_history,  dim=1),
                        'ep_len':  ep_len
                    }
                    experience_buffer.add_batch(seq, batch_dict)
                    gen_reward = batch_dict['rewards'].sum(1).mean().item()
                    if hp.enable_logging:
                        logger.log({
                            "reward": gen_reward,
                            "seq_len": ep_len.float().mean().item(),
                            "buffer_size": len(experience_buffer),
                            "n_particles": n_particles.mean().item(),
                            "n_fields":    n_fields.mean().item(),
                            "n_itracts":   n_itracts.mean().item(),
                        })

            # ───────────────── PPO update ─────────────────
            with record_function("ppo_update_phase"):
                model.train()
                if len(experience_buffer) < hp.min_buffer_size:
                    if hp.debug:
                        print(f"[DEBUG] Ep {ep} skip train (buffer "
                              f"{len(experience_buffer)} < {hp.min_buffer_size})")
                    sched.step()
                    if hp.enable_logging:
                        logger.cli_step()
                    continue

                for _ in range(hp.ppo_epochs):
                    tbatch      = experience_buffer.sample()
                    b_seq       = tbatch['seq']
                    b_actions   = b_seq[:, 1:].clone()
                    max_len     = tbatch['returns'].size(1)

                    logits, vals = model(b_seq[:, :-1])
                    dist         = Categorical(logits=logits)
                    new_logps    = dist.log_prob(b_actions)
                    ents         = dist.entropy()

                    ts_grid      = torch.arange(max_len, device=DEVICE).unsqueeze(0)
                    valid_mask   = ts_grid < tbatch['ep_len'].unsqueeze(1)
                    N_valid      = valid_mask.sum().clamp_min(1.0)

                    adv = tbatch['advantages']
                    if N_valid > 1:
                        flat = adv[valid_mask]
                        adv  = (adv - flat.mean()) / (flat.std() + EPS)

                    # PPO losses
                    log_ratio = new_logps - tbatch['logps']
                    ratio     = log_ratio.exp()
                    surr1     = ratio * adv
                    surr2     = torch.clamp(ratio, 1 - hp.clip_epsilon,
                                            1 + hp.clip_epsilon) * adv
                    pg_loss   = -torch.min(surr1, surr2)[valid_mask].sum() / N_valid
                    v_loss    = F.mse_loss(vals[valid_mask],
                                           tbatch['returns'][valid_mask])
                    e_loss    = -ents[valid_mask].sum() / N_valid

                    # grammar penalties
                    train_parser = TorchGrammarBatchParser(grammar, hp.batch_size,
                                                           device=DEVICE)
                    train_parser.set_batch(random.choices(ROOTS, k=hp.batch_size))
                    g_penalties = []
                    for t in range(max_len):
                        mask_t  = train_parser.next_token_mask(VOCAB_SIZE)
                        logits_t = logits[:, t]
                        banned   = (mask_t != 0.0).float()
                        pen_row  = F.relu(logits_t * banned).sum(-1)
                        g_penalties.append(
                            pen_row / banned.sum(-1).clamp_min(1.0)
                        )
                        train_parser.fast_is_valid(b_actions[:, t])
                    g_loss = torch.stack(g_penalties, dim=1)[valid_mask].sum() / N_valid

                    total_loss = (pg_loss +
                                  hp.value_coef * v_loss +
                                  hp.ent_coef  * e_loss +
                                  hp.sup_coef  * g_loss)

                    opt.zero_grad(set_to_none=True)
                    SCALER.scale(total_loss).backward()
                    SCALER.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   hp.grad_clip_norm)
                    SCALER.step(opt)
                    SCALER.update()

                    if hp.enable_logging:
                        logger.state_step()

            # ───────────────── LR & scalar logging ─────────────────
            sched.step()
            if hp.enable_logging:
                scale = SCALER.get_scale()
                grads_exist = any(p.grad is not None for p in model.parameters())
                if grads_exist:
                    g_l1 = sum((p.grad / scale).abs().sum().item()
                               for p in model.parameters() if p.grad is not None)
                    g_l2 = math.sqrt(sum(((p.grad / scale) ** 2).sum().item()
                               for p in model.parameters() if p.grad is not None))
                else:
                    g_l1 = g_l2 = 0.0

                base_log = {
                    "n_updates": hp.ppo_epochs,
                    "n_rollouts": 1,
                    "n_samples":  B,
                    "learning_rate": sched.get_last_lr()[0],
                    "l1_norm": sum(p.abs().sum().item() for p in model.parameters()),
                    "l2_norm": math.sqrt(sum((p ** 2).sum().item()
                                             for p in model.parameters())),
                    "grad_l1_norm": g_l1,
                    "grad_l2_norm": g_l2,
                }
                logger.log(base_log)
                logger.cli_step()

            # ───────────────── periodic checkpoint ─────────────────
            if ep % hp.ckpt_every == 0 or ep == hp.epochs:
                ckpt_path = ckptdir / f"ep{ep}.pt"
                model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save(model_to_save.state_dict(), ckpt_path)
                main_logger.info("✓ saved %s", ckpt_path)

        # ════════════════════════════════════════════════════════════════
        #              END OF TRAINING  →  Steps 1 & 2
        # ════════════════════════════════════════════════════════════════

        # STEP 1 – quick hotspot tables:
        print("\n================  PyTorch profiler (top-20 CUDA)  ================")
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
        print("\n================  PyTorch profiler (top-20 CPU)   ================")
        print(prof.key_averages().table(sort_by="self_cpu_time_total",  row_limit=20))

        # STEP 2 – friendly TensorBoard hint:
        print("\n🎛  Inspect interactive traces with:\n"
              f"    tensorboard --logdir={prof_dir}\n"
              "    # then open http://localhost:6006 → Profile tab\n")

        # also dump folded stacks for flamegraph.pl users
        prof.export_stacks(str(logdir / "profiler_cpu.stacks"),
                           metric="self_cpu_time_total")
        prof.export_stacks(str(logdir / "profiler_cuda.stacks"),
                           metric="self_cuda_time_total")
        print("✓  profiler_cpu.stacks & profiler_cuda.stacks exported\n")

    # ───────────── final checkpoint (if not already saved) ─────────────
    final_ckpt = ckptdir / f"ep{hp.epochs}.pt"
    if hp.epochs % hp.ckpt_every:
        model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
        torch.save(model_to_save.state_dict(), final_ckpt)

    score_pool.close(); score_pool.join()
    main_logger.info("Multiprocessing pool closed.")
    if "MAIN_PROCESS_LAUNCH_DIR" in os.environ:
        del os.environ["MAIN_PROCESS_LAUNCH_DIR"]

    return final_ckpt

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
            logits_all, _ = model_to_sample(torch.tensor([seq_ids], device=DEVICE))
            logits_tensor, mask_tensor = logits_all[:, -1].squeeze(0), parser.next_token_mask(VOCAB_SIZE)[0]
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
                if tok_id in ends: indent = max(0, indent - 1); print(f"{'  ' * indent}{token_str}", end="")
                elif tok_id in starts: print(f"{'  ' * indent}{token_str}", end=""); indent += 1
                else: print(f"{token_str}", end="")
            else:
                if tok_id in ends: indent = max(0, indent - 1); print(f",\n{'  ' * indent}{token_str}", end="")
                elif tok_id in starts: print(f",\n{'  ' * indent}{token_str}", end=""); indent += 1
                else: print(f", {token_str}", end="")
            seq_ids.append(tok_id)
            is_valid_token = parser.fast_is_valid(torch.tensor([tok_id], device=DEVICE)).item()
            if not is_valid_token: print("\n...(Invalid by grammar; terminating sample)]", end=""); break
            if tok_id == EOS_TOKEN or parser.next_lens[0] == 0: print("\n(Grammar complete)", end=""); break
        print()
    print("-" * 30)
    model_to_sample.train()
    
# ─────────────────────────────────────────────────────────────────────────────
# 9. Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example of turning on debug mode:
    # hp = replace(hp, debug=True, epochs=20, enable_logging=False)
    
    main_logger.info(f"Device: {DEVICE}, AMP: {_AMP_DTYPE if DEVICE.type=='cuda' else 'off'}")
    main_logger.info(f"Using {hp.num_score_processes} processes for score calculation.")
    main_logger.info(f"Algorithm: PPO with GAE. Buffer: size={hp.buffer_size}, k={hp.rank_k}")
    main_logger.info(f"Logging {'enabled' if hp.enable_logging else 'disabled'}")
    main_logger.info(f"Debug mode: {'ON' if hp.debug else 'OFF'}")
    main_logger.info(f"Main process CWD: {os.getcwd()}")

    transformer_model = TransformerPolicy(VOCAB_SIZE, max_T=hp.seq_len + 1).to(DEVICE)
    optimizer = _build_adamw(transformer_model.parameters(), hp.lr)

    final_checkpoint = train_reinforce(transformer_model, optimizer, hp)
    main_logger.info("Training complete – model saved to %s", final_checkpoint)

    # Reload for sampling
    model_to_sample = TransformerPolicy(VOCAB_SIZE, max_T=hp.seq_len + 1).to(DEVICE)
    model_to_sample.load_state_dict(torch.load(final_checkpoint, map_location=DEVICE))

    print("\n=== Greedy samples (temp=0.0) ===")
    sample(model_to_sample, hp.seq_len, n=5, temp=0.0)

    print("\n=== Temperature-0.7 samples ===")
    sample(model_to_sample, hp.seq_len, n=5, temp=0.7)

    print("\nDone.")