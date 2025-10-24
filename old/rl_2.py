# ─────────────────────────────────────────────────────────────────────────────
# 1. Imports and Global Setup
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import csv, math, sys, os, random, time, warnings, logging
from pathlib import Path
from dataclasses import dataclass
from collections import deque
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple, Union, NamedTuple, Optional

# Third-party libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sympy import Matrix, sympify

# --- Global Setup ---
warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True")
SEED = 42
EPS = 1e-9 # Epsilon for numerical stability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed everything for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# Mixed precision settings
_AMP_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
SCALER = amp.GradScaler(enabled=(DEVICE.type == "cuda"))
CAN_COMPILE = hasattr(torch, "compile")

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
main_logger = logging.getLogger(__name__)

# --- Type Aliases ---
Token     = Union[int, Tuple[int, ...], str, Dict[str, Any]]
Predicate = Callable[[int], bool]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Helper Classes
# ─────────────────────────────────────────────────────────────────────────────
class RunningStat:
    """Exponential moving average (EMA) style running mean / variance tracker."""
    def __init__(self, alpha: float = 0.99) -> None:
        if not 0 < alpha <= 1: raise ValueError("alpha must be in (0, 1]")
        self._alpha = alpha
        self._count: int = 0
        self._mean: float = 0.0
        self._var: float = 0.0
        self._most_recent: float = 0.0

    @property
    def mean(self) -> float: return self._mean
    @property
    def std(self) -> float: return math.sqrt(max(0.0, self._var))
    @property
    def most_recent(self) -> float: return self._most_recent

    def update(self, x: float) -> None:
        if not math.isfinite(x): return
        self._most_recent = x
        self._count += 1
        if self._count == 1:
            self._mean, self._var = x, 0.0
            return
        delta = x - self._mean
        self._mean = self._alpha * self._mean + (1 - self._alpha) * x
        self._var = self._alpha * (self._var + (1 - self._alpha) * delta**2)

class Logger:
    """Generic experiment logger for CLI, CSV, and TensorBoard."""
    CLI_CSV, STATE_CSV = "cli.csv", "state.csv"
    WEIGHTS_DIR, TBOARD_DIR = "weights", "tensorboard"

    def __init__(self, path: str | os.PathLike, *, cli_interval: int = 1, state_interval: int = 1, weights_interval: int = 100_000):
        self.root = Path(path); self.root.mkdir(parents=True, exist_ok=True)
        self.cli_interval, self.state_interval, self.weights_interval = cli_interval, state_interval, weights_interval
        self.weights_dir = self.root / self.WEIGHTS_DIR; self.weights_dir.mkdir(exist_ok=True)
        self.tboard_dir = self.root / self.TBOARD_DIR; self.tboard_dir.mkdir(exist_ok=True)
        self._cli_csv = (self.root / self.CLI_CSV).open("w", newline="", buffering=1)
        self._state_csv = (self.root / self.STATE_CSV).open("w", newline="", buffering=1)
        self.cli_writer, self.state_writer = csv.writer(self._cli_csv), csv.writer(self._state_csv)
        self.tb = SummaryWriter(str(self.tboard_dir), flush_secs=1)
        self.timestep, self.seconds, self._t0 = 0, 0.0, time.time()
        self.stats: dict[str, RunningStat] = {}
        self.scalars: dict[str, float] = {}
        self.state, self.action, self.logits, self.model = None, None, None, None
        self._cli_header_written, self._state_header_written = False, False
        self.add_metric("fps", kind="stat", alpha=0.9)

    def __del__(self):
        if hasattr(self, '_cli_csv'): self._cli_csv.close()
        if hasattr(self, '_state_csv'): self._state_csv.close()
        if hasattr(self, 'tb'): self.tb.close()

    def add_metric(self, name: str, *, kind: str = "stat", alpha: float = 0.99):
        if kind not in {"stat", "scalar"}: raise ValueError("kind must be 'stat' or 'scalar'")
        if name in self.stats or name in self.scalars: raise ValueError(f"Metric '{name}' already registered")
        if kind == "stat": self.stats[name] = RunningStat(alpha=alpha)
        else: self.scalars[name] = 0.0

    def log(self, values: Dict[str, Any]):
        for k, v in values.items():
            if k in self.stats: self.stats[k].update(float(v))
            elif k in self.scalars: self.scalars[k] = float(v)
            elif k in {"state", "action", "logits", "weights"}: setattr(self, k if k != "weights" else "model", v)
            else: raise KeyError(f"'{k}' not registered – call add_metric() first")

    def state_step(self):
        self.timestep += 1
        self.seconds = time.time() - self._t0
        self.stats["fps"].update(self.timestep / max(self.seconds, 1e-9))
        if not self._state_header_written: self._write_state_header(); self._state_header_written = True
        if self.timestep % self.state_interval == 0:
            self.state_writer.writerow([self.timestep, self.seconds, self.state, self.stats.get("reward", RunningStat()).most_recent, self.action, self.logits])
            self.tb.add_scalar("state/reward", self.stats.get("reward", RunningStat()).most_recent, self.timestep)
            self.tb.add_scalar("state/fps", self.stats["fps"].mean, self.timestep)
        if self.model is not None and self.timestep % self.weights_interval == 0: torch.save(self.model.state_dict(), self.weights_dir / f"weights_{self.timestep}.pt")

    def cli_step(self):
        if not self._cli_header_written: self._write_cli_header(); self._cli_header_written = True
        row = [self.timestep, self.seconds, self.stats["fps"].mean]
        for name in sorted(self.scalars): row.append(self.scalars[name])
        for name in sorted(self.stats):
            if name == "fps": continue
            st = self.stats[name]; row.extend([st.most_recent, st.mean, st.std])
        self.cli_writer.writerow(row)
        for name, st in self.stats.items():
            self.tb.add_scalar(f"train/{name}_mean", st.mean, self.timestep); self.tb.add_scalar(f"train/{name}_mr", st.most_recent, self.timestep)
        for name, val in self.scalars.items(): self.tb.add_scalar(f"train/{name}", val, self.timestep)
        self.tb.flush()

    def _write_state_header(self): self.state_writer.writerow(["timestep", "seconds", "state", "reward_most_recent", "action", "logits"])
    def _write_cli_header(self):
        hdr = ["timestep", "seconds", "fps_mean"]
        hdr.extend(sorted(self.scalars))
        for name in sorted(self.stats):
            if name != "fps": hdr.extend([f"{name}_most_recent", f"{name}_mean", f"{name}_std"])
        self.cli_writer.writerow(hdr)

class TorchGrammar:
    """Immutable container mapping symbol → rule‑list for grammar-guided generation."""
    def __init__(self) -> None:
        self.raw_rules: Dict[str, List[Token]] = {}
        self.symbol_to_index: Dict[str, int] = {}
        self.index_to_symbol: Dict[int, str] = {}

    def add_object(self, name: str, tokens: List[Token]) -> None:
        self.raw_rules[name] = tokens
        if name not in self.symbol_to_index:
            idx = len(self.symbol_to_index)
            self.symbol_to_index[name] = idx
            self.index_to_symbol[idx] = name

    @staticmethod
    def expand_literal(tok: Token) -> List[int]:
        """Expand a literal spec (int, range tuple, list tuple) to a list of token IDs."""
        if isinstance(tok, int): return [tok]
        if isinstance(tok, tuple): return list(range(tok[0], tok[1] + 1)) if len(tok) == 2 else list(tok)
        raise ValueError(f"Invalid token literal type: {type(tok)}")

    @staticmethod
    def wrap_pred(raw: Any) -> Predicate:
        """Return predicate `raw` unchanged if callable, else create a `count >= N` predicate."""
        return raw if callable(raw) else (lambda cnt, m=raw: cnt >= m)

class TorchGrammarBatchParser:
    """Fast, batch-parallel, GPU-accelerated grammar parser."""
    DEFAULT_MAX_DEPTH, DEFAULT_MAX_LEN, DEFAULT_MAX_NEXT = 16, 128, 32

    def __init__(self, grammar: TorchGrammar, batch_size: int, *, device: str | torch.device, max_depth: int | None = None, max_len: int | None = None, max_next: int | None = None):
        self.g, self.bs, self.dev = grammar, batch_size, torch.device(device)
        self.max_depth, self.max_len, self.max_next = max_depth or self.DEFAULT_MAX_DEPTH, max_len or self.DEFAULT_MAX_LEN, max_next or self.DEFAULT_MAX_NEXT
        self.stack = torch.full((self.bs, self.max_depth, 3), -1, dtype=torch.long, device=self.dev)
        self.ptr = torch.zeros(self.bs, dtype=torch.long, device=self.dev)
        self.next_cache = torch.full((self.bs, self.max_next), -1, dtype=torch.long, device=self.dev)
        self.next_lens = torch.zeros(self.bs, dtype=torch.long, device=self.dev)

    def set_batch(self, roots: List[str]) -> None:
        for i, name in enumerate(roots):
            idx = self.g.symbol_to_index[name]
            self.stack[i, 0] = torch.tensor([idx, 0, 0], device=self.dev) # [symbol_idx, token_ptr, repetition_count]
            self.ptr[i] = 1
        self._refresh_next()

    def fast_is_valid(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(self.bs, dtype=torch.bool, device=self.dev)
        for i in range(self.bs):
            if self.ptr[i] == 0: continue
            tok = tokens[i].item()
            n_cached = self.next_lens[i].item()
            if n_cached and (self.next_cache[i, :n_cached] == tok).any():
                mask[i] = True
                o_idx, o_ptr, o_rep = self.stack[i, self.ptr[i] - 1].tolist()
                obj_name = self.g.index_to_symbol[o_idx]
                curr_tok = self.g.raw_rules[obj_name][o_ptr]
                self._apply_token(i, curr_tok, tok)
        self._refresh_next()
        return mask

    def next_token_mask(self, vocab_size: int, *, fill_value: float = float("-inf")) -> torch.Tensor:
        mask = torch.full((self.bs, vocab_size), fill_value, device=self.dev)
        for i in range(self.bs):
            k = self.next_lens[i].item()
            if k: mask[i, self.next_cache[i, :k]] = 0.0
        return mask

    def _compute_valid(self, tok: Token, rep_cnt: int, tokens: List[Token], idx: int) -> List[int]:
        if isinstance(tok, dict):
            sub, raw_pred = next(iter(tok.items())); pred = self.g.wrap_pred(raw_pred)
            valid: List[int] = []
            # Check if repetition is satisfied. If not, the only valid option is to expand the sub-rule.
            if not pred(rep_cnt): return self.g.expand_literal(self.g.raw_rules[sub][0])
            # If satisfied, we can either continue repeating the sub-rule...
            valid.extend(self.g.expand_literal(self.g.raw_rules[sub][0]))
            # ...or move to the next token in the current rule.
            if idx + 1 < len(tokens):
                nxt = tokens[idx + 1]; nxt_lit = nxt if not isinstance(nxt, str) else self.g.raw_rules[nxt][0]
                valid.extend(self.g.expand_literal(nxt_lit))
            return valid
        return self.g.expand_literal(self.g.raw_rules[tok][0] if isinstance(tok, str) else tok)

    def _apply_token(self, i: int, curr: Token, tok: int) -> None:
        if isinstance(curr, dict):
            sub, _ = next(iter(curr.items()))
            # If the chosen token is not from the sub-rule, it must be from the next token in the parent rule.
            # In this case, we just pop the current `{sub: pred}` token from the stack.
            if tok not in set(self.g.expand_literal(self.g.raw_rules[sub][0])):
                self._pop_and_advance(i); return
            # Otherwise, we are descending into the sub-rule.
            depth = self.ptr[i].item(); sid = self.g.symbol_to_index[sub]
            if depth < self.max_depth:
                self.stack[i, depth] = torch.tensor([sid, 0, 0], device=self.dev); self.ptr[i] += 1
            self.stack[i, depth - 1, 2] += 1 # Increment repetition count
        elif isinstance(curr, str):
            depth, sid = self.ptr[i].item(), self.g.symbol_to_index[curr]
            if depth < self.max_depth:
                self.stack[i, depth] = torch.tensor([sid, 0, 0], device=self.dev); self.ptr[i] += 1
        else: # Literal token
            self.stack[i, self.ptr[i] - 1, 1] += 1 # Advance token pointer
            self.stack[i, self.ptr[i] - 1, 2] = 0 # Reset repetition count

    def _pop_and_advance(self, i: int) -> None:
        self.ptr[i] -= 1
        if self.ptr[i] > 0:
            self.stack[i, self.ptr[i] - 1, 1] += 1 # Advance parent rule's token pointer
            self.stack[i, self.ptr[i] - 1, 2] = 0  # Reset parent rule's repetition count

    def _refresh_next(self) -> None:
        self.next_lens.zero_()
        for i in range(self.bs):
            while 0 < self.ptr[i].item() <= self.max_depth:
                s_ptr = self.ptr[i].item() - 1
                o_idx, o_ptr, o_rep = self.stack[i, s_ptr].tolist()
                obj_name = self.g.index_to_symbol[o_idx]
                tokens = self.g.raw_rules[obj_name]
                if o_ptr >= len(tokens): self._pop_and_advance(i); continue
                curr = tokens[o_ptr]; valid = self._compute_valid(curr, o_rep, tokens, o_ptr)
                if valid:
                    k = min(len(valid), self.max_next)
                    self.next_cache[i, :k] = torch.tensor(valid[:k], device=self.dev); self.next_lens[i] = k; break
                self.ptr[i] -= 1


# ─────────────────────────────────────────────────────────────────────────────
# 3. GPU Scorer and Validator
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "MAX_GROUPS": 5, "MAX_PARTICLES": 20, "MAX_FIELDS": 10, "MAX_PARTICLES_PER_FIELD": 10,
    "MAX_INTERACTIONS": 5, "PADDING_VALUE": -1, "GROUP_TYPE_MAP": {"U": 1, "SU": 2},
    "FIELD_TYPE_MAP": {"fermion": 1, "real": 2, "complex": 3, "vector": 4},
    "REP_MAP": {"singlet": 1, "fnd": 2, "adj": 3}, "CHIRALITY_MAP": {"left": 1, "right": 2, "na": 0},
    "GROUP_PROPS": {"TYPE": 0, "DIM": 1, "IS_COLOR": 2}, "PARTICLE_PROPS": {"TYPE": 0, "CHARGE": 1, "MASS": 2},
    "FIELD_PROPS": {"TYPE": 0, "DIM": 1, "GEN": 2, "CHIRALITY": 3},
    "INTERACTION_PROPS": {"FIELD_1": 0, "FIELD_2": 1, "FIELD_3": 2, "HIGGS_LOC_IDX": 3},
    "REP_DIM_LUT": np.array([[[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]], [[0,1,1,1], [0,0,0,0], [0,0,0,0], [0,0,0,0]], [[0,0,0,0], [0,0,0,0], [0,1,2,3], [0,1,3,8]]], dtype=np.int32),
    "SCORE_NORMALIZATION": { "PER_GAUGE_GROUP": 3, "PER_INTERACTION": 4, "PER_SCALAR_FIELD": 7, "PER_FERMION_FIELD": 10 }
}

class ModelSerializer:
    """Serializes a model dictionary into a set of NumPy arrays for GPU processing."""
    def __init__(self, model_dict: Dict[str, Any]):
        self.model, self.padding = model_dict, CONFIG['PADDING_VALUE']
        self.group_id_to_idx = {g['id']: i for i, g in enumerate(self.model.get('GaugeGroups', []))}
        self.field_id_to_idx = {f['id']: i for i, f in enumerate(self.model.get('fields', []))}
        self.field_id_to_obj = {f['id']: f for f in self.model.get('fields', [])}
        self.serialized_data = {
            "groups": np.full((CONFIG['MAX_GROUPS'], 3), self.padding, dtype=np.int32),
            "fields": np.full((CONFIG['MAX_FIELDS'], 4), self.padding, dtype=np.int32),
            "field_reps": np.full((CONFIG['MAX_FIELDS'], CONFIG['MAX_GROUPS']), self.padding, dtype=np.int32),
            "interactions": np.full((CONFIG['MAX_INTERACTIONS'], 4), self.padding, dtype=np.int32)
        }
    def serialize(self) -> Dict[str, np.ndarray]:
        self._serialize_groups(); self._serialize_fields(); self._serialize_interactions()
        return self.serialized_data

    def _validate_and_derive_yukawa(self, interaction: Dict[str, Any]) -> int:
        try:
            field_ids = interaction['fields']
            if len(field_ids) != 3: return self.padding
            f_objects = [self.field_id_to_obj.get(fid) for fid in field_ids]
            if any(f is None for f in f_objects): return self.padding
            f_left, f_right, scalar = f_objects
            if f_left.get('gen',1) != f_right.get('gen',1) or f_left.get('type') != 'fermion' or f_right.get('type') != 'fermion' or scalar.get('type') != 'complex': return self.padding
            left_p = np.array([f"p{i}" for i in range(f_left.get('dim',1) * f_left.get('gen',1))]).reshape(f_left.get('gen',1), f_left['dim']).transpose()
            right_p = np.array([f"p{i}" for i in range(f_right.get('dim',1) * f_right.get('gen',1))]).reshape(f_right.get('gen',1), f_right['dim'])
            bilinear, loc = Matrix(left_p) * Matrix(right_p), -1
            for i in range(scalar['dim']):
                if str(sympify(str(bilinear[i]).replace(" ", ""))).count("**2") == f_left.get('gen',1): loc = i; break
            return loc if loc != -1 else self.padding
        except Exception: return self.padding

    def _serialize_interactions(self):
        for i, itr in enumerate(self.model.get('interactions', [])):
            if i >= CONFIG['MAX_INTERACTIONS']: break
            field_indices = [self.field_id_to_idx.get(fid, self.padding) for fid in itr.get('fields', [])]
            while len(field_indices) < 3: field_indices.append(self.padding)
            self.serialized_data['interactions'][i, 0:3] = field_indices[:3]
            if itr.get('type') == 'yukawa':
                self.serialized_data['interactions'][i, 3] = self._validate_and_derive_yukawa(itr)

    def _serialize_groups(self):
        for i, g in enumerate(self.model.get('GaugeGroups', [])):
            if i >= CONFIG['MAX_GROUPS']: break
            g_type, g_dim = g['group'].split('_'); self.serialized_data['groups'][i] = [CONFIG['GROUP_TYPE_MAP'].get(g_type, 0), int(g_dim), 1 if g.get('name') == 'SU3C' else 0]

    def _serialize_fields(self):
        for i, f in enumerate(self.model.get('fields', [])):
            if i >= CONFIG['MAX_FIELDS']: break
            self.serialized_data['fields'][i] = [CONFIG['FIELD_TYPE_MAP'].get(f.get('type'), 0), f.get('dim', 1), f.get('gen', 1), CONFIG['CHIRALITY_MAP'].get(f.get('chirality'), 0)]
            for gid, rname in f.get('reps', {}).items():
                gidx = self.group_id_to_idx.get(gid)
                if gidx is not None and gidx < CONFIG['MAX_GROUPS']:
                    rep_val = CONFIG['REP_MAP']['singlet'] if isinstance(rname, (float,int)) else CONFIG['REP_MAP'].get(rname, 0)
                    self.serialized_data['field_reps'][i, gidx] = rep_val

class GPUValidator:
    """Performs batch validation of serialized models on the GPU."""
    def __init__(self, batch_data: Dict[str, np.ndarray]):
        self.device = DEVICE
        self.tensors = {name: torch.from_numpy(arr).to(self.device) for name, arr in batch_data.items()}
        self.rep_dim_lut = torch.from_numpy(CONFIG['REP_DIM_LUT']).to(self.device)
        self.batch_size, self.padding, self.scores = self.tensors['groups'].shape[0], CONFIG['PADDING_VALUE'], CONFIG['SCORE_NORMALIZATION']

    def validate(self) -> torch.Tensor:
        """Calculates total scores for the batch of models."""
        group_scores = self.check_gauge_groups()
        field_scores = self.check_fields()
        interaction_scores = self.check_interactions()
        total_score = torch.sum(group_scores, dim=-1) + torch.sum(field_scores, dim=-1) + torch.sum(interaction_scores, dim=-1)
        return total_score

    def check_gauge_groups(self) -> torch.Tensor:
        g, mask = self.tensors['groups'], (self.tensors['groups'][:, :, 0] != self.padding)
        t, d = g[:, :, CONFIG['GROUP_PROPS']['TYPE']], g[:, :, CONFIG['GROUP_PROPS']['DIM']]
        is_valid = ((t == 1) & (d == 1)) | ((t == 2) & ((d == 2) | (d == 3)))
        return (is_valid & mask).int() * self.scores['PER_GAUGE_GROUP']

    def check_fields(self) -> torch.Tensor:
        f, fr, g = self.tensors['fields'], self.tensors['field_reps'], self.tensors['groups']
        mask = (f[:, :, 0] != self.padding)
        field_type = f[:, :, CONFIG['FIELD_PROPS']['TYPE']]
        type_check = field_type > 0

        is_fermion_mask = (field_type == CONFIG['FIELD_TYPE_MAP']['fermion'])
        chi = f[:, :, CONFIG['FIELD_PROPS']['CHIRALITY']]
        chirality_check = ((chi == CONFIG['CHIRALITY_MAP']['left']) | (chi == CONFIG['CHIRALITY_MAP']['right']))
        chirality_ok = chirality_check | ~is_fermion_mask

        gt = torch.clamp(g[:, :, CONFIG['GROUP_PROPS']['TYPE']].unsqueeze(1), min=0, max=self.rep_dim_lut.size(0) - 1)
        gd = torch.clamp(g[:, :, CONFIG['GROUP_PROPS']['DIM']].unsqueeze(1), min=0, max=self.rep_dim_lut.size(1) - 1)
        fr_clamped = torch.clamp(fr, min=0, max=self.rep_dim_lut.size(2) - 1)

        rep_dims = self.rep_dim_lut[gt, gd, fr_clamped]
        is_na_ns = (gt == CONFIG['GROUP_TYPE_MAP']['SU']) & (fr_clamped != CONFIG['REP_MAP']['singlet'])
        relevant_rep_dims = rep_dims * is_na_ns
        field_dim_exp = f[:, :, CONFIG['FIELD_PROPS']['DIM']].unsqueeze(-1)
        dim_matches_any_rep = torch.any((field_dim_exp == relevant_rep_dims) & (relevant_rep_dims != 0), dim=-1)
        is_singlet_everywhere = torch.sum(relevant_rep_dims, dim=-1) == 0
        dim_consistency_check = is_singlet_everywhere | (f[:, :, CONFIG['FIELD_PROPS']['DIM']] == 1) | dim_matches_any_rep

        is_valid = type_check & chirality_ok & dim_consistency_check & mask
        base_score = is_valid.int() * self.scores['PER_SCALAR_FIELD']
        fermion_bonus = is_valid.int() * (self.scores['PER_FERMION_FIELD'] - self.scores['PER_SCALAR_FIELD'])
        return base_score + (fermion_bonus * is_fermion_mask)

    def check_interactions(self) -> torch.Tensor:
        inter, fields = self.tensors['interactions'], self.tensors['fields']
        mask = (inter[..., 0] != self.padding)
        if not mask.any(): return torch.zeros_like(inter[..., 0], dtype=torch.int32)

        max_f_idx = fields.size(1) - 1
        i1 = torch.clamp(inter[..., 0], min=0, max=max_f_idx)
        i2 = torch.clamp(inter[..., 1], min=0, max=max_f_idx)
        i3 = torch.clamp(inter[..., 2], min=0, max=max_f_idx)

        batch_idx = torch.arange(self.batch_size, device=self.device).view(-1, 1)
        p1, p2, p3 = fields[batch_idx, i1], fields[batch_idx, i2], fields[batch_idx, i3]

        type_ok = ((p1[..., 0] == CONFIG['FIELD_TYPE_MAP']['fermion']) & (p1[..., 3] == CONFIG['CHIRALITY_MAP']['left'])) & \
                  ((p2[..., 0] == CONFIG['FIELD_TYPE_MAP']['fermion']) & (p2[..., 3] == CONFIG['CHIRALITY_MAP']['right'])) & \
                  (p3[..., 0] == CONFIG['FIELD_TYPE_MAP']['complex'])
        gen_ok = (p1[..., 2] == p2[..., 2])
        symbolic_ok = (inter[..., 3] != self.padding)
        score = type_ok.int() + gen_ok.int() + symbolic_ok.int() * 2
        return score * mask.int()

class BatchValidator:
    """Public interface for batch validation."""
    def __init__(self):
        self.device = DEVICE
        main_logger.info(f"BatchValidator initialized. Target device: {str(self.device).upper()}")

    def validate_batch(self, batch_of_models: List[Dict[str, Any]]) -> np.ndarray:
        if not batch_of_models: return np.array([], dtype=np.int32)
        serialized_models = [ModelSerializer(m).serialize() for m in batch_of_models]
        if not serialized_models: return np.array([], dtype=np.int32)
        batch_data = {k: np.stack([d[k] for d in serialized_models]) for k in serialized_models[0].keys()}
        validator = GPUValidator(batch_data)
        scores_tensor = validator.validate()
        return scores_tensor.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Grammar and Token Definitions
# ─────────────────────────────────────────────────────────────────────────────
TOKENS: List[str] = (
    ["BOS", "EOS"]
    + ["ITRACT", "END_ITRACT"] + [f"ITRACT_ID_{i}" for i in range(10)]
    + ["FIELD", "END_FIELD"] + [f"FIELD_ID_{i}" for i in range(10)]
    + ["PARTICLE", "END_PARTICLE"] + [f"PARTICLE_ID_{i}" for i in range(20)]
    + ["TYPE_yukawa", "TYPE_d_coupling"]
    + ["TYPE_complex", "TYPE_real", "TYPE_fermion", "TYPE_vector"]
    + [f"DIM_{i}" for i in [1, 2, 3]] + [f"GEN_{i}" for i in [1, 2, 3]]
    + ["SELF_CONJ_TRUE", "SELF_CONJ_FALSE"] + ["CHIRALITY_left", "CHIRALITY_right", "CHIRALITY_na"]
    + ["REP_SU3C_singlet", "REP_SU3C_fnd", "REP_SU3C_adj"] + ["REP_SU2L_singlet", "REP_SU2L_fnd", "REP_SU2L_adj"]
    + ["REP_U1Y_minus1", "REP_U1Y_0", "REP_U1Y_1"]
)
TOKEN2ID = {t: i for i, t in enumerate(TOKENS)}; ID2TOK = {i: t for t, i in TOKEN2ID.items()}
VOCAB_SIZE, BOS_TOKEN, EOS_TOKEN = len(TOKENS), TOKEN2ID["BOS"], TOKEN2ID["EOS"]

grammar = TorchGrammar()
tr = lambda s, e: (TOKEN2ID[s], TOKEN2ID[e])
grammar.add_object("Field", [ TOKEN2ID["FIELD"], tr("FIELD_ID_0", f"FIELD_ID_{9}"), tr("TYPE_complex", "TYPE_vector"), tr("DIM_1", "DIM_3"), tr("GEN_1", "GEN_3"), tr("SELF_CONJ_TRUE", "SELF_CONJ_FALSE"), tr("CHIRALITY_left", "CHIRALITY_na"), tr("REP_SU3C_singlet", "REP_SU3C_adj"), tr("REP_SU2L_singlet", "REP_SU2L_adj"), tr("REP_U1Y_minus1", "REP_U1Y_1"), TOKEN2ID["END_FIELD"] ])
grammar.add_object("Interaction", [ TOKEN2ID["ITRACT"], tr("ITRACT_ID_0", f"ITRACT_ID_{9}"), tr("TYPE_yukawa", "TYPE_d_coupling"), {"Field": 2}, TOKEN2ID["END_ITRACT"] ])
grammar.add_object("Model", [TOKEN2ID["BOS"], {"Interaction": 1}, TOKEN2ID["EOS"]])
ROOTS = ["Model"]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Sequence Detokenizer
# ─────────────────────────────────────────────────────────────────────────────
class SequenceDetokenizer:
    """Converts a list of token strings back into a structured model dictionary."""
    def parse_token(self, token: str) -> Tuple[str, Any] | None:
        parts = token.split('_')
        prefix = parts[0]
        if prefix == "TYPE": return "type", parts[1]
        if prefix == "DIM": return "dim", int(parts[1])
        if prefix == "GEN": return "gen", int(parts[1])
        if prefix == "SELF": return "self_conjugate", parts[2].lower() == "true"
        if prefix == "CHIRALITY": return "chirality", parts[1]
        if prefix == "REP":
            group_map = {"U1Y": "g1", "SU2L": "g2", "SU3C": "g3"}
            gid = group_map.get(parts[1])
            return ("rep", (gid, parts[2])) if gid else None
        return None

    def detokenize(self, token_list: List[str]) -> Dict[str, Any]:
        model = {"GaugeGroups": [ {"id": "g1", "name": "U1Y", "group": "U_1"}, {"id": "g2", "name": "SU2L", "group": "SU_2"}, {"id": "g3", "name": "SU3C", "group": "SU_3"} ], "fields": [], "interactions": []}
        stack, obj_counter = [], {"field": 0, "interaction": 0}
        for token in token_list:
            if token in ["BOS", "EOS"]: continue
            if token == "FIELD":
                obj_id = f"f{obj_counter['field']}"; obj_counter['field'] += 1
                stack.append({"type": "field", "id": obj_id, "data": {"id": obj_id, "name": f"field_{obj_id}", "particles": [], "reps": {}}})
                continue
            if token == "ITRACT":
                obj_id = f"i{obj_counter['interaction']}"; obj_counter['interaction'] += 1
                stack.append({"type": "interaction", "id": obj_id, "data": {"id": obj_id, "name": f"interaction_{obj_id}", "fields": []}})
                continue
            if token.startswith("END_") and stack:
                obj_type_ended = token.split('_')[1].lower()
                if stack[-1]['type'] == obj_type_ended:
                    completed_obj = stack.pop()
                    if not stack:
                        model[f"{obj_type_ended}s"].append(completed_obj['data'])
                    elif stack[-1]['type'] == 'interaction' and completed_obj['type'] == 'field':
                         model['fields'].append(completed_obj['data'])
                         stack[-1]['data']['fields'].append(completed_obj['id'])
                continue
            if not stack: continue
            parsed = self.parse_token(token)
            if parsed:
                key, value = parsed
                if key == "rep":
                    gid, rep_val = value
                    stack[-1]['data']['reps'][gid] = rep_val
                else: stack[-1]['data'][key] = value
        return model


# ─────────────────────────────────────────────────────────────────────────────
# 6. Fast Grammar Parser Wrapper
# ─────────────────────────────────────────────────────────────────────────────
class FastGrammarParser:
    """A user-friendly wrapper for the TorchGrammarBatchParser."""
    def __init__(self, grammar: TorchGrammar, batch_size: int, device: torch.device):
        self.parser = TorchGrammarBatchParser(grammar, batch_size, device=device, max_len=HParams.seq_len)

    def start_batch(self, roots: List[str]):
        self.parser.set_batch(roots)

    def get_mask(self) -> torch.Tensor:
        """Returns a mask of legal tokens for the current state. Shape: [B, V]."""
        return self.parser.next_token_mask(VOCAB_SIZE)

    def update(self, actions: torch.Tensor):
        """Updates the parser state with the chosen actions."""
        self.parser.fast_is_valid(actions)

    @property
    def is_finished(self) -> torch.Tensor:
        """Returns a boolean tensor indicating which sequences in the batch are complete."""
        return self.parser.next_lens == 0


# ─────────────────────────────────────────────────────────────────────────────
# 7. Transformer (Actor-Critic) & Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
class TransformerPolicy(nn.Module):
    """An actor-critic Transformer model."""
    def __init__(self, vocab: int, *, d_model: int = 256, nhead: int = 4, layers: int = 4, max_T: int = 512):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_T, d_model)
        self.register_buffer("pos", torch.arange(max_T))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, 0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.actor = nn.Linear(d_model, vocab)
        self.value = nn.Linear(d_model, 1)

    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T = seq.size(1)
        x = self.tok_emb(seq) + self.pos_emb(self.pos[:T])
        # The causal mask ensures that attention for position i can only attend to positions <= i.
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=seq.device)
        h = self.encoder(x, mask=causal_mask)
        return self.actor(h), self.value(h).squeeze(-1)

@dataclass(frozen=True)
class HParams:
    epochs: int = 500
    batch_size: int = 512
    seq_len: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    gae_lambda: float = 0.95
    pg_coef: float = 0.01    # Weight for the policy gradient loss
    value_coef: float = 0.1  # Weight for the value loss
    sup_coef: float = 1.0    # Weight for the supervised grammar loss
    ent_coef: float = 0.5    # Weight for the entropy bonus
    grad_clip_norm: float = 1.0
    ckpt_every: int = 50
    buffer_size: int = 2048
    min_buffer_fill: int = 512 # Minimum trajectories before training starts
    rank_k: float = 0.1 # Parameter for rank-based reward shaping in sampling
    enable_logging: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# 8. Experience Buffer
# ─────────────────────────────────────────────────────────────────────────────
class PPOTrajectory(NamedTuple):
    """Stores data for a single trajectory."""
    seq: torch.Tensor; rewards: torch.Tensor; logps: torch.Tensor; values: torch.Tensor; ep_len: int; total_reward: float

class ExperienceBuffer:
    """A prioritized replay buffer for PPO that stores and samples trajectories."""
    def __init__(self, hp: HParams):
        self.buffer: List[PPOTrajectory] = []
        self.hp = hp
        self._unique_seqs = set()

    def __len__(self) -> int:
        return len(self.buffer)

    def add_batch(self, generated_seqs: torch.Tensor, batch_data: Dict[str, torch.Tensor]):
        rews, logps, values = batch_data['rewards'], batch_data['logps'], batch_data['values']
        for i in range(rews.size(0)):
            ep_len = batch_data['ep_len'][i].item()
            if ep_len == 0: continue

            seq = generated_seqs[i, :ep_len + 1].cpu()
            seq_tuple = tuple(seq.tolist())
            if seq_tuple in self._unique_seqs: continue

            ep_rews = rews[i, :ep_len]
            total_reward = ep_rews.sum().item() # Using undiscounted reward for sorting

            traj = PPOTrajectory(
                seq=seq,
                rewards=ep_rews.cpu(),
                logps=logps[i, :ep_len].cpu(),
                values=values[i, :ep_len].cpu(),
                ep_len=ep_len,
                total_reward=total_reward
            )
            self.buffer.append(traj)
            self._unique_seqs.add(seq_tuple)

        # Prune buffer if it exceeds capacity, keeping the best trajectories
        if len(self.buffer) > self.hp.buffer_size:
            self.buffer.sort(key=lambda x: x.total_reward, reverse=True)
            removed_trajs = self.buffer[self.hp.buffer_size:]
            self.buffer = self.buffer[:self.hp.buffer_size]
            for traj in removed_trajs:
                self._unique_seqs.remove(tuple(traj.seq.tolist()))

    def sample(self) -> Dict[str, torch.Tensor]:
        if not self.buffer: raise RuntimeError("Cannot sample from empty buffer")

        n_samples = min(self.hp.batch_size, len(self.buffer))
        buffer_rewards = torch.tensor([t.total_reward for t in self.buffer])
        ranks = buffer_rewards.argsort(descending=True).argsort().float() + 1.0
        weights = (self.hp.rank_k * len(self.buffer) + ranks).pow(-1)
        indices = torch.multinomial(weights, n_samples, replacement=True)

        samples = [self.buffer[i] for i in indices]
        max_len = max(s.ep_len for s in samples)

        collated_seqs = torch.full((n_samples, max_len + 1), BOS_TOKEN, dtype=torch.long)
        collated_rewards = torch.zeros((n_samples, max_len))
        collated_logps = torch.zeros((n_samples, max_len))
        collated_values = torch.zeros((n_samples, max_len))
        collated_ep_len = torch.tensor([s.ep_len for s in samples], dtype=torch.long)

        for i, s in enumerate(samples):
            collated_seqs[i, :s.ep_len + 1] = s.seq
            collated_rewards[i, :s.ep_len] = s.rewards
            collated_logps[i, :s.ep_len] = s.logps
            collated_values[i, :s.ep_len] = s.values

        advantages = self._compute_gae(collated_rewards, collated_values, collated_ep_len, max_len)

        return {
            'seq': collated_seqs.to(DEVICE),
            'logps': collated_logps.to(DEVICE),
            'values': collated_values.to(DEVICE),
            'returns': (advantages + collated_values).to(DEVICE),
            'advantages': advantages.to(DEVICE),
            'ep_len': collated_ep_len.to(DEVICE)
        }

    def _compute_gae(self, rewards, values, ep_lens, max_len):
        """Computes Generalized Advantage Estimation (GAE)."""
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        for t in reversed(range(max_len)):
            is_alive = (t < ep_lens).float()
            next_values = values[:, t + 1] if t < max_len - 1 else torch.zeros_like(values[:, t])
            delta = rewards[:, t] + self.hp.gamma * next_values * is_alive - values[:, t]
            advantages[:, t] = last_gae_lam = delta + self.hp.gamma * self.hp.gae_lambda * last_gae_lam * is_alive
        return advantages


# ─────────────────────────────────────────────────────────────────────────────
# 9. PPO Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def train(model: nn.Module, opt: torch.optim.Optimizer, hp: HParams) -> Path:
    run_id  = datetime.now().strftime("phy_ppo_%Y%m%d_%H%M%S")
    logdir  = Path("./runs") / run_id
    ckptdir = logdir / "checkpoints"; ckptdir.mkdir(parents=True, exist_ok=True)

    logger: Optional[Logger] = None
    if hp.enable_logging:
        logger = Logger(logdir, cli_interval=1, weights_interval=hp.ckpt_every * 10)
        logger.add_metric("reward", kind="stat"); logger.add_metric("seq_len", kind="stat")
        logger.add_metric("pg_loss", kind="stat"); logger.add_metric("v_loss", kind="stat")
        logger.add_metric("e_loss", kind="stat"); logger.add_metric("sup_loss", kind="stat")
        logger.add_metric("total_loss", kind="stat")
        logger.add_metric("buffer_size", kind="scalar"); logger.add_metric("learning_rate", kind="scalar")
        logger.model = model

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, hp.epochs)
    experience_buffer = ExperienceBuffer(hp)
    batch_validator = BatchValidator()
    detokenizer = SequenceDetokenizer()

    for ep in range(1, hp.epochs + 1):
        # --- Rollout Generation ---
        model.eval()
        parser = FastGrammarParser(grammar, hp.batch_size, device=DEVICE)
        parser.start_batch(random.choices(ROOTS, k=hp.batch_size))

        seq = torch.full((hp.batch_size, hp.seq_len + 1), BOS_TOKEN, dtype=torch.long, device=DEVICE)
        ep_len = torch.zeros(hp.batch_size, dtype=torch.long, device=DEVICE)
        alive = torch.ones(hp.batch_size, dtype=torch.bool, device=DEVICE)
        rewards_h, logps_h, values_h = [], [], []

        for t in range(hp.seq_len):
            with torch.no_grad():
                logits_t, values_t = model(seq[:, :t+1])
            logits, mask = logits_t[:, -1].float(), parser.get_mask()
            eligible = alive & ~parser.is_finished
            if not eligible.any(): break

            probs = F.softmax(logits + mask, dim=-1).nan_to_num_(0.0)
            probs[~eligible] = 0.0 # Zero out probabilities for finished sequences
            probs[~eligible, EOS_TOKEN] = 1.0 # Ensure finished sequences sample EOS

            dist = Categorical(probs=probs)
            acts = dist.sample()
            parser.update(acts)
            seq[:, t + 1] = acts

            rewards_h.append(torch.zeros(hp.batch_size, device=DEVICE))
            logps_h.append(dist.log_prob(acts))
            values_h.append(values_t[:, -1])
            ep_len[alive] += 1
            alive &= (acts != EOS_TOKEN)

        # --- Terminal Reward Calculation ---
        final_rewards = torch.zeros(hp.batch_size, device=DEVICE)
        completed_indices = [i for i, length in enumerate(ep_len) if length > 0]
        if completed_indices:
            models_to_score = [detokenizer.detokenize([ID2TOK[tid] for tid in seq[i, 1:ep_len[i]+1].tolist()]) for i in completed_indices]
            scores = batch_validator.validate_batch(models_to_score)
            final_rewards[torch.tensor(completed_indices)] = torch.tensor(scores, device=DEVICE, dtype=torch.float32)

        # --- Update Experience Buffer ---
        if rewards_h:
            rewards_tensor = torch.stack(rewards_h, dim=1)
            for i in range(hp.batch_size):
                if ep_len[i] > 0: rewards_tensor[i, ep_len[i] - 1] += final_rewards[i]
            batch_dict = {'rewards': rewards_tensor, 'logps': torch.stack(logps_h, dim=1), 'values': torch.stack(values_h, dim=1), 'ep_len': ep_len}
            experience_buffer.add_batch(seq, batch_dict)
            if logger: logger.log({"reward": final_rewards.mean().item(), "seq_len": ep_len.float().mean().item(), "buffer_size": len(experience_buffer)})

        # --- Warm-up Check ---
        if len(experience_buffer) < hp.min_buffer_fill:
            if ep % 10 == 0: main_logger.info(f"Epoch [{ep:4d}/{hp.epochs}] | Warming up buffer ({len(experience_buffer)}/{hp.min_buffer_fill})...")
            sched.step()
            continue

        # --- PPO Update Phase ---
        model.train()
        ppo_stats = {"pg": 0.0, "v": 0.0, "e": 0.0, "sup": 0.0}
        for _ in range(hp.ppo_epochs):
            tbatch = experience_buffer.sample()
            N, T = tbatch['seq'].size(0), tbatch['seq'].size(1) - 1
            valid_mask = torch.arange(T, device=DEVICE)[None, :] < tbatch['ep_len'][:, None]
            if not valid_mask.any(): continue

            logits, vals = model(tbatch['seq'][:, :-1])
            dist = Categorical(logits=logits)
            new_lp, ents = dist.log_prob(tbatch['seq'][:, 1:]), dist.entropy()

            # PPO losses
            adv = tbatch['advantages']
            adv = (adv - adv[valid_mask].mean()) / (adv[valid_mask].std() + EPS)
            ratio = (new_lp - tbatch['logps']).exp()
            s1, s2 = ratio * adv, torch.clamp(ratio, 1 - hp.clip_epsilon, 1 + hp.clip_epsilon) * adv
            pg_loss = -torch.min(s1, s2)[valid_mask].mean()
            v_loss = F.mse_loss(vals[valid_mask], tbatch['returns'][valid_mask])
            e_loss = -ents[valid_mask].mean()

            # Supervised grammar penalty on illegal logits
            parser_sup = FastGrammarParser(grammar, N, device=DEVICE); parser_sup.start_batch(["Model"] * N)
            sup_loss_sum, sup_loss_count = 0.0, 0
            for t in range(T):
                legal_mask = (parser_sup.get_mask() == 0.0) # [N, V]
                illegal_mask = ~legal_mask
                step_logits = logits[:, t, :]
                # Penalize logits assigned to illegal tokens
                illegal_logits = step_logits.masked_select(illegal_mask & valid_mask[:, t].unsqueeze(-1))
                if illegal_logits.numel() > 0:
                    sup_loss_sum += illegal_logits.pow(2).sum()
                    sup_loss_count += illegal_logits.numel()
                parser_sup.update(tbatch['seq'][:, t+1])
            sup_loss = sup_loss_sum / max(1, sup_loss_count) if sup_loss_count > 0 else 0.0

            loss = hp.pg_coef * pg_loss + hp.value_coef * v_loss + hp.ent_coef * e_loss + hp.sup_coef * sup_loss
            opt.zero_grad(set_to_none=True)
            SCALER.scale(loss).backward()
            SCALER.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip_norm)
            SCALER.step(opt); SCALER.update()

            ppo_stats['pg'] += pg_loss.item(); ppo_stats['v'] += v_loss.item(); ppo_stats['e'] += e_loss.item(); ppo_stats['sup'] += sup_loss.item() if isinstance(sup_loss, torch.Tensor) else sup_loss

        sched.step()

        # --- Logging ---
        if logger:
            avg_stats = {f"{k}_loss": v / hp.ppo_epochs for k, v in ppo_stats.items()}
            logger.log(avg_stats)
            logger.log({"learning_rate": sched.get_last_lr()[0]})
            logger.cli_step()

        print(f"Epoch [{ep:4d}/{hp.epochs}] | Reward: {logger.stats['reward'].mean:6.2f} | Len: {logger.stats['seq_len'].mean:4.1f} | "
              f"PGL: {ppo_stats['pg']/hp.ppo_epochs:.2f} | VL: {ppo_stats['v']/hp.ppo_epochs:.2f} | EL: {ppo_stats['e']/hp.ppo_epochs:.2f} | SUP: {ppo_stats['sup']/hp.ppo_epochs:.2f} | "
              f"LR: {sched.get_last_lr()[0]:.1e}")

        if ep % hp.ckpt_every == 0 or ep == hp.epochs:
            ckpt_path = ckptdir / f"ep{ep}.pt"
            torch.save(getattr(model, "_orig_mod", model).state_dict(), ckpt_path)
            main_logger.info(f"✓ Checkpoint saved to {ckpt_path}")

    main_logger.info("Training Complete.")
    return ckptdir / f"ep{hp.epochs}.pt"


# ─────────────────────────────────────────────────────────────────────────────
# 10. Sampling and Main Execution
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def sample(model_to_sample: nn.Module, T: int, *, n: int = 4, temp: float = 1.0) -> None:
    """Generates and prints sample sequences from a trained model."""
    model_to_sample.eval()
    for i in range(n):
        parser = FastGrammarParser(grammar, 1, device=DEVICE)
        parser.start_batch(random.choices(ROOTS, k=1))
        seq_ids = [BOS_TOKEN]
        print(f"\n--- Sample {i+1} (temp={temp:.2f}) ---")
        for _ in range(T):
            logits_all, _ = model_to_sample(torch.tensor([seq_ids], device=DEVICE))
            logits, mask = logits_all[:, -1].squeeze(0), parser.get_mask()[0]
            # Apply temperature to logits for sampling diversity
            if temp > 0: logits = logits / temp
            probs = F.softmax(logits + mask, dim=0).nan_to_num_(0)
            if probs.sum() < EPS: break
            tok_id = int(Categorical(probs).sample().item()); print(f" {ID2TOK[tok_id]}", end="")
            seq_ids.append(tok_id); parser.update(torch.tensor([tok_id], device=DEVICE))
            if tok_id == EOS_TOKEN or parser.is_finished[0]: break
        print("\n" + "-"*25)
    model_to_sample.train()

if __name__ == "__main__":
    hp = HParams()
    main_logger.info(f"Device: {DEVICE}, AMP: {'ON' if DEVICE.type=='cuda' else 'OFF'}, Logging: {'ON' if hp.enable_logging else 'OFF'}, Compiling: {CAN_COMPILE}")
    policy_model = TransformerPolicy(VOCAB_SIZE, max_T=hp.seq_len + 1).to(DEVICE)
    if CAN_COMPILE:
        main_logger.info("PyTorch 2.0+ detected. Compiling model for a significant speed-up...")
        policy_model = torch.compile(policy_model)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=hp.lr)
    final_checkpoint = train(policy_model, optimizer, hp)
    main_logger.info("Training complete. Final model saved to %s", final_checkpoint)

    # --- Load final model and generate samples ---
    main_logger.info("Loading final model for sampling...")
    model_for_sampling = TransformerPolicy(VOCAB_SIZE, max_T=hp.seq_len + 1).to(DEVICE)
    model_for_sampling.load_state_dict(torch.load(final_checkpoint, map_location=DEVICE))
    if CAN_COMPILE: model_for_sampling = torch.compile(model_for_sampling)
    
    # Sample with no temperature (greedy decoding) to see the 'best' learned sequences
    sample(model_for_sampling, hp.seq_len, n=5, temp=0.0)
    # Sample with some temperature to see diversity
    sample(model_for_sampling, hp.seq_len, n=5, temp=0.8)

    print("\nDone.")
