"""
TODO:
- Top Goal : Use the prior code as an example, output a file which can be used to train a transformer via RL to learn to generate sequences within this grammar which maximize reward
- Use popular highly optimized libraries for RL (especially PPO) and Transformer (especially libraries for RL'ing Transformers)
- Keep all of the logging in-line 
- Keep all of the environment logic in the same file as the transformer & training

Think about what would be needed to learn the sparse reward ecosystems of this problem.
- Where can SFT be used since we know the tokens for the grammatically incorrect tokens should be zero
- How can more be extracted from the highest reward examples? Or the most surprising examples.
- Given the hierarchical objects-in-objects structure, how can we make the learning even faster?

Make sure that as many operations as possible are done with tensor ops and bitwise ops rather than for loops or if statements to maximize parallelization.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. Imports and Global Setup
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import csv, math, sys, os, random, time, queue, warnings, contextlib, logging, threading, multiprocessing
from pathlib import Path
from dataclasses import dataclass, replace
from collections import deque
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple, Union, Iterable, Optional, NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.distributions import Categorical
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity
import json
import numpy as np
from sympy import Matrix, sympify
from fractions import Fraction

# Global setup
warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True")
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
main_logger = logging.getLogger(__name__)

# Type Aliases (from classes.py)
Token     = Union[int, Tuple[int, ...], str, Dict[str, Any]]
Predicate = Callable[[int], bool]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Helper Classes (inlined from classes.py)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Trajectory:
    states: torch.Tensor # (T_i + 1,) token IDs incl. BOS
    actions: torch.Tensor # (T_i,) token IDs (step-wise)
    ret: float # scalar episode return

class SILBuffer:
    """Top-K buffer for Self-Imitation Learning (SIL)."""
    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self._data: deque[Trajectory] = deque(maxlen=capacity)
    def add(self, trajs: List[Trajectory]) -> None:
        self._data.extend(trajs)
        self._data = deque(sorted(self._data, key=lambda x: x.ret, reverse=True)[:self.capacity], maxlen=self.capacity)
    def sample(self, batch: int) -> List[Trajectory]:
        idx = torch.randint(0, len(self._data), (batch,))
        return [self._data[i] for i in idx]

class RunningStat:
    """Exponential moving average (EMA) style running mean / variance tracker."""
    def __init__(self, alpha: float = 0.1) -> None:
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
        self._mean += self._alpha * delta
        self._var = (1 - self._alpha) * self._var + self._alpha * delta * (x - self._mean)

class Logger:
    """Generic experiment logger."""
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
        self.add_metric("fps", kind="stat")
    def __del__(self):
        if hasattr(self, '_cli_csv'): self._cli_csv.close()
        if hasattr(self, '_state_csv'): self._state_csv.close()
        if hasattr(self, 'tb'): self.tb.close()
    def add_metric(self, name: str, *, kind: str = "stat"):
        if kind not in {"stat", "scalar"}: raise ValueError("kind must be 'stat' or 'scalar'")
        if name in self.stats or name in self.scalars: raise ValueError(f"Metric '{name}' already registered")
        if kind == "stat": self.stats[name] = RunningStat()
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
        row = [self.timestep, self.seconds, self.stats["fps"].mean, self.stats["fps"].std]
        for name in sorted(self.scalars): row.append(self.scalars[name])
        for name in sorted(self.stats):
            if name == "fps": continue
            st = self.stats[name]; row.extend([st.most_recent, st.mean, st.std])
        self.cli_writer.writerow(row); self._cli_csv.flush()
        for name, st in self.stats.items():
            self.tb.add_scalar(f"train/{name}_mean", st.mean, self.timestep); self.tb.add_scalar(f"train/{name}_std", st.std, self.timestep); self.tb.add_scalar(f"train/{name}_mr", st.most_recent, self.timestep)
        for name, val in self.scalars.items(): self.tb.add_scalar(f"train/{name}", val, self.timestep)
        self.tb.flush()
    def _write_state_header(self): self.state_writer.writerow(["timestep", "seconds", "state", "reward_most_recent", "action", "logits"])
    def _write_cli_header(self):
        hdr = ["timestep", "seconds", "fps_mean", "fps_std"]
        hdr.extend(sorted(self.scalars))
        for name in sorted(self.stats):
            if name != "fps": hdr.extend([f"{name}_most_recent", f"{name}_mean", f"{name}_std"])
        self.cli_writer.writerow(hdr)

class TorchGrammar:
    """*Immutable* container mapping *symbol → rule‑list*."""
    def __init__(self) -> None:
        self.raw_rules: Dict[str, List[Token]] = {}
        self.symbol_to_index: Dict[str, int] = {}
        self.index_to_symbol: Dict[int, str] = {}
    def add_object(self, name: str, tokens: List[Token]) -> None:
        """Register **one** grammar object."""
        self.raw_rules[name] = tokens
        if name not in self.symbol_to_index:
            idx = len(self.symbol_to_index)
            self.symbol_to_index[name] = idx
            self.index_to_symbol[idx] = name
    @staticmethod
    def expand_literal(tok: Token) -> List[int]:
        """Expand a *literal spec* → concrete list of **token IDs**."""
        if isinstance(tok, int): return [tok]
        if isinstance(tok, tuple): return list(range(tok[0], tok[1] + 1)) if len(tok) == 2 else list(tok)
        raise ValueError()
    @staticmethod
    def wrap_pred(raw: Any) -> Predicate:
        """Return predicate *raw* unchanged if callable, else ``cnt >= raw``."""
        return raw if callable(raw) else (lambda cnt, m=raw: cnt >= m)

class TorchGrammarBatchParser:
    DEFAULT_MAX_DEPTH, DEFAULT_MAX_LEN, DEFAULT_MAX_NEXT = 16, 128, 32
    def __init__(self, grammar: TorchGrammar, batch_size: int, *, device: str | torch.device = "cuda", max_depth: int | None = None, max_len: int | None = None, max_next: int | None = None):
        self.g, self.bs, self.dev = grammar, batch_size, torch.device(device)
        self.max_depth, self.max_len, self.max_next = max_depth or self.DEFAULT_MAX_DEPTH, max_len or self.DEFAULT_MAX_LEN, max_next or self.DEFAULT_MAX_NEXT
        self.stack = torch.full((self.bs, self.max_depth, 3), -1, dtype=torch.long, device=self.dev)
        self.ptr = torch.zeros(self.bs, dtype=torch.long, device=self.dev)
        self.seqs = torch.full((self.bs, self.max_len), -1, dtype=torch.long, device=self.dev)
        self.lens = torch.zeros(self.bs, dtype=torch.long, device=self.dev)
        self.next_cache = torch.full((self.bs, self.max_next), -1, dtype=torch.long, device=self.dev)
        self.next_lens = torch.zeros(self.bs, dtype=torch.long, device=self.dev)
    def set_batch(self, roots: List[str]) -> None:
        for i, name in enumerate(roots):
            idx = self.g.symbol_to_index[name]
            self.stack[i, 0] = torch.tensor([idx, 0, 0], device=self.dev)
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
                pos = self.lens[i].item()
                if pos < self.max_len: self.seqs[i, pos] = tok; self.lens[i] += 1
                o_idx, o_ptr, o_rep = self.stack[i, self.ptr[i] - 1].tolist()
                obj_name = self.g.index_to_symbol[o_idx]
                curr_tok = self.g.raw_rules[obj_name][o_ptr]
                self._apply_token(i, curr_tok, tok, o_rep)
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
            if not pred(rep_cnt): return self.g.expand_literal(self.g.raw_rules[sub][0])
            valid: List[int] = []
            if idx + 1 < len(tokens):
                nxt = tokens[idx + 1]; nxt_lit = nxt if not isinstance(nxt, str) else self.g.raw_rules[nxt][0]
                valid.extend(self.g.expand_literal(nxt_lit))
            valid.extend(self.g.expand_literal(self.g.raw_rules[sub][0]))
            return valid
        return self.g.expand_literal(self.g.raw_rules[tok][0] if isinstance(tok, str) else tok)
    def _apply_token(self, i: int, curr: Token, tok: int, rep_cnt: int) -> None:
        if isinstance(curr, dict):
            sub, _ = next(iter(curr.items()))
            if tok not in set(self.g.expand_literal(self.g.raw_rules[sub][0])): self._pop_and_advance(i); return
            depth = self.ptr[i].item(); sid = self.g.symbol_to_index[sub]
            self.stack[i, depth] = torch.tensor([sid, 1, 0], device=self.dev); self.ptr[i] += 1
            self.stack[i, depth - 1, 2] += 1
        elif isinstance(curr, str):
            depth, sid = self.ptr[i].item(), self.g.symbol_to_index[curr]
            self.stack[i, depth] = torch.tensor([sid, 1, 0], device=self.dev); self.ptr[i] += 1
        else: self.stack[i, self.ptr[i] - 1, 1] += 1; self.stack[i, self.ptr[i] - 1, 2] = 0
    def _pop_and_advance(self, i: int) -> None:
        self.ptr[i] -= 1
        if self.ptr[i] > 0:
            o_idx, o_ptr, _ = self.stack[i, self.ptr[i] - 1].tolist()
            token = self.g.raw_rules[self.g.index_to_symbol[o_idx]][o_ptr]
            if not isinstance(token, dict):
                self.stack[i, self.ptr[i] - 1, 1] = o_ptr + 1; self.stack[i, self.ptr[i] - 1, 2] = 0
    def _refresh_next(self) -> None:
        self.next_lens.zero_()
        for i in range(self.bs):
            while self.ptr[i] > 0:
                o_idx, o_ptr, o_rep = self.stack[i, self.ptr[i] - 1].tolist()
                tokens = self.g.raw_rules[self.g.index_to_symbol[o_idx]]
                if o_ptr >= len(tokens): self._pop_and_advance(i); continue
                curr = tokens[o_ptr]; valid = self._compute_valid(curr, o_rep, tokens, o_ptr)
                if valid:
                    k = len(valid)
                    self.next_cache[i, :k] = torch.tensor(valid, device=self.dev); self.next_lens[i] = k; break
                self.ptr[i] -= 1

class ModelTester:
    MASS_MAP = {"MASS_1e0": 1, "MASS_1e1": 10, "MASS_1e2": 100, "MASS_1e3": 1000}
    CHARGE_MAP = {"CHARGE_-1": -1, "CHARGE_0": 0, "CHARGE_1": 1}
    REP_U1Y_MAP = {"REP_U1Y_minus1": -1, "REP_U1Y_0": 0, "REP_U1Y_1": 1}
    @staticmethod
    def _after(tok: str, prefix: str) -> str: return tok.replace(f"{prefix}_", "")
    @staticmethod
    def _int(tok: str) -> int: return int(tok.split("_")[-1])
    def parse_particle(self, seq: List[str], i: int) -> Tuple[Dict[str, Any], int]:
        particle = {"id": f"p{self._int(seq[i + 1]) + 1}", "name": seq[i + 1], "type": self._after(seq[i + 2], "TYPE"), "mass": self.MASS_MAP.get(seq[i + 3], 0), "charge": self.CHARGE_MAP.get(seq[i + 4], 0)}
        i += 5
        while i < len(seq) and seq[i] != "END_PARTICLE": i += 1
        return particle, i + 1
    def parse_field(self, seq: List[str], i: int) -> Tuple[Dict[str, Any], int, List[Dict[str, Any]]]:
        field = {"id": f"m{self._int(seq[i + 1]) + 1}", "name": seq[i + 1], "type": self._after(seq[i + 2], "TYPE"), "dim": self._int(seq[i + 3]), "gen": self._int(seq[i + 4]), "self_conjugate": seq[i + 5] == "SELF_CONJ_TRUE", "chirality": None if seq[i + 6] == "CHIRALITY_na" else self._after(seq[i + 6], "CHIRALITY"), "reps": {"g1": self.REP_U1Y_MAP.get(seq[i + 9], 0), "g2": self._after(seq[i + 8], "REP_SU2L"), "g3": self._after(seq[i + 7], "REP_SU3C")}, "QuantumNumber": {"LeptonNumber": self._int(seq[i + 10]), "BaryonNumber": self._int(seq[i + 11])}, "particles": []}
        i += 12; particles_in_field = []
        while i < len(seq) and seq[i] != "END_FIELD":
            if seq[i] == "PARTICLE":
                p, i_after_p = self.parse_particle(seq, i); particles_in_field.append(p); field["particles"].append(p["id"]); i = i_after_p
            else: i += 1
        return field, i + 1, particles_in_field
    def parse_interaction(self, seq: List[str], i: int) -> Tuple[Dict[str, Any], int, List[Dict[str, Any]], List[Dict[str, Any]]]:
        inter = {"id": f"i{self._int(seq[i + 1]) + 1}", "type": self._after(seq[i + 2], "TYPE").lower(), "fields": []}
        i += 3; fields_in_inter, particles_in_inter = [], []
        while i < len(seq) and seq[i] != "END_ITRACT":
            if seq[i] == "FIELD":
                f, i_after_f, ps = self.parse_field(seq, i); fields_in_inter.append(f); particles_in_inter.extend(ps); inter["fields"].append(f["id"]); i = i_after_f
            else: i += 1
        return inter, i + 1, fields_in_inter, particles_in_inter
    def build_model(self, tokens: List[str]) -> Dict[str, Any]:
        particles, fields, inters, i = [], [], [], 0
        while i < len(tokens):
            if tokens[i] == "PARTICLE": p, i = self.parse_particle(tokens, i); particles.append(p)
            elif tokens[i] == "FIELD": f, i, new_ps = self.parse_field(tokens, i); fields.append(f); particles.extend(new_ps)
            elif tokens[i] == "ITRACT": it, i, new_fs, new_ps = self.parse_interaction(tokens, i); inters.append(it); fields.extend(new_fs); particles.extend(new_ps)
            else: i += 1
        particles, fields, inters = list({p["id"]: p for p in particles}.values()), list({f["id"]: f for f in fields}.values()), list({it["id"]: it for it in inters}.values())
        return {"GaugeGroups": [{"id": "g1", "name": "U1Y", "charge": "Y", "group": "U_1", "coupling": "g1", "boson": "B"}, {"id": "g2", "name": "SU2L", "charge": "I", "group": "SU_2", "coupling": "gw", "boson": "W"}, {"id": "g3", "name": "SU3C", "charge": "C", "group": "SU_3", "coupling": "gs", "boson": "G"}], "vevs": [{"id": "v1", "name": "vev", "vacuum": [0, 1], "value": 246.22}], "particles": particles, "fields": fields, "interactions": inters, "links": {"vevs": [{"source": "v1", "target": fields[0]["id"] if fields else ""}]}}
    def test_model(self, tokens: List[str]) -> Tuple[float, str]:
        try: model_json_data = self.build_model(tokens)
        except Exception as e: main_logger.debug(f"Parsing error in build_model for sequence. Error: {e}"); return 0.0, "<parsing_error>"
        try: from ._model import Model
        except ImportError: main_logger.warning("Scoring module (.model) not found. Returning 0.0 score."); return 0.0, "<json2fr_import_error>"
        except Exception as e: main_logger.error(f"An unexpected error during '.model' import: {e}"); return 0.0, "<model_import_general_error>"
        try:
            mdl = Model(model_name="Generated Model", author=Path(__file__).name, model_data_dict=model_json_data)
            return mdl.get_score_fraction(), "<in-memory-model>"
        except Exception as e: main_logger.error(f"Error creating Model instance or getting score: {e}"); return 0.0, f"<error_scoring: {str(e)[:50]}>"


# ─────────────────────────────────────────────────────────────────────────────
# 3. GPU Scorer and Validator (from scorer_gpu.py)
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "MAX_GROUPS": 5, "MAX_PARTICLES": 20, "MAX_FIELDS": 10, "MAX_PARTICLES_PER_FIELD": 10,
    "MAX_INTERACTIONS": 5, "PADDING_VALUE": -1, "GROUP_TYPE_MAP": {"U": 1, "SU": 2},
    "FIELD_TYPE_MAP": {"fermion": 1, "real": 2, "complex": 3, "vector": 4},
    "REP_MAP": {"singlet": 1, "fnd": 2, "adj": 3}, "CHIRALITY_MAP": {"left": 1, "right": 2, "none": 0},
    "GROUP_PROPS": {"TYPE": 0, "DIM": 1, "IS_COLOR": 2}, "PARTICLE_PROPS": {"TYPE": 0, "CHARGE": 1, "MASS": 2},
    "FIELD_PROPS": {"TYPE": 0, "DIM": 1, "GEN": 2, "CHIRALITY": 3},
    "INTERACTION_PROPS": {"FIELD_1": 0, "FIELD_2": 1, "FIELD_3": 2, "HIGGS_LOC_IDX": 3},
    "REP_DIM_LUT": np.array([[[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]], [[0,1,1,1], [0,0,0,0], [0,0,0,0], [0,0,0,0]], [[0,0,0,0], [0,0,0,0], [0,1,2,3], [0,1,3,8]]], dtype=np.int32),
    "SCORE_NORMALIZATION": { "PER_GAUGE_GROUP": 3, "PER_INTERACTION": 4, "PER_SCALAR_FIELD": 7, "PER_FERMION_FIELD": 10 }
}

class ModelSerializer:
    def __init__(self, model_dict: Dict[str, Any]):
        self.model, self.padding = model_dict, CONFIG['PADDING_VALUE']
        self.group_id_to_idx = {g['id']: i for i, g in enumerate(self.model.get('GaugeGroups', []))}
        self.particle_id_to_idx = {p['id']: i for i, p in enumerate(self.model.get('particles', []))}
        self.field_id_to_idx = {f['id']: i for i, f in enumerate(self.model.get('fields', []))}
        self.field_id_to_obj = {f['id']: f for f in self.model.get('fields', [])}
        self.serialized_data = { "groups": np.full((CONFIG['MAX_GROUPS'], 3), self.padding, dtype=np.int32), "particles": np.full((CONFIG['MAX_PARTICLES'], 3), self.padding, dtype=np.float32), "fields": np.full((CONFIG['MAX_FIELDS'], 4), self.padding, dtype=np.int32), "field_reps": np.full((CONFIG['MAX_FIELDS'], CONFIG['MAX_GROUPS']), self.padding, dtype=np.int32), "field_to_particles": np.full((CONFIG['MAX_FIELDS'], CONFIG['MAX_PARTICLES_PER_FIELD']), self.padding, dtype=np.int32), "interactions": np.full((CONFIG['MAX_INTERACTIONS'], 4), self.padding, dtype=np.int32) }
    def serialize(self) -> Dict[str, np.ndarray]:
        self._serialize_groups(); self._serialize_particles(); self._serialize_fields(); self._serialize_interactions()
        return self.serialized_data
    def _validate_and_derive_yukawa(self, interaction: Dict[str, Any]) -> int:
        try:
            field_ids = interaction['fields']
            if len(field_ids) != 3: return self.padding
            f_left, f_right, scalar = self.field_id_to_obj[field_ids[0]], self.field_id_to_obj[field_ids[1]], self.field_id_to_obj[field_ids[2]]
            if f_left.get('gen',1) != f_right.get('gen',1): return self.padding
            left_p, right_p = np.array(f_left['particles']).reshape(f_left.get('gen',1), f_left['dim']).transpose(), np.array(f_right['particles']).reshape(f_right.get('gen',1), f_right['dim'])
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
            self.serialized_data['interactions'][i, 0], self.serialized_data['interactions'][i, 1], self.serialized_data['interactions'][i, 2] = field_indices[:3]
            self.serialized_data['interactions'][i, 3] = self._validate_and_derive_yukawa(itr)
    def _serialize_groups(self):
        for i, g in enumerate(self.model.get('GaugeGroups', [])):
            if i >= CONFIG['MAX_GROUPS']: break
            g_type, g_dim = g['group'].split('_'); self.serialized_data['groups'][i, 0], self.serialized_data['groups'][i, 1], self.serialized_data['groups'][i, 2] = CONFIG['GROUP_TYPE_MAP'].get(g_type, 0), int(g_dim), 1 if g.get('name') == 'SU3C' else 0
    def _serialize_particles(self):
        for i, p in enumerate(self.model.get('particles', [])):
            if i >= CONFIG['MAX_PARTICLES']: break
            self.serialized_data['particles'][i, 0], self.serialized_data['particles'][i, 1], self.serialized_data['particles'][i, 2] = CONFIG['FIELD_TYPE_MAP'].get(p.get('type'), 0), p.get('charge', 0), p.get('mass', 0)
    def _serialize_fields(self):
        for i, f in enumerate(self.model.get('fields', [])):
            if i >= CONFIG['MAX_FIELDS']: break
            chirality_val = CONFIG['CHIRALITY_MAP'].get(f.get('chirality'), 0)
            self.serialized_data['fields'][i, 0], self.serialized_data['fields'][i, 1], self.serialized_data['fields'][i, 2], self.serialized_data['fields'][i, 3] = CONFIG['FIELD_TYPE_MAP'].get(f.get('type'), 0), f.get('dim', 1), f.get('gen', 1), chirality_val
            for gid, rname in f.get('reps', {}).items():
                gidx = self.group_id_to_idx.get(gid)
                if gidx is not None and gidx < CONFIG['MAX_GROUPS']: self.serialized_data['field_reps'][i, gidx] = CONFIG['REP_MAP']['singlet'] if isinstance(rname, (float,int)) else CONFIG['REP_MAP'].get(rname, 0)
            for j, pid in enumerate(f.get('particles', [])):
                if j < CONFIG['MAX_PARTICLES_PER_FIELD']: self.serialized_data['field_to_particles'][i, j] = self.particle_id_to_idx.get(pid, self.padding)

class GPUValidator:
    def __init__(self, batch_data: Dict[str, np.ndarray]):
        self.device = DEVICE
        self.tensors = {name: torch.from_numpy(arr).to(self.device) for name, arr in batch_data.items()}
        self.rep_dim_lut = torch.from_numpy(CONFIG['REP_DIM_LUT']).to(self.device)
        self.batch_size, self.padding, self.scores = self.tensors['groups'].shape[0], CONFIG['PADDING_VALUE'], CONFIG['SCORE_NORMALIZATION']

    def validate(self) -> torch.Tensor:
        try:
            # Run the fast GPU path
            scores = self._validate_gpu()
            # Synchronize here to catch any async errors now
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            return scores
        except RuntimeError as err:
            main_logger.warning(f"GPUValidator failed (OOB on CUDA). Falling back to CPU: {err}")
            # Run the safe CPU path and return CPU-tensor
            return self._validate_cpu()

    def _validate_gpu(self) -> torch.Tensor:
        # Ensure all indices are clamped before use
        group_scores = self.check_gauge_groups()
        field_scores = self.check_fields()
        interaction_scores = self._check_interactions_gpu()
        return torch.sum(group_scores, dim=-1) + torch.sum(field_scores, dim=-1) + torch.sum(interaction_scores, dim=-1)

    def _check_interactions_gpu(self) -> torch.Tensor:
        inter = self.tensors['interactions']
        fields = self.tensors['fields']
        mask = (inter[..., 0] != self.padding)
        if not mask.any():
            return torch.zeros_like(inter[..., 0], dtype=torch.int32)

        # Clamp indices to valid ranges before use
        max_f = fields.size(1) - 1
        i1 = torch.clamp(inter[..., 0], min=0, max=max_f)
        i2 = torch.clamp(inter[..., 1], min=0, max=max_f)
        i3 = torch.clamp(inter[..., 2], min=0, max=max_f)

        B = self.batch_size
        batch_idx = torch.arange(B, device=self.device).view(-1, 1)

        # Gather field records with clamped indices
        p1 = fields[batch_idx, i1]
        p2 = fields[batch_idx, i2]
        p3 = fields[batch_idx, i3]

        # Type checks with explicit bounds checking
        type_ok = (
            (p1[..., 0] == 1) & (p1[..., 3] == 1) &  # first is scalar
            (p2[..., 0] == 1) & (p2[..., 3] == 2) &  # second is fermion
            (p3[..., 0] == 3)                         # third is vector
        )
        gen_ok = (p1[..., 2] == p2[..., 2])
        symbolic_ok = (inter[..., 3] != self.padding)

        score = (
            type_ok.int() * 1 +
            gen_ok.int() * 1 +
            symbolic_ok.int() * 2
        )

        return score * mask.int()

    def check_fields(self) -> torch.Tensor:
        f, p, fp, fr, g = self.tensors['fields'], self.tensors['particles'], self.tensors['field_to_particles'], self.tensors['field_reps'], self.tensors['groups']
        mask = (f[:, :, 0] != self.padding)
        field_type = f[:, :, CONFIG['FIELD_PROPS']['TYPE']]
        type_check = field_type > 0

        # Clamp particle indices before use
        p_idx = torch.clamp(fp, min=0, max=p.size(1) - 1)
        b_idx = torch.arange(self.batch_size, device=self.device).view(-1, 1, 1)
        
        # Ensure rep_dim_lut indices are in bounds
        gt = torch.clamp(g[:, :, CONFIG['GROUP_PROPS']['TYPE']].unsqueeze(1), min=0, max=self.rep_dim_lut.size(0) - 1)
        gd = torch.clamp(g[:, :, CONFIG['GROUP_PROPS']['DIM']].unsqueeze(1), min=0, max=self.rep_dim_lut.size(1) - 1)
        fr_clamped = torch.clamp(fr, min=0, max=self.rep_dim_lut.size(2) - 1)

        # Rest of the field validation logic with clamped indices
        count_check = torch.sum(fp != self.padding, dim=-1) == f[:, :, CONFIG['FIELD_PROPS']['DIM']] * f[:, :, CONFIG['FIELD_PROPS']['GEN']]
        p_types = p[b_idx, p_idx, CONFIG['PARTICLE_PROPS']['TYPE']]
        match = (p_types == field_type.unsqueeze(2).expand_as(fp))
        match[fp == self.padding] = True
        particle_type_check = torch.all(match, dim=-1)
        is_fermion_mask = (field_type == CONFIG['FIELD_TYPE_MAP']['fermion'])
        chi = f[:, :, CONFIG['FIELD_PROPS']['CHIRALITY']]
        chirality_check = ((chi == 1) | (chi == 2)) | ~is_fermion_mask

        # Use clamped indices for rep_dim_lut lookup
        rep_dims = self.rep_dim_lut[gt, gd, fr_clamped]
        is_na_ns = (gt == 2) & (fr_clamped != 1)
        relevant_rep_dims = rep_dims * is_na_ns
        field_dim_exp = f[:, :, CONFIG['FIELD_PROPS']['DIM']].unsqueeze(-1)
        is_singlet_everywhere = torch.sum(relevant_rep_dims, dim=-1) == 0
        is_dim_one = f[:, :, CONFIG['FIELD_PROPS']['DIM']] == 1
        dim_matches_any_rep = torch.any((field_dim_exp == relevant_rep_dims) & (relevant_rep_dims != 0), dim=-1)
        dim_consistency_check = is_singlet_everywhere | is_dim_one | dim_matches_any_rep

        is_valid = type_check & count_check & particle_type_check & chirality_check & dim_consistency_check & mask
        base_score = is_valid.int() * self.scores['PER_SCALAR_FIELD']
        fermion_bonus = is_valid.int() * (self.scores['PER_FERMION_FIELD'] - self.scores['PER_SCALAR_FIELD'])
        return base_score + (fermion_bonus * is_fermion_mask)

    def check_gauge_groups(self) -> torch.Tensor:
        g, mask = self.tensors['groups'], (self.tensors['groups'][:, :, 0] != self.padding)
        t, d = g[:, :, CONFIG['GROUP_PROPS']['TYPE']], g[:, :, CONFIG['GROUP_PROPS']['DIM']]
        is_valid = ((t == 1) | (t == 2)) & ((t != 1) | (d == 1)) & ((t != 2) | ((d == 2) | (d == 3)))
        return (is_valid & mask).int() * self.scores['PER_GAUGE_GROUP']

    def _validate_cpu(self) -> torch.Tensor:
        # Pull everything to CPU numpy
        inter = self.tensors['interactions'].cpu().numpy()
        fields = self.tensors['fields'].cpu().numpy()
        B, M, _ = inter.shape
        scores = np.zeros((B, M), dtype=np.int32)

        padding = self.padding
        for b in range(B):
            for j in range(M):
                f1, f2, f3, sym = inter[b, j]
                if f1 == padding:
                    continue
                # clamp
                max_f = fields.shape[1] - 1
                i1 = min(max(f1, 0), max_f)
                i2 = min(max(f2, 0), max_f)
                i3 = min(max(f3, 0), max_f)

                p1 = fields[b, i1]
                p2 = fields[b, i2]
                p3 = fields[b, i3]

                type_ok = (p1[0] == 1 and p1[3] == 1
                        and p2[0] == 1 and p2[3] == 2
                        and p3[0] == 3)
                gen_ok = (p1[2] == p2[2])
                sym_ok = (sym != padding)

                scores[b, j] = type_ok * 1 + gen_ok * 1 + sym_ok * 2

        return torch.from_numpy(scores)

class BatchValidator:
    def __init__(self):
        self.device = DEVICE
        main_logger.info(f"BatchValidator initialized. Target device: {str(self.device).upper()}")
    def validate_batch(self, batch_of_models: List[Dict[str, Any]]) -> np.ndarray:
        if not batch_of_models: return np.array([], dtype=np.int32)
        serialized_models = []
        for m in batch_of_models:
            try: s = ModelSerializer(m); d = s.serialize(); serialized_models.append(d)
            except Exception as e: main_logger.error(f"Fatal serialization error: {e}"); return np.full(len(batch_of_models), -100, dtype=np.int32)
        if not serialized_models: return np.array([], dtype=np.int32)
        batch_data = {k: np.stack([d[k] for d in serialized_models]) for k in serialized_models[0].keys()}
        validator = GPUValidator(batch_data)
        cpu_tensor = validator.validate()
        # Ensure it's on CPU before numpy()
        if cpu_tensor.device.type != "cpu":
            cpu_tensor = cpu_tensor.cpu()
        return cpu_tensor.numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Grammar and Token Definitions
# ─────────────────────────────────────────────────────────────────────────────
TOKENS: List[str] = (
    ["BOS", "EOS"]
    + ["ITRACT", "END_ITRACT"] + [f"ITRACT_ID_{i}" for i in range(10)]
    + ["FIELD", "END_FIELD"] + [f"FIELD_ID_{i}" for i in range(10)]
    + ["PARTICLE", "END_PARTICLE"] + [f"PARTICLE_ID_{i}" for i in range(20)]
    + ["TYPE_DC", "TYPE_YUKAWA", "TYPE_VLF", "TYPE_PHI4", "TYPE_FF", "TYPE_FFDUAL"]
    + ["TYPE_complex", "TYPE_real", "TYPE_fermion", "TYPE_vector"]
    + [f"MASS_1e{i}" for i in range(4)] + [f"CHARGE_{i}" for i in [-1, 0, 1]]
    + [f"DIM_{i}" for i in [1, 2, 3]] + [f"GEN_{i}" for i in [1, 2, 3]]
    + ["SELF_CONJ_TRUE", "SELF_CONJ_FALSE"] + ["CHIRALITY_left", "CHIRALITY_right", "CHIRALITY_na"]
    + ["REP_SU3C_singlet", "REP_SU3C_fnd", "REP_SU3C_adj"] + ["REP_SU2L_singlet", "REP_SU2L_fnd", "REP_SU2L_adj"]
    + [f"REP_U1Y_{s}{i}".replace('0','') for s, i in [("minus", 1), ("0", ""), ("1", "")] ]
    + [f"QN_LeptonNumber_{s}{i}".replace('0','') for s, i in [("minus", 1), ("0", ""), ("1", "")] ]
    + [f"QN_BaryonNumber_{s}{i}".replace('0','') for s, i in [("minus", 1), ("0", ""), ("1", "")] ]
)
TOKEN2ID = {t: i for i, t in enumerate(TOKENS)}; ID2TOK = {i: t for t, i in TOKEN2ID.items()}
VOCAB_SIZE, BOS_TOKEN, EOS_TOKEN = len(TOKENS), TOKEN2ID["BOS"], TOKEN2ID["EOS"]

grammar = TorchGrammar()
tr = lambda s, e: (TOKEN2ID[s], TOKEN2ID[e])
grammar.add_object("Particle", [ TOKEN2ID["PARTICLE"], tr("PARTICLE_ID_0", f"PARTICLE_ID_{19}"), tr("TYPE_complex", "TYPE_vector"), tr("MASS_1e0", "MASS_1e3"), tr("CHARGE_-1", "CHARGE_1"), TOKEN2ID["END_PARTICLE"] ])
grammar.add_object("Field", [ TOKEN2ID["FIELD"], tr("FIELD_ID_0", f"FIELD_ID_{9}"), tr("TYPE_complex", "TYPE_vector"), tr("DIM_1", "DIM_3"), tr("GEN_1", "GEN_3"), tr("SELF_CONJ_TRUE", "SELF_CONJ_FALSE"), tr("CHIRALITY_left", "CHIRALITY_na"), tr("REP_SU3C_singlet", "REP_SU3C_adj"), tr("REP_SU2L_singlet", "REP_SU2L_adj"), tr("REP_U1Y_minus1", "REP_U1Y_1"), tr("QN_LeptonNumber_minus1", "QN_LeptonNumber_1"), tr("QN_BaryonNumber_minus1", "QN_BaryonNumber_1"), {"Particle": lambda c: c > 0}, TOKEN2ID["END_FIELD"] ])
grammar.add_object("Interaction", [ TOKEN2ID["ITRACT"], tr("ITRACT_ID_0", f"ITRACT_ID_{9}"), tr("TYPE_DC", "TYPE_FFDUAL"), {"Field": lambda c: c > 0}, TOKEN2ID["END_ITRACT"] ])
grammar.add_object("Model", [TOKEN2ID["BOS"], {"Interaction": lambda c: c >= 1}, TOKEN2ID["EOS"]])
ROOTS = ["Model"]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Sequence Detokenizer (For Scorer)
# ─────────────────────────────────────────────────────────────────────────────
class SequenceDetokenizer:
    def parse_token(self, token: str) -> Tuple[str, Any]:
        parts = token.split('_')
        prefix = parts[0]
        if prefix == "TYPE": return "type", parts[1].lower()
        if prefix == "DIM": return "dim", int(parts[1])
        if prefix == "GEN": return "gen", int(parts[1])
        if prefix == "SELF": return "self_conj", parts[2].lower() == "true"
        if prefix == "CHIRALITY": return "chirality", parts[1] if parts[1] != "na" else "none"
        if prefix == "MASS": return "mass", float(parts[1].replace('e', 'e'))
        if prefix == "CHARGE": return "charge", int(parts[1])
        if prefix == "REP": return "rep", (parts[1], parts[2])
        return None, None
    def detokenize(self, token_list: List[str]) -> Dict[str, Any]:
        model = {"GaugeGroups": [ {"id": "g1", "name": "U1Y", "group": "U_1"}, {"id": "g2", "name": "SU2L", "group": "SU_2"}, {"id": "g3", "name": "SU3C", "group": "SU_3"} ], "particles": [], "fields": [], "interactions": []}
        stack, obj_counter = [], {"particle": 0, "field": 0, "interaction": 0}
        for token in token_list:
            if token in ["BOS", "EOS"]: continue
            if token == "PARTICLE": obj_id = f"p{obj_counter['particle']}"; obj_counter['particle'] += 1; stack.append({"type": "particle", "id": obj_id, "data": { "id": obj_id }}); continue
            if token == "FIELD": obj_id = f"f{obj_counter['field']}"; obj_counter['field'] += 1; stack.append({"type": "field", "id": obj_id, "data": {"id": obj_id, "particles": [], "reps": {}}}); continue
            if token == "ITRACT": obj_id = f"i{obj_counter['interaction']}"; obj_counter['interaction'] += 1; stack.append({"type": "interaction", "id": obj_id, "data": {"id": obj_id, "fields": []}}); continue
            if token.startswith("END_") and stack:
                obj_type_ended = token.split('_')[1].lower(); completed_obj = stack.pop()
                if completed_obj['type'] == obj_type_ended:
                    if not stack: model[f"{obj_type_ended}s"].append(completed_obj['data'])
                    else:
                        parent = stack[-1]
                        if completed_obj['type'] == 'particle':
                            model['particles'].append(completed_obj['data'])
                            if parent['type'] == 'field': parent['data']['particles'].append(completed_obj['id'])
                        elif completed_obj['type'] == 'field':
                            model['fields'].append(completed_obj['data'])
                            if parent['type'] == 'interaction': parent['data']['fields'].append(completed_obj['id'])
                continue
            if not stack: continue
            key, value = self.parse_token(token)
            if key:
                if key == "rep":
                    group_name, rep_val = value; group_map = {"U1Y": "g1", "SU2L": "g2", "SU3C": "g3"}; gid = group_map.get(group_name)
                    if gid: stack[-1]['data']['reps'][gid] = rep_val
                else: stack[-1]['data'][key] = value
        return model

# ─────────────────────────────────────────────────────────────────────────────
# 6. Fast Grammar Parser
# ─────────────────────────────────────────────────────────────────────────────
class FastGrammarParser:
    def __init__(self, grammar: TorchGrammar, batch_size: int, device: torch.device):
        self.grammar, self.batch_size, self.device = grammar, batch_size, device
        self.parser = TorchGrammarBatchParser(grammar, batch_size, device=device)
    def start_batch(self, roots: List[str]):
        self.parser.set_batch(roots)
    def get_mask(self) -> torch.Tensor:
        return self.parser.next_token_mask(VOCAB_SIZE)
    def update(self, actions: torch.Tensor):
        self.parser.fast_is_valid(actions)
    @property
    def is_finished(self) -> torch.Tensor:
        return self.parser.next_lens == 0


# ─────────────────────────────────────────────────────────────────────────────
# 7. Transformer (actor-critic) & Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
class TransformerPolicy(nn.Module):
    def __init__(self, vocab: int, *, d_model: int = 256, nhead: int = 4, layers: int = 3, max_T: int = 128):
        super().__init__()
        self.tok_emb, self.pos_emb = nn.Embedding(vocab, d_model), nn.Embedding(max_T, d_model)
        self.register_buffer("pos", torch.arange(max_T))
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, 0.1, batch_first=True, norm_first=True)
        self.encoder, self.actor, self.value = nn.TransformerEncoder(enc, layers), nn.Linear(d_model, vocab), nn.Linear(d_model, 1)
    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.tok_emb(seq) + self.pos_emb(self.pos[: seq.size(1)].to(seq.device))
        h = self.encoder(x)
        return self.actor(h), self.value(h).squeeze(-1)

@dataclass(frozen=True)
class HParams:
    epochs: int = 500; batch_size: int = 512; seq_len: int = 512
    lr: float = 3e-4; gamma: float = 0.99; ppo_epochs: int = 4
    clip_epsilon: float = 0.2; gae_lambda: float = 0.95; 
    value_coef: float = 0.1; sup_coef: float = 0.9; ent_coef: float = 0.4; 
    grad_clip_norm: float = 1.0; ckpt_every: int = 20
    buffer_size: int = 100; min_buffer_size: int = 5; rank_k: float = 0.05
    enable_logging: bool = True; debug: bool = False
hp = HParams()

# ─────────────────────────────────────────────────────────────────────────────
# 8. Experience Buffer
# ─────────────────────────────────────────────────────────────────────────────
class PPOTrajectory(NamedTuple):
    seq: torch.Tensor; rewards: torch.Tensor; logps: torch.Tensor; values: torch.Tensor; ep_len: int; total_reward: float

class ExperienceBuffer:
    def __init__(self, hp: HParams):
        self.buffer = []  # Changed from deque to list
        self.hp = hp
        # Track unique sequences using a set of tuples (since tensors aren't hashable)
        self._unique_seqs = set()
        # Track sequence to index mapping for faster removal
        self._seq_to_idx = {}

    def __len__(self) -> int:
        return len(self.buffer)

    def _seq_to_tuple(self, seq: torch.Tensor) -> Tuple[int, ...]:
        """Convert a sequence tensor to a hashable tuple."""
        return tuple(seq.cpu().tolist())

    def add_batch(self, generated_seqs: torch.Tensor, batch_data: Dict[str, torch.Tensor]):
        rews, logps, values = batch_data['rewards'], batch_data['logps'], batch_data['values']
        new_trajectories = []
        
        for i in range(rews.size(0)):
            ep_len = batch_data['ep_len'][i].item()
            if ep_len > 0:
                # Get the sequence up to ep_len + 1 (including BOS)
                seq = generated_seqs[i, :ep_len + 1].clone().cpu()
                seq_tuple = self._seq_to_tuple(seq)
                
                # Skip if we've seen this sequence before
                if seq_tuple in self._unique_seqs:
                    continue

                ep_rews = rews[i, :ep_len]
                discounts = self.hp.gamma ** torch.arange(ep_len, device=DEVICE)
                total_reward = (ep_rews * discounts).sum().item()

                # Create trajectory
                traj = PPOTrajectory(
                    seq=seq,
                    rewards=ep_rews.clone().cpu(),
                    logps=logps[i, :ep_len].clone().cpu(),
                    values=values[i, :ep_len].clone().cpu(),
                    ep_len=ep_len,
                    total_reward=total_reward
                )
                new_trajectories.append((traj, seq_tuple))

        # Add new trajectories to buffer
        for traj, seq_tuple in new_trajectories:
            # If buffer is full, remove lowest scoring sequence
            if len(self.buffer) == self.hp.buffer_size:
                # Find lowest scoring sequence
                min_idx = min(range(len(self.buffer)), 
                            key=lambda i: self.buffer[i].total_reward)
                # Remove it from both buffer and unique set
                removed_seq = self._seq_to_tuple(self.buffer[min_idx].seq)
                if removed_seq in self._unique_seqs:  # Safety check
                    self._unique_seqs.remove(removed_seq)
                if removed_seq in self._seq_to_idx:  # Clean up mapping
                    del self._seq_to_idx[removed_seq]
                self.buffer.pop(min_idx)

            # Add new sequence
            self.buffer.append(traj)
            self._unique_seqs.add(seq_tuple)
            self._seq_to_idx[seq_tuple] = len(self.buffer) - 1

    def sample(self) -> Dict[str, torch.Tensor]:
        if not self.buffer:
            raise RuntimeError("Cannot sample from empty buffer")

        # Verify consistency of our tracking sets
        current_seqs = {self._seq_to_tuple(t.seq) for t in self.buffer}
        if current_seqs != self._unique_seqs:
            # If there's a mismatch, rebuild tracking sets
            self._unique_seqs = current_seqs
            self._seq_to_idx = {seq: i for i, seq in enumerate(current_seqs)}

        buffer_rewards = torch.tensor([t.total_reward for t in self.buffer], device=DEVICE)
        ranks = buffer_rewards.argsort(descending=True).argsort().float() + 1.0
        weights = (self.hp.rank_k * len(self.buffer) + ranks).pow(-1)
        
        # Sample indices, ensuring we don't try to sample more than available
        n_samples = min(self.hp.batch_size, len(self.buffer))
        indices = torch.multinomial(weights / weights.sum(), n_samples, replacement=True)
        
        samples = [self.buffer[i] for i in indices]
        max_len = max(s.ep_len for s in samples)

        # Initialize tensors
        collated_seqs = torch.full((n_samples, max_len + 1), BOS_TOKEN, dtype=torch.long)
        collated_rewards = torch.zeros((n_samples, max_len))
        collated_logps = torch.zeros((n_samples, max_len))
        collated_values = torch.zeros((n_samples, max_len))
        collated_ep_len = torch.tensor([s.ep_len for s in samples], dtype=torch.long)

        # Fill tensors
        for i, s in enumerate(samples):
            collated_seqs[i, :s.ep_len + 1] = s.seq
            collated_rewards[i, :s.ep_len] = s.rewards
            collated_logps[i, :s.ep_len] = s.logps
            collated_values[i, :s.ep_len] = s.values

        # Move to device
        collated_seqs = collated_seqs.to(DEVICE)
        collated_rewards = collated_rewards.to(DEVICE)
        collated_logps = collated_logps.to(DEVICE)
        collated_values = collated_values.to(DEVICE)
        collated_ep_len = collated_ep_len.to(DEVICE)

        # Compute advantages
        advantages, last_gae_lam = torch.zeros_like(collated_rewards), 0
        for t in reversed(range(max_len)):
            is_not_last = (t < collated_ep_len - 1).float()
            next_vals = collated_values[:, t + 1] if t < max_len - 1 else torch.zeros_like(collated_values[:, t])
            delta = collated_rewards[:, t] + self.hp.gamma * next_vals * is_not_last - collated_values[:, t]
            advantages[:, t] = last_gae_lam = delta + self.hp.gamma * self.hp.gae_lambda * last_gae_lam * is_not_last

        return {
            'seq': collated_seqs,
            'logps': collated_logps,
            'values': collated_values,
            'returns': advantages + collated_values,
            'advantages': advantages,
            'ep_len': collated_ep_len
        }

# ─────────────────────────────────────────────────────────────────────────────
# 9. PPO Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def train_reinforce(model: nn.Module, opt, hp: HParams) -> Path:
    run_id  = datetime.now().strftime("combo_%Y%m%d_%H%M%S")
    logdir  = Path("./runs") / run_id
    ckptdir = logdir / "checkpoints"
    prof_dir = logdir / "profiler"
    logdir.mkdir(parents=True, exist_ok=True)
    ckptdir.mkdir(exist_ok=True)
    prof_dir.mkdir(exist_ok=True)

    logger: Optional[Logger] = None
    if hp.enable_logging:
        logger = Logger(logdir, cli_interval=1, weights_interval=hp.ckpt_every * 10)
        logger.add_metric("reward",      kind="stat")
        logger.add_metric("seq_len",     kind="stat")
        logger.add_metric("pg_loss",     kind="stat")
        logger.add_metric("v_loss",      kind="stat")
        logger.add_metric("e_loss",      kind="stat")
        logger.add_metric("sup_loss",    kind="stat")
        logger.add_metric("total_loss",  kind="stat")
        logger.add_metric("buffer_size", kind="scalar")
        logger.add_metric("learning_rate", kind="scalar")
        logger.model = model

    sched              = torch.optim.lr_scheduler.CosineAnnealingLR(opt, hp.epochs)
    experience_buffer  = ExperienceBuffer(hp)
    batch_validator    = BatchValidator()
    detokenizer        = SequenceDetokenizer()

    for ep in range(1, hp.epochs + 1):
        # --- Rollout Generation ---
        model.eval()
        parser = FastGrammarParser(grammar, hp.batch_size, device=DEVICE)
        parser.start_batch(random.choices(ROOTS, k=hp.batch_size))

        seq    = torch.full((hp.batch_size, hp.seq_len + 1), BOS_TOKEN,
                            dtype=torch.long, device=DEVICE)
        ep_len = torch.zeros(hp.batch_size, dtype=torch.long, device=DEVICE)
        alive  = torch.ones(hp.batch_size, dtype=torch.bool, device=DEVICE)

        rewards_h, logps_h, values_h = [], [], []

        for t in range(hp.seq_len):
            with torch.no_grad():
                logits_t, values_t = model(seq[:, : t + 1])

            logits, mask = logits_t[:, -1].float(), parser.get_mask()
            eligible = alive & ~parser.is_finished
            if not eligible.any():
                break

            probs = F.softmax(logits + mask, dim=-1).nan_to_num_(0.0)
            probs[probs.sum(-1) == 0, EOS_TOKEN] = 1.0

            dist = Categorical(probs=probs)
            acts = dist.sample()

            parser.update(acts)
            seq[:, t + 1] = acts

            rewards_h.append(torch.zeros(hp.batch_size, device=DEVICE))
            logps_h.append(dist.log_prob(acts).clone())
            values_h.append(values_t[:, -1].clone())

            ep_len[alive] += 1
            alive &= (acts != EOS_TOKEN)

        # --- Terminal Reward Calculation ---
        final_rewards = torch.zeros(hp.batch_size, device=DEVICE)
        completed = [i for i in range(hp.batch_size) if ep_len[i] > 0]
        if completed:
            models_to_score = [
                detokenizer.detokenize([ID2TOK[tid] for tid in seq[i, 1:ep_len[i]+1].tolist()])
                for i in completed
            ]
            final_rewards[completed] = torch.tensor(
                batch_validator.validate_batch(models_to_score),
                device=DEVICE, dtype=torch.float32
            )

        last_avg_reward  = final_rewards.mean().item()
        last_avg_seq_len = ep_len.float().mean().item()

        # --- Fill SIL buffer ---
        if rewards_h:
            rewards_tensor = torch.stack(rewards_h, dim=1)
            for i in range(hp.batch_size):
                if ep_len[i] > 0:
                    rewards_tensor[i, ep_len[i] - 1] += final_rewards[i]

            batch_dict = {
                'rewards': rewards_tensor,
                'logps':   torch.stack(logps_h, dim=1),
                'values':  torch.stack(values_h, dim=1),
                'ep_len':  ep_len
            }
            experience_buffer.add_batch(seq, batch_dict)

            if logger:
                logger.log({
                    "reward":      last_avg_reward,
                    "seq_len":     last_avg_seq_len,
                    "buffer_size": len(experience_buffer)
                })

        # --- Warm-up check ---
        if len(experience_buffer) < hp.min_buffer_size:
            if ep % 10 == 0:
                print(f"Epoch [{ep:4d}/{hp.epochs}] | Warming up buffer "
                      f"({len(experience_buffer)}/{hp.min_buffer_size})...")
            sched.step()
            continue

        # --- PPO + supervised penalty update ---
        model.train()
        tot_pg = tot_v = tot_e = tot_sup = 0.0

        for _ in range(hp.ppo_epochs):
            tbatch = experience_buffer.sample()
            N, T   = tbatch['seq'].size(0), tbatch['seq'].size(1) - 1

            # build grammar legality masks
            parser_sup = FastGrammarParser(grammar, N, device=DEVICE)
            parser_sup.start_batch(["Model"] * N)
            mask_list = []
            for t in range(T):
                # get_mask() returns 0.0 for legal, -inf for illegal
                legal = (parser_sup.get_mask() == 0.0)   # [N, vocab]
                mask_list.append(legal)
                parser_sup.update(tbatch['seq'][:, t+1])
            legal_tensor   = torch.stack(mask_list, dim=1)  # [N, T, V]
            illegal_tensor = ~legal_tensor

            # forward
            logits, vals = model(tbatch['seq'][:, :-1])
            dist   = Categorical(logits=logits)
            new_lp = dist.log_prob(tbatch['seq'][:, 1:])
            ents   = dist.entropy()

            # construct valid_mask using ep_len
            # shape: [N, T]
            valid_mask = torch.arange(T, device=DEVICE)[None, :].expand(N, T) \
                         < tbatch['ep_len'][:, None]

            # PPO losses
            adv = tbatch['advantages']
            adv = (adv - adv[valid_mask].mean()) / (adv[valid_mask].std() + EPS)
            ratio = (new_lp - tbatch['logps']).exp()
            s1, s2 = ratio * adv, torch.clamp(ratio, 1-hp.clip_epsilon, 1+hp.clip_epsilon) * adv

            pg_loss = -torch.min(s1, s2)[valid_mask].mean()
            v_loss  = F.mse_loss(vals[valid_mask], tbatch['returns'][valid_mask])
            e_loss  = -ents[valid_mask].mean()

            # supervised grammar penalty
            sup_loss = logits.masked_select(illegal_tensor).pow(2).mean()

            # combine & step
            loss = (
                pg_loss
              + hp.value_coef * v_loss
              + hp.ent_coef  * e_loss
              + hp.sup_coef  * sup_loss
            )

            opt.zero_grad(set_to_none=True)
            SCALER.scale(loss).backward()
            SCALER.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip_norm)
            SCALER.step(opt)
            SCALER.update()

            tot_pg  += pg_loss.item()
            tot_v   += v_loss.item()
            tot_e   += e_loss.item()
            tot_sup += sup_loss.item()

            if logger:
                logger.log({
                    "pg_loss":   pg_loss.item(),
                    "v_loss":    v_loss.item(),
                    "e_loss":    e_loss.item(),
                    "sup_loss":  sup_loss.item(),
                    "total_loss": loss.item()
                })
                logger.state_step()

        sched.step()

        # console print
        print(
            f"Epoch [{ep:4d}/{hp.epochs}] | Reward: {last_avg_reward:6.2f} | "
            f"Len: {last_avg_seq_len:4.1f} | "
            f"PGLoss: {tot_pg/hp.ppo_epochs:.3f} | VLoss: {tot_v/hp.ppo_epochs:.3f} | "
            f"ELoss: {tot_e/hp.ppo_epochs:.3f} | SUP: {tot_sup/hp.ppo_epochs:.3f} | "
            f"Buffer: {len(experience_buffer):4d}"
        )

        if logger:
            logger.log({"learning_rate": sched.get_last_lr()[0]})
            logger.cli_step()

        if ep % hp.ckpt_every == 0 or ep == hp.epochs:
            ckpt_path = ckptdir / f"ep{ep}.pt"
            to_save = getattr(model, "_orig_mod", model)
            torch.save(to_save.state_dict(), ckpt_path)
            main_logger.info(f"✓ Checkpoint saved to {ckpt_path}")

    main_logger.info("Training Complete.")
    return ckptdir / f"ep{hp.epochs}.pt"





# ─────────────────────────────────────────────────────────────────────────────
# 10. Sampling and Main Execution
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def sample(model_to_sample: nn.Module, T: int, *, n: int = 4, temp: float = 1.0) -> None:
    model_to_sample.eval()
    for i in range(n):
        parser = FastGrammarParser(grammar, 1, device=DEVICE)
        parser.start_batch(random.choices(ROOTS, k=1))
        seq_ids = [BOS_TOKEN]
        print(f"\n--- Sample {i+1} (temp={temp}) ---")
        for _ in range(T):
            logits_all, _ = model_to_sample(torch.tensor([seq_ids], device=DEVICE))
            logits, mask = logits_all[:, -1].squeeze(0), parser.get_mask()[0]
            probs = F.softmax((logits + mask) / max(temp, 1e-5) if temp > 0 else (logits+mask)*1e3, 0).nan_to_num_(0)
            if probs.sum() < EPS: break
            tok_id = int(Categorical(probs).sample().item()); print(f" {ID2TOK[tok_id]}", end="")
            seq_ids.append(tok_id); parser.update(torch.tensor([tok_id], device=DEVICE))
            if tok_id == EOS_TOKEN or parser.is_finished[0]: break
        print("\n" + "-"*20)
    model_to_sample.train()

if __name__ == "__main__":
    main_logger.info(f"Device: {DEVICE}, AMP: {'ON' if DEVICE.type=='cuda' else 'OFF'}, Logging: {'ON' if hp.enable_logging else 'OFF'}")
    transformer_model = TransformerPolicy(VOCAB_SIZE, max_T=hp.seq_len + 1).to(DEVICE)
    optimizer = torch.optim.AdamW(transformer_model.parameters(), lr=hp.lr)
    final_checkpoint = train_reinforce(transformer_model, optimizer, hp)
    main_logger.info("Training complete. Model saved to %s", final_checkpoint)
    model_to_sample = TransformerPolicy(VOCAB_SIZE, max_T=hp.seq_len + 1).to(DEVICE)
    model_to_sample.load_state_dict(torch.load(final_checkpoint, map_location=DEVICE))
    sample(model_to_sample, hp.seq_len, n=5, temp=0.0)
    sample(model_to_sample, hp.seq_len, n=5, temp=0.7)
    print("\nDone.")