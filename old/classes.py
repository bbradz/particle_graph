# ==== classes.py ====
"""logger_parser.py – utilities for logging, grammar‑aware masking, and model tests.

This version is *functionally identical* to your previous file but is now fully
compatible with the tensor‑aware `Model` class (see *model.py*). No core logic
changed – every public class and method signature remains the same, so existing
imports `from classes import Logger, TorchGrammar …` are unaffected.

The only code‑level tweaks:
1. **ModelTester.test_model()** – uses the new `Model` transparently (no extra
   work required). Because `Model` auto‑unwraps tensors, the implementation
   needs no edits.
2. A type‑hint update where tuple aliases were missing (pure lint fix).

All other lines are untouched to guarantee deterministic output.
"""

from __future__ import annotations

# ───────────────────────── Imports & Global Setup ────────────────────────────
import csv, io, json, math, logging, os, re, sys, time, contextlib
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Union, Optional # Added Optional

import torch
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────── Type Aliases ──────────────────────────────
Token     = Union[int, Tuple[int, ...], str, Dict[str, Any]]
Predicate = Callable[[int], bool]


# ──────────────────────────── Self-Imitation Buffer ────────────────────────────
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
        """Insert & keep only the *capacity* highest-return trajectories."""
        self._data.extend(trajs)
        self._data = deque(sorted(self._data, key=lambda x: x.ret, reverse=True)[:self.capacity], maxlen=self.capacity)

    def sample(self, batch: int) -> List[Trajectory]:
        idx = torch.randint(0, len(self._data), (batch,))
        return [self._data[i] for i in idx]


class RunningStat:
    """Exponential moving average (EMA) style running mean / variance tracker.

    Uses an exponential decay factor (alpha) to give more weight to recent samples.
    Storage- and time-complexity: O(1) per update.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        """Initialize with decay rate alpha (0 < alpha <= 1).
        
        Smaller alpha = more smoothing (slower to adapt to changes)
        Larger alpha = less smoothing (faster to adapt to changes)
        """
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self._alpha = alpha
        self._count: int = 0
        self._mean: float = 0.0
        self._var: float = 0.0  # Running variance using EMA
        self._most_recent: float = 0.0

    # ------------------------------------------------------------------
    # Public read-only properties.
    # ------------------------------------------------------------------
    @property
    def count(self) -> int:  # noqa: D401
        """Number of samples seen so far."""
        return self._count

    @property
    def mean(self) -> float:  # noqa: D401
        """Exponential moving average of samples."""
        return self._mean

    @property
    def variance(self) -> float:  # noqa: D401
        """Exponential moving average of squared differences."""
        return self._var

    @property
    def std(self) -> float:  # noqa: D401
        """Square root of variance."""
        return math.sqrt(max(0.0, self._var))

    @property
    def most_recent(self) -> float:  # noqa: D401
        """Most recently-observed value."""
        return self._most_recent

    # ------------------------------------------------------------------
    # Mutator – EMA update.
    # ------------------------------------------------------------------
    def update(self, x: float) -> None:
        """Consume a new scalar value and update EMA stats."""
        if not math.isfinite(x):
            return

        self._most_recent = x
        self._count += 1

        # For the first sample, initialize mean and variance
        if self._count == 1:
            self._mean = x
            self._var = 0.0
            return

        # Update mean using EMA
        delta = x - self._mean
        self._mean += self._alpha * delta

        # Update variance using EMA
        # Using the formula: var = (1-alpha) * var + alpha * (x - mean)^2
        self._var = (1 - self._alpha) * self._var + self._alpha * delta * (x - self._mean)


# ───────────────────────────── Experiment Logger ────────────────────────────
class Logger:
    """
    Generic experiment logger.

    • add_metric(name, kind="stat")   – register a variable you plan to log
       kind ∈ {"stat", "scalar"}
       "stat"    → RunningStat is kept (mean / std / most-recent)
       "scalar"  → last value only (e.g. counters, LR)

    • log({...})        – push a batch of updates
    • state_step()      – per-action   write (state.csv, TensorBoard)
    • cli_step()        – per-epoch    write (cli.csv , TensorBoard)
    """

    CLI_CSV, STATE_CSV = "cli.csv", "state.csv"
    WEIGHTS_DIR, TBOARD_DIR = "weights", "tensorboard"

    def __init__(
        self, path: str | os.PathLike,
        *, cli_interval: int = 1_000, state_interval: int = 1,
        weights_interval: int = 100_000,
    ) -> None:
        self.root = Path(path); self.root.mkdir(parents=True, exist_ok=True)
        self.cli_interval, self.state_interval = cli_interval, state_interval
        self.weights_interval = weights_interval

        # destination paths
        self.weights_dir = self.root / self.WEIGHTS_DIR; self.weights_dir.mkdir(exist_ok=True)
        self.tboard_dir  = self.root / self.TBOARD_DIR ; self.tboard_dir.mkdir(exist_ok=True)

        # csv writers with explicit buffering
        self._cli_csv   = (self.root / self.CLI_CSV  ).open("w", newline="", buffering=1)
        self._state_csv = (self.root / self.STATE_CSV).open("w", newline="", buffering=1)
        self.cli_writer, self.state_writer = csv.writer(self._cli_csv), csv.writer(self._state_csv)
        self.tb = SummaryWriter(str(self.tboard_dir), flush_secs=1)  # Force more frequent flushing

        # timers / counters
        self.timestep, self.seconds = 0, 0.0
        self._t0 = time.time()

        # metric stores (filled by add_metric)
        self.stats   : dict[str, RunningStat] = {}
        self.scalars : dict[str, float]       = {}

        # mandatory metrics
        self.add_metric("fps", kind="stat")
        self.state, self.action, self.logits, self.model = None, None, None, None

        # write *placeholder* headers – finalised after first add_metric batch
        self._cli_header_written  = False
        self._state_header_written= False

    def __del__(self):
        """Ensure proper cleanup of file handles and tensorboard writer."""
        if hasattr(self, '_cli_csv'):
            self._cli_csv.close()
        if hasattr(self, '_state_csv'):
            self._state_csv.close()
        if hasattr(self, 'tb'):
            self.tb.close()

    # ──────────────────────── public helpers ─────────────────────────
    def add_metric(self, name: str, *, kind: str = "stat") -> None:
        """Register a metric **before** logging for it."""
        if kind not in {"stat", "scalar"}:
            raise ValueError("kind must be 'stat' or 'scalar'")
        if name in self.stats or name in self.scalars:
            raise ValueError(f"Metric '{name}' already registered")

        if kind == "stat":
            self.stats[name] = RunningStat()
        else:
            self.scalars[name] = 0.0

    # -----------------------------------------------------------------
    def log(self, values: Dict[str, Any]) -> None:
        """Push a dict of updates (called both per-action and per-epoch)."""
        for k, v in values.items():
            if k in self.stats:
                self.stats[k].update(float(v))
            elif k in self.scalars:
                self.scalars[k] = float(v)
            elif k in {"state", "action", "logits", "weights"}:
                setattr(self, k if k != "weights" else "model", v)
            else:
                raise KeyError(f"'{k}' not registered – call add_metric() first")

    # ───────────────────────── inner-loop write ─────────────────────
    def state_step(self) -> None:
        self.timestep += 1
        self.seconds = time.time() - self._t0
        self.stats["fps"].update(self.timestep / max(self.seconds, 1e-9))

        if not self._state_header_written:
            self._write_state_header(); self._state_header_written = True

        if self.timestep % self.state_interval == 0:
            #print("writing to tb...")
            self.state_writer.writerow([
                self.timestep, self.seconds,
                self.state, self.stats.get("reward", RunningStat()).most_recent,
                self.action, self.logits,
            ])
            # tensorboard mirrors
            self.tb.add_scalar("state/reward", self.stats.get("reward", RunningStat()).most_recent, self.timestep)
            self.tb.add_scalar("state/fps",     self.stats["fps"].mean, self.timestep)

        # periodic weight dump
        if self.model is not None and self.timestep % self.weights_interval == 0:
            torch.save(self.model.state_dict(), self.weights_dir / f"weights_{self.timestep}.pt")

    # ───────────────────────── outer-loop write ─────────────────────
    def cli_step(self) -> None:
        if not self._cli_header_written:
            self._write_cli_header(); self._cli_header_written = True

        row = [self.timestep, self.seconds,
               self.stats["fps"].mean, self.stats["fps"].std]

        # append all scalars
        for name in sorted(self.scalars):
            row.append(self.scalars[name])

        # append running-stat triples
        for name in sorted(self.stats):
            if name == "fps": continue  # already logged above
            st = self.stats[name]
            row.extend([st.most_recent, st.mean, st.std])

        self.cli_writer.writerow(row)
        self._cli_csv.flush()  # Explicitly flush the CSV file

        # tensorboard mirror with explicit flushing
        #print("writing to tb...")
        for name, st in self.stats.items():
            self.tb.add_scalar(f"train/{name}_mean", st.mean, self.timestep)
            self.tb.add_scalar(f"train/{name}_std",   st.std,   self.timestep)
            self.tb.add_scalar(f"train/{name}_mr",     st.most_recent, self.timestep)
        for name, val in self.scalars.items():
            self.tb.add_scalar(f"train/{name}", val, self.timestep)
        self.tb.flush()  # Explicitly flush tensorboard writes

    # ───────────────────────── header helpers ────────────────────────
    def _write_state_header(self) -> None:
        self.state_writer.writerow(["timestep", "seconds", "state",
                                   "reward_most_recent", "action", "logits"])

    def _write_cli_header(self) -> None:
        hdr = ["timestep", "seconds", "fps_mean", "fps_std"]
        hdr.extend(sorted(self.scalars))
        for name in sorted(self.stats):
            if name == "fps": continue
            hdr.extend([f"{name}_most_recent", f"{name}_mean", f"{name}_std"])
        self.cli_writer.writerow(hdr)




# ──────────────────────── Grammar & Batch Parser ─────────────────────────────

class TorchGrammar:
    """*Immutable* container mapping *symbol → rule‑list*."""

    def __init__(self) -> None:
        self.raw_rules: Dict[str, List[Token]] = {}
        self.symbol_to_index: Dict[str, int] = {}
        self.index_to_symbol: Dict[int, str] = {}

    # ------------------------------------------------------------------
    # API – grammar construction helpers.
    # ------------------------------------------------------------------
    def add_object(self, name: str, tokens: List[Token]) -> None:
        """Register **one** grammar object.

        *Side‑effect:* assigns a stable *integer* ID used for GPU storage.
        """
        self.raw_rules[name] = tokens
        if name not in self.symbol_to_index:
            idx = len(self.symbol_to_index)
            self.symbol_to_index[name] = idx
            self.index_to_symbol[idx] = name
        # logger.debug("TorchGrammar.add_object: %s → idx=%d", name, self.symbol_to_index[name])

    # ------------------------------------------------------------------
    # Static helpers – literal expansion & repetition predicate wrapping.
    # ------------------------------------------------------------------
    @staticmethod
    def expand_literal(tok: Token) -> List[int]:
        """Expand a *literal spec* → concrete list of **token IDs**."""
        if isinstance(tok, int):
            return [tok]
        if isinstance(tok, tuple):
            return list(range(tok[0], tok[1] + 1)) if len(tok) == 2 else list(tok)
        #raise ValueError(f"Invalid literal token: {tok!r}")
        raise ValueError()

    @staticmethod
    def wrap_pred(raw: Any) -> Predicate:
        """Return predicate *raw* unchanged if callable, else ``cnt >= raw``."""
        return raw if callable(raw) else (lambda cnt, m=raw: cnt >= m)


class TorchGrammarBatchParser:
    """Incremental **batch** parser operating entirely in *Torch* tensors.

    The parser maintains an explicit **stack** per batch row, enabling efficient
    grammar‑constrained decoding on the GPU. The public surface consists of:

    * :py:meth:`set_batch` – reset internal state for a new batch.
    * :py:meth:`next_token_mask` – produce a *logit mask* compatible with
      softmax‑based models.
    * :py:meth:`fast_is_valid` – validate & consume *one* token per batch row.
    """

    # Default limits - can be overridden by constructor
    DEFAULT_MAX_DEPTH = 16  # stack depth
    DEFAULT_MAX_LEN = 128   # sequence length
    DEFAULT_MAX_NEXT = 32   # cached valid literals

    def __init__(
        self,
        grammar: TorchGrammar,
        batch_size: int,
        *,
        device: str | torch.device = "cuda",
        max_depth: int | None = None,
        max_len: int | None = None,
        max_next: int | None = None,
    ) -> None:
        self.g = grammar
        self.bs = batch_size
        self.dev = torch.device(device)

        # Use provided values or defaults
        self.max_depth = max_depth if max_depth is not None else self.DEFAULT_MAX_DEPTH
        self.max_len = max_len if max_len is not None else self.DEFAULT_MAX_LEN
        self.max_next = max_next if max_next is not None else self.DEFAULT_MAX_NEXT

        # Initialize tensors with dynamic sizes
        self.stack = torch.full((self.bs, self.max_depth, 3), -1, dtype=torch.long, device=self.dev)
        self.ptr = torch.zeros(self.bs, dtype=torch.long, device=self.dev)

        self.seqs = torch.full((self.bs, self.max_len), -1, dtype=torch.long, device=self.dev)
        self.lens = torch.zeros(self.bs, dtype=torch.long, device=self.dev)

        self.next_cache = torch.full((self.bs, self.max_next), -1, dtype=torch.long, device=self.dev)
        self.next_lens = torch.zeros(self.bs, dtype=torch.long, device=self.dev)

    # ------------------------------------------------------------------
    # Batch initialisation.
    # ------------------------------------------------------------------
    def set_batch(self, roots: List[str]) -> None:
        """Reset parser state with new *root objects* (length == ``batch_size``)."""
        for i, name in enumerate(roots):
            idx = self.g.symbol_to_index[name]
            self.stack[i, 0] = torch.tensor([idx, 0, 0], device=self.dev)
            self.ptr[i] = 1
        self._refresh_next()
        # logger.debug("Parser.set_batch: roots=%s", roots)

    # ------------------------------------------------------------------
    # Public inference helpers.
    # ------------------------------------------------------------------
    def fast_is_valid(self, tokens: torch.Tensor) -> torch.Tensor:  # noqa: C901
        """Consume **one** token / batch row and return validity mask (bool)."""
        mask = torch.zeros(self.bs, dtype=torch.bool, device=self.dev)
        for i in range(self.bs):
            if self.ptr[i] == 0:
                continue  # sequence finished

            tok = tokens[i].item()
            n_cached = self.next_lens[i].item()

            if n_cached and (self.next_cache[i, :n_cached] == tok).any():
                mask[i] = True

                # 1) Append token to sequence.
                pos = self.lens[i].item()
                if pos < self.max_len:
                    self.seqs[i, pos] = tok
                    self.lens[i] += 1

                # 2) Update parser stack / repetition counters.
                o_idx, o_ptr, o_rep = self.stack[i, self.ptr[i] - 1].tolist()
                obj_name = self.g.index_to_symbol[o_idx]
                curr_tok = self.g.raw_rules[obj_name][o_ptr]
                self._apply_token(i, curr_tok, tok, o_rep)
        self._refresh_next()
        return mask

    def next_token_mask(self, vocab_size: int, *, fill_value: float = float("-inf")) -> torch.Tensor:
        """Return additive *logit mask* (0 where valid, −∞ elsewhere)."""
        mask = torch.full((self.bs, vocab_size), fill_value, device=self.dev)
        for i in range(self.bs):
            k = self.next_lens[i].item()
            if k:
                mask[i, self.next_cache[i, :k]] = 0.0
        return mask

    def get_sequences(self) -> List[List[int]]:
        """Return *parsed* token IDs (list per batch row)."""
        return [self.seqs[i, : self.lens[i]].tolist() for i in range(self.bs)]

    def decode_sequences(self, id2token: Dict[int, str]) -> List[List[str]]:
        """Convert ID sequences → token strings (helper)."""
        return [[id2token[t] for t in seq] for seq in self.get_sequences()]

    # ------------------------------------------------------------------
    # Internal helpers – token handling & stack book‑keeping.
    # ------------------------------------------------------------------
    def _compute_valid(self, tok: Token, rep_cnt: int, tokens: List[Token], idx: int) -> List[int]:
        """Compute *valid next literals* given current grammar state."""
        if isinstance(tok, dict):
            sub, raw_pred = next(iter(tok.items()))
            pred = self.g.wrap_pred(raw_pred)

            # Case 1 – below repetition threshold → MUST stay inside subobject.
            if not pred(rep_cnt):
                first = self.g.raw_rules[sub][0]
                return self.g.expand_literal(first)

            # Case 2 – threshold met → *exit or repeat* allowed.
            valid: List[int] = []
            # 2a) Exit path: literal following the group (if any).
            if idx + 1 < len(tokens):
                nxt = tokens[idx + 1]
                nxt_lit = nxt if not isinstance(nxt, str) else self.g.raw_rules[nxt][0]
                valid.extend(self.g.expand_literal(nxt_lit))
            # 2b) Repeat path: "first" of subobject.
            first = self.g.raw_rules[sub][0]
            valid.extend(self.g.expand_literal(first))
            return valid

        # Non‑dict token → embedded object or literal.
        if isinstance(tok, str):
            first = self.g.raw_rules[tok][0]
            return self.g.expand_literal(first)
        return self.g.expand_literal(tok)

    def _apply_token(self, i: int, curr: Token, tok: int, rep_cnt: int) -> None:  # noqa: C901
        """Update parser *stack* after accepting ``tok`` for batch row ``i``."""
        if isinstance(curr, dict):
            sub, _ = next(iter(curr.items()))
            first_lits = set(self.g.expand_literal(self.g.raw_rules[sub][0]))
            if tok not in first_lits:
                self._pop_and_advance(i)
                return
            depth = self.ptr[i].item()
            sid = self.g.symbol_to_index[sub]
            self.stack[i, depth] = torch.tensor([sid, 1, 0], device=self.dev)
            self.ptr[i] += 1
            self.stack[i, depth - 1, 2] += 1  # increment repetition counter
        elif isinstance(curr, str):
            depth = self.ptr[i].item()
            sid = self.g.symbol_to_index[curr]
            self.stack[i, depth] = torch.tensor([sid, 1, 0], device=self.dev)
            self.ptr[i] += 1
        else:
            # Plain literal – just advance pointer.
            self.stack[i, self.ptr[i] - 1, 1] += 1
            self.stack[i, self.ptr[i] - 1, 2] = 0

    def _pop_and_advance(self, i: int) -> None:
        """Pop current frame; advance parent pointer if appropriate."""
        self.ptr[i] -= 1
        if self.ptr[i] == 0:
            return
        o_idx, o_ptr, _ = self.stack[i, self.ptr[i] - 1].tolist()
        token = self.g.raw_rules[self.g.index_to_symbol[o_idx]][o_ptr]
        if not isinstance(token, dict):
            self.stack[i, self.ptr[i] - 1, 1] = o_ptr + 1
            self.stack[i, self.ptr[i] - 1, 2] = 0

    def _refresh_next(self) -> None:  # noqa: C901
        """Recompute *next‑token cache* for every batch row."""
        self.next_lens.zero_()
        for i in range(self.bs):
            while self.ptr[i] > 0:
                o_idx, o_ptr, o_rep = self.stack[i, self.ptr[i] - 1].tolist()
                obj_name = self.g.index_to_symbol[o_idx]
                tokens = self.g.raw_rules[obj_name]
                if o_ptr >= len(tokens):
                    self._pop_and_advance(i)
                    continue
                curr = tokens[o_ptr]
                valid = self._compute_valid(curr, o_rep, tokens, o_ptr)
                if valid:
                    k = len(valid)
                    self.next_cache[i, :k] = torch.tensor(valid, device=self.dev)
                    self.next_lens[i] = k
                    break
                self.ptr[i] -= 1


# ───────────────────────────── Model Tester ─────────────────────────────────

class ModelTester:
    """
    Deserializes token streams into a JSON-like structure and, if available,
    scores the resulting model using an external 'json2fr' utility via the ._model module.
    """
    # --- Mappings for token parsing ---
    MASS_MAP = {"MASS_1e0": 1, "MASS_1e1": 10, "MASS_1e2": 100, "MASS_1e3": 1000}
    CHARGE_MAP = {"CHARGE_-1": -1, "CHARGE_0": 0, "CHARGE_1": 1}
    REP_U1Y_MAP = {"REP_U1Y_minus1": -1, "REP_U1Y_0": 0, "REP_U1Y_1": 1}

    # ------------------------------------------------------------------
    # Static helper methods for post-processing tokens
    # ------------------------------------------------------------------
    @staticmethod
    def _after(tok: str, prefix: str) -> str:
        return tok.replace(f"{prefix}_", "")

    @staticmethod
    def _int(tok: str) -> int:
        return int(tok.split("_")[-1])

    # ------------------------------------------------------------------
    # Low-level parsers for model components
    # ------------------------------------------------------------------
    def parse_particle(self, seq: List[str], i: int) -> Tuple[Dict[str, Any], int]:
        """Parses a PARTICLE … END_PARTICLE block."""
        particle = {
            "id": f"p{self._int(seq[i + 1]) + 1}",
            "name": seq[i + 1],
            "type": self._after(seq[i + 2], "TYPE"),
            "mass": self.MASS_MAP.get(seq[i + 3], 0),
            "charge": self.CHARGE_MAP.get(seq[i + 4], 0),
        }
        # Advance index past the particle definition
        i += 5
        # Find the end of the block
        while i < len(seq) and seq[i] != "END_PARTICLE":
            i += 1
        return particle, i + 1

    def parse_field(self, seq: List[str], i: int) -> Tuple[Dict[str, Any], int, List[Dict[str, Any]]]:
        """Parses a FIELD … END_FIELD block."""
        field = {
            "id": f"m{self._int(seq[i + 1]) + 1}",
            "name": seq[i + 1],
            "type": self._after(seq[i + 2], "TYPE"),
            "dim": self._int(seq[i + 3]),
            "gen": self._int(seq[i + 4]),
            "self_conjugate": seq[i + 5] == "SELF_CONJ_TRUE",
            "chirality": None if seq[i + 6] == "CHIRALITY_na" else self._after(seq[i + 6], "CHIRALITY"),
            "reps": {
                "g1": self.REP_U1Y_MAP.get(seq[i + 9], 0),
                "g2": self._after(seq[i + 8], "REP_SU2L"),
                "g3": self._after(seq[i + 7], "REP_SU3C"),
            },
            "QuantumNumber": {
                "LeptonNumber": self._int(seq[i + 10]),
                "BaryonNumber": self._int(seq[i + 11]),
            },
            "particles": [],
        }
        i += 12
        particles_in_field = []
        while i < len(seq) and seq[i] != "END_FIELD":
            if seq[i] == "PARTICLE":
                p, i_after_p = self.parse_particle(seq, i)
                particles_in_field.append(p)
                field["particles"].append(p["id"])
                i = i_after_p # Correctly update index
            else:
                i += 1
        return field, i + 1, particles_in_field

    def parse_interaction(self, seq: List[str], i: int) -> Tuple[Dict[str, Any], int, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parses an ITRACT … END_ITRACT block."""
        inter = {
            "id": f"i{self._int(seq[i + 1]) + 1}",
            "type": self._after(seq[i + 2], "TYPE").lower(),
            "fields": [],
        }
        i += 3
        fields_in_inter, particles_in_inter = [], []
        while i < len(seq) and seq[i] != "END_ITRACT":
            if seq[i] == "FIELD":
                f, i_after_f, ps = self.parse_field(seq, i)
                fields_in_inter.append(f)
                particles_in_inter.extend(ps)
                inter["fields"].append(f["id"])
                i = i_after_f # Correctly update index
            else:
                i += 1
        return inter, i + 1, fields_in_inter, particles_in_inter

    # ------------------------------------------------------------------
    # High-level builder to construct the final model dictionary
    # ------------------------------------------------------------------
    def build_model(self, tokens: List[str]) -> Dict[str, Any]:
        """Convert *flat* token list → structured JSON physics model."""
        particles: List[Dict[str, Any]] = []
        fields: List[Dict[str, Any]] = []
        inters: List[Dict[str, Any]] = []
        i = 0
        while i < len(tokens):
            if tokens[i] == "PARTICLE":
                p, i = self.parse_particle(tokens, i)
                particles.append(p)
            elif tokens[i] == "FIELD":
                f, i, new_ps = self.parse_field(tokens, i)
                fields.append(f)
                particles.extend(new_ps)
            elif tokens[i] == "ITRACT":
                it, i, new_fs, new_ps = self.parse_interaction(tokens, i)
                inters.append(it)
                fields.extend(new_fs)
                particles.extend(new_ps)
            else:
                i += 1
        # Deduplicate by ``id`` (preserve first occurrence).
        particles = list({p["id"]: p for p in particles}.values())
        fields = list({f["id"]: f for f in fields}.values())
        inters = list({it["id"]: it for it in inters}.values())

        return {
            "GaugeGroups": [
                {"id": "g1", "name": "U1Y", "charge": "Y", "group": "U_1", "coupling": "g1", "boson": "B"},
                {"id": "g2", "name": "SU2L", "charge": "I", "group": "SU_2", "coupling": "gw", "boson": "W"},
                {"id": "g3", "name": "SU3C", "charge": "C", "group": "SU_3", "coupling": "gs", "boson": "G"},
            ],
            "vevs": [{"id": "v1", "name": "vev", "vacuum": [0, 1], "value": 246.22}],
            "particles": particles,
            "fields": fields,
            "interactions": inters,
            "links": {"vevs": [{"source": "v1", "target": fields[0]["id"] if fields else ""}]},
        }

    # ------------------------------------------------------------------
    # External tool integration for scoring
    # ------------------------------------------------------------------
    def test_model(self, tokens: List[str]) -> Tuple[float, str]:
        """
        Builds the model dictionary and scores it using the lazily-imported ._model module.
        """
        try:
            model_json_data = self.build_model(tokens)
        except Exception as e:
            # Catch parsing errors, which indicate a malformed sequence.
            # Log for debugging at a lower level and return a score of 0.
            logger.debug(f"Parsing error in build_model for sequence. Error: {e}")
            return 0.0, "<parsing_error>"

        try:
            # LAZY IMPORT: This is the key. The heavy module is imported here,
            # inside the worker function, preventing initialization conflicts.
            from ._model import Model
        except ImportError:
            logger.warning("Scoring module (.model) not found. Returning 0.0 score.")
            return 0.0, "<json2fr_import_error>"
        except Exception as e:
            logger.error(f"An unexpected error occurred during '.model' import: {e}")
            return 0.0, "<model_import_general_error>"

        try:
            # If import succeeds, create a Model instance and get the score
            mdl = Model(
                model_name="Generated Model",
                author=Path(__file__).name,
                model_data_dict=model_json_data
            )
            score_frac = mdl.get_score_fraction()
            return score_frac, "<in-memory-model>"
        except Exception as e:
            logger.error(f"Error creating Model instance or getting score: {e}")
            # Optionally log the JSON that caused the error
            # logger.debug(f"Problematic JSON: {json.dumps(model_json_data, indent=2)}")
            return 0.0, f"<error_scoring: {str(e)[:50]}>"

# ─────────────────────────────────── Main ────────────────────────────────────

if __name__ == "__main__":
    # Example usage of ModelTester
    TEST_TOKENS = [
        "FIELD", 
        "FIELD_ID_0", 
        "TYPE_vector", 
        "DIM_1", 
        "GEN_2", 
        "SELF_CONJ_TRUE", 
        "CHIRALITY_right", 
        "REP_SU3C_adj", 
        "REP_SU2L_adj", 
        "REP_U1Y_0", 
        "QN_LeptonNumber_0", 
        "QN_BaryonNumber_-1",
        "PARTICLE", # This PARTICLE token is *within* the FIELD block
        "PARTICLE_ID_1",
        "TYPE_fermion",
        "MASS_1e1",
        "CHARGE_1",
        "END_PARTICLE",
        "END_FIELD",
        "ITRACT",
        "ITRACT_ID_0",
        "TYPE_YUKAWA",
        "FIELD", # field 1 for Yukawa
        "FIELD_ID_1", 
        "TYPE_fermion", 
        "DIM_1", 
        "GEN_3", 
        "SELF_CONJ_FALSE", 
        "CHIRALITY_left", 
        "REP_SU3C_singlet", 
        "REP_SU2L_fnd", 
        "REP_U1Y_minus1", 
        "QN_LeptonNumber_1", 
        "QN_BaryonNumber_0",
        "PARTICLE",
        "PARTICLE_ID_0",
        "TYPE_fermion",
        "MASS_1e0",
        "CHARGE_0",
        "END_PARTICLE",
        "PARTICLE",
        "PARTICLE_ID_2",
        "TYPE_fermion",
        "MASS_1e0",
        "CHARGE_-1",
        "END_PARTICLE",
        "PARTICLE",
        "PARTICLE_ID_3",
        "TYPE_fermion",
        "MASS_1e0",
        "CHARGE_-1",
        "END_PARTICLE",
        "END_FIELD",
        "FIELD", # field 2 for Yukawa
        "FIELD_ID_2", 
        "TYPE_fermion", 
        "DIM_1", 
        "GEN_3", 
        "SELF_CONJ_FALSE", 
        "CHIRALITY_right", 
        "REP_SU3C_singlet", 
        "REP_SU2L_singlet", 
        "REP_U1Y_0", 
        "QN_LeptonNumber_1", 
        "QN_BaryonNumber_0",
        "PARTICLE",
        "PARTICLE_ID_4",
        "TYPE_fermion",
        "MASS_1e0",
        "CHARGE_0",
        "END_PARTICLE",
        "PARTICLE",
        "PARTICLE_ID_5",
        "TYPE_fermion",
        "MASS_1e0",
        "CHARGE_-1",
        "END_PARTICLE",
        "PARTICLE",
        "PARTICLE_ID_6",
        "TYPE_fermion",
        "MASS_1e0",
        "CHARGE_-1",
        "END_PARTICLE",
        "END_FIELD",
        "FIELD", # field 3 for Yukawa (Scalar Higgs)
        "FIELD_ID_3",
        "TYPE_complex",
        "DIM_2",
        "GEN_1",
        "SELF_CONJ_FALSE",
        "CHIRALITY_na",
        "REP_SU3C_singlet",
        "REP_SU2L_fnd",
        "REP_U1Y_1",
        "QN_LeptonNumber_0",
        "QN_BaryonNumber_0",
        "PARTICLE",
        "PARTICLE_ID_7",
        "TYPE_complex",
        "MASS_1e2",
        "CHARGE_0",
        "END_PARTICLE",
        "PARTICLE",
        "PARTICLE_ID_8",
        "TYPE_complex",
        "MASS_1e2",
        "CHARGE_1",
        "END_PARTICLE",
        "END_FIELD",
        "END_ITRACT"
    ]


    tester = ModelTester()
    score, path = tester.test_model(TEST_TOKENS)
    print("score:", score)
    print("file :", path)