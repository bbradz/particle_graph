"""torch_grammar_parser.py
================================
Torch‑native **incremental batch grammar parser** with embedded‑object reuse,
GPU support, and convenient masking for Transformer integration.

Key features
------------
* **Literal expansion** – supports single ints, inclusive integer ranges
  expressed as 2‑tuples, and explicit enumerations expressed as n‑tuples
  (``expand_literal``).
* **O(1) next‑token lookup** per batch element via cached "next" arrays.
* **Embedded‑object reuse** – objects may appear repeatedly until their
  repetition predicate evaluates ``True``.
* **Vectorised batch operation** on any CUDA or CPU device.
* **`next_token_mask` utility** – produces an additive mask tensor over the
  full vocabulary (``0`` for allowed tokens, ``‑inf`` for disallowed), ready
  to be added to model logits prior to softmax.

The public API is intentionally minimal:

```python
>>> g   = TorchGrammar()
>>> p   = TorchGrammarBatchParser(g, batch_size=4)
>>> p.set_batch(["Interaction", "Field", "Field", "Particle"])
>>> logits += p.next_token_mask(vocab_size=logits.shape[-1])
>>> valid  = p.fast_is_valid(predicted_token_ids)
```

All state updates occur inside :py:meth:`fast_is_valid`; call that once per
step with the model's chosen token for each batch element.  Mask generation
should be performed *after* every state update to reflect the new parser
state.
"""

from __future__ import annotations

# ───────────────────────────── Imports ──────────────────────────────
from typing import Any, Callable, Dict, List, Tuple, Union
import torch

# ────────────────────────────── Typing ──────────────────────────────
Token = Union[int, Tuple[int, ...], str, Dict[str, Any]]
Predicate = Callable[[int], bool]


# ====================================================================
#                              Grammar
# ====================================================================
class TorchGrammar:
    """Container for **static** grammar information.

    A grammar is simply a mapping from *object names* (strings) to a list of
    *tokens* that define that object's expansion.  Tokens can be:

    * ``int`` – literal token ID.
    * ``tuple`` – if *len == 2*⇢ inclusive range ; if *len > 2*⇢ explicit set.
    * ``str`` – name of another object (embedded object).
    * ``dict`` – ``{sub_object_name: repetition_predicate}`` allowing repeated
      sub‑objects until the predicate on the current repeat count returns
      ``True``.
    """

    # ------------------------------------------------------------------
    def __init__(self) -> None:
        self.raw_rules: Dict[str, List[Token]] = {}
        self.symbol_to_index: Dict[str, int] = {}
        self.index_to_symbol: Dict[int, str] = {}

    # ------------------------------------------------------------------
    def add_object(self, name: str, tokens: List[Token]) -> None:
        """Register a new *grammar object*.

        The object becomes addressable both by *name* (string) and by a stable
        integer index (used internally for fast tensor ops).
        """
        self.raw_rules[name] = tokens
        if name not in self.symbol_to_index:
            idx = len(self.symbol_to_index)
            self.symbol_to_index[name] = idx
            self.index_to_symbol[idx] = name

    # ------------------------------------------------------------------
    @staticmethod
    def expand_literal(tok: Token) -> List[int]:
        """Convert a *literal‑spec* token into a concrete **list of IDs**.

        * ``int`` → ``[tok]``
        * ``(lo, hi)`` → inclusive range ``[lo, …, hi]``
        * ``(a, b, c, …)`` with *len > 2* → explicit enumeration ``[a, b, c, …]``
        """
        if isinstance(tok, int):
            return [tok]
        if isinstance(tok, tuple):
            return list(range(tok[0], tok[1] + 1)) if len(tok) == 2 else list(tok)
        raise ValueError(f"Invalid literal token: {tok!r}")

    # ------------------------------------------------------------------
    @staticmethod
    def wrap_pred(raw: Any) -> Predicate:  # noqa: D401  (simple lambda wrapper)
        """Return *raw* unchanged if callable, else build «cnt >= raw» predicate."""
        return raw if callable(raw) else lambda cnt, m=raw: cnt >= m


# ====================================================================
#                          Batch‑state Parser
# ====================================================================
class TorchGrammarBatchParser:
    """Incremental **batch** parser operating entirely in Torch tensors."""

    # ------------------------------------------------------------------
    def __init__(self, grammar: TorchGrammar, batch_size: int, *, device: str | torch.device = "cuda") -> None:
        self.g = grammar
        self.bs = batch_size
        self.dev = torch.device(device)

        # Tunable limits – adjust if your grammars/sequences are larger.
        self.max_d = 16   # max stack depth
        self.max_l = 128  # max sequence length
        self.max_n = 32   # max cache length for next‑token candidates

        # Runtime state – all torch tensors for speed / GPU usage.
        self.stack = torch.full((self.bs, self.max_d, 3), -1, dtype=torch.long, device=self.dev)
        self.ptr   = torch.zeros(self.bs, dtype=torch.long, device=self.dev)  # depth pointer per batch

        self.seqs  = torch.full((self.bs, self.max_l), -1, dtype=torch.long, device=self.dev)
        self.lens  = torch.zeros(self.bs, dtype=torch.long, device=self.dev)

        self.next_cache = torch.full((self.bs, self.max_n), -1, dtype=torch.long, device=self.dev)
        self.next_lens  = torch.zeros(self.bs, dtype=torch.long, device=self.dev)

    # ================================================================
    #                         Public Interface
    # ================================================================
    def set_batch(self, root_objects: List[str]) -> None:
        """(Re‑)initialise batch with the given *root_objects* (one per row)."""
        for i, name in enumerate(root_objects):
            idx = self.g.symbol_to_index[name]
            self.stack[i, 0] = torch.tensor([idx, 0, 0], device=self.dev)
            self.ptr[i] = 1
        self._refresh_next()

    # ------------------------------------------------------------------
    def fast_is_valid(self, tokens: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Consume **one** token per batch row and return a boolean *validity* mask.

        ``tokens`` must be a 1‑D tensor of length ``batch_size``.
        Parser state & *next‑token cache* are updated in‑place where valid.
        """
        mask = torch.zeros(self.bs, dtype=torch.bool, device=self.dev)

        for i in range(self.bs):
            if self.ptr[i] == 0:
                continue  # sequence finished – nothing valid any more

            tok      = tokens[i].item()
            n_cached = self.next_lens[i].item()

            if n_cached and (self.next_cache[i, :n_cached] == tok).any():
                mask[i] = True

                # ── 1.  Append token to sequence ─────────────────────
                pos = self.lens[i].item()
                if pos < self.max_l:
                    self.seqs[i, pos] = tok
                    self.lens[i] += 1

                # ── 2.  Update parser stack / repetition counters ────
                o_idx, o_ptr, o_rep = self.stack[i, self.ptr[i] - 1].tolist()
                obj_name = self.g.index_to_symbol[o_idx]
                curr_tok = self.g.raw_rules[obj_name][o_ptr]
                self._apply_token(i, curr_tok, tok, o_rep)

        self._refresh_next()
        return mask

    # ------------------------------------------------------------------
    def next_token_mask(self, vocab_size: int, *, fill_value: float = float("-inf")) -> torch.Tensor:
        """Return an **additive logit mask** of shape ``(batch, vocab_size)``.

        Entries corresponding to *valid* next tokens are set to ``0``; all
        others are filled with ``fill_value`` (default ``‑inf``).  Designed to
        be added to model logits *before* the softmax.
        """
        mask = torch.full((self.bs, vocab_size), fill_value, device=self.dev)
        for i in range(self.bs):
            k = self.next_lens[i].item()
            if k:
                mask[i, self.next_cache[i, :k]] = 0.0
        return mask

    # ------------------------------------------------------------------
    def get_sequences(self) -> List[List[int]]:
        """Return parsed token IDs for each batch element (list of lists)."""
        return [self.seqs[i, : self.lens[i]].tolist() for i in range(self.bs)]

    # ------------------------------------------------------------------
    def decode_sequences(self, id2token: Dict[int, str]) -> List[List[str]]:
        """Convert :py:meth:`get_sequences` output back to token strings."""
        return [[id2token[t] for t in seq] for seq in self.get_sequences()]

    # ================================================================
    #                       Internal Helper Logic
    # ================================================================
    def _compute_valid(self, tok: Token, rep_cnt: int, tokens: List[Token], idx: int) -> List[int]:
        """Compute *raw* list of valid next literals for the current *tok*."""
        if isinstance(tok, dict):
            valid: List[int] = []

            # 1) Exit path – the literal following this group (if any).
            if idx + 1 < len(tokens):
                nxt = tokens[idx + 1]
                nxt_lit = nxt if not isinstance(nxt, str) else self.g.raw_rules[nxt][0]
                valid += self.g.expand_literal(nxt_lit)

            # 2) Repeat path – re‑enter sub‑object until predicate satisfied.
            sub, raw_pred = next(iter(tok.items()))
            if not self.g.wrap_pred(raw_pred)(rep_cnt):
                first = self.g.raw_rules[sub][0]
                valid += self.g.expand_literal(first)
            return valid

        if isinstance(tok, str):  # embedded object literal
            first = self.g.raw_rules[tok][0]
            return self.g.expand_literal(first)

        # Simple literal / range.
        return self.g.expand_literal(tok)

    # ------------------------------------------------------------------
    def _apply_token(self, i: int, curr: Token, tok: int, rep_cnt: int) -> None:
        """Advance parser *state* for batch row *i* after accepting *tok*."""
        if isinstance(curr, dict):
            sub, raw_pred = next(iter(curr.items()))
            pred  = self.g.wrap_pred(raw_pred)
            first = self.g.raw_rules[sub][0]

            if tok not in self.g.expand_literal(first):  # ⇢ exited sub‑object
                self._pop_and_advance(i)
                return

            if not pred(rep_cnt):  # ⇢ enter new sub‑object instance
                depth = self.ptr[i].item()
                sid   = self.g.symbol_to_index[sub]
                self.stack[i, depth] = torch.tensor([sid, 1, 0], device=self.dev)
                self.ptr[i] += 1
                self.stack[i, depth - 1, 2] += 1  # increment repetition counter
            else:  # literal following group consumed
                self.stack[i, self.ptr[i] - 1, 1] += 1
                self.stack[i, self.ptr[i] - 1, 2]  = 0

        elif isinstance(curr, str):  # embedded object literal
            depth = self.ptr[i].item()
            sid   = self.g.symbol_to_index[curr]
            self.stack[i, depth] = torch.tensor([sid, 0, 0], device=self.dev)
            self.ptr[i] += 1

        else:  # plain literal / range
            self.stack[i, self.ptr[i] - 1, 1] += 1
            self.stack[i, self.ptr[i] - 1, 2]  = 0

    # ------------------------------------------------------------------
    def _pop_and_advance(self, i: int) -> None:
        """Pop current stack frame; advance *parent* pointer if needed."""
        self.ptr[i] -= 1
        if self.ptr[i] == 0:
            return  # reached root – nothing further to advance

        o_idx, o_ptr, o_rep = self.stack[i, self.ptr[i] - 1].tolist()
        obj_name = self.g.index_to_symbol[o_idx]
        token    = self.g.raw_rules[obj_name][o_ptr]

        if isinstance(token, dict):
            sub, raw_pred = next(iter(token.items()))
            if not self.g.wrap_pred(raw_pred)(o_rep):
                return  # may repeat again – do *not* advance pointer yet

        # Advance to *next* token in parent object.
        self.stack[i, self.ptr[i] - 1, 1] = o_ptr + 1
        self.stack[i, self.ptr[i] - 1, 2] = 0

    # ------------------------------------------------------------------
    def _refresh_next(self) -> None:
        """Recompute *next‑token cache* for **all** batch elements."""
        self.next_lens.zero_()

        for i in range(self.bs):
            while self.ptr[i] > 0:
                o_idx, o_ptr, o_rep = self.stack[i, self.ptr[i] - 1].tolist()
                obj_name = self.g.index_to_symbol[o_idx]
                tokens   = self.g.raw_rules[obj_name]

                if o_ptr >= len(tokens):  # end of object – pop & continue
                    self._pop_and_advance(i)
                    continue

                curr = tokens[o_ptr]
                valid = self._compute_valid(curr, o_rep, tokens, o_ptr)

                if valid:
                    k = len(valid)
                    self.next_cache[i, :k] = torch.tensor(valid, device=self.dev)
                    self.next_lens[i]      = k
                    break  # found at least one valid literal

                # No valid literals at this depth – pop & retry higher frame.
                self.ptr[i] -= 1


# ====================================================================
#                               Tests
# ====================================================================
if __name__ == "__main__":  # pragma: no cover
    print("—— Abstract reuse‑grammar sanity check ——")
    g = TorchGrammar()
    g.add_object("Z", [16, 17])
    g.add_object("Y", [12, 13, {"Z": lambda x: x >= 2}, 15])
    g.add_object("X", [1, 2, (3, 5), 6])

    # Manual USE_Z token so that reuse is testable
    g.symbol_to_index["USE_Z_0"] = max(g.symbol_to_index.values()) + 1
    g.index_to_symbol[g.symbol_to_index["USE_Z_0"]] = "USE_Z_0"

    parser = TorchGrammarBatchParser(g, batch_size=3, device="cpu")
    parser.set_batch(["Y", "X", "Y"])

    tests = [
        [12, 1, 12],
        [13, 2, 13],
        [16, 3, 16],
        [17, 5, 17],
        [16, 6, 15],
        [17, 6, 16],
        [16, 6, 16],
        [15, 6, 16],
    ]
    for toks in tests:
        print(parser.fast_is_valid(torch.tensor(toks)))

    print("Sequences:", parser.get_sequences())
    print("Next‑token mask (vocab ≤ 20):\n", parser.next_token_mask(20))

    # ------------------------------------------------------------------
    # Particle‑physics demo grammar
    # ------------------------------------------------------------------
    MAX_ITRACTS, MAX_FIELDS, MAX_PARTICLE = 3, 3, 3
    TOKENS: List[str] = (
        ["BOS", "EOS"]
        + ["ITRACT", "END_ITRACT"]
        + [f"ITRACT_ID_{i}" for i in range(MAX_ITRACTS)]
        + ["FIELD", "END_FIELD"]
        + [f"FIELD_ID_{i}" for i in range(MAX_FIELDS)]
        + [f"USE_FIELD_{i}" for i in range(MAX_FIELDS)]
        + ["PARTICLE", "END_PARTICLE"]
        + [f"PARTICLE_ID_{i}" for i in range(MAX_PARTICLE)]
        + [f"USE_PARTICLE_{i}" for i in range(MAX_PARTICLE)]
        + [
            "TYPE_DC","TYPE_YUKAWA","TYPE_VLF","TYPE_PHI4","TYPE_FF","TYPE_FFDUAL",
            "TYPE_complex","TYPE_real","TYPE_fermion","TYPE_vector",
            "MASS_1e0","MASS_1e1","MASS_1e2","MASS_1e3",
            "CHARGE_-1","CHARGE_0","CHARGE_1",
            "DIM_1","DIM_2","DIM_3",
            "GEN_1","GEN_2","GEN_3",
            "SELF_CONJ_TRUE","SELF_CONJ_FALSE",
            "CHIRALITY_left","CHIRALITY_right","CHIRALITY_na",
            "REP_SU3C_singlet","REP_SU3C_fnd","REP_SU3C_adj",
            "REP_SU2L_singlet","REP_SU2L_fnd","REP_SU2L_adj"
        ]
    )