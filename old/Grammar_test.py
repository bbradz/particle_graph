from __future__ import annotations
from typing import Any, Callable, Dict, List, Tuple, Union
import torch
import logging
Token = Union[int, Tuple[int, ...], str, Dict[str, Any]]
Predicate = Callable[[int], bool]
# logger = logging.getLogger(__name__)
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
# )


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
    def _compute_valid(
        self,
        tok: Token,
        rep_cnt: int,
        tokens: List[Token],
        idx: int,
    ) -> List[int]:
        """
        Compute *raw* list of valid next literals for the current *tok*,
        but now: do _not_ allow exit until rep_cnt >= threshold(predicate).
        Once rep_cnt >= threshold, we offer both "exit" and "repeat".
        """
        if isinstance(tok, dict):
            sub, raw_pred = next(iter(tok.items()))
            pred = self.g.wrap_pred(raw_pred)

            # If rep_cnt < threshold, we can only repeat:
            if not pred(rep_cnt):
                # Must stay in the subobject. Only “first of sub” is allowed.
                first = self.g.raw_rules[sub][0]
                valid = self.g.expand_literal(first)
                # logger.debug(
                #     f"_compute_valid (dict): rep_cnt={rep_cnt} < threshold; "
                #     f"only repeat allowed → valid={valid}"
                # )
                return valid

            # If rep_cnt >= threshold, we may either exit or repeat.
            valid: List[int] = []

            # 1) Offer exit path (the literal following this group), if any.
            if idx + 1 < len(tokens):
                nxt = tokens[idx + 1]
                nxt_lit = nxt if not isinstance(nxt, str) else self.g.raw_rules[nxt][0]
                exit_literals = self.g.expand_literal(nxt_lit)
                valid += exit_literals
                # logger.debug(
                #     f"_compute_valid (dict): rep_cnt={rep_cnt} >= threshold; "
                #     f"adding exit_literals={exit_literals}"
                # )

            # 2) Also allow repeating again (i.e., push a new subobject instance).
            first = self.g.raw_rules[sub][0]
            repeat_literals = self.g.expand_literal(first)
            valid += repeat_literals
            # logger.debug(
            #     f"_compute_valid (dict): rep_cnt={rep_cnt} >= threshold; "
            #     f"adding repeat_literals={repeat_literals}"
            # )

            return valid

        # Non‐dict tokens fall back to literal or embedded‐object logic:
        if isinstance(tok, str):
            first = self.g.raw_rules[tok][0]
            valid = self.g.expand_literal(first)
            # logger.debug(
            #     f"_compute_valid (str): embedded object '{tok}', valid={valid}"
            # )
            return valid

        # Simple literal / range → just expand it:
        valid = self.g.expand_literal(tok)
        # logger.debug(f"_compute_valid (literal): tok={tok}, valid={valid}")
        return valid

    # ------------------------------------------------------------------
    def _apply_token(
        self,
        i: int,
        curr: Token,
        tok: int,
        rep_cnt: int,
    ) -> None:
        """
        Advance parser *state* for batch row *i* after accepting *tok*,
        but now: if tok is 'first of subobject', we ALWAYS push (and increment rep_cnt).
        Exiting happens only if tok is not in 'first'.
        """
        if isinstance(curr, dict):
            sub, raw_pred = next(iter(curr.items()))
            first = self.g.raw_rules[sub][0]
            first_literals = set(self.g.expand_literal(first))

            # If the incoming token is _not_ one of the subobject's 'first' literals,
            # we pop & advance (i.e. exit this repetition‑group).
            if tok not in first_literals:
                # logger.debug(
                #     f"_apply_token (dict): tok={tok} not in first_literals={first_literals}, popping..."
                # )
                self._pop_and_advance(i)
                return

            # Otherwise, tok _is_ the first of the subobject: user explicitly wants to enter.
            # Always push a new frame, increment repetition count, and continue.
            depth = self.ptr[i].item()
            sid = self.g.symbol_to_index[sub]
            self.stack[i, depth] = torch.tensor([sid, 1, 0], device=self.dev)
            self.ptr[i] += 1
            # Increment the parent’s repetition counter:
            self.stack[i, depth - 1, 2] += 1

            # logger.debug(
            #     f"_apply_token (dict): tok={tok} ∈ first_literals → pushing sub='{sub}' "
            #     f"(sid={sid}), new rep_cnt={self.stack[i, depth - 1, 2].item()}"
            # )

        elif isinstance(curr, str):  # embedded object literal
            # Behavior unchanged: push embedded object.
            depth = self.ptr[i].item()
            sid = self.g.symbol_to_index[curr]
            self.stack[i, depth] = torch.tensor([sid, 1, 0], device=self.dev)
            self.ptr[i] += 1
            # logger.debug(
            #     f"_apply_token (str): entering embedded object '{curr}' (sid={sid})"
            # )

        else:  # plain literal / range
            # Just advance the pointer in the current frame:
            self.stack[i, self.ptr[i] - 1, 1] += 1
            self.stack[i, self.ptr[i] - 1, 2] = 0
            # logger.debug(
            #     f"_apply_token (literal): consumed literal={tok}, "
            #     f"advancing pointer at depth={self.ptr[i].item() - 1}"
            # )

    # ------------------------------------------------------------------
    def _pop_and_advance(self, i: int) -> None:
        """Pop current stack frame; advance *parent* pointer if needed."""
        self.ptr[i] -= 1
        if self.ptr[i] == 0:
            return  # reached root – nothing further to advance

        o_idx, o_ptr, o_rep = self.stack[i, self.ptr[i] - 1].tolist()
        obj_name = self.g.index_to_symbol[o_idx]
        token    = self.g.raw_rules[obj_name][o_ptr]

        # ⬇️  NEW: if parent token is the repetition-group itself,
        #          do **not** auto-advance; the next token will decide.
        if isinstance(token, dict):
            return

        # Old behaviour for non-dict tokens (plain literals / embedded objects)
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
    g.add_object("Y", [12, 13, {"Z": lambda x: False}, 15])
    g.add_object("X", [1, 2, (3, 5), 6])

    # Manual USE_Z token so that reuse is testable
    g.symbol_to_index["USE_Z_0"] = max(g.symbol_to_index.values()) + 1
    g.index_to_symbol[g.symbol_to_index["USE_Z_0"]] = "USE_Z_0"

    parser = TorchGrammarBatchParser(g, batch_size=4, device="cpu")
    parser.set_batch(["Y", "Y", "X", "Y"])

    tests = [
        [12, 12, 1, 12],
        [13, 13, 2, 13],
        [16, 16, 3, 16],
        [17, 17, 5, 17],
        [16, 16, 6, 15],
        [17, 17, 6, 16],
        [16, 15, 6, 16],
        [15, 15, 6, 16],
    ]
    i = 0
    for toks in tests:
        #if i == 0:
        #print("Next‑token mask:\n", parser.next_token_mask(20))
        #i += 1
        print(parser.fast_is_valid(torch.tensor(toks)))
    print("Sequences:", parser.get_sequences())

    # ------------------------------------------------------------------
    # Particle‑physics demo grammar
    # ------------------------------------------------------------------
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
            {"Particle": lambda c: True},
            TOKEN2ID["END_FIELD"],
        ],
    )
    grammar.add_object(
        "Interaction",
        [
            TOKEN2ID["ITRACT"],
            tr("ITRACT_ID_0", "ITRACT_ID_2"),
            tr("TYPE_DC", "TYPE_FFDUAL"),
            {"Field": lambda c: True},
            TOKEN2ID["END_ITRACT"],
        ],
    )

    grammar.add_object(
        "Model",
        [
            TOKEN2ID["BOS"],
            {"Interaction": lambda c: True},
            TOKEN2ID["EOS"],
        ],
    )

    print("\n—— Integrated Model sanity check ——")
    parser = TorchGrammarBatchParser(grammar, batch_size=1, device="cpu")
    parser.set_batch(["Model"])  # root object

    # Fully‐specified model: BOS, one minimal Interaction, then one rich Interaction, then EOS
    full_model_seq = [
        TOKEN2ID["BOS"],

        # --- minimal Interaction ---
        TOKEN2ID["ITRACT"],
        TOKEN2ID["ITRACT_ID_0"],
        TOKEN2ID["TYPE_DC"],
        TOKEN2ID["END_ITRACT"],

        # --- rich Interaction: start tokens ---
        TOKEN2ID["ITRACT"],
        TOKEN2ID["ITRACT_ID_1"],
        TOKEN2ID["TYPE_YUKAWA"],

        # --- rich Interaction: first Field (field_min minus END_FIELD) ---
        TOKEN2ID["FIELD"],
        TOKEN2ID["FIELD_ID_0"],
        TOKEN2ID["TYPE_complex"],
        TOKEN2ID["DIM_1"],
        TOKEN2ID["GEN_1"],
        TOKEN2ID["SELF_CONJ_TRUE"],
        TOKEN2ID["CHIRALITY_left"],
        TOKEN2ID["REP_SU3C_singlet"],
        TOKEN2ID["REP_SU2L_singlet"],
        TOKEN2ID["REP_U1Y_minus1"],
        TOKEN2ID["QN_LeptonNumber_0"],
        TOKEN2ID["QN_BaryonNumber_0"],

        # --- rich Interaction: embedded Particle inside that Field ---
        TOKEN2ID["PARTICLE"],
        TOKEN2ID["PARTICLE_ID_0"],
        TOKEN2ID["TYPE_vector"],
        TOKEN2ID["MASS_1e2"],
        TOKEN2ID["CHARGE_1"],
        TOKEN2ID["END_PARTICLE"],

        TOKEN2ID["PARTICLE"],
        TOKEN2ID["PARTICLE_ID_1"],
        TOKEN2ID["TYPE_vector"],
        TOKEN2ID["MASS_1e2"],
        TOKEN2ID["CHARGE_1"],
        TOKEN2ID["END_PARTICLE"],

        TOKEN2ID["PARTICLE"],
        TOKEN2ID["PARTICLE_ID_2"],
        TOKEN2ID["TYPE_vector"],
        TOKEN2ID["MASS_1e2"],
        TOKEN2ID["CHARGE_1"],
        TOKEN2ID["END_PARTICLE"],

        # --- close first Field of rich Interaction ---
        TOKEN2ID["END_FIELD"],

        # --- rich Interaction: second Field (full field_min) ---
        TOKEN2ID["FIELD"],
        TOKEN2ID["FIELD_ID_0"],
        TOKEN2ID["TYPE_complex"],
        TOKEN2ID["DIM_1"],
        TOKEN2ID["GEN_1"],
        TOKEN2ID["SELF_CONJ_TRUE"],
        TOKEN2ID["CHIRALITY_left"],
        TOKEN2ID["REP_SU3C_singlet"],
        TOKEN2ID["REP_SU2L_singlet"],
        TOKEN2ID["REP_U1Y_minus1"],
        TOKEN2ID["QN_LeptonNumber_0"],
        TOKEN2ID["QN_BaryonNumber_0"],

        TOKEN2ID["PARTICLE"],
        TOKEN2ID["PARTICLE_ID_0"],
        TOKEN2ID["TYPE_vector"],
        TOKEN2ID["MASS_1e2"],
        TOKEN2ID["CHARGE_1"],
        TOKEN2ID["END_PARTICLE"],

        TOKEN2ID["PARTICLE"],
        TOKEN2ID["PARTICLE_ID_1"],
        TOKEN2ID["TYPE_vector"],
        TOKEN2ID["MASS_1e2"],
        TOKEN2ID["CHARGE_1"],
        TOKEN2ID["END_PARTICLE"],

        TOKEN2ID["PARTICLE"],
        TOKEN2ID["PARTICLE_ID_2"],
        TOKEN2ID["TYPE_vector"],
        TOKEN2ID["MASS_1e2"],
        TOKEN2ID["CHARGE_1"],
        TOKEN2ID["END_PARTICLE"],

        TOKEN2ID["END_FIELD"],

        # --- close rich Interaction ---
        TOKEN2ID["END_ITRACT"],

        TOKEN2ID["EOS"],
    ]

    for tok in full_model_seq:
        ok = parser.fast_is_valid(torch.tensor([tok]))
        assert ok.item(), f"Token {ID2TOK[tok]} was rejected!"
    print("[PASS] Full model accepted.")

    # Finally, print what the parser stored
    print("Sequences:", [ID2TOK[t] for t in parser.get_sequences()[0]])