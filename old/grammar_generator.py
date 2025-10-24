# grammar_generator.py
"""
Grammar-Constrained Sequence Generation and Reward Calculation

A trainer class for generating token sequences that adhere to predefined grammatical rules,
particularly for physics model descriptions (interactions, fields, particles). Features:

- Dynamic token generation with unique identifiers (ITRACT_ID_0, FIELD_ID_5, etc.)
- Layered grammar system with context-aware rule switching
- O(1) legality masks for efficient token validation
- Autoregressive generation with balanced structure enforcement
- Discounted survival rewards for reinforcement learning
- Model-agnostic design compatible with any torch.nn.Module

Example Usage:
    trainer = GrammarModelTrainer(device, validation_bonus=10.0, debug=True)
    sequences = trainer.generate_sequences(model, batch_size=16, seq_len=60)
    rewards = trainer.calculate_rewards_with_validation(sequences, gamma=0.99, len_target=349, len_sigma=50.0)
"""

import time
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Any, Tuple
import logging

# Import the refactored scoring and validation functions
from score_calculator import ingest_batch, parallel_validator_and_scorer, DEVICE, DTYPE, PAD_VALUE

# Configure a basic logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import multiprocessing

def parse_model_worker(token_sequence: List[str]) -> Dict[str, Any]:
    """A helper function for multiprocessing to call the parser."""
    # Each worker process gets its own parser instance.
    parser = TokenSequenceParser()
    return parser.build_model(token_sequence)

# ======================================================================================
# Phase 4: Token Sequence to Dictionary Parser (Adapted from ModelTester)
# ======================================================================================
class TokenSequenceParser:
    """
    Deserializes token streams into a JSON-like structure for the validator.
    """
    MASS_MAP = {"MASS_1e0": 1, "MASS_1e1": 10, "MASS_1e2": 100, "MASS_1e3": 1000}
    CHARGE_MAP = {"CHARGE_-1": -1, "CHARGE_0": 0, "CHARGE_1": 1}

    def __init__(self, debug: bool = False):
        self.debug = debug

    @staticmethod
    def _after(tok: str, prefix: str) -> str: return tok.replace(f"{prefix}_", "")
    @staticmethod
    def _int(tok: str) -> int: return int(tok.split("_")[-1])

    def parse_particle(self, seq: List[str], i: int) -> Tuple[Dict[str, Any], int]:
        """Parses a PARTICLE … END_PARTICLE block."""
        particle = {
            "id": f"p{self._int(seq[i + 1])}",
            "type": self._after(seq[i + 2], "TYPE"),
            "mass": self.MASS_MAP.get(seq[i + 3], 0),
            "charge": self.CHARGE_MAP.get(seq[i + 4], 0),
        }
        i += 5
        while i < len(seq) and seq[i] != "END_PARTICLE": i += 1
        return particle, i + 1

    def parse_field(self, seq: List[str], i: int) -> Tuple[Dict[str, Any], int, List[Dict[str, Any]]]:
        """Parses a FIELD … END_FIELD block."""
        field = {
            "id": f"f{self._int(seq[i + 1])}",
            "type": self._after(seq[i + 2], "TYPE"),
            "dim": self._int(seq[i + 3]),
            "gen": self._int(seq[i + 4]),
            "self_conjugate": seq[i + 5] == "SELF_CONJ_TRUE",
            "chirality": None if seq[i + 6] == "CHIRALITY_na" else self._after(seq[i + 6], "CHIRALITY"),
            "particles": [],
        }
        i += 12 # Skip to where particles might start
        particles_in_field = []
        while i < len(seq) and seq[i] != "END_FIELD":
            if seq[i] == "PARTICLE":
                p, i_after_p = self.parse_particle(seq, i)
                particles_in_field.append(p)
                field["particles"].append(p["id"])
                i = i_after_p
            else:
                i += 1
        return field, i + 1, particles_in_field

    def parse_interaction(self, seq: List[str], i: int) -> Tuple[Dict[str, Any], int, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parses an ITRACT … END_ITRACT block."""
        inter = {
            "id": f"i{self._int(seq[i + 1])}",
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
                i = i_after_f
            else:
                i += 1
        return inter, i + 1, fields_in_inter, particles_in_inter

    def build_model(self, tokens: List[str]) -> Dict[str, Any]:
        """Convert *flat* token list → structured JSON physics model."""
        particles, fields, inters = [], [], []
        i = 0
        try:
            while i < len(tokens):
                if tokens[i] == "ITRACT":
                    it, i, new_fs, new_ps = self.parse_interaction(tokens, i)
                    inters.append(it)
                    fields.extend(new_fs)
                    particles.extend(new_ps)
                else:
                    i += 1
            particles = list({p["id"]: p for p in particles}.values())
            fields = list({f["id"]: f for f in fields}.values())
            inters = list({it["id"]: it for it in inters}.values())
            return {
                "GaugeGroups": [{"id": "g1", "group": "U_1"}, {"id": "g2", "group": "SU_2"}, {"id": "g3", "group": "SU_3"}],
                "particles": particles, "fields": fields, "interactions": inters,
            }
        except (IndexError, ValueError, KeyError) as e:
            logger.debug(f"Parsing error in build_model for sequence. Error: {e}")
            return {}

# ======================================================================================
# Phase 5: Grammar Definition and Generator (Adapted from GrammarModelTrainer)
# ======================================================================================
def generate_id_tokens(prefix: str, count: int) -> List[str]: return [f"{prefix}_{i}" for i in range(count)]
NUM_IDS = 10
ITRACT_IDS, FIELD_IDS, PARTICLE_IDS = (generate_id_tokens(p, NUM_IDS) for p in ["ITRACT_ID", "FIELD_ID", "PARTICLE_ID"])
ALL_TOKEN_NAMES: List[str] = (["BOS", "ITRACT"] + ITRACT_IDS + ["TYPE_YUKAWA", "FIELD"] + FIELD_IDS + ["TYPE_complex", "TYPE_real", "TYPE_fermion", "DIM_1", "DIM_2", "DIM_3", "GEN_1", "GEN_2", "GEN_3", "SELF_CONJ_TRUE", "SELF_CONJ_FALSE", "CHIRALITY_left", "CHIRALITY_right", "CHIRALITY_na", "REP_SU3C_singlet", "REP_SU3C_fnd", "REP_SU2L_singlet", "REP_SU2L_fnd", "REP_U1Y_0", "QN_LeptonNumber_0", "QN_BaryonNumber_0", "PARTICLE"] + PARTICLE_IDS + ["MASS_1e2", "CHARGE_0", "END_PARTICLE", "END_FIELD", "END_ITRACT", "EOS", "DEAD"])
TOKENS_MODEL = {"BOS": ["ITRACT"], "END_ITRACT": ["ITRACT", "EOS"], "EOS": ["EOS"], "DEAD": ["DEAD"]}
TOKENS_INTERACTION = {"ITRACT": ITRACT_IDS, "TYPE_YUKAWA": ["FIELD"], "END_FIELD": ["FIELD", "END_ITRACT"], "DEAD": ["DEAD"]}
for it_id in ITRACT_IDS: TOKENS_INTERACTION[it_id] = ["TYPE_YUKAWA"]
TOKENS_FIELD = {"FIELD": FIELD_IDS, "TYPE_complex": ["DIM_1", "DIM_2", "DIM_3"], "TYPE_real": ["DIM_1", "DIM_2", "DIM_3"], "TYPE_fermion": ["DIM_1", "DIM_2", "DIM_3"], "DIM_1": ["GEN_1", "GEN_2", "GEN_3"], "DIM_2": ["GEN_1", "GEN_2", "GEN_3"], "DIM_3": ["GEN_1", "GEN_2", "GEN_3"], "GEN_1": ["SELF_CONJ_FALSE"], "GEN_2": ["SELF_CONJ_FALSE"], "GEN_3": ["SELF_CONJ_FALSE"], "SELF_CONJ_FALSE": ["CHIRALITY_left", "CHIRALITY_right", "CHIRALITY_na"], "CHIRALITY_na": ["REP_SU3C_singlet", "REP_SU3C_fnd"], "CHIRALITY_left": ["REP_SU3C_singlet", "REP_SU3C_fnd"], "CHIRALITY_right": ["REP_SU3C_singlet", "REP_SU3C_fnd"], "REP_SU3C_singlet": ["REP_SU2L_singlet", "REP_SU2L_fnd"], "REP_SU3C_fnd": ["REP_SU2L_singlet", "REP_SU2L_fnd"], "REP_SU2L_singlet": ["REP_U1Y_0"], "REP_SU2L_fnd": ["REP_U1Y_0"], "REP_U1Y_0": ["QN_LeptonNumber_0"], "QN_LeptonNumber_0": ["QN_BaryonNumber_0"], "QN_BaryonNumber_0": ["PARTICLE"], "END_PARTICLE": ["PARTICLE", "END_FIELD"], "DEAD": ["DEAD"]}
for f_id in FIELD_IDS: TOKENS_FIELD[f_id] = ["TYPE_complex", "TYPE_real", "TYPE_fermion"]
TOKENS_PARTICLE = {"PARTICLE": PARTICLE_IDS, "TYPE_complex": ["MASS_1e2"], "TYPE_real": ["MASS_1e2"], "TYPE_fermion": ["MASS_1e2"], "MASS_1e2": ["CHARGE_0"], "CHARGE_0": ["END_PARTICLE"], "DEAD": ["DEAD"]}
for p_id in PARTICLE_IDS: TOKENS_PARTICLE[p_id] = ["TYPE_complex", "TYPE_real", "TYPE_fermion"]
EMBED_VAL_MAP: Dict[str, int] = {"ITRACT": 0b1000, "END_ITRACT": -0b1000, "FIELD": 0b0100, "END_FIELD": -0b0100, "PARTICLE": 0b0010, "END_PARTICLE": -0b0010}
INTERACTION_BIT, FIELD_BIT, PARTICLE_BIT = 0b1000, 0b0100, 0b0010

class GrammarModelTrainer:
    def __init__(self, device: torch.device, validation_bonus: float, debug: bool = False):
        self.device = device
        self.validation_bonus = validation_bonus
        self.debug = debug
        self.token_to_idx: Dict[str, int] = {token: i for i, token in enumerate(ALL_TOKEN_NAMES)}
        self.idx_to_token: Dict[int, str] = {i: token for token, i in self.token_to_idx.items()}
        self.vocab_size = len(ALL_TOKEN_NAMES)
        self.bos_token_idx, self.eos_token_idx, self.dead_token_idx, self.itract_token_idx = (self.token_to_idx[k] for k in ["BOS", "EOS", "DEAD", "ITRACT"])
        self.parser = TokenSequenceParser(debug=self.debug)
        self.legality_tensors = self._build_legality_tensors()
        self.embed_val_tensor = self._build_embed_val_tensor()
        # OPTIMIZATION: Create a long-lived multiprocessing pool to avoid process spin-up/tear-down overhead.
        self.pool = multiprocessing.Pool()
        # OPTIMIZATION: Cache for discount tensors to avoid re-creation.
        self.discounts_cache = {}


    def __del__(self):
        """Ensure the multiprocessing pool is properly closed when the trainer is garbage collected."""
        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def _build_legality_tensors(self) -> Dict[str, torch.Tensor]:
        tensors = {}
        for token in ALL_TOKEN_NAMES:
            for ruleset in [TOKENS_MODEL, TOKENS_INTERACTION, TOKENS_FIELD, TOKENS_PARTICLE]:
                ruleset.setdefault(token, ["DEAD"])
        for name, rules in {"MODEL": TOKENS_MODEL, "INTERACTION": TOKENS_INTERACTION, "FIELD": TOKENS_FIELD, "PARTICLE": TOKENS_PARTICLE}.items():
            matrix = torch.zeros((self.vocab_size, self.vocab_size), dtype=torch.bool)
            for prev_tok, next_toks in rules.items():
                if prev_tok in self.token_to_idx:
                    prev_idx = self.token_to_idx[prev_tok]
                    next_indices = [self.token_to_idx[n] for n in next_toks if n in self.token_to_idx]
                    if next_indices: matrix[prev_idx, next_indices] = True
            tensors[name] = matrix.to(self.device)
        return tensors

    def _build_embed_val_tensor(self) -> torch.Tensor:
        embed_vals = torch.zeros(self.vocab_size, dtype=torch.long)
        for token, value in EMBED_VAL_MAP.items():
            embed_vals[self.token_to_idx[token]] = value
        return embed_vals.to(self.device)

    def get_active_grammar_mask(self, embed_vals: torch.Tensor, prev_tokens: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros((embed_vals.shape[0], self.vocab_size), dtype=torch.bool, device=self.device)
        is_particle = (embed_vals & PARTICLE_BIT) != 0
        is_field = (~is_particle) & ((embed_vals & FIELD_BIT) != 0)
        is_interaction = (~is_particle & ~is_field) & ((embed_vals & INTERACTION_BIT) != 0)
        is_model = (~is_particle & ~is_field & ~is_interaction)
        
        # This logic correctly handles the 1D inputs from generate_sequences
        mask[is_particle] = self.legality_tensors["PARTICLE"][prev_tokens[is_particle]]
        mask[is_field] = self.legality_tensors["FIELD"][prev_tokens[is_field]]
        mask[is_interaction] = self.legality_tensors["INTERACTION"][prev_tokens[is_interaction]]
        mask[is_model] = self.legality_tensors["MODEL"][prev_tokens[is_model]]
        return mask

    def get_active_grammar_mask_vectorized(self, embed_vals: torch.Tensor, prev_tokens: torch.Tensor) -> torch.Tensor:
        B, T = embed_vals.shape
        V = self.vocab_size

        # Determine the grammar state (mode) for each position in the batch
        is_particle = (embed_vals & PARTICLE_BIT) != 0
        is_field = (~is_particle) & ((embed_vals & FIELD_BIT) != 0)
        is_interaction = (~is_particle & ~is_field) & ((embed_vals & INTERACTION_BIT) != 0)
        is_model = (~is_particle & ~is_field & ~is_interaction)

        # Flatten inputs to process all B*T tokens at once
        flat_prev_tokens = prev_tokens.view(-1)

        # Gather the next legal token rules for every token
        particle_legality = self.legality_tensors["PARTICLE"][flat_prev_tokens]
        field_legality = self.legality_tensors["FIELD"][flat_prev_tokens]
        interaction_legality = self.legality_tensors["INTERACTION"][flat_prev_tokens]
        model_legality = self.legality_tensors["MODEL"][flat_prev_tokens]

        # Use the actual grammar state masks to select the correct rule for each token
        flat_is_particle = is_particle.view(-1, 1).float()
        flat_is_field = is_field.view(-1, 1).float()
        flat_is_interaction = is_interaction.view(-1, 1).float()
        flat_is_model = is_model.view(-1, 1).float()

        # Combine the rulebooks using the masks
        final_mask_flat = (particle_legality * flat_is_particle +
                           field_legality * flat_is_field +
                           interaction_legality * flat_is_interaction +
                           model_legality * flat_is_model).bool()

        return final_mask_flat.view(B, T, V)

    def generate_sequences(self, model: nn.Module, batch_size: int, seq_len: int) -> torch.Tensor:
        sequences = torch.full((batch_size, seq_len), self.dead_token_idx, dtype=torch.long, device=self.device)
        sequences[:, 0] = self.bos_token_idx
        embed_vals = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        is_successful = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        itract_counts = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        cache = None
        current_tokens = sequences[:, 0]
        for i in range(1, seq_len):
            if is_finished.all(): 
                logger.debug(f"[Generator] All sequences finished at step {i}. Breaking early.")
                break
            if self.debug and i % 10 == 0:
                logger.debug(f"[Generator] Generating token {i}/{seq_len}. {is_finished.sum().item()}/{batch_size} sequences finished.")
            
            logits, _, cache = model(current_tokens.unsqueeze(1), cache)
            logits = logits.squeeze(1)

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                if self.debug:
                    logger.warning(f"[Generator] NaN or Inf detected in raw logits. Sanitizing.")
                neg_inf = torch.finfo(logits.dtype).min
                logits = torch.nan_to_num(logits, nan=neg_inf, posinf=neg_inf, neginf=neg_inf)

            # ── Build a *combined* mask: grammar ∧ stack-ok ────────────────────
            grammar_ok = self.get_active_grammar_mask(embed_vals, current_tokens)
            grammar_ok[itract_counts < 2, self.eos_token_idx] = False

            stack_ok = (embed_vals[:, None] + self.embed_val_tensor >= 0)
            final_mask = grammar_ok & stack_ok

            no_legal_tokens_mask = ~final_mask.any(dim=1)
            if no_legal_tokens_mask.any():
                # If there are no legal moves, the only legal move becomes DEAD.
                final_mask[no_legal_tokens_mask] = False
                final_mask[no_legal_tokens_mask, self.dead_token_idx] = True
            
            # 2. Hard-mask the logits *before* soft-max.
            neg_inf = torch.finfo(logits.dtype).min
            masked_logits = logits.masked_fill(~final_mask, neg_inf)
            probs = torch.softmax(masked_logits, dim=-1).nan_to_num_(0.0)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # 3. Check if the model's chosen token is actually legal.
            is_grammatically_legal = final_mask.gather(1, next_tokens.unsqueeze(1)).squeeze(1)
            does_not_overflow_stack = (embed_vals + self.embed_val_tensor[next_tokens] >= 0)
            is_choice_acceptable = is_grammatically_legal & does_not_overflow_stack

            # 4. Determine the token to place based on the choice's validity.
            # Token to use for padding if the sequence has already finished.
            pad_token = torch.where(is_successful, self.eos_token_idx, self.dead_token_idx)
            # Token to use if the sequence is still running: the model's choice if acceptable, otherwise DEAD.
            running_token = torch.where(is_choice_acceptable, next_tokens, self.dead_token_idx)
            # Final decision: if sequence was already finished, use pad_token. Else, use running_token.
            current_tokens = torch.where(is_finished, pad_token, running_token)
            
            sequences[:, i] = current_tokens
            embed_vals += self.embed_val_tensor[current_tokens]
            itract_counts += (current_tokens == self.itract_token_idx).long()
            
            # Update state flags based on the token that was actually placed.
            is_successful |= (current_tokens == self.eos_token_idx)
            is_finished |= (current_tokens == self.dead_token_idx) | (current_tokens == self.eos_token_idx)
        return sequences

    def calculate_rewards_with_validation(
            self,
            sequences: torch.Tensor,
            *,
            gamma: float = 0.997,
            alive_r: float = 0.05,        #   NEW: tiny per-token reward
            len_target: int = 349,        # NEW
            len_sigma: float = 50.0       # NEW – width of the peak
    ) -> torch.Tensor:
        """
        ① discounted alive-time reward (small)
        ② length-shaping bonus (NEW)
        ③ big terminal bonus ∝ physical-validity
        ④ no normalisation – final scale matters for PPO
        """
        B, T = sequences.shape
        is_alive = (sequences != self.dead_token_idx)

        # ①  tiny alive-time reward
        # OPTIMIZATION: Cache the discounts tensor to avoid re-computing it on every call.
        if T in self.discounts_cache and self.discounts_cache[T].device == self.device:
            discounts = self.discounts_cache[T]
        else:
            discounts = gamma ** torch.arange(T, device=self.device, dtype=torch.float32)
            self.discounts_cache[T] = discounts
        alive_reward = (is_alive.float() * discounts).flip(1).cumsum(1).flip(1) * alive_r

        # ②  length-shaping bonus (NEW)
        length = is_alive.float().sum(1)
        # Ensure len_target and len_sigma are the right types (handle tuple case)
        len_target = float(len_target[0] if isinstance(len_target, tuple) else len_target)
        len_sigma = float(len_sigma[0] if isinstance(len_sigma, tuple) else len_sigma)
        # Gaussian bump centred on the desired length
        len_bonus = torch.exp(-0.5*((length - len_target)/len_sigma)**2)

        # ③  terminal validation bonus (same as before, just drop unnecessary .float())
        last_alive = is_alive.float().sum(1).long() - 1
        last_tok   = sequences[torch.arange(B), last_alive]
        eos_mask   = last_tok == self.eos_token_idx

        bonus = torch.zeros(B, device=self.device)
        if eos_mask.any():
            eos_idx  = eos_mask.nonzero(as_tuple=True)[0]
            token_seqs = [[self.idx_to_token[i] for i in seq if i != self.dead_token_idx]
                          for seq in sequences[eos_idx].tolist()]
            # OPTIMIZATION: Use the persistent multiprocessing pool.
            model_dicts = self.pool.map(parse_model_worker, token_seqs)

            tens = ingest_batch(model_dicts)
            scores, total = parallel_validator_and_scorer(tens)
            bonus[eos_idx] = (scores / total) * self.validation_bonus   # BIG signal

        # ④  combine everything, but *only once* per trajectory
        rewards = alive_reward
        rewards[torch.arange(B), last_alive] += len_bonus + bonus
        return rewards

    def display_results(self, sequences: torch.Tensor, rewards: torch.Tensor):
        """Pretty‑prints sequences along with rewards for debugging."""
        sequences_cpu, rewards_cpu = sequences.cpu().numpy(), rewards.cpu().numpy()
        for i in range(sequences_cpu.shape[0]):
            print(f"\n--- Sequence {i:02d} ---")
            header = "Step | Token                | Reward"
            print(header, "\n" + "-" * len(header), sep="")
            for j in range(sequences_cpu.shape[1]):
                token_idx = sequences_cpu[i, j]
                token_str = self.idx_to_token[token_idx]
                reward_val = rewards_cpu[i, j]
                print(f"{j:<4d} | {token_str:<18} | {reward_val:>6.2f}")
                if token_idx in (self.dead_token_idx, self.eos_token_idx):
                    if reward_val > self.validation_bonus - 1:
                        print("     -> ✅ Perfect physical score bonus applied!")
                    break

# ======================================================================================
# Phase 6: Main Execution and Benchmarking
# ======================================================================================

class DummyModel(nn.Module):
    """A stand‑in model that produces random logits."""
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
    def forward(self, x: torch.Tensor, cache: Optional[Any] = None) -> tuple[torch.Tensor, Any, Any]:
        logits = torch.randn(x.shape[0], x.shape[1], self.vocab_size, device=x.device)
        dummy_value = torch.zeros(x.shape[0], x.shape[1], device=x.device)
        return logits, dummy_value, cache


if __name__ == "__main__":
    BATCH_SIZE = 1_000
    VALIDATION_BONUS = 100.0

    print("--- Testing Full Pipeline: Token Sequence -> Parse -> Validate -> Reward ---")
    print(f"Using device: {DEVICE}\n")

    valid_interaction_tokens = [
        "ITRACT", "ITRACT_ID_0", "TYPE_YUKAWA",
        "FIELD", "FIELD_ID_0", "TYPE_fermion", "DIM_2", "GEN_1", "SELF_CONJ_FALSE", "CHIRALITY_left",
        "REP_SU3C_singlet", "REP_SU2L_fnd", "REP_U1Y_0", "QN_LeptonNumber_0", "QN_BaryonNumber_0",
        "PARTICLE", "PARTICLE_ID_0", "TYPE_fermion", "MASS_1e2", "CHARGE_0", "END_PARTICLE",
        "PARTICLE", "PARTICLE_ID_1", "TYPE_fermion", "MASS_1e2", "CHARGE_0", "END_PARTICLE",
        "END_FIELD",
        "FIELD", "FIELD_ID_1", "TYPE_fermion", "DIM_2", "GEN_1", "SELF_CONJ_FALSE", "CHIRALITY_right",
        "REP_SU3C_singlet", "REP_SU2L_singlet", "REP_U1Y_0", "QN_LeptonNumber_0", "QN_BaryonNumber_0",
        "PARTICLE", "PARTICLE_ID_2", "TYPE_fermion", "MASS_1e2", "CHARGE_0", "END_PARTICLE",
        "PARTICLE", "PARTICLE_ID_3", "TYPE_fermion", "MASS_1e2", "CHARGE_0", "END_PARTICLE",
        "END_FIELD",
        "FIELD", "FIELD_ID_2", "TYPE_complex", "DIM_2", "GEN_1", "SELF_CONJ_FALSE", "CHIRALITY_na",
        "REP_SU3C_singlet", "REP_SU2L_fnd", "REP_U1Y_0", "QN_LeptonNumber_0", "QN_BaryonNumber_0",
        "PARTICLE", "PARTICLE_ID_4", "TYPE_complex", "MASS_1e2", "CHARGE_0", "END_PARTICLE",
        "PARTICLE", "PARTICLE_ID_5", "TYPE_complex", "MASS_1e2", "CHARGE_0", "END_PARTICLE",
        "END_FIELD",
        "END_ITRACT"
    ]
    
    sm_token_sequence = ["BOS"] + valid_interaction_tokens + valid_interaction_tokens + ["EOS"]
    
    parser = TokenSequenceParser()
    parsed_sm_model = parser.build_model(sm_token_sequence)
    
    if not parsed_sm_model or not parsed_sm_model.get("interactions"):
        raise ValueError("Failed to parse the token sequence into a valid model.")
    
    print("Successfully parsed the token sequence into a model dictionary.")

    sm_models_batch = [parsed_sm_model] * BATCH_SIZE
    print(f"Created a batch of {len(sm_models_batch)} models from the parsed tokens.")

    t_start = time.time()
    tensors_dict = ingest_batch(sm_models_batch)
    final_scores, total_checks = parallel_validator_and_scorer(tensors_dict)
    validation_time = time.time() - t_start
    print(f"Validation for {BATCH_SIZE} models completed in {validation_time:.4f} seconds.")

    num_perfect_scores = (final_scores == total_checks).sum().item()
    print(f"\nTotal validation checks per model: {total_checks}")
    print(f"Number of models with perfect score: {num_perfect_scores}/{BATCH_SIZE}")

    if num_perfect_scores == BATCH_SIZE:
        print("✅ All 100 parsed models passed all validation checks as expected.")
        
        sequence_length = len(sm_token_sequence)
        placeholder_rewards = torch.zeros(BATCH_SIZE, sequence_length, device=DEVICE)
        last_step_idx = sequence_length - 1
        placeholder_rewards[:, last_step_idx] = 1.0

        print(f"\nInitial reward for last step: {placeholder_rewards[0, last_step_idx]:.2f}")

        print(f"Number of checks: {total_checks}")
        bonus = VALIDATION_BONUS * (final_scores == total_checks).float()
        
        placeholder_rewards[:, last_step_idx] += bonus
        
        print(f"Validation bonus of {VALIDATION_BONUS} applied.")
        print(f"Final reward for last step: {placeholder_rewards[0, last_step_idx]:.2f}")
        print("\nFunctionality test successful: The system correctly parses tokens, validates models, and applies the reward.")

    else:
        print("❌ Test failed: Not all models parsed from tokens passed the validation checks.")