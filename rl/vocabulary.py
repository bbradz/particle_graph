from transformers import AutoTokenizer
from typing import Tuple, Dict, List
from dataclasses import dataclass

try:
    # Optional import; avoid circulars if not needed
    from config import Config
except Exception:
    Config = None  # type: ignore

# A definitive list of all symbolic tokens that define the grammar's vocabulary.
GRAMMAR_TOKEN_NAMES: List[str] = [
    # Special Control Tokens
    "BOS", "EOS", "PAD",
    # Block Openers
    "ITRACT", "FIELD", "PARTICLE",
    # Block Closers
    "END_ITRACT", "END_FIELD", "END_PARTICLE",
    # Identifiers
    *[f"ITRACT_ID_{i}" for i in range(1, 6)], # e.g., ITRACT_ID_1
    *[f"FIELD_ID_{i}" for i in range(1, 15)], # e.g., FIELD_ID_1
    *[f"PARTICLE_ID_{i}" for i in range(1, 15)], # e.g., PARTICLE_ID_1
    # Interaction Types
    "TYPE_YUKAWA", "TYPE_SCALARSELFINTERACTION",
    # Field Types
    "TYPE_FIELD_fermion", "TYPE_FIELD_real", "TYPE_FIELD_complex",
    # Particle Types
    "TYPE_PARTICLE_fermion", "TYPE_PARTICLE_real", "TYPE_PARTICLE_complex",
    # Properties
    "DIM_1", "DIM_2", "DIM_3",
    "GEN_1", "GEN_2", "GEN_3",
    "SELF_CONJ_TRUE", "SELF_CONJ_FALSE",
    "CHIRALITY_left", "CHIRALITY_right", "CHIRALITY_none",
    # Representations
    "SU3C_REP_1", "SU3C_REP_2", "SU3C_REP_3", # singlet, fundamental, adjoint
    "SU2L_REP_1", "SU2L_REP_2", "SU2L_REP_3", # singlet, fundamental, adjoint
    "U1Y_CHARGE_-1", "U1Y_CHARGE_0", "U1Y_CHARGE_1", "U1Y_CHARGE_2", "U1Y_CHARGE_3", # Hypercharges
    # Quantum Numbers
    "QN_L_0", "QN_L_1", "QN_L_2", # Lepton Number
    "QN_B_0", "QN_B_1", "QN_B_2", # Baryon Number
    # Mass & Charge
    *[f"MASS_{m}" for m in ["1e-9", "1e-4", "1e-3", "1e-2", "1e-1", "1e0", "1e1", "1e2"]], # Mass values
    *[f"CHARGE_{c}" for c in ["-1", "0", "1", "2", "3"]], # Electric Charges
    # Parameters
    "LAMBDAVAR_1", # Corresponds to 0.1
    "LAMBDAVAR_2", # Corresponds to 0.5
    "LAMBDAVAR_3", # Corresponds to 1.0
    # A generic "NA" token for masking
    "NA"
]

@dataclass
class SimpleGrammarTokenizer:
    """A minimal tokenizer that treats each grammar token as an atomic token.

    This provides the small surface the rest of the code expects: __len__,
    convert_tokens_to_ids, pad_token_id, eos_token_id.
    """
    vocab: List[str]

    def __post_init__(self) -> None:
        self._tok2id: Dict[str, int] = {t: i for i, t in enumerate(self.vocab)}
        self._id2tok: Dict[int, str] = {i: t for t, i in self._tok2id.items()}
        # Required special token ids
        if 'PAD' not in self._tok2id or 'EOS' not in self._tok2id:
            raise ValueError("'PAD' and 'EOS' must be present in GRAMMAR_TOKEN_NAMES")
        self.pad_token_id: int = self._tok2id['PAD']
        self.eos_token_id: int = self._tok2id['EOS']

    def __len__(self) -> int:
        return len(self.vocab)

    def convert_tokens_to_ids(self, token: str) -> int:
        if token not in self._tok2id:
            raise KeyError(f"Unknown grammar token: {token}")
        return self._tok2id[token]

    # Convenience accessors used elsewhere
    @property
    def pad_token(self) -> str:
        return 'PAD'

    @property
    def eos_token(self) -> str:
        return 'EOS'

    # Compatibility no-op for HF API parity where called
    def add_special_tokens(self, *_args, **_kwargs) -> None:
        return None

    # Expose mappings if needed by callers
    @property
    def token_to_id(self) -> Dict[str, int]:
        return self._tok2id

    @property
    def id_to_token(self) -> Dict[int, str]:
        return self._id2tok


def initialize_tokenizer_and_mappings(config) -> Tuple[object, Dict[str, int], Dict[int, str], int]:
    """
    Initialize tokenizer and mappings.

    - If MODEL_TYPE == 'hf': use Hugging Face tokenizer and register grammar tokens
      as special tokens (legacy path).
    - If MODEL_TYPE == 'transformer': use a simple grammar tokenizer where the
      vocabulary is exactly the set of grammar tokens.
    """
    model_type = getattr(config, 'MODEL_TYPE', 'transformer')

    if model_type == 'hf':
        tokenizer = AutoTokenizer.from_pretrained(
            config.HF_MODEL_NAME,
            trust_remote_code=getattr(config, 'TRUST_REMOTE_CODE', True)
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.add_special_tokens({'additional_special_tokens': GRAMMAR_TOKEN_NAMES})

        TOKEN_TO_IDX = {tok: tokenizer.convert_tokens_to_ids(tok) for tok in GRAMMAR_TOKEN_NAMES}
        IDX_TO_TOKEN = {idx: tok for tok, idx in TOKEN_TO_IDX.items()}

        TOKEN_TO_IDX['PAD'] = tokenizer.pad_token_id
        IDX_TO_TOKEN[tokenizer.pad_token_id] = 'PAD'

        if 'EOS' not in TOKEN_TO_IDX:
            if tokenizer.eos_token_id is not None:
                TOKEN_TO_IDX['EOS'] = tokenizer.eos_token_id
                IDX_TO_TOKEN[tokenizer.eos_token_id] = 'EOS'
            else:
                raise ValueError("EOS token not found in vocabulary after adding special tokens")

        VOCAB_SIZE = len(tokenizer)
        print(f"Vocabulary initialized. Size: {VOCAB_SIZE}. PAD ID: {tokenizer.pad_token_id}, EOS ID: {tokenizer.eos_token_id}")
        return tokenizer, TOKEN_TO_IDX, IDX_TO_TOKEN, VOCAB_SIZE

    # MODEL_TYPE == 'transformer' → use grammar-only tokenizer
    tokenizer = SimpleGrammarTokenizer(GRAMMAR_TOKEN_NAMES)

    TOKEN_TO_IDX = {tok: tokenizer.convert_tokens_to_ids(tok) for tok in GRAMMAR_TOKEN_NAMES}
    IDX_TO_TOKEN = {idx: tok for tok, idx in TOKEN_TO_IDX.items()}

    VOCAB_SIZE = len(tokenizer)
    print(f"Vocabulary initialized. Size: {VOCAB_SIZE}. PAD ID: {tokenizer.pad_token_id}, EOS ID: {tokenizer.eos_token_id}")
    return tokenizer, TOKEN_TO_IDX, IDX_TO_TOKEN, VOCAB_SIZE