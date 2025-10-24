from transformers import AutoTokenizer
from typing import Tuple, Dict, List

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

def initialize_tokenizer_and_mappings(config) -> Tuple[AutoTokenizer, Dict[str, int], Dict[int, str], int]:
    """
    Loads the specified Hugging Face tokenizer, adds the grammar tokens as special
    tokens, and returns the tokenizer along with the necessary vocabulary mappings.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        config.HF_MODEL_NAME,
        trust_remote_code=config.TRUST_REMOTE_CODE
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add grammar tokens to the tokenizer's vocabulary.
    # This ensures they are treated as single, indivisible units.
    tokenizer.add_special_tokens({'additional_special_tokens': GRAMMAR_TOKEN_NAMES})

    # Create the mappings that the rest of the application will use.
    TOKEN_TO_IDX = {tok: tokenizer.convert_tokens_to_ids(tok) for tok in GRAMMAR_TOKEN_NAMES}
    IDX_TO_TOKEN = {idx: tok for tok, idx in TOKEN_TO_IDX.items()}

    # Ensure PAD token is correctly mapped
    TOKEN_TO_IDX['PAD'] = tokenizer.pad_token_id
    IDX_TO_TOKEN[tokenizer.pad_token_id] = 'PAD'
    
    # Ensure EOS token is correctly mapped - always use the grammar token EOS
    # This ensures EOS and PAD have distinct IDs
    if 'EOS' in TOKEN_TO_IDX:
        # EOS was added as a special token, use its ID
        pass  # TOKEN_TO_IDX['EOS'] is already set
    else:
        # Fallback: if EOS wasn't added as special token, use eos_token_id
        if tokenizer.eos_token_id is not None:
            TOKEN_TO_IDX['EOS'] = tokenizer.eos_token_id
            IDX_TO_TOKEN[tokenizer.eos_token_id] = 'EOS'
        else:
            # This should not happen since EOS is in GRAMMAR_TOKEN_NAMES
            raise ValueError("EOS token not found in vocabulary after adding special tokens")


    VOCAB_SIZE = len(tokenizer)

    print(f"Vocabulary initialized. Size: {VOCAB_SIZE}. PAD ID: {tokenizer.pad_token_id}, EOS ID: {tokenizer.eos_token_id}")

    return tokenizer, TOKEN_TO_IDX, IDX_TO_TOKEN, VOCAB_SIZE