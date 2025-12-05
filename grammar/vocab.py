from config import config
from dataclasses import dataclass

# Gauge Group Vocab
GROUP_TYPES = ["GAUGE_U", "GAUGE_SU"]
GROUP_IDS = [f"g_{i+1}" for i in range(config.max_gauge_groups)]
GROUP_RANKS = [f"rank_{i+1}" for i in range(config.max_group_rank)]

# Particle Vocab
PARTICLES = ["PTCL_FERMION", "PTCL_SCALAR"]
COLORS = ["COLOR", "NO_COLOR"]
SM_TAGS = ["TAG_E", "TAG_MU", "TAG_TAU", "TAG_VE", "TAG_VM", "TAG_VT", "TAG_U", "TAG_C", "TAG_D", "TAG_S", "TAG_T", "TAG_B", "TAG_Hp", "TAG_H0"]
CHARGES = [f"charge_{i}" for i in range(-config.max_charge, config.max_charge+1)]

# Multiplet Vocab
MULTIPLET_TYPES = ["MPLT_SCALAR", "MPLT_FERMION"]
MULTIPLET_IDS = [f"m_{i+1}" for i in range(config.max_multiplets)]
CHIRALITIES = ["left", "right", "null"]
#REPRESENTATIONS = ["singlet", "fnd", "adj"] 
REPRESENTATIONS = ["singlet", "fnd"]
HYPERCHARGES = ["hypercharge_0"] + [f"hypercharge_{i}" for i in range(-config.max_hypercharge, config.max_hypercharge+1)]
GENS = [f"gen_{i}" for i in range(1, config.max_gen+1)]
DIMS = [f"dim_{i}" for i in range(1, config.max_dim+1)]

# Interaction Vocab
INTERACTION_TYPES = ["TERM_YUKAWA", "TERM_PHI4"]
INTERACTION_IDS = [f"i_{i+1}" for i in range(config.max_interactions)]
PARAMETERS = [f"param_1e{i}" for i in range(-config.max_value_exp, config.max_value_exp+1)]
MASS= ["mass_0"] + [f"mass_1e{i}" for i in range(-config.max_mass_exp, config.max_mass_exp+1)]

GRAMMAR_TOKENS = [
    "BOS", "EOS", "PAD",
    # Gauge Group Tokens
    *GROUP_IDS,
    *GROUP_TYPES,
    *GROUP_RANKS,
    "END_GAUGE",

    # Particle Tokens
    *PARTICLES,
    *SM_TAGS,
    *COLORS,
    "END_PTCL",
    "END_PARTICLE_BLOCK",

    # Multiplet Tokens
    *MULTIPLET_TYPES,
    *MULTIPLET_IDS,
    *CHIRALITIES,
    *GENS,
    *REPRESENTATIONS,
    *HYPERCHARGES,
    "REPS", "END_REPS",
    "END_MULTIPLET", 

    # Interaction Tokens
    *INTERACTION_TYPES,
    *INTERACTION_IDS,
    *PARAMETERS,
    *MASS,
    "MPLTS", "END_MPLT",
    "PARAMS", "END_PARAM",
    "END_INTERACTION"
]

token2id = {token: i for i, token in enumerate[str](GRAMMAR_TOKENS)}
id2token = {i: token for i, token in enumerate(GRAMMAR_TOKENS)}

PAD_TOKEN_ID = token2id["PAD"]
BOS_TOKEN_ID = token2id["BOS"]
EOS_TOKEN_ID = token2id["EOS"]

def encode(sequence: list[str]) -> list[int]:
    return [token2id[token] for token in sequence]

def decode(ids: list[int]) -> list[str]:
    return [id2token[id] for id in ids]
