# file: config.py

from dataclasses import dataclass
from typing import Literal, List, Dict, Tuple

@dataclass
class Config:
    """Central configuration for the RL training system."""
    # Model and Tokenizer
    MODEL_TYPE: Literal["hf", "transformer"] = "transformer"
    HF_MODEL_NAME: str = "microsoft/phi-4-mini-reasoning"
    TORCH_DTYPE: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    USE_QUANTIZATION: bool = True # Use 4-bit quantization for memory efficiency
    TRUST_REMOTE_CODE: bool = True

    # Simple Transformer Parameters (used when MODEL_TYPE="transformer")
    TRANSFORMER_VOCAB_SIZE: int = 1000
    TRANSFORMER_D_MODEL: int = 512
    TRANSFORMER_NHEAD: int = 8
    TRANSFORMER_NUM_LAYERS: int = 6
    TRANSFORMER_DIM_FEEDFORWARD: int = 2048
    TRANSFORMER_DROPOUT: float = 0.1
    TRANSFORMER_MAX_LEN: int = 256

    # Training Loop
    EPOCHS: int = 1000
    STEPS_PER_EPOCH: int = 200
    BATCH_SIZE: int = 16
    GRAD_ACCUMULATION_STEPS: int = 5
    LEARNING_RATE: float = 5e-5
    MAX_GRAD_NORM: float = 1.0

    # PPO Hyperparametrs
    PPO_EPOCHS: int = 6
    CLIP_RANGE: float = 0.2
    GAMMA: float = 0.995
    LAMBDA_GAE: float = 0.98
    ENTROPY_COEF: float = 0.01
    VALUE_COEF: float = 0.5
    
    # Supervised Error Loss Coefficient
    SUPERVISED_LOSS_COEF: float = 0.1

    # Sequence Generation
    MAX_LEN: int = 256
    TEMPERATURE: float = 1.0

    # Reward Shaping
    # Penalty applied per token from the start of an unclosed block to the end of the sequence.
    # The value is multiplied by the nesting depth of the unclosed block.
    UNCLOSED_BLOCK_PENALTY_PER_DEPTH: float = -0.5

    # A neutral score for initializing per-token target vectors (e.g., for Sigmoid output)
    DEFAULT_NEUTRAL_CHECK_SCORE: float = 0.5

    # A bonus awarded to the EOS token for completing a sequence.
    EOS_BONUS: float = 0.1

    # Token IDs
    EOS_TOKEN_ID: int = -1 # Placeholder, will be set dynamically
    PAD_TOKEN_ID: int = -1 # Placeholder, will be set dynamically

    # Logging and Output
    KEEP_GENERATED_MODEL_FILES: bool = False
    PRINT_GENERATED_SEQUENCES: bool = True
    PRINT_SEQUENCE: bool = True  # Control whether to print the formatted sequence
    PRINT_DETAILED_CHECKLIST: bool = True  # Control whether to print the detailed checklist
    NUM_SEQUENCES_TO_PRINT: int = 1  # Number of sequences from batch to print (0 = print all)
    PRINT_PER_CHECK_REWARDS_DETAIL: bool = True # Set to True to enable per-token reward detail

    # File Paths
    MODEL_BASE_PATH: str = "./Models" # Default to local Models directory
    LOG_EVERY_N_STEPS: int = 25
    EVAL_EVERY_N_STEPS: int = 50
    DEBUG_PRINTS: bool = False  # Controls all debug print statements throughout the codebase

CONFIG = Config()

def get_config() -> Config:
    return CONFIG