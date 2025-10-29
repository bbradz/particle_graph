from dataclasses import dataclass
from typing import Literal, List, Dict, Tuple

@dataclass
class Config:
    """Central configuration for the RL training system."""
    
    # ========================================================================
    # MODEL ARCHITECTURE & TOKENIZER
    # ========================================================================
    # Used by: rl/network.py, rl/vocabulary.py
    MODEL_TYPE: Literal["hf", "transformer"] = "transformer"
    HF_MODEL_NAME: str = "microsoft/phi-4-mini-reasoning"
    TORCH_DTYPE: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    USE_QUANTIZATION: bool = True  # Use 4-bit quantization for memory efficiency
    TRUST_REMOTE_CODE: bool = True

    # Simple Transformer Parameters (used when MODEL_TYPE="transformer")
    # Used by: rl/network.py
    TRANSFORMER_VOCAB_SIZE: int = 1000
    TRANSFORMER_D_MODEL: int = 512
    TRANSFORMER_NHEAD: int = 8
    TRANSFORMER_NUM_LAYERS: int = 6
    TRANSFORMER_DIM_FEEDFORWARD: int = 2048
    TRANSFORMER_DROPOUT: float = 0.1
    TRANSFORMER_MAX_LEN: int = 256

    # ========================================================================
    # TRAINING LOOP & OPTIMIZATION
    # ========================================================================
    # Used by: rl/agent.py, train.py
    EPOCHS: int = 1000
    STEPS_PER_EPOCH: int = 200
    BATCH_SIZE: int = 16
    GRAD_ACCUMULATION_STEPS: int = 5
    LEARNING_RATE: float = 1e-6
    MAX_GRAD_NORM: float = 1.0

    # ========================================================================
    # PPO ALGORITHM
    # ========================================================================
    # Used by: rl/agent.py
    PPO_EPOCHS: int = 8
    CLIP_RANGE: float = 0.1
    GAMMA: float = 0.995
    LAMBDA_GAE: float = 0.98
    ENTROPY_COEF: float = 0.01
    VALUE_COEF: float = 0.5
    PPO_LOSS_EXPLOSION_THRESHOLD: float = 100.0

    # ========================================================================
    # SEQUENCE GENERATION & GRAMMAR
    # ========================================================================
    # Used by: rl/agent.py, rl/grammar.py, rl/environment.py
    MAX_LEN: int = 256
    TEMPERATURE: float = 1.0
    EOS_TOKEN_ID: int = -1  # Placeholder, will be set dynamically
    PAD_TOKEN_ID: int = -1  # Placeholder, will be set dynamically

    # ========================================================================
    # REWARD SHAPING & CRITIC ARCHITECTURE
    # ========================================================================
    # Used by: rl/reward.py, rl/agent.py, rl/network.py
    USE_REWARD_SHAPING: bool = True
    REWARD_SHAPING_LAMBDA_START: float = 0.1 
    REWARD_SHAPING_LAMBDA_END: float = 1.0
    REWARD_SHAPING_ANNEAL_STEPS: int = 1000
    
    # Reward penalties and bonuses
    UNCLOSED_BLOCK_PENALTY_PER_DEPTH: float = -0.5
    DEFAULT_NEUTRAL_CHECK_SCORE: float = 0.5
    EOS_BONUS: float = 1.0
    LENGTH_PENALTY_PER_TOKEN: float = -0.01

    # Three-Headed Critic Architecture
    CRITIC_ARCHITECTURE: Literal["single", "three_headed"] = "three_headed"
    VALUE_HEAD_COEF: float = 0.5
    REWARD_HEAD_COEF: float = 0.3
    CRITICALITY_HEAD_COEF: float = 0.2
    CRITICALITY_LOSS_COEF: float = 0.1

    # ========================================================================
    # MULTI-PHASE PRETRAINING & RL LOSSES
    # ========================================================================
    # Pretraining phases (A-C)
    PRE_TRAIN_DATASET_SIZE: int = 2_000
    PRE_TRAIN_GRAMMAR_STEPS: int = 100
    PRE_TRAIN_CRITICALITY_STEPS: int = 200
    PRE_TRAIN_EXPLORATION_STEPS: int = 200
    PRE_TRAIN_BATCH_SIZE: int = 8
    PRE_TRAIN_LEARNING_RATE: float = 1e-3
    # Cache root for storing generated pretraining data (sequences, masks, targets, scores)
    PRETRAIN_CACHE_ROOT: str = "/oscar/scratch/bpbradle/particle_graph/pretrain_cache"

    # ========================================================================
    # CHECKPOINTING & RESUMING
    # ========================================================================
    CHECKPOINT_DIR: str = "/oscar/scratch/bpbradle/particle_graph/training_checkpoints"

    # Phase D loss coefficients
    Q_HEAD_COEF: float = 0.3
    RND_COEF: float = 0.1
    CRITICALITY_COEF: float = 0.2

    # ========================================================================
    # EXPERIENCE REPLAY & MEMORY
    # ========================================================================
    # Used by: rl/agent.py, rl/replay_buffer.py
    USE_EXPERIENCE_REPLAY: bool = True
    REPLAY_BATCH_SIZE: int = 8
    PRIORITIZED_REPLAY_ENABLED: bool = True
    REPLAY_BUFFER_SIZE: int = 1000
    REPLAY_ALPHA: float = 0.6  # Prioritization exponent
    REPLAY_BETA: float = 0.4   # Importance sampling exponent
    REPLAY_BETA_ANNEAL_STEPS: int = 10000
    REPLAY_EPSILON: float = 1e-6  # Small constant to avoid zero priorities
    PER_SUCCESS_BONUS: float = 10.0

    # ========================================================================
    # CURRICULUM LEARNING
    # ========================================================================
    # Used by: rl/curriculum.py, rl/agent.py
    CURRICULUM_ENABLED: bool = True
    USE_CURRICULUM_LOSS_MASK: bool = True
    CURRICULUM_PHASE_1_STEPS: int = 1000  # Steps for basic particle/field checks
    CURRICULUM_PHASE_2_STEPS: int = 2000  # Steps for interaction checks
    CURRICULUM_PHASE_3_STEPS: int = 3000  # Steps for global anomaly checks
    CURRICULUM_PHASE_4_STEPS: int = 5000  # Steps for full complexity

    # Rolling-window advancement controls
    CURRICULUM_WINDOW_SIZE: int = 100
    CURRICULUM_TRANSITION_THRESHOLD: float = 0.95

    # ========================================================================
    # EXPLORATION & HIERARCHICAL LEARNING
    # ========================================================================
    # Used by: rl/agent.py, rl/curriculum.py
    EXPLORATION_ENABLED: bool = True
    EXPLORATION_BASE_TEMPERATURE: float = 1.0
    EXPLORATION_MIN_TEMPERATURE: float = 0.1
    EXPLORATION_DECAY_RATE: float = 0.995
    EXPLORATION_PROPORTIONAL_SCALING: bool = True
    EXPLORATION_HIERARCHICAL_WEIGHT: float = 0.3
    EXPLORATION_BETA: float = 0.5
    EXPLORATION_LOSS_COEF: float = 0.1
    UCB_C: float = 0.1
    CRITICALITY_THRESHOLD: float = 0.5

    # ========================================================================
    # LOSS COEFFICIENTS & REGULARIZATION
    # ========================================================================
    # Used by: rl/agent.py
    SUPERVISED_LOSS_COEF: float = 0.1
    SUPERVISED_ERROR_LOSS_COEF: float = 0.1
    HIERARCHICAL_CONSISTENCY_ENABLED: bool = True
    HIERARCHICAL_CONSISTENCY_COEF: float = 0.1
    HIERARCHICAL_DEPENDENCY_WEIGHT: float = 0.5
    AUXILIARY_LOSSES_ENABLED: bool = True
    CONSISTENCY_LOSS_COEF: float = 0.05
    REGULARIZATION_LOSS_COEF: float = 0.01

    # ========================================================================
    # CRITIC PRE-TRAINING
    # ========================================================================
    # Used by: test_critic.py, test_criticality.py
    PRE_TRAIN_CRITIC: bool = True
    PRE_TRAIN_DATASET_SIZE_LEGACY: int = 2e3
    PRE_TRAIN_EPOCHS: int = 10
    PRE_TRAIN_LEARNING_RATE: float = 3e-4

    # ========================================================================
    # LOGGING & OUTPUT CONTROL
    # ========================================================================
    # Used by: rl/logging.py, rl/agent.py, rl/environment.py, train.py, rl/utils.py
    
    # Master logging controls
    ENABLE_LOGGING: bool = True
    DEBUG_PRINTS: bool = False  # Master debug flag for all modules
    REDIRECT_DEBUG_TO_FILE: bool = False
    
    # Training phase logging
    PRINT_TRAINING_INIT: bool = True
    PRINT_PHASE_HEADERS: bool = True
    PRINT_PHASE_PROGRESS: bool = True
    PRINT_CHECKPOINT_INFO: bool = True
    
    # Data generation logging
    PRINT_DATA_GENERATION: bool = True
    PRINT_DATASET_STATS: bool = True
    PRINT_CACHE_INFO: bool = True
    
    # Model evaluation logging
    PRINT_EVALUATION_RESULTS: bool = True
    PRINT_PER_CHECK_RESULTS: bool = True
    PRINT_DETAILED_TABLES: bool = True
    
    # Sequence generation logging
    PRINT_GENERATED_SEQUENCES: bool = True
    PRINT_SEQUENCE: bool = True
    PRINT_DETAILED_CHECKLIST: bool = True
    NUM_SEQUENCES_TO_PRINT: int = 1
    
    # Reward and environment logging
    PRINT_PER_CHECK_REWARDS_DETAIL: bool = True
    PRINT_FULL_ENV_DEBUG: bool = False
    PRINT_REWARD_DEBUG: bool = False
    
    # File management
    KEEP_GENERATED_MODEL_FILES: bool = True
    
    # Logging frequency
    LOG_EVERY_N_STEPS: int = 25
    EVAL_EVERY_N_STEPS: int = 25

    # ========================================================================
    # FILE PATHS & PLOTTING
    # ========================================================================
    # Used by: rl/environment.py, rl/plotting.py
    DEBUG_OUTPUT_FILE: str = "debug_output.txt"
    MODEL_BASE_PATH: str = "/oscar/scratch/bpbradle/particle_graph/Models"
    PLOT_EVERY_N_STEPS: int = 10
    PLOT_OUTPUT_DIR: str = "/oscar/scratch/bpbradle/particle_graph/training_plots"

CONFIG = Config()

def get_config() -> Config:
    return CONFIG