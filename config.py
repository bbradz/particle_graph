from dataclasses import dataclass

@dataclass
class Config:

    # ========================================================================
    # Random Seed
    # ========================================================================
    random_seed: int = 42
    
    # ========================================================================
    # Model Parameters
    # ========================================================================
    max_gauge_groups: int = 3
    max_group_rank: int = 3
    max_interactions: int = 10
    max_multiplets: int = 15
    max_particles: int = 20
    max_dim: int = 3
    max_gen: int = 3
    max_charge: int = 6
    max_hypercharge: int = 9
    max_mass_exp: int = 4
    max_value_exp: int = 1
    PAD_TOKEN_ID: int = 2

    max_exotic_scalar: int = 3
    max_exotic_fermion: int = 5

    # ========================================================================
    # Debugging
    # ========================================================================
    GRAMMAR_DEBUG: bool = False
    DEBUG_MODE: bool = False

    # ========================================================================
    # Network Parameters
    # ========================================================================
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_length: int = 512
    
    # ========================================================================
    # Data Parameters
    # ========================================================================
    pretrain_data_size: int = 2000
    pretrain_data_dir: str = "/users/qniu3/physics/RL_builder_5.0/dataset/pretrain/"
    VAL_SPLIT_SEED: int = 1337

    # ========================================================================
    # Pretraining Parameters
    # ========================================================================
    PRETRAIN_LEARNING_RATE: float = 1e-4
    PRETRAIN_BATCH_SIZE: int = 8
    PRETRAIN_NUM_EPOCHS: int = 10
    PRETRAIN_GRAD_ACCUMULATION_STEPS: int = 5



config = Config()