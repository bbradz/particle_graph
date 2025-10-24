"""
Reinforcement Learning Package for Particle Physics Model Generation

This package contains all the core components for the PPO-based training
system, including the agent, environment, reward shaper, grammar masker,
and network definitions.
"""

# Import key classes and functions to the package level for easy access.
from .agent import RLTrainer
from .network import make_policy, HuggingFacePolicy, SimpleTransformerPolicy
from .environment import ParticlePhysicsEnvironment, EnvOutcome
from .reward import RewardShaper, ShapedRewards
from .grammar import GrammarMasker, GrammarState
from .vocabulary import initialize_tokenizer_and_mappings, GRAMMAR_TOKEN_NAMES
from .advantages import compute_advantages
from .curriculum import Curriculum
from .utils import finalize_sequence_inplace

# Define the public API of the 'rl' package.
__all__ = [
    # from agent.py
    "RLTrainer",
    # from network.py
    "make_policy",
    "HuggingFacePolicy",
    "SimpleTransformerPolicy",
    # from environment.py
    "ParticlePhysicsEnvironment",
    "EnvOutcome",
    # from reward.py
    "RewardShaper",
    "ShapedRewards",
    # from grammar.py
    "GrammarMasker",
    "GrammarState",
    # from vocabulary.py
    "initialize_tokenizer_and_mappings",
    "GRAMMAR_TOKEN_NAMES",
    # from advantages.py
    "compute_advantages",
    # from curriculum.py
    "Curriculum",
    # from utils.py
    "finalize_sequence_inplace",
    # from replay_buffer.py
    "ReplayBuffer",
]