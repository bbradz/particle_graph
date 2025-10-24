# test_critic_convergence.py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Any
import time # <-- Imported time
import random
import copy
import os
import sys

# Add the parent directory to the path to allow imports from rl and Token2Model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config, Config
from rl.network import make_policy
from rl.vocabulary import initialize_tokenizer_and_mappings
from rl.grammar import GrammarMasker, GrammarState
from rl.environment import ParticlePhysicsEnvironment
from rl.reward import RewardShaper
from rl.logging import format_sequence_with_rewards
from Token2Model.check import NUM_CHECKS

# --- Configuration for the Test ---
TEST_CONFIG = get_config()
TEST_CONFIG.BATCH_SIZE = 32 # Training batch size
TEST_CONFIG.MAX_LEN = 150 # Reduced max length for faster generation
TEST_CONFIG.EPOCHS = 500  # Number of *offline* training epochs
TEST_CONFIG.LEARNING_RATE = 1e-4 # Higher LR for pure supervised training
# Use a simple transformer for stability/reproducibility in a test
TEST_CONFIG.MODEL_TYPE = "transformer" 
TEST_CONFIG.TRANSFORMER_D_MODEL = 512 # Smaller model for faster convergence
TEST_CONFIG.TRANSFORMER_NUM_LAYERS = 32
TEST_CONFIG.TRANSFORMER_VOCAB_SIZE = 1000 # Will be overwritten by tokenizer

N_SEQUENCES_TO_GENERATE = 500 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Custom Dataset for Offline Value Training ---
class ValueDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]]):
        self.sequences = [item['sequence'] for item in data]
        self.masks = [item['mask'] for item in data]
        self.targets = [item['target'] for item in data]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            self.sequences[idx].to(DEVICE),
            self.masks[idx].to(DEVICE),
            self.targets[idx].to(DEVICE)
        )

# --- Data Generation Function ---
def generate_data(
    cfg: Config, 
    tokenizer, 
    token_to_idx: Dict[str, int], 
    idx_to_token: Dict[int, str], 
    n_sequences: int
) -> List[Dict[str, Any]]:
    """Generates N grammatically correct sequences and calculates the reward target."""
    print(f"\n--- Generating {n_sequences} Sequences for Offline Training ---")
    
    # Start overall timer
    total_start_time = time.time()
    
    grammar = GrammarMasker(cfg, token_to_idx, idx_to_token, len(tokenizer))
    env = ParticlePhysicsEnvironment(cfg)
    reward_shaper = RewardShaper(cfg, torch.float32) 
    
    data = []
    
    # Initialize time counters for components
    total_generation_time = 0.0
    total_target_calc_time = 0.0
    total_env_run_time = 0.0
    total_reward_calc_time = 0.0
    
    # Pre-calculate the fixed BOS token and max length
    bos_id = token_to_idx['BOS']
    pad_id = token_to_idx['PAD']
    T = cfg.MAX_LEN

    for i in range(n_sequences):
        # Print progress with accumulated time
        if i % 100 == 0 and i > 0: 
            current_time = time.time()
            elapsed = current_time - total_start_time
            print(f"  Generated {i}/{n_sequences} sequences... ({elapsed:.2f}s total)")
            print(f"    - Sequence Generation Time: {total_generation_time:.2f}s")
            print(f"    - Target Calculation Time: {total_target_calc_time:.2f}s")
            print(f"      * Environment Run Time: {total_env_run_time:.2f}s")
            print(f"      * Reward Calculation Time: {total_reward_calc_time:.2f}s")
            
        # 1. Generate a single grammatically valid sequence
        generation_start = time.perf_counter() # Start high-res timer
        
        seq_ids = [bos_id]
        attn_mask_list = [True]
        state = grammar.initial_state()
        state = grammar.step(state, bos_id)

        for t in range(T - 1):
            valid_mask_cpu = grammar.get_valid_actions(state)
            valid_ids = valid_mask_cpu.nonzero(as_tuple=True)[0].tolist()
            
            if not valid_ids:
                # Should not happen for a grammatically-open state unless max_len is reached
                next_tok_id = pad_id
            else:
                next_tok_id = random.choice(valid_ids)
            
            seq_ids.append(next_tok_id)
            attn_mask_list.append(True)
            
            if next_tok_id == token_to_idx['EOS']:
                break
            
            state = grammar.step(state, next_tok_id)

        # Pad remaining length
        while len(seq_ids) < T:
            seq_ids.append(pad_id)
            attn_mask_list.append(False)
            
        sequence_tensor = torch.tensor(seq_ids, dtype=torch.long)
        attn_mask_tensor = torch.tensor(attn_mask_list, dtype=torch.bool)
        
        total_generation_time += time.perf_counter() - generation_start # End high-res timer

        # 2. Process the sequence to get the ground-truth target
        target_calc_start = time.perf_counter() # Start high-res timer
        
        effective_len = attn_mask_tensor.sum().item()
        token_strs = [idx_to_token.get(i, "PAD") for i in sequence_tensor[:effective_len].tolist()]
        
        # NOTE: Using environment.run_episode is SLOW due to I/O and Model initialization. 
        # We process the sequence once to get the target reward.
        try:
            # Time the environment run separately
            env_run_start = time.perf_counter()
            episode_outcome = env.run_episode(sequence_tensor[:effective_len].tolist(), idx_to_token)
            total_env_run_time += time.perf_counter() - env_run_start
            
            # Time the reward calculation separately
            reward_calc_start = time.perf_counter()
            rewards_object = reward_shaper.calculate_rewards(episode_outcome, sequence_tensor)
            total_reward_calc_time += time.perf_counter() - reward_calc_start
            
            # 3. Store sequence, mask, and ground-truth R_target
            data.append({
                'sequence': sequence_tensor,
                'mask': attn_mask_tensor,
                'target': rewards_object.per_token_value_targets
            })
        except Exception as e:
            # Skip sequence if environment/reward processing fails
            print(f"Skipping sequence {i} due to critical error: {e}")
            continue

        total_target_calc_time += time.perf_counter() - target_calc_start # End high-res timer

    total_elapsed_time = time.time() - total_start_time # Calculate total time
    print(f"\n--- Data Generation Complete. Total sequences: {len(data)} ---")
    print(f"--- Data Generation Total Time: {total_elapsed_time:.2f}s ---")
    print(f"  - Total Sequence Generation (Grammar): {total_generation_time:.2f}s ({total_generation_time/total_elapsed_time*100:.1f}%)")
    print(f"  - Total Target Calculation (Env/Reward): {total_target_calc_time:.2f}s ({total_target_calc_time/total_elapsed_time*100:.1f}%)")
    print(f"    * Environment Run Time: {total_env_run_time:.2f}s ({total_env_run_time/total_elapsed_time*100:.1f}%)")
    print(f"    * Reward Calculation Time: {total_reward_calc_time:.2f}s ({total_reward_calc_time/total_elapsed_time*100:.1f}%)")
    return data

# --- Offline Training Function ---
def train_critic(cfg: Config, policy, dataloader: DataLoader, N_epochs: int):
    """Trains the critic head in a purely supervised manner on the fixed dataset."""
    print(f"\n--- Starting Offline Critic Training ({N_epochs} epochs) ---")
    
    # Start overall timer
    total_train_start_time = time.time()
    
    # Freeze the main transformer body, only train the value head
    for param in policy.parameters():
        param.requires_grad = False
    for param in policy.value_head.parameters():
        param.requires_grad = True
        
    optimizer = torch.optim.Adam(policy.value_head.parameters(), lr=cfg.LEARNING_RATE)
    
    policy.to(DEVICE)
    policy.train()
    
    best_loss = float('inf')
    
    # Per-component time tracking across all epochs
    total_forward_time = 0.0
    total_backward_time = 0.0

    for epoch in range(N_epochs):
        epoch_start_time = time.time() # Start epoch timer
        epoch_loss = 0.0
        n_samples = 0
        
        for sequences, masks, targets in dataloader:
            # sequences, masks, targets are already on DEVICE from custom dataset
            optimizer.zero_grad()
            
            # Forward pass: get hidden states from the (frozen) transformer
            forward_start_time = time.perf_counter() # Start high-res timer
            with torch.no_grad():
                # We need to run the full forward pass to get the hidden states
                outputs = policy(sequences, attention_mask=masks.to(torch.long))
                hidden_states = outputs.hidden_states[-1]
            
            # Forward pass through the (trainable) critic head
            predictions = policy.value(hidden_states)
            
            # Calculate MSE loss only on non-padded tokens
            active_predictions = predictions[masks]
            active_targets = targets[masks]
            
            loss = F.mse_loss(active_predictions, active_targets)
            forward_end_time = time.perf_counter() # End high-res timer
            total_forward_time += forward_end_time - forward_start_time
            
            # Backward pass and optimization
            backward_start_time = time.perf_counter() # Start high-res timer
            loss.backward()
            optimizer.step()
            total_backward_time += time.perf_counter() - backward_start_time # End high-res timer
            
            epoch_loss += loss.item() * len(sequences)
            n_samples += len(sequences)
        
        avg_loss = epoch_loss / n_samples
        best_loss = min(best_loss, avg_loss)
        epoch_duration = time.time() - epoch_start_time # End epoch timer
        
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == N_epochs - 1:
            print(f"Epoch {epoch + 1}/{N_epochs}, Avg MSE Loss: {avg_loss:.6f}, Time: {epoch_duration:.2f}s")

    total_train_time = time.time() - total_train_start_time # Calculate total training time
    print(f"--- Offline Training Complete. Total Training Time: {total_train_time:.2f}s ---")
    print(f"  - Total Forward Pass Time (incl. Frozen Body): {total_forward_time:.2f}s ({total_forward_time/total_train_time*100:.1f}%)")
    print(f"  - Total Backward Pass & Step Time: {total_backward_time:.2f}s ({total_backward_time/total_train_time*100:.1f}%)")
    print(f"Final MSE Loss: {avg_loss:.6f}")
    print(f"Best MSE Loss Achieved: {best_loss:.6f}")
    return best_loss


# --- Post-Training Evaluation Function ---
def evaluate_sequence(
    cfg: Config,
    policy,
    tokenizer,
    token_to_idx: Dict[str, int],
    idx_to_token: Dict[int, str],
    env: ParticlePhysicsEnvironment,
    reward_shaper: RewardShaper,
    sample_data: Dict[str, Any]
):
    """Evaluate a single sequence and print detailed reward analysis."""
    print("\n" + "="*60)
    print("      POST-TRAINING SEQUENCE EVALUATION")
    print("="*60)
    
    # Get the sequence data
    sequence_tensor = sample_data['sequence']
    attn_mask_tensor = sample_data['mask']
    target_rewards = sample_data['target']
    
    # Convert sequence to readable format
    effective_len = attn_mask_tensor.sum().item()
    sequence_tokens = sequence_tensor[:effective_len].tolist()
    sequence_str = [idx_to_token.get(token_id, "UNK") for token_id in sequence_tokens]
    
    print(f"Sequence: {' '.join(sequence_str)}")
    print(f"Sequence Length: {effective_len} tokens")
    print()
    
    # Run environment to get raw rewards
    episode_outcome = env.run_episode(sequence_tokens, idx_to_token)
    rewards_object = reward_shaper.calculate_rewards(episode_outcome, sequence_tensor)
    
    # Get predicted rewards from the trained model
    policy.eval()
    with torch.no_grad():
        # Add batch dimension and move to device
        sequence_batch = sequence_tensor.unsqueeze(0).to(DEVICE)
        mask_batch = attn_mask_tensor.unsqueeze(0).to(DEVICE)
        
        # Get hidden states from transformer
        outputs = policy(sequence_batch, attention_mask=mask_batch.to(torch.long))
        hidden_states = outputs.hidden_states[-1]
        
        # Get value predictions
        predicted_values = policy.value(hidden_states)
        predicted_values = predicted_values.squeeze(0)  # Remove batch dimension
    
    # Get the correct attributes from ShapedRewards
    per_token_instantaneous_rewards = rewards_object.per_token_instantaneous_rewards[:effective_len]
    per_token_target_rewards = target_rewards[:effective_len]
    predicted_token_rewards = predicted_values[:effective_len]
    
    # Calculate summary statistics (sum across all checks for each token)
    total_target_sum = per_token_target_rewards.sum().item()
    total_predicted_sum = predicted_token_rewards.sum().item()
    scalar_total_reward = rewards_object.scalar_total_reward
    
    print("REWARD ANALYSIS:")
    print("-" * 40)
    print(f"Sequence Length: {effective_len} tokens")
    print(f"Scalar Total Reward: {scalar_total_reward:.4f}")
    print(f"Target V_Sum: {total_target_sum:.4f}")
    print(f"Predicted V_Sum: {total_predicted_sum:.4f}")
    print(f"V_Sum Error: {abs(total_target_sum - total_predicted_sum):.4f}")
    print(f"V_Sum Error %: {abs(total_target_sum - total_predicted_sum) / abs(total_target_sum) * 100:.2f}%")
    print()
    
    # Print per-check rewards
    print("DETAILED SEQUENCE WITH REWARDS (Per Check):")
    print("-" * 40)
    
    # Get the number of checks
    num_checks = per_token_instantaneous_rewards.size(1)
    
    for check_idx in range(num_checks):
        print(f"\n--- Check {check_idx} ---")
        formatted_output = format_sequence_with_rewards(
            sequence_str,
            per_token_instantaneous_rewards[:, check_idx],
            per_token_target_rewards[:, check_idx],
            predicted_token_rewards[:, check_idx],
            print_sequence=True
        )
        print(formatted_output)
    
    print("="*60)


# --- Main Execution ---
if __name__ == "__main__":
    
    main_start_time = time.time() # NEW: Start overall script timer
    
    # 1. Setup and Initialization
    print(f"Running Critic Convergence Test on {DEVICE}")
    tokenizer, token_to_idx, idx_to_token, vocab_size = initialize_tokenizer_and_mappings(TEST_CONFIG)
    TEST_CONFIG.EOS_TOKEN_ID = token_to_idx['EOS']
    TEST_CONFIG.PAD_TOKEN_ID = token_to_idx['PAD']
    TEST_CONFIG.TRANSFORMER_VOCAB_SIZE = vocab_size # Set final vocab size

    # Initialize the policy network (includes the Critic Head)
    policy = make_policy(TEST_CONFIG, tokenizer)

    # 2. Data Generation
    data_gen_start = time.time() # NEW
    collected_data = generate_data(TEST_CONFIG, tokenizer, token_to_idx, idx_to_token, N_SEQUENCES_TO_GENERATE)
    data_gen_end = time.time() # NEW
    
    if not collected_data:
        print("CRITICAL: Failed to generate any valid sequences. Test aborted.")
        sys.exit(1)

    # 3. Data Preparation
    dataset = ValueDataset(collected_data)
    dataloader = DataLoader(dataset, batch_size=TEST_CONFIG.BATCH_SIZE, shuffle=True)
    
    # 4. Offline Training
    training_start = time.time() # NEW
    final_loss = train_critic(TEST_CONFIG, policy, dataloader, TEST_CONFIG.EPOCHS)
    training_end = time.time() # NEW

    # 5. Post-Training Evaluation
    # Create environment and reward shaper for evaluation
    env = ParticlePhysicsEnvironment(TEST_CONFIG)
    reward_shaper = RewardShaper(TEST_CONFIG, torch.float32)
    
    # Evaluate the first sequence from the collected data
    if collected_data:
        evaluate_sequence(
            TEST_CONFIG, policy, tokenizer, token_to_idx, idx_to_token,
            env, reward_shaper, collected_data[0]
        )

    # 6. Conclusion
    main_end_time = time.time() # NEW: End overall script timer
    
    print("\n" + "="*50)
    print("      CRITIC CONVERGENCE TEST RESULTS")
    print(f"      Total Execution Time: {(main_end_time - main_start_time):.2f}s") # NEW
    print(f"      Time spent on Data Generation: {(data_gen_end - data_gen_start):.2f}s") # NEW
    print(f"      Time spent on Training: {(training_end - training_start):.2f}s") # NEW
    print(f"      Total Sequences: {len(collected_data)}")
    print(f"      Total Epochs:    {TEST_CONFIG.EPOCHS}")
    print(f"      Final Avg MSE Loss: {final_loss:.6f}")
    print("="*50)
    
    if final_loss < 0.01:
        print("CONCLUSION: The Critic can learn the value function with low error (< 1% MSE) when trained offline on sufficient, high-quality data.")
    elif final_loss < 0.05:
        print("CONCLUSION: The Critic is moderately successful, suggesting the issue is likely insufficient on-policy updates or high variance/noise in the online RL loop.")
    else:
        print("CONCLUSION: The Critic is struggling to converge even with offline supervised learning, indicating a potential issue with the model architecture (capacity) or the target signal (density/complexity).")