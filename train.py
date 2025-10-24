import torch
from config import get_config
from rl.agent import RLTrainer
from rl.network import make_policy
from rl.vocabulary import initialize_tokenizer_and_mappings

def main():
    """Main entry point to start the RL training process."""
    config = get_config()
    
    print("--- Initializing Training ---")
    
    # 1. Initialize tokenizer and get vocabulary mappings
    tokenizer, token_to_idx, idx_to_token, vocab_size = initialize_tokenizer_and_mappings(config)
    
    # Update config with actual EOS/PAD token IDs
    config.EOS_TOKEN_ID = token_to_idx['EOS']
    config.PAD_TOKEN_ID = token_to_idx['PAD']
    
    # 2. Initialize the policy network
    print("Loading policy network...")
    policy = make_policy(config, tokenizer)
    
    # 3. Instantiate the trainer
    trainer = RLTrainer(
        cfg=config,
        policy=policy,
        tokenizer=tokenizer,
        token_maps=(token_to_idx, idx_to_token, vocab_size)
    )
    
    # 4. Start training
    print("--- Starting Training Loop ---")
    trainer.train()
    print("--- Training Finished ---")

if __name__ == "__main__":
    main()