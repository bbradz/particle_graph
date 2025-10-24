import torch
import torch.nn.functional as F
import random
from tqdm import tqdm

from config import get_config
import os
import json
from rl.agent import RLTrainer
from rl.network import make_policy
from rl.vocabulary import initialize_tokenizer_and_mappings
from rl.grammar import GrammarMasker
from rl.environment import ParticlePhysicsEnvironment
from rl.reward import RewardShaper
from Token2Model.check import CHECK_TO_IDX, IDX_TO_CHECK, NUM_CHECKS
from rl.dependencies import HIERARCHICAL_DEPS
from rl.utils import _compute_per_check_train_val_splits, _combined_phase_BC_reporting, _calculate_losses_on_split, save_checkpoint, load_checkpoint, log_print
from rl.plotting import PreTrainingPlotter
from torch.utils.data import DataLoader

def _pretrain_cache_paths(config):
    os.makedirs(config.PRETRAIN_CACHE_ROOT, exist_ok=True)
    meta_path = os.path.join(config.PRETRAIN_CACHE_ROOT, "pretrain_meta.json")
    data_path = os.path.join(config.PRETRAIN_CACHE_ROOT, "pretrain_data.pt")
    return meta_path, data_path


def _save_pretrain_cache(config, token_maps, pretrain_data, success_scores_tensor):
    meta_path, data_path = _pretrain_cache_paths(config)
    meta = {
        "vocab_size": len(token_maps['token_to_idx']),
        "dataset_size": len(pretrain_data['sequences_ids_list']),
        "num_checks": int(NUM_CHECKS),
        "max_len": int(getattr(config, 'MAX_LEN', 256)),
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f)
    torch.save({
        'sequences_ids_list': pretrain_data['sequences_ids_list'],
        'sequences_masks_list': [[m.cpu() for m in masks] for masks in pretrain_data['sequences_masks_list']],
        'criticality_targets_list': [t.detach().cpu() for t in pretrain_data['criticality_targets_list']],
        'exploration_targets': {int(k): v.detach().cpu() for k, v in pretrain_data['exploration_targets'].items()},
        'success_scores_tensor': None if success_scores_tensor is None else success_scores_tensor.detach().cpu(),
    }, data_path)


def _load_pretrain_cache(config):
    meta_path, data_path = _pretrain_cache_paths(config)
    if not (os.path.exists(meta_path) and os.path.exists(data_path)):
        return None, None
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        data = torch.load(data_path, map_location='cpu')
        return meta, data
    except Exception:
        return None, None


# Placeholder: dataset generator for phases A-C
def generate_pretrain_data(config, grammar, token_maps):
    # Try to load cached pretraining data
    meta_cached, data_cached = _load_pretrain_cache(config)
    if data_cached is not None:
        log_print("Loaded pretraining data from cache.", config, "data")
        return {
            'sequences_ids_list': data_cached['sequences_ids_list'],
            'sequences_masks_list': data_cached['sequences_masks_list'],
            'criticality_targets_list': data_cached['criticality_targets_list'],
            'exploration_targets': data_cached['exploration_targets'],
        }
    token_to_idx, idx_to_token = token_maps['token_to_idx'], token_maps['idx_to_token']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ParticlePhysicsEnvironment(config)
    reward_shaper = RewardShaper(config, torch.float32)

    def generate_random_sequence():
        bos_id, eos_id, pad_id = token_to_idx['BOS'], token_to_idx['EOS'], token_to_idx['PAD']
        seq_ids = [bos_id]
        grammar_masks = []
        state = grammar.initial_state()
        grammar_masks.append(grammar.get_valid_actions(state))
        state = grammar.step(state, bos_id)
        for _ in range(config.MAX_LEN - 1):
            valid_mask = grammar.get_valid_actions(state)
            valid_ids = valid_mask.nonzero(as_tuple=True)[0].tolist()
            if not valid_ids or valid_ids == [pad_id]:
                break
            next_tok_id = random.choice(valid_ids)
            seq_ids.append(next_tok_id)
            grammar_masks.append(valid_mask)
            if next_tok_id == eos_id:
                break
            state = grammar.step(state, next_tok_id)
        return seq_ids, grammar_masks

    dataset_size = int(getattr(config, 'PRE_TRAIN_DATASET_SIZE'))
    sequences_ids_list = []
    sequences_masks_list = []
    for n in range(dataset_size):
        if n % 10 == 0:
            log_print(f"Generating dataset {n}/{dataset_size}", config, "data")
        seq_ids, grammar_masks = generate_random_sequence()
        sequences_ids_list.append(seq_ids)
        sequences_masks_list.append(grammar_masks)

    targets_list = []
    i = 0
    for seq_ids in sequences_ids_list:
        if i % 10 == 0:
            log_print(f"Calculating targets for sequence {i}/{len(sequences_ids_list)}", config, "data")
        seq_tensor = torch.tensor(seq_ids, dtype=torch.long, device=device)
        episode_outcome = env.run_episode(seq_ids, idx_to_token)
        r = reward_shaper.calculate_rewards(episode_outcome, seq_tensor)
        targets_list.append(r.per_token_criticality_targets.to(device))
        i += 1

    # Initialize empty exploration targets - will be filled with data-driven approach in main()
    exploration_targets = {}

    pretrain_data = {
        'sequences_ids_list': sequences_ids_list,
        'sequences_masks_list': sequences_masks_list,
        'criticality_targets_list': targets_list,
        'exploration_targets': exploration_targets,
    }
    # Save initial cache (without success scores yet)
    _save_pretrain_cache(config, token_maps, pretrain_data, success_scores_tensor=None)
    return pretrain_data

def main():
    """Main entry point to start the RL training process."""
    config = get_config()
    
    log_print("--- Initializing Training ---", config, "training")
    
    # 1. Initialize tokenizer and get vocabulary mappings
    tokenizer, token_to_idx, idx_to_token, vocab_size = initialize_tokenizer_and_mappings(config)
    
    # Update config with actual EOS/PAD token IDs
    config.EOS_TOKEN_ID = token_to_idx['EOS']
    config.PAD_TOKEN_ID = token_to_idx['PAD']
    
    # 2. Initialize the policy network
    log_print("Loading policy network...", config, "training")
    policy = make_policy(config, tokenizer)
    device = policy.model.device if hasattr(policy, 'model') else policy.device

    # Performance knobs: enable TF32, channels-last, and try compile (PyTorch 2.x)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    try:
        policy = policy.to(memory_format=torch.channels_last)
    except Exception:
        pass
    # Guard compilation behind an env toggle to avoid CUDAGraphs issues with dynamic workloads
    try:
        if os.environ.get("TORCH_COMPILE", "0") == "1":
            policy = torch.compile(policy, mode="reduce-overhead", dynamic=False)  # type: ignore[attr-defined]
    except Exception:
        pass

    # Optional helpers
    grammar = GrammarMasker(config, token_to_idx, idx_to_token, vocab_size)
    pretrain_data = generate_pretrain_data(config, grammar, {'token_to_idx': token_to_idx, 'idx_to_token': idx_to_token})
    
    # Initialize pre-training plotter
    pretrain_plotter = PreTrainingPlotter(config.PLOT_OUTPUT_DIR)
    phases_history = {}  # Store history for all phases

    # ======================================
    # CHECKPOINT OPTIMIZATION: Skip data preparation if all phases are checkpointed
    # ======================================
    import os
    checkpoint_a_exists = os.path.exists(os.path.join(config.CHECKPOINT_DIR, "checkpoint_phase_A.pt"))
    checkpoint_b_exists = os.path.exists(os.path.join(config.CHECKPOINT_DIR, "checkpoint_phase_B.pt"))
    checkpoint_c_exists = os.path.exists(os.path.join(config.CHECKPOINT_DIR, "checkpoint_phase_C.pt"))
    
    all_phases_checkpointed = checkpoint_a_exists and checkpoint_b_exists and checkpoint_c_exists
    
    if all_phases_checkpointed:
        log_print("\n✅ All pre-training phases (A, B, C) are checkpointed. Skipping data preparation steps 2.1-2.4.", config, "checkpoint")
        log_print("   Data has already been processed and cached from previous runs.", config, "checkpoint")
    else:
        log_print(f"\n📊 Pre-training checkpoint status: A={checkpoint_a_exists}, B={checkpoint_b_exists}, C={checkpoint_c_exists}", config, "checkpoint")
        log_print("   Running data preparation steps 2.1-2.4 for phases that need training.", config, "checkpoint")

    # ======================================
    # PRECOMPUTE: Pad sequences, grammar masks, and targets to tensors on device
    # ======================================
    if not all_phases_checkpointed:
        log_print("\n[Step 2.1] Precomputing padded tensors for sequences, grammar masks, and targets...", config, "data")
        N = len(pretrain_data['sequences_ids_list'])
        T_max = min(getattr(config, 'MAX_LEN', 256), max(len(s) for s in pretrain_data['sequences_ids_list']))
        V = vocab_size
        C = int(NUM_CHECKS)
        sequences_tensor = torch.full((N, T_max), fill_value=config.PAD_TOKEN_ID, dtype=torch.long, device=device)
        grammar_mask_tensor = torch.zeros((N, T_max, V), dtype=torch.bool, device=device)
        # Store targets on CPU in compact uint8 and pin for fast transfer per-batch
        targets_cpu = torch.zeros((N, T_max, C), dtype=torch.uint8)
        for i, (seq_ids, masks_list) in enumerate(zip(pretrain_data['sequences_ids_list'], pretrain_data['sequences_masks_list'])):
            t_len = min(len(seq_ids), T_max)
            if t_len > 0:
                sequences_tensor[i, :t_len] = torch.tensor(seq_ids[:t_len], dtype=torch.long, device=device)
                m_stack = torch.stack([m.to(device) for m in masks_list[:t_len]], dim=0)
                grammar_mask_tensor[i, :t_len] = m_stack
        for i, tgt in enumerate(pretrain_data['criticality_targets_list']):
            t_len = min(tgt.size(0), T_max)
            if t_len > 0:
                targets_cpu[i, :t_len] = (tgt[:t_len] > 0.5).to(torch.uint8).cpu()
        try:
            targets_cpu = targets_cpu.pin_memory()
        except Exception:
            pass

        log_print("\n[Step 2.1a] Generating data-driven exploration targets from criticality data...", config, "data")

        # Instead of the old heuristic, we build targets by aggregating all tokens
        # that are actually critical for each check across the entire dataset.
    
        # This dictionary will hold a set of token IDs for each check index.
        critical_tokens_per_check = {idx: set() for idx in range(NUM_CHECKS)}

        # Iterate over the entire pre-training dataset (N sequences)
        for i in range(targets_cpu.size(0)):
            # Get the sequence of token IDs for this example
            seq_ids = pretrain_data['sequences_ids_list'][i]
            t_len = len(seq_ids)
        
            # Get the criticality targets for this sequence (T, C)
            criticality_for_seq = targets_cpu[i, :t_len, :]
        
            # For each check (column in the criticality tensor)
            for check_idx in range(NUM_CHECKS):
                # Find the time steps 't' where this check was critical
                critical_time_steps = (criticality_for_seq[:, check_idx] > 0).nonzero(as_tuple=True)[0]
            
                # For each of those time steps, get the corresponding token ID and add it to our set
                for t in critical_time_steps:
                    token_id = seq_ids[t.item()]
                    critical_tokens_per_check[check_idx].add(token_id)
    
        # Now, build the final exploration_targets dictionary from these sets
        exploration_targets = {}
        for check_idx, token_set in critical_tokens_per_check.items():
            # Create a zero vector for the entire vocabulary
            target_vector = torch.zeros(vocab_size, dtype=torch.float32, device=device)
            if token_set:
                # If we found any critical tokens for this check, set their indices to 1.0
                indices = torch.tensor(list(token_set), dtype=torch.long, device=device)
                target_vector[indices] = 1.0
            exploration_targets[check_idx] = target_vector
        
        # Overwrite the pretrain_data dictionary with our new, smarter targets
        pretrain_data['exploration_targets'] = exploration_targets
    
        log_print("  - Data-driven exploration targets created successfully.", config, "data")

        # Create global 5% holdout validation set for loss reporting
        N_all = len(pretrain_data['sequences_ids_list'])
        val_frac_global = 0.05
        val_size_global = max(1, int(N_all * val_frac_global))
    
        # Use fixed seed for reproducible validation split
        rng_global = random.Random(getattr(config, 'VAL_SPLIT_SEED', 1337))
        all_indices = list(range(N_all))
        val_indices_global = set(rng_global.sample(all_indices, val_size_global))
        train_indices_global = set(all_indices) - val_indices_global
    
        log_print(f"\n[Step 2.1b] Global holdout validation set created: {len(val_indices_global)}/{N_all} sequences ({len(val_indices_global)/N_all*100:.1f}%)", config, "data")

        # Build per-check train/val splits (~5% of "multi-positive" sequences go to val) ===
        check_train_indices, check_val_indices, fallback_flags = _compute_per_check_train_val_splits(
            pretrain_data['criticality_targets_list'],
            val_frac=getattr(config, 'VAL_FRACTION', 0.05),
            seed=getattr(config, 'VAL_SPLIT_SEED', 1337),
            pos_threshold=0.5
        )
        log_print("\n[Step 2.1c] Per-check train/val splits created (validation starts at ≥2 eligible sequences; ~5% target).", config, "data")

        # Dataset stats: per-check pass counts and avg non-zero scores
        log_print("\n[Step 2.2] Calculating per-check sequence success statistics...", config, "data")
        env_stats_env = ParticlePhysicsEnvironment(config)
        env_stats_reward = RewardShaper(config, torch.float32)
        # Try load cached success scores
        meta_cached, data_cached = _load_pretrain_cache(config)
        success_scores_list = []
        if data_cached is not None and data_cached.get('success_scores_tensor') is not None:
            log_print("  Loaded cached success scores.", config, "data")
            r_success_tensor_all = data_cached['success_scores_tensor']
            # Rebuild list form for existing downstream code
            success_scores_list = [r_success_tensor_all[i] for i in range(r_success_tensor_all.size(0))]
        else:
            # Reuse rewards computed earlier by recalculating once here and caching, but avoid recomputing later runs
            for i, seq_ids in enumerate(pretrain_data['sequences_ids_list']):
                if i % 10 == 0:
                    log_print(f"  Scoring sequences: {i}/{len(pretrain_data['sequences_ids_list'])}", config, "data")
                episode_outcome = env_stats_env.run_episode(seq_ids, idx_to_token)
                r = env_stats_reward.calculate_rewards(episode_outcome, torch.tensor(seq_ids))
                success_scores_list.append(r.diagnostics.get('r_success_tensor'))

        check_stats = {idx: {'non_zero_count': 0, 'total_score': 0.0, 'total_positive_tokens': 0.0, 'total_sequences': 0} for idx in range(NUM_CHECKS)}
    
        # Calculate success scores statistics
        for r_success_tensor in success_scores_list:
            if r_success_tensor is None:
                continue
            for check_idx in range(NUM_CHECKS):
                score = float(r_success_tensor[check_idx].item())
                if score > 1e-6:
                    check_stats[check_idx]['non_zero_count'] += 1
                    check_stats[check_idx]['total_score'] += score
    
        # Calculate positive target tokens statistics
        for i, tgt in enumerate(pretrain_data['criticality_targets_list']):
            tgt_cpu = tgt.detach().cpu()
            T_i = tgt_cpu.size(0)
            for check_idx in range(NUM_CHECKS):
                # Count positive tokens for this check in this sequence
                positive_tokens = (tgt_cpu[:, check_idx] > 0.5).sum().item()
                check_stats[check_idx]['total_positive_tokens'] += positive_tokens
                check_stats[check_idx]['total_sequences'] += 1

        log_print("\n--- Per-Check Success Statistics (Grouped by Level) ---", config, "data")
        log_print(f"{'Check Name':<35} | {'Sequences with Non-Zero Score':>30} | {'Avg. Score (if > 0)':>25} | {'Avg. Positive Tokens/Seq':>25}", config, "data")
        log_print("-" * 120, config, "data")
        # Save/Update cache with success scores tensor for future runs
        # Persist success scores to cache and keep an in-memory tensor for Phase C
        success_scores_tensor_all = None
        try:
            r_success_all = torch.stack([s if s is not None else torch.zeros(NUM_CHECKS) for s in success_scores_list])
            success_scores_tensor_all = r_success_all
            _save_pretrain_cache(config, {'token_to_idx': token_to_idx, 'idx_to_token': idx_to_token}, pretrain_data, r_success_all)
        except Exception:
            pass
        level_names = {4: 'Global', 3: 'Interaction', 2: 'Field', 1: 'Particle'}
        level_to_indices = {1: [], 2: [], 3: [], 4: []}
        for check_idx in range(NUM_CHECKS):
            check_name = IDX_TO_CHECK.get(check_idx, f"Check_{check_idx}")
            level = HIERARCHICAL_DEPS.get_check_hierarchical_level(check_name)
            if level in level_to_indices:
                level_to_indices[level].append(check_idx)
        for level in [4, 3, 2, 1]:
            group = level_to_indices.get(level, [])
            if not group:
                continue
            log_print(f"\n[{level_names[level]} Level]", config, "data")
            for check_idx in sorted(group, key=lambda idx: IDX_TO_CHECK.get(idx, '')):
                stats = check_stats[check_idx]
                check_name = IDX_TO_CHECK.get(check_idx, f"Check_{check_idx}")
                count = stats['non_zero_count']
                avg_score_str = 'N/A'
                if count > 0:
                    avg_score = stats['total_score'] / count
                    avg_score_str = f"{avg_score:.4f}"
            
                # Calculate average positive tokens per sequence
                avg_positive_tokens_str = 'N/A'
                if stats['total_sequences'] > 0:
                    avg_positive_tokens = stats['total_positive_tokens'] / stats['total_sequences']
                    avg_positive_tokens_str = f"{avg_positive_tokens:.2f}"
            
                log_print(f"{check_name:<35} | {count:>30} | {avg_score_str:>25} | {avg_positive_tokens_str:>25}", config, "data")
        log_print("-" * 120, config, "data")


        # ======================================
        # Create indices for balanced batching
        # ======================================
        log_print("\n[Step 2.3] Creating indices for balanced batching...", config, "data")
        check_to_indices = {idx: {'non_zero': [], 'zero': []} for idx in range(NUM_CHECKS)}
        for i, r_success_tensor in enumerate(success_scores_list):
            if r_success_tensor is None:
                continue
            for check_idx in range(NUM_CHECKS):
                score = float(r_success_tensor[check_idx].item())
                # Use a small epsilon to avoid floating point inaccuracies
                if score > 1e-6:
                    check_to_indices[check_idx]['non_zero'].append(i)
                else:
                    check_to_indices[check_idx]['zero'].append(i)
        log_print("  - Balanced batching indices created successfully.", config, "data")

        # ======================================
        # PRECOMPUTE: rarity-aware sampling + pos_weight stats
        # ======================================
        log_print("\n[Step 2.4] Precomputing rarity-aware sampling/probabilities...", config, "data")

        N = len(pretrain_data['sequences_ids_list'])
        device_cpu = torch.device("cpu")

        # Per-sequence "has any positive for check c"
        seq_has_pos = torch.zeros((N, NUM_CHECKS), dtype=torch.bool)
        pos_tokens = torch.zeros(NUM_CHECKS, dtype=torch.float32)
        total_tokens = torch.zeros(NUM_CHECKS, dtype=torch.float32)

        for i, tgt in enumerate(pretrain_data['criticality_targets_list']):
            # tgt: (T, NUM_CHECKS) on CUDA; move to CPU for aggregation
            tgt_cpu = tgt.detach().to(device_cpu)
            T_i = tgt_cpu.size(0)
            any_pos = (tgt_cpu > 0.5).any(dim=0)           # (NUM_CHECKS,)
            seq_has_pos[i] = any_pos
            pos_tokens += tgt_cpu.sum(dim=0)               # sum over tokens per check
            total_tokens += float(T_i)

        # Per-check freq over sequences
        freq_seq = seq_has_pos.sum(dim=0).clamp_min(1)     # (#seqs with ≥1 positive), avoid div0
        inv_freq_seq = 1.0 / freq_seq.float()              # rarity over sequences

        # Per-check token-level pos_weight ≈ (#neg / #pos) - dampened to reduce extreme values
        neg_tokens = (total_tokens - pos_tokens).clamp_min(0.0)
        pos_tokens_clamped = pos_tokens.clamp_min(1.0)
        raw_pos_weight = (neg_tokens / pos_tokens_clamped) # (NUM_CHECKS,)
        # Apply square root dampening to reduce extreme pos_weight values
        pos_weight_vec = torch.sqrt(raw_pos_weight.clamp_min(1.0))
        # Further stabilize extremely rare checks
        pos_weight_vec = pos_weight_vec.clamp_max(10.0)

        # Per-sequence sampling weight: sum_c [1/freq_seq[c]] over checks present in this sequence
        seq_weight = (seq_has_pos.float() * inv_freq_seq.unsqueeze(0)).sum(dim=1)  # (N,)
        # Ensure non-zero probability for sequences with no positives at all
        seq_weight = seq_weight + 1e-6
        seq_prob = (seq_weight / seq_weight.sum()).to(device_cpu)

        # Goal (check) sampling prob for Phase C: p(c) ∝ 1/freq_seq[c]
        goal_prob = (inv_freq_seq / inv_freq_seq.sum()).to(device_cpu)

        # Precompute exploration targets and per-goal pos_weight for Phase C (vectorized)
        # Build tensors on device for all goals to avoid repeated dict lookups and transfers
        vocab_size_probe = None
        for _g_idx, _vec in pretrain_data['exploration_targets'].items():
            vocab_size_probe = int(_vec.numel())
            break
        if vocab_size_probe is None:
            vocab_size_probe = 0
        exploration_targets_mat = torch.zeros(NUM_CHECKS, vocab_size_probe, device=device)
        valid_goal_mask = torch.zeros(NUM_CHECKS, dtype=torch.bool, device=device)
        pos_weight_goal = torch.ones(NUM_CHECKS, device=device)
        for g in range(NUM_CHECKS):
            vec = pretrain_data['exploration_targets'].get(g)
            if vec is None:
                continue
            vec_dev = vec.to(device)
            exploration_targets_mat[g] = vec_dev
            num_pos_g = vec_dev.sum()

            # Only treat a goal as valid if it actually has ≥1 positive tokens
            if num_pos_g.item() >= 1:
                valid_goal_mask[g] = True
                raw_pos_weight_g = (vec_dev.numel() - num_pos_g) / num_pos_g
                # Apply square root dampening to reduce extreme pos_weight values
                pos_weight_goal[g] = torch.sqrt(raw_pos_weight_g.clamp_min(1.0)).clamp_max(10.0).detach()
            else:
                # No positives => do not train on this goal (keep mask False and neutral pos_weight)
                valid_goal_mask[g] = False
                pos_weight_goal[g] = 1.0

    # End of data preparation steps 2.1-2.4
    # ======================================

    # ======================================
    # PHASE A: Grammar Pre-training (Uniform)
    # ======================================
    # Freeze auxiliary heads
    for name, module in getattr(policy, 'named_children', lambda: [])():
        pass
    for head_name in ['value_head', 'reward_head', 'criticality_head', 'exploration_q_head', 'rnd_predictor_network']:
        if hasattr(policy, head_name):
            for p in getattr(policy, head_name).parameters():
                p.requires_grad = False
    for p in policy.parameters():
        if not hasattr(p, 'requires_grad') or p.requires_grad is None:
            continue
    optimizer_a = torch.optim.Adam((p for p in policy.parameters() if p.requires_grad), lr=config.PRE_TRAIN_LEARNING_RATE)
    scaler_a = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    scheduler_a = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_a, 'min', patience=5, factor=0.5)

    checkpoint_loaded = load_checkpoint("A", policy, optimizer_a, scheduler_a, config, device)
    log_print(f"DEBUG: Phase A checkpoint loaded = {checkpoint_loaded}", config, "training")
    
    # Force Phase A to run for debugging (comment out this line to restore normal behavior)
    # checkpoint_loaded = False
    # log_print("DEBUG: Forcing Phase A to run (checkpoint_loaded = False)", config, "training")
    
    if not checkpoint_loaded:
        log_print("\n--- Starting Phase A: Grammar Pre-training (Batched Whole Dataset) ---", config, "training")
        log_print(f"  > Using effective batch size: {min(config.PRE_TRAIN_BATCH_SIZE, 8)} (capped to prevent OOM)", config, "training")
        policy.train()
        pbar_a = tqdm(range(config.PRE_TRAIN_GRAMMAR_STEPS), disable=False, desc="Phase A")
        # Initialize validation loss variables
        last_val_grammar_loss = None
        
        # Initialize Phase A history tracking
        phase_a_history = {
            'step': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'grad_norm': []
        }
        
        for step in pbar_a:
            optimizer_a.zero_grad()
            total_loss = 0.0
            total_valid_sequences = 0
            grad_accum_steps = max(1, int(getattr(config, 'GRAD_ACCUMULATION_STEPS', 1)))
            accum_counter = 0
            last_grad_norm = 0.0
            
            # Process dataset in smaller batches using precomputed tensors
            dataset_size = sequences_tensor.size(0)
            effective_batch_size = min(config.PRE_TRAIN_BATCH_SIZE, 8)
            for batch_start in range(0, dataset_size, effective_batch_size):
                batch_end = min(batch_start + effective_batch_size, dataset_size)
                idxs = slice(batch_start, batch_end)
                batch_tensor = sequences_tensor[idxs]
                autocast_enabled_a = torch.cuda.is_available()
                with torch.amp.autocast('cuda', enabled=autocast_enabled_a):
                    try:
                        torch.compiler.cudagraph_mark_step_begin()
                    except Exception:
                        pass
                    outputs = policy(batch_tensor)
                    batch_logits = outputs.logits  # (B, T, V)

            # Vectorized Grammar KL over batch
            B, T, V = batch_logits.size()
            # Slice precomputed valid mask tensor (B, T, V)
            valid_mask = grammar_mask_tensor[idxs, :T, :]

            log_probs = F.log_softmax(batch_logits, dim=-1)
            denom = valid_mask.sum(dim=-1, keepdim=True).clamp_min(1)
            target_dist = valid_mask.float() / denom
            kl_elem = F.kl_div(log_probs, target_dist, reduction='none')  # (B,T,V)
            kl_masked = (kl_elem * valid_mask).sum(dim=-1)  # (B,T) sum over V
            step_has_valid = (denom.squeeze(-1) > 0)  # (B,T)
            # Per-sequence mean over valid steps
            steps_per_seq = step_has_valid.sum(dim=-1).clamp_min(1)  # (B,)
            seq_loss = (kl_masked * step_has_valid).sum(dim=-1) / steps_per_seq  # (B,)
            seq_has_any = step_has_valid.any(dim=-1)
            
            # DEBUG: Print batch statistics and example invalid sequence
            batch_total_sequences = B
            valid_count = int(seq_has_any.sum().item())
            invalid_count = batch_total_sequences - valid_count
            
            # Find and print an example invalid sequence if one exists
            if invalid_count > 0:
                invalid_indices = (~seq_has_any).nonzero(as_tuple=True)[0]
                if len(invalid_indices) > 0:
                    invalid_idx = invalid_indices[0].item()
                    invalid_seq = batch_tensor[invalid_idx]
                    # Convert to tokens for readability
                    invalid_tokens = [idx_to_token.get(token_id.item(), f"UNK_{token_id.item()}") for token_id in invalid_seq if token_id.item() != config.PAD_TOKEN_ID]
                    
                    # Also check the grammar mask for this sequence
                    invalid_grammar_mask = valid_mask[invalid_idx]
                    valid_steps_count = step_has_valid[invalid_idx].sum().item()
            
            if seq_has_any.any():
                batch_kl = seq_loss[seq_has_any].mean()
                total_loss += batch_kl.detach()
                total_valid_sequences += int(seq_has_any.sum().item())
                # Backprop on this mini-batch
                scaler_a.scale(batch_kl / float(grad_accum_steps)).backward()
            else:
                # If no sequences have valid steps, still count them and add minimal loss
                total_valid_sequences += B
                # Add a small loss to ensure we have something to track
                total_loss += torch.tensor(0.001, device=device, dtype=torch.float32)
            
            accum_counter += 1
            # Step on accumulation boundary or at end
            if (accum_counter % grad_accum_steps == 0) or (batch_end >= dataset_size):
                    # Optional: compute grad norm before step
                    scaler_a.unscale_(optimizer_a)
                    total_norm_val = 0.0
                    for p in policy.parameters():
                        if p.requires_grad and p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm_val += (param_norm.item() ** 2)
                    last_grad_norm = total_norm_val ** 0.5
                    scaler_a.step(optimizer_a)
                    scaler_a.update()
                    optimizer_a.zero_grad()
        
            # Always update progress bar, even if total_valid_sequences is 0
            if total_valid_sequences > 0:
                avg_loss = total_loss / total_valid_sequences
            else:
                avg_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            
            current_lr = optimizer_a.param_groups[0]['lr']
            
            # Track history for plotting
            phase_a_history['step'].append(step + 1)
            phase_a_history['train_loss'].append(avg_loss.item())
            phase_a_history['learning_rate'].append(current_lr)
            phase_a_history['grad_norm'].append(last_grad_norm)
            
            # Calculate validation loss every few steps
            if (step + 1) % max(1, config.PRE_TRAIN_GRAMMAR_STEPS // 10) == 0:
                losses_a = _calculate_losses_on_split(
                    policy=policy, device=device, sequences_tensor=sequences_tensor, targets_cpu=targets_cpu,
                    grammar_mask_tensor=grammar_mask_tensor, exploration_targets_mat=None, pos_weight_vec=pos_weight_vec,
                    pos_weight_goal=pos_weight_goal, train_indices=train_indices_global, val_indices=val_indices_global,
                    vocab_size=vocab_size, valid_goal_mask=None
                )
                last_val_grammar_loss = losses_a['val']['grammar_loss']
                phase_a_history['val_loss'].append(last_val_grammar_loss)
                
                # Generate plots periodically
                if (step + 1) % max(1, config.PRE_TRAIN_GRAMMAR_STEPS // 5) == 0:
                    log_print(f"\n--- Generating Phase A Plots (Step {step + 1}) ---", config, "training")
                    try:
                        pretrain_plotter.plot_phase_losses("Phase A", phase_a_history)
                        log_print("Phase A plots generated successfully.", config, "training")
                    except Exception as e:
                        log_print(f"Warning: Failed to generate Phase A plots: {e}", config, "training")
            else:
                phase_a_history['val_loss'].append(None)
            
            # Show validation loss if available, otherwise show N/A
            val_display = "N/A"
            if last_val_grammar_loss is not None:
                val_display = f"Gram={last_val_grammar_loss:.4f}"
            
            pbar_a.set_postfix({
                'Gram': f"{avg_loss.item():.4f}",
                'Grad': f"{last_grad_norm:.3f}",
                'LR': f"{current_lr:.2e}",
                'Val': val_display
            })

        # Save checkpoint at the end of Phase A
        save_checkpoint("A", policy, optimizer_a, scheduler_a, config)
        
        # Store Phase A history and generate final plots
        phases_history['Phase A'] = phase_a_history
        log_print("\n--- Generating Final Phase A Plots ---", config, "training")
        try:
            pretrain_plotter.plot_phase_losses("Phase A", phase_a_history)
            log_print("Final Phase A plots generated successfully.", config, "training")
        except Exception as e:
            log_print(f"Warning: Failed to generate final Phase A plots: {e}", config, "training")

    # ======================================
    # PHASE B: Criticality Pre-training (BCE) with Grammar Regularization
    # ======================================
    # Ensure only the correct parameters are trained
    for name, param in policy.named_parameters():
        name_ok = (
            ('criticality' in name) or ('transformer' in name) or ('embedding' in name)
            or ('token_embedding' in name) or ('block_depth_embedding' in name) or ('intra_block_position_embedding' in name)
            or ('model' in name and 'lm_head' not in name)
        )
        param.requires_grad = bool(name_ok)

    optimizer_b = torch.optim.Adam((p for p in policy.parameters() if p.requires_grad), lr=config.PRE_TRAIN_LEARNING_RATE)
    scaler_b = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    scheduler_b = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_b, 'min', patience=5, factor=0.5)

    if not load_checkpoint("B", policy, optimizer_b, scheduler_b, config, device):
        log_print("\n--- Starting Phase B: Criticality Pre-training with Grammar Regularization (Batched Whole Dataset) ---", config, "training")
        log_print(f"  > Using effective batch size: {min(config.PRE_TRAIN_BATCH_SIZE, 8)} (capped to prevent OOM)", config, "training")
        policy.train()
        lambda_grammar_b = 0.2
        log_print(f"  > Using grammar regularization with lambda = {lambda_grammar_b}", config, "training")
        log_print("  > Phase B uses rarity-weighted sequence sampling and positive-only weighting via pos_weight_vec per check.", config, "training")

        steps_b = max(1, config.PRE_TRAIN_CRITICALITY_STEPS)
        batch_size = config.PRE_TRAIN_BATCH_SIZE
        pos_weight_vec_dev = pos_weight_vec.to(device)

        pbar_b = tqdm(range(steps_b), disable=False, desc="Phase B")
        # Initialize validation loss variables
        last_val_crit_loss = None
        last_val_grammar_loss = None
        
        # Initialize Phase B history tracking
        phase_b_history = {
            'step': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'grad_norm': [],
            'crit_loss': [],
            'grammar_loss': []
        }
        
        for step in pbar_b:
            optimizer_b.zero_grad()

            total_crit_loss = 0.0
            total_grammar_loss = 0.0
            total_valid_sequences = 0
            grad_accum_steps_b = max(1, int(getattr(config, 'GRAD_ACCUMULATION_STEPS', 1)))
            accum_counter_b = 0
            last_grad_norm_b = 0.0

            # Process dataset using precomputed tensors
            dataset_size = sequences_tensor.size(0)
            effective_batch_size = min(config.PRE_TRAIN_BATCH_SIZE, 8)
            for batch_start in range(0, dataset_size, effective_batch_size):
                batch_end = min(batch_start + effective_batch_size, dataset_size)
                idxs = slice(batch_start, batch_end)
                batch_tensor = sequences_tensor[idxs]
                autocast_enabled_b = torch.cuda.is_available()
                with torch.amp.autocast('cuda', enabled=autocast_enabled_b):
                    try:
                        torch.compiler.cudagraph_mark_step_begin()
                    except Exception:
                        pass
                    outputs = policy(batch_tensor)
                hidden_states = outputs.hidden_states[-1]  # (B, T, H)
                logits = outputs.logits                    # (B, T, V)
                crit_logits = policy.criticality(hidden_states)  # (B, T, C)

                # Vectorized Criticality BCE across batch over all tokens (no last-positive masking)
                B, T, C = crit_logits.size()
                # Bring CPU-pinned uint8 targets to device for this batch only, cast to float
                target_tensor = targets_cpu[idxs, :T, :].to(device=device, non_blocking=True).to(torch.float32)

                # BCE with pos weight broadcast over (B,T,C);
                # then mean over checks per token before applying the time mask
                pos_w = pos_weight_vec_dev.view(1, 1, C)
                bce_elem = F.binary_cross_entropy_with_logits(crit_logits, target_tensor, reduction='none')  # (B,T,C)
                weight = 1.0 + (pos_w - 1.0) * target_tensor
                weighted = bce_elem * weight  # (B,T,C)
                per_token_mean_over_checks = weighted.mean(dim=-1)  # (B,T)
                # Mean over all tokens and batch (no masking to last positive)
                crit_loss_batch = per_token_mean_over_checks.mean()
                total_crit_loss += crit_loss_batch.detach()

                # Vectorized Grammar KL over batch (with gradients for proper regularization)
                B, T, V = logits.size()
                valid_mask = grammar_mask_tensor[idxs, :T, :]
                log_probs = F.log_softmax(logits, dim=-1)  # no detach - let gradients flow
                denom = valid_mask.sum(dim=-1, keepdim=True).clamp_min(1)
                target_dist = valid_mask.float() / denom
                kl_elem = F.kl_div(log_probs, target_dist, reduction='none')
                kl_masked = (kl_elem * valid_mask).sum(dim=-1)  # (B,T)
                step_has_valid = (denom.squeeze(-1) > 0)
                steps_per_seq = step_has_valid.sum(dim=-1).clamp_min(1)
                seq_loss = (kl_masked * step_has_valid).sum(dim=-1) / steps_per_seq
                seq_has_any = step_has_valid.any(dim=-1)
                batch_kl = seq_loss[seq_has_any].mean() if seq_has_any.any() else torch.tensor(0.0, device=device)
                total_grammar_loss += batch_kl.detach()
                # Count all sequences in this batch
                total_valid_sequences += B
                
                # Backprop per mini-batch with accumulation
                batch_total_loss = crit_loss_batch + lambda_grammar_b * batch_kl
                scaler_b.scale(batch_total_loss / float(grad_accum_steps_b)).backward()
                accum_counter_b += 1
                if (accum_counter_b % grad_accum_steps_b == 0) or (batch_end >= dataset_size):
                    scaler_b.unscale_(optimizer_b)
                    # Gradient clipping to stabilize updates
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                    total_norm_val_b = 0.0
                    for p in policy.parameters():
                        if p.requires_grad and p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm_val_b += (param_norm.item() ** 2)
                    last_grad_norm_b = total_norm_val_b ** 0.5
                    scaler_b.step(optimizer_b)
                    scaler_b.update()
                    optimizer_b.zero_grad()

            if total_valid_sequences > 0:
                avg_crit_loss = total_crit_loss / total_valid_sequences
                avg_grammar_loss = total_grammar_loss / total_valid_sequences
                current_lr = optimizer_b.param_groups[0]['lr']
                
                # Track history for plotting
                phase_b_history['step'].append(step + 1)
                phase_b_history['train_loss'].append((avg_crit_loss + lambda_grammar_b * avg_grammar_loss).item())
                phase_b_history['crit_loss'].append(avg_crit_loss.item())
                phase_b_history['grammar_loss'].append(avg_grammar_loss.item())
                phase_b_history['learning_rate'].append(current_lr)
                phase_b_history['grad_norm'].append(last_grad_norm_b)
                
                # Calculate validation loss every few steps
                if (step + 1) % max(1, steps_b // 20) == 0:
                    losses_b = _calculate_losses_on_split(
                        policy=policy, device=device, sequences_tensor=sequences_tensor, targets_cpu=targets_cpu,
                        grammar_mask_tensor=grammar_mask_tensor, exploration_targets_mat=None, pos_weight_vec=pos_weight_vec,
                        pos_weight_goal=pos_weight_goal, train_indices=train_indices_global, val_indices=val_indices_global,
                        vocab_size=vocab_size, valid_goal_mask=None
                    )
                    last_val_crit_loss = losses_b['val']['crit_loss']
                    last_val_grammar_loss = losses_b['val']['grammar_loss']
                    phase_b_history['val_loss'].append(last_val_crit_loss + lambda_grammar_b * last_val_grammar_loss)
                    
                    # Generate plots periodically
                    if (step + 1) % max(1, steps_b // 5) == 0:
                        log_print(f"\n--- Generating Phase B Plots (Step {step + 1}) ---", config, "training")
                        try:
                            pretrain_plotter.plot_phase_losses("Phase B", phase_b_history)
                            log_print("Phase B plots generated successfully.", config, "training")
                        except Exception as e:
                            log_print(f"Warning: Failed to generate Phase B plots: {e}", config, "training")
                        
                        # Generate per-check performance plots
                        try:
                            # Initialize check_performance if it doesn't exist
                            if not hasattr(pretrain_plotter, 'check_performance_b'):
                                pretrain_plotter.check_performance_b = {}
                            
                            # Accumulate data over time
                            for check_name, check_idx in CHECK_TO_IDX.items():
                                if check_name not in pretrain_plotter.check_performance_b:
                                    pretrain_plotter.check_performance_b[check_name] = {
                                        'train_loss': [],
                                        'val_loss': [],
                                        'steps': []
                                    }
                                
                                pretrain_plotter.check_performance_b[check_name]['train_loss'].append(losses_b['train']['crit_loss'])
                                pretrain_plotter.check_performance_b[check_name]['val_loss'].append(losses_b['val']['crit_loss'])
                                pretrain_plotter.check_performance_b[check_name]['steps'].append(step + 1)
                            
                            pretrain_plotter.plot_per_check_performance("Phase B", pretrain_plotter.check_performance_b)
                            log_print("Phase B per-check performance plots generated successfully.", config, "training")
                        except Exception as e:
                            log_print(f"Warning: Failed to generate Phase B per-check performance plots: {e}", config, "training")
                else:
                    phase_b_history['val_loss'].append(None)
                
                # Show validation loss if available, otherwise show N/A
                val_display = "N/A"
                if last_val_crit_loss is not None and last_val_grammar_loss is not None:
                    val_display = f"Crit={last_val_crit_loss:.4f} Gram={last_val_grammar_loss:.4f}"
                
                pbar_b.set_postfix({
                    'Total': f"{(avg_crit_loss + lambda_grammar_b * avg_grammar_loss).item():.4f}",
                    'Crit': f"{avg_crit_loss.item():.4f}",
                    'Gram': f"{avg_grammar_loss.item():.4f}",
                    'Grad': f"{last_grad_norm_b:.3f}",
                    'Val': val_display
                })

        # Combined reporting after Phase B
        _combined_phase_BC_reporting(
            policy=policy,
            device=device,
            sequences_tensor=sequences_tensor,
            targets_cpu=targets_cpu,
            idx_to_token=idx_to_token,
            pretrain_data=pretrain_data,
            vocab_size=vocab_size,
            check_train_indices=check_train_indices,
            check_val_indices=check_val_indices,
            fallback_flags=fallback_flags,
            valid_goal_mask=valid_goal_mask,
            grammar_mask_tensor=grammar_mask_tensor,
            train_indices_global=train_indices_global,
            val_indices_global=val_indices_global,
            config=config
        )

        # Save checkpoint at the end of Phase B
        save_checkpoint("B", policy, optimizer_b, scheduler_b, config)
        
        # Store Phase B history and generate final plots
        phases_history['Phase B'] = phase_b_history
        log_print("\n--- Generating Final Phase B Plots ---", config, "training")
        try:
            pretrain_plotter.plot_phase_losses("Phase B", phase_b_history)
            log_print("Final Phase B plots generated successfully.", config, "training")
        except Exception as e:
            log_print(f"Warning: Failed to generate final Phase B plots: {e}", config, "training")

    # ================================================
    # PHASE C: Multi-Task Pre-training (Criticality + Exploration) with Grammar Regularization
    # ================================================
    # Activate parameters for both heads and the shared body
    for name, param in policy.named_parameters():
        name_ok = (
            ('criticality' in name) or
            ('exploration_q_head' in name) or
            ('transformer' in name) or 
            ('embedding' in name) or
            ('token_embedding' in name) or 
            ('block_depth_embedding' in name) or 
            ('intra_block_position_embedding' in name) or
            ('model' in name and 'lm_head' not in name)
        )
        param.requires_grad = bool(name_ok)

    optimizer_c = torch.optim.Adam((p for p in policy.parameters() if p.requires_grad), lr=config.PRE_TRAIN_LEARNING_RATE)
    scaler_c = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    scheduler_c = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_c, 'min', patience=5, factor=0.5)

    if not load_checkpoint("C", policy, optimizer_c, scheduler_c, config, device):
        log_print("\n--- Starting Phase C: Multi-Task Pre-training (Criticality + Dynamic Q-Head) ---", config, "training")
        policy.train()

        # Define weights for the composite loss
        lambda_crit = 1.0
        lambda_q = 1.0  # Weight for the dynamic Q-head loss
        lambda_grammar_c = 0.2
        log_print(f"  > Using composite loss with weights: Crit={lambda_crit}, Q={lambda_q}, Gram={lambda_grammar_c}", config, "training")

        steps_c = max(1, config.PRE_TRAIN_EXPLORATION_STEPS)
        batch_size = config.PRE_TRAIN_BATCH_SIZE

        # --- Precompute and move required tensors to the device ---
        pos_weight_vec_dev = pos_weight_vec.to(device)
        exploration_targets_mat_dev = exploration_targets_mat.to(device).transpose(0, 1) # Shape: (V, C)

        pbar_c = tqdm(range(steps_c), disable=False, desc="Phase C")
        # Initialize validation loss variables
        last_val_crit_loss = None
        last_val_q_loss = None
        last_val_grammar_loss = None
        
        # Initialize Phase C history tracking
        phase_c_history = {
            'step': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'grad_norm': [],
            'crit_loss': [],
            'q_loss': [],
            'grammar_loss': []
        }
        
        for step in pbar_c:
            optimizer_c.zero_grad()
            total_q_loss = 0.0
            total_crit_loss = 0.0
            total_grammar_loss = 0.0
            total_valid_sequences_c = 0
            last_grad_norm_c = 0.0
            
            # Process dataset in smaller batches to manage memory
            dataset_size = sequences_tensor.size(0)
            effective_batch_size = min(batch_size, 8) # Keep batch small for memory
            for batch_start in range(0, dataset_size, effective_batch_size):
                batch_end = min(batch_start + effective_batch_size, dataset_size)
                idxs = slice(batch_start, batch_end)
                batch_seq_tensor = sequences_tensor[idxs]
                
                autocast_enabled = torch.cuda.is_available()
                with torch.amp.autocast('cuda', enabled=autocast_enabled):
                    # --- Main Forward Pass ---
                    outputs = policy(batch_seq_tensor)
                    hidden_states = outputs.hidden_states[-1]
                    logits = outputs.logits

                    # --- 1. Criticality Head Loss (same as Phase B) ---
                    crit_logits = policy.criticality(hidden_states)
                    B, T, C = crit_logits.size()
                    target_tensor_crit = targets_cpu[idxs, :T, :].to(device, non_blocking=True).float()
                    pos_w_crit = pos_weight_vec_dev.view(1, 1, C)
                    # Use binary_cross_entropy_with_logits for autocast safety
                    bce_elem_crit = F.binary_cross_entropy_with_logits(crit_logits, target_tensor_crit, reduction='none')
                    weight_crit = 1.0 + (pos_w_crit - 1.0) * target_tensor_crit
                    crit_loss_batch = (bce_elem_crit * weight_crit).mean()
                    
                    # 2. Dynamic Q-Head Loss (Maximizing Predictions for Legal Tokens at Critical Steps)
                    q_preds_matrix_logits = policy.exploration_q_values(hidden_states) # Shape: (B, T, V, C)

                    # a. Define the LOSS MASK: We only care about tokens that are BOTH grammatically
                    #    legal AND occur at a timestep/check that is marked as critical.
                    batch_grammar_mask = grammar_mask_tensor[idxs, :T, :] # Shape: (B, T, V)
                    criticality_target_mask = targets_cpu[idxs, :T, :].to(device, non_blocking=True).bool() # Shape: (B, T, C)

                    # Combine the masks via broadcasting to find the exact elements for our loss.
                    # (B,T,V,1) & (B,T,1,C) -> (B,T,V,C)
                    loss_application_mask = batch_grammar_mask.unsqueeze(-1) & criticality_target_mask.unsqueeze(2)

                    # If there are no such tokens in this batch, the loss is zero.
                    if not loss_application_mask.any():
                        q_loss_batch = torch.tensor(0.0, device=device)
                    else:
                        # b. Select only the logits where the mask is True.
                        preds_for_loss = q_preds_matrix_logits[loss_application_mask]
                        
                        # c. The target for ALL of these selected logits is 1.0.
                        # We create a tensor of ones with the same shape as our predictions.
                        targets_for_loss = torch.ones_like(preds_for_loss)

                        # d. Calculate the BCE loss. This will push all selected logits towards +infinity,
                        # which corresponds to a predicted probability of 1.0.
                        q_loss_batch = F.binary_cross_entropy_with_logits(preds_for_loss, targets_for_loss)

                    # --- 3. Grammar Regularization Loss (same as Phase B) ---
                    valid_mask_gram = grammar_mask_tensor[idxs, :T, :]
                    log_probs = F.log_softmax(logits, dim=-1)  # Remove detach to allow gradients to flow
                    denom = valid_mask_gram.sum(dim=-1, keepdim=True).clamp_min(1)
                    target_dist = valid_mask_gram.float() / denom
                    kl_elem_gram = F.kl_div(log_probs, target_dist, reduction='none')
                    kl_masked_gram = (kl_elem_gram * valid_mask_gram).sum(dim=-1)
                    step_has_valid_gram = (denom.squeeze(-1) > 0)
                    steps_per_seq_gram = step_has_valid_gram.sum(dim=-1).clamp_min(1)
                    seq_loss_gram = (kl_masked_gram * step_has_valid_gram).sum(dim=-1) / steps_per_seq_gram
                    batch_kl_c = seq_loss_gram[step_has_valid_gram.any(dim=-1)].mean() if step_has_valid_gram.any() else torch.tensor(0.0, device=device)

                    # --- 4. Combine losses and backpropagate ---
                    batch_total_loss = (lambda_crit * crit_loss_batch) + \
                                    (lambda_q * q_loss_batch) + \
                                    (lambda_grammar_c * batch_kl_c)
                    
                    scaler_c.scale(batch_total_loss).backward()
                    
                    total_crit_loss += crit_loss_batch.detach()
                    total_q_loss += q_loss_batch.detach()
                    total_grammar_loss += batch_kl_c.detach()
                    total_valid_sequences_c += batch_seq_tensor.size(0)

            # --- Optimizer Step and Logging (after processing all mini-batches) ---
            scaler_c.unscale_(optimizer_c)
            torch.nn.utils.clip_grad_norm_((p for p in policy.parameters() if p.requires_grad), 1.0)
            
            # Calculate gradient norm
            total_norm_val_c = 0.0
            for p in policy.parameters():
                if p.requires_grad and p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm_val_c += (param_norm.item() ** 2)
            last_grad_norm_c = total_norm_val_c ** 0.5
            
            scaler_c.step(optimizer_c)
            scaler_c.update()
            
            if total_valid_sequences_c > 0:
                num_batches = max(1, dataset_size // effective_batch_size)
                avg_crit_loss = total_crit_loss / num_batches
                avg_q_loss = total_q_loss / num_batches
                avg_grammar_loss = total_grammar_loss / num_batches
                avg_total_loss = avg_crit_loss + avg_q_loss + avg_grammar_loss
                current_lr = optimizer_c.param_groups[0]['lr']

                # Track history for plotting
                phase_c_history['step'].append(step + 1)
                phase_c_history['train_loss'].append(avg_total_loss.item())
                phase_c_history['crit_loss'].append(avg_crit_loss.item())
                phase_c_history['q_loss'].append(avg_q_loss.item())
                phase_c_history['grammar_loss'].append(avg_grammar_loss.item())
                phase_c_history['learning_rate'].append(current_lr)
                phase_c_history['grad_norm'].append(last_grad_norm_c)

                # Calculate validation loss every few steps
                if (step + 1) % max(1, steps_c // 20) == 0:
                    losses_c = _calculate_losses_on_split(
                        policy=policy, device=device, sequences_tensor=sequences_tensor, targets_cpu=targets_cpu,
                        grammar_mask_tensor=grammar_mask_tensor, exploration_targets_mat=exploration_targets_mat_dev,
                        pos_weight_vec=pos_weight_vec, pos_weight_goal=pos_weight_goal, train_indices=train_indices_global,
                        val_indices=val_indices_global, vocab_size=vocab_size, valid_goal_mask=valid_goal_mask
                    )
                    last_val_crit_loss = losses_c['val']['crit_loss']
                    last_val_q_loss = losses_c['val']['q_loss']
                    last_val_grammar_loss = losses_c['val']['grammar_loss']
                    phase_c_history['val_loss'].append(last_val_crit_loss + last_val_q_loss + lambda_grammar_c * last_val_grammar_loss)
                    
                    # Generate plots periodically
                    if (step + 1) % max(1, steps_c // 5) == 0:
                        log_print(f"\n--- Generating Phase C Plots (Step {step + 1}) ---", config, "training")
                        try:
                            pretrain_plotter.plot_phase_losses("Phase C", phase_c_history)
                            log_print("Phase C plots generated successfully.", config, "training")
                        except Exception as e:
                            log_print(f"Warning: Failed to generate Phase C plots: {e}", config, "training")
                        
                        # Generate per-check performance plots
                        try:
                            # Initialize check_performance if it doesn't exist
                            if not hasattr(pretrain_plotter, 'check_performance_c'):
                                pretrain_plotter.check_performance_c = {}
                            
                            losses_c = _calculate_losses_on_split(
                                policy=policy, device=device, sequences_tensor=sequences_tensor, targets_cpu=targets_cpu,
                                grammar_mask_tensor=grammar_mask_tensor, exploration_targets_mat=exploration_targets_mat_dev,
                                pos_weight_vec=pos_weight_vec, pos_weight_goal=pos_weight_goal, train_indices=train_indices_global,
                                val_indices=val_indices_global, vocab_size=vocab_size, valid_goal_mask=valid_goal_mask
                            )
                            
                            # Accumulate data over time
                            for check_name, check_idx in CHECK_TO_IDX.items():
                                if check_name not in pretrain_plotter.check_performance_c:
                                    pretrain_plotter.check_performance_c[check_name] = {
                                        'train_loss': [],
                                        'val_loss': [],
                                        'steps': []
                                    }
                                
                                pretrain_plotter.check_performance_c[check_name]['train_loss'].append(losses_c['train']['crit_loss'])
                                pretrain_plotter.check_performance_c[check_name]['val_loss'].append(losses_c['val']['crit_loss'])
                                pretrain_plotter.check_performance_c[check_name]['steps'].append(step + 1)
                            
                            pretrain_plotter.plot_per_check_performance("Phase C", pretrain_plotter.check_performance_c)
                            log_print("Phase C per-check performance plots generated successfully.", config, "training")
                        except Exception as e:
                            log_print(f"Warning: Failed to generate Phase C per-check performance plots: {e}", config, "training")
                else:
                    phase_c_history['val_loss'].append(None)
                
                # Show validation loss if available, otherwise show N/A
                val_display = "N/A"
                if last_val_crit_loss is not None and last_val_q_loss is not None and last_val_grammar_loss is not None:
                    val_display = f"Crit={last_val_crit_loss:.6f} Q={last_val_q_loss:.6f} Gram={last_val_grammar_loss:.6f}"
                
                pbar_c.set_postfix({
                    'Total': f"{avg_total_loss.item():.4f}",
                    'Crit': f"{avg_crit_loss.item():.4f}",
                    'Q': f"{avg_q_loss.item():.4f}",
                    'Gram': f"{avg_grammar_loss.item():.4f}",
                    'Val': val_display
                })

        # Combined reporting after Phase C
        _combined_phase_BC_reporting(
            policy=policy,
            device=device,
            sequences_tensor=sequences_tensor,
            targets_cpu=targets_cpu,
            idx_to_token=idx_to_token,
            pretrain_data=pretrain_data,
            vocab_size=vocab_size,
            check_train_indices=check_train_indices,
            check_val_indices=check_val_indices,
            fallback_flags=fallback_flags,
            valid_goal_mask=valid_goal_mask,
            grammar_mask_tensor=grammar_mask_tensor,
            train_indices_global=train_indices_global,
            val_indices_global=val_indices_global,
            config=config
        )

        # Save checkpoint at the end of Phase C
        save_checkpoint("C", policy, optimizer_c, scheduler_c, config)
        
        # Store Phase C history and generate final plots
        phases_history['Phase C'] = phase_c_history
        log_print("\n--- Generating Final Phase C Plots ---", config, "training")
        try:
            pretrain_plotter.plot_phase_losses("Phase C", phase_c_history)
            log_print("Final Phase C plots generated successfully.", config, "training")
        except Exception as e:
            log_print(f"Warning: Failed to generate final Phase C plots: {e}", config, "training")
        
        # Generate phase comparison plot
        log_print("\n--- Generating Pre-training Phase Comparison Plot ---", config, "training")
        try:
            pretrain_plotter.plot_phase_comparison(phases_history)
            log_print("Pre-training phase comparison plot generated successfully.", config, "training")
        except Exception as e:
            log_print(f"Warning: Failed to generate phase comparison plot: {e}", config, "training")
        
        # Generate final per-check performance plots
        log_print("\n--- Generating Final Per-Check Performance Plots ---", config, "training")
        try:
            # Get final validation losses
            final_losses = _calculate_losses_on_split(
                policy=policy, device=device, sequences_tensor=sequences_tensor, targets_cpu=targets_cpu,
                grammar_mask_tensor=grammar_mask_tensor, exploration_targets_mat=exploration_targets_mat_dev,
                pos_weight_vec=pos_weight_vec, pos_weight_goal=pos_weight_goal, train_indices=train_indices_global,
                val_indices=val_indices_global, vocab_size=vocab_size, valid_goal_mask=valid_goal_mask
            )
            
            # Combine all accumulated data for final plotting
            final_check_performance = {}
            for check_name, check_idx in CHECK_TO_IDX.items():
                final_check_performance[check_name] = {
                    'train_loss': [],
                    'val_loss': [],
                    'steps': []
                }
                
                # Combine Phase B and Phase C data with adjusted step numbers
                if hasattr(pretrain_plotter, 'check_performance_b') and check_name in pretrain_plotter.check_performance_b:
                    final_check_performance[check_name]['train_loss'].extend(pretrain_plotter.check_performance_b[check_name]['train_loss'])
                    final_check_performance[check_name]['val_loss'].extend(pretrain_plotter.check_performance_b[check_name]['val_loss'])
                    # Keep Phase B steps as-is (they start around step 40)
                    final_check_performance[check_name]['steps'].extend(pretrain_plotter.check_performance_b[check_name]['steps'])
                
                if hasattr(pretrain_plotter, 'check_performance_c') and check_name in pretrain_plotter.check_performance_c:
                    final_check_performance[check_name]['train_loss'].extend(pretrain_plotter.check_performance_c[check_name]['train_loss'])
                    final_check_performance[check_name]['val_loss'].extend(pretrain_plotter.check_performance_c[check_name]['val_loss'])
                    # Adjust Phase C steps to be continuous after Phase B
                    phase_c_steps = pretrain_plotter.check_performance_c[check_name]['steps']
                    if phase_c_steps:
                        # Find the last step from Phase B
                        last_phase_b_step = max(final_check_performance[check_name]['steps']) if final_check_performance[check_name]['steps'] else 0
                        # Adjust Phase C steps to start after Phase B
                        adjusted_phase_c_steps = [step - phase_c_steps[0] + last_phase_b_step + 1 for step in phase_c_steps]
                        final_check_performance[check_name]['steps'].extend(adjusted_phase_c_steps)
            
            pretrain_plotter.plot_per_check_performance("Final", final_check_performance)
            log_print("Final per-check performance plots generated successfully.", config, "training")
        except Exception as e:
            log_print(f"Warning: Failed to generate final per-check performance plots: {e}", config, "training")

        # ================================================
        # BENCHMARKING BEFORE RL PHASE
        # ================================================
        log_print("\n--- Benchmarking Final Pre-trained Weights ---", config, "eval")
        _combined_phase_BC_reporting(
            policy=policy,
            device=device,
            sequences_tensor=sequences_tensor,
            targets_cpu=targets_cpu,
            idx_to_token=idx_to_token,
            pretrain_data=pretrain_data,
            vocab_size=vocab_size,
            check_train_indices=check_train_indices,
            check_val_indices=check_val_indices,
            fallback_flags=fallback_flags,
            valid_goal_mask=valid_goal_mask,
            grammar_mask_tensor=grammar_mask_tensor,
            train_indices_global=train_indices_global,
            val_indices_global=val_indices_global,
            config=config
        )
        log_print("--- Benchmarking Complete ---", config, "eval")
        
        # Save the final pre-trained model before starting RL
        save_checkpoint("pre_RL", policy, optimizer_c, scheduler_c, config)

    # ==============================
    # PHASE D: Full RL
    # ==============================
    for p in policy.parameters():
        p.requires_grad = True
    
    # 3. Instantiate the trainer
    trainer = RLTrainer(
        cfg=config,
        policy=policy,
        tokenizer=tokenizer,
        token_maps=(token_to_idx, idx_to_token, vocab_size)
    )
    
    # 4. Start training
    log_print("--- Starting Training Loop ---", config, "training")
    trainer.train()
    log_print("--- Training Finished ---", config, "training")

if __name__ == "__main__":
    main()