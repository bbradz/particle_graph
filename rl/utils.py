import torch
import torch.nn.functional as F
import random
from typing import Tuple, Dict, List
from Token2Model.check import CHECK_TO_IDX, IDX_TO_CHECK


def log_print(message: str, config=None, log_type: str = "general", force: bool = False):
    """
    Centralized logging function that respects config settings.
    
    Args:
        message: The message to print
        config: Config object with logging settings
        log_type: Type of log message ("general", "training", "data", "eval", "checkpoint", "debug")
        force: If True, always print regardless of config settings
    """
    if force:
        print(message)
        return
    
    if config is None:
        # If no config provided, always print (fallback behavior)
        print(message)
        return
    
    # Check if logging is enabled
    if not config.ENABLE_LOGGING:
        return
    
    # Check specific log type settings
    should_print = False
    
    if log_type == "general":
        should_print = True
    elif log_type == "training":
        should_print = config.PRINT_TRAINING_INIT or config.PRINT_PHASE_HEADERS or config.PRINT_PHASE_PROGRESS
    elif log_type == "data":
        should_print = config.PRINT_DATA_GENERATION or config.PRINT_DATASET_STATS or config.PRINT_CACHE_INFO
    elif log_type == "eval":
        should_print = config.PRINT_EVALUATION_RESULTS or config.PRINT_PER_CHECK_RESULTS or config.PRINT_DETAILED_TABLES
    elif log_type == "checkpoint":
        should_print = config.PRINT_CHECKPOINT_INFO
    elif log_type == "debug":
        should_print = config.DEBUG_PRINTS
    else:
        should_print = True  # Default to printing for unknown types
    
    if should_print:
        if config.REDIRECT_DEBUG_TO_FILE and hasattr(config, 'DEBUG_OUTPUT_FILE'):
            # Redirect to file
            with open(config.DEBUG_OUTPUT_FILE, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
        else:
            print(message)

def _find_last_valid_closure(sequence_list: List[int], token_ids: Dict[str, int]) -> int:
    """
    Finds the index of the last token that concludes a structurally balanced block.
    Returns 0 (BOS token) if no other valid closure is found.
    """
    openers = {token_ids['ITRACT'], token_ids['FIELD'], token_ids['PARTICLE']}
    token_map = {
        token_ids['END_ITRACT']: token_ids['ITRACT'],
        token_ids['END_FIELD']: token_ids['FIELD'],
        token_ids['END_PARTICLE']: token_ids['PARTICLE'],
    }

    best_pos = 0  # Default to the position of the BOS token

    open_stack = []
    for i, token_id in enumerate(sequence_list):
        if token_id in openers:
            open_stack.append(token_id)
        elif token_id in token_map:
            if open_stack and open_stack[-1] == token_map[token_id]:
                open_stack.pop()
                # If the stack is now empty, this is a valid top-level closure point
                if not open_stack:
                    best_pos = i

    return best_pos


def finalize_sequence_inplace(
    seq_tensor: torch.LongTensor,
    attn_mask: torch.BoolTensor,
    token_ids: Dict[str, int],
    max_len: int
) -> Tuple[int, int]:
    """
    Cleans up a generated sequence by truncating it to the last valid closure point
    WITHOUT adding an EOS token. This function modifies the input tensors in-place 
    for efficiency.
    
    The sequence will cut off naturally at the last valid closure point, ensuring
    that the percentage of sequences which reach EOS equals the percentage of 
    sequences which were actually resolved/completed.
    """
    effective_len = int(attn_mask.sum())
    if effective_len == 0:
        return -1, 0
        
    sequence_list = seq_tensor[:effective_len].tolist()

    eos_id = token_ids['EOS']
    pad_id = token_ids['PAD']

    # Case 1: The sequence already contains a valid EOS token.
    if eos_id in sequence_list:
        eos_pos = sequence_list.index(eos_id)
        # Pad everything after the EOS token
        if eos_pos + 1 < max_len:
            seq_tensor[eos_pos + 1:] = pad_id
            attn_mask[eos_pos + 1:] = False
        return eos_pos, 0

    # Case 2: The sequence was truncated. Find the last valid structural closure.
    last_closure_pos = _find_last_valid_closure(sequence_list, token_ids)
    
    # The sequence will end at the last closure point (no EOS added)
    cut_pos = last_closure_pos + 1
    
    # Ensure cut_pos is at least 1 (to keep BOS) if sequence is not empty
    if cut_pos < 1 and effective_len > 0:
         cut_pos = 1

    truncated_count = effective_len - cut_pos
    
    # In-place modification of the tensors - just pad without adding EOS
    if cut_pos < max_len:
        # Pad everything from cut_pos onwards
        seq_tensor[cut_pos:] = pad_id
        attn_mask[cut_pos:] = False
            
    # If the sequence is full, just pad the last position
    elif cut_pos == max_len:
        seq_tensor[max_len - 1] = pad_id
        attn_mask[max_len - 1] = False
        cut_pos = max_len - 1
        
    return cut_pos, truncated_count


def _compute_per_check_train_val_splits(criticality_targets_list, val_frac=0.05, seed=1337, pos_threshold=0.5):
    """
    Creates per-check train/validation splits. For each check, it identifies
    sequences with at least one positive token ("eligible pool"). If the pool is
    large enough, it creates a dedicated validation set from it. Otherwise, all
    data is used for training for that specific check.
    """
    rng = random.Random(seed)
    num_checks = criticality_targets_list[0].size(-1)
    N_all = len(criticality_targets_list)
    
    # 1. Identify eligible sequences for each check
    eligible_sequences_per_check = {c: [] for c in range(num_checks)}
    for i, tgt in enumerate(criticality_targets_list):
        pos_counts = (tgt > pos_threshold).sum(dim=0)
        for c in range(num_checks):
            if int(pos_counts[c].item()) >= 1:
                eligible_sequences_per_check[c].append(i)

    check_train_indices = {}
    check_val_indices = {}
    fallback_flags = {}

    # 2. Create a unique split for each check
    for c in range(num_checks):
        pool = eligible_sequences_per_check[c]
        n = len(pool)
        
        # If not enough positive sequences to make a split, use all data for training
        if n < 2:
            check_val_indices[c] = set()
            check_train_indices[c] = set(range(N_all))
            fallback_flags[c] = True # Mark that no dedicated validation set was created
        else:
            # Create a dedicated validation set from the pool of positive sequences
            k = max(1, int(round(val_frac * n)))
            k = min(k, n - 1) # Ensure at least one positive sequence remains for training
            
            val_set_for_c = set(rng.sample(pool, k))
            
            # The validation set for this check contains only these chosen sequences
            check_val_indices[c] = val_set_for_c
            # The training set is ALL other sequences
            check_train_indices[c] = set(range(N_all)) - val_set_for_c
            fallback_flags[c] = False

    return check_train_indices, check_val_indices, fallback_flags


def _print_phaseB_seq_table_for_check(seq_ids, preds_seq_tc, targets_seq_tc, cidx, idx_to_token, check_name, config=None):
    """
    Pretty-print per-token Phase-B predictions vs targets for one sequence and one check.
    preds_seq_tc, targets_seq_tc: (T, C)
    """
    display_name = check_name.replace('*check', '').strip('*')
    log_print(f"\nCheck: {display_name}", config, "eval")
    log_print(f"  {'Token':<30} | {'Target':>6} | {'Pred':>8}", config, "eval")
    log_print("  " + "-" * 50, config, "eval")
    T = min(len(seq_ids), preds_seq_tc.size(0), targets_seq_tc.size(0))
    for t in range(T):
        token_str = idx_to_token.get(int(seq_ids[t]), "UNK")
        target_val = float(targets_seq_tc[t, cidx].item())
        pred_val = float(preds_seq_tc[t, cidx].item())
        log_print(f"  {token_str:<30} | {target_val:>6.1f} | {pred_val:>8.4f}", config, "eval")


def _calculate_losses_on_split(
    policy, device, sequences_tensor, targets_cpu, grammar_mask_tensor,
    exploration_targets_mat, pos_weight_vec, pos_weight_goal,
    train_indices, val_indices, vocab_size, valid_goal_mask=None
):
    """
    Calculate average losses (Criticality, Q-head, Grammar) on train and validation splits.
    Returns a dictionary with loss values for both splits.
    """
    policy.eval()
    
    def compute_losses_on_indices(indices):
        if not indices:
            return {'crit_loss': float('nan'), 'q_loss': float('nan'), 'grammar_loss': float('nan')}
        
        indices_list = list(indices)
        batch_size = 4  # Small batch size for evaluation
        
        total_crit_loss = 0.0
        total_q_loss = 0.0
        total_grammar_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(indices_list), batch_size):
                batch_indices = indices_list[i:i + batch_size]
                batch_seq = sequences_tensor[batch_indices]
                batch_targets = targets_cpu[batch_indices].to(device).float()
                batch_grammar = grammar_mask_tensor[batch_indices]
                
                # Forward pass
                outputs = policy(batch_seq)
                hidden_states = outputs.hidden_states[-1]
                logits = outputs.logits
                
                # 1. Criticality Loss
                crit_logits = policy.criticality(hidden_states)
                B, T, C = crit_logits.size()
                pos_w = pos_weight_vec.to(device).view(1, 1, C)
                bce_elem = F.binary_cross_entropy_with_logits(crit_logits, batch_targets[:, :T, :], reduction='none')
                weight = 1.0 + (pos_w - 1.0) * batch_targets[:, :T, :]
                crit_loss = (bce_elem * weight).mean()
                
                # 2. Q-head Loss (validation) - Match training: manual class-balanced BCE (50/50)
                q_loss = torch.tensor(0.0, device=device)
                if exploration_targets_mat is not None and valid_goal_mask is not None:
                    try:
                        q_preds_matrix = policy.exploration_q_values(hidden_states) # (B, T, V, C)

                        # Define the LOSS MASK: We only care about tokens that are BOTH grammatically
                        # legal AND occur at a timestep/check that is marked as critical.
                        batch_grammar_mask = batch_grammar[:, :T, :] # Shape: (B, T, V)
                        criticality_target_mask = batch_targets[:, :T, :].bool() # Shape: (B, T, C)

                        # Combine the masks via broadcasting to find the exact elements for our loss.
                        # (B,T,V,1) & (B,T,1,C) -> (B,T,V,C)
                        loss_application_mask = batch_grammar_mask.unsqueeze(-1) & criticality_target_mask.unsqueeze(2)

                        # If there are no such tokens in this batch, the loss is zero.
                        if loss_application_mask.any():
                            # Select only the logits where the mask is True.
                            preds_for_loss = q_preds_matrix[loss_application_mask]
                            
                            # The target for ALL of these selected logits is 1.0.
                            targets_for_loss = torch.ones_like(preds_for_loss)

                            # Calculate the BCE loss.
                            q_loss = F.binary_cross_entropy_with_logits(preds_for_loss, targets_for_loss)
                    except Exception:
                        q_loss = torch.tensor(0.0, device=device)
                
                # 3. Grammar Loss
                valid_mask = batch_grammar[:, :T, :]
                log_probs = F.log_softmax(logits, dim=-1)
                denom = valid_mask.sum(dim=-1, keepdim=True).clamp_min(1)
                target_dist = valid_mask.float() / denom
                kl_elem = F.kl_div(log_probs, target_dist, reduction='none')
                kl_masked = (kl_elem * valid_mask).sum(dim=-1)
                step_has_valid = (denom.squeeze(-1) > 0)
                steps_per_seq = step_has_valid.sum(dim=-1).clamp_min(1)
                seq_loss = (kl_masked * step_has_valid).sum(dim=-1) / steps_per_seq
                grammar_loss = seq_loss[step_has_valid.any(dim=-1)].mean() if step_has_valid.any() else torch.tensor(0.0, device=device)
                
                total_crit_loss += crit_loss.item()
                total_q_loss += q_loss.item()
                total_grammar_loss += grammar_loss.item()
                num_batches += 1
        
        if num_batches > 0:
            return {
                'crit_loss': total_crit_loss / num_batches,
                'q_loss': total_q_loss / num_batches,
                'grammar_loss': total_grammar_loss / num_batches
            }
        else:
            return {'crit_loss': float('nan'), 'q_loss': float('nan'), 'grammar_loss': float('nan')}
    
    train_losses = compute_losses_on_indices(train_indices)
    val_losses = compute_losses_on_indices(val_indices)
    
    policy.train()
    return {'train': train_losses, 'val': val_losses}


def _combined_phase_BC_reporting(
    policy, device,
    sequences_tensor, targets_cpu, idx_to_token,
    pretrain_data, vocab_size,
    check_train_indices, check_val_indices, fallback_flags,
    valid_goal_mask=None, grammar_mask_tensor=None, train_indices_global=None, val_indices_global=None, config=None
):
    """
    Runs *batched* Phase B and Phase C style evaluations and prints:
      - Phase B Table A: per-token preds vs targets for a single maximally-informative sequence (for each check)
      - Phase B Table B: mean(pred) and accuracy on critical vs non-critical tokens, for TRAIN and VAL
      - Phase C Table C: mean(sigmoid(Q)) and accuracy for critical vs non-critical vocab tokens, for TRAIN and VAL
    """
    def _fmt_mean(v):
        return "NA" if (v is None or (isinstance(v, float) and (v != v))) else f"{v:6.3f}"

    policy.eval()
    N, T_max = sequences_tensor.size(0), sequences_tensor.size(1)
    C = targets_cpu.size(-1)

    # ====== Phase B: get predictions for all sequences in batches ======
    all_phaseB_preds = torch.zeros((N, T_max, C), dtype=torch.float32, device='cpu')
    batch_sz = 8
    with torch.no_grad():
        for b in range(0, N, batch_sz):
            be = min(b + batch_sz, N)
            batch = sequences_tensor[b:be]                           # (B, T)
            outputs = policy(batch)                                  # logits + hidden states
            hidden_states = outputs.hidden_states[-1]                # (B, T, H)
            preds = policy.criticality(hidden_states).float()        # (B, T, C) in [0,1]
            Bt, Tt, Ct = preds.size()
            all_phaseB_preds[b:be, :Tt, :Ct] = preds.detach().cpu()

    # Choose a single "maximally-informative" sequence (max sum of targets)
    best_idx = -1
    best_sum = -1.0
    for i, tgt in enumerate(pretrain_data['criticality_targets_list']):
        s = tgt.sum().item()
        if s > best_sum:
            best_sum, best_idx = s, i

    # ====== Print Phase B Table A for each check on that chosen sequence ======
    if best_idx >= 0:
        seq_ids = pretrain_data['sequences_ids_list'][best_idx]
        preds_seq = all_phaseB_preds[best_idx]                        # (T, C) on CPU
        targets_seq = pretrain_data['criticality_targets_list'][best_idx].cpu()  # (T, C)
        log_print("\n--- Combined Reporting (Phase B/C): Per-Check Per-Token (Phase B preds vs targets) ---", config, "eval")
        ordered_checks = sorted(list(CHECK_TO_IDX.keys()), key=lambda name: name)
        for check_name in ordered_checks:
            cidx = CHECK_TO_IDX[check_name]
            _print_phaseB_seq_table_for_check(seq_ids, preds_seq, targets_seq, cidx, idx_to_token, check_name, config)

    # ====== Phase B Table B: dataset means and accuracy for TRAIN vs VAL per check ======
    log_print("\n--- Phase B Dataset Means: mean(pred) and accuracy on critical vs non-critical tokens (TRAIN vs VAL) ---", config, "eval")
    targets_all = targets_cpu.float()   # (N, T, C)

    def split_means_and_accuracy(seq_ids_split, cidx):
        if not seq_ids_split:
            return (float('nan'), 0, float('nan'), 0, float('nan'), float('nan'), 0, float('nan'), float('nan'), float('nan'))
        
        # Convert to list for indexing
        seq_ids_list = list(seq_ids_split)
        
        tgt_split = targets_all[seq_ids_list, :, cidx]        # (S, T)
        pred_split = all_phaseB_preds[seq_ids_list, :, cidx]   # (S, T)
        crit_mask = (tgt_split > 0.5)
        noncrit_mask = ~crit_mask
        n_crit = int(crit_mask.sum().item())
        n_non  = int(noncrit_mask.sum().item())
        m_crit = float(pred_split[crit_mask].mean().item()) if n_crit > 0 else float('nan')
        m_non  = float(pred_split[noncrit_mask].mean().item()) if n_non  > 0 else float('nan')
        
        # Probabilities for accuracy and distance
        pred_probs = torch.sigmoid(pred_split)
        
        # Accuracies
        crit_correct = ((pred_probs > 0.5) & crit_mask).sum().item()
        noncrit_correct = ((pred_probs < 0.5) & noncrit_mask).sum().item()
        total_tokens = n_crit + n_non
        accuracy = (crit_correct + noncrit_correct) / total_tokens if total_tokens > 0 else float('nan')
        pos_acc = (crit_correct / n_crit) if n_crit > 0 else float('nan')
        neg_acc = (noncrit_correct / n_non) if n_non > 0 else float('nan')
        
        # Distance (MSE between predicted prob and target)
        dist = float(F.mse_loss(pred_probs, tgt_split.float()).item()) if total_tokens > 0 else float('nan')
        
        # Count sequences that actually contributed positive examples
        contributing_seqs = int(crit_mask.any(dim=1).sum().item())  # Sequences with at least one critical token
        
        return (m_crit, n_crit, m_non, n_non, accuracy, total_tokens, contributing_seqs, pos_acc, neg_acc, dist)

    for check_name, cidx in sorted(CHECK_TO_IDX.items(), key=lambda kv: kv[0]):
        train_ids = list(check_train_indices.get(cidx, set()))
        val_ids   = list(check_val_indices.get(cidx, set()))
        tr_mc, tr_nc, tr_mn, tr_nn, tr_acc, tr_total, tr_contrib_seqs, tr_pos_acc, tr_neg_acc, tr_dist = split_means_and_accuracy(train_ids, cidx)
        va_mc, va_nc, va_mn, va_nn, va_acc, va_total, va_contrib_seqs, va_pos_acc, va_neg_acc, va_dist = split_means_and_accuracy(val_ids, cidx)

        # Add fallback indicator to the printout
        val_label = " (fallback)" if fallback_flags.get(cidx, False) else ""
        
        disp = check_name.replace('_check', '').strip('_')
        
        # More descriptive print format with critical/non-critical means, accuracy, Pos/Neg accuracies, and distance
        train_str = (
            f"TRAIN μ(crit)={_fmt_mean(tr_mc)} [toks={tr_nc}] μ(non)={_fmt_mean(tr_mn)} [toks={tr_nn}] "
            f"Acc={tr_acc:.3f} Pos={tr_pos_acc:.3f} Neg={tr_neg_acc:.3f} Dist={tr_dist:.4f} "
            f"from {tr_contrib_seqs}/{len(train_ids)} seqs"
        )
        val_str = (
            f"VAL{val_label} μ(crit)={_fmt_mean(va_mc)} [toks={va_nc}] μ(non)={_fmt_mean(va_mn)} [toks={va_nn}] "
            f"Acc={va_acc:.3f} Pos={va_pos_acc:.3f} Neg={va_neg_acc:.3f} Dist={va_dist:.4f} "
            f"from {va_contrib_seqs}/{len(val_ids)} seqs"
        )
        log_print(f"Check: {disp:<28} | {train_str} | {val_str}", config, "eval")

    # ====== Phase C: Q-head "Maximize Legal" Performance (at Critical Timesteps) ======
    log_print("\n--- Phase C Dataset Means: Q-Head 'Maximize Legal' Performance (at Critical Timesteps) ---", config, "eval")

    # Precompute last_hidden per sequence without assuming policy.hidden_size
    all_last_hidden_list = []
    with torch.no_grad():
        for b in range(0, N, batch_sz):
            be = min(b + batch_sz, N)
            batch = sequences_tensor[b:be]
            outputs = policy(batch)
            hs = outputs.hidden_states[-1][:, -1, :].detach().cpu()   # (B, H)
            all_last_hidden_list.append(hs)
    all_last_hidden = torch.cat(all_last_hidden_list, dim=0)  # (N, H)

    def q_head_report_for_split(seq_ids_split):
        """
        Calculates metrics for the Q-head's new task: predicting 1.0 for all legal
        tokens at critical timesteps.
        """
        if not seq_ids_split:
            return (float('nan'), float('nan'), float('nan'), 0)

        all_legal_preds = []
        all_illegal_preds = []
        
        for i in seq_ids_split:
            h = all_last_hidden[i].to(device).unsqueeze(0).unsqueeze(1)  # (1, 1, H)
            
            # Get Q-head predictions (average over the check dim as it's now a shared task)
            q_logits_matrix = policy.exploration_q_values(h)
            q_logits = q_logits_matrix.mean(dim=-1).squeeze(0)
            q_probs = torch.sigmoid(q_logits) # (1, V)

            # Get the timesteps in this sequence that are critical for ANY check
            crit_mask_for_seq = targets_cpu[i, :, :].any(dim=-1).bool() # Shape (T,)
            
            if crit_mask_for_seq.any():
                # Get grammar masks (our targets) for the critical timesteps
                legal_mask_at_crit_steps = grammar_mask_tensor[i, crit_mask_for_seq, :] # (num_crit_steps, V)
                
                # Get Q-head predictions and expand to match
                q_probs_at_crit_steps = q_probs.expand(legal_mask_at_crit_steps.size(0), -1)

                # Separate predictions for legal and illegal tokens
                all_legal_preds.append(q_probs_at_crit_steps[legal_mask_at_crit_steps])
                all_illegal_preds.append(q_probs_at_crit_steps[~legal_mask_at_crit_steps])

        if not all_legal_preds:
            return (float('nan'), float('nan'), float('nan'), 0)

        # Concatenate all collected predictions
        legal_preds_tensor = torch.cat(all_legal_preds)
        illegal_preds_tensor = torch.cat(all_illegal_preds)

        # Calculate metrics
        mean_legal_pred = legal_preds_tensor.mean().item()
        mean_illegal_pred = illegal_preds_tensor.mean().item()
        
        # Distance is MSE from the target of 1.0 for legal predictions
        dist = F.mse_loss(legal_preds_tensor, torch.ones_like(legal_preds_tensor)).item()
        
        return (dist, mean_legal_pred, mean_illegal_pred, len(seq_ids_split))

    # --- New Print Logic ---
    log_print("\n--- Phase C Dataset Means: Q-Head Performance (at Critical Timesteps) ---", config, "eval")
    log_print("    > 'Dist' is MSE from target of 1.0 for legal tokens.", config, "eval")
    log_print("    > 'μ(legal)' is the avg prediction for grammatically LEGAL tokens.", config, "eval")
    log_print("    > 'μ(illegal)' is the avg prediction for grammatically ILLEGAL tokens.", config, "eval")

    train_ids_global_list = list(train_indices_global)
    val_ids_global_list = list(val_indices_global)

    tr_dist, tr_legal, tr_illegal, tr_seqs = q_head_report_for_split(train_ids_global_list)
    va_dist, va_legal, va_illegal, va_seqs = q_head_report_for_split(val_ids_global_list)

    train_str = f"TRAIN Dist={tr_dist:.4f} μ(legal)={_fmt_mean(tr_legal)} μ(illegal)={_fmt_mean(tr_illegal)} [seqs={tr_seqs}]"
    val_str = f"VAL Dist={va_dist:.4f} μ(legal)={_fmt_mean(va_legal)} μ(illegal)={_fmt_mean(va_illegal)} [seqs={va_seqs}]"
    log_print(f"{'Q-Head Overall':<28} | {train_str} | {val_str}", config, "eval")

    policy.train()


def save_checkpoint(phase_name: str, policy: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler, config):
    """Saves model, optimizer, and scheduler state for a given phase."""
    import os
    
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_phase_{phase_name}.pt")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # For compiled models, get the original model
    model_to_save = getattr(policy, '_orig_mod', policy)

    save_dict = {
        'policy_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    
    torch.save(save_dict, checkpoint_path)
    log_print(f"✅ Checkpoint for Phase '{phase_name}' saved to {checkpoint_path}", config, "checkpoint")


def load_checkpoint(phase_name: str, policy: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler, config, device) -> bool:
    """Loads a checkpoint if it exists. Returns True if successful, False otherwise."""
    import os
    
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_phase_{phase_name}.pt")
    
    if not os.path.exists(checkpoint_path):
        log_print(f"ℹ️ No checkpoint found for Phase '{phase_name}'. Starting from scratch.", config, "checkpoint")
        return False
        
    log_print(f"➡️ Loading checkpoint for Phase '{phase_name}' from {checkpoint_path}...", config, "checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # For compiled models, load into the original model
    model_to_load = getattr(policy, '_orig_mod', policy)

    model_to_load.load_state_dict(checkpoint['policy_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    log_print(f"✅ Checkpoint for Phase '{phase_name}' loaded successfully.", config, "checkpoint")
    return True