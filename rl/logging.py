from typing import List, Dict, Any, Tuple, Optional
import json
import torch
import torch.nn.functional as F
from collections import defaultdict
from Token2Model.check import CHECK_TO_IDX, IDX_TO_CHECK

def _trim_pad_tokens(tokens: List[str]) -> List[str]:
    """Removes PAD tokens from the end of a token list."""
    try:
        first_pad_index = tokens.index("PAD")
        return tokens[:first_pad_index]
    except ValueError:
        return tokens

def format_sequence_with_full_details(
    tokens: List[str],
    immediate_rewards: torch.Tensor,
    target_rewards: torch.Tensor,
    predicted_values: torch.Tensor,
    criticality_scores: torch.Tensor,
    exploration_head_scores: torch.Tensor,
    check_idx: int
) -> str:
    """
    Formats a token sequence with detailed reward, value, criticality, and exploration info.
    """
    tokens = _trim_pad_tokens(tokens)
    if not tokens: return ""
    
    output = []
    header = f"{'Token':<20} | {'r_t':>6} | {'R_target':>8} | {'V(s_t)':>8} | {'Crit':>6} | {'Expl_head':>9} | {'Crit*Expl':>9}"
    output.append(header)
    output.append("-" * len(header))
    
    max_len = len(tokens)
    for i, token in enumerate(tokens):
        r_t = immediate_rewards[i].item() if i < immediate_rewards.numel() else float('nan')
        R_target = target_rewards[i].item() if i < target_rewards.numel() else float('nan')
        V_st = predicted_values[i].item() if i < predicted_values.numel() else float('nan')
        crit_score = criticality_scores[i, check_idx].item() if i < criticality_scores.size(0) and criticality_scores.size(1) > check_idx else float('nan')
        expl_head = exploration_head_scores[i, check_idx].item() if i < exploration_head_scores.size(0) and exploration_head_scores.size(1) > check_idx else float('nan')
        crit_times_expl = crit_score * expl_head if not (torch.isnan(torch.tensor(crit_score)) or torch.isnan(torch.tensor(expl_head))) else float('nan')
        output.append(f"{token:<20} | {r_t:>6.3f} | {R_target:>8.3f} | {V_st:>8.3f} | {crit_score:>6.3f} | {expl_head:>9.3f} | {crit_times_expl:>9.3f}")
        
    return "\n".join(output)

def format_sequence_with_indents(tokens: List[str]) -> str:
    """
    Formats a token sequence by grouping related tokens onto single lines 
    with hierarchical indentation.
    """
    tokens = _trim_pad_tokens(tokens)
    if not tokens: return ""

    output, current_line_tokens, indent_level = [], [], 0

    def flush_line():
        nonlocal current_line_tokens
        if current_line_tokens:
            output.append("  " * indent_level + " ".join(current_line_tokens))
            current_line_tokens = []

    for token in tokens:
        if token in ['ITRACT', 'FIELD', 'PARTICLE', 'END_FIELD', 'END_ITRACT', 'BOS', 'EOS']:
            flush_line()

        if token == 'ITRACT': indent_level = 0
        elif token == 'FIELD': indent_level = 1
        elif token == 'PARTICLE': indent_level = 2
        elif token == 'END_FIELD': indent_level = 1
        elif token == 'END_ITRACT': indent_level = 0
        
        current_line_tokens.append(token)
        
        if token == 'END_PARTICLE':
            flush_line()

    flush_line()
    return "\n".join(output)

def format_sequence_with_rewards(
    tokens: List[str],
    per_token_instantaneous_rewards: torch.Tensor,
    per_token_target_rewards: torch.Tensor,
    predicted_token_rewards: torch.Tensor,
    print_sequence: bool = True
) -> str:
    """
    Formats a token sequence with rewards, showing raw reward, discounted target, and prediction.
    """
    tokens = _trim_pad_tokens(tokens)
    if not tokens: return ""

    output, current_line_tokens, indent_level = [], [], 0
    current_line_raw_rewards, current_line_target_rewards, current_line_predicted_rewards = [], [], []

    def flush_line():
        nonlocal current_line_tokens, current_line_raw_rewards, current_line_target_rewards, current_line_predicted_rewards
        if current_line_tokens:
            token_part = "  " * indent_level + " ".join(current_line_tokens)
            raw_str = [f"{r:.3f}" for r in current_line_raw_rewards]
            target_str = [f"{r:.3f}" for r in current_line_target_rewards]
            predicted_str = [f"{r:.3f}" for r in current_line_predicted_rewards]
            output.append(f"{token_part:<80} [Raw R: {','.join(raw_str)}] [Discount R_Sum: {','.join(target_str)}] [Predicted R_Sum: {','.join(predicted_str)}]")
            current_line_tokens, current_line_raw_rewards, current_line_target_rewards, current_line_predicted_rewards = [], [], [], []

    for i, token in enumerate(tokens):
        if token in ['ITRACT', 'FIELD', 'PARTICLE', 'END_FIELD', 'END_ITRACT', 'BOS', 'EOS']:
            flush_line()

        if token == 'ITRACT': indent_level = 0
        elif token == 'FIELD': indent_level = 1
        elif token == 'PARTICLE': indent_level = 2
        elif token == 'END_FIELD': indent_level = 1
        elif token == 'END_ITRACT': indent_level = 0

        current_line_tokens.append(token)
        if i < per_token_instantaneous_rewards.size(0):
            current_line_raw_rewards.append(per_token_instantaneous_rewards[i].item())
        if i < per_token_target_rewards.size(0):
            current_line_target_rewards.append(per_token_target_rewards[i].item())
        if i < predicted_token_rewards.size(0):
            current_line_predicted_rewards.append(predicted_token_rewards[i].item())

        if token == 'END_PARTICLE':
            flush_line()

    flush_line()
    return "\n".join(output) if print_sequence else "[Sequence content hidden]"

def _get_check_category_priority(check_name: str) -> int:
    if check_name.startswith('(global)'): return 0
    if check_name.startswith('(itract)'): return 1
    if check_name.startswith('(field)'): return 2
    if check_name.startswith('(particle)'): return 3
    return 99

def log_batch_summary(
    batch_idx: int, outcome: Any, sequence_tokens: List[str], rewards_info: Any,
    predicted_score_vectors: torch.Tensor, per_token_instantaneous_rewards: torch.Tensor, 
    per_token_target_scores: torch.Tensor, averaged_check_type_scores: torch.Tensor, 
    active_checks: List[str], per_token_exploration_bonus: torch.Tensor,
    criticality_scores: torch.Tensor, exploration_head_scores: torch.Tensor,
    print_sequence: bool = True, print_detailed_checklist: bool = True, 
    print_per_check_rewards_detail: bool = False,
    step_number: Optional[int] = None,
    timings: Optional[Dict[str, float]] = None
):
    print("\n" + "="*40 + f" Sequence {batch_idx + 1} " + "="*40)
    
    trimmed_tokens = _trim_pad_tokens(sequence_tokens)
    if print_sequence:
        print(f"Generated Sequence (Length: {len(trimmed_tokens)}):\n{format_sequence_with_indents(trimmed_tokens)}")
    else:
        print(f"Generated Sequence (Length: {len(trimmed_tokens)}) - [Sequence content hidden]")

    # Overall Summary
    total_seq_reward = (
        rewards_info.get('scalar_total_reward', 0.0)
        if isinstance(rewards_info, dict)
        else getattr(rewards_info, 'scalar_total_reward', 0.0)
    )
    if step_number is not None:
        print(f"\n--- Overall Summary: Step {step_number} ---\n  Total Sequence Reward (Curriculum-Masked): {total_seq_reward:.4f}")
    else:
        print(f"\n--- Overall Summary ---\n  Total Sequence Reward (Curriculum-Masked): {total_seq_reward:.4f}")
    
    # Pipeline Timings (New Section)
    if timings:
        # Define the grand total time for all percentages
        total_pipeline_time = timings.get('generate_trajectories_total', 0.0) + timings.get('env_and_reward_total', 0.0)
        if total_pipeline_time == 0.0: total_pipeline_time = 1e-9

        print("\n--- Pipeline Timings (Averages per Step) ---")
        
        # Helper to print a line of timing info
        def print_timing_line(label, time_key, indent=0, is_subtotal=False):
            time_val = timings.get(time_key, 0.0)
            percent_of_total = (time_val / total_pipeline_time) * 100
            
            line_prefix = f"{'  ' * indent}{label:<38}:"
            line_suffix = f"{time_val:.4f}s ({percent_of_total:.1f}%)"
            
            if is_subtotal:
                print(f"{line_prefix} {'=' * 10} {line_suffix}")
            else:
                print(f"{line_prefix} {line_suffix}")
            return time_val

        # Total Model Time (Model Forward Passes + Value Head)
        total_model_time = timings.get('total_model_forward_time', 0.0)
        
        # --- 1. Trajectory Generation (RL Loop) ---
        print_timing_line("1. Trajectory Generation (Total - GPU/CPU)", 'generate_trajectories_total', 0, is_subtotal=True)
        
        # The new breakdown for the old 'Inference' part
        model_fwd_time = print_timing_line("  - Model Forward Passes (Total - GPU)", 'total_model_forward_time', 1)
        print_timing_line("    - Single-token Inference (Autoreg.)", 'single_token_inference_time', 2)
        print_timing_line("    - Full-seq Forward (Critic Prep)", 'full_forward_time', 2)
        print_timing_line("    - Value Head Calculation", 'value_head_time', 2)
        
        print_timing_line("  - Grammar Masking (CPU)", 'grammar_mask_time', 1)

        # --- 2. Environment & Reward ---
        env_reward_total = print_timing_line("2. Environment & Reward (Total - CPU)", 'env_and_reward_total', 0, is_subtotal=True)
        
        # 2.1. Environment Processing (Parsing + Model Init) - Average per sequence
        parse_time = timings.get('avg_env_total', 0.0)  # Use new timing key
        model_init_total = timings.get('avg_env_total', 0.0)  # Use new timing key
        env_processing_time_avg = parse_time + model_init_total

        print(f"  - Environment Processing (Avg per seq): {env_processing_time_avg:.4f}s ({(env_processing_time_avg/total_pipeline_time)*100:.1f}%)")
        print_timing_line("    - Parsing Sequence", 'avg_env_total', 2)
        
        model_init_and_val_total = print_timing_line("    - Model Init & Validation (Total)", 'avg_env_total', 2)
        print_timing_line("      - Check Propagate & Anomaly", 'avg_env_total', 3)

        # 2.2. Reward Shaping - Average per sequence
        reward_avg_total = timings.get('avg_reward_total', 0.0)  # Use new timing key
        print(f"  - Reward Shaping (Avg per seq): {reward_avg_total:.4f}s ({(reward_avg_total/total_pipeline_time)*100:.1f}%)")
        print_timing_line("    - Aggregate Check Data (Token Map)", 'avg_reward_total', 2)
        print_timing_line("    - Calculate Instantaneous Rewards", 'avg_reward_total', 2)
        # New breakdown for Discounted Return
        print_timing_line("    - Discounted Return Loop (R_target)", 'avg_reward_total', 2)
        print_timing_line("    - Misc Scalar/Mask Calculation", 'avg_reward_total', 2)
        
        print("-" * 90)

    # Print parsed model dict if available
    outcome_model_dict = outcome.get('model_data_dict') if isinstance(outcome, dict) else getattr(outcome, 'model_data_dict', None)
    if outcome_model_dict:
        try:
            print("\n--- Parsed Model Dict ---")
            print(json.dumps(outcome_model_dict, indent=2))
        except Exception:
            print("\n--- Parsed Model Dict (raw) ---")
            print(outcome_model_dict)
    
    effective_seq_len = min(len(trimmed_tokens), predicted_score_vectors.size(0))
    if effective_seq_len > 0:
        avg_value_loss = F.l1_loss(predicted_score_vectors[:effective_seq_len], per_token_target_scores[:effective_seq_len]).item()
        print(f"  Average Value Loss (L1): {avg_value_loss:.4f}")

    outcome_success = outcome.get('success', False) if isinstance(outcome, dict) else getattr(outcome, 'success', False)
    outcome_meta = outcome.get('meta', {}) if isinstance(outcome, dict) else getattr(outcome, 'meta', {})
    outcome_unclosed = outcome.get('unclosed_block_info') if isinstance(outcome, dict) else getattr(outcome, 'unclosed_block_info', None)
    if not outcome_success:
        print(f"  {outcome_meta.get('error') or outcome_meta.get('message', 'CHECK FAILURE')}")
    if outcome_unclosed:
        start, depth = outcome_unclosed
        print(f"  Structural Penalty applied for {depth} unclosed block(s) from token index {start}.")
        
    if print_detailed_checklist:
        print("\n--- Detailed Checklist & Reward Breakdown (Active Checks Only) ---")
        header = f"{'Check Name':<35} | {'Target Avg Score':>16} | {'Value Pred (Avg)':>16} | {'Value Loss (L1)':>16} | {'Overall Status'}"
        print(header); print("-" * len(header))

        unique_check_summary = defaultdict(lambda: {'instance_count': 0, 'passed_instances': 0, 'failed_instances': 0, 'messages': set(), 'error_vars': set(), 'good_vars': set(), 'check_idx': -1, 'obj_types_found': set()})
        obj_type_map = {'f': 'particle', 's': 'particle', 'm': 'field', 'i': 'itract', 'g': 'global'}
        
        diagnostics = rewards_info.get('diagnostics', {}) if isinstance(rewards_info, dict) else getattr(rewards_info, 'diagnostics', {})
        checklist_data = diagnostics.get("checks", {})
        for obj_id, checks in checklist_data.items():
            for check_name, result in checks.items():
                if check_name not in active_checks: continue  # CURRICULUM FILTER
                if check_name not in CHECK_TO_IDX: continue
                check_idx = CHECK_TO_IDX[check_name]
                summary = unique_check_summary[check_name]
                summary.update({'instance_count': summary['instance_count'] + 1, 'check_idx': check_idx})
                
                if obj_id == 'global' or not obj_id: summary['obj_types_found'].add('global')
                else: summary['obj_types_found'].add(obj_type_map.get(obj_id[0], 'unknown'))

                if result.get('score', 0) == result.get('max_score', 1): summary['passed_instances'] += 1
                else: summary['failed_instances'] += 1
                
                if result.get('message') and result['message'] != "Passed": summary['messages'].add(result['message'])
                summary['error_vars'].update(v for v in result.get('error_var', []) if v)
                summary['good_vars'].update(v for v in result.get('good_var', []) if v)
        
        sorted_check_names = sorted([name for name in unique_check_summary.keys() if name in active_checks], key=lambda k: (_get_check_category_priority(f"({'/'.join(sorted(list(unique_check_summary[k]['obj_types_found'])))} ) {k}"), k))

        for check_name in sorted_check_names:
            summary = unique_check_summary[check_name]
            check_idx = summary['check_idx']
            prefix = f"({'/'.join(sorted(list(summary['obj_types_found'])))})" if summary['obj_types_found'] else ''
            display_name = f"{prefix} {check_name}".strip()
            
            target_avg = float(averaged_check_type_scores[check_idx].item()) if check_idx >= 0 else float('nan')
            pred_avg = predicted_score_vectors[:effective_seq_len, check_idx].mean().item() if check_idx >= 0 and effective_seq_len > 0 else float('nan')
            l1 = F.l1_loss(predicted_score_vectors[:effective_seq_len, check_idx], per_token_target_scores[:effective_seq_len, check_idx]).item() if check_idx >= 0 and effective_seq_len > 0 else float('nan')

            status = "N/A"
            if summary['instance_count'] > 0:
                status = f"✅ All {summary['instance_count']} Passed" if summary['failed_instances'] == 0 else f"❌ {summary['failed_instances']}/{summary['instance_count']} Failed"
            
            details = []
            if summary['good_vars']: details.append(f"G:({', '.join(sorted(list(summary['good_vars'])))} )")
            if summary['error_vars']: details.append(f"B:({', '.join(sorted(list(summary['error_vars'])))} )")
            if summary['messages']: details.append(f"Msg:({'; '.join(sorted(list(summary['messages'])))} )")
            
            print(f"{display_name:<35} | {target_avg:>16.4f} | {pred_avg:>16.4f} | {l1:>16.4f} | {status} {' - '.join(details)}")

            if print_per_check_rewards_detail and check_idx != -1:
                print(f"  Token-by-token details for '{display_name}':")
                print(format_sequence_with_full_details(
                    tokens=trimmed_tokens,
                    immediate_rewards=per_token_instantaneous_rewards[:effective_seq_len, check_idx],
                    target_rewards=per_token_target_scores[:effective_seq_len, check_idx],
                    predicted_values=predicted_score_vectors[:effective_seq_len, check_idx],
                    criticality_scores=criticality_scores[:effective_seq_len],
                    exploration_head_scores=exploration_head_scores[:effective_seq_len],
                    check_idx=check_idx
                ))
                print("-" * len(header))
        print("-" * len(header) + "\n")