import torch
from typing import Tuple, Dict, List

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