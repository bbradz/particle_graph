from dataclasses import dataclass, field
from typing import Dict, List, Any, Set, Optional, Tuple
import torch
import re
import copy
from config import Config
from rl.vocabulary import GRAMMAR_TOKEN_NAMES

# ANSI color codes for debugging
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

@dataclass
class GrammarState:
    last_token_str: str = "BOS"
    length: int = 0
    open_itracts: List[int] = field(default_factory=list)
    open_fields: List[int] = field(default_factory=list)
    open_particles: List[int] = field(default_factory=list)

    completed_blocks: Dict[str, List[int]] = field(default_factory=dict)
    
    replay_stack: List[Tuple[List[int], int]] = field(default_factory=list)
    
    current_itract_type: Optional[str] = None
    fields_closed_in_current_itract: int = 0
    current_block_sequence: List[int] = field(default_factory=list)
    block_sequence_stack: List[List[int]] = field(default_factory=list)
    
    current_field_type: Optional[str] = None
    current_field_dim: Optional[int] = None
    current_field_gen: Optional[int] = None
    current_su2l_rep: Optional[int] = None
    current_u1y_charge: Optional[int] = None
    particles_in_current_field: int = 0
    
    itract_id_next: int = 1
    field_id_next: int = 1
    particle_id_next: int = 1
    
    used_ids_in_open_blocks: Dict[str, Set[str]] = field(default_factory=lambda: {'i': set(), 'm': set(), 'f': set()})


class GrammarMasker:
    def __init__(self, cfg: Config, token_to_idx: Dict[str, int], idx_to_token: Dict[int, str], vocab_size: int):
        self.cfg = cfg
        self.token_to_idx = token_to_idx
        self.idx_to_token = idx_to_token
        self.vocab_size = vocab_size
        self.eos_id = self.token_to_idx['EOS']
        self.pad_id = self.token_to_idx['PAD']
        
        # --- DEBUGGING FLAG ---
        self.DEBUG_GRAMMAR = False # Set to True to see detailed grammar logs

        self.transitions = {
            'BOS': ['ITRACT'], 'ITRACT_ID_': ['TYPE_YUKAWA', 'TYPE_SCALARSELFINTERACTION'],
            'TYPE_YUKAWA': ['FIELD'], 'TYPE_SCALARSELFINTERACTION': ['LAMBDAVAR_1', 'LAMBDAVAR_2', 'LAMBDAVAR_3'],
            'FIELD_ID_': ['TYPE_FIELD_fermion', 'TYPE_FIELD_real', 'TYPE_FIELD_complex'],
            'TYPE_FIELD_fermion': ['DIM_1', 'DIM_2', 'DIM_3'], 'TYPE_FIELD_real': ['DIM_1', 'DIM_2', 'DIM_3'], 'TYPE_FIELD_complex': ['DIM_1', 'DIM_2', 'DIM_3'],
            'DIM_': ['GEN_1', 'GEN_2', 'GEN_3'], 'GEN_': ['SELF_CONJ_TRUE', 'SELF_CONJ_FALSE'],
            'SELF_CONJ_TRUE': ['CHIRALITY_left', 'CHIRALITY_right', 'CHIRALITY_none'], 'SELF_CONJ_FALSE': ['CHIRALITY_left', 'CHIRALITY_right', 'CHIRALITY_none'],
            'CHIRALITY_left': ['SU3C_REP_1', 'SU3C_REP_2', 'SU3C_REP_3'], 'CHIRALITY_right': ['SU3C_REP_1', 'SU3C_REP_2', 'SU3C_REP_3'], 'CHIRALITY_none': ['SU3C_REP_1', 'SU3C_REP_2', 'SU3C_REP_3'],
            'SU3C_REP_': ['SU2L_REP_1', 'SU2L_REP_2', 'SU2L_REP_3'], 'SU2L_REP_': ['U1Y_CHARGE_-1', 'U1Y_CHARGE_0', 'U1Y_CHARGE_1', 'U1Y_CHARGE_2', 'U1Y_CHARGE_3'],
            'U1Y_CHARGE_': ['QN_L_0', 'QN_L_1', 'QN_L_2'], 'QN_L_': ['QN_B_0', 'QN_B_1', 'QN_B_2'], 'QN_B_': ['PARTICLE', 'END_FIELD'],
            'PARTICLE_ID_': ['TYPE_PARTICLE_fermion', 'TYPE_PARTICLE_real', 'TYPE_PARTICLE_complex'],
            'TYPE_PARTICLE_fermion': ['MASS_1e-9', 'MASS_1e-4', 'MASS_1e-3', 'MASS_1e-2', 'MASS_1e-1', 'MASS_1e0', 'MASS_1e1', 'MASS_1e2'],
            'TYPE_PARTICLE_real': ['MASS_1e-9', 'MASS_1e-4', 'MASS_1e-3', 'MASS_1e-2', 'MASS_1e-1', 'MASS_1e0', 'MASS_1e1', 'MASS_1e2'],
            'TYPE_PARTICLE_complex': ['MASS_1e-9', 'MASS_1e-4', 'MASS_1e-3', 'MASS_1e-2', 'MASS_1e-1', 'MASS_1e0', 'MASS_1e1', 'MASS_1e2'],
            'MASS_': ['CHARGE_-1', 'CHARGE_0', 'CHARGE_1', 'CHARGE_2', 'CHARGE_3'], 'CHARGE_': ['END_PARTICLE'],
            'END_PARTICLE': ['PARTICLE', 'END_FIELD'], 'END_FIELD': ['FIELD', 'END_ITRACT'],
            'END_ITRACT': ['ITRACT', 'EOS'], 'EOS': ['PAD'], 'PAD': ['PAD']
        }
        for i in range(1, 4): self.transitions[f'LAMBDAVAR_{i}'] = ['FIELD']
        self.prefix_patterns = {
            'ITRACT_ID_': r'ITRACT_ID_\d+', 'FIELD_ID_': r'FIELD_ID_\d+', 'PARTICLE_ID_': r'PARTICLE_ID_\d+',
            'DIM_': r'DIM_\d+', 'GEN_': r'GEN_\d+', 'SU3C_REP_': r'SU3C_REP_\d+', 'SU2L_REP_': r'SU2L_REP_\d+',
            'U1Y_CHARGE_': r'U1Y_CHARGE_-?\d+', 'QN_L_': r'QN_L_\d+', 'QN_B_': r'QN_B_\d+',
            'MASS_': r'MASS_1e-?\d+', 'CHARGE_': r'CHARGE_-?\d+', 'LAMBDAVAR_': r'LAMBDAVAR_\d+'
        }

    def initial_state(self) -> GrammarState: return GrammarState()
    def _get_id_str(self, token_str: str) -> Optional[str]:
        if token_str.startswith('ITRACT_ID_'): return f"i{token_str.split('_')[-1]}"
        if token_str.startswith('FIELD_ID_'): return f"m{token_str.split('_')[-1]}"
        if token_str.startswith('PARTICLE_ID_'): return f"f{token_str.split('_')[-1]}"
        return None

    def step(self, state: GrammarState, token_id: int) -> GrammarState:
        token_str = self.idx_to_token.get(token_id, "PAD")
        new_state = copy.deepcopy(state)
        new_state.length += 1
        new_state.last_token_str = token_str

        if self.DEBUG_GRAMMAR: print(f"{bcolors.OKBLUE}Step {new_state.length}: Processing token '{token_str}'{bcolors.ENDC}")

        if new_state.replay_stack:
            current_replay_ids, current_cursor = new_state.replay_stack[-1]
            current_cursor += 1
            if current_cursor >= len(current_replay_ids):
                if self.DEBUG_GRAMMAR: print(f"{bcolors.WARNING}  - Replay task finished. Popping from stack.{bcolors.ENDC}")
                new_state.replay_stack.pop()
            else:
                new_state.replay_stack[-1] = (current_replay_ids, current_cursor)
        
        new_state.current_block_sequence.append(token_id)
        
        is_opener = token_str in ['ITRACT', 'FIELD', 'PARTICLE']
        is_id_token = '_ID_' in token_str and any(token_str.startswith(p) for p in ['ITRACT_', 'FIELD_', 'PARTICLE_'])
        is_closer = token_str in ['END_ITRACT', 'END_FIELD', 'END_PARTICLE']

        if is_opener:
            new_state.block_sequence_stack.append(new_state.current_block_sequence[:-1])
            new_state.current_block_sequence = [token_id]
            if token_str == 'ITRACT':
                new_state.open_itracts.append(new_state.length - 1)
                new_state.used_ids_in_open_blocks['m'] = set()
                new_state.fields_closed_in_current_itract = 0
                new_state.current_itract_type = None
            elif token_str == 'FIELD':
                new_state.open_fields.append(new_state.length - 1)
                new_state.used_ids_in_open_blocks['f'] = set()
                new_state.particles_in_current_field = 0
                new_state.current_field_type = None
            elif token_str == 'PARTICLE':
                new_state.open_particles.append(new_state.length - 1)
                new_state.particles_in_current_field += 1
        
        elif is_id_token:
            # --- THE FIX: Only trigger a new replay if not already replaying ---
            if not new_state.replay_stack:
                id_str = self._get_id_str(token_str)
                if id_str in new_state.completed_blocks:
                    full_block_seq = new_state.completed_blocks[id_str]
                    if len(full_block_seq) > 2:
                        replay_info = (full_block_seq[2:], 0)
                        if self.DEBUG_GRAMMAR: print(f"{bcolors.WARNING}  - Triggering replay for '{id_str}'. Pushing to stack.{bcolors.ENDC}")
                        new_state.replay_stack.append(replay_info)
            elif self.DEBUG_GRAMMAR: print(f"{bcolors.OKCYAN}  - In replay mode, ignoring nested replay trigger for '{token_str}'{bcolors.ENDC}")

            id_num_str = re.search(r'\d+', token_str).group()
            if token_str.startswith('ITRACT_ID_'): new_state.used_ids_in_open_blocks['i'].add(id_num_str); new_state.itract_id_next = max(new_state.itract_id_next, int(id_num_str) + 1)
            elif token_str.startswith('FIELD_ID_'): new_state.used_ids_in_open_blocks['m'].add(id_num_str); new_state.field_id_next = max(new_state.field_id_next, int(id_num_str) + 1)
            elif token_str.startswith('PARTICLE_ID_'): new_state.used_ids_in_open_blocks['f'].add(id_num_str); new_state.particle_id_next = max(new_state.particle_id_next, int(id_num_str) + 1)

        elif is_closer:
            if len(new_state.current_block_sequence) > 1:
                id_token_str = self.idx_to_token.get(new_state.current_block_sequence[1])
                block_id_str = self._get_id_str(id_token_str)
                if block_id_str and not new_state.replay_stack:
                    new_state.completed_blocks[block_id_str] = new_state.current_block_sequence.copy()
                    if self.DEBUG_GRAMMAR: print(f"{bcolors.OKGREEN}  - Completed and saved block '{block_id_str}'{bcolors.ENDC}")

            if new_state.block_sequence_stack:
                parent_sequence = new_state.block_sequence_stack.pop()
                parent_sequence.extend(new_state.current_block_sequence)
                new_state.current_block_sequence = parent_sequence
            
            if token_str == 'END_ITRACT' and new_state.open_itracts: new_state.open_itracts.pop()
            elif token_str == 'END_FIELD' and new_state.open_fields:
                new_state.open_fields.pop()
                if new_state.open_itracts: new_state.fields_closed_in_current_itract += 1
            elif token_str == 'END_PARTICLE' and new_state.open_particles: new_state.open_particles.pop()

        if token_str == 'TYPE_YUKAWA': new_state.current_itract_type = 'YUKAWA'
        elif token_str == 'TYPE_SCALARSELFINTERACTION': new_state.current_itract_type = 'SCALAR'
        if token_str.startswith('TYPE_FIELD_'): new_state.current_field_type = token_str.replace('TYPE_FIELD_', '').lower()
        if token_str.startswith('DIM_'): new_state.current_field_dim = int(re.search(r'\d+', token_str).group())
        if token_str.startswith('GEN_'): new_state.current_field_gen = int(re.search(r'\d+', token_str).group())
        if token_str.startswith('SU2L_REP_'): new_state.current_su2l_rep = int(re.search(r'\d+', token_str).group())
        if token_str.startswith('U1Y_CHARGE_'):
            match = re.search(r'U1Y_CHARGE_(-?\d+)', token_str)
            if match: new_state.current_u1y_charge = int(match.group(1))
        
        if self.DEBUG_GRAMMAR: print(f"{bcolors.OKCYAN}  - Replay Stack: {[(len(ids), cur) for ids, cur in new_state.replay_stack]}{bcolors.ENDC}")
        return new_state

    def _get_next_options_from_transitions(self, state: GrammarState) -> List[str]:
        last_tok = state.last_token_str
        if last_tok == 'ITRACT': return [f'ITRACT_ID_']
        if last_tok == 'FIELD': return [f'FIELD_ID_']
        if last_tok == 'PARTICLE': return [f'PARTICLE_ID_']
        if last_tok in self.transitions: return self.transitions[last_tok]
        for prefix, pattern in self.prefix_patterns.items():
            if re.fullmatch(pattern, last_tok):
                if prefix in self.transitions: return self.transitions[prefix]
        return []

    def get_valid_actions(self, state: GrammarState) -> torch.BoolTensor:
        if state.replay_stack:
            replay_ids, cursor = state.replay_stack[-1]
            if cursor < len(replay_ids):
                mask = torch.zeros(self.vocab_size, dtype=torch.bool)
                mask[replay_ids[cursor]] = True
                if self.DEBUG_GRAMMAR: print(f"{bcolors.WARNING}get_valid_actions: Replaying token '{self.idx_to_token[replay_ids[cursor]]}'{bcolors.ENDC}")
                return mask

        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        for option_str in self._get_next_options_from_transitions(state):
            if option_str.endswith('_'):
                for name, idx in self.token_to_idx.items():
                    if name.startswith(option_str): mask[idx] = True
            elif option_str in self.token_to_idx:
                mask[self.token_to_idx[option_str]] = True

        if state.last_token_str == 'ITRACT':
            mask.zero_()
            tok = f"ITRACT_ID_{state.itract_id_next}"
            if tok in self.token_to_idx: mask[self.token_to_idx[tok]] = True
        elif state.last_token_str == 'FIELD' and state.open_itracts:
            mask.zero_()
            tok_new = f"FIELD_ID_{state.field_id_next}"
            if tok_new in self.token_to_idx: mask[self.token_to_idx[tok_new]] = True
            for block_id in state.completed_blocks:
                if block_id.startswith('m') and block_id[1:] not in state.used_ids_in_open_blocks['m']:
                    tok_reuse = f"FIELD_ID_{block_id[1:]}"
                    if tok_reuse in self.token_to_idx: mask[self.token_to_idx[tok_reuse]] = True
        elif state.last_token_str == 'PARTICLE' and state.open_fields:
            mask.zero_()
            tok_new = f"PARTICLE_ID_{state.particle_id_next}"
            if tok_new in self.token_to_idx: mask[self.token_to_idx[tok_new]] = True
            for block_id in state.completed_blocks:
                if block_id.startswith('f') and block_id[1:] not in state.used_ids_in_open_blocks['f']:
                    tok_reuse = f"PARTICLE_ID_{block_id[1:]}"
                    if tok_reuse in self.token_to_idx: mask[self.token_to_idx[tok_reuse]] = True

        if state.last_token_str.startswith('DIM_') and state.current_field_type in ['real', 'complex']:
             mask &= (torch.arange(self.vocab_size) == self.token_to_idx['GEN_1'])
        if state.last_token_str.startswith('SU3C_REP_') and state.current_field_dim is not None:
            dim_map = {1: 'SU2L_REP_1', 2: 'SU2L_REP_2', 3: 'SU2L_REP_3'}
            if state.current_field_dim in dim_map:
                 mask &= (torch.arange(self.vocab_size) == self.token_to_idx[dim_map[state.current_field_dim]])
        if state.last_token_str.startswith('PARTICLE_ID_') and state.current_field_type:
            type_map = {'fermion': 'TYPE_PARTICLE_fermion', 'real': 'TYPE_PARTICLE_real', 'complex': 'TYPE_PARTICLE_complex'}
            if state.current_field_type in type_map:
                mask.zero_()
                mask[self.token_to_idx[type_map[state.current_field_type]]] = True
        if state.last_token_str.startswith('MASS_') and state.current_su2l_rep is not None and state.current_u1y_charge is not None:
            T3_map = {1: [0.0], 2: [0.5, -0.5], 3: [1.0, 0.0, -1.0]}
            allowed_charges = {int(3 * (T3 + state.current_u1y_charge)) for T3 in T3_map.get(state.current_su2l_rep, [])}
            charge_mask = torch.zeros_like(mask)
            for charge in allowed_charges:
                tok = f"CHARGE_{charge}"
                if tok in self.token_to_idx: charge_mask[self.token_to_idx[tok]] = True
            if charge_mask.any(): mask &= charge_mask

        if state.open_itracts and state.current_itract_type == 'SCALAR':
            if state.last_token_str.startswith('FIELD_ID_'):
                mask[self.token_to_idx['TYPE_FIELD_fermion']] = False
            if state.fields_closed_in_current_itract >= 1:
                mask[self.token_to_idx['FIELD']] = False
        if state.open_itracts and state.current_itract_type == 'YUKAWA':
             if state.fields_closed_in_current_itract >= 3:
                mask[self.token_to_idx['FIELD']] = False

        if state.open_fields and (state.last_token_str.startswith('QN_B_') or state.last_token_str == 'END_PARTICLE'):
            required = (state.current_field_dim or 1) * (state.current_field_gen or 1)
            mask.zero_()
            if state.particles_in_current_field < required:
                mask[self.token_to_idx['PARTICLE']] = True
            else:
                mask[self.token_to_idx['END_FIELD']] = True
        
        is_any_block_open = bool(state.open_itracts or state.open_fields or state.open_particles)
        if is_any_block_open:
            mask[self.token_to_idx['EOS']] = False
        else:
            temp_mask = torch.zeros_like(mask)
            temp_mask[self.token_to_idx['ITRACT']] = True
            temp_mask[self.token_to_idx['EOS']] = True
            mask &= temp_mask

        if state.length >= self.cfg.MAX_LEN - 5:
            for opener in ['ITRACT', 'FIELD', 'PARTICLE']:
                if opener in self.token_to_idx:
                    mask[self.token_to_idx[opener]] = False
        if not mask.any(): mask[self.pad_id] = True
            
        if self.DEBUG_GRAMMAR:
            valid_tokens = [self.idx_to_token[i] for i, v in enumerate(mask) if v]
            print(f"{bcolors.HEADER}get_valid_actions: After '{state.last_token_str}', valid options: {valid_tokens}{bcolors.ENDC}")

        return mask