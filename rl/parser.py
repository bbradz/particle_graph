import re
from typing import List, Dict, Any, Tuple, Optional

class ParsingError(Exception):
    """Custom exception for parsing errors."""
    pass

class SequenceParser:
    """
    Parses a token sequence into a valid model dictionary (containing only
    fully closed objects) and creates a detailed token map for credit assignment.
    This is a top-down, recursive-descent style parser.
    """
    # Mappings from tokens to values
    MASS_MAP = {
        "MASS_1e-9": 1e-9, "MASS_1e-4": 1e-4, "MASS_1e-3": 1e-3, "MASS_1e-2": 1e-2,
        "MASS_1e-1": 1e-1, "MASS_1e0": 1, "MASS_1e1": 10, "MASS_1e2": 100,
    }
    CHARGE_MAP = { "CHARGE_-1": -1, "CHARGE_0": 0, "CHARGE_1": 1, "CHARGE_2": 2, "CHARGE_3": 3 }
    LAMBDAVAR_MAP = {
        "LAMBDAVAR_1": 0.1,
        "LAMBDAVAR_2": 0.5,
        "LAMBDAVAR_3": 1.0,
    }
    SU3C_REP_MAP = {"SU3C_REP_1": "singlet", "SU3C_REP_2": "fnd", "SU3C_REP_3": "adj"}
    SU2L_REP_MAP = {"SU2L_REP_1": "singlet", "SU2L_REP_2": "fnd", "SU2L_REP_3": "adj"}
    U1Y_CHARGE_MAP = {
        "U1Y_CHARGE_-1": -1, "U1Y_CHARGE_0": 0, "U1Y_CHARGE_1": 1, "U1Y_CHARGE_2": 2, "U1Y_CHARGE_3": 3,
    }
    # Map specific type tokens to their generic string values
    TYPE_MAP = {
        "TYPE_YUKAWA": "yukawa",
        "TYPE_SCALARSELFINTERACTION": "ScalarSelfInteraction",
        "TYPE_FIELD_fermion": "fermion",
        "TYPE_FIELD_real": "real",
        "TYPE_FIELD_complex": "complex",
        "TYPE_PARTICLE_fermion": "fermion",
        "TYPE_PARTICLE_real": "real",
        "TYPE_PARTICLE_complex": "complex"
    }
    CHIRALITY_MAP = {"CHIRALITY_left": "left", "CHIRALITY_right": "right", "CHIRALITY_none": None}

    def __init__(self):
        self.tokens: List[str] = []
        self.cursor: int = 0
        # token_map key format: (object_type_str, object_id_str, attribute_path_str) -> List[int]
        self.token_map: Dict[Tuple[str, str, str], List[int]] = {}
        # Keep track of generated IDs for current sequence to avoid duplicates
        self._generated_ids: Dict[str, Dict[str, bool]] = {'i': {}, 'm': {}, 'f': {}} # {prefix: {id_str: used}}
        # Diagnostics about parsing attempts and failures
        self.parsing_diagnostics: Dict[str, List[str]] = {}

    def _get_value_from_token(self, token_str: str, value_map: Dict[str, Any] = None, regex_pattern: str = None) -> Any:
        if value_map:
            return value_map.get(token_str)
        if regex_pattern:
            match = re.search(regex_pattern, token_str)
            if match:
                return match.group() # Return the full matched string
        return None

    def _consume(self, expected_token_prefix: str, store_map_key: Tuple = None, value_map: Dict = None, regex_pattern: str = None) -> Any:
        if self.cursor >= len(self.tokens):
            raise ParsingError(f"Expected token starting with '{expected_token_prefix}' but reached end of sequence prematurely at index {self.cursor}.")

        token = self.tokens[self.cursor]
        if not token.startswith(expected_token_prefix):
            raise ParsingError(f"Expected token starting with '{expected_token_prefix}' but got unexpected token '{token}' at index {self.cursor}.")

        value = self._get_value_from_token(token, value_map, regex_pattern)

        if store_map_key:
            self.token_map.setdefault(store_map_key, []).append(self.cursor)

        self.cursor += 1
        return value

    def parse(self, tokens: List[str]) -> Tuple[Dict[str, Any], Dict[Tuple, List[int]], Optional[Tuple[int, int]], Dict[str, List[str]]]:
        self.tokens = tokens
        self.cursor = 0
        self.token_map = {}
        self._generated_ids = {'i': {}, 'm': {}, 'f': {}} # Reset IDs for each parse call
        self.parsing_diagnostics = {'failed_fields': [], 'failed_particles': []}

        model_dict = {
            "GaugeGroups": {
                "g1": {"name": "hypercharge", "charge": "Y", "group": "U_1", "boson": "B"},
                "g2": {"name": "left", "charge": "I", "group": "SU_2", "boson": "WB"},
                "g3": {"name": "color", "charge": "C", "group": "SU_3", "boson": "G"}
            },
            "particles": {}, "fields": {}, "interactions": {}
        }

        if not self.tokens or self.tokens[self.cursor] != 'BOS':
            # Critical parsing failure at the very start.
            # We must still find unclosed blocks for penalty, scanning the given tokens.
            unclosed_info = self._find_first_unclosed_block(self.tokens)
            raise ParsingError("Sequence must start with BOS token.")

        self.cursor += 1 # Consume BOS

        # Parse interactions (top-level blocks)
        while self.cursor < len(self.tokens) and self.tokens[self.cursor] == "ITRACT":
            start_itract_idx = self.cursor
            try:
                # Consume ITRACT opener token
                self._consume("ITRACT")

                # Consume ITRACT_ID_X token
                itract_id_token = self._consume("ITRACT_ID_", regex_pattern=r'\d+')
                itract_num = self._get_value_from_token(itract_id_token, regex_pattern=r'\d+')
                itract_id = f"i{itract_num}"

                if itract_id in self._generated_ids['i']:
                    raise ParsingError(f"Duplicate interaction ID: {itract_id}")
                self._generated_ids['i'][itract_id] = True

                interaction_dict, field_list, particle_list = self._parse_interaction_block(itract_id)

                # If parsing successful, add to model_dict and update token_map
                model_dict["interactions"][interaction_dict['id']] = interaction_dict
                for p_dict in particle_list:
                    model_dict["particles"][p_dict['id']] = p_dict
                for f_dict in field_list:
                    model_dict["fields"][f_dict['id']] = f_dict

                # Map the entire interaction span after successful parse
                for token_idx in range(start_itract_idx, self.cursor):
                    self.token_map.setdefault(('interaction', interaction_dict['id'], 'block_span'), []).append(token_idx)

            except ParsingError:
                # If an interaction block is incomplete or invalid, stop parsing it and subsequent interactions.
                # Move cursor past current ITRACT token to avoid infinite loop
                while self.cursor < len(self.tokens) and not self.tokens[self.cursor].startswith("END_ITRACT"):
                    self.cursor += 1
                if self.cursor < len(self.tokens) and self.tokens[self.cursor].startswith("END_ITRACT"):
                    self.cursor += 1 # Consume the END_ITRACT if found
                break # Stop processing further top-level interactions

        # After attempting to parse all interactions, determine if there are unclosed blocks
        unclosed_info = self._find_first_unclosed_block(self.tokens)

        return model_dict, self.token_map, unclosed_info, self.parsing_diagnostics

    def _parse_interaction_block(self, itract_id: str) -> Tuple[Dict, List, List]:
        """Parses a single ITRACT block."""

        interaction_type = self._consume("TYPE_", ('interaction', itract_id, 'type'), value_map=self.TYPE_MAP)

        interaction_dict = {"id": itract_id, "type": interaction_type, "fields": []}
        
        if interaction_type == "ScalarSelfInteraction":
            lambda_token = self._consume("LAMBDAVAR_", ('interaction', itract_id, 'LambdaVar'))
            lambda_val = self.LAMBDAVAR_MAP.get(lambda_token, 0.5)
            interaction_dict['LambdaVar'] = lambda_val

        fields_in_interaction = []
        particles_in_interaction = []

        # Parse nested fields
        while self.cursor < len(self.tokens) and self.tokens[self.cursor] == "FIELD":
            start_field_idx = self.cursor
            try:
                # Consume FIELD opener token
                self._consume("FIELD")

                # Consume FIELD_ID_X token
                field_id_token = self._consume("FIELD_ID_", regex_pattern=r'\d+')
                field_num = self._get_value_from_token(field_id_token, regex_pattern=r'\d+')
                field_id = f"m{field_num}"

                if field_id in self._generated_ids['m']:
                    raise ParsingError(f"Duplicate field ID: {field_id}")
                self._generated_ids['m'][field_id] = True

                field_dict, particles_in_field = self._parse_field_block(field_id)
                fields_in_interaction.append(field_dict)
                particles_in_interaction.extend(particles_in_field)
                interaction_dict["fields"].append(field_id) # Store field ID, not full dict

                # Map the entire field span after successful parse
                for token_idx in range(start_field_idx, self.cursor):
                    self.token_map.setdefault(('field', field_id, 'block_span'), []).append(token_idx)

            except ParsingError:
                # If a field block is incomplete or invalid, stop parsing fields within this interaction
                # Move cursor past current FIELD opener to avoid infinite loop on a bad field.
                # Record failed field id if we could parse it
                try:
                    # Attempt to extract FIELD_ID token if present next
                    # Note: We only append if field_id is already defined
                    if 'field_id' in locals():
                        self.parsing_diagnostics['failed_fields'].append(field_id)
                except Exception:
                    pass
                while self.cursor < len(self.tokens) and not self.tokens[self.cursor].startswith("END_FIELD"):
                    self.cursor += 1
                if self.cursor < len(self.tokens) and self.tokens[self.cursor].startswith("END_FIELD"):
                    self.cursor += 1 # Consume the END_FIELD if found
                continue # Use continue to proceed to the next token, potentially finding the next valid FIELD

        # Consume END_ITRACT
        self._consume("END_ITRACT", ('interaction', itract_id, 'END_ITRACT'))

        return interaction_dict, fields_in_interaction, particles_in_interaction

    def _parse_field_block(self, field_id: str) -> Tuple[Dict, List]:
        """Parses a single FIELD block."""

        # Expect TYPE_FIELD_X tokens
        field_type = self._consume("TYPE_FIELD_", ('field', field_id, 'type'), value_map=self.TYPE_MAP)

        dim = int(self._consume("DIM_", ('field', field_id, 'dim'), regex_pattern=r'\d+'))
        gen = int(self._consume("GEN_", ('field', field_id, 'gen'), regex_pattern=r'\d+'))
        self_conjugate_str = self._consume("SELF_CONJ_", ('field', field_id, 'self_conjugate'))
        self_conjugate = (self_conjugate_str == "SELF_CONJ_TRUE")
        chirality = self._consume("CHIRALITY_", ('field', field_id, 'chirality'), value_map=self.CHIRALITY_MAP)

        reps = {
            "g3": self._consume("SU3C_REP_", ('field', field_id, 'reps.g3'), value_map=self.SU3C_REP_MAP),
            "g2": self._consume("SU2L_REP_", ('field', field_id, 'reps.g2'), value_map=self.SU2L_REP_MAP),
            "g1": self._consume("U1Y_CHARGE_", ('field', field_id, 'reps.g1'), value_map=self.U1Y_CHARGE_MAP)
        }

        # Quantum Numbers (QN_L_, QN_B_) - consume but do not store for now as not in base JSON structure
        self._consume("QN_L_", ('field', field_id, 'QN_L'), regex_pattern=r'\d+')
        self._consume("QN_B_", ('field', field_id, 'QN_B'), regex_pattern=r'\d+')

        field_dict = {
            "id": field_id, "name": field_id, "type": field_type,
            "dim": dim, "gen": gen, "self_conjugate": self_conjugate,
            "chirality": chirality, "reps": reps, "particles": []
        }
        particles_in_field = []

        # Parse nested particles
        while self.cursor < len(self.tokens) and self.tokens[self.cursor] == "PARTICLE":
            start_particle_idx = self.cursor
            try:
                # Consume PARTICLE opener token
                self._consume("PARTICLE")

                # Consume PARTICLE_ID_X token
                particle_id_token = self._consume("PARTICLE_ID_", regex_pattern=r'\d+')
                particle_num = self._get_value_from_token(particle_id_token, regex_pattern=r'\d+')
                particle_id = f"f{particle_num}"

                if particle_id in self._generated_ids['f']:
                    raise ParsingError(f"Duplicate particle ID: {particle_id}")
                self._generated_ids['f'][particle_id] = True

                particle_dict = self._parse_particle_block(particle_id, parent_field_type=field_type)

                particles_in_field.append(particle_dict)
                field_dict["particles"].append(particle_id) # Store particle ID, not full dict

                # Map the entire particle span after successful parse
                for token_idx in range(start_particle_idx, self.cursor):
                    self.token_map.setdefault(('particle', particle_id, 'block_span'), []).append(token_idx)

            except ParsingError:
                # If a particle block is incomplete or invalid, stop parsing particles within this field
                # Move cursor past current PARTICLE opener to avoid infinite loop on a bad particle.
                # Record failed particle id if available
                try:
                    if 'particle_id' in locals():
                        self.parsing_diagnostics['failed_particles'].append(particle_id)
                except Exception:
                    pass
                while self.cursor < len(self.tokens) and not self.tokens[self.cursor].startswith("END_PARTICLE"):
                    self.cursor += 1
                if self.cursor < len(self.tokens) and self.tokens[self.cursor].startswith("END_PARTICLE"):
                    self.cursor += 1 # Consume the END_PARTICLE if found
                break # Stop processing further particles in this field

        # Consume END_FIELD
        self._consume("END_FIELD", ('field', field_id, 'END_FIELD'))

        return field_dict, particles_in_field

    def _parse_particle_block(self, particle_id: str, parent_field_type: str) -> Dict:
        """Parses a single PARTICLE block."""

        # Expect TYPE_PARTICLE_X tokens
        particle_type = self._consume("TYPE_PARTICLE_", ('particle', particle_id, 'type'), value_map=self.TYPE_MAP)

        # Contextual validation for particle type here - moved from grammar
        if particle_type != parent_field_type:
            raise ParsingError(f"Particle type '{particle_type}' for {particle_id} "
                               f"does not match parent field type '{parent_field_type}'.")

        mass = self._consume("MASS_", ('particle', particle_id, 'mass'), value_map=self.MASS_MAP)
        charge = self._consume("CHARGE_", ('particle', particle_id, 'charge'), value_map=self.CHARGE_MAP)

        # Consume END_PARTICLE
        self._consume("END_PARTICLE", ('particle', particle_id, 'END_PARTICLE'))

        particle_dict = {
            "id": particle_id, "name": particle_id, "type": particle_type,
            "mass": mass, "charge": charge
        }
        return particle_dict

    def _find_first_unclosed_block(self, tokens_to_scan: List[str]) -> Optional[Tuple[int, int]]:
        """
        Identifies the starting index and depth of the first unclosed block
        within a given list of tokens.
        """
        stack = []  # Stack holds tuples of (block_type, start_index)
        openers = {"ITRACT": "ITRACT", "FIELD": "FIELD", "PARTICLE": "PARTICLE"}
        closers = {"END_ITRACT": "ITRACT", "END_FIELD": "FIELD", "END_PARTICLE": "PARTICLE"}

        for i, token in enumerate(tokens_to_scan):
            if token in openers:
                stack.append((token, i))
            elif token in closers:
                if stack and stack[-1][0] == closers[token]:
                    stack.pop()
            # If EOS or PAD is encountered, stop scanning, as these terminate structure.
            if token == 'EOS' or token == 'PAD':
                break

        if stack:
            # The first item left on the stack is the outermost unclosed block
            return stack[0][1], len(stack)  # Return start_index and depth
        return None