from __future__ import annotations
import os, sys, json, tempfile, re
from typing import List, Dict, Any, Tuple

# Assuming AdvancedModel is in a sibling directory as per the context
from Token2Model.model import Model as AdvancedModel

# Import DEBUG from config
from config import get_config
DEBUG = get_config().DEBUG_PRINTS

class ModelTester:
    """
    Deserializes token streams into a JSON-like structure, augmenting each
    value with the index of the token(s) it originated from.
    This version includes robust error handling to prevent crashes.
    """
    # Mappings from tokens to values (updated to match grammar)
    MASS_MAP = {
        "MASS_1e-9": 1e-9, "MASS_1e-4": 1e-4, "MASS_1e-3": 1e-3, "MASS_1e-2": 1e-2,
        "MASS_1e-1": 1e-1, "MASS_1e0": 1, "MASS_1e1": 10, "MASS_1e2": 100,
    }
    CHARGE_MAP = { "CHARGE_-1": -1, "CHARGE_0": 0, "CHARGE_1": 1, "CHARGE_2": 2, "CHARGE_3": 3 }
    SU3C_REP_MAP = {"SU3C_REP_1": "singlet", "SU3C_REP_2": "fnd", "SU3C_REP_3": "adj"}
    SU2L_REP_MAP = {"SU2L_REP_1": "singlet", "SU2L_REP_2": "fnd", "SU2L_REP_3": "adj"}
    U1Y_CHARGE_MAP = {
        "U1Y_CHARGE_-1": -1, "U1Y_CHARGE_0": 0, "U1Y_CHARGE_1": 1, "U1Y_CHARGE_2": 2, "U1Y_CHARGE_3": 3,
    }
    TYPE_MAP = {
        "TYPE_YUKAWA": "yukawa", "TYPE_SCALARSELFINTERACTION": "ScalarSelfInteraction",
        "TYPE_fermion": "fermion", "TYPE_real": "real", "TYPE_complex": "complex"
    }
    CHIRALITY_MAP = {"CHIRALITY_left": "left", "CHIRALITY_right": "right", "CHIRALITY_none": None}

    @staticmethod
    def _after(tok: str, prefix: str) -> str:
        return tok.replace(f"{prefix}_", "")

    @staticmethod
    def _int(tok: str) -> int:
        numbers = re.findall(r'\d+', tok)
        return int(numbers[-1]) if numbers else 0

    def _v(self, value: Any, token_index: int) -> Dict[str, Any]:
        return {"value": value, "token_indices": [token_index]}

    def parse_particle(self, seq: List[str], i: int) -> Tuple[Dict[str, Any], int]:
        start_index = i
        if DEBUG: print(f"  [DEBUG] Parsing PARTICLE at index {i}")
        try:
            # Look for the end of the particle block first
            try:
                end_index = seq.index("END_PARTICLE", start_index)
            except ValueError:
                raise IndexError(f"Unclosed PARTICLE block starting at index {start_index}")

            # Validate header length
            if start_index + 4 >= end_index:
                raise IndexError(f"Incomplete PARTICLE header starting at index {start_index}")

            particle_id_val = f"f{self._int(seq[i+1])}"
            name_val = f"p{self._int(seq[i+1])}"
            type_val = self.TYPE_MAP.get(seq[i+2], "unknown")
            mass_val = self.MASS_MAP.get(seq[i+3], 0.0)
            charge_val = self.CHARGE_MAP.get(seq[i+4], 0)

            particle = {
                "id": self._v(particle_id_val, i+1), "name": self._v(name_val, i+1),
                "type": self._v(type_val, i+2), "mass": self._v(mass_val, i+3),
                "charge": self._v(charge_val, i+4),
            }
            return particle, end_index + 1
        except IndexError as e:
            if DEBUG: print(f"  [DEBUG] PARSE ERROR in parse_particle at index {start_index}: {e}")
            raise

    def parse_field(self, seq: List[str], i: int) -> Tuple[Dict[str, Any], int, List[Dict[str, Any]]]:
        start_index = i
        if DEBUG: print(f"  [DEBUG] Parsing FIELD at index {i}")
        try:
            # Find the corresponding END_FIELD, respecting nested PARTICLE blocks
            nesting_level = 0
            end_index = -1
            for j in range(start_index + 1, len(seq)):
                if seq[j] == "FIELD": nesting_level += 1
                if seq[j] == "END_FIELD":
                    if nesting_level == 0:
                        end_index = j
                        break
                    nesting_level -= 1
            if end_index == -1:
                 raise IndexError(f"Unclosed FIELD block starting at index {start_index}")

            if i + 11 >= end_index: raise IndexError(f"Incomplete FIELD header starting at index {start_index}")

            field = {
                "id": self._v(f"m{self._int(seq[i+1])}", i+1),
                "name": self._v(f"f{self._int(seq[i+1])}", i+1),
                "type": self._v(self.TYPE_MAP.get(seq[i+2], "unknown"), i+2),
                "dim": self._v(self._int(seq[i+3]), i+3),
                "gen": self._v(self._int(seq[i+4]), i+4),
                "self_conjugate": self._v(seq[i+5] == "SELF_CONJ_TRUE", i+5),
                "chirality": self._v(self.CHIRALITY_MAP.get(seq[i+6], None), i+6),
                "reps": {
                    "value": {
                        "g1": self._v(self.U1Y_CHARGE_MAP.get(seq[i+9], 0), i+9),
                        "g2": self._v(self.SU2L_REP_MAP.get(seq[i+8], "singlet"), i+8),
                        "g3": self._v(self.SU3C_REP_MAP.get(seq[i+7], "singlet"), i+7)
                    }, "token_indices": [i+7, i+8, i+9]
                },
                "particles": self._v([], i),
            }
            
            j = i + 12
            particles_in_field, particle_ids_val, particle_tokens = [], [], []

            while j < end_index:
                if seq[j] == "PARTICLE":
                    p, j_after_p = self.parse_particle(seq, j)
                    particles_in_field.append(p)
                    if p and "id" in p and "value" in p["id"]:
                        particle_ids_val.append(p["id"]["value"])
                    particle_tokens.extend(range(j, j_after_p))
                    j = j_after_p
                else: j += 1

            field["particles"] = {"value": particle_ids_val, "token_indices": particle_tokens}
            return field, end_index + 1, particles_in_field
        except IndexError as e:
            if DEBUG: print(f"  [DEBUG] PARSE ERROR in parse_field at index {start_index}: {e}")
            raise

    def parse_interaction(self, seq: List[str], i: int) -> Tuple[Dict[str, Any], int, List[Dict[str, Any]], List[Dict[str, Any]]]:
        start_index = i
        try:
            # Find the corresponding END_ITRACT, respecting nested FIELD blocks
            nesting_level = 0
            end_index = -1
            for j in range(start_index + 1, len(seq)):
                if seq[j] == "ITRACT": nesting_level += 1
                if seq[j] == "END_ITRACT":
                    if nesting_level == 0:
                        end_index = j
                        break
                    nesting_level -= 1
            if end_index == -1:
                raise IndexError(f"Unclosed ITRACT block starting at index {start_index}")

            if i + 2 >= end_index: raise IndexError(f"Incomplete ITRACT header starting at index {start_index}")

            inter_type_val = self.TYPE_MAP.get(seq[i+2], "unknown")
            inter = {
                "id": self._v(f"i{self._int(seq[i+1])}", i+1),
                "type": self._v(inter_type_val, i+2), "fields": self._v([], i),
            }
            if inter_type_val == "ScalarSelfInteraction":
                inter["LambdaVar"] = self._v([0.1, 1.0], i) 
            
            j = i + 3
            fields_in_inter, particles_in_inter, field_ids_val, field_tokens = [], [], [], []

            while j < end_index:
                if seq[j] == "FIELD":
                    f, j_after_f, ps = self.parse_field(seq, j)
                    fields_in_inter.append(f)
                    particles_in_inter.extend(ps)
                    if f and "id" in f and "value" in f["id"]:
                        field_ids_val.append(f["id"]["value"])
                    field_tokens.extend(range(j, j_after_f))
                    j = j_after_f
                else: j += 1
                    
            inter["fields"] = {"value": field_ids_val, "token_indices": field_tokens}
            return inter, end_index + 1, fields_in_inter, particles_in_inter
        except IndexError as e:
            if DEBUG: print(f"  [DEBUG] PARSE ERROR in parse_interaction at index {start_index}: {e}")
            raise

    def build_model(self, tokens: List[str]) -> Dict[str, Any]:
        particles, fields, inters = [], [], []
        i = 0
        try:
            # The top-level structure must be a series of ITRACT blocks
            while i < len(tokens):
                token = tokens[i]
                if token == "BOS":
                    i += 1
                    continue
                if token == "EOS":
                    break # End of sequence
                if token == "ITRACT":
                    if DEBUG: print(f"[DEBUG] Parsing ITRACT at index {i}")
                    it, i_after, new_fs, new_ps = self.parse_interaction(tokens, i)
                    inters.append(it)
                    fields.extend(new_fs)
                    particles.extend(new_ps)
                    i = i_after
                else:
                    # If we find a token that is not a top-level block opener, it's an error
                    raise ValueError(f"Unexpected token '{token}' at top level of model definition at index {i}")
        except (IndexError, ValueError) as e:
            if DEBUG: print(f"[DEBUG] Parsing failed: {e}")
            # Re-raise as a generic error that the environment can catch
            raise ValueError(f"Parsing failed due to structural error: {e}")

        # Robust de-duplication
        unique_particles = {p["id"]["value"]: p for p in particles if p and "id" in p and "value" in p["id"]}
        unique_fields = {f["id"]["value"]: f for f in fields if f and "id" in f and "value" in f["id"]}
        unique_inters = {it["id"]["value"]: it for it in inters if it and "id" in it and "value" in it["id"]}

        return {
            "GaugeGroups": [
                {"id": {"value": "g1", "token_indices": []}, "name": {"value": "hypercharge", "token_indices": []}, "charge": {"value": "Y", "token_indices": []}, "group": {"value": "U_1", "token_indices": []}, "boson": {"value": "B", "token_indices": []}},
                {"id": {"value": "g2", "token_indices": []}, "name": {"value": "left", "token_indices": []}, "charge": {"value": "I", "token_indices": []}, "group": {"value": "SU_2", "token_indices": []}, "boson": {"value": "WB", "token_indices": []}},
                {"id": {"value": "g3", "token_indices": []}, "name": {"value": "color", "token_indices": []}, "charge": {"value": "C", "token_indices": []}, "group": {"value": "SU_3", "token_indices": []}, "boson": {"value": "G", "token_indices": []}},
            ],
            "particles": list(unique_particles.values()), "fields": list(unique_fields.values()), "interactions": list(unique_inters.values())
        }

# (The blame attribution functions remain unchanged)
def collect_all_indices(obj: Any) -> set[int]:
    indices = set()
    if isinstance(obj, dict):
        if "token_indices" in obj and isinstance(obj["token_indices"], list): indices.update(obj["token_indices"])
        for value in obj.values(): indices.update(collect_all_indices(value))
    elif isinstance(obj, list):
        for item in obj: indices.update(collect_all_indices(item))
    return indices
def resolve_path(augmented_dict: Dict, path_str: str) -> List[int]:
    keys = path_str.split('.')
    if DEBUG: print(f"  [DEBUG_BLAME] Resolving path: '{path_str}'")
    try:
        current_level = augmented_dict
        all_indices_for_path = set()
        
        for k_idx, key in enumerate(keys):
            if k_idx == 0: # Top level (e.g., 'fields')
                if key in current_level:
                    current_level = current_level[key]
                else:
                    if DEBUG: print(f"  [DEBUG_BLAME]   - Path part '{key}' not found at top level.")
                    return []
            elif k_idx == 1: # Object ID (e.g., 'm6')
                obj_id = re.sub(r'(_[RL])$', '', key) # Remove _L/_R suffix for lookup
                if obj_id in current_level:
                    # Add all indices for the entire object (field/particle/interaction)
                    all_indices_for_path.update(collect_all_indices(current_level[obj_id]))
                    current_level = current_level[obj_id]
                else:
                    if DEBUG: print(f"  [DEBUG_BLAME]   - Object ID '{obj_id}' not found in '{keys[0]}'.")
                    return []
            else: # Nested properties (e.g., 'reps.g1')
                if isinstance(current_level, dict) and "value" in current_level:
                    current_level = current_level["value"] # Dive into the 'value' part of augmented dict
                
                if key in current_level:
                    if isinstance(current_level[key], dict) and "token_indices" in current_level[key]:
                        all_indices_for_path.update(current_level[key]["token_indices"])
                    else: # If it's a simple value, its parent (the object itself) already added its tokens
                        pass 
                    current_level = current_level[key]
                else:
                    if DEBUG: print(f"  [DEBUG_BLAME]   - Path part '{key}' not found in current level.")
                    return []

        if DEBUG: print(f"  [DEBUG_BLAME]   - Resolved to tokens: {sorted(list(all_indices_for_path))}")
        return sorted(list(all_indices_for_path))
    except (KeyError, TypeError, IndexError) as e: 
        if DEBUG: print(f"  [DEBUG_BLAME]   - Error resolving path '{path_str}': {e}")
        return []
def map_failures_to_tokens(checklist: Dict, augmented_model: Dict) -> Dict[Tuple[str, str], List[int]]:
    blame_map = {}; root_causes = {obj_id: set() for obj_id in checklist}
    if 'global' not in root_causes: root_causes['global'] = set()
    fermion_field_ids = {fid for fid, f in augmented_model.get("fields", {}).items() if f.get("type", {}).get("value") == "fermion"}
    
    if DEBUG: print("\n[DEBUG_BLAME] Starting map_failures_to_tokens.")

    # Iteratively propagate blame
    for iteration in range(5): # Limit iterations to prevent infinite loops on complex dependencies
        blame_added_this_pass = False
        if DEBUG: print(f"[DEBUG_BLAME] Iteration {iteration + 1}")
        
        for obj_id, checks in checklist.items():
            if DEBUG: print(f"[DEBUG_BLAME] Processing object '{obj_id}' for blame.")
            for check_name, results in checks.items():
                if DEBUG: print(f"[DEBUG_BLAME]   - Check '{check_name}': Score={results.get('score')}/{results.get('max_score')}, Message='{results.get('message')}'")
                
                relevant_tokens_for_check = set() # Collect tokens for this specific check

                if results['score'] == results['max_score']: 
                    # PART 1 FIX: For passing checks, attribute credit to 'good_var' tokens
                    if DEBUG: print(f"[DEBUG_BLAME]     -> Check PASSED. Attributing credit from good_var.")
                    for path in results.get('good_var', []): # Use .get with default empty list
                        resolved_indices = resolve_path(augmented_model, path)
                        relevant_tokens_for_check.update(resolved_indices)
                        if DEBUG: print(f"[DEBUG_BLAME]         - Good var path '{path}' resolved to indices: {resolved_indices}")
                else: # Check failed or skipped
                    if DEBUG: print(f"[DEBUG_BLAME]     -> Check FAILED or SKIPPED. Attributing blame from error_var.")
                    
                    if results['message'] != "Skipped":
                        if check_name == "_all_field_pass_checks":
                            dependency_ids = [path.split('.')[-1] for path in results.get('error_var', [])]
                            if DEBUG: print(f"[DEBUG_BLAME]       - _all_field_pass_checks dependency IDs: {dependency_ids}")
                            for dep_id in dependency_ids: 
                                if dep_id in root_causes:
                                    relevant_tokens_for_check.update(root_causes.get(dep_id, set()))
                                    if DEBUG: print(f"[DEBUG_BLAME]         - Added blame from dependent object '{dep_id}': {root_causes.get(dep_id, set())}")
                        else:
                            for path in results.get('error_var', []): # Use .get with default empty list
                                resolved_indices = resolve_path(augmented_model, path)
                                relevant_tokens_for_check.update(resolved_indices)
                                if DEBUG: print(f"[DEBUG_BLAME]         - Error var path '{path}' resolved to indices: {resolved_indices}")
                            
                            # PART 2 FIX (part 1): Fallback if no specific path blamed
                            if not relevant_tokens_for_check and obj_id != 'global': 
                                # Attempt to find the full object if the path doesn't point to a specific sub-attribute
                                component_type_key = None
                                if obj_id.startswith('f') and 'particles' in augmented_model: component_type_key = 'particles'
                                elif obj_id.startswith('m') and 'fields' in augmented_model: component_type_key = 'fields'
                                elif obj_id.startswith('i') and 'interactions' in augmented_model: component_type_key = 'interactions'

                                if component_type_key and obj_id in augmented_model.get(component_type_key, {}):
                                    obj_tokens = collect_all_indices(augmented_model[component_type_key][obj_id])
                                    relevant_tokens_for_check.update(obj_tokens)
                                    if DEBUG: print(f"[DEBUG_BLAME]         - Fallback: No specific var blamed, blaming entire object '{obj_id}': {obj_tokens}")

                    else: # Message is "Skipped"
                        if obj_id == 'global':
                            # For global anomaly checks skipped, blame all fermion fields involved
                            for fid in fermion_field_ids: 
                                if fid in root_causes:
                                    relevant_tokens_for_check.update(root_causes.get(fid, set()))
                                    if DEBUG: print(f"[DEBUG_BLAME]         - Global check skipped, blaming fermion field '{fid}': {root_causes.get(fid, set())}")
                        else: 
                            # For non-global skipped checks, propagate blame from its direct object
                            if obj_id in root_causes:
                                relevant_tokens_for_check.update(root_causes.get(obj_id, set()))
                                if DEBUG: print(f"[DEBUG_BLAME]         - Skipped check, blaming object '{obj_id}': {root_causes.get(obj_id, set())}")

                # Update the blame map for this specific check
                current_blame_for_check_key = blame_map.get((obj_id, check_name), [])
                if set(current_blame_for_check_key) != relevant_tokens_for_check:
                    blame_map[(obj_id, check_name)] = sorted(list(relevant_tokens_for_check))
                    blame_added_this_pass = True

                # Propagate to root_causes for next iteration (important for transitive blame)
                old_root_causes_for_obj = root_causes.get(obj_id, set())
                if not relevant_tokens_for_check.issubset(old_root_causes_for_obj):
                    root_causes.setdefault(obj_id, set()).update(relevant_tokens_for_check)
                    blame_added_this_pass = True
                    if DEBUG: print(f"[DEBUG_BLAME]     -> Updated root_causes for '{obj_id}' with new tokens. Total for '{obj_id}': {root_causes.get(obj_id)}")

        if not blame_added_this_pass:
            if DEBUG: print("[DEBUG_BLAME] No new blame added in this pass, stopping iteration.")
            break
    
    if DEBUG: print("[DEBUG_BLAME] map_failures_to_tokens completed.")
    return blame_map
def convert_to_keyed_dict_with_provenance(list_of_dicts: List[Dict]) -> Dict[str, Dict]:
    return {item["id"]["value"]: item for item in list_of_dicts}
def strip_provenance(data):
    if isinstance(data, dict):
        if "value" in data and "token_indices" in data: return strip_provenance(data["value"])
        return {k: strip_provenance(v) for k, v in data.items()}
    if isinstance(data, list): return [strip_provenance(i) for i in data]
    return data

def process_tokens(
    tokens: List[str], model_name: str = "Generated Model", author: str = "Bohr Network", output_path: str = "./output"
) -> Tuple[Optional[AdvancedModel], Dict]:
    if DEBUG:
        print("\n" + "="*20 + " New Sequence to Process " + "="*20)
        print(f"[DEBUG] Received {len(tokens)} tokens: {' '.join(tokens)}")

    try:
        tester = ModelTester()
        augmented_model_list_format = tester.build_model(tokens)
        clean_model_dict_list_format = strip_provenance(augmented_model_list_format)
        
        clean_model_dict_keyed = {
            "GaugeGroups": {d.pop('id'): d for d in clean_model_dict_list_format.get("GaugeGroups", [])},
            "particles": {d.pop('id'): d for d in clean_model_dict_list_format.get("particles", [])},
            "fields": {d.pop('id'): d for d in clean_model_dict_list_format.get("fields", [])},
            "interactions": {d.pop('id'): d for d in clean_model_dict_list_format.get("interactions", [])},
        }
        
        if DEBUG:
            print("[DEBUG] Successfully parsed into dictionary structure.")
            print(json.dumps(clean_model_dict_keyed, indent=2))

        json_path = None
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump(clean_model_dict_keyed, tf, indent=4)
            json_path = tf.name
        
        model = AdvancedModel(model_name, author, json_path, output_path)
        
        augmented_model_keyed = {
            "particles": convert_to_keyed_dict_with_provenance(augmented_model_list_format["particles"]),
            "fields": convert_to_keyed_dict_with_provenance(augmented_model_list_format["fields"]),
            "interactions": convert_to_keyed_dict_with_provenance(augmented_model_list_format["interactions"]),
        }
        blame_map = map_failures_to_tokens(model.checklist, augmented_model_keyed)
        
        if DEBUG:
            print("\n[DEBUG] Final blame map (failed checks and their token indices):")
            for (obj_id, check_name), indices in sorted(blame_map.items()):
                if indices:
                    print(f"  {obj_id} -> {check_name}: {indices}")
        
        return model, blame_map
    
    except Exception as e:
        if DEBUG:
            import traceback
            print(f"\n[DEBUG] FATAL ERROR during token processing: {type(e).__name__}: {e}")
            traceback.print_exc()
        return None, {} # Return a failed state
    finally:
        if 'json_path' in locals() and json_path and os.path.exists(json_path):
            os.unlink(json_path)

if __name__ == '__main__':
    # This block is now safe to run for standalone testing of the parsing logic.
    example_tokens = [
        "ITRACT", "ITRACT_ID_0", "TYPE_YUKAWA",
        "FIELD", "FIELD_ID_2", "TYPE_FERMION", "DIM_2", "GEN_1", "SELF_CONJ_FALSE", "CHIRALITY_LEFT",
        "SU3C_REP_1", "SU2L_REP_2", "U1Y_CHARGE_-3", "QN_L_1", "QN_B_0",
        "PARTICLE", "PARTICLE_ID_0", "TYPE_FERMION", "MASS_1e2", "CHARGE_0", "END_PARTICLE",
        "PARTICLE", "PARTICLE_ID_1", "TYPE_FERMION", "MASS_1e2", "CHARGE_-3", "END_PARTICLE",
        "END_FIELD",
        "END_ITRACT"
    ]
    model_instance, token_blame = process_tokens(tokens=example_tokens)

    print("\n--- Token Blame Report (Example) ---")
    if not token_blame:
        print("No failed checks found.")
    else:
        for (obj_id, check_name), indices in sorted(token_blame.items()):
            if indices:
                print(f"\nFailed Check: {obj_id} -> {check_name}")
                print(f"  Token Indices: {indices}")