import os
import tempfile
import shutil
import traceback
import copy
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import time

from config import Config, get_config
from .parser import SequenceParser, ParsingError
from Token2Model.model import Model as PhysicsModel 
from Token2Model.param import GlobalParameterRegistry # Import GlobalParameterRegistry 

@dataclass
class EnvOutcome:
    checklist: Dict[str, Any]
    token_map: Dict[Tuple, List[int]]
    unclosed_block_info: Optional[Tuple[int, int]] # (start_index, depth)
    parsing_diagnostics: Dict[str, List[str]] = None
    success: bool = True
    meta: Dict[str, Any] = None


def propagate_skipped_errors(checklist: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Propagates error_var/good_var from a preceding failed check to subsequent 
    skipped checks within the same object instance's checklist.
    Also handles the cascading failure to global checks.
    """
    propagated_checklist = copy.deepcopy(checklist)
    all_failed_vars = {'error_var': set(), 'good_var': set()}

    # 1. Iterate over object instances (P, F, I)
    for obj_id, checks in propagated_checklist.items():
        if obj_id == 'global':
            continue
            
        last_failed_result = None

        # Sort checks to ensure sequential processing (keys are insertion ordered in Python >= 3.7)
        for check_name, result in checks.items():
            if result['message'] == "Passed":
                last_failed_result = None
            elif result['message'] == "Skipped" and last_failed_result is not None:
                # Propagate from the last failed result
                result['error_var'] = last_failed_result['error_var']
                result['good_var'] = last_failed_result['good_var']
                propagated_checklist[obj_id][check_name] = result
                
                # Keep last_failed_result as it is (a skip does not clear it)
            elif result['score'] < result['max_score'] and result['message'] != "Skipped":
                # New failure detected, store it
                last_failed_result = result
            elif result['message'] == "CRASHED":
                 # Keep crash result active for propagation
                 last_failed_result = result

        # 2. After checking an object, consolidate all failure indicators for global checks
        for check_name, result in checks.items():
             if result['score'] < result['max_score'] and result['message'] not in ["Skipped", "CRASHED"]:
                all_failed_vars['error_var'].update(result['error_var'])
                all_failed_vars['good_var'].update(result['good_var'])
    
    # 3. Propagate to Global Checks (if any check failed)
    if all_failed_vars['error_var'] or any(result['score'] < result['max_score'] for obj_id, checks in checklist.items() if obj_id != 'global' for result in checks.values()):
        if 'global' in propagated_checklist:
            for check_name, result in propagated_checklist['global'].items():
                if result['message'] == "Skipped":
                    # Global checks are skipped if *any* object failed, so propagate
                    result['error_var'] = list(all_failed_vars['error_var'])
                    result['good_var'] = list(all_failed_vars['good_var'])
                    propagated_checklist['global'][check_name] = result
                # Note: Global checks are directly calculated, so only 'Skipped' state is updated here.

    return propagated_checklist


class ParticlePhysicsEnvironment:
    """
    Acts as the bridge between the RL agent and the physics model validation logic.
    Uses a robust parser to construct a (potentially partial) model.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.parser = SequenceParser()
        self.meta = {"error": None}
        self.model_base_path = cfg.MODEL_BASE_PATH # Retrieve from config

    def run_episode(self, sequence_ids: List[int], idx_to_token: Dict[int, str]) -> EnvOutcome:
        token_strs = [idx_to_token.get(i, "PAD") for i in sequence_ids]
        
        temp_output_dir = None
        model_instance = None
        parsed_model_dict = {} 
        token_map_from_parser = {} 
        unclosed_info_from_parser = None 
        parsing_diagnostics = {'failed_fields': [], 'failed_particles': []}
        
        env_timing: Dict[str, float] = {}

        try:
            # Clear global state at the start of every episode
            GlobalParameterRegistry.clear() 

            # [TIME] Environment & Reward: Parsing Sequence
            # 1. Parse the sequence
            start_parse = time.perf_counter()
            parsed_model_dict, token_map_from_parser, unclosed_info_from_parser, parsing_diagnostics = self.parser.parse(token_strs)
            env_timing['parsing_time'] = time.perf_counter() - start_parse
            
            # 2. Prepare a temporary directory
            temp_output_dir = tempfile.mkdtemp()

            # [TIME] Environment & Reward: Model Init & Validation
            # 3. Instantiate the Physics Model (triggers internal checks & validation)
            start_model_init = time.perf_counter()
            model_instance = PhysicsModel(
                model_name="RLGeneratedModel",
                author="RL Agent",
                output_directory=temp_output_dir, 
                model_data_dict=parsed_model_dict
            )
            env_timing['model_init_total'] = time.perf_counter() - start_model_init
            
            # [TIME] Environment & Reward: Check Propagate & Anomaly
            # 4. Propagate errors to skipped checks
            start_propagate = time.perf_counter()
            propagated_checklist = propagate_skipped_errors(model_instance.checklist)
            env_timing['check_propagate_time'] = time.perf_counter() - start_propagate

            # Check if model passes all internal physics checks
            model_passes_checks = model_instance.pass_all_checks()
            
            env_meta = {"message": "Model built and checked successfully." if model_passes_checks else "CHECK FAILURE: Model built but failed internal checks."}
            env_meta['timing'] = env_timing
            return EnvOutcome(
                checklist=propagated_checklist, 
                token_map=token_map_from_parser,
                unclosed_block_info=unclosed_info_from_parser,
                parsing_diagnostics=parsing_diagnostics,
                success=model_passes_checks, 
                meta=env_meta
            )

        except ParsingError as e:
            # Errors from the parser
            env_timing['parsing_time'] = time.perf_counter() - start_parse if 'start_parse' in locals() else 0.0
            env_meta = {"error": f"CRITICAL FAILURE: ParsingError: {e}"}
            env_meta['timing'] = env_timing
            return EnvOutcome(
                checklist={}, 
                token_map=token_map_from_parser, 
                unclosed_block_info=self.parser._find_first_unclosed_block(token_strs),
                parsing_diagnostics=parsing_diagnostics,
                success=False,
                meta=env_meta
            )
        except Exception as e:
            # Other unexpected errors
            if get_config().DEBUG_PRINTS:
                print("\n" + "="*50)
                print(">>> CRITICAL MODEL INITIALIZATION ERROR CAUGHT <<<")
                traceback.print_exc()
                print("="*50 + "\n")
            if 'start_model_init' in locals():
                env_timing['model_init_total'] = time.perf_counter() - start_model_init
            if model_instance:
                propagated_checklist = propagate_skipped_errors(model_instance.checklist)
                env_meta = {"error": f"CRITICAL FAILURE: ModelError: {e}"}
                env_meta['timing'] = env_timing
                return EnvOutcome(
                    checklist=propagated_checklist, 
                    token_map=token_map_from_parser,
                    unclosed_block_info=unclosed_info_from_parser,
                    parsing_diagnostics=parsing_diagnostics,
                    success=False,
                    meta=env_meta
                )
            else: 
                env_meta = {"error": f"CRITICAL FAILURE: Model initialization error: {e}"}
                env_meta['timing'] = env_timing
                return EnvOutcome(
                    checklist={}, 
                    token_map=token_map_from_parser,
                    unclosed_block_info=unclosed_info_from_parser,
                    parsing_diagnostics=parsing_diagnostics,
                    success=False,
                    meta=env_meta
                )
        finally:
            if temp_output_dir and os.path.exists(temp_output_dir):
                if not self.cfg.KEEP_GENERATED_MODEL_FILES:
                    shutil.rmtree(temp_output_dir)
                else:
                    # If keeping files, rename the temp dir to something more meaningful
                    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    perm_dir = os.path.join(self.model_base_path, f"RL_run_{run_id}") # Use self.model_base_path
                    os.makedirs(os.path.dirname(perm_dir), exist_ok=True)
                    shutil.move(temp_output_dir, perm_dir)
                    if self.cfg.DEBUG_PRINTS: print(f"Kept model files in: {perm_dir}")