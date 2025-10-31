# Import DEBUG from config
from typing import Any
from sympy import false
from config import get_config
DEBUG = get_config().DEBUG_PRINTS


def get_all_possible_check_names():
    """
    Returns a definitive, ordered list of all possible check names.
    This list defines the output vector of the new Critic network.
    """
    # This list must be comprehensive and the order must be static.
    # We will build it from the various _all_checks and _all_validations methods.
    
    # Particle Checks (from particle.py)
    particle_checks = [
        '_type_check', '_name_check', '_mass_check', '_charge_check'
    ]
    
    # Field Checks (from field.py - Field class)
    field_checks = [
        '_name_check', '_type_check', '_groups_check', 
        '_reps_type_check', '_reps_length_check', '_reps_value_check', 
        '_dim_check', '_gen_check', '_self_conjugate_check', 
        '_particle_list_check', '_all_particle_pass',
        '_particle_type_check', '_particle_number_check', 
        '_particle_field_consistency', '_gen_type_consistency', 
        '_chirality_check', '_sort_reps', '_reps_dim_consistency', 
        '_conj_charges_consistency', '_rep_charge_consistency', 
        '_sort_particles'
    ]
    
    # FermionField-specific Checks (from field.py - FermionField class)
    fermion_field_checks = ['_assign_colors']
    
    # Field Validation Checks (from field.py)
    field_validation_checks = ['_mass_term_check', '_potential_term_check']
    
    # Interaction Checks (from interaction.py - Interaction class)
    interaction_checks = [
        '_params_check', '_fields_integrity_check', '_field_check', 
        '_all_field_pass_checks', '_sort_field'
    ]
    
    # Yukawa-specific Checks (from interaction.py - Yukawa class)
    yukawa_checks = ['_gen_check', '_dim_check']
    
    # Yukawa Validation Checks (from interaction.py - Yukawa class)
    yukawa_validation_checks = [
        '_dirac_bilinear_product', '_get_massive_particles', 
        '_check_U1Y_gauge_symmetry', '_yukawa_mass', '_yukawa_matrix', 
        '_mixing_matrix', '_yukawa_lagrangian'
    ]

    # ScalarSelfInteraction Validation Checks (from interaction.py - ScalarSelfInteraction class)
    scalar_self_interaction_checks = [
        '_scalar_mass', '_scalar_quartic', '_scalar_lagrangian'
    ]

    # Global Anomaly Checks
    global_checks = [
        '(left)^3', '(color)^3', '(hypercharge)x(left)^2', 
        '(hypercharge)x(color)^2', '(hypercharge)^3', '(hypercharge)^2-grav'
    ]
    
    # Combine, ensure uniqueness, and sort to guarantee a fixed order
    all_checks = sorted(list(set(
        particle_checks + field_checks + fermion_field_checks + 
        field_validation_checks + interaction_checks + 
        yukawa_checks + yukawa_validation_checks + 
        scalar_self_interaction_checks + global_checks
    )))
    
    return all_checks

# We can also create a mapping for convenience
CHECK_NAMES = get_all_possible_check_names()
CHECK_TO_IDX = {name: i for i, name in enumerate(CHECK_NAMES)}
IDX_TO_CHECK = {i: name for i, name in enumerate(CHECK_NAMES)}
NUM_CHECKS = len(CHECK_NAMES)

def skip_check(max_score, level):
    result = {"score": 0, 
             "error_var": [], 
             "good_var": [],
             "mattered_vars": [],
             "message": "Skipped", 
             "max_score": max_score, 
             "level": level,
             }
    
    return result

    
def run_checks(all_checks, checklists, level, skip_results=False):
    if DEBUG: print(f"[DEBUG_CHECKS] Starting run_checks for {NUM_CHECKS} checks.")

    for check_item in all_checks:
        check_func, max_score = check_item
        check_name = check_func.__name__
        fail_previous_check = False

        if DEBUG: print(f"[DEBUG_CHECKS]   - Running check: {check_name} (Max Score: {max_score})")

        if skip_results:
            for result in checklists.values():
                if isinstance(result, dict) and 'message' in result and result['message'] not in ["Skipped", "CRASHED"] and result.get("score", 0) != result.get("max_score", 1):
                    fail_previous_check = True
                    break

        if fail_previous_check and skip_results:
            result = skip_check(max_score, level)
            if DEBUG: print(f"[DEBUG_CHECKS]     -> SKIPPED (previous check failed). Result: {result}")

        else:
            try:
                result = check_func()
                if not isinstance(result, dict) or 'max_score' not in result:
                    raise ValueError(f"Check function {check_name} returned an invalid result.")

                good_var = result.get('good_var', [])
                error_var = result.get('error_var', [])
                mattered_var = list[Any]({*good_var, *error_var})
                result.update({"mattered_vars": mattered_var})
                result.update({"level": level})
            
                if DEBUG: print(f"[DEBUG_CHECKS]     -> COMPLETED. Result: {result}")

            except Exception as e:
                result = {"score": 0, 
                          "error_var": [], 
                          "good_var": [],
                          "mattered_vars": [],
                          "message": f"CRASHED: {str(e)}", 
                          "max_score": max_score, 
                          "level": level,
                          }                
                if DEBUG: print(f"[DEBUG_CHECKS]     -> CRASHED. Error: {e}. Result: {result}")
        checklists[check_name] = result
            
    if DEBUG: print(f"[DEBUG_CHECKS] Finished run_checks.")
    return checklists
