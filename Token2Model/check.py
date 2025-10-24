# Import DEBUG from config
from config import get_config
DEBUG = get_config().DEBUG_PRINTS

def get_all_possible_check_names():
    """
    Returns a definitive, ordered list of all possible check names.
    This list defines the output vector of the new Critic network.
    """
    # This list must be comprehensive and the order must be static.
    # We will build it from the various _all_checks and _all_validations methods.
    
    # Particle Checks
    particle_checks = [
        '_type_check', '_name_check', '_mass_check', '_charge_check'
    ]
    
    # Field Checks
    field_checks = [
        '_name_check', '_type_check', '_groups_check', '_reps_check', 
        '_dim_check', '_gen_check', '_particles_check', '_self_conjugate_check', 
        '_sort_reps', '_reps_dim_consistency', '_gen_type_consistency', 
        '_allowed_charges', '_deplicate_particles', '_particle_numbers', 
        '_particle_types', '_particle_charges', '_all_particle_pass', 
        '_sort_particles'
    ]
    
    # Fermion-specific Field Checks
    fermion_field_checks = ['_chirality_check', '_assign_colors']
    
    # Field Validation Checks
    field_validation_checks = ['_mass_term_check', '_potential_term_check']
    
    # Interaction Checks
    interaction_checks = [
        '_field_length_check', '_field_check', '_params_check', 
        '_all_field_pass_checks', '_check_replicate_fields', '_sort_field'
    ]
    
    # Yukawa-specific Checks
    yukawa_checks = ['_gen_check', '_dim_check', '_dirac_bilinear_product', 
                     '_get_massive_particles', '_check_U1Y_gauge_symmetry', 
                     '_yukawa_mass', '_yukawa_matrix', '_mixing_matrix', 
                     '_yukawa_lagrangian']

    # Scalar Self-Interaction Checks
    scalar_self_interaction_checks = ['_scalar_mass', '_scalar_quartic', '_scalar_lagrangian']

    # Global Anomaly Checks
    global_checks = [
        '(left)^3', '(color)^3', '(hypercharge)x(left)^2', 
        '(hypercharge)x(color)^2', '(hypercharge)^3', '(hypercharge)^2-grav'
    ]
    
    # Combine, ensure uniqueness, and sort to guarantee a fixed order
    all_checks = sorted(list(set(
        particle_checks + field_checks + fermion_field_checks + 
        field_validation_checks + interaction_checks + 
        yukawa_checks + scalar_self_interaction_checks + global_checks
    )))
    
    return all_checks

# We can also create a mapping for convenience
CHECK_NAMES = get_all_possible_check_names()
CHECK_TO_IDX = {name: i for i, name in enumerate(CHECK_NAMES)}
IDX_TO_CHECK = {i: name for i, name in enumerate(CHECK_NAMES)}
NUM_CHECKS = len(CHECK_NAMES)

def run_checks(all_checks, checklists, skip_results=True):
    if DEBUG: print(f"[DEBUG_CHECKS] Starting run_checks for {len(all_checks)} checks.")
    for check_item in all_checks:
        check_func, max_score = check_item
        check_name = check_func.__name__
        if DEBUG: print(f"[DEBUG_CHECKS]   - Running check: {check_name} (Max Score: {max_score})")

        fail_previous_check = False
        if skip_results:
            for result in checklists.values():
                # Check for failure: score < max_score (only applies to checks that ran)
                if isinstance(result, dict) and 'message' in result and result['message'] not in ["Skipped", "CRASHED"] and result.get("score", 0) != result.get("max_score", 1):
                    fail_previous_check = True
                    break
        
        if fail_previous_check and skip_results:
            # Mark as skipped if a prior check in this instance's list failed
            result = {"score": 0, "max_score": max_score, "error_var": [], "good_var": [], "message": "Skipped"}
            if DEBUG: print(f"[DEBUG_CHECKS]     -> SKIPPED (previous check failed). Result: {result}")
            checklists[check_name] = result
        else:
            try:
                # Run the check function
                result = check_func()
                # Ensure the result is a dict and contains max_score
                if not isinstance(result, dict) or 'max_score' not in result:
                    raise TypeError(f"Check {check_name} returned invalid result type or missing 'max_score'.")
                    
                if DEBUG: print(f"[DEBUG_CHECKS]     -> COMPLETED. Result: {result}")
                checklists[check_name] = result
            except Exception as e:
                # Catch unexpected crashes
                result = {"score": 0, "max_score": max_score, "error_var": [], "good_var": [], "message": f"CRASHED: {e}"}
                if DEBUG: print(f"[DEBUG_CHECKS]     -> CRASHED. Error: {e}. Result: {result}")
                checklists[check_name] = result
    if DEBUG: print(f"[DEBUG_CHECKS] Finished run_checks.")
    return checklists