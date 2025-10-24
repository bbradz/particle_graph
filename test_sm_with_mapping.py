import sys
import os
import tempfile
import shutil
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the SM tokens
SM = [
    'BOS',
    # --- Lepton Sector ---
    'ITRACT', 'ITRACT_ID_1', 'TYPE_YUKAWA',
        # Left-handed Lepton Doublets (L)
        'FIELD', 'FIELD_ID_1', 'TYPE_fermion', 'DIM_2', 'GEN_3', 'SELF_CONJ_FALSE', 'CHIRALITY_left',
            'SU3C_REP_1', 'SU2L_REP_2', 'U1Y_CHARGE_-1', 'QN_L_1', 'QN_B_0',
                # Generation 1: (nu_e, e-)
                'PARTICLE', 'PARTICLE_ID_1', 'TYPE_fermion', 'MASS_1e-9', 'CHARGE_0', 'END_PARTICLE',     # nu_e (small mass)
                'PARTICLE', 'PARTICLE_ID_2', 'TYPE_fermion', 'MASS_1e-4', 'CHARGE_-1', 'END_PARTICLE',    # e-
                # Generation 2: (nu_mu, mu-)
                'PARTICLE', 'PARTICLE_ID_3', 'TYPE_fermion', 'MASS_1e-9', 'CHARGE_0', 'END_PARTICLE',     # nu_mu (small mass)
                'PARTICLE', 'PARTICLE_ID_4', 'TYPE_fermion', 'MASS_1e-1', 'CHARGE_-1', 'END_PARTICLE',    # mu-
                # Generation 3: (nu_tau, tau-)
                'PARTICLE', 'PARTICLE_ID_5', 'TYPE_fermion', 'MASS_1e-9', 'CHARGE_0', 'END_PARTICLE',     # nu_tau (small mass)
                'PARTICLE', 'PARTICLE_ID_6', 'TYPE_fermion', 'MASS_1e0', 'CHARGE_-1', 'END_PARTICLE',     # tau-
            'END_FIELD',
        # Right-handed Charged Lepton Singlets (eR)
        'FIELD', 'FIELD_ID_2', 'TYPE_fermion', 'DIM_1', 'GEN_3', 'SELF_CONJ_FALSE', 'CHIRALITY_right',
            'SU3C_REP_1', 'SU2L_REP_1', 'U1Y_CHARGE_-1', 'QN_L_1', 'QN_B_0',
                'PARTICLE', 'PARTICLE_ID_2', 'TYPE_fermion', 'MASS_1e-4', 'CHARGE_-1', 'END_PARTICLE',   # e-
                'PARTICLE', 'PARTICLE_ID_4', 'TYPE_fermion', 'MASS_1e-1', 'CHARGE_-1', 'END_PARTICLE',   # mu-
                'PARTICLE', 'PARTICLE_ID_6', 'TYPE_fermion', 'MASS_1e0', 'CHARGE_-1', 'END_PARTICLE',    # tau-
            'END_FIELD',
        # Higgs Doublet
        'FIELD', 'FIELD_ID_3', 'TYPE_complex', 'DIM_2', 'GEN_1', 'SELF_CONJ_FALSE', 'CHIRALITY_none',
            'SU3C_REP_1', 'SU2L_REP_2', 'U1Y_CHARGE_1', 'QN_L_0', 'QN_B_0',
                'PARTICLE', 'PARTICLE_ID_7', 'TYPE_complex', 'MASS_1e2', 'CHARGE_1', 'END_PARTICLE',    # H+
                'PARTICLE', 'PARTICLE_ID_8', 'TYPE_complex', 'MASS_1e2', 'CHARGE_0', 'END_PARTICLE',    # h0
            'END_FIELD',
    'END_ITRACT',
    # --- Up-type Quark Sector ---
    'ITRACT', 'ITRACT_ID_2', 'TYPE_YUKAWA',
        # Left-handed Quark Doublets (Q)
        'FIELD', 'FIELD_ID_4', 'TYPE_fermion', 'DIM_2', 'GEN_3', 'SELF_CONJ_FALSE', 'CHIRALITY_left',
            'SU3C_REP_3', 'SU2L_REP_2', 'U1Y_CHARGE_1', 'QN_L_0', 'QN_B_1',
                # Generation 1: (u, d)
                'PARTICLE', 'PARTICLE_ID_9', 'TYPE_fermion', 'MASS_1e-3', 'CHARGE_1', 'END_PARTICLE',   # u
                'PARTICLE', 'PARTICLE_ID_10', 'TYPE_fermion', 'MASS_1e-3', 'CHARGE_-1', 'END_PARTICLE', # d
                # Generation 2: (c, s)
                'PARTICLE', 'PARTICLE_ID_11', 'TYPE_fermion', 'MASS_1e0', 'CHARGE_1', 'END_PARTICLE',   # c
                'PARTICLE', 'PARTICLE_ID_12', 'TYPE_fermion', 'MASS_1e-2', 'CHARGE_-1', 'END_PARTICLE', # s
                # Generation 3: (t, b)
                'PARTICLE', 'PARTICLE_ID_13', 'TYPE_fermion', 'MASS_1e2', 'CHARGE_1', 'END_PARTICLE',   # t
                'PARTICLE', 'PARTICLE_ID_14', 'TYPE_fermion', 'MASS_1e0', 'CHARGE_-1', 'END_PARTICLE',  # b
            'END_FIELD',
        # Right-handed Up-type Quark Singlets (uR)
        'FIELD', 'FIELD_ID_5', 'TYPE_fermion', 'DIM_1', 'GEN_3', 'SELF_CONJ_FALSE', 'CHIRALITY_right',
            'SU3C_REP_3', 'SU2L_REP_1', 'U1Y_CHARGE_2', 'QN_L_0', 'QN_B_1',
                'PARTICLE', 'PARTICLE_ID_9', 'TYPE_fermion', 'MASS_1e-3', 'CHARGE_1', 'END_PARTICLE',   # u
                'PARTICLE', 'PARTICLE_ID_11', 'TYPE_fermion', 'MASS_1e0', 'CHARGE_1', 'END_PARTICLE',   # c
                'PARTICLE', 'PARTICLE_ID_13', 'TYPE_fermion', 'MASS_1e2', 'CHARGE_1', 'END_PARTICLE',   # t <--- TOP QUARK
            'END_FIELD',
        # Higgs Doublet
        'FIELD', 'FIELD_ID_3', 'TYPE_complex', 'DIM_2', 'GEN_1', 'SELF_CONJ_FALSE', 'CHIRALITY_none',
            'SU3C_REP_1', 'SU2L_REP_2', 'U1Y_CHARGE_1', 'QN_L_0', 'QN_B_0',
                'PARTICLE', 'PARTICLE_ID_7', 'TYPE_complex', 'MASS_1e2', 'CHARGE_1', 'END_PARTICLE',
                'PARTICLE', 'PARTICLE_ID_8', 'TYPE_complex', 'MASS_1e2', 'CHARGE_0', 'END_PARTICLE',
            'END_FIELD',
    'END_ITRACT',
    # --- Down-type Quark Sector ---
    'ITRACT', 'ITRACT_ID_3', 'TYPE_YUKAWA',
        # Left-handed Quark Doublets (Q)
        'FIELD', 'FIELD_ID_4', 'TYPE_fermion', 'DIM_2', 'GEN_3', 'SELF_CONJ_FALSE', 'CHIRALITY_left',
            'SU3C_REP_3', 'SU2L_REP_2', 'U1Y_CHARGE_1', 'QN_L_0', 'QN_B_1',
                'PARTICLE', 'PARTICLE_ID_9', 'TYPE_fermion', 'MASS_1e-3', 'CHARGE_1', 'END_PARTICLE',
                'PARTICLE', 'PARTICLE_ID_10', 'TYPE_fermion', 'MASS_1e-3', 'CHARGE_-1', 'END_PARTICLE',
                'PARTICLE', 'PARTICLE_ID_11', 'TYPE_fermion', 'MASS_1e0', 'CHARGE_1', 'END_PARTICLE',
                'PARTICLE', 'PARTICLE_ID_12', 'TYPE_fermion', 'MASS_1e-2', 'CHARGE_-1', 'END_PARTICLE',
                'PARTICLE', 'PARTICLE_ID_13', 'TYPE_fermion', 'MASS_1e2', 'CHARGE_1', 'END_PARTICLE',
                'PARTICLE', 'PARTICLE_ID_14', 'TYPE_fermion', 'MASS_1e0', 'CHARGE_-1', 'END_PARTICLE',
            'END_FIELD',
        # Right-handed Down-type Quark Singlets (dR)
        'FIELD', 'FIELD_ID_6', 'TYPE_fermion', 'DIM_1', 'GEN_3', 'SELF_CONJ_FALSE', 'CHIRALITY_right',
            'SU3C_REP_3', 'SU2L_REP_1', 'U1Y_CHARGE_-1', 'QN_L_0', 'QN_B_1',
                'PARTICLE', 'PARTICLE_ID_10', 'TYPE_fermion', 'MASS_1e-3', 'CHARGE_-1', 'END_PARTICLE', # d
                'PARTICLE', 'PARTICLE_ID_12', 'TYPE_fermion', 'MASS_1e-2', 'CHARGE_-1', 'END_PARTICLE', # s
                'PARTICLE', 'PARTICLE_ID_14', 'TYPE_fermion', 'MASS_1e0', 'CHARGE_-1', 'END_PARTICLE',  # b
            'END_FIELD',
        # Higgs Doublet
        'FIELD', 'FIELD_ID_3', 'TYPE_complex', 'DIM_2', 'GEN_1', 'SELF_CONJ_FALSE', 'CHIRALITY_none',
            'SU3C_REP_1', 'SU2L_REP_2', 'U1Y_CHARGE_1', 'QN_L_0', 'QN_B_0',
                'PARTICLE', 'PARTICLE_ID_7', 'TYPE_complex', 'MASS_1e2', 'CHARGE_1', 'END_PARTICLE',
                'PARTICLE', 'PARTICLE_ID_8', 'TYPE_complex', 'MASS_1e2', 'CHARGE_0', 'END_PARTICLE',
            'END_FIELD',
    'END_ITRACT',
    'EOS'
]
print(len(SM))


WITHOUT_TOP_QUARK = [
    'BOS',
    # --- Lepton Sector ---
    'ITRACT', 'ITRACT_ID_1', 'TYPE_YUKAWA',
        # Left-handed Lepton Doublets (L)
        'FIELD', 'FIELD_ID_1', 'TYPE_fermion', 'DIM_2', 'GEN_3', 'SELF_CONJ_FALSE', 'CHIRALITY_left',
            'SU3C_REP_1', 'SU2L_REP_2', 'U1Y_CHARGE_-1', 'QN_L_1', 'QN_B_0',
                # Generation 1: (nu_e, e-)
                'PARTICLE', 'PARTICLE_ID_1', 'TYPE_fermion', 'MASS_1e-9', 'CHARGE_0', 'END_PARTICLE',    
                'PARTICLE', 'PARTICLE_ID_2', 'TYPE_fermion', 'MASS_1e-4', 'CHARGE_-1', 'END_PARTICLE',    
                # Generation 2: (nu_mu, mu-)
                'PARTICLE', 'PARTICLE_ID_3', 'TYPE_fermion', 'MASS_1e-9', 'CHARGE_0', 'END_PARTICLE',     
                'PARTICLE', 'PARTICLE_ID_4', 'TYPE_fermion', 'MASS_1e-1', 'CHARGE_-1', 'END_PARTICLE',    
                # Generation 3: (nu_tau, tau-)
                'PARTICLE', 'PARTICLE_ID_5', 'TYPE_fermion', 'MASS_1e-9', 'CHARGE_0', 'END_PARTICLE',     
                'PARTICLE', 'PARTICLE_ID_6', 'TYPE_fermion', 'MASS_1e0', 'CHARGE_-1', 'END_PARTICLE',     
            'END_FIELD',
        # Right-handed Charged Lepton Singlets (eR)
        'FIELD', 'FIELD_ID_2', 'TYPE_fermion', 'DIM_1', 'GEN_3', 'SELF_CONJ_FALSE', 'CHIRALITY_right',
            'SU3C_REP_1', 'SU2L_REP_1', 'U1Y_CHARGE_-1', 'QN_L_1', 'QN_B_0',
                'PARTICLE', 'PARTICLE_ID_2', 'TYPE_fermion', 'MASS_1e-4', 'CHARGE_-1', 'END_PARTICLE',   
                'PARTICLE', 'PARTICLE_ID_4', 'TYPE_fermion', 'MASS_1e-1', 'CHARGE_-1', 'END_PARTICLE',   
                'PARTICLE', 'PARTICLE_ID_6', 'TYPE_fermion', 'MASS_1e0', 'CHARGE_-1', 'END_PARTICLE',    
            'END_FIELD',
        # Higgs Doublet
        'FIELD', 'FIELD_ID_3', 'TYPE_complex', 'DIM_2', 'GEN_1', 'SELF_CONJ_FALSE', 'CHIRALITY_none',
            'SU3C_REP_1', 'SU2L_REP_2', 'U1Y_CHARGE_1', 'QN_L_0', 'QN_B_0',
                'PARTICLE', 'PARTICLE_ID_7', 'TYPE_complex', 'MASS_1e2', 'CHARGE_1', 'END_PARTICLE',    
                'PARTICLE', 'PARTICLE_ID_8', 'TYPE_complex', 'MASS_1e2', 'CHARGE_0', 'END_PARTICLE',    
            'END_FIELD',
    'END_ITRACT',
    'EOS'
]

def get_all_possible_check_names():
    """Define all possible check names that could appear in the checklist."""
    return {
        # Gauge group checks
        'id', 'name', 'charge', 'group', 'group_type', 'N', 'coupling', 'boson',
        
        # Field checks (base Field class)
        '_id_check', '_name_check', '_type_check', '_groups_check', '_reps_check', 
        '_dim_check', '_gen_check', '_particles_check', '_self_conjugate_check', 
        '_QuantumNumber_check', '_ptcl_check', '_create_generation_index', 
        '_check_gen_type_consistency', '_check_reps',
        
        # FermionField specific checks
        '_chirality_check', '_assign_colors', '_sort_unphy_fields', '_sort_phy_fields', 
        '_check_mass_type',
        
        # Interaction checks (base Interaction class)
        '_field_length_check', '_field_type_check', '_field_check', 
        '_all_field_pass_checks', '_sort_field',
        
        # Yukawa specific checks
        '_gen_check', '_dim_check', '_assign_mass_type', '_yukawa_mass',
        
        # Particle checks
        '_type_check', '_id_check', '_name_check', '_mass_check', '_charge_check',
        
        # Group checks
        '_dim_check',
        
        # Fermion specific checks
        '_WeylSpinor'
    }

# Import the conversion function and scorer
from token2fr import tokens_to_fr, ModelTester
from old.bufffr import model_based_scorer, model_based_scorer_with_diagnostics, ALL_TOKEN_NAMES

def test_sm():
    """Test SM tokens with existing vocabulary"""
    print("Testing Standard Model tokens...")
    print(f"SM tokens: {len(SM)}")
    print(f"Current vocabulary size: {len(ALL_TOKEN_NAMES)}")
    print()
    
    # Create token to index mapping
    token_to_idx = {tok: i for i, tok in enumerate(ALL_TOKEN_NAMES)}
    
    # Test 1: Convert SM tokens to FeynRules format
    print("=" * 60)
    print("TEST 1: Converting SM tokens to FeynRules format")
    print("=" * 60)
    
    output_dir = tempfile.mkdtemp(prefix="sm_test_")
    
    try:
        # Convert SM tokens to FeynRules
        fr_path, free_params = tokens_to_fr(
            tokens=SM,
            model_name="Standard Model",
            author="SM Test",
            output_path=output_dir
        )
        
        print("✅ FeynRules conversion successful!")
        print(f"   • Main FR file: {fr_path}")
        print(f"   • Free parameters: {', '.join(free_params) if free_params else '(none)'}")
        
        tester = ModelTester()
        model_dict = tester.build_model(SM)
        
        print("Model built successfully from SM tokens!")
        print(f"   • Gauge groups: {len(model_dict.get('GaugeGroups', []))}")
        print(f"   • Particles: {len(model_dict.get('particles', []))}")
        print(f"   • Fields: {len(model_dict.get('fields', []))}")
        print(f"   • Interactions: {len(model_dict.get('interactions', []))}")
        
        # Test 3: Use model-based scorer with SM tokens
        print("\n" + "=" * 60)
        print("TEST 3: Using model-based scorer with SM tokens")
        print("=" * 60)
        
        # Convert SM tokens to indices
        sm_indices = []
        for token in SM:
            if token in token_to_idx:
                sm_indices.append(token_to_idx[token])
            else:
                sm_indices.append(0)  # Use index 0 as fallback
        
        # Convert to tensor and score
        sm_sequence = torch.tensor([sm_indices], device='cpu')
        
        # Score using model-based scorer
        scores, num_passed = model_based_scorer(
            sequences=sm_sequence,
            idx_to_token={i: tok for tok, i in token_to_idx.items()},
            level="all",
            verbose=True
        )
        
        print(f"✅ Model-based scoring completed!")
        print(f"   • Score: {scores[0].item()}")
        print(f"   • Checks passed: {num_passed}")
        print(f"   • Sequence length: {len(SM)}")
        
        # Test 4: Direct validation using json2fr
        print("\n" + "=" * 60)
        print("TEST 4: Direct validation using json2fr Model class")
        print("=" * 60)
        
        try:
            from json2fr.model import Model
            import json
            
            # Create temporary JSON file
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
                json.dump(model_dict, tf, indent=2)
                json_path = tf.name
            
            try:
                # Create model and validate
                model = Model("Standard Model", "SM Test", json_path, output_dir)
                model._read_model()  # This populates the score
                
                print(f"✅ Direct model validation completed!")
                print(f"   • Model score: {model.score}")
                print(f"   • Passes all checks: {model.pass_all_checks()}")
                
                # Show detailed checklist if available
                checks_passed = {}
                if hasattr(model, 'checklist') and model.checklist:
                    print(f"   • Detailed checks:")
                    passed_checks = 0
                    total_checks = 0
                    for item_id, item_checks in model.checklist.items():
                        if hasattr(item_checks, 'items'):
                            for check_name, check_result in item_checks.items():
                                status = "✅" if check_result is True else "❌"
                                print(f"     - {item_id}.{check_name}: {status}")
                                total_checks += 1
                                if check_result is True:
                                    passed_checks += 1
                                
                                # Aggregate checks by name
                                if check_name not in checks_passed:
                                    checks_passed[check_name] = []
                                checks_passed[check_name].append(1 if check_result is True else 0)
                    
                    # Calculate average scores for each check type
                    check_averages = {}
                    for check_name, results in checks_passed.items():
                        avg_score = sum(results) / len(results)
                        check_averages[check_name] = avg_score
                    
                    print(f"   • Summary: {passed_checks}/{total_checks} checks passed")
                    print(f"   • Score:   {passed_checks}")
                    print(f"   • Check averages: {check_averages}")
                    checks_passed = check_averages
                
            finally:
                os.unlink(json_path)
                
        except ImportError:
            print("⚠️  json2fr.model not available, skipping direct validation")
        except Exception as e:
            print(f"❌ Direct validation failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(output_dir)
            print(f"\nCleaned up temporary directory: {output_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up {output_dir}: {e}")

def io_sequence_test(tokens=None, keep_files=False, show=False):
    if tokens is None:
        tokens = SM
        model_name = "Standard Model"
        author = "SM Test"
    else:
        model_name = "Custom Model"
        author = "Custom Test"
    
    if keep_files:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"fr_model_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = tempfile.mkdtemp(prefix="sm_test_")
    
    try:
        fr_path, free_params = tokens_to_fr(
            tokens=tokens,
            model_name=model_name,
            author=author,
            output_path=output_dir
        )
        tester = ModelTester()
        model_dict = tester.build_model(tokens)
        total_checks = 0
        checks_passed = {}
        model_score = 0
        pass_all_checks = False
        
        try:
            from json2fr.model import Model
            import json
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
                json.dump(model_dict, tf, indent=2)
                json_path = tf.name
            try:
                model = Model(model_name, author, json_path, output_dir)
                model._read_model()
                
                # Initialize checks_passed with all possible checks set to 0.0
                all_possible_checks = get_all_possible_check_names()
                checks_passed = {check_name: 0.0 for check_name in all_possible_checks}
                
                if hasattr(model, 'checklist') and model.checklist:
                    passed_checks = 0
                    total_checks = 0
                    check_results = {}
                    
                    for item_id, item_checks in model.checklist.items():
                        if hasattr(item_checks, 'items'):
                            for check_name, check_result in item_checks.items():
                                total_checks += 1
                                if check_result is True:
                                    passed_checks += 1
                                
                                # Aggregate checks by name
                                if check_name not in check_results:
                                    check_results[check_name] = []
                                check_results[check_name].append(1 if check_result is True else 0)
                    
                    # Calculate average scores for each check type and overwrite the defaults
                    for check_name, results in check_results.items():
                        avg_score = sum(results) / len(results)
                        checks_passed[check_name] = avg_score
                    
                    model_score = passed_checks
                    pass_all_checks = (passed_checks == total_checks)
                else:
                    if hasattr(model, 'score'):
                        model_score = model.score
                    if hasattr(model, 'pass_all_checks'):
                        pass_all_checks = model.pass_all_checks()
            finally:
                os.unlink(json_path)
                
        except ImportError:
            print("⚠️  json2fr.model not available, skipping direct validation")
        except Exception as e:
            print(f"❌ Direct validation failed: {e}")

        results_dict = {
            "main_fr_file": fr_path,
            "free_params": free_params,
            "particles": len(model_dict.get('particles', [])),
            "fields": len(model_dict.get('fields', [])),
            "interactions": len(model_dict.get('interactions', [])),
            "total_checks": total_checks,
            "score": model_score,
            "checks_passed": checks_passed,
            "percent_passed": sum(checks_passed.values()) / len(checks_passed.values()),
            "pass_all_checks": sum(checks_passed.values()) == len(checks_passed.values()),
            "sequence_length": len(tokens),
            "output_directory": output_dir,
            "files_kept": keep_files,
        }
        if show: print(json.dumps(results_dict, indent=2))
        return results_dict
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up temporary directory only if keep_files is False
        if not keep_files:
            try:
                shutil.rmtree(output_dir)
                #print(f"\nCleaned up temporary directory: {output_dir}")
            except Exception as e:
                print(f"Warning: Could not clean up {output_dir}: {e}")
        else:
            print(f"\nFiles kept in directory: {output_dir}")

if __name__ == "__main__":
    print("TEST 1: Standard Model with temporary files")
    print("=" * 60)
    results = io_sequence_test(keep_files=False, show=True)

    print("\nTEST 3: Custom token sequence (subset of SM)")
    print("=" * 60)
    custom_seq = ['BOS', 'EOS']
    io_sequence_test(tokens=custom_seq, keep_files=False, show=True)