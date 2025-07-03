import os 
import subprocess
import time
from typing import Dict


FEYNRULES_PATH = "/users/qniu3/physics/FeynRules"
FEYNAFTS_PATH = "/users/qniu3/physics/FeynArts"
NLOCT_PATH = "/users/qniu3/physics/NLO"
MADGRAPH5_PATH = "/users/qniu3/physics/MadGraph5_v1_5_15"
MODEL_PATH = "/users/qniu3/physics/RL_model_builder/theories/SM"
MODEL_NAME = "SM"

def write_fr_script(feynrules_path: str, 
                    model_path: str, 
                    model_name: str
                    ) -> str:
    fr_script = f"""
$FeynRulesPath = SetDirectory["{feynrules_path}"];
<< FeynRules`
$ModelPath = SetDirectory["{model_path}"];
LoadModel["{model_name}.fr"];
FeynmanGauge = False;

(* Mass spectrum check *)
result1 = Check[CheckMassSpectrum[LSM], $Failed];
If[result1 === $Failed, Print["result1: failed"], Print["result1: passed"]];
If[result1 === $Failed, Quit[]];

(* Hermiticity check with flavor expansion *)
result2 = Check[CheckHermiticity[LSM, FlavorExpand -> True], $Failed];
If[result2 === $Failed, Print["result2: failed"], Print["result2: passed"]];
If[result2 === $Failed, Quit[]];

(* Kinetic term normalization check *)
result3 = Check[CheckKineticTermNormalisation[LSM, FlavorExpand -> SU2W], $Failed];
If[result3 === $Failed, Print["result3: failed"], Print["result3: passed"]];
If[result3 === $Failed, Quit[]];
"""
    return fr_script

FEYNRULES_PATH = "/users/qniu3/physics/FeynRules"
FEYNAFTS_PATH = "/users/qniu3/physics/FeynArts"
NLOCT_PATH = "/users/qniu3/physics/FeynRules"
MADGRAPH5_PATH = "/users/qniu3/physics/MG5_aMC_v3_6_2"

class Verifier:
    def __init__(self, 
                 feynrules_path = FEYNRULES_PATH, 
                 feynarts_path = FEYNAFTS_PATH,
                 nloct_path = NLOCT_PATH,
                 madgraph5_path = MADGRAPH5_PATH
                 ) -> None:
        
        self.feynrules_path = feynrules_path
        self.feynarts_path = feynarts_path
        self.nloct_path = nloct_path
        self.madgraph5_path = madgraph5_path

    def run_fr(self, model_path: str, model_name: str) -> None:
        fr_script = write_fr_script(self.feynrules_path, model_path, model_name)
        script_file = os.path.join(model_path, "temp_verifier.m")

        print("Writing script file...")
        with open(script_file, "w") as f:
            f.write(fr_script)
        print("Script file written successfully at ", script_file)
        
        print("Running script file...")
        try:
            process = subprocess.run(
                ["/bin/bash", "-c", f"module load mathematica && math -script {script_file}"],
                capture_output=True,
                text=True,
                timeout=300 
            )
            results = {"mass_spectrum": 0, "hermiticity": 0, "kinetic_term": 0}

            for line in process.stdout.split('\n'):
                if line.startswith('result1:'):
                    results['mass_spectrum'] = 0 if line.split(':')[1].strip() == "failed" else 1
                elif line.startswith('result2:'):
                    results['hermiticity'] = 0 if line.split(':')[1].strip() == "failed" else 1
                elif line.startswith('result3:'):
                    results['kinetic_term'] = 0 if line.split(':')[1].strip() == "failed" else 1
            
            print("\nCheck Results:")
            for check_name, result in results.items():
                status = "PASSED" if result == 1 else "FAILED"
                print(f"{check_name}: {status}")
            
            if process.stderr:
                print("Mathematica errors:")
                print(process.stderr)
                
            if process.returncode != 0:
                print(f"Mathematica exited with code: {process.returncode}")
                
        except subprocess.TimeoutExpired:
            print("Mathematica execution timed out after 5 minutes")
        except Exception as e:
            print(f"Error running Mathematica: {e}")
        finally:
            if os.path.exists(script_file):
                os.remove(script_file)

if __name__ == "__main__":
    verifier = Verifier(FEYNRULES_PATH, FEYNAFTS_PATH, NLOCT_PATH, MADGRAPH5_PATH)
    verifier.run_fr(MODEL_PATH, MODEL_NAME)