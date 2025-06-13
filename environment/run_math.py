import subprocess
import os
import time
import signal
import sys

FEYNRULES_PATH = "/oscar/home/qniu3/physics/FeynRules"
MODEL_PATH = "/users/qniu3/physics/RL_model_builder/theories/SM_2025-06-09-22-31-11"

def run_math():
    # First load the module using Lmod
    load_process = subprocess.run(
        ["/bin/bash", "-c", "module load mathematica"],
        capture_output=True,
        text=True
    )
    
    if load_process.returncode != 0:
        print("Failed to load Mathematica module:")
        print(load_process.stderr)
        return None

    process = subprocess.Popen(
        ["/bin/bash", "-c", "module load mathematica && math"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    open_feyrules = [f"$FeynRulesPath = SetDirectory[\"{FEYNRULES_PATH}\"];", 
                     "<< FeynRules`",
                     f"$ModelPath = SetDirectory[\"{MODEL_PATH}\"];",
                     "LoadModel[\"SM.fr\"];",
                     "CheckMassSpectrum[LSM];",
                     "CheckHermiticity[LSM];",
                     ]
    for command in open_feyrules:
        process.stdin.write(command + "\n")
    process.stdin.flush()
    return process


class RunFeynRules:
    def __init__(self, feynrules_path: str) -> None:
        self.feynrules_path = feynrules_path
        self.process = None

        try:
            subprocess.run(["/bin/bash", "-c", "module load mathematica"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to load Mathematica module: {e}")

    def open_feyrules(self, model_path: str) -> None:
        self.process = subprocess.Popen(
            ["/bin/bash", "-c", "module load mathematica && math"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        open_feyrules = [f"$FeynRulesPath = SetDirectory[\"{self.feynrules_path}\"];", 
                         "<< FeynRules`",
                         f"$ModelPath = SetDirectory[\"{model_path}\"];",
                         "LoadModel[\"SM.fr\"];"
                         ]
        for command in open_feyrules:
            self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def check_hermiticity(self) -> bool:
        if self.process is None:
            pass
        self.process.stdin.write("CheckHermiticity[LSM]\n")
        self.process.stdin.flush()
        time.sleep(1)
        stdout, stderr = self.process.communicate()
        print(stdout)
        print(stderr)
        return True  # You'll need to parse the output to determine the actual result

    def check_mass_spectrum(self) -> bool:
        self.process.stdin.write("CheckMassSpectrum[LSM]\n")
        self.process.stdin.flush()
        time.sleep(1)
        stdout, stderr = self.process.communicate()
        print(stdout)
        print(stderr)
        return True  # You'll need to parse the output to determine the actual result




if __name__ == "__main__":
    # math_process = run_math()
    # if math_process:
    #     # You can interact with the process here if needed
    #     stdout, stderr = math_process.communicate()
    #     print(stdout)
    #     print(stderr)

    run_feynrules = RunFeynRules(FEYNRULES_PATH)
    run_feynrules.open_feyrules(MODEL_PATH)
