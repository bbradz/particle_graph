import os 
import subprocess
import time
from typing import Dict

class Verifier:
    def __init__(self, 
                 feynrules_path: str, 
                 feynarts_path: str, 
                 nloct_path: str, 
                 madgraph5_path: str, 
                 model_path: str, 
                 model_name: str,
                 verbose: bool = False
                 ) -> None:
        self.feynrules_path = feynrules_path
        self.feynarts_path = feynarts_path
        self.nloct_path = nloct_path
        self.madgraph5_path = madgraph5_path
        self.model_path = model_path
        self.model_name = model_name
        self.verbose = verbose

    def run_feynrules(self) -> None:
        fr_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feynrules.m")
        print(f"Running FeynRules checks for {self.model_name}...")

        fr_process = subprocess.run(
            ["math", "-script", fr_file_path, 
             self.feynrules_path, 
             self.feynarts_path,
             self.model_path, 
             self.model_name],
            capture_output=True,
            text=True
            )
        print("FeynRules checks completed!")
        if self.verbose:
            print(fr_process.stdout)
        return None
    
    def run_feynarts(self) -> None:
        print(f"Running FeynArts computations for {self.model_name}...")
        fa_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feynarts.m")
        fa_process = subprocess.run(
            ["math", "-script", fa_file_path, 
             self.feynrules_path, 
             self.feynarts_path,
             self.model_path, 
             self.model_name],
            capture_output=True,
            text=True
            )
        print("FeynArts computations completed!")
        if self.verbose:
            print(fa_process.stdout)
        return None
    
    def run_nloct(self) -> None:
        print(f"Running NLO computations for {self.model_name}...")
        nlo_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feynnlo.m")
        nlo_process = subprocess.run(
            ["math", "-script", nlo_file_path, 
             self.feynrules_path, 
             self.feynarts_path,
             self.model_path, 
             self.model_name],  
            capture_output=True,
            text=True
            )
        print("NLO computations completed!")
        if self.verbose:
            print(nlo_process.stdout)
        return None
    
    def output_UFO(self) -> None:
        self.run_feynrules()
        txt_path = os.path.join(self.model_path, "feynrules_checks.txt")
        pass_fr_checks = True
        if not os.path.exists(txt_path):
            print("FeynRules checks not found!")
            return None
        else:
            print("FeynRules checks found!")
            with open(txt_path, "r") as file:
                lines = file.readlines()
                for line in lines[1:]:
                    if line.strip():
                        check, status = line.strip().split('\t') 
                        if "Fail" in status:
                            pass_fr_checks = False
                        print(f"{check}: {status}")

        if pass_fr_checks:
            print(f"{self.model_name} passed FeynRules checks!")
        else:
            print(f"{self.model_name} failed FeynRules checks!")
            return None
            
        self.run_feynarts()
        self.run_nloct()

    def run_madgraph(self) -> None:
        print(f"Running MadGraph5 computations for {self.model_name}...")
        process = subprocess.Popen(
            ["cd", self.madgraph5_path], 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        process.stdin.write("./bin/mg5_aMC\n")
        process.stdin.flush()
        print(process.stdout)
        return process

if __name__ == "__main__":

    FEYNRULES_PATH = "/users/qniu3/physics/FeynRules"
    FEYNARTS_PATH = "/users/qniu3/physics/FeynArts-3.12/Models" 
    NLOCT_PATH = "/users/qniu3/physics/FeynRules"
    MADGRAPH5_PATH = "/users/qniu3/physics/MG5_aMC_v3_6_3"
    MODEL_PATH = "/oscar/home/qniu3/physics/RL_model_builder/theories/SM_2025-06-26-17-00-45"
    MODEL_NAME = "SM"

    verifier = Verifier(
        feynrules_path = FEYNRULES_PATH,
        feynarts_path = FEYNARTS_PATH,
        nloct_path = NLOCT_PATH,
        madgraph5_path = MADGRAPH5_PATH,
        model_path = MODEL_PATH,
        model_name = MODEL_NAME,
        verbose = True
        )

    verifier.run_madgraph()
 