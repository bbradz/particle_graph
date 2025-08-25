import subprocess
import os
import shutil
from typing import Dict, List
import datetime
import json
import multiprocessing

import numpy as np
from scipy import stats
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

class ObservableCalc:
    """
    Python wrapper to run SARAH/SPheno to compute observables.
    """
    def __init__(self, 
                 model_name,
                 model_base= "./Models",
                 sarah_path= "../SARAH-4.15.4", 
                 spheno_path= "../SPheno-4.0.5",
                 obv_list_path = None,
                 timeout = 1,
                 keep_log = True,
                 loop_mass = True,
                 include_tachyon = False,
                 calc_decays = False
                 ):
        self.MODEL_NAME = model_name
        self.MODEL_BASE = os.path.abspath(model_base)
        self.MODEL_PATH = os.path.join(model_base, model_name)
        self.SARAH_PATH = os.path.abspath(sarah_path)
        self.SPHENO_PATH = os.path.abspath(spheno_path)
        self.OBV_LIST_PATH = os.path.abspath(obv_list_path)
        self.free_params_path = os.path.join(self.MODEL_PATH, "free_params.json")
        self.calc_spheno_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calc_spheno.m")
        self.input_path = os.path.join(self.MODEL_PATH, "EWSB", "SPheno", "Input_Files", f"LesHouches.in.{self.MODEL_NAME}")
        self.output_path = os.path.join(self.MODEL_PATH, "Results")
        os.makedirs(self.output_path, exist_ok = True)

        self.N_CPU_CORES = multiprocessing.cpu_count()

        self.pre_check()
        self.timeout = timeout
        self.keep_log = keep_log
        self.loop_mass = loop_mass
        self.include_tachyon = include_tachyon
        self.calc_decays = calc_decays

    def pre_check(self):
        # Check if SPheno.m, parameters.m, and particles.m exist in the model path
        exist_spheno_m = os.path.isfile(os.path.join(self.MODEL_PATH, "SPheno.m"))
        exist_parameters_m = os.path.isfile(os.path.join(self.MODEL_PATH, "parameters.m"))
        exist_particles_m = os.path.isfile(os.path.join(self.MODEL_PATH, "particles.m"))
        exist_model_m = os.path.isfile(os.path.join(self.MODEL_PATH, f"{self.MODEL_NAME}.m"))

        self.pass_pre_check = True
        if not exist_spheno_m:
            print("SPheno.m not found.")
            self.pass_pre_check = False
        if not exist_parameters_m:
            print("parameters.m not found.")
            self.pass_pre_check = False
        if not exist_particles_m:
            print("particles.m not found.")
            self.pass_pre_check = False
        if not exist_model_m:
            print(f"{self.MODEL_NAME}.m not found.")
            self.pass_pre_check = False

        try:
            with open(self.OBV_LIST_PATH, "r") as f:
                self.obv_list = json.load(f)
        except:
            self.obv_list = {}
            print("No observable list file found.")
            self.pass_pre_check = False
        
        try:
            with open(self.free_params_path, "r") as f:
                self.free_params = json.load(f)
            self.num_params = len(self.free_params)
        except:
            self.free_params = {}
            print("No parameter range file found.")
            self.pass_pre_check = False

    def run_sarah(self):
        if not self.pass_pre_check:
            print("Error: Model has failed the pre-check.")
            return None
        
        # Use absolute path for the math script 
        subprocess.run(["math", "-script", self.calc_spheno_path, self.SARAH_PATH, self.MODEL_BASE, self.MODEL_NAME])
    
    def compile_spheno(self, n_cpu = 4):
        if not self.pass_pre_check:
            print("Error: Model has failed the pre-check.")
            return None
        
        # Copy the SPheno files to the SPheno directory
        src_dir = os.path.join(self.MODEL_BASE, self.MODEL_NAME, "EWSB", "SPheno")
        dst_dir = os.path.join(self.SPHENO_PATH, self.MODEL_NAME)
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)
        
        # Store current directory and change to spheno path
        original_dir = os.getcwd()
        os.chdir(self.SPHENO_PATH)
        subprocess.run(["make", "clean"], shell=True)
        subprocess.run(f"make F90=gfortran Model={self.MODEL_NAME}", shell=True)
        # Return to original directory
        os.chdir(original_dir)

    def write_lha(self, 
                  G_F = 1.166370E-05, 
                  alpha_s = 1.187000E-01, 
                  m_Z = 9.118870E+01, 
                  m_b = 4.180000E+00, 
                  m_t = 1.728900E+02, 
                  m_tau = 1.776690E+00,
                  **kwargs
                  ):
        if not self.pass_pre_check:
            print("Error: Model has failed the pre-check.")
            return None
        
        """write the LHA file"""
        sminputs = {2: G_F, 3: alpha_s, 4: m_Z, 5: m_b, 6: m_t, 7: m_tau}
        minpar = {}
        for i, (key, value) in enumerate(kwargs.items()):
            minpar[i+1] = value

        with open(self.input_path, 'r') as f:
            lines = f.readlines()
        current_block = None
        modified_file = []
        for line in lines:
            if line.startswith("Block"):
                current_block = line.split()[1]
                modified_file.append(line)
                continue

            if current_block == "SMINPUTS":
                key, _ = line.split("#")[0].split()
                comment = line.split("#")[1]
                line = f" {key} {sminputs[int(key)]}  #{comment}"
            elif current_block == "MINPAR":
                key, _ = line.split("#")[0].split()
                comment = line.split("#")[1]
                line = f" {key} {minpar[int(key)]}  #{comment}"
            elif current_block == "SPhenoInput":
                key, _ = line.split("#")[0].split()
                comment = line.split("#")[1]
                if key == "52":
                    line = f" {key} {1 if self.include_tachyon else 0}  #{comment}"
                elif key == "55":
                    line = f" {key} {1 if self.loop_mass else 0}  #{comment}"
                else:
                    pass
            elif current_block == "DECAYOPTIONS":
                key, _ = line.split("#")[0].split()
                comment = line.split("#")[1]
                line = f" {key} {1 if self.calc_decays else 0}  #{comment}"
            modified_file.append(line)

        with open(f"{self.input_path}", 'w') as f:
            f.writelines(modified_file)

    def run_spheno(self):
        if not self.pass_pre_check:
            print("Error: Model has failed the pre-check.")
            return None

        # Use absolute paths for both the executable and input file
        timestamp = datetime.datetime.now().strftime("%m%d.%H%M%S.%f")
        self.spheno_exe = os.path.join(self.SPHENO_PATH, "bin", f"SPheno{self.MODEL_NAME}")
        self.spheno_out = os.path.join(self.output_path, f"{self.MODEL_NAME}.spc.{timestamp}")

        try:
            subprocess.run([self.spheno_exe, self.input_path, self.spheno_out], timeout = self.timeout)
            subprocess.run("rm *.dat *.out", shell=True)
        except subprocess.TimeoutExpired:
            print("SPheno timed out")
            return False
        return True

    def parse_spc(self):
        if not self.pass_pre_check:
            print("Error: Model has failed the pre-check.")
            return None, None, None
        
        blocks: Dict[str, Dict[str, float]] = {}
        decays: Dict[str, Dict[str, float]] = {}
        decays1l: Dict[str, Dict[str, float]] = {}

        if not os.path.exists(self.spheno_out):
            return None, None, None
        
        with open(self.spheno_out, 'r') as f:
            is_block = False
            is_decay = False
            key = None
            for line in f:
                if line.startswith("#"):
                    continue
    
                if line.startswith("Block "):
                    is_block, is_decay = True, False
                    key = line.split()[1]
                    blocks[key] = {}

                    if "Q=" in line:
                        scale = float(line.split("Q=")[1].split()[0])
                        blocks[key]["scale"] = scale
                    continue 

                elif line.startswith("DECAY"):
                    is_decay, is_block = True, False
                    is_1loop = True if "1L" in line else False
                    key = line.split()[1]
                    if is_1loop:
                        decays1l[key] = {}
                    else:
                        decays[key] = {}
                    continue 

                else:
                    if '#' in line:
                        line = line.split('#')[0]
                    
                    if is_block:
                        if len(line.split()) == 2:
                            blocks[key][line.split()[0]] = line.split()[1]
                        else: # we ignore the matrix blocks and other blocks that are not relevant to us
                            pass
                        
                    elif is_decay:
                        NDA = int(line.split()[1])
                        final_states = {f"ID{i+1}": line.split()[i+2] for i in range(NDA)}
                        if is_1loop:
                            decays1l[key] = {"BR": line.split()[0], "NDA": NDA, **final_states}
                        else:
                            decays[key] = {"BR": line.split()[0], "NDA": NDA, **final_states}

        return blocks, decays, decays1l
    

    def compute_obv(self, args):
        assert len(args) == self.num_params, "Number of parameters does not match"
        input_param = {}
        for i, (key, value) in enumerate(self.free_params.items()):
            if args[i] < value[0] or args[i] > value[1]:
                print(f"Parameter {key} out of range: {args[i]}")
                return None
            input_param[key] = args[i]
        self.write_lha(**input_param)
        self.run_spheno()
        blocks, decays, decays1l = self.parse_spc()
        if blocks is None:
            print("SPheno failed.")
            return None

        if not self.keep_log:
            subprocess.run(f"rm {self.spheno_out}", shell=True)
        
        if self.obv_list is None:
            print("No observable list provided. Return all.")
            return blocks, decays, decays1l

        obvs = {} 
        for obv, loc in self.obv_list.items():
            if loc["LHA_loc"][0] == "Block":
                obvs[obv] = blocks[loc["LHA_loc"][1]][loc["LHA_loc"][2]]
            elif loc["LHA_loc"][0] == "DECAY":
                obvs[obv] = decays[loc["LHA_loc"][1]][loc["LHA_loc"][2]]
            elif loc["LHA_loc"][0] == "DECAY1L":
                obvs[obv] = decays1l[loc["LHA_loc"][1]][loc["LHA_loc"][2]]
        return obvs
    


    # ------------------------------------------------------------
    # Parameter Scanning
    # ------------------------------------------------------------
    def chi2(self, params):
        """chi-squared function"""
        predicted_value = self.compute_obv(params)
        if predicted_value is None:
            return 1e4
        chi2 = 0
        sub_chi2 = []
        for obv, measured_value in self.obv_list.items():
            sub_chi2.append((float(predicted_value[obv]) - float(measured_value["measured"]))**2 / float(measured_value["sigma"])**2)
        chi2 = sum(sub_chi2)
        return chi2, sub_chi2

    def log_likelihood(self, params):
        """log-likelihood function"""
        return -0.5 * self.chi2(params)
    
    def log_prior(self, params):
        """log-prior function"""
        log_prior = 0
        for i, (key, value) in enumerate(self.free_params.items()):
            if params[i] < value[0] or params[i] > value[1]:
                return -1e4
        return log_prior
    
    def log_prob(self, params):
        """log-probability function"""
        return self.log_prior(params) + self.log_likelihood(params)
    
    def confidence_level(self, n_sigma = 3):
        """confidence level of the chi-squared distribution"""
        return stats.norm.cdf(n_sigma) - stats.norm.cdf(-n_sigma)

    def chi2_threshold(self, n_dof, n_sigma = 3):
        """threshold of the chi-squared distribution"""
        return stats.chi2.ppf(self.confidence_level(n_sigma), n_dof)

    def generate_initial_conditions(self, n_minimizers, method = "latin_hypercube"):
        """generate initial conditions for the minimizers"""
        samples = []
        if method == "uniform":
            samples = np.random.uniform(0, 1, size = (n_minimizers, self.num_params))
        elif method == "latin_hypercube":
            sampler = stats.qmc.LatinHypercube(d=self.num_params)
            samples = sampler.random(n=n_minimizers)
        else:
            raise ValueError(f"Invalid method: {method}")
        
        for index, (param_name, ranges) in enumerate(self.free_params.items()):
            low, high = ranges
            samples[:, index] = low + samples[:, index] * (high - low)
        return samples
    
    def minimize_chi2(self, maxiter = 10, seed = 42):
        """minimize the chi-squared function"""

        self.keep_log = False

        chi2_history = []
        sub_chi2_history = []
        params_history = []

        def tracked_chi2(params):
            chi2, sub_chi2 = self.chi2(params)
            chi2_history.append(chi2)
            sub_chi2_history.append(sub_chi2)
            params_history.append(params)
            return chi2
  
        bounds = list(map(tuple, self.free_params.values()))
        result = differential_evolution(tracked_chi2, 
                                        popsize = 5,
                                        bounds = bounds, 
                                        maxiter = maxiter,
                                        seed = seed
                                        )

        chi2_history = np.array(chi2_history)
        params_history = np.array(params_history)
        sub_chi2_history = np.array(sub_chi2_history)
        np.savez(os.path.join(self.output_path, "chi2_data.npz"), 
                 chi2_history = chi2_history, 
                 sub_chi2_history = sub_chi2_history,
                 params_history = params_history)
        print(f"Minimized chi2: {result.fun}")
        print(f"Minimized parameters: {result.x}")
        return result

    def make_plot(self):
        chi2_data = np.load(os.path.join(self.output_path, "chi2_data.npz"))
        chi2_history = chi2_data["chi2_history"]
        params_history = chi2_data["params_history"]
        for i in range(self.num_params):
            plt.figure()
            plt.scatter(params_history[:, i], chi2_history, label=list(self.free_params.keys())[i])
            plt.xlabel(list(self.free_params.keys())[i])
            plt.ylabel("chi2")
            plt.ylim(0, 100)
            plt.legend()
            plt.savefig(os.path.join(self.output_path, f"chi2_plot_{list(self.free_params.keys())[i]}.png"))
            plt.close()

    # ------------------------------------------------------------
    # Clean cache
    # ------------------------------------------------------------
    def clean_cache(self):
        pass
