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
                 obs_list_path = None,
                 timeout = 1,
                 sigma_threshold = 3,
                 keep_log = True,
                 loop_mass = True,
                 include_tachyon = True,
                 calc_decays = False,
                 mass_precision = 1e-6,
                 three_body_decays = False,
                 higgs_bounds = False,
                 eft_higgs_coupling = False,
                 diphoton_width = False
                 ):
        self.MODEL_NAME = model_name
        self.MODEL_BASE = os.path.abspath(model_base)
        self.MODEL_PATH = os.path.join(model_base, model_name)
        self.SARAH_PATH = os.path.abspath(sarah_path)
        self.SPHENO_PATH = os.path.abspath(spheno_path)
        self.OBS_LIST_PATH = os.path.abspath(obs_list_path)
        self.free_params_path = os.path.join(self.MODEL_PATH, "free_params.json")
        self.calc_spheno_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calc_spheno.m")
        self.input_path = os.path.join(self.MODEL_PATH, "EWSB", "SPheno", "Input_Files", f"LesHouches.in.{self.MODEL_NAME}")
        self.output_path = os.path.join(self.MODEL_PATH, "Results")
        os.makedirs(self.output_path, exist_ok = True)
        

        self.N_CPU_CORES = multiprocessing.cpu_count()
        self.sigma_threshold = sigma_threshold

        self.pre_check()
        self.timeout = timeout
        self.keep_log = keep_log
        self.loop_mass = loop_mass
        self.include_tachyon = include_tachyon
        self.calc_decays = calc_decays
        self.mass_precision = mass_precision
        self.three_body_decays = three_body_decays
        self.higgs_bounds = higgs_bounds
        self.eft_higgs_coupling = eft_higgs_coupling
        self.diphoton_width = diphoton_width

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
            with open(self.OBS_LIST_PATH, "r") as f:
                self.obs_list = json.load(f)
        except:
            self.obs_list = {}
            print("No observable list file found.")
            self.pass_pre_check = False
        
        try:
            with open(self.free_params_path, "r") as f:
                self.free_params = json.load(f)
            self.free_param_keys = list(self.free_params.keys())
            self.num_params = len(self.free_params)
        except:
            self.free_params = {}
            self.free_param_keys = []
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
                elif key == "34":
                    line = f" {key} {self.mass_precision}  #{comment}"
                elif key == "13":
                    line = f" {key} {1 if self.three_body_decays else 0}  #{comment}"
                elif key == "76":
                    line = f" {key} {2 if self.higgs_bounds else 0}  #{comment}"
                elif key == "520":
                    line = f" {key} {1 if self.eft_higgs_coupling else 0}  #{comment}"
                elif key == "521":
                    line = f" {key} {1 if self.diphoton_width else 0}  #{comment}"
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
    

    def compute_obs(self, args):
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
        
        if self.obs_list is None:
            print("No observable list provided. Return all.")
            return blocks, decays, decays1l

        obss = {} 
        for obs, loc in self.obs_list.items():
            if loc["LHA_loc"][0] == "Block":
                obss[obs] = blocks[loc["LHA_loc"][1]][loc["LHA_loc"][2]]
            elif loc["LHA_loc"][0] == "DECAY":
                obss[obs] = decays[loc["LHA_loc"][1]][loc["LHA_loc"][2]]
            elif loc["LHA_loc"][0] == "DECAY1L":
                obss[obs] = decays1l[loc["LHA_loc"][1]][loc["LHA_loc"][2]]
        return obss
    


    # ------------------------------------------------------------
    # Parameter Scanning
    # ------------------------------------------------------------
    def chi2(self, params):
        """chi-squared function"""

        all_obs = list(self.obs_list.keys())
        chi2_dict = {obs: 1e4 for obs in all_obs}
        chi2_dict["total"] = sum(chi2_dict.values())

        predicted_value = self.compute_obs(params)
        if predicted_value is None:
            return chi2_dict
        
        chi2_dict["total"] = 0
        for obs, measured_value in self.obs_list.items():
            chi2_dict[obs] = (float(predicted_value[obs]) - float(measured_value["measured"]))**2 / float(measured_value["sigma"])**2
        chi2_dict["total"] = sum(chi2_dict.values())
        return chi2_dict

    def log_likelihood(self, params):
        """log-likelihood function"""
        chi2_dict = self.chi2(params)
        return -0.5 * chi2_dict["total"]
    
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
    
    def confidence_level(self, n_sigma):
        """confidence level of the chi-squared distribution"""
        return stats.norm.cdf(n_sigma) - stats.norm.cdf(-n_sigma)

    def chi2_threshold(self):
        """threshold of the chi-squared distribution"""
        df = self.num_params
        return stats.chi2.ppf(self.confidence_level(self.sigma_threshold), df)

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
    
    def minimize_chi2(self, maxiter = 10, popsize = 5, seed = 42):
        """minimize the chi-squared function"""

        self.keep_log = False
        chi2_history = []
        params_history = []

        class EarlyStopException(Exception):
            pass
        
        def tracked_chi2(params):
            chi2_dict = self.chi2(params)
            chi2_history.append(chi2_dict)
            params_history.append(params)
            print(f"tracked_chi2 called {len(chi2_history)} times, chi2 = {chi2_dict['total']}")

            # Early stopping condition
            if chi2_dict["total"] < self.chi2_threshold():
                raise EarlyStopException("chi2 below threshold")

            return chi2_dict["total"]
        
        bounds = list(map(tuple, self.free_params.values()))
        try:
            result = differential_evolution(tracked_chi2, 
                                            popsize = popsize,
                                            bounds = bounds, 
                                            maxiter = maxiter,
                                            seed = seed
                                            )
            print(f"Minimized chi2: {result.fun}")
            print(f"Minimized parameters: {result.x}")
        except EarlyStopException as e:
            result = {"fun": chi2_history[-1]["total"], "x": params_history[-1]}
            print(f"Early stop: {e}")
            print(f"Minimized chi2: {result['fun']}")
            print(f"Minimized parameters: {result['x']}")

        # Store all chi2 values (total and individual observables)
        chi2_total = np.array([d["total"] for d in chi2_history])
        
        # Get all observable names from the first chi2_dict
        if chi2_history:
            obs_names = [key for key in chi2_history[0].keys() if key != "total"]
            chi2_obs = {}
            for obs_name in obs_names:
                chi2_obs[obs_name] = np.array([d[obs_name] for d in chi2_history])

        # Convert params_history to numpy arrays for each parameter
        params_history_arrays = {}
        for i, key in enumerate(self.free_param_keys):
            params_history_arrays[key] = np.array([params[i] for params in params_history])

        # Save all data
        save_dict = {
            "chi2_total": chi2_total,
            **chi2_obs,
            **params_history_arrays
        }
        np.savez(os.path.join(self.output_path, "chi2_data.npz"), **save_dict)

        self.chi2_result = result

    def make_plot(self):
        chi2_data = np.load(os.path.join(self.output_path, "chi2_data.npz"))
        chi2_total = chi2_data["chi2_total"]
        
        # Get the parameter arrays from the NPZ file
        params_history = {}
        for param_name in self.free_param_keys:
            params_history[param_name] = chi2_data[param_name]

        plt.figure()
        for obs_name in self.obs_list.keys():
            plt.plot(range(len(chi2_total)), chi2_data[obs_name], marker='o', label=obs_name)
        plt.plot(range(len(chi2_total)), chi2_total, marker='o', label="total")
        plt.xlabel("Iteration")
        plt.ylabel("chi2")
        plt.title("chi2 vs Iteration")
        plt.ylim(0.01, 1000)
        plt.yscale("log")
        plt.legend()
        plt.savefig(os.path.join(self.output_path, "chi2_history.png"))
        plt.close()

        for param_name in self.free_param_keys:
            plt.figure()
            for obs_name in self.obs_list.keys():
                plt.scatter(params_history[param_name], chi2_data[obs_name], label=obs_name)
            plt.scatter(params_history[param_name], chi2_total, label="total")
            plt.xlabel(param_name)
            plt.ylabel("chi2")
            plt.ylim(0.01, 1000)
            plt.yscale("log")
            plt.legend()
            plt.savefig(os.path.join(self.output_path, f"chi2_plot_{param_name}.png"))
            plt.close()

    # ------------------------------------------------------------
    # Clean cache
    # ------------------------------------------------------------
    def clean_cache(self):
        pass
