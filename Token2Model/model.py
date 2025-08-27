### ========================================================================== ###
###                                                                            ###
###                             Model Class                                    ###
###                                                                            ###
### ========================================================================== ###

# Standard library imports
import os
import json
from datetime import datetime
import random
import shutil


# ====================================================================
#                            Model
# ====================================================================
class Model:
    """
    Particle Physics Model Class: read JSON file and write FR model file
    """
    def __init__(self, model_name, author, JSON_PATH, OUTPUT_PATH, 
                 sm_particles_json = "sm_particles.json", 
                 sm_parameters_json = "sm_parameters.json",
                 simplify_checklist = True):
        self.model_name = model_name
        self.model_symbol = ''.join(word[0].upper() for word in self.model_name.split() if word)
        self.author = author
        self.ai = True if author == 'Bohr Network' else False
        self.FeynmanGauge = True
        self.JSON_PATH = JSON_PATH
        self.OUTPUT_PATH = OUTPUT_PATH
        self.current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.output_dir = os.path.join(OUTPUT_PATH, self.model_symbol)

        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

        os.makedirs(self.output_dir)

        self.sm_particles_json = os.path.join(os.path.dirname(__file__), sm_particles_json)
        self.sm_parameters_json = os.path.join(os.path.dirname(__file__), sm_parameters_json)
        self.simplify_checklist = simplify_checklist

        self.free_params = {}
        self.checklist = {}
        self._read_model()

    def __str__(self):
        return f"{self.model_name}"

    def __repr__(self):
        repr = "# ------------------------------------\n"
        repr += f"# {self.model_name}\n"
        repr += f"# by {self.author} at {self.current_time}\n"
        repr += f"# -----------------------------------\n\n"
        repr += f"# Gauge Groups ({len(self.gauge_groups)})\n"
        repr += f"{self.gauge_groups}\n\n"
        repr += f"# Particles ({len(self.particles)})\n"
        repr += f"{self.particles}\n\n"
        repr += f"# Fields ({len(self.fields)})\n"
        repr += f"{self.fields}\n\n"
        repr += f"# Interactions ({len(self.interactions)})\n"
        repr += f"{self.interactions}\n"
        return repr
    
    # ------------------------------------------------------------------
    #                           Read Model
    # ------------------------------------------------------------------
    def _read_gauge_group(self, model_data):
        from .group import GaugeGroup
        self.gauge_groups = {}
        for g in model_data['GaugeGroups']:
            self.gauge_groups[g["id"]] = GaugeGroup(**g)

    # def _read_vev(self, model_data):
    #     from .vev import vev
    #     self.vevs = {}
    #     for v in model_data['vevs']:
    #         self.vevs[v["id"]] = vev(**v)

    def _read_particles(self, model_data):
        from .particle import Fermion, RealScalar, ComplexScalar
        self.scalar_particles = {}
        self.fermion_particles = {}
        self.vector_particles = {}
        
        for p in model_data['particles']:

            # find free parameters
            if isinstance(p["mass"], list):
                free_param_name = f"M{p['name']}INPUT"
                self.free_params[free_param_name] = p["mass"]
                p["mass"] = random.uniform(p["mass"][0], p["mass"][1])

            if p["type"] == "fermion":
                p.pop("type")
                self.fermion_particles[p["id"]] = Fermion(**p, simplify_checklist = self.simplify_checklist)
            elif p["type"] == "real":
                p.pop("type")
                self.scalar_particles[p["id"]] = RealScalar(**p, simplify_checklist = self.simplify_checklist)
            elif p["type"] == "complex":
                p.pop("type")
                self.scalar_particles[p["id"]] = ComplexScalar(**p, simplify_checklist = self.simplify_checklist)
            else:
                type = p["type"]
                print(f"invalid field type {type}")

        self.particles = {**self.scalar_particles, **self.fermion_particles, **self.vector_particles}

    def _read_scalar_fields(self, model_data):
        from .field import ScalarField
        self.scalar_fields = {}
        scalar_particles = self.scalar_particles.copy()
        for sf in model_data["fields"]: # sf stands for "scalar field"
            if sf["type"] in ["real", "complex", "scalar"]:
                try:
                    scalar_list = [self.scalar_particles[id] for id in sf["particles"]]
                    for s in scalar_list:
                        scalar_particles.pop(s.id, None)
                except:
                    scalar_list = sf["particles"]

                sf.pop('chirality')
                sf["groups"] = self.gauge_groups
                sf["particles"] = scalar_list
                new_scalar_field = ScalarField(**sf, simplify_checklist = self.simplify_checklist)
                self.scalar_fields[sf["id"]] = new_scalar_field

    def _read_vector_fields(self, model_data):
        from .field import VectorField
        self.vector_fields = {}

    def _read_fermion_fields(self, model_data):
        from .field import FermionField
        self.fermion_fields = {}

        # we separate all fermion particles into left and right chiral fermions
        chiral_fermions = {**{f"{id}_left": ptcl.left for id, ptcl in self.fermion_particles.items()},
                           **{f"{id}_right": ptcl.right for id, ptcl in self.fermion_particles.items()}}
        
        for ff in model_data["fields"]:
            if ff["type"] == "fermion":
                try:
                    weyl_list = [chiral_fermions[f"{p}_left"] if ff["chirality"] == "left" else chiral_fermions[f"{p}_right"] for p in ff["particles"]]
                    for weyl_fermion in weyl_list:
                        chiral_fermions.pop(weyl_fermion.id, None)
                except:
                    weyl_list = ff["particles"]
        
                # TO-DO: add a check to see if the chiral fermion is in the chiral_fermions dictionary
                ff["groups"] = self.gauge_groups
                ff["particles"] = weyl_list
                ff.pop('type')
                new_fermion_field = FermionField(**ff, simplify_checklist = self.simplify_checklist)
                self.fermion_fields[ff["id"]] = new_fermion_field

        try:        
            self.weyl_spinors = {}
            for f in self.fermion_fields.values():
                self.weyl_spinors.update(f.phy_field_info)
        except:
            pass

        self.dirac_spinors = {}
        for key, value in self.weyl_spinors.items():
            new_key = f"F{key[:-1]}"
            if new_key not in self.dirac_spinors.keys():
                dirac_spinor_info = self.weyl_spinors[key].copy()
                dirac_spinor_info["left"] = 0
                dirac_spinor_info["right"] = 0
                dirac_spinor_info["LaTeX"] = dirac_spinor_info["LaTeX"][:-1]
            else:
                dirac_spinor_info = self.dirac_spinors[new_key]

            if value["chirality"] == "left":
                dirac_spinor_info["left"] = key
            elif value["chirality"] == "right":
                dirac_spinor_info["right"] = f"{key}"
            
            self.dirac_spinors[new_key] = dirac_spinor_info        
        
    def _read_fields(self, model_data):
        self._read_scalar_fields(model_data)
        self._read_fermion_fields(model_data)
        self._read_vector_fields(model_data)
        self.fields = {**self.scalar_fields, **self.fermion_fields, **self.vector_fields}

    def _read_interactions(self, model_data):
        from .interaction import Yukawa, ScalarSelfInteraction
        self.interactions = {}
        for itr in model_data["interactions"]:
            try:
                fields = [self.fields[id] for id in itr["fields"]]
            except:
                fields = itr["fields"]

            free_param_list = [k for k in itr.keys() if k not in ("id", "type", "fields")]
            for free_param in free_param_list:
                if isinstance(itr[free_param], list):
                    free_param_name = f"{free_param}{itr['id']}INPUT"
                    self.free_params[free_param_name] = itr[free_param]
                    itr[free_param] = random.uniform(itr[free_param][0], itr[free_param][1])
                
            if itr["type"] == "yukawa":
                itr.pop("type")
                itr["fields"] = fields
                new_interaction = Yukawa(**itr)
            elif itr["type"] == "ScalarSelfInteraction":
                itr.pop("type")
                itr["fields"] = fields
                new_interaction = ScalarSelfInteraction(**itr)
            else:
                print(f"invalid interaction type {itr['type']}")
                continue

            new_interaction.__validate__()
            self.interactions[itr["id"]] = new_interaction
        pass 

    def _read_parameters(self):
        self.ext_params = {}
        self.int_params = {}
        self.matching_conditions = ["v, vSM", "g1, g1SM", "g2, g2SM", "g3, g3SM"]
        self.parameters_to_solve_tadpoles = []
        self.EWSB_matter_sector = []
        self.lagNoHC = []
        self.lagHC = []
        for itr in self.interactions.values():
            self.ext_params.update(itr.ExtParams)
            self.int_params.update(itr.IntParams)
            self.matching_conditions.extend(itr.MatchingConditions)
            self.lagNoHC.extend(itr.LagNoHC)
            self.lagHC.extend(itr.LagHC)
            self.parameters_to_solve_tadpoles.extend(itr.ParametersToSolveTadpoles)
            if itr.EWSB_matter_sector is not None:
                self.EWSB_matter_sector.append(f"    {{{itr.EWSB_matter_sector}}}")
        self.parameters = {**self.ext_params, **self.int_params}

    def _read_check_list(self):
        model_score = 0
        max_score = 0
        model_components = [self.particles, self.fields, self.interactions]

        for component in model_components:
            for item in component.values():
                self.checklist[item.id] = item.checklist
                score, max_val = map(int, item.score.split("/"))
                model_score += score
                max_score += max_val

        self.score = f"{model_score}/{max_score}"

    def pass_all_checks(self):
        score, max_score = map(int, self.score.split("/"))
        return score == max_score

    def _read_model(self):
        from .utility import read_json
        model_data = read_json(self.JSON_PATH)

        self._read_gauge_group(model_data)
        self._read_particles(model_data)
        self._read_fields(model_data)
        self._read_interactions(model_data)
        [p.__validate__() for p in self.particles.values()]

        self._read_parameters()
        self._read_check_list()

    # ------------------------------------------------------------------
    #                           Write Model
    # ------------------------------------------------------------------
    def write_model(self):
        if not self.pass_all_checks():
            print(f"Error: {self.model_name} has failed the checks")
            return None

        model_file = os.path.join(self.output_dir, f"{self.model_symbol}.m")
        particle_file = os.path.join(self.output_dir, f"particles.m")
        parameter_file = os.path.join(self.output_dir, f"parameters.m")
        SPheno_file = os.path.join(self.output_dir, f"SPheno.m")

        from .write_model_files import write_particles, write_parameters, write_SPheno

        # Model.m
        with open(model_file, "w") as f:
            f.write("Off[General::spell];\n")
            f.write("\n")
            f.write(f"Model`Name = \"{self.model_symbol}\";\n")
            f.write(f"Model`NameLaTeX = \"{self.model_name}\";\n")
            f.write(f"Model`Authors = \"{self.author}\";\n")
            f.write(f"Model`Date = \"{self.current_time}\";\n")
            f.write("\n")
            f.write("(*-------------------------------------------*)\n")
            f.write("(*   Particle Content*)\n")
            f.write("(*-------------------------------------------*)\n")
            f.write("\n")
            f.write("(* Gauge Groups *)\n")
            f.write("Gauge[[1]]={B,   U[1], hypercharge, g1,False};\n")
            f.write("Gauge[[2]]={WB, SU[2], left,        g2,True};\n")
            f.write("Gauge[[3]]={G,  SU[3], color,       g3,False};\n")
            f.write("\n")
            f.write("(* Matter Fields *)\n")
            for idx, field in enumerate(self.fermion_fields.values()):
                if field.type == "fermion":
                    f.write(f"FermionFields[[{idx+1}]] = {{ {field.unphy_field_info} }};\n")
            f.write("\n")
            for idx, field in enumerate(self.scalar_fields.values()):
                if field.type == "complex" or field.type == "real":
                    f.write(f"ScalarFields[[{idx+1}]] = {{ {field.unphy_field_info} }};\n")
   
            f.write("\n")
            f.write("(*----------------------------------------------*)\n")
            f.write("(*   DEFINITION                                 *)\n")
            f.write("(*----------------------------------------------*)\n")
            f.write("NameOfStates={GaugeES, EWSB};\n")
            f.write("\n")
            f.write("(* ----- Before EWSB ----- *)\n")
            f.write("\n")
            f.write("DEFINITION[GaugeES][LagrangianInput]= \n")
            f.write("{\n")
            f.write("   {LagHC, {AddHC->True}},\n")
            f.write("   {LagNoHC,{AddHC->False}}\n")
            f.write("};\n")
            f.write("\n")
            f.write(f"LagHC = {' '.join(self.lagHC)};\n")
            f.write(f"LagNoHC = {' '.join(self.lagNoHC)};\n")

            f.write("\n")
            f.write("(* Gauge Sector *)\n")
            f.write("\n")
            f.write("DEFINITION[EWSB][GaugeSector] = \n")
            f.write("{")
            f.write("   {{VB,VWB[3]},{VP,VZ},ZZ},\n")
            f.write("   {{VWB[1],VWB[2]},{VWp,conj[VWp]},ZW}\n")
            f.write("};\n")
            f.write("\n")
            f.write("(* ----- VEVs ---- *)\n")
            f.write("\n")
            f.write("DEFINITION[EWSB][VEVs]= \n")
            f.write("{\n")
            f.write("    {H0, {v, 1/Sqrt[2]}, {Ah, \[ImaginaryI]/Sqrt[2]},{hh, 1/Sqrt[2]}}\n")
            f.write("};\n")

            f.write("\n")
            f.write("DEFINITION[EWSB][MatterSector]= {\n")
            f.write(',\n'.join(self.EWSB_matter_sector))
            f.write("\n};\n\n")

            f.write("(*------------------------------------------------------*)\n")
            f.write("(* Dirac-Spinors *)\n")
            f.write("(*------------------------------------------------------*)\n")
            f.write("\n")
            f.write("DEFINITION[EWSB][DiracSpinors]= {\n")
            DiracSpinors = []
            for key, value in self.dirac_spinors.items(): 
                if value["left"] != 0 and value["right"] != 0:
                    DiracSpinors.append(f"    {key} -> {{{value['left'].upper()}, conj[{value['right'].upper()}]}}")
                elif value["left"] != 0 and value["right"] == 0:
                    DiracSpinors.append(f"    {key} -> {{{value['left']}, 0}}")
                elif value["left"] == 0 and value["right"] != 0:
                    DiracSpinors.append(f"    {key} -> {{0, conj[{value['right']}]}}")
            f.write(',\n'.join(DiracSpinors))
            f.write("\n};\n")
            f.write("\n")

            f.write("DEFINITION[EWSB][GaugeES]= {\n")
            WeylSpinors = []
            for key, value in self.dirac_spinors.items():
                if value["left"] != 0 and value["right"] != 0:
                    WeylSpinors.append(f"    {key}1 -> {{{key}L, 0}}")
                    WeylSpinors.append(f"    {key}2 -> {{0, {key}R}}")
            WeylSpinors.sort()
            f.write(',\n'.join(WeylSpinors))
            f.write("\n};\n")
            f.write("\n")

        # --- parameters.m ---
        with open(self.sm_parameters_json, "r") as f:
            all_parameters = json.load(f)

        parameters = {key: value.__dict__() for key, value in self.parameters.items()}
        all_parameters.update(parameters)
        write_parameters(all_parameters, parameter_file)

        # --- particles.m ---
        with open(self.sm_particles_json, "r") as f:
            all_particles = json.load(f)

        all_particles["ParticleDefinitions[EWSB]"].update(self.dirac_spinors)

        weyl_spinors_latex = {key: {"LaTeX": value["LaTeX"]} for key, value in self.weyl_spinors.items()}
        weyl_spinors_EWSB_latex = {key.upper(): {"LaTeX": value["LaTeX"].upper()} for key, value in self.weyl_spinors.items()}
        all_particles["WeylFermionAndIndermediate"].update(weyl_spinors_latex)
        all_particles["WeylFermionAndIndermediate"].update(weyl_spinors_EWSB_latex)
        write_particles(all_particles, particle_file)
            
        # --- SPheno.m ---
        self.list_decay_particles = ["hh"]
        self.list_decay_particles.extend(list(self.dirac_spinors.keys()))
        write_SPheno(self.matching_conditions, 
                     self.parameters_to_solve_tadpoles, 
                     self.ext_params,
                     SPheno_file,
                     self.list_decay_particles)
        
        # --- write free parameters ---
        with open(os.path.join(self.output_dir, "free_params.json"), "w") as f:
            json.dump(self.free_params, f)


    def write_checklist(self):
        with open(os.path.join(self.output_dir, "checklist.csv"), "w") as f:
            f.write("id, check, score, error_var, message\n")
            for id, checklist in self.checklist.items():
                for key, value in checklist.items():
                    f.write(f"{id}, {key}, {value['score']}, {value['error_var']}, {value['message']}\n")


    