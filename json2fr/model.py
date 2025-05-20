### ========================================================================== ###
###                                                                            ###
###                             Model Class                                    ###
###                                                                            ###
### ========================================================================== ###

from datetime import datetime
import os
from fractions import Fraction
from index import Index
import name
# ====================================================================
#                            Model
# ====================================================================
class Model:
    """
    Particle Physics Model Class: read JSON file and write FR model file
    """
    def __init__(self, model_name, author, JSON_PATH, OUTPUT_PATH = None):
        self.model_name = model_name
        self.author = author
        self.ai = True if author == 'Bohr Network' else False
        self.FeynmanGauge = True
        
        self.JSON_PATH = JSON_PATH
        self.OUTPUT_PATH = OUTPUT_PATH
        self.current_time = datetime.now().strftime("%Y-%m-%d")
        self.checklist = {}
        self.indices = {}
        self._read_model()

    # ------------------------------------------------------------------
    #                           Read Model
    # ------------------------------------------------------------------
    def _read_gauge_group(self, model_data):
        from group import GaugeGroup
        self.gauge_groups = {}
        for g in model_data['nodes']['GaugeGroups']:
            self.gauge_groups[g["id"]] = GaugeGroup(**g)

    def _read_vev(self, model_data):
        from vev import vev
        self.vevs = {}
        for v in model_data["nodes"]['vevs']:
            self.vevs[v["id"]] = vev(**v)

    def _read_particles(self, model_data):
        from particle import Fermion, RealScalar, ComplexScalar
        self.scalar_particles = {}
        self.fermion_particles = {}
        self.vector_particles = {}
        
        for p in model_data['nodes']['particles']:
            if p["type"] == "fermion":
                p.pop("type")
                self.fermion_particles[p["id"]] = Fermion(**p)
            elif p["type"] == "real":
                p.pop("type")
                self.scalar_particles[p["id"]] = RealScalar(**p)
            elif p["type"] == "complex":
                p.pop("type")
                self.scalar_particles[p["id"]] = ComplexScalar(**p)
            else:
                type = p["type"]
                print(f"invalid field type {type}")
    
    def _read_scalar_fields(self, model_data):
        from field import ScalarField
        self.scalar_fields = {}
        scalar_particles = self.scalar_particles.copy()
        for sf in model_data["nodes"]["fields"]: # sf stands for "scalar field"
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
                new_scalar_field = ScalarField(**sf)
                new_scalar_field.__validate__()
                self.scalar_fields[sf["id"]] = new_scalar_field
                for idx in new_scalar_field.indices:
                    if idx.name not in self.indices.keys():
                        self.indices[idx.name] = idx

    def _read_vector_fields(self, model_data):
        from field import VectorField
        self.vector_fields = {}

        self.indices["SU2W"] = Index("SU2W", 3, "Unfold")
        self.indices["Gluon"] = Index("Gluon", 8, "NoUnfold", color=True)
        
    def _read_fermion_fields(self, model_data):
        from field import FermionField
        self.fermion_fields = {}
        # we separate all fermion particles into left and right chiral fermions
        chiral_fermions = {**{f"{id}_left": ptcl.left for id, ptcl in self.fermion_particles.items()},
                          **{f"{id}_right": ptcl.right for id, ptcl in self.fermion_particles.items()}}
        
        for f in model_data["nodes"]["fields"]:
            if f["type"] == "fermion":
                try:
                    cf_list = [chiral_fermions[f"{p}_left"] if f["chirality"] == "left" else chiral_fermions[f"{p}_right"] for p in f["particles"]]
                    for cf in cf_list:
                        chiral_fermions.pop(cf.id, None)
                except:
                    cf_list = f["particles"]
        
                # TO-DO: add a check to see if the chiral fermion is in the chiral_fermions dictionary
                f["groups"] = self.gauge_groups
                f["particles"] = cf_list
                f.pop('type')
                new_fermion_field = FermionField(**f)
                new_fermion_field.__validate__()
                self.fermion_fields[f["id"]] = new_fermion_field
                for idx in new_fermion_field.indices:
                    if idx.name not in self.indices.keys():
                        self.indices[idx.name] = idx

        try:        
            self.phy_fermion = []
            self.unphy_fermion = []
            for f in self.fermion_fields.values():
                self.phy_fermion.extend(f.phy_field_info)
                self.unphy_fermion.append(f.unphy_field_info)
        except:
            pass

        seen = []
        unique_fermion = []
        for f in self.phy_fermion:
            if f["PDG"] not in seen:
                seen.append(f["PDG"])
                unique_fermion.append(f)
        self.phy_fermion = unique_fermion

    def _read_fields(self, model_data):
        self._read_scalar_fields(model_data)
        self._read_fermion_fields(model_data)
        self._read_vector_fields(model_data)
        self.fields = {**self.scalar_fields, **self.fermion_fields, **self.vector_fields}

    def _read_interactions(self, model_data):
        from interaction import Yukawa
        self.interactions = {}
        for itr in model_data["nodes"]["interactions"]:
            try:
                fields = [self.fields[id] for id in itr["fields"]]
            except:
                fields = itr["fields"]
            itr.pop("type")
            itr["fields"] = fields
            new_interaction = Yukawa(**itr)
            new_interaction.__validate__()
            self.interactions[itr["id"]] = new_interaction

    def _read_check_list(self):
        model_score = 0
        max_score = 0
        model_components = [self.gauge_groups, self.fields, self.interactions]

        for component in model_components:
            for item in component.values():
                self.checklist[item.id] = item.checklist
                score, max_val = map(int, item.score.split("/"))
                model_score += score
                max_score += max_val
        self.score = f"{Fraction(model_score, max_score)}"

    def pass_all_checks(self):
        return self.score == "1"

    def _read_model(self):
        from utility import read_json
        model_data = read_json(self.JSON_PATH)
        self._read_gauge_group(model_data)
        self._read_particles(model_data)
        self._read_fields(model_data)
        self._read_interactions(model_data)
        self._read_check_list()


    # ------------------------------------------------------------------
    #                        Write FeynRules file
    # ------------------------------------------------------------------
    def write_info(self, file):
        file.write(f"(* ************************** *)\n")
        file.write(f"(* *****  Information   ***** *)\n")
        file.write(f"(* ************************** *)\n")
        file.write(f"M$ModelName = \"{self.model_name}\";\n")
        file.write("\n")
        file.write(f"M$Information = {{ \n")
        file.write(f"  Authors      -> {self.author}, \n")
        file.write(f"  Date         -> \"{self.current_time}\" \n")
        file.write(f"}};\n")
        file.write("\n")

    def write_gauge(self, file):
        file.write(f"FeynmanGauge = {self.FeynmanGauge};\n")
        file.write("\n")    
    
    def write_vevs(self, file):
        file.write(f"(* ************************** *)\n")
        file.write(f"(* *****      vevs      ***** *)\n")
        file.write(f"(* ************************** *)\n")
        file.write("M$vevs = { {Phi[2],vev} };\n\n")

    def write_gauge_group(self, file):
        file.write(f"(* ************************** *)\n")
        file.write(f"(* *****  Gauge groups  ***** *)\n")
        file.write(f"(* ************************** *)\n")
        file.write(f"M$GaugeGroups = {{\n")
        gauge_group_entries = []
        for _, group in self.gauge_groups.items():
            group_entry = f"  {group.name} == {{\n"
            group_info = []
            for key, value in group.gauge_group_info().items():
                if value is not None:
                    group_info.append(f"{key:20} -> {value}")
            group_entry += "    " + ",\n    ".join(group_info) + "\n"
            group_entry += "  }"
            gauge_group_entries.append(group_entry)
        file.write(",\n".join(gauge_group_entries) + "\n")
        file.write(f"}};\n\n")

    def write_FeynArts(self, file):
        from sm_setting import write_sm_gauge_parameters
        write_sm_gauge_parameters(file)
        
    def write_Indices(self, file):
        file.write("(* ************************** *)\n")
        file.write("(* *****    Indices     ***** *)\n")
        file.write("(* ************************** *)\n\n")
        Indices = self.indices
        for _, idx in Indices.items():
            file.write(f"{idx.IndexRange()};\n")
        file.write("\n")
        for _, idx in Indices.items():
            style = name.generate_ijk()
            file.write(f"{idx.IndexStyle(style)};\n")
        file.write("\n")

    def write_Interaction_orders(self, file):
        from sm_setting import write_sm_interaction_orders
        write_sm_interaction_orders(file)
        
    def write_fermion(self, file):
        file.write("(* Fermions: physical fields *)\n")
        for idx, field in enumerate(self.phy_fermion):
            fermion_entry = f"  F[{idx+1}] == {{\n"
            fermion_info = []
            for key, value in field.items():
                fermion_info.append(f"{key:20} -> {value}")
            fermion_entry += "    " + ",\n    ".join(fermion_info) + "\n"
            fermion_entry += "  },\n"
            file.write(fermion_entry)
    
        file.write("\n")
        file.write("(* Fermions: unphysical fields *)\n")
        for idx, field in enumerate(self.unphy_fermion):
            fermion_entry = f"  F[1{idx+1}] == {{\n"
            fermion_info = []
            for key, value in field.items():
                fermion_info.append(f"{key:20} -> {value}")
            fermion_entry += "    " + ",\n    ".join(fermion_info) + "\n"
            fermion_entry += "  },\n"
            file.write(fermion_entry)

    def write_scalar(self, file):
        from sm_setting import write_sm_higgs
        write_sm_higgs(file)

    def write_vector(self, file):
        from sm_setting import write_sm_gauge_boson
        write_sm_gauge_boson(file)

    def write_fields(self, file):
        file.write(f"(* ************************** *)\n")
        file.write(f"(* **** Particle classes **** *)\n")
        file.write(f"(* ************************** *)\n\n")
        file.write(f"M$ClassesDescription = {{\n\n")
        self.write_vector(file)
        self.write_fermion(file)
        self.write_scalar(file)
        file.write("};\n\n")

    def write_ExtParams(self, file):
        for itr in self.interactions.values():
            for param in itr.ExtParams:
                file.write(param.to_fr())

    def write_IntParams(self, file):
        param_str_list = []
        for itr in self.interactions.values():
            param_str_list.extend([param.to_fr() for param in itr.IntParams])
        param_str = ",\n".join(param_str_list)
        param_str += "\n"
        file.write(param_str)

    def write_parameters(self, file):
        from sm_setting import write_sm_higgs_parameters_external, write_sm_higgs_parameters_internal
        file.write("(* ************************** *)\n")
        file.write("(* *****   Parameters   ***** *)\n")
        file.write("(* ************************** *)\n")
        file.write("M$Parameters = {\n")
        write_sm_higgs_parameters_external(file)
        write_sm_higgs_parameters_internal(file)
        self.write_ExtParams(file)
        self.write_IntParams(file)
        file.write("};\n")
    
    def write_yukawa(self, file):
        terms = []
        dummy_idx = []
        for itr in self.interactions.values():
            terms.append(itr.to_fr())
            dummy_idx.extend(itr.dummy_idx)
        dummy_idx = list(set(dummy_idx))
        dummy_idx = ", ".join(dummy_idx)

        file.write(f"LYukawa := Block[{{sp, {dummy_idx}, yuk, feynmangaugerules}},\n")
        file.write("  feynmangaugerules = If[Not[FeynmanGauge], {G0|GP|GPbar ->0}, {}];\n")
        file.write("\n")
        file.write("  yuk = ExpandIndices[\n")
        file.write("\n".join(terms))
        file.write("\n  ];\n")
        file.write("  yuk + HC[yuk]/. feynmangaugerules\n")
        file.write("];\n")

    def write_lagrangian(self, file):
        from sm_setting import write_sm_lagrangian
        file.write("(* ************************** *)\n")
        file.write("(* *****   Lagrangian   ***** *)\n")
        file.write("(* ************************** *)\n")
        file.write("\n")
        self.write_yukawa(file)
        write_sm_lagrangian(file)

    def _write_feynrules_file(self, OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        model_dir_name = self.model_name + "_" + self.current_time  
        Output_dir = os.path.join(OUTPUT_PATH, model_dir_name)
        os.makedirs(Output_dir, exist_ok=True)
        
        model_file = os.path.join(Output_dir, f"{self.model_name}.fr")
        particle_file = os.path.join(Output_dir, f"{self.model_name}_particles.fr")
        parameter_file = os.path.join(Output_dir, f"{self.model_name}_parameters.fr")
        lagrangian_file = os.path.join(Output_dir, f"{self.model_name}_lagrangian.fr")

        fr_files = [model_file, particle_file, parameter_file, lagrangian_file]

        for file in fr_files:
            if os.path.exists(file):
                os.remove(file)
        
        with open(model_file, "w") as f:
            f.write("(******************************************************************************************************************)\n")
            f.write("(****** " + f"This is the FeynRules mod-file for {self.model_name}" + " " * (65 - len(self.model_name) ) + " ******)\n")
            f.write("(****** " + " " * 100                                                                                    + " ******)\n")
            f.write("(****** " + f"Author: {self.author}" + " " * (100 - len(self.author) - 8)                                + " ******)\n")
            f.write("(****** " + " " * 100                                                                                    + " ******)\n")
            f.write("(****** Choose whether Feynman gauge is desired.                                                             ******)\n")
            f.write("(****** If set to False, unitary gauge is assumed.                                                           ******)\n")
            f.write("(****** Feynman gauge is especially useful for CalcHEP/CompHEP where the calculation is 10-100 times faster. ******)\n")
            f.write("(****** Feynman gauge is not supported in MadGraph and Sherpa.                                               ******)\n")
            f.write("(******************************************************************************************************************)\n")
            f.write("\n")

            self.write_info(f)
            self.write_gauge(f)
            self.write_vevs(f)
            self.write_gauge_group(f)
            self.write_Indices(f)
            self.write_Interaction_orders(f)
            f.write("\n")
            f.write(f"Get[\"{self.model_name}_particles.fr\"];\n")
            f.write(f"Get[\"{self.model_name}_parameters.fr\"];\n")
            f.write(f"Get[\"{self.model_name}_lagrangian.fr\"];\n")

            with open(particle_file, "w") as f: 
                self.write_fields(f)
                self.write_FeynArts(f)

            with open(parameter_file, "w") as f:
                self.write_parameters(f)

            with open(lagrangian_file, "w") as f:
                self.write_lagrangian(f)

    def write_checklist(self, OUTPUT_PATH):
        with open(os.path.join(OUTPUT_PATH, "checklist.csv"), "w") as f:
            f.write("id, check, result\n")
            for id, checklist in self.checklist.items():
                for key, value in checklist.items():
                    f.write(f"{id}, {key}, {value}\n")

    def to_fr(self, OUTPUT_PATH):
        if self.pass_all_checks():
            self._write_feynrules_file(OUTPUT_PATH)
            print(f"{self.model_name} get score {self.score}.")
            print(f"{self.model_name} passed all checks!")
            print(f"FeynRules file is written to {OUTPUT_PATH}")
        else:
            print(f"{self.model_name} get score {self.score}.")
            print(f"Please check the checklist.log for more details.")

        self.write_checklist(OUTPUT_PATH)
    
if __name__ == "__main__":
    JSON_PATH = '/Users/cooperniu/Documents/codes/particle_game/pygame/particle_graph/SM.json'
    MODEL_PATH = '/Users/cooperniu/Documents/codes/particle_game/pygame/particle_graph/Models'

    model = Model("SM", "Cooper", JSON_PATH)
    model.to_fr(MODEL_PATH)

    