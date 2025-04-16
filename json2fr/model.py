from datetime import datetime
import os
import naming 
import numpy as np
from multiplet import UnphysicalMultiplet, PhysicalMultiplet


class model:
    """
    Particle Physics Model Class: read JSON file and write FR model file
    """
    def __init__(self, 
                 model_name, 
                 author, 
                 JSON_PATH                
                 ):
        
        self.model_name = model_name
        self.author = author
        
        self.ai = True if author == 'Bohr Network' else False

        self.FeynmanGauge = True
        self.JSON_PATH = JSON_PATH
        self.current_time = datetime.now().strftime("%Y-%m-%d")

        # read JSON file and write FR model file
        self.read_model()


    def read_json(self):
        """Read JSON file and return the data as a dictionary."""
        import json
        
        try:
            with open(self.JSON_PATH, 'r') as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError:
            print(f"Error: {self.JSON_PATH} is not a valid JSON file.")
            return None
        except FileNotFoundError:
            print(f"Error: File {self.JSON_PATH} not found.")
            return None
        
    @staticmethod
    def read_gauge_groups(model_data):
        """Read gauge groups from JSON file"""
        
        from group import U, SU, GaugeGroup
        
        gauge_groups = {}
        for g in model_data['nodes']['GaugeGroups']:
            group_type = g['group'][:g['group'].find("(")]
            N = int(g['group'][g['group'].find("(")+1:g['group'].find(")")])

            if group_type == "U":
                group = U(N)
            elif group_type == "SU":
                group = SU(N)
            else:
                raise ValueError(f"Invalid group: {g['group']}")    

            new_gauge_group = GaugeGroup(g['id'],
                                         g['name'],
                                         g['charge'],
                                         group,
                                         g['CouplingConstant'],
                                         g['confinement'],
                                         g['gauge_boson']
                                         )
            gauge_groups[g['id']] = new_gauge_group
        return gauge_groups

    @staticmethod
    def read_particles(model_data):
        """Read particles from JSON file"""

        from particles import Fermion, Scalar, VectorBoson
        
        scalars = {}
        fermions = {}
        vector_bosons = {}
        for particle in model_data['nodes']['particles']:
            #= fermions =#
            if particle['type'] == 'fermion':
                if not particle['self_conjugate']:
                    # self-conjugate fermions can have antiparticles
                    new_fermion = Fermion(particle['id'], 
                                          particle['full_name'], 
                                          particle['name'], 
                                          particle['mass'], 
                                          particle['self_conjugate'], 
                                          particle['QuantumNumber'])
                    
                else:
                    # self-conjugate fermions are their own antiparticles
                    new_fermion = Fermion(particle['id'], 
                                          particle['full_name'], 
                                          particle['name'], 
                                          particle['mass'], 
                                          particle['self_conjugate'], 
                                          particle['QuantumNumber'])
                fermions[particle['id']] = new_fermion
                            
            #= scalars =#
            elif particle['type'] == 'scalar':
                new_scalar = Scalar(particle['id'], 
                                    particle["scalar_type"],
                                    particle['full_name'],
                                    particle['name'],
                                    particle['mass'], 
                                    particle['self_conjugate'], 
                                    particle['QuantumNumber'])
                scalars[particle['id']] = new_scalar

            #= gauge bosons =#
            elif particle['type'] == 'gauge':
                raise ValueError(f"Gauge bosons are not automatically generated from Gauge Groups.")
                # new_gauge = VectorBoson(id = particle['id'], 
                #                         name = particle['name'],
                #                         mass = 0, # Gauge Bosons are massless before SSB
                #                         self_conjugate = particle['self_conjugate'])
                # vector_bosons[particle['id']] = new_gauge
                
            else:
                raise ValueError(f"Invalid particle type: {particle['type']}")
            
        return scalars, fermions


    @staticmethod
    def read_multiplets(gauge_groups, particles, model_data):
        """Read multiplets from JSON file"""
        unphy_mltplt = {}

        for multiplet in model_data['nodes']['multiplets']:
            multiplet_id = multiplet['id']
            
            # initialize particles_list
            particles_list = np.zeros((multiplet["gen"], multiplet["dim"])).tolist()
            for link in model_data['links']['multiplets']:
                if link['source'] == multiplet_id:
                    where = "target"
                elif link['target'] == multiplet_id:
                    where = "source"
                else:
                    continue

                gen_idx = link['loc'][0]-1
                dim_idx = link['loc'][1]-1

                particles[link[where]].gen_idx = link['loc'][0]
                particles[link[where]].dim_idx = link['loc'][1]

                if link['spin_state'] == 'left':
                    particles_list[gen_idx][dim_idx] = particles[link[where]].left()
                elif link['spin_state'] == 'right':
                    particles_list[gen_idx][dim_idx] = particles[link[where]].right()
                elif link['spin_state'] is None:
                    particles_list[gen_idx][dim_idx] = particles[link[where]]
                else:
                    raise ValueError(f"Invalid Chirality for particle {link[where]}")
            
            new_unphy_mltplt = UnphysicalMultiplet(id = multiplet_id, 
                                                   name = multiplet["name"],
                                                   dim = multiplet["dim"], 
                                                   gen = multiplet["gen"],
                                                   reps = multiplet["reps"],
                                                   gauge_groups = gauge_groups,
                                                   particles = particles_list,
                                                   QuantumNumber = multiplet["QuantumNumber"]
                                                   )
            unphy_mltplt[multiplet_id] = new_unphy_mltplt
        return unphy_mltplt
    
    @staticmethod
    def read_fermions(fermions):
        """Read fermions from JSON file"""
        phy_mltplt = {}

        qnumbers_tuples = {tuple(f.QuantumNumber.items()) for f in fermions.values()}
        unique_qnumbers = [dict(qn_tuple) for qn_tuple in qnumbers_tuples]

        for idx, qnumbers in enumerate(unique_qnumbers):
            full_name = ""
            name = ""
            if qnumbers["BaryonNumber"] != 0 and qnumbers["LeptonNumber"] == 0:
                if qnumbers["Q"] == 2:
                    full_name = "Up-Type Quark"
                    name = "uq"
                elif qnumbers["Q"] == -1:
                    full_name = "Down-Type Quark"
                    name = "dq"
                elif qnumbers["Q"] == 0:
                    full_name = "Neutral Quark"
                    name = "nq"
                else:
                    charge = qnumbers["Q"]
                    full_name = naming.number_to_alphabet(charge) + "Quark"
                    name = naming.number_to_alphabet(charge) + "q"

            elif qnumbers["BaryonNumber"] == 0 and qnumbers["LeptonNumber"] != 0:
                if qnumbers["Q"] == 0:
                    full_name = "Neutrino"
                    name = "vl"
                elif qnumbers["Q"] == -3:
                    full_name = "Charged Lepton"
                    name = "l"
                else:
                    charge = qnumbers["Q"]
                    full_name = naming.number_to_alphabet(charge) + "Lepton"
                    name = naming.number_to_alphabet(charge) + "l"
        
            fermions_in_multiplet = [fermion for fermion in fermions.values() if fermion.QuantumNumber == qnumbers]
            fermions_in_multiplet.sort(key=lambda fermion: fermion.gen_idx)

            id = "mp" + str(idx + 1)
            new_phy_mltplt = PhysicalMultiplet(id = id,
                                               full_name = full_name,
                                               name = name,
                                               dim = 1,
                                               gen = len(fermions_in_multiplet),
                                               particles = fermions_in_multiplet,
                                               QuantumNumber = qnumbers
                                               )
            phy_mltplt[id] = new_phy_mltplt
        return phy_mltplt
        
    @staticmethod
    def read_scalar_fields(scalars):
        """Read scalar fields from JSON file"""
        pass

    def read_model(self):
        """Read model from JSON file"""
        self.indices = []
        model_data = self.read_json()
        self.gauge_groups = model.read_gauge_groups(model_data)

        self.scalars, self.fermions, self.vector_bosons = model.read_particles(model_data)
        self.particles = {**self.scalars, **self.fermions, **self.vector_bosons}
        self.unphy_mltplts = model.read_multiplets(self.gauge_groups, self.particles, model_data)

        self.unphy_fermions = {}
        self.unphy_scalars = {}
        self.unphy_vector_bosons = {}

        for m in self.unphy_mltplts.values():
            if m.particle_type == 'fermion':
                self.unphy_fermions[m.id] = m
            elif m.particle_type == 'scalar':
                self.unphy_scalars[m.id] = m
            elif m.particle_type == 'gauge':
                self.unphy_vector_bosons[m.id] = m

        self.phy_fermions = model.read_fermions(self.fermions)

        for m in self.unphy_mltplts.values():
            self.indices.extend(m.indices)

        self.indices = list(set(self.indices))
        
        for idx in self.indices:
            if idx[3] is not None:
                self.gauge_groups[idx[3]].reps.append(idx[1])


    def write_info(self, file):
        file.write(f"(* ************************** *)\n")
        file.write(f"(* *****  Information   ***** *)\n")
        file.write(f"(* ************************** *)\n")
        file.write(f"M$ModelName = \"{self.model_name}\";\n")
        file.write("\n")
        file.write(f"M$Information = {{ \n")
        file.write(f"  Authors      -> {self.author}, \n")
        file.write(f"  Date         -> \"{self.current_time}\", \n")
        file.write(f"}};\n")
        file.write("\n")

    def write_gauge(self, file):
        file.write(f"FeynmanGauge = {self.FeynmanGauge};\n")
        file.write("\n")

    def write_vev(self, file):
        file.write(f"(* ************************** *)\n")
        file.write(f"(* *****       VEV      ***** *)\n")
        file.write(f"(* ************************** *)\n")
        file.write(f"M$vevs = {{}}\n")
        file.write(f"\n")
        
    def write_group(self, file):
        file.write(f"(* ************************** *)\n")
        file.write(f"(* *****  Gauge groups  ***** *)\n")
        file.write(f"(* ************************** *)\n")
        file.write(f"M$gaugeGroup = {{\n")
        for group in self.gauge_groups.values():
            file.write(f"  {group.name} == {{\n")
            file.write(f"    Abelian           -> {group.abelian},\n")
            file.write(f"    CouplingConstant  -> {group.coupling_constant},\n")
            file.write(f"    GaugeBoson        -> {group.boson},\n")
            if group.abelian:
                file.write(f"    Charge            -> {group.charge}\n")
                file.write(f"  }},\n")
            else:
                file.write(f"    StructureConstant -> {group.structure_constants},\n")
                file.write(f"    Representations   -> {group.rep_list},\n")
                file.write(f"    SymmetricTensor   -> {group.sym_tensors}\n") if group.sym_tensors else None
                file.write(f"    Definition        -> {group.definition}\n") if group.definition else None
                file.write(f"  }},\n") 
        file.write(f"}};\n")
        file.write("\n" * 2)
        
    def write_indices(self, file):
        file.write(f"(* ************************** *)\n")
        file.write(f"(* *****    Indices     ***** *)\n")
        file.write(f"(* ************************** *)\n")
        file.write("\n")
        for idx in self.indices:
            if idx[0] == "fold":
                file.write(f"IndexRange[Index[{idx[1]}]] = Range[{idx[2]}],\n")
            elif idx[0] == "Unfold":
                file.write(f"IndexRange[Index[{idx[1]}]] = Unfold[Range[{idx[2]}]],\n")
            elif idx[0] == "NoUnfold":
                file.write(f"IndexRange[Index[{idx[1]}]] = NoUnfold[Range[{idx[2]}]],\n")
        file.write("\n")
        for i, idx in enumerate(self.indices):
            file.write(f"IndexStyle[{idx[1]}, {naming.index_style(i)}];\n")
        file.write("\n" * 2)

    def write_interaction_orders(self, file):
        file.write(f"(* ************************** *)\n")
        file.write(f"(* *** Interaction orders *** *)\n")
        file.write(f"(* ***  (as used by mg5)  *** *)\n")
        file.write(f"(* ************************** *)\n")
        file.write(f"M$InteractionOrderHierarchy = {{ }}\n")
        file.write("\n" * 2)

    def write_particle_classes(self, file):
        file.write(f"(* ************************** *)\n")
        file.write(f"(* **** Particle classes **** *)\n")
        file.write(f"(* ************************** *)\n")
        file.write(f"\n")
        file.write(f"M$ClassesDescription = {{\n")
        file.write("\n")
        file.write("(* Fermions: physical fields *)\n")

        for idx, multiplet in enumerate(self.phy_fermions.values()):
            file.write(f"  F[{idx+1}] == {{\n")
            file.write(f"    ClassName        -> {multiplet.name},\n")
            file.write(f"    ClassMembers     -> {multiplet.write_ParticleName()},\n")
            file.write(f"    Mass             -> {multiplet.write_Mass()},\n")
            file.write(f"    Width            -> 0,\n")
            file.write(f"    QuantumNumbers   -> {multiplet.write_QuantumNumbers()},\n")
            file.write(f"    Indices          -> {multiplet.write_Indices()},\n")
            file.write(f"    FlavorIndex      -> {multiplet.GenerationIndex},\n")
            file.write(f"    PropagatorLabel  -> {multiplet.write_PropagatorLabel()},\n")
            file.write(f"    PropagatorType   -> Straight,\n")
            file.write(f"    PropagatorArrow  -> Forward,\n")
            file.write(f"    SelfConjugate    -> {multiplet.self_conjugate},\n")
            file.write(f"    ParticleName     -> {multiplet.write_ParticleName()},\n")
            file.write(f"    AntiParticleName -> {multiplet.write_AntiParticleName()},\n")
            file.write(f"    FullName         -> {multiplet.write_FullName()}\n")
            file.write(f"  }},\n")

        file.write("\n")
        file.write("(* Fermions: unphysical fields *)\n")

        for idx, multiplet in enumerate(self.unphy_fermions.values()):
            file.write(f"  F[1{idx+1}] == {{\n")
            file.write(f"    ClassName        -> {multiplet.name}\n")
            file.write(f"    Unphysical       -> True,\n")
            file.write(f"    Indices          -> {multiplet.write_Indices()},\n")
            file.write(f"    FlavorIndex      -> {multiplet.GenerationIndex},\n")
            file.write(f"    SelfConjugate    -> {multiplet.self_conjugate},\n")
            file.write(f"    QuantumNumbers   -> {multiplet.write_QuantumNumbers()},\n")
            file.write(f"    Definitions      -> {multiplet.write_Definition()}\n")
            file.write(f"  }},\n")
        file.write("}\n")
        file.write("\n" * 2)

    def write_parameter(self, file):
        file.write(f"(* ************************** *)\n")
        file.write(f"(* *****   Parameters   ***** *)\n")
        file.write(f"(* ************************** *)\n")
        file.write(f"\n")
        file.write(f"M$Parameters = {{\n")
        file.write("\n" * 2)

    def write_lagrangian(self, file):
        file.write(f"(* ************************** *)\n")
        file.write(f"(* *****   Lagrangian   ***** *)\n")
        file.write(f"(* ************************** *)\n")
        file.write(f"\n")
        
    def write_model(self, OUTPUT_PATH):

        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

        model_file = os.path.join(OUTPUT_PATH, f"{self.model_name}.fr")
        particle_file = os.path.join(OUTPUT_PATH, f"{self.model_name}_particles.fr")
        parameter_file = os.path.join(OUTPUT_PATH, f"{self.model_name}_parameters.fr")
        lagrangian_file = os.path.join(OUTPUT_PATH, f"{self.model_name}_lagrangian.fr")

        if os.path.exists(model_file):
            os.remove(model_file)
        if os.path.exists(particle_file):
            os.remove(particle_file)
        if os.path.exists(parameter_file):
            os.remove(parameter_file)
        if os.path.exists(lagrangian_file):
            os.remove(lagrangian_file)

        with open(model_file, "w") as f:

            f.write("(******************************************************************************************************************)\n")
            f.write("(****** " + f"This is the FeynRules mod-file for {self.model_name}" + " " * (65 - len(self.model_name) ) + " ******)\n")
            f.write("(****** " + " " * 100 + " ******)\n")
            f.write("(****** " + f"Author: {self.author}" + " " * (100 - len(self.author) - 8) + " ******)\n")
            f.write("(****** " + " " * 100 + " ******)\n")
            f.write("(****** Choose whether Feynman gauge is desired.                                                             ******)\n")
            f.write("(****** If set to False, unitary gauge is assumed.                                                           ******)\n")
            f.write("(****** Feynman gauge is especially useful for CalcHEP/CompHEP where the calculation is 10-100 times faster. ******)\n")
            f.write("(****** Feynman gauge is not supported in MadGraph and Sherpa.                                               ******)\n")
            f.write("(******************************************************************************************************************)\n")
            f.write("\n")

            self.write_info(f)
            self.write_gauge(f)
            self.write_vev(f)
            self.write_group(f)
            self.write_indices(f)
            self.write_interaction_orders(f)
            f.write("\n")
            f.write(f"Get[\"{self.model_name}_particles.fr\"];\n\n")
            f.write(f"Get[\"{self.model_name}_parameters.fr\"];\n\n")
            f.write(f"Get[\"{self.model_name}_lagrangian.fr\"];\n\n")

        with open(particle_file, "w") as f:
            self.write_particle_classes(f)

        with open(parameter_file, "w") as f:
            self.write_parameter(f)

        with open(lagrangian_file, "w") as f:
            self.write_lagrangian(f)

        #print(f"Model {self.model_name} is successfully written to {OUTPUT_PATH}!")
    

if __name__ == "__main__":
    JSON_PATH = 'particle_graph/model/sm.json'
    MODEL_PATH = 'particle_graph/model'
    a = model('SM', 'Bohr Network', JSON_PATH)
    a.write_model(MODEL_PATH)