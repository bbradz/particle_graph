from particles import Scalar, Fermion, VectorBoson

from interaction import Yukawa, CovariantDerivative

from multiplet import UnphysicalMultiplet, PhysicalMultiplet

import naming
import numpy as np
import json
import os

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

from group import U, SU, GaugeGroup, GaugeSector
SM_GaugeSector = GaugeSector([GaugeGroup("SU3C", "C", SU(3), confinement=True), 
                         GaugeGroup("SU2W", "W", SU(2)), 
                         GaugeGroup("U1Y", "Y", U(1))],)

#===============================================#
#===             read particles              ===#
#===============================================#
def read_particles(sm_data):
    #= initialize particles =#
    scalars = {}
    fermions = {}
    vector_bosons = {}
    # Dictionary to group particles by quantum number
    
    for particle in sm_data['nodes']['particles']:
        #= fermions =#
        if particle['type'] == 'fermion':
            if not particle['self_conjugate']:
                # self-conjugate fermions can have antiparticles
                new_fermion = Fermion(particle['id'], 
                                      particle['name'], 
                                      particle['label'], 
                                      particle['mass'], 
                                      particle['self_conjugate'], 
                                      False, # the fermion is not an antiparticle type
                                      particle['QuantumNumber'])
                
            else:
                # self-conjugate fermions are their own antiparticles
                new_fermion = Fermion(particle['id'], 
                                      particle['name'], 
                                      particle['label'], 
                                      particle['mass'], 
                                      particle['self_conjugate'], 
                                      None, # antiparticles are not defined for self-conjugate fermions
                                      particle['QuantumNumber'])
            fermions[particle['id']] = new_fermion
                        
        #= scalars =#
        elif particle['type'] == 'scalar':
            new_scalar = Scalar(particle['id'], 
                                scalar_type = particle["scalar_type"],
                                name = particle['name'],
                                label = particle['label'], 
                                goldstone = particle['goldstone'],
                                mass = particle['mass'], 
                                self_conjugate = particle['self_conjugate'], 
                                QuantumNumber = particle['QuantumNumber'])
            scalars[particle['id']] = new_scalar
        
        #= gauge bosons =#
        elif particle['type'] == 'gauge':
            new_gauge = VectorBoson(particle['id'], 
                                    name = particle['name'],
                                    label = particle['label'], 
                                    mass = particle['mass'], 
                                    self_conjugate = particle['self_conjugate'],
                                    abelian = particle['abelian'],
                                    QuantumNumber = particle['QuantumNumber'])
            vector_bosons[particle['id']] = new_gauge

    return scalars, fermions, vector_bosons

#===============================================#
#===             read multiplets             ===#
#===============================================#
def read_multiplets(sm_data, scalars, fermions, vector_bosons):
    unphysical_multiplets = {}
    particles = {**scalars, **fermions, **vector_bosons}

    for multiplet in sm_data['nodes']['multiplets']:
        multiplet_id = multiplet['id']
        
        # initialize particles_list
        particles_list = np.zeros((multiplet["gen"], multiplet["dim"])).tolist()

        for link in sm_data['links']['multiplets']:
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
            elif link['spin_state'] == None:
                particles_list[gen_idx][dim_idx] = particles[link[where]]
            else:
                raise ValueError(f"Invalid Chirality for particle {link[where]}")
        
        new_multiplet = UnphysicalMultiplet(id = multiplet_id, 
                                  label = multiplet["label"],
                                  dim = multiplet["dim"], 
                                  gen = multiplet["gen"],
                                  reps = multiplet["reps"],
                                  gauge_groups = SM_GaugeSector.gauge_groups,
                                  particles = particles_list,
                                  QuantumNumber = multiplet["QuantumNumber"])
        #print(new_multiplet)
        unphysical_multiplets[multiplet_id] = new_multiplet

    # Convert tuple of items back to dictionary for each unique quantum number
    unique_qnumbers_tuples = {tuple(fermion.QuantumNumber.items()) for fermion in fermions.values()}
    unique_qnumbers = [dict(qn_tuple) for qn_tuple in unique_qnumbers_tuples]
    
    physical_multiplets = {}

    for idx, qnumbers in enumerate(unique_qnumbers):

        name = ""
        label = ""
        if qnumbers["BaryonNumber"] != 0 and qnumbers["LeptonNumber"] == 0:
            if qnumbers["Q"] == 2:
                name = "Up-Type Quark"
                label = "uq"
            elif qnumbers["Q"] == -1:
                name = "Down-Type Quark"
                label = "dq"
            elif qnumbers["Q"] == 0:
                name = "Neutral Quark"
                label = "nq"
            else:
                charge = qnumbers["Q"]
                name = naming.number_to_alphabet(charge) + "Quark"
                label = naming.number_to_alphabet(charge) + "q"

        elif qnumbers["BaryonNumber"] == 0 and qnumbers["LeptonNumber"] != 0:
            if qnumbers["Q"] == 0:
                name = "Neutrino"
                label = "vl"
            elif qnumbers["Q"] == -3:
                name = "Charged Lepton"
                label = "l"
            else:
                charge = qnumbers["Q"]
                name = naming.number_to_alphabet(charge) + "Lepton"
                label = naming.number_to_alphabet(charge) + "l"
    
        fermions_in_multiplet = [fermion for fermion in fermions.values() if fermion.QuantumNumber == qnumbers]
        fermions_in_multiplet.sort(key=lambda fermion: fermion.gen_idx)

        id = "mp" + str(idx + 1)
        new_physical_multiplet = PhysicalMultiplet(id = id,
                                            name = name,
                                            label = label,
                                            dim = 1,
                                            gen = len(fermions_in_multiplet),
                                            particles = fermions_in_multiplet,
                                            QuantumNumber = qnumbers)

        physical_multiplets[id] = new_physical_multiplet
    return physical_multiplets, unphysical_multiplets

#===============================================#
#===             write indices               ===#
#===============================================#
def write_gen_indices(multiplets, f):
    if isinstance(multiplets, tuple):
        all_multiplets = {}
        for multiplet in multiplets:
            all_multiplets.update(multiplet)
    elif isinstance(multiplets, dict):
        all_multiplets = multiplets
    else:
        raise ValueError("multiplets must be a tuple or a dictionary")

    idx_list = []
    for multiplet in all_multiplets.values():
        if multiplet.gen not in idx_list:
            idx_list.append(multiplet.gen)
            gen_name = naming.number_to_english(multiplet.gen) + "Gen"
            f.write(f"IndexRange[Index[{gen_name}]] = Range[{multiplet.gen}]\n")

def write_rep_indices(multiplets, f):
    if isinstance(multiplets, tuple):
        all_multiplets = {}
        for multiplet in multiplets:
            all_multiplets.update(multiplet)
    elif isinstance(multiplets, dict):
        all_multiplets = multiplets
    else:
        raise ValueError("multiplets must be a tuple or a dictionary")
    
    idx_list = []
    for multiplet in all_multiplets.values():
        for idx in multiplet.indices:
            if idx not in idx_list:
                f.write(f"IndexRange[{idx}] = Range[{multiplet.dim}]\n")

                idx_list.append(idx)

#===============================================#
#===             write fermions              ===#
#===============================================#
def write_fermions(multiplets, f):
    physical_multiplets, unphysical_multiplets = multiplets

    f.write("(* Fermions: physical fields *)\n")
    f.write("\n")

    for idx, multiplet in enumerate(physical_multiplets.values()):
        f.write(f"  F[{idx+1}] == {{\n")
        f.write(f"    ClassName        -> {multiplet.label},\n")
        f.write(f"    ClassMembers     -> {multiplet.write_ParticleName()},\n")
        f.write(f"    Mass             -> {multiplet.write_Mass()},\n")
        f.write(f"    Width            -> 0,\n")
        f.write(f"    QuantumNumbers   -> {multiplet.write_QuantumNumbers()},\n")
        f.write(f"    Indices          -> {multiplet.write_Indices()},\n")
        f.write(f"    FlavorIndex      -> {multiplet.FlavorIndex},\n")
        f.write(f"    PropagatorLabel  -> {multiplet.write_PropagatorLabel()},\n")
        f.write(f"    PropagatorType   -> Straight,\n")
        f.write(f"    PropagatorArrow  -> Forward,\n")
        f.write(f"    SelfConjugate    -> {multiplet.self_conjugate},\n")
        f.write(f"    ParticleName     -> {multiplet.write_ParticleName()},\n")
        f.write(f"    AntiParticleName -> {multiplet.write_AntiParticleName()},\n")
        f.write(f"    FullName         -> {multiplet.write_FullName()}\n")
        f.write(f"  }},\n")

    f.write("\n")
    f.write("(* Fermions: unphysical fields *)\n")
    f.write("\n")

    for idx, multiplet in enumerate(unphysical_multiplets.values()):
        f.write(f"  F[1{idx+1}] == {{\n")
        f.write(f"    ClassName        -> {multiplet.label}\n")
        f.write(f"    Unphysical       -> True,\n")
        f.write(f"    Indices          -> {multiplet.write_Indices()},\n")
        f.write(f"    FlavorIndex      -> {multiplet.FlavorIndex},\n")
        f.write(f"    SelfConjugate    -> {multiplet.self_conjugate},\n")
        f.write(f"    QuantumNumbers   -> {multiplet.write_QuantumNumbers()},\n")
        f.write(f"    Definitions      -> {multiplet.write_Definition()}\n")
        f.write(f"  }},\n")

def write_particles(fermions, file):
    if os.path.exists(file):
        os.remove(file)

    with open(file, "w") as f:

        f.write("(* ************************** *)\n")
        f.write("(* *****    Indices     ***** *)\n")
        f.write("(* ************************** *)\n")
        f.write("\n")
        write_gen_indices(fermions, f)
        write_rep_indices(fermions, f)
        f.write("\n")

        f.write("(* ************************** *)\n")
        f.write("(* **** Particle classes **** *)\n")
        f.write("(* ************************** *)\n")
        f.write("M$ClassesDescription = {\n")
        f.write("\n")
        
        #write_gauge_bosons(f)
        write_fermions(fermions, f)
        f.write("}\n")

if __name__ == "__main__":

    sm_data = read_json("json2fr/sm.json")
    scalars, fermions, vector_bosons = read_particles(sm_data)
    physical_multiplets, unphysical_multiplets = read_multiplets(sm_data, scalars, fermions, vector_bosons)
    print(unphysical_multiplets['m1'].full_reps)