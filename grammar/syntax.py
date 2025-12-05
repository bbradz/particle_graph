from config import config
import grammar.vocab as vocab

# get allowed charge 
def get_allowed_charge(U1_rep, U2_rep) -> list[str]:
    Y = int(U1_rep.split("_")[1])/6 # hypercharge
    SU2_T3_map = {"singlet": [0],
                 "fnd": [1/2, -1/2],
                 #"adj": [1, 0, -1]
                 }
    T3 = SU2_T3_map.get(U2_rep, [])
    Q = [int((t3 + Y) * 6) for t3 in T3]

    if any(q > config.max_charge or q < -config.max_charge for q in Q):
        return []
    else:
        return [f"charge_{q}" for q in Q]

def get_chirality(particles: dict) -> list[str]:
    candidate_chiralities = []
    for chirality, color_dict in particles.items():
        for color, charge_dict in color_dict.items():
            for charge, p in charge_dict.items():
                if p.get("num_ptcls", 0) > 0: 
                    candidate_chiralities.append(chirality)
    if candidate_chiralities:
        return candidate_chiralities
    else:
        return vocab.CHIRALITIES

# get SU3 representation
def get_SU3_rep(chirality, particles: dict) -> list[str]:
    candidate_reps = []
    for color, charge_dict in particles[chirality].items():
        for charge, p in charge_dict.items():
            if p.get("num_ptcls", 0) > 0: 
                candidate_reps.append("fnd" if color == "COLOR" else "singlet")
    if candidate_reps:
        return candidate_reps
    else:
        return ["singlet", "fnd"]

# get SU2 representation
def get_SU2_rep(chirality) -> list[str]:
    if chirality == "left": return ["fnd"]
    elif chirality == "right": return ["singlet"]
    else: return vocab.REPRESENTATIONS

# get hypercharge
def get_allowed_hypercharge(Q_list, SU2_rep):
    Q_list = [int(Q.split("_")[1])/6 for Q in Q_list]
    T3_map = {
        "singlet": [0],
        "fnd": [1/2, -1/2],
        "adj": [1, 0, -1]}
    allowed_Y_list = []
    for T3 in T3_map[SU2_rep]:
        Y_list = [f"hypercharge_{int(round((Q - T3)*6))}" for Q in Q_list]
        for Y in Y_list:
            allowed_charge = get_allowed_charge(Y, SU2_rep)
            if allowed_charge: allowed_Y_list.append(Y)
    return allowed_Y_list

def get_non_duplicate_hypercharge(Y_list: list[str], rep_list: list[str], multiplets: dict) -> list[str]:
    existing_rep_lists = [value["rep_list"] for value in multiplets.values() if "rep_list" in value]
    candidates = Y_list
    options = []
    for candidate in candidates:
        potential_rep_list = rep_list + [candidate]
        is_duplicate = False
        for existing_rep in existing_rep_lists:
            if existing_rep == potential_rep_list:
                is_duplicate = True
                break
        if not is_duplicate:
            options.append(candidate)
    return options


# get preferred generation
def get_preferred_gen(chirality, color, charge_list, particles) -> int:
    if chirality == "null": return ["gen_1"]
    preferred_gen = 1
    for charge in charge_list:
        num_particles = particles[chirality][color].get(charge, {}).get("num_ptcls", 0)
        if num_particles > preferred_gen:
            preferred_gen = num_particles
    preferred_gen_tokens = [f"gen_{i}" for i in range(preferred_gen, config.max_gen+1)]
    return preferred_gen_tokens 

# get multiplet type
def get_type(particles: dict) -> str:
    for chirality, color_dict in particles.items():
        for color, charge_dict in color_dict.items():
            for charge, p in charge_dict.items():
                if p.get("num_ptcls", 0) > 0: 
                    return "MPLT_SCALAR" if chirality == "null" else "MPLT_FERMION"
    return vocab.MULTIPLET_TYPES

# check if all required particles are included
def req_fullfilled(particles: dict) -> bool:
    for chirality, color_dict in particles.items():
        for color, charge_dict in color_dict.items():
            for charge, p in charge_dict.items():
                if p.get("num_ptcls", 0) > 0: 
                    return False
    return True

# get target charge
def get_target_charge(chirality, particles):
    charge_list = []
    for color, charge_dict in particles[chirality].items():
        for charge, p in charge_dict.items():
            if p.get("num_ptcls", 0) > 0: 
                charge_list.append(charge)
    return charge_list

# get allowed phi4 interactions
def allowed_phi4(multiplets: dict) -> list[str]:
    return [[key] for key, value in multiplets.items() if value["chirality"] == "null" and value["type"] == "MPLT_SCALAR"]

# get allowed yukawa interactions
def allowed_yukawas(multiplets: dict) -> list[str]:
    left_fermions = [key for key, value in multiplets.items() if value["chirality"] == "left" and value["type"] == "MPLT_FERMION"]
    right_fermions = [key for key, value in multiplets.items() if value["chirality"] == "right" and value["type"] == "MPLT_FERMION"]
    scalars = [key for key, value in multiplets.items() if value["chirality"] == "null" and value["type"] == "MPLT_SCALAR"]

    def check_charges():
        return multiplets[rf]["charges"][0] in multiplets[lf]["charges"]

    def check_U1Y():
        Y_scalar = int(multiplets[scalar]["rep_list"][2].split("_")[1])
        Y_left = int(multiplets[lf]["rep_list"][2].split("_")[1])   
        Y_right = int(multiplets[rf]["rep_list"][2].split("_")[1])
        if Y_scalar - Y_left + Y_right == 0: return True
        elif -Y_scalar - Y_left + Y_right == 0: return True
        else: return False

    def check_color():
        color_left = multiplets[lf]["rep_list"][0]
        color_right = multiplets[rf]["rep_list"][0]
        return color_left == color_right
    
    valid_yukawas = []
    for lf in left_fermions:
        for rf in right_fermions:
            for scalar in scalars:
                if check_charges() and check_U1Y() and check_color():
                    valid_yukawas.append([scalar, lf, rf])
    return valid_yukawas

# get next multiplet
def get_next_mplt(mplt_list, query):
    valid_list = [mplts for mplts in mplt_list if mplts[:len(query)] == query]
    idx = len(query)
    next_mplt = [candidate[idx] for candidate in valid_list]
    return next_mplt

if __name__ == "__main__":
    particles = {'null': {'NO_COLOR': {'charge_6': {'ptcls': ['TAG_Hp'], 'num_ptcls': 1}, 'charge_0': {'ptcls': ['TAG_H0'], 'num_ptcls': 1}}}, 'left': {'COLOR': {'charge_4': {'ptcls': ['TAG_U', 'TAG_C', 'TAG_T'], 'num_ptcls': 3}, 'charge_-2': {'ptcls': ['TAG_D', 'TAG_S', 'TAG_B'], 'num_ptcls': 3}}, 'NO_COLOR': {'charge_-6': {'ptcls': ['TAG_E', 'TAG_MU', 'TAG_TAU'], 'num_ptcls': 3}, 'charge_0': {'ptcls': ['TAG_VE', 'TAG_VM', 'TAG_VT'], 'num_ptcls': 3}}}, 'right': {'COLOR': {'charge_4': {'ptcls': ['TAG_U', 'TAG_C', 'TAG_T'], 'num_ptcls': 3}, 'charge_-2': {'ptcls': ['TAG_D', 'TAG_S', 'TAG_B'], 'num_ptcls': 3}}, 'NO_COLOR': {'charge_-6': {'ptcls': ['TAG_E', 'TAG_MU', 'TAG_TAU'], 'num_ptcls': 3}}}}

    print(get_SU3_rep("null", particles))