from dataclasses import dataclass, field
from typing import Dict, List, Any, Set, Optional, Tuple

from config import config
import grammar.vocab as vocab
import grammar.syntax as syntax 
from grammar.parser import print_model, print_particles

group_rank_constraints = {"GAUGE_SU": ["rank_2", "rank_3"], "GAUGE_U": ["rank_1"]}

SU2_dim_map ={"singlet": "dim_1", "fnd": "dim_2", "adj": "dim_3"}

interaction_map = {"TERM_YUKAWA": {"num_mplts": 3, 'mplt_chiralities': ["null","left", "right"]}, 
                   "TERM_PHI4": {"num_mplts": 1, 'mplt_chiralities': ["null"]}}

TAGS = {"charge_-6": ["TAG_E", "TAG_MU", "TAG_TAU"], 
        "charge_0": ["TAG_VE", "TAG_VM", "TAG_VT"], 
        "charge_4": ["TAG_U", "TAG_C"], # no top quark
        "charge_-2": ["TAG_D", "TAG_S", "TAG_B"],
        "charge_6": ["TAG_Hp"], 
        "charge_0": ["TAG_H0"]}

@dataclass
class GrammarState:
    last_token: str = None
    length: int = 1
    current_block: str = None  # gauge/particle/multiplet/interaction

    # model dicts
    particles: dict = field(default_factory=dict)
    gauge_groups: dict = field(default_factory=dict)
    multiplets: dict = field(default_factory=dict)
    interactions: dict = field(default_factory=dict)
    gauge_count: int = 0

    # gauge block
    group_id: str = None
    group_type: str = None
    group_rank: str = None

    # particle block
    particle_type: str = None
    color: str = None
    charge: str = None

    #multiplet block
    multiplet_id: str = None
    multiplet_type: str = None
    chirality: str = None
    charge_list: list[int] = field(default_factory=list)
    rep_list: list[str] = field(default_factory=list)
    dim: int = 0
    gen: int = 0

    # interaction block
    allowed_phi4: list[str] = field(default_factory=list)
    allowed_yukawas: list[str] = field(default_factory=list)
    interaction_id: str = None
    interaction_type: str = None
    param_list: list[str] = field(default_factory=list)
    num_params: int = 0
    mplt_list: list[str] = field(default_factory=list)
    charge_tag: str = None

class GrammarMasker:
    def __init__(self):
        self.token2id = vocab.token2id
        self.id2token = vocab.id2token
        self.vocab_size = len(vocab.GRAMMAR_TOKENS)

    def init_state(self) -> GrammarState:
        return GrammarState()
        
    def step(self, state: GrammarState, token: str) -> GrammarState:
        
        # ======================== GAUGE BLOCK ======================== 
        if token in vocab.GROUP_TYPES: 
            state.current_block = "GAUGE"
            state.group_type = token
        elif token in vocab.GROUP_IDS: state.group_id = token
        elif token in vocab.GROUP_RANKS: state.group_rank = token
        elif token == "END_GAUGE":
            state.gauge_groups[state.group_id] = {
                "id": state.group_id,
                "type": state.group_type,
                "rank": state.group_rank
            }
            state.gauge_count += 1
            state.group_id = None
            state.group_type = None
            state.group_rank = None

        # ======================== PARTICLE BLOCK ======================== 
        elif token in vocab.PARTICLES:
            state.current_block = "PARTICLE"
            state.particle_type = token
        elif state.current_block == "PARTICLE" and token in vocab.CHIRALITIES: 
            state.chirality = token
            state.particles[state.chirality] = {}
        elif token in vocab.COLORS: 
            state.color = token
            state.particles[state.chirality][state.color] = {}
        elif state.current_block == "PARTICLE" and token in vocab.CHARGES:
            state.charge = token
            state.particles[state.chirality][state.color][state.charge] = {"ptcls": [], "num_ptcls": 0}
        elif state.current_block == "PARTICLE" and token in vocab.SM_TAGS:
            state.particles[state.chirality][state.color][state.charge]["ptcls"].append(token)
            state.particles[state.chirality][state.color][state.charge]["num_ptcls"] += 1
        elif token == "END_PARTICLE_BLOCK":
            state.charge = None
            state.color = None
            state.chirality = None
            state.particle_type = None
        
        # ======================== MULTIPLET BLOCK ======================== 
        elif token in vocab.MULTIPLET_TYPES: 
            state.current_block = "MULTIPLET"
            state.multiplet_type = token
        elif token in vocab.MULTIPLET_IDS and state.current_block == "MULTIPLET": state.multiplet_id = token
        elif token in vocab.CHIRALITIES: state.chirality = token
        elif token == "REPS": pass
        elif token in (*vocab.REPRESENTATIONS, *vocab.HYPERCHARGES): state.rep_list.append(token)
        elif token == "END_REPS":
            SU2_rep = state.rep_list[1]
            U1_rep = state.rep_list[2]
            state.charge_list = syntax.get_allowed_charge(U1_rep, SU2_rep)       
        elif token in vocab.DIMS: state.dim = token
        elif token in vocab.GENS: state.gen = token
        elif token == "END_MULTIPLET":
            # remove particles from the particles dict
            color = "NO_COLOR" if state.rep_list[0] == "singlet" else "COLOR"
            for charge in state.charge_list:
                gen = int(state.gen.split("_")[1])
                if state.particles[state.chirality][color].get(charge, []):
                    state.particles[state.chirality][color][charge]["num_ptcls"] -= gen
                else:  
                    pass

            #print_particles(state.particles)
            # save multiplet
            state.multiplets[state.multiplet_id] = {
                "id": state.multiplet_id,
                "type": state.multiplet_type,
                "chirality": state.chirality,
                "rep_list": state.rep_list,
                "gen": state.gen,
                "dim": state.dim,
                "charges": state.charge_list
            }
            state.multiplet_id = None
            state.multiplet_type = None
            state.chirality = None
            state.charge_list = []
            state.rep_list = []
            state.gen = 0
            state.dim = 0

        # ======================== INTERACTION BLOCK ========================
        elif token in vocab.INTERACTION_TYPES:
            state.current_block = "INTERACTION"
            state.interaction_type = token
        elif token in vocab.INTERACTION_IDS: state.interaction_id = token
        elif token in (*vocab.MASS, *vocab.PARAMETERS): state.param_list.append(token)
        elif token in vocab.MULTIPLET_IDS and state.current_block == "INTERACTION": state.mplt_list.append(token)
        elif token == "END_MPLT":
            if state.interaction_type == "TERM_YUKAWA":
                gen_left = int(state.multiplets[state.mplt_list[1]]["gen"].split("_")[1])
                gen_right = int(state.multiplets[state.mplt_list[2]]["gen"].split("_")[1])
                state.num_params = max(gen_left, gen_right)
                state.allowed_yukawas.remove(state.mplt_list)
            elif state.interaction_type == "TERM_PHI4":
                state.num_params = 1
                state.allowed_phi4.remove(state.mplt_list)
        elif token == "END_INTERACTION":
            state.interactions[state.interaction_id] = {
                "id": state.interaction_id,
                "type": state.interaction_type,
                "param_list": state.param_list,
                "mplt_list": state.mplt_list
            }
            state.interaction_id = None
            state.interaction_type = None
            state.param_list = []
            state.mplt_list = []
            state.num_params = 0

        elif token == "EOS":
            pass
            #print(syntax.req_fullfilled(state.particles))
            


        # ======================== UPDATE LAST TOKEN AND LENGTH ========================
        state.last_token= token
        state.length += 1
        return state



    def get_valid_tokens(self, state: GrammarState, token: str) -> list[str]:

        # ======================== BOS ======================== 
        if token == "BOS": return vocab.GROUP_TYPES

        # ======================== GAUGE BLOCK ======================== 
        elif token in vocab.GROUP_TYPES: return [f"g_{len(state.gauge_groups)+1}"]
        elif state.current_block == "GAUGE": 
            if token.startswith("g_"): return group_rank_constraints[state.group_type]
            elif token in vocab.GROUP_RANKS: return ["END_GAUGE"]
            elif token == "END_GAUGE": return vocab.GROUP_TYPES + vocab.PARTICLES
            else: print(f"Invalid token in GAUGE block: {token}")


        # ======================== PARTICLE BLOCK ======================== 
        elif token == "PTCL_SCALAR": return ["null"]
        elif token == "PTCL_FERMION": return ["left", "right"]
        elif state.current_block == "PARTICLE":
            if token in vocab.CHIRALITIES: 
                return vocab.COLORS
            elif token in vocab.COLORS: return vocab.CHARGES
            elif token in vocab.CHARGES: return vocab.SM_TAGS
            elif token in vocab.SM_TAGS: return vocab.SM_TAGS + ["END_PTCL"]
            elif token == "END_PTCL": return ["END_PARTICLE_BLOCK"] + vocab.CHARGES + vocab.COLORS + vocab.PARTICLES
            elif token == "END_PARTICLE_BLOCK": return vocab.MULTIPLET_TYPES
            else: print(f"Invalid token in PARTICLE block: {token}")


        # ======================== MULTIPLET BLOCK ======================== 
        elif token in vocab.MULTIPLET_TYPES: return [f"m_{len(state.multiplets)+1}"]
        elif state.current_block == "MULTIPLET": 
            if token.startswith("m_"): 
                if state.multiplet_type == "MPLT_SCALAR": return ["null"]
                elif state.multiplet_type == "MPLT_FERMION":
                    return syntax.get_chirality(state.particles)
            elif token in vocab.CHIRALITIES: return ["REPS"]
            elif token in ("REPS", *vocab.REPRESENTATIONS, *vocab.HYPERCHARGES):
                group_key = f"g_{len(state.rep_list)+1}"
                if group_key == "g_1": 
                    return syntax.get_SU3_rep(state.chirality, state.particles)
                elif group_key == "g_2":
                    return syntax.get_SU2_rep(state.chirality)
                elif group_key == "g_3": 
                    target_charge = syntax.get_target_charge(state.chirality, state.particles)
                    allowed_HYCHARGES = syntax.get_allowed_hypercharge(target_charge, state.rep_list[1])
                    non_duplicate_HYCHARGES = syntax.get_non_duplicate_hypercharge(allowed_HYCHARGES, state.rep_list, state.multiplets)
                    if not non_duplicate_HYCHARGES: raise ValueError(f"No non-duplicate hypercharge found for {state.rep_list}")
                    return non_duplicate_HYCHARGES
                else: return ["END_REPS"]
            elif token == "END_REPS":
                # Chain-of-thought: return DIMS
                SU2_rep = state.rep_list[1]
                return [SU2_dim_map[SU2_rep]]
            elif token in vocab.DIMS: return vocab.GENS
            elif token in vocab.GENS: return ["END_MULTIPLET"]
            elif token == "END_MULTIPLET": 
                if (syntax.req_fullfilled(state.particles)
                    or len(state.multiplets) == config.max_multiplets):
                    state.allowed_phi4 = syntax.allowed_phi4(state.multiplets)
                    state.allowed_yukawas = syntax.allowed_yukawas(state.multiplets)
                    if state.allowed_phi4: return ["TERM_PHI4"]
                    elif state.allowed_yukawas: return ["TERM_YUKAWA"]
                    else: print(f"Invalid multiplet block: {state.multiplets}")
                else: 
                    return [syntax.get_type(state.particles)]
            else: print(f"Invalid token in MULTIPLET block: {token}")

        # ======================== INTERACTION BLOCK ========================
        elif token in vocab.INTERACTION_TYPES: return [f"i_{len(state.interactions)+1}"]
        elif state.current_block == "INTERACTION":
            if token.startswith("i_"): return ["MPLTS"]
            elif token in ("MPLTS", *vocab.MULTIPLET_IDS): 
                if len(state.mplt_list) == interaction_map[state.interaction_type]["num_mplts"]:
                    return ["END_MPLT"]
                if state.interaction_type == "TERM_YUKAWA":
                    return syntax.get_next_mplt(state.allowed_yukawas, state.mplt_list)
                elif state.interaction_type == "TERM_PHI4":
                    return syntax.get_next_mplt(state.allowed_phi4, state.mplt_list)
            elif token == "END_MPLT": return ["PARAMS"]
            elif token in ("PARAMS", *vocab.MASS, *vocab.PARAMETERS): 
                if state.interaction_type == "TERM_YUKAWA": 
                    if len(state.param_list) < state.num_params:
                        # there must be a mass hierarchy
                        if len(state.param_list) == 0: return vocab.MASS
                        else: 
                            return vocab.MASS[vocab.MASS.index(state.last_token):]
                    else: return ["END_PARAM"]
                elif state.interaction_type == "TERM_PHI4":
                    if len(state.param_list) < state.num_params:
                        return vocab.PARAMETERS
                    else: return ["END_PARAM"]
            elif token == "END_PARAM": return ["END_INTERACTION"]
            elif token == "END_INTERACTION": 
                if len(state.interactions) == config.max_interactions: return ["EOS"]
                elif state.allowed_phi4: return ["TERM_PHI4"]
                elif state.allowed_yukawas: return ["TERM_YUKAWA"]
                else: return ["EOS"]
            else: print(f"Invalid token in INTERACTION block: {token}")