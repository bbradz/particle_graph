import random, json, hashlib, os

pdg = {
    "particles": [
      {
        "full_name": "Photon",
        "name": "gamma",
        "pdgid": 22,
        "spin": 2,
        "mass": 0.0,
        "width": 0.0,
        "charge": 0,
        "color": 1
      },
      {
        "full_name": "Z Boson",
        "name": "Z",
        "pdgid": 23,
        "spin": 2,
        "mass": 91.1876,
        "width": 2.4952,
        "charge": 0,
        "color": 1
      },
      {
        "full_name": "W Boson",
        "name": "W",
        "pdgid": 24,
        "spin": 2,
        "mass": 80.379,
        "width": 2.0476,
        "charge": 3,
        "color": 1
      },
      {
        "full_name": "Gluon",
        "name": "G",
        "pdgid": 21,
        "spin": 2,
        "mass": 0.0,
        "width": 0.0,
        "charge": 0,
        "color": 8
      },
      {
        "full_name": "Higgs Boson",
        "name": "H",
        "pdgid": 25,
        "spin": 0,
        "mass": 125.25,
        "width": 0.00407,
        "charge": 0,
        "color": 1
      },
      {
        "full_name": "Electron",
        "name": "e",
        "pdgid": 11,
        "spin": 1,
        "mass": 0.000511,
        "width": 0.0,
        "charge": -3,
        "color": 1,
        "flavor": "e"
      },
      {
        "full_name": "Muon",
        "name": "mu",
        "pdgid": 13,
        "spin": 1,
        "mass": 0.1057,
        "width": 0.0,
        "charge": -3,
        "color": 1,
        "flavor": "mu"
      },
      {
        "full_name": "Tau",
        "name": "ta",
        "pdgid": 15,
        "spin": 1,
        "mass": 1.777,
        "width": 0.0,
        "charge": -3,
        "color": 1,
        "flavor": "ta"
      },
      {
        "full_name": "Electron-neutrino",
        "name": "ve",
        "pdgid": 12,
        "spin": 1,
        "mass": 0.0,
        "width": 0.0,
        "charge": 0,
        "color": 1,
        "flavor": "e"
      },
      {
        "full_name": "Muon-neutrino",
        "name": "vm",
        "pdgid": 14,
        "spin": 1,
        "mass": 0.0,
        "width": 0.0,
        "charge": 0,
        "color": 1,
        "flavor": "mu"
      },
      {
        "full_name": "Tau-neutrino",
        "name": "vt",
        "pdgid": 16,
        "spin": 1,
        "mass": 0.0,
        "width": 0.0,
        "charge": 0,
        "color": 1,
        "flavor": "ta"
      },
      {
        "full_name": "u-quark",
        "name": "u",
        "pdgid": 2,
        "spin": 1,
        "mass": 0.0023,
        "width": 0.0,
        "charge": 2,
        "color": 3,
        "flavor": "u"
      },
      {
        "full_name": "d-quark",
        "name": "d",
        "pdgid": 1,
        "spin": 1,
        "mass": 0.0047,
        "width": 0.0,
        "charge": -1,
        "color": 3,
        "flavor": "d"
      },
      {
        "full_name": "c-quark",
        "name": "c",
        "pdgid": 4,
        "spin": 1,
        "mass": 1.27,
        "width": 0.0,
        "charge": 2,
        "color": 3,
        "flavor": "c"
      },
      {
        "full_name": "s-quark",
        "name": "s",
        "pdgid": 3,
        "spin": 1,
        "mass": 0.095,
        "width": 0.0,
        "charge": -1,
        "color": 3,
        "flavor": "s"
      },
      {
        "full_name": "t-quark",
        "name": "t",
        "pdgid": 6,
        "spin": 1,
        "mass": 172.76,
        "width": 1.50833649,
        "charge": 2,
        "color": 3,
        "flavor": "t"
      },
      {
        "full_name": "b-quark",
        "name": "b",
        "pdgid": 5,
        "spin": 1,
        "mass": 4.18,
        "width": 0.0,
        "charge": -1,
        "color": 3,
        "flavor": "b"
      },
      {
        "full_name": "Charged Goldstone",
        "name": "GP",
        "pdgid": 251,
        "spin": 0,
        "mass": 0.0,
        "width": 2.085,
        "charge": 3,
        "color": 1
      },
      {
        "full_name": "Neutral Goldstone",
        "name": "G0",
        "pdgid": 250,
        "spin": 0,
        "mass": 0.0,
        "width": 2.4952,
        "charge": 0,
        "color": 1
      }
    ]
  }

def get_pdg(mass, charge, color, spin, flavor, PDG_DATA = pdg, mass_tol = 1e-2):
    """
    Get the particle from the PDG data
    """
    candidates = PDG_DATA['particles']

    if color is not None:
        candidates = [p for p in candidates if p['color'] == color]
    if spin is not None:
        candidates = [p for p in candidates if p['spin'] == spin]
    if flavor is not None:
        candidates = [p for p in candidates if p['flavor'] == flavor]
    if charge is not None:
        candidates = [p for p in candidates if p['charge'] == charge]    
    if mass is not None:
        candidates = [p for p in candidates if abs(p['mass'] - mass) <= mass_tol * mass or (p['mass'] == mass)]
    
    if len(candidates) == 1:
            return candidates[0]
    else:
        return None


def BSM_id(mass, charge, color, spin, flavor, offset = 9000000):
    """
    Get the BSM particle ID from the PDG data
    """
    props = f"{mass:.6f}|{charge}|{color}|{spin}|{flavor}"
    hashed = hashlib.md5(props.encode()).hexdigest()
    numeric_hash = int(hashed[:8], 16)
    bsm_pdg_id = offset + (numeric_hash % (9999999 - offset))

    return bsm_pdg_id


# generate random inputs
def random_inputs(count=6):
    """Generate a list of random values that can be strings, integers, floats, or None."""
    result = []
    for _ in range(count):
        value_type = random.choice(['str', 'int'])
        
        if value_type == 'str':
            # Generate a random string of length 3-8
            length = random.randint(3, 8)
            random_string = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz_') for _ in range(length))
            result.append(random_string)
        elif value_type == 'int':
            result.append(random.randint(-100, 100))
        elif value_type == 'float':
            result.append(random.uniform(-100.0, 100.0))
        else:  # None
            result.append(None)
            
    return result

def count_calls(func):
    def wrapper(*args, **kwargs):
        wrapper.call_count += 1
        return func(*args, **kwargs)
    wrapper.call_count = 0
    return wrapper

def run_checks(all_checks, checklist, skip_check = False):
        for check in all_checks:
            fail_previous_check = any(isinstance(value, Exception) or value == False for value in checklist.values())
            if skip_check and fail_previous_check:
                checklist[check.__name__] = Exception(f"Skipped due to previous check failure")
            else:
                try:
                    check()
                    checklist[check.__name__] = True
                except Exception as e:
                    checklist[check.__name__] = e

### ========================================================================== ###
###                                                                            ###
###                           Particles Classes                                ###
###                                                                            ###
### ========================================================================== ###

import numpy as np
from fractions import Fraction

# ====================================================================
#                              Field
# ====================================================================
class Field:
    """
    Base class for fields.
    id: str
    name: str
    type: str ["complex", "real", "fermion", "vector"]
    reps: dict of str (non-abelian group reps) or int (abelian group charge)
    groups: dict of group
    dim: int > 0
    gen: int > 0
    particles: list of Particle
    self_conjugate: bool
    QuantumNumber: dict
    """
    def __init__(self, id, name, type, groups, reps, dim, gen, particles, self_conjugate, QuantumNumber):
        self.id = id
        self.name = name
        self.type = type
        self.groups = groups
        self.reps = reps
        self.dim = dim
        self.gen = gen
        self.particles = particles
        self.self_conjugate = self_conjugate
        self.QuantumNumber = QuantumNumber
        self.indices = []
        self.full_reps = {}
        self.abelian_charges = {}
        self.__check__()
        
        
    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __dict__(self):
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "reps": self.reps,
            "groups": self.groups,
            "dim": self.dim,
            "gen": self.gen,
            "particles": self.particles,
            "self_conjugate": self.self_conjugate,
            "QuantumNumber": self.QuantumNumber
        }
    
    def _all_checks(self):
        """ All checks for the initial INPUTs of the 'Field' class. """
        self.all_checks = []

        # ------------------------------------------------------------------
        def _id_check():
            assert isinstance(self.id, str), \
                f"Error: 'id' must be a string."
        
        def _name_check():
            assert isinstance(self.name, str), \
                f"Error: 'name' must be a string."
        
        def _type_check():
            assert isinstance(self.type, str), \
                f"Error: 'type' must be a string."
            assert self.type in ["complex", "real", "fermion", "vector"], \
                f"Error: 'type' must be one of complex/real/fermion/vector."
        
        def _groups_check():
            assert isinstance(self.groups, dict), \
                f"Error: 'groups' must be a dictionary."
            assert all(isinstance(g, str) for g in self.groups), \
                f"Error: 'groups' must be a dictionary of strings."

        def _reps_check():
            assert isinstance(self.reps, dict), \
                f"Error: 'reps' must be a dict."
            assert all(isinstance(x, int) or isinstance(x, str) for x in self.reps.values()), \
                f"Error: 'reps' must be a dict of int or str."
            assert len(self.reps) == len(self.groups), \
                f"Error: 'reps' and 'groups' must be the same length."
        
        def _dim_check():
            assert isinstance(self.dim, int) and self.dim > 0, \
                f"Error: 'dim' must be a positive integer."
        
        def _gen_check():
            assert isinstance(self.gen, int) and self.gen > 0, \
                f"Error: 'gen' must be a positive integer."
        
        def _particles_check():
            assert isinstance(self.particles, list), \
                f"Error: 'particles' must be a list."
            if self.type == "fermion":
                assert all(isinstance(p, WeylSpinor) for p in self.particles), \
                    f"Error: 'particles' for fermion fields must be a list of WeylSpinor."
            else:
                assert all(isinstance(p, Particle) for p in self.particles), \
                    f"Error: 'particles' for non-fermion fields must be a list of Particle."
            
        def _self_conjugate_check():
            assert isinstance(self.self_conjugate, bool), \
                f"Error: 'self_conjugate' must be a bool."
        
        def _QuantumNumber_check():
            assert isinstance(self.QuantumNumber, dict), \
                f"Error: 'QuantumNumber' must be a dict."
            assert all(isinstance(x, int) for x in self.QuantumNumber.values()), \
                f"Error: 'QuantumNumber' must be a dict of int."
        
        def _ptcl_check():
            assert len(self.particles) == self.dim * self.gen, \
                f"Error: there must be (dim * gen) number of particles"
            
            for p in self.particles:
                assert p.type == self.type, \
                f"Error: this field is a {self.type} field, but the particle is a {p.type} particle."

            assert all(p.charge == 0 for p in self.particles) or not self.self_conjugate, \
                f"Error: this field is self-conjugate, but the particles have non-zero charges."
            
            for p in self.particles:
                if p.type == "fermion":     
                    assert p.fermion.pass_all_checks(), \
                        f"Error: {p.fermion} does NOT pass ALL checks with a score of {p.fermion.score}."
                else:
                    assert p.pass_all_checks(), \
                        f"Error: {p} does NOT pass ALL checks with a score of {p.score}."
        
        # ------------------------------------------------------------------
        # Generation Index 
        def _create_generation_index():
            if self.gen > 1:
                gen_idx_name = num2words(self.gen).capitalize() + "Gen"
                self.gen_idx = Index(gen_idx_name, self.gen, "Fold")
                self.indices.append(self.gen_idx)

        # Generation Type Consistency
        def _check_gen_type_consistency():
            if self.type != "fermion":
                assert self.gen == 1, \
                    f"Error: Non-fermion fields can have only one generation"

        # Assign reps to the groups
        def _check_reps():
            for key, group in self.groups.items():
                rep_dict = {}
                if group.abelian:
                    rep_dict["reps"] = self.reps[key]
                    rep_dict["dim"] = 1
                    rep_dict["group"] = group
                    rep_dict["isColor"] = False
                    rep_dict["abelian"] = True
                    self.abelian_charges[group.charge] = self.reps[key]
                else:
                    rep_dict["reps"] = self.reps[key]
                    rep_dict["dim"] = group.rep_list[self.reps[key]]
                    rep_dict["group"] = group
                    rep_dict["isColor"] = group.isSU3C
                    rep_dict["abelian"] = False
                    rep_idx = self._index(group, rep_dict["dim"])
                    if rep_idx is not None:
                        group.reps.append(str(rep_idx))
                        self.indices.append(rep_idx)
                self.full_reps[group.id] = rep_dict

            # Representation-Dimension Consistency
            allow_dim = [1]
            for _, rep_dict in self.full_reps.items():
                if rep_dict["abelian"]:
                    pass
                elif rep_dict["isColor"]:
                    self.color = rep_dict["group"].rep_list[rep_dict["reps"]]
                else:
                    allow_dim.append(rep_dict["dim"])
            
            allow_dim = list(set(allow_dim))
            assert self.dim in allow_dim, \
                f"Error: {self.name} with dim-{self.dim} is not in allowed dims: {allow_dim}"
            
            if self.dim != 1:
                allow_dim.remove(self.dim)

            assert len(allow_dim) == 1 and allow_dim[0] == 1, \
                f"Error: {self.name} with dim-{self.dim} is not in allowed dims: {allow_dim}"

        self.all_checks = [_id_check, 
                           _name_check, 
                           _type_check, 
                           _groups_check, 
                           _reps_check, 
                           _dim_check, 
                           _gen_check, 
                           _particles_check, 
                           _self_conjugate_check, 
                           _QuantumNumber_check, 
                           _ptcl_check, 
                           _create_generation_index, 
                           _check_gen_type_consistency, 
                           _check_reps, 
                           ]

    def __check__(self):
        """ Input checks for the field class. """
        self.checklist = {}
        self._all_checks()
        run_checks(self.all_checks, self.checklist)

    
    @staticmethod
    def _index(group, dim):
        if dim == 1:
            return None
        elif group.isSU3C and dim == 3:
            return Index("Colour", 3, "NoUnfold", color = True)
        elif group.isSU3C and dim == 8:
            return Index("Gluon", 8, "NoUnfold", color = True)
        else:
            idx_name = str(group.group).replace("(", "").replace(")", "") + num2tuple(int(dim))[0].upper()
            return Index(idx_name, int(dim), "Unfold", color = False)
        
    # def _all_validations(self):
    #     self.all_validations = [
    #         self._create_generation_index, 
    #         self._check_gen_type_consistency, 
    #         self._check_reps, 
    #         ]

    # def __validate__(self):
    #     self._all_validations()
    #     run_checks(self.all_validations, self.checklist, skip_check = True)

    @property
    def score(self):
        max_score = len(self.checklist)
        score = sum(1 for value in self.checklist.values() if value is True)
        return f"{score}/{max_score}"

    def pass_all_checks(self):
        return len(self.checklist) == sum(1 for value in self.checklist.values() if value is True)
        


# ====================================================================
#                          Fermion Field
# ====================================================================
class FermionField(Field):
    """
    Class for fermion fields.
    """
    def __init__(self, id, name, groups, reps, dim, gen, particles, self_conjugate, QuantumNumber, chirality):
        self.chirality = chirality
        self._unphy_fields = None
        self._phy_fields = None
        self.color = 1
        self.mass_type = None
        super().__init__(id, name, "fermion", groups, reps, dim, gen, particles, self_conjugate, QuantumNumber)
        
    def __dict__(self):
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "reps": self.reps,
            "groups": self.groups,
            "dim": self.dim,
            "gen": self.gen,
            "particles": self.particles,
            "self_conjugate": self.self_conjugate,
            "QuantumNumber": self.QuantumNumber,
            "chirality": self.chirality
        }

    def _all_checks(self):
        super()._all_checks()
        def _chirality_check():
            assert self.chirality in ["left", "right"], f"Error: {self.chirality} is not a valid chirality"

        def _assign_colors():
            for p in self.particles:
                p.fermion.color = self.color if self.color is not None else p.fermion.color

        def _sort_unphy_fields():
            self._unphy_fields = np.array(self.particles).reshape(self.gen, self.dim).tolist()
            if self.QuantumNumber.get('LeptonNumber', 0) != 0:
                for gen in self._unphy_fields:
                    flavor = [p.fermion.name for p in gen if p.charge != 0]
                    for p in gen:
                        p.fermion.flavor = flavor[0] if len(flavor) == 1 else None

        def _sort_phy_fields():
            self._phy_fields = np.array(self._unphy_fields).transpose().tolist()
            for idx, gen in enumerate(self._phy_fields):
                assert all(cf.charge == gen[0].charge for cf in gen), f"Error: {self.name} has inconsistent charges in generation {idx+1}"
        
        # This is the corrected list of initial checks.
        # The problematic _check_mass_type is NOT included here, and the duplicated block is gone.
        self.all_checks.extend([_chirality_check, _assign_colors, _sort_unphy_fields, _sort_phy_fields])

    def _check_mass_type(self):
        """ Checks if a field with massive particles has a mass type assigned. """
        is_massive = any(p.fermion.mass > 0 for p in self.particles)
        if is_massive:
            assert self.mass_type is not None, f"Error: {self.name} is massive but has no mass_type assigned from an interaction."

    # The rest of the methods (phy_field_info, etc.) are correct but omitted here for brevity.
    # Please ensure your file has the rest of the FermionField class methods as they were.
    @property
    def phy_field_info(self):
        assert self.pass_all_checks(), f"Error: {self.name} has failed the checks"
        def _class_name(idx):
            charge = self._phy_fields[idx][0].charge
            color = self.color
            return fermion_field_name(charge, color)[0]
        def _class_members(idx):
            member_names = [p.fermion.name for p in self._phy_fields[idx]]
            return self.rewrite(member_names)
        def _indices():
            indices = [self.gen_idx] + [i for i in self.indices if i.color == True]
            indices = [repr(i) for i in indices]
            return "{" + ", ".join(indices) + "}"
        def _flavor_index():
            return self.gen_idx
        def _mass(idx):
            class_name = _class_name(idx)
            particle_list = self._phy_fields[idx]
            if all(particle.fermion.mass == 0 for particle in particle_list): return 0
            else:
                Mass = ["M"+str(class_name).upper()]
                for particle in particle_list:
                    mass_name = "M" + particle.fermion.name.upper()
                    Mass.append([mass_name, particle.fermion.mass])
                return self.rewrite(Mass)
        def _width(idx):
            class_name = _class_name(idx)
            particle_list = self._phy_fields[idx]
            if all(particle.fermion.width == 0 for particle in particle_list): return 0
            else:
                Width = ["W"+str(class_name).upper()]
                for particle in particle_list:
                    width_name = "W" + particle.fermion.name.upper()
                    Width.append([width_name, particle.fermion.width])
                return self.rewrite(Width)
        def _quantum_number(idx):
            qnumber = self.QuantumNumber.copy()
            qnumber.update({"Q": self._phy_fields[idx][0].charge})
            qnumber = {key: str(Fraction(value,3)) for key, value in qnumber.items() if value != 0}
            return str(qnumber).replace(":", " ->").replace("'", "")
        def _propagator_label(idx):
            class_name = _class_name(idx)
            member_name = [p.fermion.name for p in self._phy_fields[idx]]
            prop_label_str = [class_name] + member_name
            return self.rewrite(prop_label_str, quote=True)
        def _PDG(idx):
            PDG_list = [p.fermion.pdg_id for p in self._phy_fields[idx]]
            return self.rewrite(PDG_list, quote=False)
        def _particle_name(idx):
            particle_name_list = [p.fermion.name for p in self._phy_fields[idx]]
            return self.rewrite(particle_name_list, quote=True)
        def _anti_particle_name(idx):
            anti_particle_name_list = [p.fermion.name + "~" for p in self._phy_fields[idx]]
            return self.rewrite(anti_particle_name_list, quote=True)
        def _particle_fullname(idx):
            particle_fullname_list = [p.fermion.full_name for p in self._phy_fields[idx]]
            return self.rewrite(particle_fullname_list, quote=True)
        phy_fields = []
        for i in range(self.dim):
            phy_field = {
                "ClassName": _class_name(i), "ClassMembers": _class_members(i), "Indices": _indices(),
                "FlavorIndex": _flavor_index(), "SelfConjugate": self.self_conjugate, "Mass": _mass(i),
                "Width": _width(i), "QuantumNumber": _quantum_number(i), "PropagatorLabel": _propagator_label(i),
                "PropagatorType": "Straight", "PropagatorArrow": "Forward", "PDG": _PDG(i),
                "ParticleName": _particle_name(i), "AntiParticleName": _anti_particle_name(i),
                "FullName": _particle_fullname(i)
            }
            phy_fields.append(phy_field)
        return phy_fields

    @property
    def unphy_field_info(self):
        assert self.pass_all_checks(), f"Error: {self.name} has failed the checks"
        def _class_name(): return self.name
        def _indices():
            indices = [repr(i) for i in self.indices]
            return "{" + ", ".join(indices) + "}"
        def _flavor_index(): return self.gen_idx
        def _quantum_number():
            qnumber = self.abelian_charges
            qnumber = {key: str(Fraction(value,3)) for key, value in qnumber.items() if value != 0}
            return str(qnumber).replace(":", " ->").replace("'", "")
        def _definition():
            definition = []
            proj_matrix = "ProjM" if self.chirality == "left" else "ProjP"
            if self.color > 1:
                if self.dim > 1:
                    for i in range(self.dim):
                        class_name = fermion_field_name(self._phy_fields[i][0].charge, self.color)[0]
                        definition.append(f"{self.name}[sp1_, {i+1}, ff_, cc_] :> Module[{{sp2}}, {proj_matrix}[sp1, sp2] {class_name}[sp2, ff, cc]]")
                else:
                    class_name = fermion_field_name(self._phy_fields[0][0].charge, self.color)[0]
                    definition.append(f"{self.name}[sp1_, ff_, cc_] :> Module[{{sp2}}, {proj_matrix}[sp1, sp2] {class_name}[sp2, ff, cc]]")
            else:
                for i in range(self.dim):
                    class_name = fermion_field_name(self._phy_fields[i][0].charge, self.color)[0]
                    definition.append(f"{self.name}[sp1_, {i+1}, ff_] :> Module[{{sp2}}, {proj_matrix}[sp1, sp2] {class_name}[sp2, ff]]")
            return "{ " + (",\n" + " "*30).join(definition) + " }"
        unphy_field = {
            "ClassName": _class_name(), "Unphysical": True, "Indices": _indices(),
            "FlavorIndex": _flavor_index(), "SelfConjugate": self.self_conjugate,
            "QuantumNumber": _quantum_number(), "Definition": _definition()
        }
        return unphy_field

    @staticmethod
    def rewrite(list, quote = False):
        list = str(list).replace("[", "{").replace("]", "}")
        if quote: list = str(list).replace("'", "\"")
        else: list = str(list).replace("'", "")
        return list
    


# ====================================================================
#                            Scalar Field
# ====================================================================
class ScalarField(Field):
    def __init__(self, id, name, type, groups, reps, dim, gen, particles, self_conjugate, QuantumNumber):
        super().__init__(id, name, type, groups, reps, dim, gen, particles, self_conjugate, QuantumNumber)
        self.potential = None
        self.vev = None
        self.get_vev = False

    def _check_vev_potential(self):
        """ Check the potential allowed the vev """
        assert True, \
            f"{self.vev} does not minimize {self.name}'s potential"

    def acquire_vev(self, vev):
        pass

    def _all_validations(self):
        super()._all_validations()
        self.all_validations.extend([self._check_vev_potential])

    def phy_field_info(self):
        pass

    def unphy_field_info(self):
        pass



# ====================================================================
#                            Vector Fields
# ====================================================================
class VectorField(Field):
    def __init__(self, id, name, groups, reps, dim, gen, particles, self_conjugate, QuantumNumber):
        super().__init__(id, name, "vector", groups, reps, dim, gen, particles, self_conjugate, QuantumNumber)
        self.__check__()

    def __check__(self):
        super().__check__()

### =============================== ###
###          Group Classes          ###
### =============================== ###
class Group:
    """
    Base class for groups
    """
    def __init__(self, name, dim):
        self.name = str(name)
        self.dim = dim
        self.__check__()

    def __str__(self):
        return self.name
    
    def _all_checks(self):
        self.all_checks = []

    @staticmethod
    def run_checks(all_checks, checklist, skip_check = False):
        for check in all_checks:
            fail_previous_check = any(isinstance(value, Exception) or value == False for value in checklist.values())
            if skip_check and fail_previous_check:
                checklist[check.__name__] = Exception(f"Skipped due to previous check failure")
            else:
                try:
                    check()
                    checklist[check.__name__] = True
                except Exception as e:
                    checklist[check.__name__] = e

    def __check__(self):
        self.checklist = {}
        self._all_checks()
        self.run_checks(self.all_checks, self.checklist)

    def _all_validations(self):
        self.all_validations = []

    def __validate__(self):
        self._all_validations()
        self.run_checks(self.all_validations, self.checklist, skip_check = True)
    
    def reps(self):
        pass

    def irreps(self):
        pass

    def generators(self):
        pass


class U(Group):
    """
    Unitary Group (only for U(1))
    """
    def __init__(self, dim):
        super().__init__(f"U({dim})", dim)
        self.type = "U"
        self.abelian = True
        self.__check__()

    def _all_checks(self):
        super()._all_checks()

        def _dim_check():
            assert isinstance(self.dim, int) and self.dim == 1, \
                f"Error: Dimension must be 1"

        self.all_checks.append(_dim_check)

    @property
    def fnd_rep(self):
        return self.dim
    
    @property
    def adj_rep(self):
        return self.dim**2
    
    @property
    def rep_list(self):
        return {"singlet": 1,
                "fnd": self.fnd_rep, 
                "adj": self.adj_rep}
    
class SU(Group):
    """
    Special Unitary Group (only for SU(N))
    """
    def __init__(self, dim):
        super().__init__(f"SU({dim})", dim)
        self.type = "SU"
        self.abelian = False
        self.__check__()

    def _all_checks(self):
        super()._all_checks()

        def _dim_check():
            assert isinstance(self.dim, int) and self.dim > 1 and self.dim <= 3, \
                f"Error: Dimension must be an integer between 2 and 3"

        self.all_checks.append(_dim_check)

    @property
    def fnd_rep(self):
        return self.dim
    
    @property
    def adj_rep(self):
        return self.dim**2 - 1
    
    @property
    def rep_list(self):
        return {"singlet": 1,
                "fnd": self.fnd_rep, 
                "adj": self.adj_rep}
    
    @property
    def structure_constants(self):
        if self.dim == 2:
            return "Eps"
        elif self.dim == 3:
            return "f"
        else:
            return None
        
    @property
    def definition(self):
        if self.dim == 2:
            return "{Ta[a_,b_,c_]->PauliSigma[a,b,c]/2, FSU2L[i_,j_,k_]:> I Eps[i,j,k]}"
        else:
            return None
        
    @property
    def sym_tensors(self):
        if self.dim == 3:
            return "dSUN"
        else:
            return None
    
class GaugeGroup:
    """
    Base class for gauge groups
    """
    def __init__(self, id, name, charge, group, coupling, boson):
        self.id = id
        self.name = str(name)
        self.charge = charge
        self.group = group
        self.coupling = coupling
        self.boson = boson
        self.__check__()

        if self.group == "SU_2":
            self.definition = "{Ta[a_,b_,c_]->PauliSigma[a,b,c]/2, FSU2L[i_,j_,k_]:> I Eps[i,j,k]}"
            self.reps = ["Ta"]
            self.sym_tensors = None

        elif self.group == "SU_3":
            self.definition = None
            self.sym_tensors = "dSUN"
            self.reps = ["T"]
        else:
            self.definition = None
            self.sym_tensors = None
            self.reps = None
            
        self._check_in_SM()
        self.define_group()
        self.checklist.update(self.group.checklist)
        
        try:
            self.abelian = self.group.abelian
        except:
            self.abelian = False

        try:
            self.fnd_rep = self.group.fnd_rep
        except:
            self.fnd_rep = -1

        try:
            self.adj_rep = self.group.adj_rep
        except:
            self.adj_rep = -1

        try:
            self.rep_list = self.group.rep_list
        except:
            self.rep_list = {"fnd": -1, "adj": -1}

        try:
            self.structure_constants = self.group.structure_constants
        except:
            self.structure_constants = None

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name 

    def _all_checks(self):
        self.all_checks = []

    def _check_in_SM(self):
        if self.name == "SU3C" and self.group == "SU_3" and self.boson == "G":
            self.isSU3C = True
        else:
            self.isSU3C = False
        if self.name == "SU2L" and self.group == "SU_2" and self.boson == "W":
            self.isSU2L = True
        else:
            self.isSU2L = False
        if self.name == "U1Y" and self.group == "U_1" and self.boson == "B":
            self.isU1Y = True
        else:
            self.isU1Y = False

    def define_group(self):
        try:
            type, N = self.group.split("_")
            N = int(N)
        except:
            N = -1
            type = "U"

        if type == "U":
            self.group = U(N)
        elif type == "SU":
            self.group = SU(N)
        else:
            self.group = U(-1)

    def __check__(self):
        id_check = isinstance(self.id, str)
        name_check = isinstance(self.name, str)
        charge_check = isinstance(self.charge, str)
        group_check = isinstance(self.group, str) and len(self.group.split("_")) == 2
        coupling_check = isinstance(self.coupling, str)
        boson_check = isinstance(self.boson, str)

        try:
            group_type_check = self.group.split("_")[0] in ["U", "SU", "O", "SO"]
            N_check = self.group.split("_")[1].isdigit()
        except:
            group_type_check = False
            N_check = False

        self.checklist = {"id": id_check, 
                          "name": name_check, 
                          "charge": charge_check, 
                          "group": group_check, 
                          "group_type": group_type_check,
                          "N": N_check,
                          "coupling": coupling_check, 
                          "boson": boson_check}

    @property
    def score(self):
        max_score = len(self.checklist)
        score = sum(1 for value in self.checklist.values() if value is True)
        return f"{score}/{max_score}"

    def pass_all_checks(self):
        return len(self.checklist) == sum(1 for value in self.checklist.values() if value is True)

    @staticmethod
    def write_reps(reps):
        return str(list(dict.fromkeys(reps))).replace("'", "").replace("[", "{").replace("]", "}")

    def gauge_group_info(self):
        gg_info = {
            "Abelian": self.abelian,
            "CouplingConstant": self.coupling,
            "GaugeBoson": self.boson
        }
        if self.abelian:
            gg_info["Charge"] = self.charge
        else:
            gg_info["StructureConstant"] = self.structure_constants
            gg_info["Representations"] = self.write_reps(self.reps)
            gg_info["SymmetricTensor"] = self.sym_tensors
            gg_info["Definitions"] = self.definition
        return gg_info

### =============================== ###
###          Index Classes          ###
### =============================== ###
class Index:
    """
    Index class for indices of fields.
    name: str
    dim: int
    fold: str
    color: bool
    """
    def __init__(self, name, dim, fold, color = False):
        self.name = name
        self.dim = dim
        self.fold = fold
        self.color = color

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Index[{self.name}]"
    
    def IndexRange(self):
        if self.fold != "Fold":
            idx_range = f"IndexRange[Index[{self.name:8}]] = {self.fold}[Range[{self.dim}]]"
        else:
            idx_range = f"IndexRange[Index[{self.name:8}]] = Range[{self.dim}]"
        return idx_range
    
    def IndexStyle(self, style):
        name = self.name + ','
        return f"IndexStyle[{name:10} {style}]"


import numpy as np
# ====================================================================
#                              Interaction
# ====================================================================
class Interaction:
    """
    Interaction class for interactions.
    id: str
    type: str
    requirements: dict
    fields: list
    """
    def __init__(self, id, type, requirements, fields):
        self.id = id
        self.type = type
        self.requirements = requirements
        self.fields = fields
        self.ExtParams = []
        self.IntParams = []
        self.sorted_fields = {}
        self.__check__()

    def __str__(self):
        return f"{self.type} ({self.id})"

    def __repr__(self):
        return f"{self.type} ({self.id})"

    def _all_checks(self):
        """ All checks for the initial INPUTs of the 'Interaction' class. """
        self.all_checks = []

        def _id_check():
            assert isinstance(self.id, str), f"AssertionError: {self.id} is not a string"

        def _field_length_check():
            assert len(self.fields) == len(self.requirements), \
                f"AssertionError: {self.id} has {len(self.fields)} fields but {len(self.requirements)} requirements"

        def _field_type_check():
            assert all(field_type in self.requirements for field_type in self.requirements), \
                f"AssertionError: {self.id} has invalid field types"

        def _field_check():
            for f in self.fields:
                assert isinstance(f, Field), \
                    f"AssertionError: {f} is not a Field"

        def _all_field_pass_checks():
            for f in self.fields:
                assert f.pass_all_checks(), \
                    f"AssertionError: {f} has failed checks"

        def _sort_field():
            self.sorted_fields = {}
            for pos, reqs in self.requirements.items():
                candidate = self.fields
                for key, value in reqs.items():
                    candidate = [f for f in candidate if f.__dict__()[key] == value or f.__dict__()[key] in value]
                assert len(candidate) == 1, \
                    f"AssertionError: Multiple fields with \"{key} = {value}\" found for {self.id}: {candidate}"
                self.sorted_fields[pos] = candidate[0]

        self.all_checks.extend([_id_check, 
                                _field_length_check, 
                                _field_type_check, 
                                _field_check, 
                                _all_field_pass_checks, 
                                _sort_field])
    
    @staticmethod
    def run_checks(all_checks, checklist, skip_check = False):
        for check in all_checks:
            fail_previous_check = any(isinstance(value, Exception) or value == False for value in checklist.values())
            if skip_check and fail_previous_check:
                checklist[check.__name__] = Exception(f"Skipped due to previous check failure")
            else:
                try:
                    check()
                    checklist[check.__name__] = True
                except Exception as e:
                    checklist[check.__name__] = e

    def __check__(self):
        self.checklist = {}
        self._all_checks()
        self.run_checks(self.all_checks, self.checklist)

    def _all_validations(self):
        """ All validations for the 'Interaction' class. """
        self.all_validations = []

    def __validate__(self):
        self._all_validations()
        self.run_checks(self.all_validations, self.checklist, skip_check = True)

    @property
    def score(self):
        max_score = len(self.checklist)
        score = sum(1 for value in self.checklist.values() if value is True)
        return f"{score}/{max_score}"

    def pass_all_checks(self):
        return all(self.checklist.values()) 



# ====================================================================
#                              Yukawa
# ====================================================================
class Yukawa(Interaction):
    """
    Yukawa interaction class.
    """

    field1 = {"type": "fermion", 
              "chirality": "left"}
    field2 = {"type": "fermion", 
              "chirality": "right"}
    field3 = {"type": ["complex", "real", "scalar"]}

    field_types = {0: field1, 1: field2, 2: field3}

    def __init__(self, id, fields):
        super().__init__(id, "Yukawa", self.field_types, fields)
        self.dummy_idx = []
        self.higgs_loc = None

    def _all_checks(self):
        super()._all_checks()

        def _gen_check():
            assert self.sorted_fields[0].gen == self.sorted_fields[1].gen,\
                f"AssertionError: {self.sorted_fields[0].gen} != {self.sorted_fields[1].gen}"

        def _dim_check():
            assert (self.sorted_fields[0].dim == self.sorted_fields[2].dim) or (self.sorted_fields[1].dim == self.sorted_fields[2].dim), f"AssertionError: {self.sorted_fields[0].dim} != {self.sorted_fields[2].dim} and {self.sorted_fields[1].dim} != {self.sorted_fields[2].dim}"
        
        # We remove the assignment from the checks list. It will be called explicitly by the Model.
        self.all_checks.extend([_gen_check, 
                                _dim_check])

    def assign_mass_type_to_fields(self):
        """Assigns the mass type to the constituent fermion fields."""
        # This method depends on self.sorted_fields, which is created during the initial __check__
        self.sorted_fields[0].mass_type = "Yukawa"
        self.sorted_fields[1].mass_type = "Yukawa"

    def _yukawa_mass(self):
        left = np.array([[p.id for p in gen] for gen in self.sorted_fields[0]._unphy_fields]).transpose()
        right = np.array([[p.id for p in gen] for gen in self.sorted_fields[1]._unphy_fields])

        rows_left, cols_left = left.shape
        rows_right, cols_right = right.shape

        # Matrix Multiplication of the Left-Handed and Right-Handed Fermions
        result = [[f"{left[i][k].replace('_L', '')}*{right[k][j].replace('_R', '')}" 
                   for k in range(cols_left)] 
                  for j in range(cols_right) 
                 for i in range(rows_left)]
        
        particle_ids = []
        for idx, row in enumerate(result):
            correct_term = all([col.split("*")[0] == col.split("*")[1] for col in row])   
            if correct_term: 
                particle_ids = [col.split("*")[0] for col in row]
                self.higgs_loc = idx
                break

        particle_ids = list(set(particle_ids))
        assert len(particle_ids) == self.sorted_fields[0].gen, \
            f"AssertionError: {self.id} has incompatible dimensions"

        all_particles = [f.fermion for f in self.sorted_fields[1].particles if f.fermion.id in particle_ids]
        for particle in all_particles:
            mass_param = ExtParam(name = "ym" + particle.name, 
                                  BLOCKNAME = "YUKAWA", 
                                  OrderBlock = particle.pdg_id, 
                                  Value = particle.mass, 
                                  Description = f"\"Yukawa mass for {particle.full_name}\"")
            self.ExtParams.append(mass_param)

        # Generate the Yukawa matrix
        suffix = self.sorted_fields[0].name + self.sorted_fields[1].name
        name = "y" + suffix
        Indices = f"{{{repr(self.sorted_fields[0].gen_idx)}, {repr(self.sorted_fields[1].gen_idx)}}}"
        Definitions = f"{{{name}[i_?NumericQ, j_?NumericQ] :> 0  /; (i =!= j)}}"
        mass_name = [p.name for p in self.ExtParams]
        Value = f"{{{name}[1,1] -> Sqrt[2] {mass_name[0]}/vev, {name}[2,2] -> Sqrt[2] {mass_name[1]}/vev, {name}[3,3] -> Sqrt[2] {mass_name[2]}/vev}}"
        InteractionOrder = "{QED, 1}"
        
        ParameterName = f"{{{name}[1,1] -> {mass_name[0]}, {name}[2,2] -> {mass_name[1]}, {name}[3,3] -> {mass_name[2]}}}"
        Tex = f"Superscript[y, {suffix}]"
        Description = f"\"Yukawa coupling for {self.sorted_fields[0].name} and {self.sorted_fields[1].name}\""
        
        self.yukawa_matrix = IntParam(name, 
                                      Indices, 
                                      Definitions, 
                                      Value, 
                                      InteractionOrder, 
                                      ParameterName, 
                                      Tex, 
                                      Description)
        
        self.IntParams.append(self.yukawa_matrix)

    def _all_validations(self):
        super()._all_validations()
        self.all_validations.extend([self._yukawa_mass])

    def to_fr(self):        
        
        suffix = self.sorted_fields[0].name + self.sorted_fields[1].name
        ym = "y" + suffix
        ff = generate_dummy_idx()
        self.dummy_idx.extend([f"{ff}1", f"{ff}2"])
        color_idx = ", cc" if self.sorted_fields[0].color > 1 and self.sorted_fields[1].color > 1 else ""
        if self.higgs_loc == 1:
            self.dummy_idx.extend(["ii", "cc"])
            return f"    - {ym}[{ff}1, {ff}2] QLbar[sp, ii, {ff}1{color_idx}].uR [sp, {ff}2{color_idx}] Phi[ii]"
        elif self.higgs_loc == 0:
            self.dummy_idx.extend(["ii", "jj", "cc"])
            return f"    - {ym}[{ff}1, {ff}2] QLbar[sp, ii, {ff}1{color_idx}].dR [sp, {ff}2{color_idx}] Phibar[jj] Eps[ii, jj]"
        

### ========================================================================== ###
###                                                                            ###
###                             Model Class                                    ###
###                                                                            ###
### ========================================================================== ###

# Standard library imports
import os
from datetime import datetime

# ====================================================================
#                            Model
# ====================================================================
class Model:
    """
    Particle Physics Model Class: read JSON file and write FR model file
    """
    def __init__(self, model_name, author, model_data, OUTPUT_PATH):
        self.model_name = model_name
        self.model_symbol = ''.join(word[0].upper() for word in self.model_name.split() if word)
        self.author = author
        self.ai = True if author == 'Bohr Network' else False
        self.FeynmanGauge = True
        
        self.OUTPUT_PATH = OUTPUT_PATH
        self.current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.checklist = {}
        self.indices = {}
        self._read_model(model_data)

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
        self.gauge_groups = {}
        for g in model_data['GaugeGroups']:
            self.gauge_groups[g["id"]] = GaugeGroup(**g)

    def _read_vev(self, model_data):
        self.vevs = {}
        for v in model_data['vevs']:
            self.vevs[v["id"]] = vev(**v)

    def _read_particles(self, model_data):
        self.scalar_particles = {}
        self.fermion_particles = {}
        self.vector_particles = {}
        
        for p in model_data['particles']:
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
        self.particles = {**self.scalar_particles, **self.fermion_particles, **self.vector_particles}

    def _read_scalar_fields(self, model_data):
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
                new_scalar_field = ScalarField(**sf)
                #new_scalar_field.__validate__()
                self.scalar_fields[sf["id"]] = new_scalar_field
                for idx in new_scalar_field.indices:
                    if idx.name not in self.indices.keys():
                        self.indices[idx.name] = idx

    def _read_vector_fields(self, model_data):
        self.vector_fields = {}

        self.indices["SU2W"] = Index("SU2W", 3, "Unfold")
        self.indices["Gluon"] = Index("Gluon", 8, "NoUnfold", color=True)
        
    def _read_fermion_fields(self, model_data):
        self.fermion_fields = {}

        # we separate all fermion particles into left and right chiral fermions
        chiral_fermions = {**{f"{id}_left": ptcl.left for id, ptcl in self.fermion_particles.items()},
                           **{f"{id}_right": ptcl.right for id, ptcl in self.fermion_particles.items()}}
        
        for f in model_data["fields"]:
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
                #new_fermion_field.__validate__()
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
            self.interactions = {}
            for itr in model_data.get("interactions", []):
                try:
                    kwargs = {k: v for k, v in itr.items() if k != 'type'}
                    field_ids = kwargs.get("fields", [])
                    fields = [self.fields[id] for id in field_ids]
                    kwargs["fields"] = fields
                    new_interaction = Yukawa(**kwargs)
                    new_interaction.__validate__()
                    self.interactions[itr["id"]] = new_interaction
                except Exception as e:
                    print(f"CRITICAL ERROR: Failed to process interaction '{itr.get('id', 'N/A')}'. Reason: {e}")

            # --- DEBUGGING AND VALIDATION LOGIC ---
            # print("\n[DEBUG] --- STARTING MODEL VALIDATION ---")

            # print("[DEBUG] Field mass_types BEFORE assignment:")
            # for field in self.fields.values():
            #     if isinstance(field, FermionField):
            #         print(f"[DEBUG] \t{field.name}: {field.mass_type}")

            # After all interactions are created, have them configure the fields.
            #print("\n[DEBUG] Assigning mass_types from interactions...")
            for interaction in self.interactions.values():
                if hasattr(interaction, 'assign_mass_type_to_fields'):
                    f1_name = interaction.sorted_fields[0].name
                    f2_name = interaction.sorted_fields[1].name
                    # print(f"[DEBUG] \tInteraction '{interaction.id}' is assigning mass_type to '{f1_name}' and '{f2_name}'")
                    interaction.assign_mass_type_to_fields()

            # print("\n[DEBUG] Field mass_types AFTER assignment:")
            # for field in self.fields.values():
            #     if isinstance(field, FermionField):
            #         print(f"[DEBUG] \t{field.name}: {field.mass_type}")

            # Now, run the deferred checks on the fields now that they are configured.
            # print("\n[DEBUG] Running deferred checks...")
            for field in self.fields.values():
                if isinstance(field, FermionField):
                    try:
                        field._check_mass_type()
                        field.checklist['_check_mass_type'] = True
                    except Exception as e:
                        field.checklist['_check_mass_type'] = e
            # print("[DEBUG] --- FINISHED MODEL VALIDATION ---\n")

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
        self.score = f"{model_score}/{max_score}"

    def pass_all_checks(self):
        score, max_score = map(int, self.score.split("/"))
        return score == max_score

    def _read_model(self, model_data):
        self._read_gauge_group(model_data)
        self._read_particles(model_data)
        self._read_fields(model_data)
        self._read_interactions(model_data)
        self._read_check_list()

    def _validate_model_consistency(self):
        for interaction in self.interactions.values():
            if isinstance(interaction, Yukawa):
                interaction.assign_mass_type_to_fields()
        for field in self.fields.values():
            if isinstance(field, FermionField):
                try:
                    field._check_mass_type()
                    field.checklist['_check_mass_type'] = True
                except Exception as e:
                    field.checklist['_check_mass_type'] = e


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
            style = generate_ijk()
            file.write(f"{idx.IndexStyle(style)};\n")
        file.write("\n")

    def write_Interaction_orders(self, file):
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
        write_sm_higgs(file)

    def write_vector(self, file):
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
        file.write("(* ************************** *)\n")
        file.write("(* *****   Lagrangian   ***** *)\n")
        file.write("(* ************************** *)\n")
        file.write("\n")
        self.write_yukawa(file)
        write_sm_lagrangian(file)

    def _write_feynrules_file(self, output_dir):
        
        model_file = os.path.join(output_dir, f"{self.model_symbol}.fr")
        particle_file = os.path.join(output_dir, f"{self.model_symbol}_particles.fr")
        parameter_file = os.path.join(output_dir, f"{self.model_symbol}_parameters.fr")
        lagrangian_file = os.path.join(output_dir, f"{self.model_symbol}_lagrangian.fr")

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
            f.write(f"Get[\"{self.model_symbol}_particles.fr\"];\n")
            f.write(f"Get[\"{self.model_symbol}_parameters.fr\"];\n")
            f.write(f"Get[\"{self.model_symbol}_lagrangian.fr\"];\n")
            f.write("\n")
            with open(particle_file, "w") as f: 
                self.write_fields(f)
                self.write_FeynArts(f)
            with open(parameter_file, "w") as f:
                self.write_parameters(f)
            with open(lagrangian_file, "w") as f:
                self.write_lagrangian(f)

    def write_checklist(self, output_dir):
        with open(os.path.join(output_dir, "checklist.csv"), "w") as f:
            f.write("id, check, result\n")
            for id, checklist in self.checklist.items():
                for key, value in checklist.items():
                    f.write(f"{id}, {key}, {value}\n")

    def to_fr(self):
        # make output directory
        os.makedirs(self.OUTPUT_PATH, exist_ok=True)
        model_dir_name = self.model_symbol + "_" + self.current_time  
        self.output_dir = os.path.join(self.OUTPUT_PATH, model_dir_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # write checklist
        self.write_checklist(self.output_dir)
        
        # write feynrules files
        if self.pass_all_checks():
            self._write_feynrules_file(self.output_dir)

            print(f"{self.model_name} ({self.model_symbol}) get score {self.score}.")
            print(f"{self.model_name} ({self.model_symbol}) passed all checks!")
            print(f"FeynRules file is written to {self.output_dir}")
        else:
            print(f"{self.model_name} ({self.model_symbol}) get score {self.score}.")
            print(f"Please check the checklist.log for more details.")
            return None

    def run_mathematica_checks(self):
        import subprocess
        if self.pass_all_checks():
            return None 
        if self.output_dir is None:
            print("Please run to_fr() first to output a FeynRules file.")
            return None

        FEYNRULES_PATH = "/oscar/home/qniu3/physics/FeynRules"
        load_process = subprocess.run(
            ["/bin/bash", "-c", "module load mathematica"],
            capture_output=True,
            text=True
        )
        
        if load_process.returncode != 0:
            print("Failed to load Mathematica module:")
            print(load_process.stderr)
            return None

@count_calls
def generate_id():
    return f"id-{generate_id.call_count}"

@count_calls
def generate_dummy_idx():
    return f"{num2iii(generate_dummy_idx.call_count)}"

@count_calls
def generate_ijk():
    return f"{num2ijk(generate_ijk.call_count)}"

def num2ijk(num):
    if num <= 0:
        return ""
    num = num + 9
    letter = chr(ord('a') + (num - 1) % 26)
    repeat = (num - 1) // 26 + 1
    
    return letter * repeat

def num2iii(num):
    """
    Convert a number to a sequence of letters where:
    """
    if num <= 0:
        return ""
    num = num + 26
    letter = chr(ord('a') + (num - 1) % 26)
    repeat = (num - 1) // 26 + 1
    
    return letter * repeat

def num2words(num):
    """
    Convert a number to its English words.
    Only handles numbers less than or equal to 100.
    """
    if num < 0:
        return "minus " + num2words(abs(num))
    
    ones = {
        0: 'zero',
        1: 'one',
        2: 'two',
        3: 'three',
        4: 'four',
        5: 'five',
        6: 'six',
        7: 'seven',
        8: 'eight',
        9: 'nine',
        10: 'ten',
        11: 'eleven',
        12: 'twelve',
        13: 'thirteen',
        14: 'fourteen',
        15: 'fifteen',
        16: 'sixteen',
        17: 'seventeen',
        18: 'eighteen',
        19: 'nineteen'
    }
    
    tens = {
        2: 'twenty',
        3: 'thirty',
        4: 'forty',
        5: 'fifty',
        6: 'sixty',
        7: 'seventy',
        8: 'eighty',
        9: 'ninety'
    }
    if num > 100:
        return num2words(num % 100)
    elif num < 20:
        return ones.get(num, '')
    elif num < 100:
        return tens.get(num // 10, '') + ('-' + ones.get(num % 10, '') if num % 10 != 0 else '')
    else:
        return "one hundred"

def num2abc(number):
    """
    Convert a number to Excel-style column letters.
    """
    if number <= 0:
        raise ValueError("Number must be positive")
    
    result = ""
    while number > 0:
        remainder = (number - 1) % 26
        result = chr(remainder + 97) + result  # 97 is ASCII for 'a'
        number = (number - 1) // 26
    
    return result

def num2greek(number):
    """
    Convert a number to its corresponding Greek letter name.
    Examples: 1→alpha, 2→beta, etc.
    """
    if number <= 0:
        number = abs(number)
    
    # Modulo to handle numbers greater than 24
    number = ((number - 1) % 24) + 1
    
    greek_letters = {
        1: 'alpha', 
        2: 'beta', 
        3: 'gamma', 
        4: 'delta',
        5: 'epsilon',
        6: 'zeta',
        7: 'eta',
        8: 'theta',
        9: 'iota',
        10: 'kappa',
        11: 'lambda',
        12: 'mu',
        13: 'nu',
        14: 'xi',
        15: 'omicron',
        16: 'pi',
        17: 'rho',
        18: 'sigma',
        19: 'tau',
        20: 'upsilon',
        21: 'phi',
        22: 'chi',
        23: 'psi',
        24: 'omega'
    }
    
    return greek_letters[number]

def num2tuple(number):
    """
    Convert a number to its corresponding tuple name (singlet, doublet, triplet, etc.)
    """
    if number <= 0:
        number = abs(number)
    
    tuple_names = {
        1: 'singlet',
        2: 'doublet',
        3: 'triplet',
        4: 'quartet',
        5: 'quintet',
        6: 'sextet',
        7: 'septet',
        8: 'octet',
        9: 'nonet',
        10: 'decuplet'
    }
    
    if number in tuple_names:
        return tuple_names[number]
    else:
        return f"{num2words(number)}-plet"


def fermion_field_name(charge, color):
    if color == 1:
        if charge == 0:
            return "vl", "neutrino"
        elif charge == -3:
            return "l", "charged lepton"
        else:
            symbol = f"{num2words(charge)} charged lepton"
            return f"{symbol[0]}l", symbol
    else:
        if charge == 2:
            return "uq", "up quark"
        elif charge == -1:
            return "dq", "down quark"
        else:
            symbol = f"{num2words(charge)} quark"
            return f"{symbol[0]}q", symbol
        


### ============================ ###
###       External Parameter     ###
### ============================ ###
class ExtParam:
    def __init__(self, name, BLOCKNAME, OrderBlock, Value, Description):
        self.name = name
        self.BLOCKNAME = BLOCKNAME
        self.OrderBlock = OrderBlock
        self.Value = Value
        self.Description = Description

    def __str__(self):
        return f"{self.BLOCKNAME} {self.OrderBlock} {self.Value} {self.Description}"

    def __dict__(self):
        return {
            "ParameterType": "External",
            "BLOCKNAME": self.BLOCKNAME,
            "OrderBlock": self.OrderBlock,
            "Value": self.Value,
            "Description": self.Description
        }

    def to_fr(self):
        param_entry = f"  {self.name} == {{\n"
        param_info = []
        for key, value in self.__dict__().items():
            param_info.append(f"{key:20} -> {value}")
        param_entry += "    " + ",\n    ".join(param_info) + "\n"
        param_entry += "  },\n"
        return param_entry


### ============================= ###
###       Internal Parameter      ###
### ============================= ###
class IntParam:
    def __init__(self, name, indices, definition, value, InteractionOrder, ParameterName, TeX, Description):
        self.name = name
        self.indices = indices
        self.definition = definition
        self.value = value
        self.InteractionOrder = InteractionOrder
        self.ParameterName = ParameterName
        self.TeX = TeX
        self.Description = Description

    def __str__(self):
        return f"{self.name} = {self.definition} {self.value} {self.InteractionOrder} {self.ParameterName} {self.TeX} {self.Description}"

    def __dict__(self):
        return {
            "ParameterType": "Internal",
            "Indices": self.indices,
            "Definition": self.definition,
            "Value": self.value,
            "InteractionOrder": self.InteractionOrder,
            "ParameterName": self.ParameterName,
            "TeX": self.TeX,
            "Description": self.Description
        }

    def to_fr(self):
        param_entry = f"  {self.name} == {{\n"
        param_info = []
        for key, value in self.__dict__().items():
            param_info.append(f"{key:20} -> {value}")
        param_entry += "    " + ",\n    ".join(param_info) + "\n"
        param_entry += "  }"
        return param_entry
    
### ========================================================================== ###
###                                                                            ###
###                           Particles Classes                                ###
###                                                                            ###
### ========================================================================== ###

allowed_particle_types = ["scalar", "real", "pseudo", "complex", "fermion", "vector"]

# ====================================================================
#                              Particle
# ====================================================================
class Particle:
    """
    Base class for particles
    id: str
    type: str
    name: str
    mass: float
    charge: int
    color: int
    flavor: str
    """
    def __init__(self, id, name, type, mass, charge, color=1, flavor=None):
        self.id = id
        self._name = name # _name can be overridden by the pdg_info
        self.type = type
        self.mass = mass
        self.charge = charge
        self._width = "Auto" # _width can be overridden by the pdg_info
        self.color = color # color can be assigned by field class
        self.flavor = flavor # flavor can be assigned by field class
        self.__check__()

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
 
    def _all_checks(self):
        """ All checks for the initial INPUTs of the 'Particle' class. """
        self.all_checks = []
        
        def _type_check():
            assert self.type in allowed_particle_types, \
                f"Error: Type must be one of {allowed_particle_types}"

        def _id_check():
            assert isinstance(self.id, str), \
                f"Error: ID must be a string"

        def _name_check():
            assert isinstance(self.name, str), \
                f"Error: Name must be a string"

        def _mass_check():
            assert (isinstance(self.mass, float) or isinstance(self.mass, int)) and self.mass >= 0, \
                f"Error: Mass must be a number and non-negative"

        def _charge_check():
            assert isinstance(self.charge, int), \
                f"Error: Charge must be an integer"
    
        self.all_checks = [_type_check, 
                           _id_check, 
                           _name_check, 
                           _mass_check, 
                           _charge_check]
        
    @staticmethod
    def run_checks(all_checks, checklist, skip_check = False):
        for check in all_checks:
            fail_previous_check = any(isinstance(value, Exception) or value == False for value in checklist.values())
            if skip_check and fail_previous_check:
                checklist[check.__name__] = Exception(f"Skipped due to previous check failure")
            else:
                try:
                    check()
                    checklist[check.__name__] = True
                except Exception as e:
                    checklist[check.__name__] = e

    def __check__(self):
        """ Input checks for the particle class. """
        self.checklist = {}
        self._all_checks()
        self.run_checks(self.all_checks, self.checklist)

    @property
    def spin(self):
        if self.type in ["scalar", "real", "pseudo", "complex"]:
            return 0
        elif self.type == "fermion":
            return 1/2
        elif self.type == "vector":
            return 1
        else: 
            return -1
    
    @property
    def pdg_info(self):
        return get_pdg(mass=self.mass, charge=self.charge, color=self.color, spin=self.spin * 2, flavor=self.flavor)

    @property
    def full_name(self):
        return self.pdg_info['full_name'] if self.pdg_info is not None else self._name
        
    @property
    def name(self):
        return self.pdg_info['name'] if self.pdg_info is not None else self._name
        
    @property
    def width(self):
        return self.pdg_info['width'] if self.pdg_info is not None else self._width
        
    @property
    def pdg_id(self):
        if self.pdg_info is not None:
            return self.pdg_info['pdgid']
        else:
            return BSM_id(self.mass, self.charge, self.color, self.spin * 2, self.flavor)    

    @property
    def score(self):
        max_score = len(self.checklist)
        score = sum(1 for value in self.checklist.values() if value is True)
        return f"{score}/{max_score}"

    def pass_all_checks(self):
        return len(self.checklist) == sum(1 for value in self.checklist.values() if value is True)

# ====================================================================
#                              WeylSpinor
# ====================================================================
class WeylSpinor:
    """
    Each Fermion can be decomposed into two Weyl spinors, left and right.
    """
    def __init__(self, fermion, chirality):
        self.id = fermion.id + f"_{chirality[0].upper()}"
        self.name = fermion.name + f"_{chirality[0].upper()}"
        self.mass = fermion.mass
        self.type = fermion.type
        self.charge = fermion.charge
        self.fermion = fermion
        self.color = fermion.color
        self.flavor = fermion.flavor
        self.chirality = chirality
        self.pdg_info = fermion.pdg_info

    def __str__(self):
        return f"{self.name}"

# ====================================================================
#                              Fermion
# ====================================================================
class Fermion(Particle):
    def __init__(self, id, name, mass, charge):
        super().__init__(id, name, "fermion", mass, charge)
        self.left = WeylSpinor(self, "left")
        self.right = WeylSpinor(self, "right")

    def _all_checks(self):
        super()._all_checks()



# ====================================================================
#                            Real Scalar
# ====================================================================
class RealScalar(Particle):
    def __init__(self, id, name, mass, charge):
        super().__init__(id, "real", name, mass, charge)



# ====================================================================
#                           Complex Scalar
# ====================================================================
class ComplexScalar(Particle):
    def __init__(self, id, name, mass, charge):
        super().__init__(id, name, "complex", mass, charge)
        self.isDecomposed = False

    def __str__(self):
        if self.isDecomposed:
            return f"{self.name}1 + I{self.name}2"
        else:
            return f"{self.name}"

    def decompose(self):
        real_dof = RealScalar(id = self.id + "1", 
                              name = self.name + "1", 
                              mass = self.mass, 
                              charge = self.charge)
        
        imaginary_dof = RealScalar(id = self.id + "2", 
                                   name = self.name + "2", 
                                   mass = self.mass, 
                                   charge = self.charge)
        self.isDecomposed = True
        return real_dof, imaginary_dof



# ====================================================================
#                           Vector Boson
# ====================================================================
class VectorBoson(Particle):
    def __init__(self, id, name, mass, charge):
        super().__init__(id, "vector", name, mass, charge)


def write_sm_gauge_boson(file):
    """
    Write the SM gauge bosons to a file
    """
    file.write("""(* Gauge bosons: physical vector fields *)        
  V[1] == { 
    ClassName       -> A, 
    SelfConjugate   -> True,  
    Mass            -> 0,  
    Width           -> 0,  
    ParticleName    -> "a", 
    PDG             -> 22, 
    PropagatorLabel -> "a", 
    PropagatorType  -> W, 
    PropagatorArrow -> None,
    FullName        -> "Photon"
  },
  V[2] == { 
    ClassName       -> Z, 
    SelfConjugate   -> True,
    Mass            -> {MZ, 91.1876},
    Width           -> {WZ, 2.4952},
    ParticleName    -> "Z", 
    PDG             -> 23, 
    PropagatorLabel -> "Z",
    PropagatorType  -> Sine,
    PropagatorArrow -> None,
    FullName        -> "Z"
  },
  V[3] == {
    ClassName        -> W,
    SelfConjugate    -> False,
    Mass             -> {MW, Internal},
    Width            -> {WW, 2.085},
    ParticleName     -> "W+",
    AntiParticleName -> "W-",
    QuantumNumbers   -> {Q -> 1},
    PDG              -> 24, 
    PropagatorLabel  -> "W",
    PropagatorType   -> Sine,
    PropagatorArrow  -> Forward,
    FullName         -> "W"
  },
  V[4] == {
    ClassName        -> G,
    SelfConjugate    -> True,
    Indices          -> {Index[Gluon]},
    Mass             -> 0,
    Width            -> 0,
    ParticleName     -> "g", 
    PDG              -> 21,
    PropagatorLabel  -> "G",
    PropagatorType   -> C,
    PropagatorArrow  -> None,
    FullName         -> "G"
  },

(* Ghosts: related to physical gauge bosons *)
  U[1] == { 
    ClassName       -> ghA, 
    SelfConjugate   -> False,
    Ghost           -> A,
    QuantumNumbers  -> {GhostNumber -> 1},
    Mass            -> 0,
    Width	    -> 0,
    PropagatorLabel -> "uA",
    PropagatorType  -> GhostDash,
    PropagatorArrow -> Forward
  },
  U[2] == {
    ClassName       -> ghZ,
    SelfConjugate   -> False,
    Ghost           -> Z,
    QuantumNumbers  -> {GhostNumber -> 1},
    Mass            -> {MZ,91.1876},  
    Width	    -> {WZ, 2.4952},
    PropagatorLabel -> "uZ",
    PropagatorType  -> GhostDash,
    PropagatorArrow -> Forward
  },
  U[31] == { 
    ClassName       -> ghWp,
    SelfConjugate   -> False, 
    Ghost           -> W,
    QuantumNumbers  -> {GhostNumber -> 1, Q -> 1},
    Mass            -> {MW,Internal}, 
    Width           -> {WW, 2.085}, 
    PropagatorLabel -> "uWp",
    PropagatorType  -> GhostDash, 
    PropagatorArrow -> Forward
  },
  U[32] == { 
    ClassName       -> ghWm,
    SelfConjugate   -> False, 
    Ghost           -> Wbar,
    QuantumNumbers  -> {GhostNumber -> 1, Q -> -1},
    Mass            -> {MW,Internal}, 
    Width           -> {WW, 2.085},
    PropagatorLabel -> "uWm",
    PropagatorType  -> GhostDash, 
    PropagatorArrow -> Forward
  },
  U[4] == { 
    ClassName       -> ghG, 
    SelfConjugate   -> False,
    Indices         -> {Index[Gluon]},
    Ghost           -> G,
    PDG             -> 82,
    QuantumNumbers  ->{GhostNumber -> 1}, 
    Mass            -> 0,
    Width	        -> 0,
    PropagatorLabel -> "uG",
    PropagatorType  -> GhostDash,
    PropagatorArrow -> Forward
  },

(* Gauge bosons: unphysical vector fields *)
  V[11] == { 
    ClassName     -> B, 
    Unphysical    -> True, 
    SelfConjugate -> True, 
    Definitions   -> { B[mu_] -> -sw Z[mu]+cw A[mu]} 
  },
  V[12] == { 
    ClassName     -> Wi,
    Unphysical    -> True,
    SelfConjugate -> True, 
    Indices       -> {Index[SU2W]},
    FlavorIndex   -> SU2W,
    Definitions   -> { Wi[mu_,1] -> (Wbar[mu]+W[mu])/Sqrt[2], Wi[mu_,2] -> (Wbar[mu]-W[mu])/(I*Sqrt[2]), Wi[mu_,3] -> cw Z[mu] + sw A[mu]}
  },

(* Ghosts: related to unphysical gauge bosons *)
  U[11] == {
    ClassName     -> ghB, 
    Unphysical    -> True,
    SelfConjugate -> False,
    Ghost         -> B, 
    Definitions   -> { ghB -> -sw ghZ + cw ghA}
  },
  U[12] == {
    ClassName     -> ghWi,
    Unphysical    -> True,
    SelfConjugate -> False,
    Ghost         -> Wi,
    Indices       -> {Index[SU2W]},
    FlavorIndex   -> SU2W,
    Definitions   -> { ghWi[1] -> (ghWp+ghWm)/Sqrt[2], ghWi[2] -> (ghWm-ghWp)/(I*Sqrt[2]), ghWi[3] -> cw ghZ+sw ghA}
  },\n""")
    
def write_sm_higgs(file):
    """
    Write the SM Higgs to a file
    """
    file.write("""
(* Higgs: physical scalars  *)
  S[1] == {
    ClassName       -> H,
    SelfConjugate   -> True,
    Mass            -> {MH,125},
    Width           -> {WH,0.00407},
    PropagatorLabel -> "H",
    PropagatorType  -> D,
    PropagatorArrow -> None,
    PDG             -> 25,
    ParticleName    -> "H",
    FullName        -> "H"
  },
  S[2] == {
    ClassName       -> G0,
    SelfConjugate   -> True,
    Goldstone       -> Z,
    Mass            -> {MZ, 91.1876},
    Width           -> {WZ, 2.4952},
    PropagatorLabel -> "Go",
    PropagatorType  -> D,
    PropagatorArrow -> None,
    PDG             -> 250,
    ParticleName    -> "G0",
    FullName        -> "G0"
  },
  S[3] == {
    ClassName        -> GP,
    SelfConjugate    -> False,
    Goldstone        -> W,
    Mass             -> {MW, Internal},
    QuantumNumbers   -> {Q -> 1},
    Width            -> {WW, 2.085},
    PropagatorLabel  -> "GP",
    PropagatorType   -> D,
    PropagatorArrow  -> None,
    PDG              -> 251,
    ParticleName     -> "G+",
    AntiParticleName -> "G-",
    FullName         -> "GP"
  },

(* Higgs: unphysical scalars  *)
  S[11] == { 
    ClassName      -> Phi, 
    Unphysical     -> True, 
    Indices        -> {Index[SU2D]},
    FlavorIndex    -> SU2D,
    SelfConjugate  -> False,
    QuantumNumbers -> {Y -> 1/2},
    Definitions    -> { Phi[1] -> -I GP, Phi[2] -> (vev + H + I G0)/Sqrt[2]  }
  }
    """)


def write_sm_higgs_parameters_external(file):
    """
    Write the SM Higgs parameters to a file
    """
    file.write("""
  aEWM1 == { 
    ParameterType    -> External, 
    BlockName        -> SMINPUTS, 
    OrderBlock       -> 1, 
    Value            -> 127.9,
    InteractionOrder -> {QED,-2},
    Description      -> "Inverse of the EW coupling constant at the Z pole"
  },
  Gf == {
    ParameterType    -> External,
    BlockName        -> SMINPUTS,
    OrderBlock       -> 2,
    Value            -> 1.16637*^-5, 
    InteractionOrder -> {QED,2},
    TeX              -> Subscript[G,f],
    Description      -> "Fermi constant"
  },
  aS    == { 
    ParameterType    -> External,
    BlockName        -> SMINPUTS,
    OrderBlock       -> 3,
    Value            -> 0.1184, 
    InteractionOrder -> {QCD,2},
    TeX              -> Subscript[\[Alpha],s],
    Description      -> "Strong coupling constant at the Z pole"
  },
    """)


def write_sm_higgs_parameters_internal(file):
    """
    Write the SM Higgs parameters to a file
    """
    file.write("""
  aEW == {
    ParameterType    -> Internal,
    Value            -> 1/aEWM1,
    InteractionOrder -> {QED,2},
    TeX              -> Subscript[\[Alpha], EW],
    Description      -> "Electroweak coupling contant"
  },
  MW == { 
    ParameterType -> Internal, 
    Value         -> Sqrt[MZ^2/2+Sqrt[MZ^4/4-Pi/Sqrt[2]*aEW/Gf*MZ^2]], 
    TeX           -> Subscript[M,W], 
    Description   -> "W mass"
  },
  sw2 == { 
    ParameterType -> Internal, 
    Value         -> 1-(MW/MZ)^2, 
    Description   -> "Squared Sin of the Weinberg angle"
  },
  ee == { 
    ParameterType    -> Internal, 
    Value            -> Sqrt[4 Pi aEW], 
    InteractionOrder -> {QED,1}, 
    TeX              -> e,  
    Description      -> "Electric coupling constant"
  },
  cw == { 
    ParameterType -> Internal, 
    Value         -> Sqrt[1-sw2], 
    TeX           -> Subscript[c,w], 
    Description   -> "Cosine of the Weinberg angle"
  },
  sw == { 
    ParameterType -> Internal, 
    Value         -> Sqrt[sw2], 
    TeX           -> Subscript[s,w], 
    Description   -> "Sine of the Weinberg angle"
  },
  gw == { 
    ParameterType    -> Internal, 
    Definitions      -> {gw->ee/sw}, 
    InteractionOrder -> {QED,1},  
    TeX              -> Subscript[g,w], 
    Description      -> "Weak coupling constant at the Z pole"
  },
  g1 == { 
    ParameterType    -> Internal, 
    Definitions      -> {g1->ee/cw}, 
    InteractionOrder -> {QED,1},  
    TeX              -> Subscript[g,1], 
    Description      -> "U(1)Y coupling constant at the Z pole"
  },
  gs == { 
    ParameterType    -> Internal, 
    Value            -> Sqrt[4 Pi aS],
    InteractionOrder -> {QCD,1},  
    TeX              -> Subscript[g,s], 
    ParameterName    -> G,
    Description      -> "Strong coupling constant at the Z pole"
  },
  vev == {
    ParameterType    -> Internal,
    Value            -> 2*MW*sw/ee, 
    InteractionOrder -> {QED,-1},
    Description      -> "Higgs vacuum expectation value"
  },
  lam == {
    ParameterType    -> Internal,
    Value            -> MH^2/(2*vev^2),
    InteractionOrder -> {QED, 2},
    Description      -> "Higgs quartic coupling"
  },
  muH == {
    ParameterType -> Internal,
    Value         -> Sqrt[vev^2 lam],
    TeX           -> \[Mu],
    Description   -> "Coefficient of the quadratic piece of the Higgs potential"
  },\n""")

def write_sm_lagrangian(file):
    """
    Write the SM Lagrangian to a file
    """
    file.write("""
LGauge := Block[{mu,nu,ii,aa}, 
  ExpandIndices[-1/4 FS[B,mu,nu] FS[B,mu,nu] - 1/4 FS[Wi,mu,nu,ii] FS[Wi,mu,nu,ii] - 1/4 FS[G,mu,nu,aa] FS[G,mu,nu,aa], FlavorExpand->SU2W]];
               
LHiggs := Block[{ii,mu, feynmangaugerules},
  feynmangaugerules = If[Not[FeynmanGauge], {G0|GP|GPbar ->0}, {}];
 
  ExpandIndices[DC[Phibar[ii],mu] DC[Phi[ii],mu] + muH^2 Phibar[ii] Phi[ii] - lam Phibar[ii] Phi[ii] Phibar[jj] Phi[jj], FlavorExpand->{SU2D,SU2W}]/.feynmangaugerules
 ];  

LGhost := Block[{LGh1,LGhw,LGhs,LGhphi,mu, generators,gh,ghbar,Vectorize,phi1,phi2,togoldstones,doublet,doublet0},
  (* Pure gauge piece *) 	
  LGh1 = -ghBbar.del[DC[ghB,mu],mu];
  LGhw = -ghWibar[ii].del[DC[ghWi[ii],mu],mu];
  LGhs = -ghGbar[ii].del[DC[ghG[ii],mu],mu];

  (* Scalar pieces: see Peskin pages 739-742 *)
  (* phi1 and phi2 are the real degrees of freedom of GP *)
  (* Vectorize transforms a doublet in a vector in the phi-basis, i.e. the basis of real degrees of freedom *)
  gh    = {ghB, ghWi[1], ghWi[2], ghWi[3]};
  ghbar = {ghBbar, ghWibar[1], ghWibar[2], ghWibar[3]};
  generators = {-I/2 g1 IdentityMatrix[2], -I/2 gw PauliSigma[1], -I/2 gw PauliSigma[2], -I/2 gw PauliSigma[3]};
  doublet = Expand[{(-I phi1 - phi2)/Sqrt[2], Phi[2]} /. MR$Definitions /. vev -> 0]; 
  doublet0 = {0, vev/Sqrt[2]};
  Vectorize[{a_, b_}]:= Simplify[{Sqrt[2] Re[Expand[a]], Sqrt[2] Im[Expand[a]], Sqrt[2] Re[Expand[b]], Sqrt[2] Im[Expand[b]]}/.{Im[_]->0, Re[num_]->num}];
  togoldstones := {phi1 -> (GP + GPbar)/Sqrt[2], phi2 -> (-GP + GPbar)/(I Sqrt[2])};
  LGhphi=Plus@@Flatten[Table[-ghbar[[kkk]].gh[[lll]] Vectorize[generators[[kkk]].doublet0].Vectorize[generators[[lll]].(doublet+doublet0)],{kkk,4},{lll,4}]] /.togoldstones;

ExpandIndices[ LGhs + If[FeynmanGauge, LGh1 + LGhw + LGhphi,0], FlavorExpand->SU2W]];
               
LSM:= LGauge + LFermions + LHiggs + LYukawa + LGhost;\n
    """)

def write_sm_gauge_parameters(file):
    """
    Write the SM gauge parameters to a file
    """
    file.write("""
(* ************************** *)
(* *****     Gauge      ***** *)
(* *****   Parameters   ***** *)
(* *****   (FeynArts)   ***** *)
(* ************************** *)
(* *)
GaugeXi[ V[1]  ] = GaugeXi[A];
GaugeXi[ V[2]  ] = GaugeXi[Z];
GaugeXi[ V[3]  ] = GaugeXi[W];
GaugeXi[ V[4]  ] = GaugeXi[G];
GaugeXi[ S[1]  ] = 1;
GaugeXi[ S[2]  ] = GaugeXi[Z];  
GaugeXi[ S[3]  ] = GaugeXi[W];
GaugeXi[ U[1]  ] = GaugeXi[A];
GaugeXi[ U[2]  ] = GaugeXi[Z];
GaugeXi[ U[31] ] = GaugeXi[W];
GaugeXi[ U[32] ] = GaugeXi[W];
GaugeXi[ U[4]  ] = GaugeXi[G];
    """)

def write_sm_interaction_orders(file):
    """
    Write the SM interaction orders to a file
    """
    file.write("(* ************************** *)\n")
    file.write("(* *** Interaction orders *** *)\n")
    file.write("(* ***  (as used by mg5)  *** *)\n")
    file.write("(* ************************** *)\n")
    file.write("\n")
    file.write("M$InteractionOrderHierarchy = {\n")
    file.write("  {QCD, 1},\n")
    file.write("  {QED, 2}\n")
    file.write("};\n")

class vev:
    def __init__(self, id, name, vacuum, value):
        self.id = id
        self.name = name
        self.vacuum = vacuum
        self.value = value
        self._input_validate()

    def _input_validate(self):
        id_check = isinstance(self.id, str)
        name_check = isinstance(self.name, str)
        vacuum_check = isinstance(self.vacuum, list)
        value_check = isinstance(self.value, int) or isinstance(self.value, float)

        self.checklist = {"id": id_check, 
                          "name": name_check,
                          "vacuum": vacuum_check,
                          "value": value_check}
        
        max_score = len(self.checklist)
        score = sum(1 for value in self.checklist.values() if value is True)
        self.score = f"{score}/{max_score}"
        self.errors = {key: value for key, value in self.checklist.items() if value is not True}

    @property
    def nonzero_idx(self):
        import numpy as np
        return np.nonzero(self.vacuum)[0].tolist()
    

import json

if __name__ == "__main__":
    # The provided code has its own __main__ block for testing.
    # This block replaces it to run the Standard Model check.

    # Define the Standard Model using the dictionary structure.
    # Modifications are made to the dictionary from the prompt to ensure
    # compatibility with the class constructors and their validation checks.
    # Specifically, U(1) representations are converted to integers (6 * Y),
    # and 'self_conjugate' and 'QuantumNumber' attributes are added.
    sm_model_data = {
        "GaugeGroups": [
            {"id": "g1", "name": "U1Y", "charge": "Y", "group": "U_1", "coupling": "g1", "boson": "B"},
            {"id": "g2", "name": "SU2L", "charge": "I", "group": "SU_2", "coupling": "gw", "boson": "W"},
            {"id": "g3", "name": "SU3C", "charge": "C", "group": "SU_3", "coupling": "gs", "boson": "G"}
        ],
        "vevs": [
            {"id": "higgs_vev", "name": "vev", "vacuum": [0, 1], "value": 246.22}
        ],
        "particles": [
            # Leptons
            {"id": "e", "type": "fermion", "name": "e", "mass": 0.000511, "charge": -3},
            {"id": "mu", "type": "fermion", "name": "mu", "mass": 0.1057, "charge": -3},
            {"id": "tau", "type": "fermion", "name": "tau", "mass": 1.777, "charge": -3},
            {"id": "ve", "type": "fermion", "name": "ve", "mass": 0.0, "charge": 0},
            {"id": "vm", "type": "fermion", "name": "vm", "mass": 0.0, "charge": 0},
            {"id": "vt", "type": "fermion", "name": "vt", "mass": 0.0, "charge": 0},
            # Quarks
            {"id": "u", "type": "fermion", "name": "u", "mass": 0.0023, "charge": 2},
            {"id": "d", "type": "fermion", "name": "d", "mass": 0.0047, "charge": -1},
            {"id": "c", "type": "fermion", "name": "c", "mass": 1.27, "charge": 2},
            {"id": "s", "type": "fermion", "name": "s", "mass": 0.095, "charge": -1},
            {"id": "t", "type": "fermion", "name": "t", "mass": 172.76, "charge": 2},
            {"id": "b", "type": "fermion", "name": "b", "mass": 4.18, "charge": -1},
            # Higgs Components (as complex scalars)
            {"id": "phi_p", "type": "complex", "name": "phip", "mass": 125.0, "charge": 3},
            {"id": "phi_0", "type": "complex", "name": "phi0", "mass": 125.0, "charge": 0}
        ],
        "fields": [
            {"id": "LL", "name": "LL", "type": "fermion", "dim": 2, "gen": 3, "self_conjugate": False, "QuantumNumber": {"LeptonNumber" : 3}, "chirality": "left", "reps": {"g1": -3, "g2": "fnd", "g3": "singlet"}, "particles": ["ve", "e", "vm", "mu", "vt", "tau"]},
            {"id": "eR", "name": "eR", "type": "fermion", "dim": 1, "gen": 3, "self_conjugate": False, "QuantumNumber": {"LeptonNumber" : 3}, "chirality": "right", "reps": {"g1": -6, "g2": "singlet", "g3": "singlet"}, "particles": ["e", "mu", "tau"]},
            {"id": "QL", "name": "QL", "type": "fermion", "dim": 2, "gen": 3, "self_conjugate": False, "QuantumNumber": {"BaryonNumber" : 1}, "chirality": "left", "reps": {"g1": 1, "g2": "fnd", "g3": "fnd"}, "particles": ["u", "d", "c", "s", "t", "b"]},
            {"id": "uR", "name": "uR", "type": "fermion", "dim": 1, "gen": 3, "self_conjugate": False, "QuantumNumber": {"BaryonNumber" : 1}, "chirality": "right", "reps": {"g1": 4, "g2": "singlet", "g3": "fnd"}, "particles": ["u", "c", "t"]},
            {"id": "dR", "name": "dR", "type": "fermion", "dim": 1, "gen": 3, "self_conjugate": False, "QuantumNumber": {"BaryonNumber" : 1}, "chirality": "right", "reps": {"g1": -2, "g2": "singlet", "g3": "fnd"}, "particles": ["d", "s", "b"]},
            {"id": "Phi", "name": "Phi", "type": "complex", "dim": 2, "gen": 1, "self_conjugate": False, "QuantumNumber": {}, "chirality": "none", "reps": {"g1": 3, "g2": "fnd", "g3": "singlet"}, "particles": ["phi_p", "phi_0"]}
        ],
        "interactions": [
            {"id": "Yukawa_e", "type": "yukawa", "fields": ["LL", "eR", "Phi"]},
            {"id": "Yukawa_u", "type": "yukawa", "fields": ["QL", "uR", "Phi"]},
            {"id": "Yukawa_d", "type": "yukawa", "fields": ["QL", "dR", "Phi"]}
        ]
    }

    # Instantiate the Model class with the Standard Model data
    print("Initializing the Standard Model...")
    sm_model = Model(
        model_name="Standard Model",
        author="Gemini",
        model_data=sm_model_data,
        OUTPUT_PATH="SM_FeynRules"
    )

    # Run the checks and generate the FeynRules files
    print("\nRunning validation checks and generating FeynRules files...")
    sm_model.to_fr()