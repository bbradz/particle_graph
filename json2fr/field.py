### ========================================================================== ###
###                                                                            ###
###                           Particles Classes                                ###
###                                                                            ###
### ========================================================================== ###

import numpy as np
from particle import Particle, WeylSpinor
from fractions import Fraction
from index import Index
import name

def run_checks(all_checks, checklist):
    for check in all_checks:
        try:
            check()
            checklist[check.__name__] = True
        except Exception as e:
            checklist[check.__name__] = e
 

### ======================= ###
###          Field          ###
### ======================= ###
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

        self.__check__()
        self.indices = []
        self.full_reps = {}
        self.abelian_charges = {}
        
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
    
    def __check__(self):
        """
        Input checks for the field class.
        """

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
        
        self.all_checks = [
            _id_check,
            _name_check,
            _type_check,
            _groups_check,
            _reps_check,
            _dim_check,
            _gen_check,
            _particles_check,
            _self_conjugate_check,
            _QuantumNumber_check,
            _ptcl_check
        ]

        self.checklist = {}
        run_checks(self.all_checks, self.checklist)
    
    # after checking all the inputs, the rest of the checks are done in the sequence
    # if a pre-requisite is not met, the rest of the checks will not be performed and automatically fail

    # Generation Index 
    def _create_generation_index(self):
        if self.gen > 1:
            gen_idx_name = name.num2words(self.gen).capitalize() + "Gen"
            self.gen_idx = Index(gen_idx_name, self.gen, "Fold")
            self.indices.append(self.gen_idx)
    
    # Generation Type Consistency
    def _check_gen_type_consistency(self):
        if self.type != "fermion":
            assert self.gen == 1, \
                f"Error: Non-fermion fields can have only one generation"

    @staticmethod
    def _index(group, dim):
        if dim == 1:
            return None
        elif group.isSU3C and dim == 3:
            return Index("Colour", 3, "Fold", color = True)
        elif group.isSU3C and dim == 8:
            return Index("Gluon", 8, "Fold", color = True)
        else:
            idx_name = str(group.group).replace("(", "").replace(")", "") + name.num2tuple(int(dim))[0].upper()
            return Index(idx_name, int(dim), "Fold", color = False)
        
    # Assign reps to the groups
    def _create_full_reps(self):
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
    def _check_dim_reps_consistency(self):
        for _, rep_dict in self.full_reps.items():
            if rep_dict["abelian"]:
                pass
            elif rep_dict["isColor"]:
                self.color = rep_dict["group"].rep_list[rep_dict["reps"]]
            elif rep_dict["reps"] == "singlet":
                pass
            else:
                assert rep_dict["dim"] == self.dim, \
                    f"Error: {self.name} with dim-{self.dim} is inconsistent with \"{rep_dict['reps']}\" ({rep_dict['dim']}) in {rep_dict['group']}"

    # Validate the field
    def __validate__(self):
        self.all_validations = [
            self._create_generation_index, 
            self._check_gen_type_consistency, 
            self._create_full_reps, 
            self._check_dim_reps_consistency
            ]
        run_checks(self.all_validations, self.checklist)

    @property
    def score(self):
        max_score = len(self.checklist)
        score = sum(1 for value in self.checklist.values() if value is True)
        return f"{score}/{max_score}"

    def pass_all_checks(self):
        return len(self.checklist) == sum(1 for value in self.checklist.values() if value is True)
        


### ======================== ###
###       Fermion Field      ###
### ======================== ###
class FermionField(Field):
    """
    Class for fermion fields.
    groups: list of str
    reps: list of int
    dim: int > 0
    gen: int > 0
    particles: list of Particle
    self_conjugate: bool
    QuantumNumber: dict
    chirality: str
    """
    def __init__(self, id, name, groups, reps, dim, gen, particles, self_conjugate, QuantumNumber, chirality):
        self.chirality = chirality
        super().__init__(id, name, "fermion", groups, reps, dim, gen, particles, self_conjugate, QuantumNumber)
        
        self.__check__()
        self.color = 1
        self.mass_type = None
        self._unphy_fields = None
        self._phy_fields = None

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

    def __check__(self):
        super().__check__()

        def _chirality_check():
            assert self.chirality in ["left", "right"], \
                f"Error: {self.chirality} is not a valid chirality"

        self.all_checks.append(_chirality_check)
        run_checks(self.all_checks, self.checklist)

    # Assign colors to the particles
    def _assign_colors(self):
        for p in self.particles:
            if self.color is None:
                pass
            else:                
                p.fermion.color = self.color
            
    # Sort the particles into generations and flavors
    def _sort_unphy_fields(self):
        self._unphy_fields = np.array(self.particles).reshape(self.gen, self.dim).tolist()

    # Sort the particles into generations and flavors
    def _sort_phy_fields(self):
        self._phy_fields = np.array(self._unphy_fields).transpose().tolist()

    # Assign Flavors to the particles
    # This is a hardcoded function that assigns flavors to the particles based on the QuantumNumber
    def _assign_flavors(self):
        if self.QuantumNumber['LeptonNumber'] != 0:
            for gen in self._unphy_fields:
                flavor = [p.fermion.name for p in gen if p.charge != 0]
                for p in gen:
                    if len(flavor) == 1:
                        p.fermion.flavor = flavor[0]
                    else:
                        p.fermion.flavor = None

    # Check if the charges of the particles are consistent
    def _check_charge_consistency(self):
        for idx, gen in enumerate(self._phy_fields):
            assert all(cf.charge == gen[0].charge for cf in gen), \
                f"Error: {self.name} has inconsistent charges in generation {idx+1}"

    # massive particle must acquire mass from interactions
    def _check_mass_type(self):
        self.is_massive = all(p.fermion.mass == 0 for p in self.particles)
        if self.is_massive:
            assert self.mass_type is not None, \
                f"Error: {self.name} is massive but has no mass type"

    def __validate__(self):
        super().__validate__()
        all_checks = [
            self._assign_colors, 
            self._sort_unphy_fields, 
            self._sort_phy_fields, 
            self._assign_flavors, 
            self._check_charge_consistency, 
            self._check_mass_type
            ]   
        run_checks(all_checks, self.checklist)

    def to_matrix(self):
        return np.array([[p.id for p in gen] for gen in self._unphy_fields])
    
    # ================================ #
    #         Write FeynRules          #
    # ================================ #
    @property
    def phy_field_info(self):
        assert self.pass_all_checks(), \
            f"Error: {self.name} has failed the checks"

        # Format propagator label for FeynRules PropagatorLabel
        def write_propagator_label(class_name, member_name):
            prop_label_str = [class_name] + member_name
            return str(prop_label_str).replace("[", "{").replace("]", "}")

        # Format particle masses for FeynRules Mass
        def write_Mass(class_name, particle_list):
            if all(particle.fermion.mass == 0 for particle in particle_list):
                return 0
            else:
                Mass = ["M"+str(class_name).upper()]
                for particle in particle_list:
                    mass_name = "M" + particle.fermion.name.upper()
                    Mass.append((mass_name, particle.fermion.mass))
                Mass = str(Mass).translate(str.maketrans({"'": "", "[": "{", "]": "}", "(": "{", ")": "}"}))
                return Mass

        # Format particle widths for FeynRules Width
        def write_Width(class_name, particle_list):
            if all(particle.fermion.width == 0 for particle in particle_list):
                return 0
            else:
                Width = ["W"+str(class_name).upper()]
                for particle in particle_list:
                    width_name = "W" + particle.fermion.name.upper()
                    Width.append((width_name, particle.fermion.width))
                Width = str(Width).translate(str.maketrans({"'": "", "[": "{", "]": "}", "(": "{", ")": "}"}))
                return Width

        phy_fields = []
        for i in range(self.dim):
            class_name = name.fermion_field_name(self._phy_fields[i][0].charge, self.color)
            particle_name_list = [p.fermion.name for p in self._phy_fields[i]]
            particle_fullname_list = [p.fermion.full_name for p in self._phy_fields[i]]
            indices = [self.gen_idx] + [i for i in self.indices if i.color == True]
            qnumber = self.QuantumNumber.copy()
            qnumber.update({"Q": self._phy_fields[i][0].charge})
            phy_field = {
                "ClassName": f"{class_name[0]}",
                "ClassMembers": self.rewrite_list(particle_name_list),
                "Indices": self.rewrite_Indices(indices),
                "FlavorIndex": self.gen_idx, #  GenIdx and ColorIdx
                "Mass": write_Mass(class_name[0], self._phy_fields[i]), 
                "Width": write_Width(class_name[0], self._phy_fields[i]), 
                "QuantumNumber": self.rewrite_QuantumNumber(qnumber), 
                "PropagatorLabel": write_propagator_label(class_name[0], particle_name_list), 
                "PropagatorType": "Straight", 
                "PropagatorArrow": "Forward", 
                "PDG": [p.fermion.pdg_id for p in self._phy_fields[i]], 
                "ParticleName": particle_name_list, 
                "AntiParticleName": [p.fermion.name + "~" for p in self._phy_fields[i]], 
                "FullName": particle_fullname_list
            }
            phy_fields.append(phy_field)
        return phy_fields

    @property
    def unphy_field_info(self):
        assert self.pass_all_checks(), \
            f"Error: {self.name} has failed the checks"

        def write_Definition():
            definition = []
            proj_matrix = "ProjM" if self.chirality == "left" else "ProjP"
            if self.color > 1:
                if self.dim > 1:
                    for i in range(self.dim):
                        class_name = name.fermion_field_name(self._phy_fields[i][0].charge, self.color)[0]
                        definition.append(f"{self.name}[sp1_, {i+1}, ff_, cc_] :> Module[{{sp2}}, {proj_matrix}[sp1, sp2] {class_name}[sp2, ff, cc]]")
                else:
                    class_name = name.fermion_field_name(self._phy_fields[0][0].charge, self.color)[0]
                    definition.append(f"{self.name}[sp1_, ff_, cc_] :> Module[{{sp2}}, {proj_matrix}[sp1, sp2] {class_name}[sp2, ff, cc]]")
            else:
                for i in range(self.dim):
                    class_name = name.fermion_field_name(self._phy_fields[i][0].charge, self.color)[0]
                    definition.append(f"{self.name}[sp1_, {i+1}, ff_] :> Module[{{sp2}}, {proj_matrix}[sp1, sp2] {class_name}[sp2, ff]]")
            return "{ " + (",\n" + " "*34).join(definition) + " }"

        unphy_field = {
            "ClassName": self.name,
            "Unphysical": True, 
            "Indices": self.rewrite_Indices(self.indices), # GenIdx and ColorIdx and RepIdx 
            "FlavorIndex": self.gen_idx, # non-color RepIdx
            "SelfConjugate": self.self_conjugate,
            "QuantumNumber": self.rewrite_QuantumNumber(self.abelian_charges),
            "Definition": write_Definition()    
        }
        return unphy_field

    @staticmethod
    def rewrite_list(list):
        return str(list).translate(str.maketrans({"'": "", "[": "{", "]": "}"}))

    @staticmethod
    def rewrite_QuantumNumber(QuantumNumber):
        QuantumNumber = {key: str(Fraction(value,3)) for key, value in QuantumNumber.items() if value != 0}
        return str(QuantumNumber).replace(":", " -> ").replace("'", "")

    @staticmethod
    def rewrite_Indices(indices):
        indices = [repr(i) for i in indices]
        indices_str = "{" + ", ".join(indices) + "}"
        return indices_str


### ======================== ###
###       Scalar Field       ###
### ======================== ###
class ScalarField(Field):
    def __init__(self, id, name, type, groups, reps, dim, gen, particles, self_conjugate, QuantumNumber):
        super().__init__(id, name, type, groups, reps, dim, gen, particles, self_conjugate, QuantumNumber)
        self.__check__()
        self.potential = None
        self.vev = None
        self.get_vev = False

    def __check__(self):
        super().__check__()
        # TO-DO: Override the ptcl_type from Field to ScalarField's subtype

    def _check_vev_potential(self):
        """ Check the potential allowed the vev """
        assert True, \
            f"{self.vev} does not minimize {self.name}'s potential"

    def acquire_vev(self, vev):
        """ ScalarField Promotion to Higgs (similar to Pawn Promotion in Chess)"""
        pass

    def __validate__(self):
        super().__validate__()
        all_checks = [self._check_vev_potential]
        run_checks(all_checks, self.checklist)


    def phy_field_info(self):
        pass

    def unphy_field_info(self):
        pass

### ============================ ###
###        Vector Field          ###
### ============================ ###
class VectorField(Field):
    def __init__(self, id, name, groups, reps, dim, gen, particles, self_conjugate, QuantumNumber):
        super().__init__(id, name, "vector", groups, reps, dim, gen, particles, self_conjugate, QuantumNumber)
        self.__check__()

    def __check__(self):
        super().__check__()



if __name__ == "__main__":
    pass