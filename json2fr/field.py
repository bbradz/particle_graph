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
    
    def _all_checks(self):
        """ All checks for the initial INPUTs of the 'Field' class. """
        self.all_checks = []

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
                           _ptcl_check]

    def __check__(self):
        """ Input checks for the field class. """
        from utility import run_checks
        self.checklist = {}
        self._all_checks()
        run_checks(self.all_checks, self.checklist)

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
            return Index("Colour", 3, "NoUnfold", color = True)
        elif group.isSU3C and dim == 8:
            return Index("Gluon", 8, "NoUnfold", color = True)
        else:
            idx_name = str(group.group).replace("(", "").replace(")", "") + name.num2tuple(int(dim))[0].upper()
            return Index(idx_name, int(dim), "Unfold", color = False)
        
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
        
    def _all_validations(self):
        self.all_validations = [
            self._create_generation_index, 
            self._check_gen_type_consistency, 
            self._create_full_reps, 
            self._check_dim_reps_consistency
            ]

    def __validate__(self):
        from utility import run_checks
        self._all_validations()
        run_checks(self.all_validations, self.checklist, skip_check = True)

    @property
    def score(self):
        max_score = len(self.checklist)
        score = sum(1 for value in self.checklist.values() if value is True)
        return f"{score}/{max_score}"

    def pass_all_checks(self):
        return len(self.checklist) == sum(1 for value in self.checklist.values() if value is True)
        


# ====================================================================
#                            Fermion Field
# ====================================================================
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

    def _all_checks(self):
        super()._all_checks()

        def _chirality_check():
            assert self.chirality in ["left", "right"], \
                f"Error: {self.chirality} is not a valid chirality"

        self.all_checks.append(_chirality_check)

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

    def _all_validations(self):
        super()._all_validations()
        self.all_validations.extend([self._assign_colors, 
                                     self._sort_unphy_fields, 
                                     self._sort_phy_fields, 
                                     self._assign_flavors, 
                                     self._check_charge_consistency, 
                                     self._check_mass_type])

    def to_matrix(self):
        return np.array([[p.id for p in gen] for gen in self._unphy_fields])
    
    # ------------------------------------------------------------------
    #                        Write FeynRules
    # ------------------------------------------------------------------
    @property
    def phy_field_info(self):
        """
        ClassName
        ClassMembers
        Indices
        FlavorIndex
        Mass
        Width
        QuantumNumber
        PropagatorLabel
        PropagatorType
        PropagatorArrow
        PDG
        ParticleName
        AntiParticleName
        FullName
        """
        # ------------------------------------------------------------
        # Check if the field passes all checks
        assert self.pass_all_checks(), \
            f"Error: {self.name} has failed the checks"

        # ------------------------------------------------------------
        def _class_name(idx):
            charge = self._phy_fields[idx][0].charge
            color = self.color
            class_name = name.fermion_field_name(charge, color)[0]
            return class_name

        def _class_members(idx):
            member_names = [p.fermion.name for p in self._phy_fields[idx]]
            member_names = self.rewrite(member_names)
            return member_names
        
        def _indices():
            indices = [self.gen_idx] + [i for i in self.indices if i.color == True]
            indices = [repr(i) for i in indices]
            indices_str = "{" + ", ".join(indices) + "}"
            return indices_str

        def _flavor_index():
            return self.gen_idx

        def _mass(idx):
            class_name = _class_name(idx)
            particle_list = self._phy_fields[idx]
            if all(particle.fermion.mass == 0 for particle in particle_list):
                return 0
            else:
                Mass = ["M"+str(class_name).upper()]
                for particle in particle_list:
                    mass_name = "M" + particle.fermion.name.upper()
                    Mass.append([mass_name, particle.fermion.mass])
                Mass = self.rewrite(Mass)
                return Mass

        def _width(idx):
            class_name = _class_name(idx)
            particle_list = self._phy_fields[idx]
            if all(particle.fermion.width == 0 for particle in particle_list):
                return 0
            else:
                Width = ["W"+str(class_name).upper()]
                for particle in particle_list:
                    width_name = "W" + particle.fermion.name.upper()
                    Width.append([width_name, particle.fermion.width])
                Width = self.rewrite(Width)
                return Width

        def _quantum_number(idx):
            qnumber = self.QuantumNumber.copy()
            qnumber.update({"Q": self._phy_fields[idx][0].charge})
            qnumber = {key: str(Fraction(value,3)) for key, value in qnumber.items() if value != 0}
            qnumber = str(qnumber).replace(":", " ->").replace("'", "")
            return qnumber

        def _propagator_label(idx):
            class_name = _class_name(idx)
            member_name = [p.fermion.name for p in self._phy_fields[idx]]
            prop_label_str = [class_name] + member_name
            prop_label_str = self.rewrite(prop_label_str, quote = True)
            return prop_label_str

        def _PDG(idx):
            PDG_list = [p.fermion.pdg_id for p in self._phy_fields[idx]]
            PDG_list = self.rewrite(PDG_list, quote = False)
            return PDG_list

        def _particle_name(idx):
            particle_name_list = [p.fermion.name for p in self._phy_fields[idx]]
            particle_name_list = self.rewrite(particle_name_list, quote = True)
            return particle_name_list

        def _anti_particle_name(idx):
            anti_particle_name_list = [p.fermion.name + "~" for p in self._phy_fields[idx]]
            anti_particle_name_list = self.rewrite(anti_particle_name_list, quote = True)
            return anti_particle_name_list

        def _particle_fullname(idx):
            particle_fullname_list = [p.fermion.full_name for p in self._phy_fields[idx]]
            particle_fullname_list = self.rewrite(particle_fullname_list, quote = True)
            return particle_fullname_list
            
        phy_fields = []
        for i in range(self.dim):
            phy_field = {
                "ClassName": _class_name(i),
                "ClassMembers": _class_members(i),
                "Indices": _indices(),
                "FlavorIndex": _flavor_index(),
                "SelfConjugate": self.self_conjugate,
                "Mass": _mass(i),
                "Width": _width(i), 
                "QuantumNumber": _quantum_number(i), 
                "PropagatorLabel": _propagator_label(i), 
                "PropagatorType": "Straight", 
                "PropagatorArrow": "Forward", 
                "PDG": _PDG(i), 
                "ParticleName": _particle_name(i), 
                "AntiParticleName": _anti_particle_name(i), 
                "FullName": _particle_fullname(i)
            }
            phy_fields.append(phy_field)
        return phy_fields

    @property
    def unphy_field_info(self):
        assert self.pass_all_checks(), \
            f"Error: {self.name} has failed the checks"

        def _class_name():
            return self.name
        
        def _indices():
            indices = [repr(i) for i in self.indices]
            indices_str = "{" + ", ".join(indices) + "}"
            return indices_str
        
        def _flavor_index():
            return self.gen_idx

        def _quantum_number():
            qnumber = self.abelian_charges
            qnumber = {key: str(Fraction(value,3)) for key, value in qnumber.items() if value != 0}
            qnumber = str(qnumber).replace(":", " ->").replace("'", "")
            return qnumber

        def _definition():
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
            return "{ " + (",\n" + " "*30).join(definition) + " }"

        unphy_field = {
            "ClassName": _class_name(),
            "Unphysical": True, 
            "Indices": _indices(), 
            "FlavorIndex": _flavor_index(),
            "SelfConjugate": self.self_conjugate,
            "QuantumNumber": _quantum_number(),
            "Definition": _definition()    
        }
        return unphy_field

    @staticmethod
    def rewrite(list, quote = False):
        list = str(list).replace("[", "{").replace("]", "}")
        if quote:
            list = str(list).replace("'", "\"")
        else:
            list = str(list).replace("'", "")
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



# ------------------------------------------------------------------
if __name__ == "__main__":
    pass