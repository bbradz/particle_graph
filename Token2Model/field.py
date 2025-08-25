### ========================================================================== ###
###                                                                            ###
###                           Particles Classes                                ###
###                                                                            ###
### ========================================================================== ###

import numpy as np
from fractions import Fraction
from . import name
from .particle import Particle, WeylSpinor
from .utility import run_checks

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
    """
    def __init__(self, id, name, type, groups, reps, dim, gen, particles, self_conjugate):
        self.id = id
        self.name = name
        self.type = type
        self.groups = groups
        self.reps = reps
        self.dim = dim
        self.gen = gen
        self.particles = particles
        self.self_conjugate = self_conjugate
        self.full_reps = {}
        self.is_massive = False
        self.mass_type = None
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
            "self_conjugate": self.self_conjugate
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

        # Generation Type Consistency
        def _check_gen_type_consistency():
            if self.type != "fermion":
                assert self.gen == 1, \
                    f"Error: Non-fermion fields can have only one generation"

        def _sort_reps():
            rep_dict = {}
            self.allow_dim = [1]
            for key, group in self.groups.items():
                if group.abelian:
                    rep_dict[key] = Fraction(self.reps[key], 6)
                else:
                    rep_dict[key] = group.rep_list[self.reps[key]]
                    if group.isSU3C:
                        self.color = rep_dict[key]
                    else:
                        self.allow_dim.append(rep_dict[key])
            self.full_reps = rep_dict
            self.allow_dim = list(set(self.allow_dim))

        def _check_reps_dim_consistency():          
            assert self.dim in self.allow_dim, \
                f"Error: {self.name} with dim-{self.dim} is not in allowed dims: {self.allow_dim}"
            
            if self.dim != 1:
                self.allow_dim.remove(self.dim)

            assert len(self.allow_dim) == 1 and self.allow_dim[0] == 1, \
                f"Error: {self.name} with dim-{self.dim} is not in allowed dims: {self.allow_dim}"
            
        def _sort_particle_ordering():
            charge_eigenvec = {}

            for p in self.particles:
                if p.charge not in charge_eigenvec:
                    charge_eigenvec[p.charge] = [p]
                else:
                    charge_eigenvec[p.charge].append(p)
            
            for _, ptcls in charge_eigenvec.items():
                ptcls.sort(key=lambda p: p.mass)
                assert len(ptcls) == self.gen, \
                    f"Error: {self.name} has {len(ptcls)} particles for charge {p.charge}, but {self.gen} generations are required."

            charge_eigenval = sorted(charge_eigenvec.keys(), reverse=True)
            assert len(charge_eigenval) == self.dim, \
                f"Error: {self.name} has {len(charge_eigenval)} charges, but {self.gen} generations are required."
            
            self._unphy_fields = np.column_stack([charge_eigenvec[charge] for charge in charge_eigenval])

            self._phy_fields = np.array(self._unphy_fields).transpose().tolist()
            print(self._phy_fields)

        self.all_checks = [_id_check, 
                           _name_check, 
                           _type_check, 
                           _groups_check, 
                           _reps_check, 
                           _dim_check, 
                           _gen_check, 
                           _particles_check, 
                           _self_conjugate_check, 
                           _ptcl_check, 
                           _check_gen_type_consistency,
                           _sort_reps,
                           _check_reps_dim_consistency,
                           _sort_particle_ordering
                           ]

    def __check__(self):
        """ Input checks for the field class. """
        self.checklist = {}
        self._all_checks()
        run_checks(self.all_checks, self.checklist)

    def _all_validations(self):
        """ All validations for the 'Interaction' class. """
        self.all_validations = []

    def __validate__(self):
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
    chirality: str
    """
    def __init__(self, id, name, groups, reps, dim, gen, particles, self_conjugate, chirality):
        self.chirality = chirality
        self._unphy_fields = None
        self._phy_fields = None
        self.color = 1
        super().__init__(id, name, "fermion", groups, reps, dim, gen, particles, self_conjugate)
        

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
            "chirality": self.chirality
        }

    def _all_checks(self):
        super()._all_checks()

        def _chirality_check():
            assert self.chirality in ["left", "right"], \
                f"Error: {self.chirality} is not a valid chirality"
        
        # ------------------------------------------------------------------
        
        # Assign colors to the particles
        def _assign_colors():
            for p in self.particles:
                if self.color is None:
                    pass
                else:
                    p.fermion.color = self.color

        self.all_checks.extend([
            _chirality_check,
            _assign_colors
        ])
        
    # ------------------------------------------------------------------
    #                        Write Model File
    # ------------------------------------------------------------------

    def _class_name(self, idx):
            charge = self._phy_fields[idx][0].charge
            color = self.color
            class_name, description = name.fermion_field_name(charge, color)
            if self.chirality == "left":
                return f"{class_name}L", description
            elif self.chirality == "right":
                return f"{class_name}R", description

    @property
    def phy_field_info(self):
        # Check if the field passes all checks
        if not self.pass_all_checks():
            print(f"Error: {self.name} has failed the checks")
            return None
        
        def _class_members(idx):
            member_names = [p.fermion.name for p in self._phy_fields[idx]]
            return member_names

        def _PDG(idx):
            PDG_list = [p.fermion.pdg_id for p in self._phy_fields[idx]]
            return PDG_list

        def _mass(idx):
            mass = [p.fermion.mass for p in self._phy_fields[idx]]
            return mass

        def _width(idx):
            particle_list = self._phy_fields[idx]
            width = [p.fermion.width for p in particle_list]
            if all(w == "Automatic" for w in width):
                width = "Automatic"
            return width

        def _charge(idx):
            charge = self._phy_fields[idx][0].charge
            charge = Fraction(charge, 3)
            return charge

        phy_fields = {}
        for i in range(self.dim):
            class_name = self._class_name(i)
            phy_field = {
                "Description": class_name[1],
                "LaTeX": class_name[0],
                "PDG": _PDG(i),
                "Mass": _mass(i),
                "OutputName": class_name[0][:-1],
                "ElectricCharge": _charge(i),
                "Width": _width(i),
                # additional info
                "chirality": self.chirality,
                "class_members": _class_members(i),
                "location": i
            }
            phy_fields[class_name[0]] = phy_field
        return phy_fields
    
    @property
    def unphy_field_info(self):
        if not self.pass_all_checks():
            print(f"Error: {self.name} has failed the checks")
            return None

        if self.chirality == "left":
            element = [f"{self._class_name(i)[0]}" for i in range(self.dim)]
        elif self.chirality == "right":
            element = [f"conj[{self._class_name(i)[0]}]" for i in range(self.dim)]
            if self.full_reps["g3"] == 1:
                pass
            else:
                self.full_reps["g3"] = - self.full_reps["g3"]
    
        if self.dim == 1:
            multiplet = element[0]
        else:
            multiplet = "{" + ", ".join(element) + "}"
        reps_info = ", ".join([str(value) for value in self.full_reps.values()])

        return f"{self.name}, {self.gen}, {multiplet}, {reps_info}"
        

# ====================================================================
#                            Scalar Field
# ====================================================================
class ScalarField(Field):
    def __init__(self, id, name, type, groups, reps, dim, gen, particles, self_conjugate):
        super().__init__(id, name, type, groups, reps, dim, gen, particles, self_conjugate)
        self.potential = None
        self.vev = None
        self.get_vev = False

    def _all_validations(self):
        super()._all_validations()
        self.all_validations.extend([])

    def phy_field_info(self):
        pass

    @property
    def unphy_field_info(self):
        reps_info = ", ".join([str(value) for value in self.full_reps.values()])
        if self.dim == 1:
            multiplet = f"{self.particles[0].name}"
        else:
            multiplet = "{" + ", ".join([f"{p.name}" for p in self.particles]) + "}"
        
        return f"{self.name}, {self.gen}, {multiplet}, {reps_info}"



# ====================================================================
#                            Vector Fields
# ====================================================================
class VectorField(Field):
    def __init__(self, id, name, groups, reps, dim, gen, particles, self_conjugate):
        super().__init__(id, name, "vector", groups, reps, dim, gen, particles, self_conjugate)
        self.__check__()

    def __check__(self):
        super().__check__()

# ------------------------------------------------------------------
if __name__ == "__main__":
    pass