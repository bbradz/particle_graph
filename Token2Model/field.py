### ========================================================================== ###
###                                                                            ###
###                           Particles Classes                                ###
###                                                                            ###
### ========================================================================== ###

import numpy as np
from fractions import Fraction
from . import name
from .particle import Particle, WeylSpinor
#from .utility import run_checks
from .check import run_checks

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
    def __init__(self, id, name, type, groups, reps, dim, gen, particles, self_conjugate, simplify_checklist):
        self.id = id
        self.name = name
        self.type = type
        self.groups = groups
        self.reps = reps
        self.dim = dim
        self.gen = gen
        self.particles = particles
        self.self_conjugate = self_conjugate
        self.simplify_checklist = simplify_checklist
        self.full_reps = {}
        self.is_massive = False
        self.mass_term = []
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

        # ------------------------- Simple Checks ------------------------------
        # check if the id is a string
        def _id_check():
            result = {"score": 1, "error_var": [], "message": "Passed", "max_score": 1}
            if not isinstance(self.id, str):
                error_var = ["id"]
                message = f"'id' must be a string"
                result.update({"score": 0, "error_var": error_var, "message": message})
            return result

        # check if the name is a string
        def _name_check():
            result = {"score": 1, "error_var": [], "message": "Passed", "max_score": 1}
            if not isinstance(self.name, str):
                error_var = ["name"]
                message = f"'name' must be a string."
                result.update({"score": 0, "error_var": error_var, "message": message})
            return result
        
        # check if the type is a string and one of complex/real/fermion/vector
        def _type_check():
            result = {"score": 1, "error_var": [], "message": "Passed", "max_score": 1}
            if not isinstance(self.type, str) or self.type not in ["complex", "real", "fermion", "vector"]:
                error_var = ["type"]
                message = f"'type' must be a string and one of complex/real/fermion/vector"
                result.update({"score": 0, "error_var": error_var, "message": message})
            return result
        
        # check if the groups are a dictionary
        def _groups_check():
            result = {"score": 1, "error_var": [], "message": "Passed", "max_score": 1}
            if not isinstance(self.groups, dict):
                error_var = ["groups"]
                message = f"'groups' must be a dictionary"
                result.update({"score": 0, "error_var": error_var, "message": message})
            return result

        # check if the reps are a dictionary of int or str
        def _reps_check():
            result = {"score": 3, "error_var": [], "message": "Passed", "max_score": 3}
            if not isinstance(self.reps, dict):
                error_var = ["reps"]
                message = f"'reps' must be a dictionary"
                result.update({"score": 0, "error_var": error_var, "message": message})
                return result
            
            if len(self.reps) != len(self.groups):
                error_var = ["reps"]
                message = f"'reps' and 'groups' must be the same length"
                result.update({"score": 1, "error_var": error_var, "message": message})
                return result
            
            error_var = [f"reps:{key}" for key, value in self.reps.items() if not isinstance(value, (int, str))]
            if error_var:
                message = f"'reps' must be a dictionary of int or str"
                result.update({"score": 2, "error_var": error_var, "message": message})
                return result
            return result
        
        # check if the dimension is a positive integer
        def _dim_check():
            result = {"score": 1, "error_var": [], "message": "Passed", "max_score": 1}
            if not isinstance(self.dim, int) or self.dim <= 0:
                error_var = ["dim"]
                message = f"'dim' must be a positive integer"
                result.update({"score": 0, "error_var": error_var, "message": message})
            return result
        
        # check if the generation is a positive integer
        def _gen_check():
            result = {"score": 1, "error_var": [], "message": "Passed", "max_score": 1}
            if not isinstance(self.gen, int) or self.gen <= 0:
                error_var = ["gen"]
                message = f"'gen' must be a positive integer"
                result.update({"score": 0, "error_var": error_var, "message": message})
            return result
        
        # check if the particles are a list of Particle
        def _particles_check():
            result = {"score": 2, "error_var": [], "message": "Passed", "max_score": 2}
            if not isinstance(self.particles, list):
                error_var = ["particles"]
                message = f"'particles' must be a list"
                result.update({"score": 0, "error_var": error_var, "message": message})
                return result
            
            fermion_check = self.type == "fermion" and all(isinstance(p, WeylSpinor) for p in self.particles)
            non_fermion_check = self.type != "fermion" and all(isinstance(p, Particle) for p in self.particles)
            if not fermion_check and not non_fermion_check:
                error_var = ["particles"]
                message = f"'particles' must be a list of Particle"
                result.update({"score": 1, "error_var": error_var, "message": message})
            return result
        
        # check if self-conjugate is a bool
        def _self_conjugate_check():
            result = {"score": 1, "error_var": [], "message": "Passed", "max_score": 1}
            if not isinstance(self.self_conjugate, bool):
                error_var = ["self_conjugate"]
                message = f"'self_conjugate' must be a bool"
                result.update({"score": 0, "error_var": error_var, "message": message})
            return result   
        
        # ------------------------- Necessary Checks ------------------------------
        
        # check if the field is consistent with the type
        def _gen_type_consistency():
            result = {"score": 1, "error_var": [], "message": "Passed", "max_score": 1}
            if self.type != "fermion" and self.gen != 1:
                error_var = ["gen"]
                message = f"Non-fermion fields can have only one generation"
                result.update({"score": 0, "error_var": error_var, "message": message})
            return result

        # Sort reps
        def _sort_reps():
            result = {"score": 1, "error_var": [], "message": "Passed", "max_score": 1}
            try:
                rep_dict = {}
                self.allow_dim = [1]
                for key, group in self.groups.items():
                    if group.abelian:
                        rep_dict[key] = Fraction(self.reps[key], 6)
                    else:
                        rep_dict[key] = group.dim(self.reps[key])
                        if group.isSU3C:
                            self.color = rep_dict[key]
                        else:
                            self.allow_dim.append(rep_dict[key])
                self.full_reps = rep_dict
                self.allow_dim = sorted(list(set(self.allow_dim)))
            except Exception as e:
                error_var = ["reps"]
                message = f"Error: {e}"
                result.update({"score": 0, "error_var": error_var, "message": message})
            return result

        # Check reps and dim consistency
        def _reps_dim_consistency():          
            result = {"score": 1, "error_var": [], "message": "Passed", "max_score": 1}
            if self.dim == 1 and self.allow_dim != [1]:
                error_var = ["dim"]
                message = f"reps and dim are not consistent"
                result.update({"score": 0, "error_var": error_var, "message": message})
                return result
            elif self.dim != 1 and self.allow_dim != [1, self.dim]:
                error_var = ["dim"]
                message = f"reps and dim are not consistent"
                result.update({"score": 0, "error_var": error_var, "message": message})
                return result
            return result

        # compute allowed charges
        def _allowed_charges():
            result = {"score": 1, "error_var": [], "message": "Passed", "max_score": 1}
            
            def _compute_allowed_charges():
                if self.reps["g2"] == "singlet":
                    T3 = [0]
                elif self.reps["g2"] == "fnd":
                    T3 = [1/2, -1/2]
                elif self.reps["g2"] == "adj":
                    T3 = [1, 0, -1]
                Y = self.reps["g1"]/6
                Q = np.array(T3) + Y
                Q = Q * 3            # times 3 to get integer charges
                #self.allowed_Q = -Q if self.chirality == "right" else Q
                self.allowed_Q = Q
            try:
                _compute_allowed_charges()
            except Exception as e:
                error_var = ["reps"]
                message = f"Error: {e}"
                result.update({"score": 0, "error_var": error_var, "message": message})
            return result
        
        # check if there are duplicate particles
        def _deplicate_particles():
            result = {"score": 1, "error_var": [], "message": "Passed", "max_score": 1}
            if len(self.particles) != len(set(self.particles)):
                error_var = ["particles"]
                message = f"Duplicate particles found"
                result.update({"score": 0, "error_var": error_var, "message": message})
            return result

        # check if the number of particles is consistent with the dim and gen
        def _particle_numbers():
            result = {"score": 1, "error_var": [], "message": "Passed", "max_score": 1}
            if len(self.particles) != self.dim * self.gen:
                error_var = ["particles", "gen"]
                message = f"there must be (dim * gen) number of particles"
                result.update({"score": 0, "error_var": error_var, "message": message})
            return result
        

        # check if the particles are consistent with the type
        def _particle_types():
            result = {"score": 1, "error_var": [], "message": "Passed", "max_score": 1}
            for p in self.particles:
                if p.type != self.type:
                    error_var = ["particles"]
                    message = f"this field is a {self.type} field, but the particle is a {p.type} particle."
                    result.update({"score": 0, "error_var": error_var, "message": message})
                    return result
            return result
        
        # check if self-conjugate is consistent with the charges
        def _particle_charges():
            result = {"score": 1, "error_var": [], "message": "Passed", "max_score": 1}
            error_var = []
            if self.self_conjugate:
                error_var = [f"particle:{p.id}:charge" for p in self.particles if p.charge != 0]
            if error_var:
                error_var.append("self_conjugate")
                message = f"this field is self-conjugate, but the particles have non-zero charges."
                result.update({"score": 0, "error_var": error_var, "message": message})
            return result
        
        # check if all particles pass all checks
        def _all_particle_pass():
            result = {"score": 1, "error_var": [], "message": "Passed", "max_score": 1}
            if self.type == "fermion":
                error_var = [f"particle:{p.name}" for p in self.particles if not p.fermion.pass_all_checks()]
            else: 
                error_var = [f"particle:{p.name}" for p in self.particles if not p.pass_all_checks()]
            if error_var:
                message = f"the following particles do NOT pass ALL checks: {error_var}"
                result.update({"score": 0, "error_var": error_var, "message": message})
            return result

        # Sort particle ordering
        def _sort_particles():
            result = {"score": 2, "error_var": [], "message": "Passed", "max_score": 2}
            error_var = []

            charge_eigenvec = {q: [] for q in self.allowed_Q}

            for p in self.particles:
                if p.charge not in self.allowed_Q:
                    error_var.extend([f"particle:{p.name}", "reps:g1", "reps:g2"])
                    message = f"Charge {p.charge} is not in allowed charges"
                    result.update({"score": 0, "error_var": error_var, "message": message})
                    return result

                charge_eigenvec[p.charge].append(p)
            
            charge_eigenval = sorted(charge_eigenvec.keys(), reverse=True)
            for _, ptcls in charge_eigenvec.items():
                ptcls.sort(key=lambda p: p.mass)

            if len(charge_eigenval) != self.dim:
                error_var.append("particles")
                message = f"Particle number is not consistent with the dim"
                result.update({"score": 1, "error_var": error_var, "message": message})
                return result
        
            self._unphy_fields = np.column_stack([charge_eigenvec[charge] for charge in charge_eigenval])
            self._phy_fields = np.array(self._unphy_fields).transpose().tolist()

            return result

        if self.simplify_checklist:
            self.all_checks = [_sort_reps,
                               _reps_dim_consistency,
                               _gen_type_consistency,
                               _allowed_charges,
                               _deplicate_particles,
                               _particle_numbers, 
                               _particle_types,
                               _particle_charges,
                               _all_particle_pass,
                               _sort_particles
                               ]
        else:
            self.all_checks = [_id_check, 
                               _name_check, 
                               _type_check, 
                               _groups_check, 
                               _reps_check, 
                               _dim_check, 
                               _gen_check, 
                               _particles_check, 
                               _self_conjugate_check, 
                               _sort_reps,
                               _reps_dim_consistency,
                               _gen_type_consistency,
                               _allowed_charges,
                               _deplicate_particles,
                               _particle_numbers, 
                               _particle_types,
                               _particle_charges,
                               _all_particle_pass,
                               _sort_particles
                               ]

    def __check__(self):
        """ Input checks for the field class. """
        self.checklist = {}
        self._all_checks()
        run_checks(self.all_checks, self.checklist)

    def _all_validations(self):
        """ All validations for the 'Field' class. """

        def _mass_term_check():
            result = {"score": 10, "error_var": [], "message": "Passed", "max_score": 10}
            if self.type == "fermion":
                self.is_massive = any(p.fermion.mass != 0 for p in self.particles)
            else:
                self.is_massive = any(p.mass != 0 for p in self.particles)
            if self.is_massive and self.mass_term == []:
                error_var = ["interactions"]
                message = f"this field is massive, but no mass term is defined"
                result.update({"score": 0, "error_var": error_var, "message": message})
            elif not self.is_massive and self.mass_term != []:
                error_var = ["interactions"]
                message = f"this field is massless, but a mass term is defined"
                result.update({"score": 0, "error_var": error_var, "message": message})
            return result
        
        self.all_validations = [_mass_term_check]

    def __validate__(self):
        self._all_validations()
        run_checks(self.all_validations, self.checklist, skip_results = True)

    @property
    def score(self):
        max_score = sum(value["max_score"] for value in self.checklist.values())
        score = sum(value["score"] for value in self.checklist.values())
        return score, max_score

    def pass_all_checks(self):
        score, max_score = self.score
        return score == max_score
        

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
    def __init__(self, id, name, groups, reps, dim, gen, particles, self_conjugate, chirality, simplify_checklist):
        self.chirality = chirality
        self._unphy_fields = None
        self._phy_fields = None
        self.color = 1
        super().__init__(id, name, "fermion", groups, reps, dim, gen, particles, self_conjugate, simplify_checklist)
        

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
            result = {"score": 1, "error_var": [], "message": "Passed", "max_score": 1}
            if self.chirality not in ["left", "right"]:
                error_var = ["chirality"]
                message = f"chirality must be left or right"
                result.update({"score": 0, "error_var": error_var, "message": message})
            return result
        
        # ------------------------------------------------------------------
        
        # Assign colors to the particles
        def _assign_colors():
            result = {"score": 1, "error_var": [], "message": "Passed", "max_score": 1}
            try:
                for p in self.particles:
                    if self.color is not None:
                        p.fermion.color = self.color
            except Exception as e:
                error_var = ["particles"]
                message = f"Error: {e}"
                result.update({"score": 0, "error_var": error_var, "message": message})
            return result

        if self.simplify_checklist:
            self.all_checks.extend([
                _assign_colors
            ])
        else:
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

            self.full_reps["g1"] = - self.full_reps["g1"]
    
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
    def __init__(self, id, name, type, groups, reps, dim, gen, particles, self_conjugate, simplify_checklist):
        self.chirality = None
        super().__init__(id, name, type, groups, reps, dim, gen, particles, self_conjugate, simplify_checklist)
        self.potential = None
        self.vev = None
        self.get_vev = False
        self.potential = []
        

    def _all_validations(self):
        """ All validations for the 'Field' class. """
        super()._all_validations()
        
        def _potential_term_check():
            result = {"score": 10, "error_var": [], "message": "Passed", "max_score": 10}
            if self.potential == []:
                error_var = ["interactions"]
                message = f"scalar field must have a potential term"
                result.update({"score": 0, "error_var": error_var, "message": message})
            return result
        
        self.all_validations.extend([_potential_term_check])

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
    def __init__(self, id, name, groups, reps, dim, gen, particles, self_conjugate, simplify_checklist):
        super().__init__(id, name, "vector", groups, reps, dim, gen, particles, self_conjugate, simplify_checklist)
        self.__check__()

    def __check__(self):
        super().__check__()

# ------------------------------------------------------------------
if __name__ == "__main__":
    pass