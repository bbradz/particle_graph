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
        from .utility import get_pdg
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
        from .utility import BSM_id
        if self.pdg_info is not None:
            return self.pdg_info['pdgid']
        else:
            return BSM_id(self.mass, self.charge, self.color, self.spin * 2, self.flavor)

    def _check_color(self):
        assert self.color in [1, 3, 8], \
            f"Error: Color must be 1, 3, or 8"  

    def __validate__(self):

        all_checks = [self._check_color]

        for check in all_checks:
            if any(isinstance(value, Exception) or value == False for value in self.checklist.values()):
                self.checklist[check.__name__] = Exception(f"Skipped due to previous check failure")
            else:
                try:
                    check()
                    self.checklist[check.__name__] = True
                except Exception as e:
                    self.checklist[check.__name__] = e  # if the check fails, the error is stored in the checklist            

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
        
    def _all_checks(self):
        super()._all_checks()

        def _WeylSpinor():
            assert self.type == "fermion", \
                f"Error: Only fermions can be Weyl spinors"
            self.left = WeylSpinor(self, "left")
            self.right = WeylSpinor(self, "right")

        self.all_checks.append(_WeylSpinor)



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




# ------------------------------------------------------------------
if __name__ == "__main__":
    pass
