### ========================================================================== ###
###                                                                            ###
###                           Particles Classes                                ###
###                                                                            ###
### ========================================================================== ###
from .check import run_checks

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
        self._width = "Automatic" # _width can be overridden by the pdg_info
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
        
        from functools import wraps

        def _finalize_result(result: dict) -> dict:
            good_var = result.get('good_var', []) or []
            error_var = result.get('error_var', []) or []
            mattered = list({*good_var, *error_var})
            result['mattered_vars'] = mattered
            # If neither good nor error vars were specified, treat as block-level
            result['level'] = 'particle' if mattered else 'block'
            return result
        
        def _wrap(fn):
            @wraps(fn)
            def wrapped():
                return _finalize_result(fn())
            return wrapped
        
        def _type_check():
            result = {"score": 1, 
                      "error_var": [], 
                      "good_var": [f"particles.{self.id}.type"], 
                      "message": "Passed", 
                      "max_score": 1,
                      }
            if self.type not in allowed_particle_types:
                result.update({"score": 0, 
                               "error_var": [f"particles.{self.id}.type"], 
                               "good_var": [], 
                               "message": f"Type must be one of scalar/real/pseudo/complex/fermion/vector"
                               })
            return result

        def _name_check():
            result = {"score": 1, 
                      "error_var": [], 
                      "good_var": [f"particles.{self.id}.name"], 
                      "message": "Passed", 
                      "max_score": 1,
                      }
            if not isinstance(self.name, str):
                result.update({"score": 0, 
                               "error_var": [f"particles.{self.id}.name"], 
                               "good_var": [], 
                               "message": f"Name must be a string"
                               })
            return result

        def _mass_check():
            result = {"score": 1, 
                      "error_var": [], 
                      "good_var": [f"particles.{self.id}.mass"], 
                      "message": "Passed", 
                      "max_score": 1,
                      }
            if not (isinstance(self.mass, float) or isinstance(self.mass, int)) or self.mass < 0:
                result.update({"score": 0, 
                               "error_var": [f"particles.{self.id}.mass"], 
                               "good_var": [], 
                               "message": f"Mass must be a number and non-negative"
                               })
            return result

        def _charge_check():
            result = {"score": 1, 
                      "error_var": [], 
                      "good_var": [f"particles.{self.id}.charge"], 
                      "message": "Passed", 
                      "max_score": 1,
                      }
            if not isinstance(self.charge, int):
                result.update({"score": 0, 
                               "error_var": [f"particles.{self.id}.charge"], 
                               "good_var": [], 
                               "message": f"Charge must be an integer"
                               })
            return result
    
        self.all_checks = [(_wrap(_type_check), 1), 
                           (_wrap(_name_check), 1), 
                           (_wrap(_mass_check), 1), 
                           (_wrap(_charge_check), 1)
                           ]

    def __check__(self):
        """ Input checks for the particle class. """
        self.checklist = {}
        self._all_checks()
        run_checks(self.all_checks, self.checklist)

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

    @property
    def score(self):
        max_score = sum(value["max_score"] for value in self.checklist.values())
        score = sum(value["score"] for value in self.checklist.values())
        return score, max_score

    def _all_validations(self):
        self.all_validations = []

    def __validate__(self):
        self._all_validations()
        run_checks(self.all_validations, self.checklist, skip_results=True)

    def pass_all_checks(self):
        score, max_score = self.score
        return score == max_score

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
    
    def __repr__(self):
        return f"{self.name}"

# ====================================================================
#                              Fermion
# ====================================================================
class Fermion(Particle):
    def __init__(self, id, name, mass, charge):
        super().__init__(id, name, "fermion", mass, charge)
        self.left = WeylSpinor(self, "left")
        self.right = WeylSpinor(self, "right")



# ====================================================================
#                            Real Scalar
# ====================================================================
class RealScalar(Particle):
    def __init__(self, id, name, mass, charge):
        super().__init__(id, name, "real", mass, charge)



# ====================================================================
#                           Complex Scalar
# ====================================================================
class ComplexScalar(Particle):
    def __init__(self, id, name, mass, charge):
        super().__init__(id, name, "complex", mass, charge)


# ====================================================================
#                           Vector Boson
# ====================================================================
class VectorBoson(Particle):
    def __init__(self, id, name, mass, charge):
        super().__init__(id, name, "vector", mass, charge)



# ------------------------------------------------------------------
if __name__ == "__main__":
    pass