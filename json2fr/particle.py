### ========================================================================== ###
###                                                                            ###
###                           Particles Classes                                ###
###                                                                            ###
### ========================================================================== ###

allowed_particle_types = ["scalar", "real", "pseudo", "complex", "fermion", "vector"]

### ======================= ###
###         Particle        ###
### ======================= ###
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
        return f"{self.__class__.__name__}(id={self.id}, type={self.type}, name={self.name}, mass={self.mass}, charge={self.charge}, color={self.color}, flavor={self.flavor})"
 
    def __check__(self):
        type_check = self.type in allowed_particle_types
        id_check = isinstance(self.id, str)
        name_check = isinstance(self.name, str)
        mass_check = (isinstance(self.mass, float) or isinstance(self.mass, int)) and self.mass >= 0
        charge_check = isinstance(self.charge, int)
    
        self.checklist = {"id": id_check, 
                          "type": type_check, 
                          "name": name_check, 
                          "mass": mass_check, 
                          "charge": charge_check}
    
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
        from utility import get_pdg
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
        from utility import BSM_id
        if self.pdg_info is not None:
            return self.pdg_info['pdgid']
        else:
            return BSM_id(self.mass, self.charge, self.color, self.spin * 2, self.flavor)

    def _check_color(self):
        assert self.color in [1, 3, 8], f"AssertionError: Color must be 1, 3, or 8"  

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

### WeylSpinor ###
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

### ======================== ###
###         Fermion          ###
### ======================== ###
class Fermion(Particle):
    def __init__(self, id, name, mass, charge):
        super().__init__(id, name, "fermion", mass, charge)
        self.__check__()
        
    def __check__(self):
        super().__check__()
        
        try:
            self._WeylSpinor()
            check_WeylSpinor = True
        except Exception as e:
            check_WeylSpinor = e

        self.checklist["WeylSpinor"] = check_WeylSpinor

    def _WeylSpinor(self):
        assert self.type == "fermion", f"AssertionError: Only fermions can be Weyl spinors"
        self.left = WeylSpinor(self, "left")
        self.right = WeylSpinor(self, "right")


### ========================== ###
###         Real Scalar        ###
### ========================== ###
class RealScalar(Particle):
    def __init__(self, id, name, mass, charge):
        super().__init__(id, "real", name, mass, charge)

### ============================ ###
###         Complex Scalar       ###
### ============================ ###
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



### ============================ ###
###         Vector Boson         ###
### ============================ ###
class VectorBoson(Particle):
    def __init__(self, id, name, mass, charge):
        super().__init__(id, "vector", name, mass, charge)





if __name__ == "__main__":
    from utility import random_inputs

    random_values = random_inputs()
    particle = ComplexScalar(random_values[0],random_values[1],random_values[2],random_values[3])
    
    print(f"ParticleName: {particle}")
    print(f"Particle: {repr(particle)}")
    print(f"ParticleScore: {particle.score}")

    for key, value in particle.checklist.items():
        print(f"{key:20} -> {value}")
