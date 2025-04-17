class Particle:
    """
    Base class for particles
    """
    def __init__(self, type, id, full_name, name, mass, self_conjugate, QuantumNumber):
        self.id = id
        self.full_name = full_name
        self.type = type

        if type == "scalar":
            self.spin = 0
        elif type == "fermion":
            self.spin = 1/2
        elif type == "vector":
            self.spin = 1
        else:
            raise ValueError(f"Invalid particle type: {type}")

        self.name = name 
        self.mass = mass
        self.self_conjugate = self_conjugate
        self.QuantumNumber = QuantumNumber
        self.flavor = None
        self.color = None
        self.charge = self.QuantumNumber['Q']

    @property
    def pdg_id(self):
        from pdg_id import get_pdg_id
        if self.color is not None:
            return get_pdg_id(mass=self.mass, 
                              charge=self.charge, 
                              color=self.color, 
                              spin=self.spin * 2,
                              flavor=self.flavor
                              )
        else:
            return None


class ComplexScalar(Particle):
    """
    Complex scalar particle
    """
    def __init__(self, 
                 id, 
                 full_name, 
                 name, 
                 mass, 
                 self_conjugate, 
                 QuantumNumber
                 ):
        super().__init__("scalar", id, full_name, name, mass, self_conjugate, QuantumNumber)    
        assert self_conjugate == False
        self.scalar_type = "complex"
    
    def __str__(self):
        return f"ComplexScalar(id={self.id}, full_name=\"{self.full_name}\", name=\"{self.name}\", mass={self.mass}, self_conjugate={self.self_conjugate}, QuantumNumber={self.QuantumNumber})"

    def to_real(self):
        pass
    
class RealScalar(Particle):
    """
    Real scalar particle
    """
    def __init__(self, 
                 id, 
                 full_name, 
                 name, 
                 mass, 
                 self_conjugate, 
                 QuantumNumber
                 ):
        super().__init__("scalar", id, full_name, name, mass, self_conjugate, QuantumNumber)
        assert self_conjugate == True
        self.scalar_type = "real"

    def __str__(self):
        return f"RealScalar(id={self.id}, full_name=\"{self.full_name}\", name=\"{self.name}\", mass={self.mass}, self_conjugate={self.self_conjugate}, QuantumNumber={self.QuantumNumber})"
    

class Scalar(Particle):
    """
    Scalar particle
    input:
    id: [string]
    name: [string]
    label: [string]
    mass: [float]
    self_conjugate: [bool]
    QuantumNumber: [dict]
    """
    def __init__(self, 
                 id,
                 scalar_type,
                 full_name, 
                 name, 
                 mass, 
                 self_conjugate, 
                 QuantumNumber):
        super().__init__("scalar", id, full_name, name, mass, self_conjugate, QuantumNumber)
        self.scalar_type = scalar_type
    
        if not self_conjugate:
            Q = abs(self.QuantumNumber["Q"])
            self.positive_charge = ComplexScalar(id + "_p", 
                                                 full_name=full_name,
                                                 name=name + "+",
                                                 mass=mass,
                                                 self_conjugate=False,
                                                 QuantumNumber={"Q": Q})
            
            self.negative_charge = ComplexScalar(id + "_m", 
                                                 full_name=full_name,
                                                 name="anti-" + name,
                                                 mass=mass,
                                                 self_conjugate=False,
                                                 QuantumNumber={"Q": -Q})
    @property
    def plus(self):
        try:
            return self.positive_charge
        except:
            return None
    
    @property
    def minus(self):
        try:
            return self.negative_charge
        except:
            return None
    
    def __str__(self):
        return f"Scalar(id={self.id}, scalar_type=\"{self.scalar_type}\", name=\"{self.name}\", label=\"{self.label}\", mass={self.mass}, self_conjugate={self.self_conjugate}, QuantumNumber={self.QuantumNumber}, goldstone={self.goldstone})"



class WeylSpinor(Particle):
    """
    Chiral fermion particle

    input:
    id: [string]
    name: [string]
    label: [string]
    mass: [float]
    self_conjugate: [bool]
    chirality: [string]
    QuantumNumber: [dict]
    """
    def __init__(self, 
                 id, 
                 full_name, 
                 name, 
                 mass = 0, 
                 self_conjugate = False, 
                 spin_state = "left",
                 QuantumNumber = None):
        super().__init__("fermion", id, full_name, name, mass, self_conjugate, QuantumNumber)
        assert spin_state in ["left", "right"]
        self.spin_state = spin_state
        self.color = None
    
    @property
    def handedness(self):
        return self.spin_state
    
    def __str__(self):
        return f"WeylSpinor(id={self.id}, full_name=\"{self.full_name}\", name=\"{self.name}\", mass={self.mass}, self_conjugate={self.self_conjugate}, chirality=\"{self.spin_state}\", QuantumNumber={self.QuantumNumber})"
    
    def get_fermion(self):
        if self.handedness() == "left":
            fermion = Fermion(self.id[:-2], 
                                  full_name = self.full_name, 
                                  name = self.name[3:], 
                                  mass = self.mass, 
                                  self_conjugate = self.self_conjugate, 
                                  QuantumNumber = self.QuantumNumber)
        elif self.handedness() == "right":
            fermion = Fermion(self.id[:-2], 
                                  full_name = self.full_name, 
                                  name = self.name[3:], 
                                  mass = self.mass, 
                                  self_conjugate = self.self_conjugate, 
                                  QuantumNumber = self.QuantumNumber)
        return fermion


class Fermion(Particle):
    """
    Fermion particle

    input:
    id: [string]
    name: [string]
    full_name: [string]
    mass: [float]
    self_conjugate: [bool]
    QuantumNumber: [dict]
    """
    def __init__(self, 
                 id, 
                 full_name, 
                 name, 
                 mass, 
                 self_conjugate, 
                 QuantumNumber):
        super().__init__("fermion", id, full_name, name, mass, self_conjugate, QuantumNumber)        

        self.left_component = WeylSpinor(id + "_l", 
                                  full_name = "LH " + full_name, 
                                  name = name + "_L", 
                                  mass = mass, 
                                  self_conjugate = self_conjugate, 
                                  spin_state = "left", 
                                  QuantumNumber = QuantumNumber)
        
        self.right_component = WeylSpinor(id + "_r", 
                                   full_name = "RH " + full_name, 
                                   name = name + "_R", 
                                   mass = mass, 
                                   self_conjugate = self_conjugate, 
                                   spin_state = "right", 
                                   QuantumNumber = QuantumNumber)
    
    def left(self):
        return self.left_component
    
    def right(self):
        return self.right_component
    
    def __str__(self):
        return f"Fermion(id={self.id}, full_name=\"{self.full_name}\", name=\"{self.name}\", mass={self.mass}, self_conjugate={self.self_conjugate}, QuantumNumber={self.QuantumNumber})"
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "full_name": self.full_name,
            "mass": self.mass,
            "self_conjugate": self.self_conjugate,
            "QuantumNumber": self.QuantumNumber
        }


class VectorBoson(Particle):
    """
    Vector boson

    input:
    id: [string]
    full_name: [string]
    name: [string]
    mass: [float]
    self_conjugate: [bool]
    abelian: [bool]
    QuantumNumber: [dict]
    """
    def __init__(self, 
                 id, 
                 full_name, 
                 name,
                 group, 
                 mass, 
                 self_conjugate, 
                 QuantumNumber):
        super().__init__("vector", id, full_name, name, mass, self_conjugate, QuantumNumber)
        
        self.abelian = group.abelian
        self.group = group

