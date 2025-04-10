class Particle:
    """
    Base class for particles
    """
    def __init__(self, type, id, name, label, mass, self_conjugate, QuantumNumber):
        self.id = id
        self.type = type # scalar/fermion/vector

        if type == "scalar":
            self.spin = 0
        elif type == "fermion":
            self.spin = 1/2
        elif type == "vector":
            self.spin = 1

        self.name = name 
        self.label = label
        self.mass = mass
        self.self_conjugate = self_conjugate
        self.QuantumNumber = QuantumNumber

    def get_id(self):
        return self.id
    
    def get_name(self):
        return self.name
    
    def get_label(self):
        return self.label

    def get_mass(self):
        return self.mass
    
    def is_self_conjugate(self):
        return self.self_conjugate
    
    def get_QuantumNumber(self):
        return self.QuantumNumber

    def get_spin(self):
        return self.spin

class ComplexScalar(Particle):
    """
    Complex scalar particle
    """
    def __init__(self, id, scalar_type, name, label, mass, self_conjugate, QuantumNumber):
        super().__init__("scalar", id, name, label, mass, self_conjugate, QuantumNumber)    
        assert self_conjugate == False
        assert scalar_type == "complex"
    
    def __str__(self):
        return f"ComplexScalar(id={self.id}, name=\"{self.name}\", label=\"{self.label}\", mass={self.mass}, self_conjugate={self.self_conjugate}, QuantumNumber={self.QuantumNumber})"

    def to_real(self):
        pass
    
class RealScalar(Particle):
    """
    Real scalar particle
    """
    def __init__(self, id, name, label, mass, self_conjugate, QuantumNumber):
        super().__init__("scalar", id, name, label, mass, self_conjugate, QuantumNumber)
        assert self_conjugate == True

    def __str__(self):
        return f"RealScalar(id={self.id}, name=\"{self.name}\", label=\"{self.label}\", mass={self.mass}, self_conjugate={self.self_conjugate}, QuantumNumber={self.QuantumNumber})"
    

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
                 scalar_type = "real",
                 name = "phi", 
                 label = "phi", 
                 goldstone = None,
                 mass = 0, 
                 self_conjugate = True, 
                 QuantumNumber = None):
        super().__init__("scalar", id, name, label, mass, self_conjugate, QuantumNumber)
        self.scalar_type = scalar_type
        self.goldstone = goldstone
        
        if not self_conjugate:
            Q = abs(self.QuantumNumber["Q"])
            self.positive_charge = ComplexScalar(id + "_p", 
                                                 scalar_type=scalar_type,
                                                 name=name,
                                                 label=label + "+",
                                                 mass=mass,
                                                 self_conjugate=False,
                                                 QuantumNumber={"Q": Q})
            
            self.negative_charge = ComplexScalar(id + "_m", 
                                                 scalar_type=scalar_type,
                                                 name="anti-" + name,
                                                 label=label + "-",
                                                 mass=mass,
                                                 self_conjugate=False,
                                                 QuantumNumber={"Q": -Q})

    def plus(self):
        try:
            return self.positive_charge
        except:
            return None
    
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
                 name = "psi", 
                 label = "psi", 
                 mass = 0, 
                 self_conjugate = False, 
                 spin_state = "left",
                 is_antiparticle = False,
                 QuantumNumber = None):
        super().__init__("fermion", id, name, label, mass, self_conjugate, QuantumNumber)
        assert spin_state in ["left", "right"]
        self.spin_state = spin_state
        self.is_antiparticle = is_antiparticle
    
    def handedness(self):
        return self.spin_state
    
    def __str__(self):
        return f"WeylSpinor(id={self.id}, name=\"{self.name}\", label=\"{self.label}\", mass={self.mass}, self_conjugate={self.self_conjugate}, chirality=\"{self.spin_state}\", is_antiparticle={self.is_antiparticle}, QuantumNumber={self.QuantumNumber})"
    
    def get_fermion(self):
        if self.handedness() == "left":
            fermion = Fermion(self.id[:-2], 
                                  name = self.name[3:], 
                                  label = self.label[:-2], 
                                  mass = self.mass, 
                                  self_conjugate = self.self_conjugate, 
                                  is_antiparticle = self.is_antiparticle,
                                  QuantumNumber = self.QuantumNumber)
        elif self.handedness() == "right":
            fermion = Fermion(self.id[:-2], 
                                  name = self.name[3:], 
                                  label = self.label[:-2], 
                                  mass = self.mass, 
                                  self_conjugate = self.self_conjugate, 
                                  is_antiparticle = self.is_antiparticle,
                                  QuantumNumber = self.QuantumNumber)
        return fermion
    
class Fermion(Particle):
    """
    Fermion particle

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
                 name, 
                 label, 
                 mass, 
                 self_conjugate, 
                 is_antiparticle,
                 QuantumNumber):
        super().__init__("fermion", id, name, label, mass, self_conjugate, QuantumNumber)        
    
        if self_conjugate:
            assert is_antiparticle is None
        else:
            assert is_antiparticle is not None
            self.is_antiparticle = is_antiparticle

        self.left_component = WeylSpinor(id + "_l", 
                                  name = "LH " + name, 
                                  label = label + "_L", 
                                  mass = mass, 
                                  self_conjugate = self_conjugate, 
                                  spin_state = "left", 
                                  is_antiparticle = is_antiparticle,
                                  QuantumNumber = QuantumNumber)
        
        self.right_component = WeylSpinor(id + "_r", 
                                   name = "RH " + name, 
                                   label = label + "_R", 
                                   mass = mass, 
                                   self_conjugate = self_conjugate, 
                                   spin_state = "right", 
                                   is_antiparticle = is_antiparticle,
                                   QuantumNumber = QuantumNumber)
    
    def left(self):
        return self.left_component
    
    def right(self):
        return self.right_component
    
    def __str__(self):
        return f"Fermion(id={self.id}, name=\"{self.name}\", label=\"{self.label}\", mass={self.mass}, self_conjugate={self.self_conjugate}, is_antiparticle={self.is_antiparticle}, QuantumNumber={self.QuantumNumber})"
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "label": self.label,
            "mass": self.mass,
            "self_conjugate": self.self_conjugate,
            "is_antiparticle": self.is_antiparticle,
            "QuantumNumber": self.QuantumNumber
        }

    def antiparticle(self):
        anti_QuantumNumber = self.QuantumNumber.copy()  
        
        Q = anti_QuantumNumber["Q"]
        
        if isinstance(Q, int) or isinstance(Q, float):
            anti_QuantumNumber["Q"] = -Q
        elif isinstance(Q, str):
            if Q.startswith("-"):
                anti_QuantumNumber["Q"] = Q[1:]
            else:
                anti_QuantumNumber["Q"] = "-" + Q
        else:
            raise ValueError("Invalid Q quantum number.")
        
        if not self.is_antiparticle:
            anti_fermion = Fermion(self.id + "~", 
                                    name = "anti-" + self.name, 
                                    label = self.label + "~", 
                                    mass = self.mass, 
                                    self_conjugate = self.self_conjugate, 
                                    is_antiparticle = True,
                                    QuantumNumber = anti_QuantumNumber)
        else:
            anti_fermion = Fermion(self.id[:-1], 
                                    name = self.name[5:], 
                                    label = self.label[:-1], 
                                    mass = self.mass, 
                                    self_conjugate = self.self_conjugate, 
                                    is_antiparticle = False,
                                    QuantumNumber = anti_QuantumNumber)
        return anti_fermion
    
class VectorBoson(Particle):
    """
    Vector boson

    input:
    id: [string]
    name: [string]
    label: [string]
    mass: [float]
    self_conjugate: [bool]
    abelian: [bool]
    QuantumNumber: [dict]
    """
    def __init__(self, 
                 id, 
                 name = "A", 
                 label = "A", 
                 mass = 0, 
                 self_conjugate = False, 
                 abelian = True,
                 QuantumNumber = None):
        super().__init__("vector", id, name, label, mass, self_conjugate, QuantumNumber)
        self.abelian = abelian

    def is_abelian(self):
        return self.abelian
