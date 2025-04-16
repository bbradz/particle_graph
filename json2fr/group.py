import numpy as np

class Group:
    """
    Base class for groups
    """
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name
    
    def _validate(self):
        pass
    
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
        super().__init__(f"U({dim})")
        self.dim = dim
        self.group_type = "U"
        self.group_name = f"U({dim})"
        self.abelian = True
        self.validate()

    def __str__(self):
        return f"{self.group_name}: Unitary Group of dimension {self.dim}"

    def validate(self):
        if self.dim != 1:
            raise ValueError("Dimension must be 1")
        if type(self.dim) != int:
            raise ValueError("Dimension must be an integer")



class SU(Group):
    """
    Special Unitary Group
    """
    def __init__(self, dim):
        super().__init__(f"SU({dim})")
        self.dim = dim
        self.group_type = "SU"
        self.group_name = f"SU({dim})"
        self.abelian = False
        self.validate()

    def __str__(self):
        return f"{self.group_name}: Special Unitary Group of dimension {self.dim}"
    
    def validate(self):
        if self.dim < 2:
            raise ValueError("Dimension must be at least 2")
        if type(self.dim) != int:
            raise ValueError("Dimension must be an integer")

    @property
    def structure_constants(self):
        if self.dim == 2:
            return "Eps"
        elif self.dim == 3:
            return "f"
        else:
            raise ValueError("Current Version only supports SU(2) and SU(3)")
        
    @property
    def adjoint_rep(self):
        return self.dim**2 - 1
    
    @property
    def fundamental_rep(self):
        return self.dim
    
    @property
    def symmetric_rep(self):
        return self.dim * (self.dim + 1) // 2

    @property
    def irreps(self):
        return {
            "adj": self.adjoint_rep,
            "fnd": self.fundamental_rep,
            "sym": self.symmetric_rep,
        }



class GaugeGroup:
    """
    Base class for gauge groups
    """
    def __init__(self, id, name, charge, group, coupling_constant, confinement, boson):
        self.id = id
        self.name = name
        self.charge = charge
        self.group = group

        self.definition = None
        self.sym_tensors = None

        self.coupling_constant = coupling_constant
        self.confinement = confinement
        self.boson = boson
        self.reps = []
        
        self._set_SM()
        
    def __str__(self):
        return f"{self.group.name} Gauge Group"

    def _set_SM(self):
        # Definition and Symmetric Tensor of SU(2) and SU(3) in FR are fixed
        if self.name == "SU2L":
            self.definition = "{Ta[a_,b_,c_]->PauliSigma[a,b,c]/2, FSU2L[i_,j_,k_]:> I Eps[i,j,k]}"
            self.reps = ["Ta"]

        elif self.name == "SU3C":
            self.sym_tensors = "dSUN"
            self.reps = ["T"]

    @property
    def dim(self):
        return self.group.dim
    
    @property
    def group_type(self):
        return self.group.group_type
    
    @property
    def abelian(self):
        return self.group.abelian
    
    @property
    def structure_constants(self):
        return self.group.structure_constants

    @property
    def rep_list(self):
        rep_list = str(self.reps)
        rep_list = rep_list.replace("'", "").replace("[", "{").replace("]", "}")
        return rep_list
