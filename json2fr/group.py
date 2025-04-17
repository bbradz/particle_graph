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

    @property
    def adjoint_rep(self):
        return self.dim**2

    @property
    def fnd_generators(self):
        if self.dim == 1:
            return 1
        if self.dim == 2:
            sigma0 = np.identity(2) * 0.5
            sigma1 = np.array([[0, 1], [1, 0]]) * 0.5
            sigma2 = np.array([[0, -1j], [1j, 0]]) * 0.5
            sigma3 = np.array([[1, 0], [0, -1]]) * 0.5
            return [sigma0, sigma1, sigma2, sigma3]
        else:
            raise ValueError("Dimension must be 1 or 2")


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

    @property
    def fnd_generators(self):
        if self.dim == 2:
            sigma1 = np.array([[0, 1], [1, 0]]) * 0.5
            sigma2 = np.array([[0, -1j], [1j, 0]]) * 0.5
            sigma3 = np.array([[1, 0], [0, -1]]) * 0.5
            return [sigma1, sigma2, sigma3]
        
        elif self.dim == 3:

            lambda1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
            lambda2 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
            lambda3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
            lambda4 = np.array([[0, 0, 1], [0, 0, -1j], [1j, 0, 0]])
            lambda5 = np.array([[0, 0, -1j], [0, 0, 1], [-1j, 0, 0]])
            lambda6 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
            lambda7 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
            lambda8 = np.array([[0, 0, 0], [0, 0, 1j], [-1j, 0, 0]]) * (1/np.sqrt(3))

            return [lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7, lambda8]
        
        else:
            raise ValueError("Dimension must be 2 or 3")
            
            





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
    def fnd_generators(self):
        return self.group.fnd_generators

    @property
    def dim(self):
        return self.group.dim

    @property
    def adjoint_rep(self):
        return self.group.adjoint_rep

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
