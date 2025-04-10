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
        self.label = f"U{dim}"
        self.abelian = True

        self._validate()
        self.description = f"Unitary Group of dimension {dim}"

    def _validate(self):
        if self.dim != 1:
            raise ValueError("Dimension must be 1")

        if type(self.dim) != int:
            raise ValueError("Dimension must be an integer")
        
    def __str__(self):
        return self.description

class SU(Group):
    """
    Special Unitary Group
    """
    def __init__(self, dim):
        super().__init__(f"SU({dim})")
        self.dim = dim
        self.label = f"SU{dim}"
        self.abelian = False
        self._validate()
        self.description = f"Special Unitary Group of dimension {dim}"

    def _validate(self):
        if self.dim < 2:
            raise ValueError("Dimension must be at least 2")

        if type(self.dim) != int:
            raise ValueError("Dimension must be an integer")

    def __str__(self):
        return self.description
    
    def adjoint_rep(self):
        return self.dim**2 - 1
    
    def fundamental_rep(self):
        return self.dim
    
    def anti_fundamental_rep(self):
        return self.dim
    
    def symmetric_rep(self):
        return self.dim * (self.dim + 1) // 2
    
    def antisymmetric_rep(self):
        return self.dim * (self.dim - 1) // 2
    
    def get_rep_dimension(self):
        return {
            "adj": self.adjoint_rep(),
            "fnd": self.fundamental_rep(),
            "anti-fnd": self.anti_fundamental_rep(),
            "sym": self.symmetric_rep(),
            "asym": self.antisymmetric_rep(),
        }

    def reps(self):
        return list(self.irreps().values())

    def generators(self):
        
        N = self.dim
        
        if N < 2:
            raise ValueError("Dimension must be at least 2")
    
        elif N >= 2:
            from sympy import Matrix, zeros, sqrt, Rational
            generators = []
            
            # Symmetric generators
            for i in range(N):
                for j in range(i+1, N):
                    matrix = zeros(N, N)
                    matrix[i, j] = 1
                    matrix[j, i] = 1
                    generators.append(matrix)
            
            # Anti-symmetric generators
            for i in range(N):
                for j in range(i+1, N):
                    matrix = zeros(N, N)
                    matrix[i, j] = -1j
                    matrix[j, i] = 1j
                    generators.append(matrix)
            
            # Diagonal generators
            for d in range(1, N):
                matrix = zeros(N, N)
                norm = sqrt(Rational(2, d*(d+1)))
                
                for i in range(d):
                    matrix[i, i] = norm
                matrix[d, d] = -d * norm
                
                generators.append(matrix)
                
            # Normalize
            for i in range(len(generators)):
                generators[i] = generators[i] / sqrt(2) 

            return generators
        else: 
            raise ValueError("Invalid dimension.")

class GaugeGroup():
    """
    Base class for gauge groups
    """
    def __init__(self, name, label, group, confinement = False):
        self.name = name
        self.label = label
        self.group = group
        self.type = group.label
        self.dim = group.dim
        self.abelian = group.abelian
        self.confinement = confinement

    def __str__(self):
        return self.name
    
    def reps(self):
        return self.group.reps()

    def irreps(self):
        return self.group.irreps()
    
    def generators(self):
        return self.group.generators()
    
    def abelian(self):
        return self.group.abelian
    
    def get_rep_dimension(self):
        return self.group.get_rep_dimension()
    
    def create_gauge_boson(self):
        pass 

class GaugeSector:
    """
    Gauge sector of the SM
    """
    def __init__(self, gauge_groups):
        self.gauge_groups = gauge_groups
        self.num_groups = len(self.gauge_groups)
        self.group_names = self.get_group_names()

    def get_group_names(self):
        return [group.name for group in self.gauge_groups]