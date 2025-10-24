### =============================== ###
###          Group Classes          ###
### =============================== ###
import numpy as np
#from .check import run_checks

class Group:
    """
    Base class for groups
    """
    def __init__(self, name, N):
        self.name = str(name)
        self.N = N
        self.__check__()

    def __str__(self):
        return self.name
    
    def _all_checks(self):
        self.all_checks = []

    def __check__(self):
        self.checklist = {}
        self._all_checks()
        #run_checks(self.all_checks, self.checklist)

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
    def __init__(self, N):
        super().__init__(f"U({N})", N)
        self.type = "U"
        self.abelian = True

    def dim(self, rep):
        if rep == "singlet":
            return 1
        else:
            return 0

class SU(Group):
    """
    Special Unitary Group (only for SU(N))
    """
    def __init__(self, N):
        super().__init__(f"SU({N})", N)
        self.type = "SU"
        self.abelian = False

    # ---------- dim ----------
    def dim(self, rep):
        if rep == "singlet":
            return 1
        elif rep == "fnd":
            return self.N
        elif rep == "adj":
            return self.N**2 - 1
        else:
            return 0

    # ---------- generators ----------
    def generators(self, representation):
        def fnd_rep(N):
            generators = []
            pass

        def adj_rep(N):
            generators = []
            pass

        def anti_fnd_rep(N):
            generators = []
            pass

        def sym2_rep():
            generators = []
            pass

        def anti_sym2_rep():
            generators = []
            pass

        if representation == "fnd":
            return fnd_rep(self.N)
        elif representation == "adj":
            return adj_rep(self.N)
        elif representation == "anti_fnd":
            return anti_fnd_rep(self.N)
        elif representation == "sym2":
            return sym2_rep(self.N)
        elif representation == "anti_sym2":
            return anti_sym2_rep(self.N)
        else:
            raise ValueError(f"Invalid representation: {representation}")


    @property
    def fnd_rep(self):
        N = self.N
        generators = []

        # Off-diagonal generators (symmetric)
        for i in range(N):
            for j in range(i+1, N):
                mat = np.zeros((N, N), dtype=complex)
                mat[i, j] = 1
                mat[j, i] = 1
                generators.append(mat)

        # Off-diagonal generators (anti-symmetric)
        for i in range(N):
            for j in range(i+1, N):
                mat = np.zeros((N, N), dtype=complex)
                mat[i, j] = -1j
                mat[j, i] = 1j
                generators.append(mat)

        # Diagonal generators (Gell-Mann style normalization)
        for k in range(1, N):
            mat = np.zeros((N, N), dtype=complex)
            for i in range(k):
                mat[i, i] = 1
            mat[k, k] = -k
            mat *= np.sqrt(2 / (k*(k+1)))
            generators.append(mat)

        # Normalize the generators
        for idx, matrix in enumerate(generators):
            abs_matrix = np.abs(matrix)
            total_sum = np.sum(abs_matrix)
            generators[idx] = matrix / total_sum * 2

        return generators

    @property
    def adj_rep(self):
        dim = self.N**2 - 1

        fund_gens = self.fnd_rep
        # Compute structure constants f^{abc}
        f = np.zeros((dim, dim, dim), dtype=float)
        for a in range(dim):
            for b in range(dim):
                comm = 1j * (np.dot(fund_gens[a], fund_gens[b]) - np.dot(fund_gens[b], fund_gens[a]))
                for c in range(dim):
                    # Take the real part and divide by 2 for proper normalization
                    f[a, b, c] = np.trace(np.dot(comm, fund_gens[c])).real

        # Adjoint generators: (T^a)_{bc} = -i f^{abc}
        adj_gens = []
        for a in range(dim):
            mat = np.zeros((dim, dim), dtype=complex)
            for b in range(dim):
                for c in range(dim):
                    mat[b, c] = -1j * f[a, b, c]
            adj_gens.append(mat)

        # Normalize the generators
        for idx, matrix in enumerate(adj_gens):
            abs_matrix = np.abs(matrix)
            total_sum = np.sum(abs_matrix)
            adj_gens[idx] = matrix / total_sum * 2

        return adj_gens
    
    # ---------- Dynkin label ----------
    def Dynkin_label(self, rep):
        if rep == "singlet":
            return [0] * (self.N - 1)
        elif rep == "fnd":
            return [1] + [0] * (self.N - 2)
        elif rep == "anti_fnd":
            return [0] * (self.N - 2) + [1]
        elif rep == "adj":
            adjoint = [0] * (self.N - 1)
            adjoint[0] += 1
            adjoint[-1] += 1
            return adjoint
        else:
            raise ValueError(f"Invalid representation: {rep}")

    # ---------- Dynkin index ----------
    def Dynkin_index(self, rep):
        label = self.Dynkin_label(rep)
        dimR = self.dim(rep)

        G = [[0.0] * (self.N - 1) for _ in range(self.N - 1)]
        for i in range(1, self.N):
            for j in range(1, self.N):
                G[i - 1][j - 1] = min(i, j) - (i * j) / self.N

        a_vec = label.copy()
        v = [x + 2.0 for x in a_vec]
        w = [sum(G[i][j] * v[j] for j in range(self.N - 1)) for i in range(self.N - 1)]
        inner = sum(a_vec[i] * w[i] for i in range(self.N - 1))
        C2 = 0.5 * inner
        dimG = self.N**2 - 1
        T = (dimR * C2) / dimG

        return float(dimR), float(C2), float(T)

    # ---------- Cubic anomaly ----------
    def cubic_anomaly(self, rep):
        if rep == "singlet":
            return 0
        elif rep == "fnd":
            return 1 if self.N >= 3 else 0
        elif rep == "anti_fnd":
            return -1 if self.N >= 3 else 0
        elif rep == "adj":
            return 0
        else:
            raise ValueError(f"Invalid representation: {rep}")
    
class GaugeGroup:
    """
    Base class for gauge groups
    """
    def __init__(self, id, name, charge, group, boson):
        self.id = id
        self.name = str(name)
        self.charge = charge
        self.group = group
        self.boson = boson

        self.__check__()
        self.initialize_group()
            
        self.isSU3C = self.name == "color"
        self.isSU2L = self.name == "left"
        self.isU1Y = self.name == "hypercharge"
        self.abelian = self.group.abelian
    
    # ----------------------------Initialize-----------------------------
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
    
    def initialize_group(self):
        group_type, N = self.group.split("_")
        N = int(N)
        if group_type == "U":
            self.group = U(N)
        elif group_type == "SU":
            self.group = SU(N)
        else:
            raise ValueError(f"Invalid group type: {group_type}")
        
    # --------------------------Properties-----------------------------

    def dim(self, rep):
        return self.group.dim(rep)

    def Dynkin_label(self, rep):
        return self.group.Dynkin_label(rep)

    def Dynkin_index(self, rep):
        return self.group.Dynkin_index(rep)
    
    def cubic_anomaly(self, rep):
        return self.group.cubic_anomaly(rep)

    # ----------------------------Checks-----------------------------

    def _all_checks(self):
        self.all_checks = []

    def __check__(self):
        self.checklist = {}
        self._all_checks()
        #run_checks(self.all_checks, self.checklist)

if __name__ == "__main__":
    group = SU(3)
    print(group.Dynkin_label("singlet"))
    print(group.Dynkin_label("fnd"))
    print(group.Dynkin_label("anti_fnd"))
    print(group.Dynkin_label("adj"))
    print(group.Dynkin_index("singlet")[2])
    print(group.Dynkin_index("fnd")[2])
    print(group.Dynkin_index("anti_fnd")[2])
    print(group.Dynkin_index("adj")[2])

