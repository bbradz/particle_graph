### =============================== ###
###          Group Classes          ###
### =============================== ###

import numpy as np

class Group:
    """
    Base class for groups
    """
    def __init__(self, name, dim):
        self.name = name
        self.dim = dim
        self.__check__()

    def __str__(self):
        return self.name
    
    def __check__(self):
        pass

    def __validate__(self):
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
        super().__init__(f"U({dim})", dim)
        self.type = "U"
        self.abelian = True
        self.__check__()

    def __check__(self):
        dim_check = self.dim == 1
        self.checklist = {"dim": dim_check}

    @property
    def fnd_rep(self):
        return self.dim
    
    @property
    def adj_rep(self):
        return self.dim**2
    
    @property
    def rep_list(self):
        return {"singlet": 1,
                "fnd": self.fnd_rep, 
                "adj": self.adj_rep}
    
class SU(Group):
    """
    Special Unitary Group (only for SU(N))
    """
    def __init__(self, dim):
        super().__init__(f"SU({dim})", dim)
        self.type = "SU"
        self.abelian = False
        self.__check__()

    def __check__(self):
        dim_check = isinstance(self.dim, int) and self.dim > 1 and self.dim <= 3
        self.checklist = {"dim": dim_check}

    @property
    def fnd_rep(self):
        return self.dim
    
    @property
    def adj_rep(self):
        return self.dim**2 - 1
    
    @property
    def rep_list(self):
        return {"singlet": 1,
                "fnd": self.fnd_rep, 
                "adj": self.adj_rep}
    
    @property
    def structure_constants(self):
        if self.dim == 2:
            return "Eps"
        elif self.dim == 3:
            return "f"
        else:
            return None
        

class O(Group):
    """
    Orthogonal Group (only for O(N))
    """
    def __init__(self, dim):
        super().__init__(f"O({dim})", dim)
        self.type = "O"
        self.abelian = False
        self.__check__()

    def __check__(self):
        dim_check = isinstance(self.dim, int) and self.dim > 1 and self.dim <= 3
        self.checklist = {"dim": dim_check}

    @property
    def fnd_rep(self):
        return self.dim
    
    @property
    def adj_rep(self):
        return self.dim * (self.dim - 1) / 2
    
    @property
    def rep_list(self):
        return {"singlet": 1,
                "fnd": self.fnd_rep, 
                "adj": self.adj_rep}
    
class SO(Group):
    """
    Special Orthogonal Group (only for SO(N))
    """
    def __init__(self, dim):
        super().__init__(f"SO({dim})", dim)
        self.type = "SO"
        self.abelian = False
        self.__check__()

    def __check__(self):
        dim_check = isinstance(self.dim, int) and self.dim > 1 and self.dim <= 3
        self.checklist = {"dim": dim_check}

    @property
    def fnd_rep(self):
        return self.dim
    
    @property
    def adj_rep(self):
        return self.dim * (self.dim - 1) / 2
    
    @property
    def rep_list(self):
        return {"singlet": 1,
                "fnd": self.fnd_rep, 
                "adj": self.adj_rep}
    
class GaugeGroup:
    """
    Base class for gauge groups
    """
    def __init__(self, id, name, charge, group, coupling, boson):
        self.id = id
        self.name = name
        self.charge = charge
        self.group = group
        self.coupling = coupling
        self.boson = boson
        self.__check__()

        if self.group == "SU_2":
            self.definition = "{Ta[a_,b_,c_]->PauliSigma[a,b,c]/2, FSU2L[i_,j_,k_]:> I Eps[i,j,k]}"
            self.reps = ["Ta"]
            self.sym_tensors = None

        elif self.group == "SU_3":
            self.definition = None
            self.sym_tensors = "dSUN"
            self.reps = ["T"]
        else:
            self.definition = None
            self.sym_tensors = None
            self.reps = None
            
        self._check_in_SM()
        self.define_group()
        self.checklist.update(self.group.checklist)

        
        try:
            self.abelian = self.group.abelian
        except:
            self.abelian = False

        try:
            self.fnd_rep = self.group.fnd_rep
        except:
            self.fnd_rep = -1

        try:
            self.adj_rep = self.group.adj_rep
        except:
            self.adj_rep = -1

        try:
            self.rep_list = self.group.rep_list
        except:
            self.rep_list = {"fnd": -1, "adj": -1}

        try:
            self.structure_constants = self.group.structure_constants
        except:
            self.structure_constants = None

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"[GROUP] {self.name:4} | {self.charge:1} | {self.boson:1} | {self.group}"

    def _check_in_SM(self):
        if self.name == "SU3C" and self.group == "SU_3" and self.boson == "G":
            self.isSU3C = True
        else:
            self.isSU3C = False
        if self.name == "SU2L" and self.group == "SU_2" and self.boson == "W":
            self.isSU2L = True
        else:
            self.isSU2L = False
        if self.name == "U1Y" and self.group == "U_1" and self.boson == "B":
            self.isU1Y = True
        else:
            self.isU1Y = False

    def define_group(self):
        try:
            type, N = self.group.split("_")
            N = int(N)
        except:
            N = -1
            type = "U"

        if type == "U":
            self.group = U(N)
        elif type == "SU":
            self.group = SU(N)
        elif type == "O":
            self.group = O(N)
        elif type == "SO":
            self.group = SO(N)
        else:
            self.group = U(-1)

    def __check__(self):
        id_check = isinstance(self.id, str)
        name_check = isinstance(self.name, str)
        charge_check = isinstance(self.charge, str)
        group_check = isinstance(self.group, str) and len(self.group.split("_")) == 2
        coupling_check = isinstance(self.coupling, str)
        boson_check = isinstance(self.boson, str)

        try:
            group_type_check = self.group.split("_")[0] in ["U", "SU", "O", "SO"]
            N_check = self.group.split("_")[1].isdigit()
        except:
            group_type_check = False
            N_check = False

        self.checklist = {"id": id_check, 
                          "name": name_check, 
                          "charge": charge_check, 
                          "group": group_check, 
                          "group_type": group_type_check,
                          "N": N_check,
                          "coupling": coupling_check, 
                          "boson": boson_check}

    @property
    def score(self):
        max_score = len(self.checklist)
        score = sum(1 for value in self.checklist.values() if value is True)
        return f"{score}/{max_score}"

    def pass_all_checks(self):
        return len(self.checklist) == sum(1 for value in self.checklist.values() if value is True)

    @staticmethod
    def write_reps(reps):
        return str(list(dict.fromkeys(reps))).replace("'", "").replace("[", "{").replace("]", "}")

    def gauge_group_info(self):
        gg_info = {
            "Abelian": self.abelian,
            "CouplingConstant": self.coupling,
            "GaugeBoson": self.boson
        }
        if self.abelian:
            gg_info["Charge"] = self.charge
        else:
            gg_info["StructureConstants"] = self.structure_constants
            gg_info["Representations"] = self.write_reps(self.reps)
            gg_info["SymmetryTensors"] = self.sym_tensors
            gg_info["Definition"] = self.definition
        return gg_info

if __name__ == "__main__":
    from utility import random_inputs
    # Generate and print 6 random values
    random_values = random_inputs()
    print(random_values)
    g = GaugeGroup(random_values[0], random_values[1], random_values[2], random_values[3], random_values[4], random_values[5])
    print(g)
    print(g.score())

    