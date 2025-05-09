from field import Field
from param import IntParam, ExtParam

### =========================== ###
###       Interaction Class     ###
### =========================== ###
class Interaction:
    def __init__(self, id, type, requirements, fields):
        self.id = id
        self.type = type
        self.requirements = requirements
        self.fields = fields
        self.__check__()
        self.ExtParams = []
        self.IntParams = []

    def __str__(self):
        return f"{self.type}"

    def __repr__(self):
        return f"{self.id} ({self.type})"

    def __check__(self):
        id_check = isinstance(self.id, str)
        field_length_check = len(self.fields) == len(self.requirements)
        field_type_check = all(field_type in self.requirements for field_type in self.requirements)
        field_check = all([isinstance(f, Field) for f in self.fields])

        try:
            self.sort_field()
            sort_field_check = True
        except Exception as e:
            sort_field_check = e


        self.checklist = {"id_check": id_check,
                          "field_length_check": field_length_check,
                          "field_type_check": field_type_check,
                          "field_check": field_check,
                          "sort_field_check": sort_field_check}

    @property
    def score(self):
        max_score = len(self.checklist)
        score = sum(1 for value in self.checklist.values() if value is True)
        return f"{score}/{max_score}"

    def pass_all_checks(self):
        return all(self.checklist.values()) 

    def sort_field(self):
        self.sorted_fields = {}
        for pos, reqs in self.requirements.items():
            candidate = self.fields
            for key, value in reqs.items():
                candidate = [f for f in candidate if f.__dict__()[key] == value]
            if len(candidate) == 1:
                self.sorted_fields[pos] = candidate[0]
            else:
                raise Exception(f"Multiple fields found for {self.id}: {candidate}")

### =========================== ###
###       Yukawa Interaction    ###
### =========================== ###
class Yukawa(Interaction):
    field1 = {"type": "fermion", 
              "chirality": "left"}
    field2 = {"type": "fermion", 
              "chirality": "right"}
    field3 = {"type": "scalar"}

    field_types = {0: field1, 1: field2, 2: field3}

    def __init__(self, id, fields):
        super().__init__(id, "Yukawa", self.field_types, fields)
        self.left = self.sorted_fields[0]
        self.right = self.sorted_fields[1]
        self.higgs = self.sorted_fields[2]
        
        self.left._mass_type = "YUKAWA"
        self.right._mass_type = "YUKAWA"
        self.dummy_idx = []
        self.higgs_loc = None
        self.yukawa_mass()
        self.yukawa_matrix()

    def __check__(self):
        super().__check__()
        gen_check = self.sorted_fields[0].gen == self.sorted_fields[1].gen
        self.checklist["gen_check"] = gen_check

    def _mass_term_check(self):
        left = self.left.__matrix__().transpose()
        right = self.right.__matrix__()
        assert left.shape[1] == right.shape[0], f"Assertion failed: {self.id} has incompatible dimensions"

    def yukawa_mass(self):
        left = self.left.__matrix__().transpose()
        right = self.right.__matrix__()
        from sympy import Matrix, sympify
        result = Matrix(left) * Matrix(right)
        assert len(result) == self.higgs.dim, f"Assertion failed: {self.id} has incompatible dimensions"
        for i in range(self.higgs.dim):
            terms = str(result[i]).replace(" ", "").replace("_L", "").replace("_R", "")
            terms = sympify(terms)
            count = str(terms).count("**2")
            if count == self.left.gen:
                ids = str(terms).replace("**2", "").replace(" ", "").split("+")
                self.higgs_loc = i

        ids = list(set(ids))
        assert len(ids) == self.left.gen, f"Assertion failed: {self.id} has incompatible dimensions"

        all_particles = [f.fermion for f in self.right.particles if f.fermion.id in ids]
        for particle in all_particles:
            yuk_mass_name = "ym" + particle.name 
            BlockName = "YUKAWA"
            OrderBlock = particle.pdg_id
            Value = particle.mass
            Description = f"\"Yukawa mass for {particle.full_name}\""
            mass_param = ExtParam(yuk_mass_name, 
                                  BlockName, 
                                  OrderBlock, 
                                  Value, 
                                  Description)
            self.ExtParams.append(mass_param)

    def yukawa_matrix(self):
        suffix = self.left.name + self.right.name
        name = "y" + suffix
        Indices = f"{{{repr(self.left.gen_idx)}, {repr(self.right.gen_idx)}}}"
        Definitions = f"{{{name}[i_?NumericQ, j_?NumericQ] :> 0  /; (i =!= j)}}"
        mass_name = [p.name for p in self.ExtParams]
        Value = f"{{{name}[1,1] -> Sqrt[2] {mass_name[0]}/vev, {name}[2,2] -> Sqrt[2] {mass_name[1]}/vev, {name}[3,3] -> Sqrt[2] {mass_name[2]}/vev}}"
        InteractionOrder = "{QED, 1}"
        ParameterName = f"{{{name}[1,1] -> {mass_name[0]}, {name}[2,2] -> {mass_name[1]}, {name}[3,3] -> {mass_name[2]}}}"
        Tex = f"Superscript[y, {suffix}]"
        Description = f"\"Yukawa coupling for {self.left.name} and {self.right.name}\""

        self._yukawa_matrix = IntParam(name, 
                                       Indices, 
                                       Definitions, 
                                       Value, 
                                       InteractionOrder, 
                                       ParameterName, 
                                       Tex, 
                                       Description)
        
        self.IntParams.append(self._yukawa_matrix)

    def __validate__(self):
        pass

    def to_fr(self):
        from name import generate_dummy_idx
        ym = self._yukawa_matrix.name 
        ff = generate_dummy_idx()
        self.dummy_idx.extend([f"{ff}1", f"{ff}2"])
        color_idx = ", cc" if self.left.color > 1 and self.right.color > 1 else ""
        if self.higgs_loc == 0:
            self.dummy_idx.extend(["ii", "cc"])
            return f"    - {ym}[{ff}1, {ff}2] QLbar[sp, ii, {ff}1{color_idx}].uR [sp, {ff}2{color_idx}] Phi[ii]"
        elif self.higgs_loc == 1:
            self.dummy_idx.extend(["ii", "jj", "cc"])
            return f"    - {ym}[{ff}1, {ff}2] QLbar[sp, ii, {ff}1{color_idx}].dR [sp, {ff}2{color_idx}] Phibar[jj] Eps[ii, jj]"
     


### =========================== ###
###       Higgs Interaction     ###
### =========================== ###
class ComplexPhi234(Interaction):
    field = {"type": "complex"}

    field_types = {0: field}

    def __init__(self, id, fields):
        super().__init__(id, "Higgs", self.field_types, fields)
        self.field = self.sorted_fields[0]

    def _mass_term_check(self):
        pass

    def to_fr(self):
        pass
        
        
        