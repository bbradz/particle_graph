from field import Field
from param import IntParam, ExtParam

### =========================== ###
###       Interaction Class     ###
### =========================== ###
class Interaction:
    """
    Interaction class for interactions.
    id: str
    type: str
    requirements: dict
    fields: list
    """
    def __init__(self, id, type, requirements, fields):
        self.id = id
        self.type = type
        self.requirements = requirements
        self.fields = fields
        self.__check__()
        self.ExtParams = []
        self.IntParams = []

    def __str__(self):
        return f"{self.type} ({self.id})"

    def __repr__(self):
        return f"{self.type} ({self.id})"

    def __check__(self):
        id_check = isinstance(self.id, str)
        field_length_check = len(self.fields) == len(self.requirements)
        field_type_check = all(field_type in self.requirements for field_type in self.requirements)
        field_check = all([isinstance(f, Field) for f in self.fields])
        field_check_pass = all([f.pass_all_checks() for f in self.fields])

        try:
            self.sort_field()
            sort_field_check = True
        except Exception as e:
            sort_field_check = e

        self.checklist = {"id_check": id_check,
                          "field_length_check": field_length_check,
                          "field_type_check": field_type_check,
                          "field_check": field_check,
                          "field_check_pass": field_check_pass,
                          "sort_field_check": sort_field_check}

    # Validate the field
    def __validate__(self):
        all_checks = []
        for check in all_checks:
            if any(isinstance(value, Exception) or value == False for value in self.checklist.values()):
                self.checklist[check.__name__] = Exception(f"Skipped due to previous check failure")
            else:
                try:
                    check()
                    self.checklist[check.__name__] = True
                except Exception as e:
                    self.checklist[check.__name__] = e

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
            elif len(candidate) > 1:
                raise Exception(f"Multiple fields with \"{key} = {value}\" found for {self.id}: {candidate}")
            else:
                raise Exception(f"No field with \"{key} = {value}\" found for {self.id}")

### =========================== ###
###       Yukawa Interaction    ###
### =========================== ###

class Yukawa(Interaction):
    """
    Yukawa interaction class.
    """
    field1 = {"type": "fermion", 
              "chirality": "left"}
    field2 = {"type": "fermion", 
              "chirality": "right"}
    field3 = {"type": "scalar"}

    field_types = {0: field1, 1: field2, 2: field3}

    def __init__(self, id, fields):
        super().__init__(id, "Yukawa", self.field_types, fields)
        self.dummy_idx = []
        self.higgs_loc = None

    def __check__(self):
        super().__check__()

        try:
            gen_check = self.sorted_fields[0].gen == self.sorted_fields[1].gen
        except Exception as e:
            gen_check = e

        try:
            dim_check = (self.sorted_fields[0].dim == self.sorted_fields[2].dim) or (self.sorted_fields[1].dim == self.sorted_fields[2].dim)
        except Exception as e:
            dim_check = e
        
        try:
            self.sorted_fields[0].mass_type = "Yukawa"
            self.sorted_fields[1].mass_type = "Yukawa"
            mass_type_check = True
        except Exception as e:
            mass_type_check = e

        self.checklist.update({"gen_check": gen_check, 
                               "dim_check": dim_check,
                               "mass_type_check": mass_type_check})

    def _yukawa_mass(self):
        from sympy import Matrix, sympify

        left = self.sorted_fields[0].to_matrix().transpose()
        right = self.sorted_fields[1].to_matrix()
        self.fermion_bilinear = Matrix(left) * Matrix(right)        

        for i in range(self.sorted_fields[2].dim):
            terms = str(self.fermion_bilinear[i]).replace(" ", "").replace("_L", "").replace("_R", "")
            terms = sympify(terms)
            count = str(terms).count("**2")
            if count == self.sorted_fields[0].gen:
                ids = str(terms).replace("**2", "").replace(" ", "").split("+")
                self.higgs_loc = i

        ids = list(set(ids))
        assert len(ids) == self.sorted_fields[0].gen, f"Assertion failed: {self.id} has incompatible dimensions"

        all_particles = [f.fermion for f in self.sorted_fields[1].particles if f.fermion.id in ids]
        for particle in all_particles:
            mass_param = ExtParam(name = "ym" + particle.name, 
                                  BLOCKNAME = "YUKAWA", 
                                  OrderBlock = particle.pdg_id, 
                                  Value = particle.mass, 
                                  Description = f"\"Yukawa mass for {particle.full_name}\"")
            self.ExtParams.append(mass_param)

    def _yukawa_matrix(self):
        suffix = self.sorted_fields[0].name + self.sorted_fields[1].name
        name = "y" + suffix
        Indices = f"{{{repr(self.sorted_fields[0].gen_idx)}, {repr(self.sorted_fields[1].gen_idx)}}}"
        Definitions = f"{{{name}[i_?NumericQ, j_?NumericQ] :> 0  /; (i =!= j)}}"
        mass_name = [p.name for p in self.ExtParams]
        Value = f"{{{name}[1,1] -> Sqrt[2] {mass_name[0]}/vev, {name}[2,2] -> Sqrt[2] {mass_name[1]}/vev, {name}[3,3] -> Sqrt[2] {mass_name[2]}/vev}}"
        InteractionOrder = "{QED, 1}"
        ParameterName = f"{{{name}[1,1] -> {mass_name[0]}, {name}[2,2] -> {mass_name[1]}, {name}[3,3] -> {mass_name[2]}}}"
        Tex = f"Superscript[y, {suffix}]"
        Description = f"\"Yukawa coupling for {self.sorted_fields[0].name} and {self.sorted_fields[1].name}\""

        self.yukawa_matrix = IntParam(name, 
                                       Indices, 
                                       Definitions, 
                                       Value, 
                                       InteractionOrder, 
                                       ParameterName, 
                                       Tex, 
                                       Description)
        
        self.IntParams.append(self.yukawa_matrix)

    def __validate__(self):
        super().__validate__()
        all_checks = [self._yukawa_mass, 
                      self._yukawa_matrix]
        
        for check in all_checks:
            if any(isinstance(value, Exception) or value == False for value in self.checklist.values()):
                self.checklist[check.__name__] = Exception(f"Skipped due to previous check failure")
            else:
                try:
                    check()
                    self.checklist[check.__name__] = True
                except Exception as e:
                    self.checklist[check.__name__] = e

    def to_fr(self):
        assert self.pass_all_checks(), f"Assertion failed: {self.id} has failed checks"
        
        from name import generate_dummy_idx
        suffix = self.sorted_fields[0].name + self.sorted_fields[1].name
        ym = "y" + suffix
        ff = generate_dummy_idx()
        self.dummy_idx.extend([f"{ff}1", f"{ff}2"])
        color_idx = ", cc" if self.sorted_fields[0].color > 1 and self.sorted_fields[1].color > 1 else ""
        if self.higgs_loc == 0:
            self.dummy_idx.extend(["ii", "cc"])
            return f"    - {ym}[{ff}1, {ff}2] QLbar[sp, ii, {ff}1{color_idx}].uR [sp, {ff}2{color_idx}] Phi[ii]"
        elif self.higgs_loc == 1:
            self.dummy_idx.extend(["ii", "jj", "cc"])
            return f"    - {ym}[{ff}1, {ff}2] QLbar[sp, ii, {ff}1{color_idx}].dR [sp, {ff}2{color_idx}] Phibar[jj] Eps[ii, jj]"

        