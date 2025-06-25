from .field import Field
from .param import IntParam, ExtParam
import numpy as np
# ====================================================================
#                              Interaction
# ====================================================================
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
        self.ExtParams = []
        self.IntParams = []
        self.sorted_fields = {}
        self.__check__()

    def __str__(self):
        return f"{self.type} ({self.id})"

    def __repr__(self):
        return f"{self.type} ({self.id})"

    def _all_checks(self):
        """ All checks for the initial INPUTs of the 'Interaction' class. """
        self.all_checks = []

        def _id_check():
            assert isinstance(self.id, str), f"AssertionError: {self.id} is not a string"

        def _field_length_check():
            assert len(self.fields) == len(self.requirements), \
                f"AssertionError: {self.id} has {len(self.fields)} fields but {len(self.requirements)} requirements"

        def _field_type_check():
            assert all(field_type in self.requirements for field_type in self.requirements), \
                f"AssertionError: {self.id} has invalid field types"

        def _field_check():
            for f in self.fields:
                assert isinstance(f, Field), \
                    f"AssertionError: {f} is not a Field"

        def _all_field_pass_checks():
            for f in self.fields:
                assert f.pass_all_checks(), \
                    f"AssertionError: {f} has failed checks"

        def _sort_field():
            self.sorted_fields = {}
            for pos, reqs in self.requirements.items():
                candidate = self.fields
                for key, value in reqs.items():
                    candidate = [f for f in candidate if f.__dict__()[key] == value or f.__dict__()[key] in value]
                assert len(candidate) == 1, \
                    f"AssertionError: Multiple fields with \"{key} = {value}\" found for {self.id}: {candidate}"
                self.sorted_fields[pos] = candidate[0]

        self.all_checks.extend([_id_check, 
                                _field_length_check, 
                                _field_type_check, 
                                _field_check, 
                                _all_field_pass_checks, 
                                _sort_field])
    
    @staticmethod
    def run_checks(all_checks, checklist, skip_check = False):
        for check in all_checks:
            fail_previous_check = any(isinstance(value, Exception) or value == False for value in checklist.values())
            if skip_check and fail_previous_check:
                checklist[check.__name__] = Exception(f"Skipped due to previous check failure")
            else:
                try:
                    check()
                    checklist[check.__name__] = True
                except Exception as e:
                    checklist[check.__name__] = e

    def __check__(self):
        self.checklist = {}
        self._all_checks()
        self.run_checks(self.all_checks, self.checklist)

    def _all_validations(self):
        """ All validations for the 'Interaction' class. """
        self.all_validations = []

    def __validate__(self):
        self._all_validations()
        self.run_checks(self.all_validations, self.checklist, skip_check = True)

    @property
    def score(self):
        max_score = len(self.checklist)
        score = sum(1 for value in self.checklist.values() if value is True)
        return f"{score}/{max_score}"

    def pass_all_checks(self):
        return all(self.checklist.values()) 



# ====================================================================
#                              Yukawa
# ====================================================================
class Yukawa(Interaction):
    """
    Yukawa interaction class.
    """

    field1 = {"type": "fermion", 
              "chirality": "left"}
    field2 = {"type": "fermion", 
              "chirality": "right"}
    field3 = {"type": ["complex", "real", "scalar"]}

    field_types = {0: field1, 1: field2, 2: field3}

    def __init__(self, id, fields):
        super().__init__(id, "Yukawa", self.field_types, fields)
        self.dummy_idx = []
        self.higgs_loc = None

    def _all_checks(self):
        super()._all_checks()

        def _gen_check():
            assert self.sorted_fields[0].gen == self.sorted_fields[1].gen,\
                f"AssertionError: {self.sorted_fields[0].gen} != {self.sorted_fields[1].gen}"

        def _dim_check():
            assert (self.sorted_fields[0].dim == self.sorted_fields[2].dim) or (self.sorted_fields[1].dim == self.sorted_fields[2].dim), f"AssertionError: {self.sorted_fields[0].dim} != {self.sorted_fields[2].dim} and {self.sorted_fields[1].dim} != {self.sorted_fields[2].dim}"
        
        def _assign_mass_type():
            self.sorted_fields[0].mass_type = "Yukawa"
            self.sorted_fields[1].mass_type = "Yukawa"

        self.all_checks.extend([_gen_check, 
                                _dim_check, 
                                _assign_mass_type])

    def _yukawa_mass(self):
        left = np.array([[p.id for p in gen] for gen in self.sorted_fields[0]._unphy_fields]).transpose()
        right = np.array([[p.id for p in gen] for gen in self.sorted_fields[1]._unphy_fields])

        rows_left, cols_left = left.shape
        rows_right, cols_right = right.shape

        # Matrix Multiplication of the Left-Handed and Right-Handed Fermions
        result = [[f"{left[i][k].replace('_L', '')}*{right[k][j].replace('_R', '')}" 
                   for k in range(cols_left)] 
                  for j in range(cols_right) 
                 for i in range(rows_left)]
        
        particle_ids = []
        for idx, row in enumerate(result):
            correct_term = all([col.split("*")[0] == col.split("*")[1] for col in row])   
            if correct_term: 
                particle_ids = [col.split("*")[0] for col in row]
                self.higgs_loc = idx
                break

        particle_ids = list(set(particle_ids))
        assert len(particle_ids) == self.sorted_fields[0].gen, \
            f"AssertionError: {self.id} has incompatible dimensions"

        all_particles = [f.fermion for f in self.sorted_fields[1].particles if f.fermion.id in particle_ids]
        for particle in all_particles:
            mass_param = ExtParam(name = "ym" + particle.name, 
                                  BLOCKNAME = "YUKAWA", 
                                  OrderBlock = particle.pdg_id, 
                                  Value = particle.mass, 
                                  Description = f"\"Yukawa mass for {particle.full_name}\"")
            self.ExtParams.append(mass_param)

        # Generate the Yukawa matrix
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

    def _all_validations(self):
        super()._all_validations()
        self.all_validations.extend([self._yukawa_mass])

    def to_fr(self):        
        from .name import generate_dummy_idx
        
        suffix = self.sorted_fields[0].name + self.sorted_fields[1].name
        ym = "y" + suffix
        ff = generate_dummy_idx()
        self.dummy_idx.extend([f"{ff}1", f"{ff}2"])
        color_idx = ", cc" if self.sorted_fields[0].color > 1 and self.sorted_fields[1].color > 1 else ""
        if self.higgs_loc == 1:
            self.dummy_idx.extend(["ii", "cc"])
            return f"    - {ym}[{ff}1, {ff}2] {self.sorted_fields[0]}bar[sp, ii, {ff}1{color_idx}].{self.sorted_fields[1].name} [sp, {ff}2{color_idx}] Phi[ii]"
        elif self.higgs_loc == 0:
            self.dummy_idx.extend(["ii", "jj", "cc"])
            return f"    - {ym}[{ff}1, {ff}2] {self.sorted_fields[0]}bar[sp, ii, {ff}1{color_idx}].{self.sorted_fields[1].name} [sp, {ff}2{color_idx}] Phibar[jj] Eps[ii, jj]"
        
