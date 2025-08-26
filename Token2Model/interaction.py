from .field import Field
from .param import InternalParameter, ExternalParameter
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
    def __init__(self, id, type, requirements, fields, param_list, **params):
        self.id = id
        self.type = type
        self.requirements = requirements
        self.fields = fields
        self.param_list = param_list
        self.params = params

        self.ExtParams = {}
        self.IntParams = {}
        self.MatchingConditions = []
        self.ParametersToSolveTadpoles = []
        self.EWSB_matter_sector = None
        self.LagNoHC = []
        self.LagHC = []
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

        def _params_check():
            assert all(param_name in self.params.keys() for param_name in self.param_list), \
                f"AssertionError: {self.id} has invalid parameters: {self.param_list} not in {self.params.keys()}"

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
                
        def _check_replicate_fields():
            assert len(self.fields) == len(set(self.fields)), \
                f"AssertionError: {self.id} has duplicate fields"

        def _check_replicate_particles():
            all_components = [p for f in self.fields for p in f.particles]
            assert len(all_components) == len(set(all_components)), \
                f"AssertionError: {self.id} has duplicate particles"
            self.all_particles = [p.fermion if f.type == "fermion" else p for f in self.fields for p in f.particles]
            self.all_particles = {p.id: p for p in self.all_particles}

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
                                _params_check,
                                _all_field_pass_checks, 
                                _check_replicate_fields,
                                _check_replicate_particles,
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
    field3 = {"type": ["complex", "real"]}

    field_types = {0: field1, 1: field2, 2: field3}

    param_list = []

    all_yukawa = []

    def __init__(self, id, fields):
        super().__init__(id, "Yukawa", self.field_types, fields, self.param_list)
        self.higgs_loc = None
        self.is_SM_yukawa = self.check_SM_yukawa()
        self.scalar_name = self.sorted_fields[2].name

    def check_SM_yukawa(self):
        """check if the Yukawa interaction is a SM Yukawa interaction"""
        if not all(key in self.sorted_fields for key in [0, 1, 2]):
            return False
        
        self.is_LL = self.sorted_fields[0].reps == {"g1": -3, "g2": "fnd", "g3": "singlet"}
        self.is_QL = self.sorted_fields[0].reps == {"g1": 1, "g2": "fnd", "g3": "fnd"}
        self.is_uR = self.sorted_fields[1].reps == {"g1": -4, "g2": "singlet", "g3": "fnd"}
        self.is_dR = self.sorted_fields[1].reps == {"g1": 2, "g2": "singlet", "g3": "fnd"}
        self.is_lR = self.sorted_fields[1].reps == {"g1": 6, "g2": "singlet", "g3": "singlet"}
        self.is_Phi = self.sorted_fields[2].reps == {"g1": 3, "g2": "fnd", "g3": "singlet"}

        if self.is_Phi and self.is_LL and self.is_lR:
            self.name = "Ye"
            self.Description = "Lepton-Yukawa-Coupling"
            self.LaTeX = "Y_e"
            return True
        
        elif self.is_Phi and self.is_QL and self.is_uR:
            self.name = "Yu"
            self.Description = "Up-Yukawa-Coupling"
            self.LaTeX = "Y_u"
            return True 
        
        elif self.is_Phi and self.is_QL and self.is_dR:
            self.name = "Yd"
            self.Description = "Down-Yukawa-Coupling"
            self.LaTeX = "Y_d"
            return True   
        
        else:
            return False

    def _all_checks(self):
        super()._all_checks()

        def _gen_check():
            assert self.sorted_fields[0].gen == self.sorted_fields[1].gen,\
                f"AssertionError: {self.sorted_fields[0].gen} != {self.sorted_fields[1].gen}"

        def _dim_check():
            assert (self.sorted_fields[0].dim == self.sorted_fields[2].dim) or (self.sorted_fields[1].dim == self.sorted_fields[2].dim), \
                f"AssertionError: {self.sorted_fields[0].dim} != {self.sorted_fields[2].dim} and {self.sorted_fields[1].dim} != {self.sorted_fields[2].dim}"
        
        self.all_checks.extend([_gen_check, 
                                _dim_check])

    # --------------------------------------------------------------------
    #                       Validation Checks
    # --------------------------------------------------------------------
    def _dirac_bilinear_product(self):
        left = np.array([[p.id for p in gen] for gen in self.sorted_fields[0]._unphy_fields]).transpose()
        right = np.array([[p.id for p in gen] for gen in self.sorted_fields[1]._unphy_fields])

        rows_left, cols_left = left.shape
        rows_right, cols_right = right.shape
        assert cols_left == rows_right, \
            f"AssertionError: {self.sorted_fields[0].name} and {self.sorted_fields[1].name} have incompatible dimensions"

        # Matrix Multiplication of the Left-Handed and Right-Handed Fermions
        self.dirac_bilinears = [[f"{left[i][k].replace('_L', '')}*{right[k][j].replace('_R', '')}" 
                   for k in range(cols_left)] 
                  for j in range(cols_right) 
                 for i in range(rows_left)]

    def _get_massive_particles(self):
        for idx, row in enumerate(self.dirac_bilinears):
            correct_term = all([col.split("*")[0] == col.split("*")[1] for col in row])   
            if correct_term: 
                self.particle_ids = [col.split("*")[0] for col in row]
                self.massive_particles = {pid: self.all_particles[pid] for pid in self.particle_ids}
                self.higgs_loc = idx
                break
    
    # def _check_massive_particles(self):
    #     [p.assign_mass_type("yukawa") for p in self.massive_particles.values()]
    #     assert all(p.mass > 0 for p in self.massive_particles.values()), \
    #         f"AssertionError: {self.id} has massless particles"

    def _check_U1Y_gauge_symmetry(self):
        Y_psi_L = self.sorted_fields[0].reps["g1"]
        Y_psi_R = self.sorted_fields[1].reps["g1"]
        Y_Phi = self.sorted_fields[2].reps["g1"]
        sum = Y_psi_L + Y_psi_R + (-1 if self.higgs_loc == 1 else 1) * Y_Phi
        sign = "+" if self.higgs_loc == 1 else "-"
        assert sum == 0, \
            f"AssertionError: {self.id} has violates U(1)Y gauge symmetry. {Y_psi_L} + {Y_psi_R} {sign} {Y_Phi} = {sum}"

    def _set_yukawa_name(self):
        if not self.is_SM_yukawa:
            try:
                exotic_yukawa = self.all_yukawa.remove("Yu")
            except:
                exotic_yukawa = self.all_yukawa
            try:
                exotic_yukawa = exotic_yukawa.remove("Yd")
            except:
                exotic_yukawa = self.all_yukawa
            try:
                exotic_yukawa = exotic_yukawa.remove("Ye")
            except:
                exotic_yukawa = self.all_yukawa

            self.name = f"Y{len(exotic_yukawa)}"
            self.Description = f"{self.name}-Yukawa-Coupling"
            self.LaTeX = f"Y_{len(exotic_yukawa)}"
        self.all_yukawa.append(self.name)

    def _yukawa_mass(self):
        # !!! we omit top quark here !!!
        sm_fermion = ["e", "mu", "tau", "ve", "vm", "vt", "u", "c", "d", "s", "b"]
        for particle in self.massive_particles.values():
            if particle.name in sm_fermion:
                pass
            else:
                particle_mass_name = f"M{particle.name}"
                mf = ExternalParameter(particle_mass_name, 
                                       f"Mass of {particle.name}", 
                                       particle_mass_name, 
                                       "YUKAWA", 
                                       Real = True, 
                                       Value = particle.mass,
                                       LesHouches = particle_mass_name,
                                       LaTeX = f"m_{particle.name}")
                self.ExtParams[mf.name] = mf
        
    def _yukawa_matrix(self):
        self._set_yukawa_name()

        yukawa_matrix = InternalParameter(self.name,
                                          self.Description,
                                          self.name,
                                          "YUKAWA",
                                          LaTeX = self.LaTeX,
                                          LesHouches = self.name)
        self.IntParams[self.name] = yukawa_matrix

        for i, (_, value) in enumerate(self.massive_particles.items()):
            math_idx = i + 1
            if value.name == "e":
                self.MatchingConditions.append(f"{self.name}[{math_idx}, {math_idx}], YeSM[1,1]")
            elif value.name == "mu":
                self.MatchingConditions.append(f"{self.name}[{math_idx}, {math_idx}], YeSM[2,2]")
            elif value.name == "tau":
                self.MatchingConditions.append(f"{self.name}[{math_idx}, {math_idx}], YeSM[3,3]")
            elif value.name == "u":
                self.MatchingConditions.append(f"{self.name}[{math_idx}, {math_idx}], YuSM[1,1]")
            elif value.name == "c":
                self.MatchingConditions.append(f"{self.name}[{math_idx}, {math_idx}], YuSM[2,2]")
            elif value.name == "d":
                self.MatchingConditions.append(f"{self.name}[{math_idx}, {math_idx}], YdSM[1,1]")
            elif value.name == "s":
                self.MatchingConditions.append(f"{self.name}[{math_idx}, {math_idx}], YdSM[2,2]")
            elif value.name == "b":
                self.MatchingConditions.append(f"{self.name}[{math_idx}, {math_idx}], YdSM[3,3]")
            # elif value.name == "t":
            #     self.MatchingConditions.append(f"{self.name}[{math_idx}, {math_idx}], YuSM[3,3]")
            else:
                self.MatchingConditions.append(f"{self.name}[{math_idx}, {math_idx}], Sqrt[2]/vSM*M{value.name}")
        
    def _yukawa_lagrangian(self):
        if self.higgs_loc == 1:
            lag = f"- {self.name} conj[{self.scalar_name}].{self.sorted_fields[1].name}.{self.sorted_fields[0].name}"
        else:
            lag = f"- {self.name} {self.sorted_fields[1].name}.{self.sorted_fields[0].name}.{self.scalar_name}"
        self.LagHC.append(lag)
        
    def _mixing_matrix(self):
        all_left_spinors = self.sorted_fields[0].phy_field_info
        self.left_spinor = [key for key, value in all_left_spinors.items() if value["location"] == self.higgs_loc][0]
        self.right_spinor = list(self.sorted_fields[1].phy_field_info.keys())[0]
        self.dirac_spinor = self.right_spinor[:-1]

        if self.is_lR:
            left_description = "Left-Lepton-Mixing-Matrix"
            right_description = "Right-Lepton-Mixing-Matrix"
        elif self.is_uR:
            left_description = "Left-Up-Mixing-Matrix"
            right_description = "Right-Up-Mixing-Matrix"
        elif self.is_dR:
            left_description = "Left-Down-Mixing-Matrix"
            right_description = "Right-Down-Mixing-Matrix"
        else:
            left_description = f"Left-{self.dirac_spinor}-Mixing-Matrix"
            right_description = f"Right-{self.dirac_spinor}-Mixing-Matrix"

        if f"V{self.dirac_spinor}" not in InternalParameter.all_parameters:
            left_MIX = InternalParameter(f"V{self.dirac_spinor}", 
                                        left_description, 
                                        f"Z{self.left_spinor.upper()}", 
                                        Block = "YUKAWA", 
                                        LaTeX = f"U^{{{self.dirac_spinor}}}_L", 
                                        LesHouches = f"U{self.left_spinor.upper()}MIX")
            self.IntParams[left_MIX.name] = left_MIX
        
        if f"U{self.dirac_spinor}" not in InternalParameter.all_parameters:
            right_MIX = InternalParameter(f"U{self.dirac_spinor}", 
                                        right_description, 
                                        f"Z{self.right_spinor.upper()}", 
                                        Block = "YUKAWA", 
                                        LaTeX = f"U^{{{self.dirac_spinor}}}_R", 
                                        LesHouches = f"U{self.right_spinor.upper()}MIX")
            self.IntParams[right_MIX.name] = right_MIX

        self.EWSB_matter_sector = f"{{{{{self.left_spinor}}}, {{conj[{self.right_spinor}]}}}}, {{{{{self.left_spinor.upper()}, V{self.dirac_spinor}}}, {{{self.right_spinor.upper()}, U{self. dirac_spinor}}}}}"
       
    def _all_validations(self):
        super()._all_validations()
        self.all_validations.extend([self._dirac_bilinear_product, 
                                     self._get_massive_particles, 
                                     #self._check_massive_particles,
                                     self._check_U1Y_gauge_symmetry,
                                     self._yukawa_mass,
                                     self._yukawa_matrix,
                                     self._mixing_matrix,
                                     self._yukawa_lagrangian])

# ====================================================================
#                     Vector-Like Fermion
# ====================================================================
class VectorLikeFermion(Interaction):
    """
    Vector-like fermion class.
    """

    field1 = {"type": "fermion", "chirality": "left"}
    field2 = {"type": "fermion", "chirality": "right"}
    field_types = {0: field1, 1: field2}
    param_list = []

    def __init__(self, id, fields):
        super().__init__(id, "VectorLikeFermion", self.field_types, fields, self.param_list)
        

# ====================================================================
#                     Scalar Self-Interaction
# ====================================================================
class ScalarSelfInteraction(Interaction):
    """
    Scalar self-interaction class.
    """

    field1 = {"type": ["complex", "real"]}
    field_types = {0: field1}
    param_list = ["LambdaVar"]

    def __init__(self, id, fields, LambdaVar):
        lambda_input = {"LambdaVar": LambdaVar}
        super().__init__(id, "ScalarSelfInteraction", self.field_types, fields, self.param_list, **lambda_input)
        self.is_SM_Higgs = self.check_SM_Higgs()
        self.scalar_name = self.sorted_fields[0].name

    def check_SM_Higgs(self):
        """check if the scalar self-interaction is a SM Higgs self-interaction"""
        if not all(key in self.sorted_fields for key in [0]):
            return False
        is_EW_doublet = self.sorted_fields[0].reps == {"g1": 3, "g2": "fnd", "g3": "singlet"}
        is_SM_Phi = self.sorted_fields[0].name == "H"
        
        if is_EW_doublet and is_SM_Phi:
            return True
        else:
            return False
        
    def _scalar_mass(self):
        if self.is_SM_Higgs:
            self.mu2 = ExternalParameter("mu2", 
                                         "SM Mu Parameter", 
                                         "m2SM", 
                                         "SM", 
                                         Real = False, 
                                         LesHouches = "{SM, 1}",
                                         LaTeX = "\\\\mu")
        else:
            self.mu2 = ExternalParameter(f"mu2{self.scalar_name}", 
                                         f"Mu2 Parameter of {self.scalar_name}", 
                                         f"mu2{self.scalar_name}", 
                                         "BSM", 
                                         Real = False, 
                                         LaTeX = f"\\\\mu_{self.scalar_name}")

        self.IntParams[self.mu2.name] = self.mu2
        self.ParametersToSolveTadpoles.append(self.mu2.name)

    def _scalar_quartic(self):
        lambda_name = f"LambdaVar{self.id}"
        if self.is_SM_Higgs:
            self.LambdaVar = ExternalParameter("\[Lambda]", 
                                               "SM Higgs Selfcouplings", 
                                               lambda_name, 
                                               "SM", 
                                               DependenceNum = "Mass[hh]^2/v^2",
                                               Real = True,
                                               Value = self.params["LambdaVar"], 
                                               LesHouches = "{SM, 1}",
                                               LaTeX = "\\\\lambda")
        else:
            self.LambdaVar = ExternalParameter(f"\[Lambda]_{self.scalar_name}", 
                                               f"Quartic Lambda Parameter of {self.scalar_name}", 
                                               lambda_name, 
                                               "BSM", 
                                               Real = True,
                                               Value = self.params["LambdaVar"], 
                                               LaTeX = f"\\\\lambda_{self.scalar_name}")

        self.ExtParams[self.LambdaVar.name] = self.LambdaVar
       
    def _scalar_lagrangian(self):
        lag = f"- {self.mu2.name} conj[{self.scalar_name}].{self.scalar_name}" 
        lag += f" - 1/2 {self.LambdaVar.name} conj[{self.scalar_name}].{self.scalar_name}.conj[{self.scalar_name}].{self.scalar_name}"
        self.LagNoHC.append(lag)

    def _all_validations(self):
        super()._all_validations()
        self.all_validations.extend([self._scalar_mass, 
                                     self._scalar_quartic, 
                                     self._scalar_lagrangian])
        
    

# ====================================================================
#                       Scalar-Scalar Mixing
# ====================================================================
class ScalarScalarMixing(Interaction):
    """
    Scalar-scalar mixing class.
    """

    field1 = {"type": ["complex", "real"]}
    field2 = {"type": ["complex", "real"]}
    field_types = {0: field1, 1: field2}
    param_list = []

    def __init__(self, id, fields):
        super().__init__(id, "ScalarScalarMixing", self.field_types, fields, self.param_list)
        self.scalar_name = self.sorted_fields[0].name
        self.scalar_name2 = self.sorted_fields[1].name

    def _all_validations(self):
        super()._all_validations()
        