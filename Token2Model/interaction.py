from .field import Field
from .param import InternalParameter, ExternalParameter
import numpy as np
from .check import run_checks
from .param import GlobalParameterRegistry

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

        # -------------------------- basic checks --------------------------
        def _id_check():
            result = {"score": 1, "error_var": [], "message": "Passed", "max_score": 1}
            if not isinstance(self.id, str):
                result.update({"score": 0, "error_var": ["id"], "message": f"'id' must be a string"})
            return result

        def _params_check():
            result = {"score": 1, "error_var": [], "message": "Passed", "max_score": 1}
            error_var = [f"param:{par}" for par in self.param_list if par not in self.params.keys()]
            if error_var:
                error_message = f"Invalid parameters: {error_var}"
                result.update({"score": 0, "error_var": error_var, "message": error_message})
            return result

        def _field_length_check():
            result = {"score": 1, "error_var": [], "message": "Passed", "max_score": 1}
            if len(self.fields) != len(self.requirements):
                error_var = ["fields"]
                error_message = f"Number of fields ({len(self.fields)}) does not match number of requirements ({len(self.requirements)})"
                result.update({"score": 0, "error_var": error_var, "message": error_message})
            return result

        def _field_check():
            result = {"score": 1, "error_var": [], "message": "Passed", "max_score": 1}
            error_var = [f"field:{field.id}" for field in self.fields if not isinstance(field, Field)]
            if error_var:
                error_message = f"Fields are not instances of Field: {error_var}"
                result.update({"score": 0, "error_var": error_var, "message": error_message})
            return result

        def _all_field_pass_checks():
            result = {"score": 1, "error_var": [], "message": "Passed", "max_score": 1}
            error_var = [f"field:{field.id}" for field in self.fields if not field.pass_all_checks()]
            if error_var:
                error_message = f"Fields failed checks: {error_var}"
                result.update({"score": 0, "error_var": error_var, "message": error_message})
            return result
                
        def _check_replicate_fields():
            result = {"score": 1, "error_var": [], "message": "Passed", "max_score": 1}
            all_fields = [f.id for f in self.fields]
            if len(all_fields) != len(set(all_fields)):
                error_var = ["fields"]
                error_message = f"Duplicate fields found"
                result.update({"score": 0, "error_var": error_var, "message": error_message})
            else:
                self.all_particles = [p.fermion if f.type == "fermion" else p for f in self.fields for p in f.particles]
                self.all_particles = {p.id: p for p in self.all_particles}
            return result

        def _sort_field():
            result = {"score": 1, "error_var": [], "message": "Passed", "max_score": 1}
            try:
                self.sorted_fields = {}
                for pos, reqs in self.requirements.items():
                    candidate = self.fields
                    for key, value in reqs.items():
                        candidate = [f for f in candidate if f.__dict__()[key] == value or f.__dict__()[key] in value]
                    assert len(candidate) == 1, \
                        f"AssertionError: Multiple fields with \"{key} = {value}\" found for {self.id}: {candidate}"
                    self.sorted_fields[pos] = candidate[0]
            except Exception as e:
                error_var = ["fields"]
                error_message = f"Error: {e}"
                result.update({"score": 0, "error_var": error_var, "message": error_message})
            return result

        self.all_checks.extend([_id_check, 
                                _field_length_check, 
                                _field_check, 
                                _params_check,
                                _all_field_pass_checks, 
                                _check_replicate_fields,
                                _sort_field])

    def __check__(self):
        self.checklist = {}
        self._all_checks()
        run_checks(self.all_checks, self.checklist, skip_results = True)

    def _all_validations(self):
        """ All validations for the 'Interaction' class. """
        self.all_validations = []

    def __validate__(self):
        self._all_validations()
        run_checks(self.all_validations, self.checklist, skip_results = True)

    @property
    def score(self):
        max_score = sum(value["max_score"] for value in self.checklist.values())
        score = sum(value["score"] for value in self.checklist.values())
        return score, max_score

    def pass_all_checks(self):
        score, max_score = self.score
        return score == max_score



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

    field_requirements = {0: field1, 1: field2, 2: field3}

    param_list = []

    all_such_interactions = []

    def __init__(self, id, fields):
        super().__init__(id, "Yukawa", self.field_requirements, fields, self.param_list)
        self.higgs_loc = None
        self.is_SM_yukawa = self.check_SM_yukawa()
        if self.sorted_fields:
            self.scalar_name = self.sorted_fields[2].name


    def check_SM_yukawa(self):
        """check if the Yukawa interaction is a SM Yukawa interaction"""
        if not all(key in self.sorted_fields for key in [0, 1, 2]):
            return False
        
        self.is_LL = self.sorted_fields[0].reps == {"g1": -3, "g2": "fnd", "g3": "singlet"}
        self.is_QL = self.sorted_fields[0].reps == {"g1": 1, "g2": "fnd", "g3": "fnd"}
        self.is_uR = self.sorted_fields[1].reps == {"g1": 4, "g2": "singlet", "g3": "fnd"}
        self.is_dR = self.sorted_fields[1].reps == {"g1": -2, "g2": "singlet", "g3": "fnd"}
        self.is_lR = self.sorted_fields[1].reps == {"g1": -6, "g2": "singlet", "g3": "singlet"}
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
            result = {"score": 1, "error_var": [], "message": "Passed", "max_score": 1}
            if self.sorted_fields[0].gen != self.sorted_fields[1].gen:
                error_var = ["fields"]
                error_message = f"The two fermions have different generations: {self.sorted_fields[0].gen} != {self.sorted_fields[1].gen}"
                result.update({"score": 0, "error_var": error_var, "message": error_message})
            return result

        def _dim_check():
            result = {"score": 1, "error_var": [], "message": "Passed", "max_score": 1}
            if self.sorted_fields[0].dim != self.sorted_fields[2].dim and self.sorted_fields[1].dim != self.sorted_fields[2].dim:
                error_var = ["fields"]
                error_message = f"The scalar and fermions have different dimensions: {self.sorted_fields[0].dim} != {self.sorted_fields[2].dim} and {self.sorted_fields[1].dim} != {self.sorted_fields[2].dim}"
                result.update({"score": 0, "error_var": error_var, "message": error_message})
            return result
        
        self.all_checks.extend([_gen_check, 
                                _dim_check])

    # --------------------------------------------------------------------
    #                       Validation Checks
    # --------------------------------------------------------------------
    def _dirac_bilinear_product(self):
        result = {"score": 1, "error_var": [], "message": "Passed", "max_score": 1}
        try:
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
            
        except Exception as e:
            error_var = ["fields"]
            error_message = f"Error: {e}"
            result.update({"score": 0, "error_var": error_var, "message": error_message})
        return result

    def _get_massive_particles(self):
        result = {"score": 1, "error_var": [], "message": "Passed", "max_score":1}
        try:
            for idx, row in enumerate(self.dirac_bilinears):
                correct_term = all([col.split("*")[0] == col.split("*")[1] for col in row])   
                if correct_term: 
                    self.particle_ids = [col.split("*")[0] for col in row]
                    self.massive_particles = {pid: self.all_particles[pid] for pid in self.particle_ids}
                    self.higgs_loc = idx
                    break
        except Exception as e:
            error_var = ["fields"]
            error_message = f"Error: {e}"
            result.update({"score": 0, "error_var": error_var, "message": error_message})
        return result

    def _check_U1Y_gauge_symmetry(self):
        result = {"score": 1, "error_var": [], "message": "Passed", "max_score": 1}
        try:
            Y_psi_L = self.sorted_fields[0].reps["g1"]
            Y_psi_R = self.sorted_fields[1].reps["g1"]
            Y_Phi = self.sorted_fields[2].reps["g1"]
            sum = -Y_psi_L + Y_psi_R + (-1 if self.higgs_loc == 0 else 1) * Y_Phi
            sign = "+" if self.higgs_loc == 1 else "-"
            
            if sum != 0:
                error_var = ["fields:0:reps:g1", "fields:1:reps:g1", "fields:2:reps:g1"]
                error_message = f"Violates U(1)Y gauge symmetry. {Y_psi_L} + {Y_psi_R} {sign} {Y_Phi} = {sum}"
                result.update({"score": 0, "error_var": error_var, "message": error_message})
        except Exception as e:
            error_var = ["fields"]
            error_message = f"Error: {e}"
            result.update({"score": 0, "error_var": error_var, "message": error_message})
        return result

    def _set_yukawa_name(self):
        result = {"score": 1, "error_var": [], "message": "Passed", "max_score":1}
        try:
            if not self.is_SM_yukawa:
                self.name = f"Y{self.sorted_fields[0].name}"
                self.Description = f"{self.name}-Yukawa-Coupling"
                self.LaTeX = f"Y_{{{self.sorted_fields[0].name}}}"
        except Exception as e:
            error_var = ["fields"]
            error_message = f"Error: {e}"
            result.update({"score": 0, "error_var": error_var, "message": error_message})
        return result

    def _yukawa_mass(self):
        result = {"score": 1, "error_var": [], "message": "Passed", "max_score":1}
        try:
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
        except Exception as e:
            error_var = ["fields"]
            error_message = f"Error: {e}"
            result.update({"score": 0, "error_var": error_var, "message": error_message})
        return result

    def _yukawa_matrix(self):
        result = {"score": 1, "error_var": [], "message": "Passed", "max_score":1}
        try:
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
        except Exception as e:
            error_var = ["fields"]
            error_message = f"Error: {e}"
            result.update({"score": 0, "error_var": error_var, "message": error_message})
        return result
        
    def _yukawa_lagrangian(self):
        result = {"score": 1, "error_var": [], "message": "Passed", "max_score":1}
        try:
            if self.higgs_loc == 1:
                lag = f"- {self.name} conj[{self.scalar_name}].{self.sorted_fields[1].name}.{self.sorted_fields[0].name}"
            else:
                lag = f"- {self.name} {self.sorted_fields[1].name}.{self.sorted_fields[0].name}.{self.scalar_name}"
            self.LagHC.append(lag)
            self.sorted_fields[0].mass_term.append(lag)
            self.sorted_fields[1].mass_term.append(lag)
        except Exception as e:
            error_var = ["fields"]
            error_message = f"Error: {e}"
            result.update({"score": 0, "error_var": error_var, "message": error_message})
        return result
        
    def _mixing_matrix(self):
        result = {"score": 1, "error_var": [], "message": "Passed", "max_score":1}
        try:
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

            if f"V{self.dirac_spinor}" not in GlobalParameterRegistry.all_parameters():
                left_MIX = InternalParameter(f"V{self.dirac_spinor}", 
                                            left_description, 
                                            f"Z{self.left_spinor.upper()}", 
                                            Block = "YUKAWA", 
                                            LaTeX = f"U^{{{self.dirac_spinor}}}_L", 
                                            LesHouches = f"U{self.left_spinor.upper()}MIX")
                self.IntParams[left_MIX.name] = left_MIX
            
            if f"U{self.dirac_spinor}" not in GlobalParameterRegistry.all_parameters():
                right_MIX = InternalParameter(f"U{self.dirac_spinor}", 
                                            right_description, 
                                            f"Z{self.right_spinor.upper()}", 
                                            Block = "YUKAWA", 
                                            LaTeX = f"U^{{{self.dirac_spinor}}}_R", 
                                            LesHouches = f"U{self.right_spinor.upper()}MIX")
                self.IntParams[right_MIX.name] = right_MIX

            self.EWSB_matter_sector = f"{{{{{self.left_spinor}}}, {{conj[{self.right_spinor}]}}}}, {{{{{self.left_spinor.upper()}, V{self.dirac_spinor}}}, {{{self.right_spinor.upper()}, U{self. dirac_spinor}}}}}"
        except Exception as e:
            error_var = ["fields"]
            error_message = f"Error: {e}"
            result.update({"score": 0, "error_var": error_var, "message": error_message})
        return result
       
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
    field_requirements = {0: field1, 1: field2}
    param_list = []
    all_such_interactions = []



    def __init__(self, id, fields):
        super().__init__(id, "VectorLikeFermion", self.field_requirements, fields, self.param_list)
        

# ====================================================================
#                     Scalar Self-Interaction
# ====================================================================
class ScalarSelfInteraction(Interaction):
    """
    Scalar self-interaction class.
    """

    field1 = {"type": ["complex", "real"]}
    field_requirements = {0: field1}
    param_list = ["LambdaVar"]
    all_such_interactions = []



    def __init__(self, id, fields, LambdaVar):
        lambda_input = {"LambdaVar": LambdaVar}
        super().__init__(id, "ScalarSelfInteraction", self.field_requirements, fields, self.param_list, **lambda_input)
        self.is_SM_Higgs = self.check_SM_Higgs()
        #self.scalar_name = self.sorted_fields[0].name

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
        result = {"score": 1, "error_var": [], "message": "Passed", "max_score":1   }
        try:
            if self.is_SM_Higgs:
                self.mu2 = ExternalParameter("mu2", 
                                         "SM Mu Parameter", 
                                         "m2SM", 
                                         "SM", 
                                         Real = False, 
                                         LesHouches = "{SM, 1}",
                                         LaTeX = "\\\\mu")
            else:
                self.mu2 = ExternalParameter(f"mu2{self.sorted_fields[0].name}", 
                                            f"Mu2 Parameter of {self.sorted_fields[0].name}", 
                                            f"mu2{self.sorted_fields[0].name}", 
                                            "BSM", 
                                            Real = False, 
                                            LaTeX = f"\\\\mu_{self.sorted_fields[0].name}")

                self.IntParams[self.mu2.name] = self.mu2
            self.ParametersToSolveTadpoles.append(self.mu2.name)
        except Exception as e:
            error_var = ["fields"]
            error_message = f"Error: {e}"
            result.update({"score": 0, "error_var": error_var, "message": error_message})
        return result

    def _scalar_quartic(self):
        result = {"score": 1, "error_var": [], "message": "Passed", "max_score":1}
        try:
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
                self.LambdaVar = ExternalParameter(f"\[Lambda]_{self.sorted_fields[0].name}", 
                                                f"Quartic Lambda Parameter of {self.sorted_fields[0].name}", 
                                                lambda_name, 
                                                "BSM", 
                                                Real = True,
                                                Value = self.params["LambdaVar"], 
                                                LaTeX = f"\\\\lambda_{self.sorted_fields[0].name}")

            self.ExtParams[self.LambdaVar.name] = self.LambdaVar
        except Exception as e:
            error_var = ["fields"]
            error_message = f"Error: {e}"
            result.update({"score": 0, "error_var": error_var, "message": error_message})
        return result
       
    def _scalar_lagrangian(self):
        result = {"score": 1, "error_var": [], "message": "Passed", "max_score":1}
        try:
            lag = f"- {self.mu2.name} conj[{self.sorted_fields[0].name}].{self.sorted_fields[0].name}" 
            lag += f" - 1/2 {self.LambdaVar.name} conj[{self.sorted_fields[0].name}].{self.sorted_fields[0].name}.conj[{self.sorted_fields[0].name}].{self.sorted_fields[0].name}"
            self.LagNoHC.append(lag)
            self.sorted_fields[0].potential.append(lag)
        except Exception as e:
            error_var = ["fields"]
            error_message = f"Error: {e}"
            result.update({"score": 0, "error_var": error_var, "message": error_message})
        return result

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
    field_requirements = {0: field1, 1: field2}
    param_list = []
    all_such_interactions = []



    def __init__(self, id, fields):
        super().__init__(id, "ScalarScalarMixing", self.field_requirements, fields, self.param_list)
        self.scalar_name = self.sorted_fields[0].name
        self.scalar_name2 = self.sorted_fields[1].name

    def _all_validations(self):
        super()._all_validations()
        