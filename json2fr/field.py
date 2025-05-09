### ========================================================================== ###
###                                                                            ###
###                           Particles Classes                                ###
###                                                                            ###
### ========================================================================== ###

import numpy as np
from particle import Particle, WeylSpinor
import name
from fractions import Fraction
from index import Index

### ======================= ###
###          Field          ###
### ======================= ###
class Field:
    """
    Base class for fields.
    id: str 
    name: str
    type: str ["fermion", "scalar", "vector"]
    dim: int > 0
    gen: int > 0
    particles: list of Particle
    self_conjugate: bool
    QuantumNumber: dict
    """
    def __init__(self, id, name, type, dim, gen, particles, self_conjugate, QuantumNumber):
        self.id = id
        self.name = name
        self.type = type
        self.dim = dim
        self.gen = gen
        self.particles = particles
        self.self_conjugate = self_conjugate
        self.QuantumNumber = QuantumNumber
        self.__check__()
        self.indices = []

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __dict__(self):
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "dim": self.dim,
            "gen": self.gen,
            "particles": self.particles,
            "self_conjugate": self.self_conjugate,
            "QuantumNumber": self.QuantumNumber
        }
    
    def __check__(self):
        """
        Input validation for the field class.
        """
        id_check = isinstance(self.id, str)
        name_check = isinstance(self.name, str)
        type_check = isinstance(self.type, str) and self.type in ["fermion", "scalar", "vector"]
        dim_check = isinstance(self.dim, int) and self.dim > 0
        gen_check = isinstance(self.gen, int) and self.gen > 0
        ptcl_list_check = isinstance(self.particles, list) 
        ptcl_check = all(isinstance(p, Particle) for p in self.particles)
        self_conjugate_check = isinstance(self.self_conjugate, bool)
        QuantumNumber_dict_check = isinstance(self.QuantumNumber, dict)
        QuantumNumber_value_check = all(isinstance(x, int) for x in self.QuantumNumber.values())
        
        try:
            # check if the number of particles is consistent with the dimension and generation
            ptcl_num = (len(self.particles) == self.dim * self.gen)
        except:
            ptcl_num = False
        try:
            # check if the type of the particles is consistent with the type of the field
            ptcl_type = all(p.type == self.type for p in self.particles)
        except:
            ptcl_type = False
        try:
            # check if the particles are self-conjugate
            ptcl_self_conjugate = all(p.charge == 0 for p in self.particles) or not self.self_conjugate
        except:
            ptcl_self_conjugate = False

        self.checklist = {"id": id_check, 
                          "name": name_check, 
                          "type": type_check,
                          "dim": dim_check, 
                          "gen": gen_check,
                          # particle checks
                          "ptcl_list_check": ptcl_list_check,
                          "ptcl_check": ptcl_check,
                          "ptcl_num": ptcl_num,
                          "ptcl_type": ptcl_type,
                          "ptcl_self_conjugate": ptcl_self_conjugate,
                          # field checks
                          "self_conjugate": self_conjugate_check, 
                          "QuantumNumber_dict_check": QuantumNumber_dict_check,
                          "QuantumNumber_value_check": QuantumNumber_value_check, 
                          }
    
    def _generation_index(self):
        if self.gen > 1:
            gen_idx_name = name.num2words(self.gen).capitalize() + "Gen"
            self.gen_idx = Index(gen_idx_name, self.gen, "Fold")
            self.indices.append(self.gen_idx)
        else:
            pass
    
    def _gen_type_consistency(self):
        if self.type != "fermion":
            assert self.gen == 1, f"AssertionError: Non-fermion fields can have only one generation"
        else:
            pass

    def __validate__(self):
        all_checks = [self._generation_index, self._gen_type_consistency]

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
        return len(self.checklist) == sum(1 for value in self.checklist.values() if value is True)
        


### ======================== ###
###       Fermion Field      ###
### ======================== ###
class FermionField(Field):
    """
    Class for fermion fields.
    reps: list of int
    groups: list of str
    chirality: str
    """
    def __init__(self, id, name, reps, groups, dim, gen, particles, self_conjugate, QuantumNumber, chirality):
        self.reps = reps
        self.groups = groups
        self.chirality = chirality
        super().__init__(id, name, "fermion", dim, gen, particles, self_conjugate, QuantumNumber)
        self.__check__()

        self.color = 1
        self._mass_type = None
        self._unphy_fields = None
        self._phy_fields = None
        self.full_reps = {}
        self.abelian_charges = {}

    def __dict__(self):
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "dim": self.dim,
            "gen": self.gen,
            "chirality": self.chirality,
            "particles": self.particles,
            "self_conjugate": self.self_conjugate,
            "QuantumNumber": self.QuantumNumber
        }

    def __check__(self):
        super().__check__()
        reps_dict_check = isinstance(self.reps, dict) 
        reps_check = all(isinstance(x, int) or isinstance(x, str) for x in self.reps.values())
        reps_len_check = len(self.reps) == len(self.groups)
        groups_dict_check = isinstance(self.groups, dict)
        groups_check = all(isinstance(g, str) for g in self.groups)
        chirality_check = self.chirality in ["left", "right"]
        # Override the ptcl_check from ParticleClass to WeylSpinorClass
        ptcl_check = all(isinstance(p, WeylSpinor) for p in self.particles)

        self.checklist.update(
            {"reps_dict": reps_dict_check, 
             "reps": reps_check, 
             "reps_len": reps_len_check,
             "groups_dict": groups_dict_check, 
             "groups": groups_check, 
             "chirality": chirality_check,
             "ptcl_check": ptcl_check
             }
        )

    # after checking all the inputs, the rest of the checks are done in the sequence
    # if a pre-requisite is not met, the rest of the checks will not be performed and automatically fail
    # this is to save time and avoid errors
    # In unphyField, the checks are:
    # 0. pass all the input checks
    # 1. Can one write the reps under the groups in the form of a dictionary?
    # 2. Is the dimension of the field consistent with the dimension of the corresponding representation?
    # 3. Assign colors to the particles
    # 4. Sort the particles into generations and flavors
    # 5. Assign Flavors to the particles
    # 6. Check if the charges of the particles are consistent
    # 7. Check if all the particles pass all the checks

    @staticmethod
    def _index(group, dim):
        if dim == 1:
            return None
        elif group.isSU3C and dim == 3:
            return Index("Colour", 3, "Fold", color = True)
        elif group.isSU3C and dim == 8:
            return Index("Gluon", 8, "Fold", color = True)
        else:
            idx_name = str(group.group).replace("(", "").replace(")", "") + name.num2tuple(int(dim))[0].upper()
            return Index(idx_name, int(dim), "Fold", color = False)
        
    # Task 1: Assign reps to the groups
    def _full_reps(self):
        for key, group in self.groups.items():
            rep_dict = {}
            if group.abelian:
                rep_dict["reps"] = self.reps[key]
                rep_dict["dim"] = 1
                rep_dict["group"] = group
                rep_dict["isColor"] = False
                rep_dict["abelian"] = True
                self.abelian_charges[group.charge] = self.reps[key]
            else:
                rep_dict["reps"] = self.reps[key]
                rep_dict["dim"] = group.rep_list[self.reps[key]]
                rep_dict["group"] = group
                rep_dict["isColor"] = group.isSU3C
                rep_dict["abelian"] = False

                rep_idx = self._index(group, rep_dict["dim"])
                if rep_idx is not None:
                    group.reps.append(str(rep_idx))
                    self.indices.append(rep_idx)
                else:
                    pass

            self.full_reps[group.id] = rep_dict

    # Task 2: Check if the dimension of the field is consistent with the dimension of the corresponding representation
    def _dim_consistency(self):
        for _, rep_dict in self.full_reps.items():
            if rep_dict["abelian"]:
                pass
            elif rep_dict["isColor"]:
                # TO-DO: Assign colors to the particles
                self.color = rep_dict["group"].rep_list[rep_dict["reps"]]
            elif rep_dict["reps"] == "singlet":
                pass
            else:
                assert rep_dict["dim"] == self.dim, f"AssertionError: {self.name} with dim-{self.dim} is inconsistent with \"{rep_dict['reps']}\" ({rep_dict['dim']}) in {rep_dict['group']}"

    # Task 3: Assign colors to the particles
    def _assign_colors(self):
        for p in self.particles:
            if self.color is None:
                pass
            else:                
                p.fermion.color = self.color
            
    # Task 4: Sort the particles into generations and flavors
    def _sort_unphy_fields(self):
        self._unphy_fields = np.array(self.particles).reshape(self.gen, self.dim).tolist()

    # Task 4.5: Sort the particles into generations and flavors
    # No check is needed for this task, as _sort_particles is the prerequisite
    def _sort_phy_fields(self):
        self._phy_fields = np.array(self._unphy_fields).transpose().tolist()

    # Task 5: Assign Flavors to the particles
    # This is a hardcoded function that assigns flavors to the particles based on the QuantumNumber
    def _assign_flavors(self):
        if self.QuantumNumber['LeptonNumber'] != 0:
            for gen in self._unphy_fields:
                flavor = [p.fermion.name for p in gen if p.charge != 0]
                for p in gen:
                    if len(flavor) == 1:
                        p.fermion.flavor = flavor[0]
                    else:
                        p.fermion.flavor = None

    # Task 6: Check if the charges of the particles are consistent
    def _check_charge_consistency(self):
        for idx, gen in enumerate(self._phy_fields):
            assert all(cf.charge == gen[0].charge for cf in gen), f"AssertionError: {self.name} has inconsistent charges in generation {idx+1}"

    # Task 7: Check if all the particles pass all the checks
    def _ptcl_pass_all_checks(self):
        for p in self.particles:
            if not p.fermion.pass_all_checks():
                return False
        return True

    def __validate__(self):
        super().__validate__()
        all_checks = [self._full_reps, self._dim_consistency, 
                      self._assign_colors, self._sort_unphy_fields, 
                      self._sort_phy_fields, self._assign_flavors, 
                      self._check_charge_consistency, self._ptcl_pass_all_checks]
        
        for check in all_checks:
            # Check if any previous check has failed
            if any(isinstance(value, Exception) or value == False for value in self.checklist.values()):
                # If so, mark this check as failed without running it
                self.checklist[check.__name__] = Exception(f"Skipped due to previous check failure")
            else:
                try:
                    check()
                    self.checklist[check.__name__] = True
                except Exception as e:
                    self.checklist[check.__name__] = e  # if the check fails, the error is stored in the checklist

    def __matrix__(self):
        return np.array([[p.id for p in gen] for gen in self._unphy_fields])

    # Final Goal: Write the field in the form of a dictionary
    # This is the function that will be used to write the field to the model file
    
    @staticmethod
    def rewrite_list(list):
        return str(list).replace("[", "{").replace("]", "}").replace("'", "")

    @staticmethod
    def rewrite_QuantumNumber(QuantumNumber):
        QuantumNumber = {key: str(Fraction(value,3)) for key, value in QuantumNumber.items() if value != 0}
        return str(QuantumNumber).replace(":", " -> ").replace("'", "")

    @staticmethod
    def rewrite_Indices(indices):
        indices = [repr(i) for i in indices]
        indices_str = "{" + ", ".join(indices) + "}"
        return indices_str

    @staticmethod
    def _propagator_label(class_name, member_name):
        prop_label_str = [class_name] + member_name
        return str(prop_label_str).replace("[", "{").replace("]", "}")

    @staticmethod
    def write_Mass(class_name, particle_list):
        """Format particle masses for FeynRules Mass."""
        if all(particle.fermion.mass == 0 for particle in particle_list):
            return 0
        else:
            Mass = ["M"+str(class_name).upper()]
            for particle in particle_list:
                mass_name = "M" + particle.fermion.name.upper()
                Mass.append((mass_name, particle.fermion.mass))
            Mass = str(Mass).replace("'", "")
            Mass = Mass.replace("[", "{").replace("]", "}")
            Mass = Mass.replace("(", "{").replace(")", "}")
            return Mass

    @staticmethod
    def write_Width(class_name, particle_list):
        """Format particle widths for FeynRules Width."""
        if all(particle.fermion.width == 0 for particle in particle_list):
            return 0
        else:
            Width = ["W"+str(class_name).upper()]
            for particle in particle_list:
                width_name = "W" + particle.fermion.name.upper()
                Width.append((width_name, particle.fermion.width))
            Width = str(Width).replace("'", "")
            Width = Width.replace("[", "{").replace("]", "}")
            Width = Width.replace("(", "{").replace(")", "}")
            return Width

    def write_Definition(self):
        """Write FeynRules definition for the multiplet."""
        definition = []
        proj_matrix = "ProjM" if self.chirality == "left" else "ProjP"
        # TODO: Check if the definition is correct
        if self.color > 1:
            if self.dim > 1:
                for i in range(self.dim):
                    class_name = name.fermion_field_name(self._phy_fields[i][0].charge, self.color)[0]
                    definition.append(f"{self.name}[sp1_, {i+1}, ff_, cc_] :> Module[{{sp2}}, {proj_matrix}[sp1, sp2] {class_name}[sp2, ff, cc]]")
            else:
                class_name = name.fermion_field_name(self._phy_fields[0][0].charge, self.color)[0]
                definition.append(f"{self.name}[sp1_, ff_, cc_] :> Module[{{sp2}}, {proj_matrix}[sp1, sp2] {class_name}[sp2, ff, cc]]")
        else:
            for i in range(self.dim):
                class_name = name.fermion_field_name(self._phy_fields[i][0].charge, self.color)[0]
                definition.append(f"{self.name}[sp1_, {i+1}, ff_] :> Module[{{sp2}}, {proj_matrix}[sp1, sp2] {class_name}[sp2, ff]]")

        return "{ " + (",\n" + " "*34).join(definition) + " }"


    def _phy_field_info(self):
        phy_fields = []
        for i in range(len(self._phy_fields)):
            class_name = name.fermion_field_name(self._phy_fields[i][0].charge, self.color)
            particle_name_list = [p.fermion.name for p in self._phy_fields[i]]
            particle_fullname_list = [p.fermion.full_name for p in self._phy_fields[i]]
            indices = [self.gen_idx] + [i for i in self.indices if i.color == True]
            qnumber = self.QuantumNumber.copy()
            qnumber.update({"Q": self._phy_fields[i][0].charge})
            
            phy_field = {
                "ClassName": f"{class_name[0]}",
                "ClassMembers": self.rewrite_list(particle_name_list),
                "Indices": self.rewrite_Indices(indices),
                "FlavorIndex": self.gen_idx, #  GenIdx and ColorIdx
                "Mass": self.write_Mass(class_name[0], self._phy_fields[i]), 
                "Width": self.write_Width(class_name[0], self._phy_fields[i]), 
                "QuantumNumber": self.rewrite_QuantumNumber(qnumber), 
                "PropagatorLabel": self._propagator_label(class_name[0], particle_name_list), 
                "PropagatorType": "Straight", 
                "PropagatorArrow": "Forward", 
                "PDG": [p.fermion.pdg_id for p in self._phy_fields[i]], 
                "ParticleName": particle_name_list, 
                "AntiParticleName": [p.fermion.name + "~" for p in self._phy_fields[i]], 
                "FullName": particle_fullname_list
            }
            phy_fields.append(phy_field)
        return phy_fields

    def _unphy_field_info(self):
        unphy_field = {
            "ClassName": self.name,
            "Unphysical": True, 
            "Indices": self.rewrite_Indices(self.indices), # GenIdx and ColorIdx and RepIdx 
            "FlavorIndex": self.gen_idx, # non-color RepIdx
            "SelfConjugate": self.self_conjugate,
            "QuantumNumber": self.rewrite_QuantumNumber(self.abelian_charges),
            "Definition": self.write_Definition()    
        }
        return unphy_field

    def covariant_derivative(self):
        return f"{self.name}bar.Ga[mu].DC[{self.name}, mu]"
    

### ======================== ###
###       Scalar Field       ###
### ======================== ###
class ScalarField(Field):
    def __init__(self, id, name, type, reps, groups, dim, gen, particles, self_conjugate, QuantumNumber):
        self.subtype = type
        self.reps = reps
        self.groups = groups
        
        super().__init__(id, name, "scalar", dim, gen, particles, self_conjugate, QuantumNumber)
        self.__check__()
        self.potential = None
        self.vev = None
        self.get_vev = False

    def __check__(self):
        super().__check__()
        subtype_check = self.subtype in ["real", "complex", "scalar"]
        reps_dict_check = isinstance(self.reps, dict) 
        reps_check = all(isinstance(x, int) or isinstance(x, str) for x in self.reps.values())
        reps_len_check = len(self.reps) == len(self.groups)
        groups_dict_check = isinstance(self.groups, dict)
        groups_check = all(isinstance(g, str) for g in self.groups)

        self.checklist.update(
            {"subtype_check": subtype_check,
             "reps_dict": reps_dict_check, 
             "reps": reps_check, 
             "reps_len": reps_len_check,
             "groups_dict": groups_dict_check, 
             "groups": groups_check
             }
        )
        
        # Override the ptcl_type from Field to ScalarField's subtype
        try:
            ptcl_type = all(p.type == self.subtype for p in self.particles)
        except:
            ptcl_type = False
        self.checklist["ptcl_type"] = ptcl_type

    def check_vev_potential(self):
        """ Check the potential allowed the vev """
        assert True, f"{self.vev} does not minimize {self.name}'s potential"

    def acquire_vev(self, vev):
        """ ScalarField Promotion to Higgs (similar to Pawn Promotion in Chess)"""
        assert isinstance(vev, list), f"AssertionError: {self.name}'s vev must be a list"
        assert self.dim == len(vev), f"AssertionError: {self.name} has dim-{self.dim} inconsistent with vev-{vev}"
        assert all(isinstance(v, (int, float)) for v in vev), f"AssertionError: {self.name}'s vev must be a list of numbers"
        self.get_vev = True
        self.vev = vev

    def __validate__(self):
        super().__validate__()
        all_checks = [self.check_vev_potential]
        for check in all_checks:
            try:
                check()
                self.checklist[check.__name__] = True
            except Exception as e:
                self.checklist[check.__name__] = e

    def _phy_field_info(self):
        pass

    def _unphy_field_info(self):
        pass

    def covariant_derivative(self):
        return f"DC[{self.name}bar, mu] DC[{self.name}, mu]"

### ============================ ###
###        Vector Field          ###
### ============================ ###
class VectorField(Field):
    def __init__(self, id, name, reps, groups, dim, gen, particles, self_conjugate, QuantumNumber):
        super().__init__(id, name, "vector", dim, gen, particles, self_conjugate, QuantumNumber)
        self.reps = reps
        self.groups = groups
        self.__check__()

    def __check__(self):
        super().__check__()
        reps_dict_check = isinstance(self.reps, dict) 
        reps_check = all(isinstance(x, int) or isinstance(x, str) for x in self.reps.values())
        reps_len_check = len(self.reps) == len(self.groups)
        groups_dict_check = isinstance(self.groups, dict)
        groups_check = all(isinstance(g, str) for g in self.groups)

        self.checklist.update(
            {"reps_dict": reps_dict_check, 
             "reps": reps_check, 
             "reps_len": reps_len_check,
             "groups_dict": groups_dict_check, 
             "groups": groups_check}
        )

if __name__ == "__main__":
    pass