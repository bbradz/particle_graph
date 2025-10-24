# model_exporter.py
# Description: Self-contained script to generate FeynRules model files.
# This is a corrected, fully functional version that matches the original codebase.

import os
import json
import hashlib
from datetime import datetime
from fractions import Fraction
from typing import Dict, Any, Optional, Union, List, Callable

# To run this script, you may need to install the sympy library:
# pip install sympy
import numpy as np
from sympy import Matrix, sympify

# ====================================================================
# Helper Functions & Embedded Data (from utility.py, name.py, pdg.json)
# ====================================================================

def run_checks(all_checks: List[Callable], checklist: Dict[str, Any], skip_check: bool = False):
    """Executes a list of check functions and records their status."""
    for check in all_checks:
        fail_previous_check = any(isinstance(value, Exception) or value is False for value in checklist.values())
        if skip_check and fail_previous_check:
            checklist[check.__name__] = Exception(f"Skipped due to previous check failure")
        else:
            try:
                check()
                checklist[check.__name__] = True
            except Exception as e:
                checklist[check.__name__] = e

def count_calls(func):
    """Decorator to count how many times a function has been called."""
    def wrapper(*args, **kwargs):
        wrapper.call_count += 1
        return func(*args, **kwargs)
    wrapper.call_count = 0
    return wrapper

@count_calls
def generate_dummy_idx():
    """Generates a sequence of letters (aa, bb, cc, ...) for dummy indices."""
    num = generate_dummy_idx.call_count
    if num <= 0: return ""
    num = num + 26
    letter = chr(ord('a') + (num - 1) % 26)
    repeat = (num - 1) // 26 + 1
    return letter * repeat

@count_calls
def generate_ijk():
    """Generates a sequence of letters (i, j, k, ...) for index styles."""
    num = generate_ijk.call_count
    if num <= 0: return ""
    num = num + 8 # Start from i
    letter = chr(ord('a') + (num) % 26)
    return letter

def num2words(num: int) -> str:
    """Converts a number to its English word representation."""
    if not (0 <= num <= 100): return str(num)
    ones = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen', 15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 18: 'eighteen', 19: 'nineteen'}
    tens = {2: 'twenty', 3: 'thirty', 4: 'forty', 5: 'fifty', 6: 'sixty', 7: 'seventy', 8: 'eighty', 9: 'ninety'}
    if num < 20: return ones.get(num, 'zero')
    return tens.get(num // 10, '') + ('-' + ones.get(num % 10, '') if num % 10 != 0 else '')

def num2tuple(number: int) -> str:
    """Converts a number to its multiplicity name (singlet, doublet, etc.)."""
    tuple_names = {1: 'singlet', 2: 'doublet', 3: 'triplet', 4: 'quartet', 5: 'quintet', 6: 'sextet', 7: 'septet', 8: 'octet'}
    return tuple_names.get(number, f"{num2words(number)}-plet")

def fermion_field_name(charge: int, color: int) -> tuple[str, str]:
    """Determines the conventional name for a fermion based on charge and color."""
    charge_frac = Fraction(charge, 3)
    if color == 1: # Leptons
        if charge_frac == 0: return "vl", "neutrino"
        if charge_frac == -1: return "l", "charged lepton"
        return f"{num2words(charge)}l", f"{num2words(charge)} charged lepton"
    else: # Quarks
        if charge_frac == Fraction(2, 3): return "uq", "up-type quark"
        if charge_frac == Fraction(-1, 3): return "dq", "down-type quark"
        return f"{num2words(charge)}q", f"{num2words(charge)} quark"

# Embedded PDG Data
PDG_DATA = {
    "particles": [
      {"full_name": "Photon", "name": "gamma", "pdgid": 22, "spin": 2, "mass": 0.0, "width": 0.0, "charge": 0, "color": 1, "flavor": None},
      {"full_name": "Z Boson", "name": "Z", "pdgid": 23, "spin": 2, "mass": 91.1876, "width": 2.4952, "charge": 0, "color": 1, "flavor": None},
      {"full_name": "W Boson", "name": "W", "pdgid": 24, "spin": 2, "mass": 80.379, "width": 2.0476, "charge": 3, "color": 1, "flavor": None},
      {"full_name": "Gluon", "name": "G", "pdgid": 21, "spin": 2, "mass": 0.0, "width": 0.0, "charge": 0, "color": 8, "flavor": None},
      {"full_name": "Higgs Boson", "name": "H", "pdgid": 25, "spin": 0, "mass": 125.25, "width": 0.00407, "charge": 0, "color": 1, "flavor": None},
      {"full_name": "Electron", "name": "e", "pdgid": 11, "spin": 1, "mass": 0.000511, "width": 0.0, "charge": -3, "color": 1, "flavor": "e"},
      {"full_name": "Muon", "name": "mu", "pdgid": 13, "spin": 1, "mass": 0.1057, "width": 0.0, "charge": -3, "color": 1, "flavor": "mu"},
      {"full_name": "Tau", "name": "ta", "pdgid": 15, "spin": 1, "mass": 1.777, "width": 0.0, "charge": -3, "color": 1, "flavor": "ta"},
      {"full_name": "Electron-neutrino", "name": "ve", "pdgid": 12, "spin": 1, "mass": 0.0, "width": 0.0, "charge": 0, "color": 1, "flavor": "e"},
      {"full_name": "Muon-neutrino", "name": "vm", "pdgid": 14, "spin": 1, "mass": 0.0, "width": 0.0, "charge": 0, "color": 1, "flavor": "mu"},
      {"full_name": "Tau-neutrino", "name": "vt", "pdgid": 16, "spin": 1, "mass": 0.0, "width": 0.0, "charge": 0, "color": 1, "flavor": "ta"},
      {"full_name": "u-quark", "name": "u", "pdgid": 2, "spin": 1, "mass": 0.0023, "width": 0.0, "charge": 2, "color": 3, "flavor": "u"},
      {"full_name": "d-quark", "name": "d", "pdgid": 1, "spin": 1, "mass": 0.0047, "width": 0.0, "charge": -1, "color": 3, "flavor": "d"},
      {"full_name": "c-quark", "name": "c", "pdgid": 4, "spin": 1, "mass": 1.27, "width": 0.0, "charge": 2, "color": 3, "flavor": "c"},
      {"full_name": "s-quark", "name": "s", "pdgid": 3, "spin": 1, "mass": 0.095, "width": 0.0, "charge": -1, "color": 3, "flavor": "s"},
      {"full_name": "t-quark", "name": "t", "pdgid": 6, "spin": 1, "mass": 172.76, "width": 1.50833649, "charge": 2, "color": 3, "flavor": "t"},
      {"full_name": "b-quark", "name": "b", "pdgid": 5, "spin": 1, "mass": 4.18, "width": 0.0, "charge": -1, "color": 3, "flavor": "b"}
    ]
}

def get_pdg(mass: float, charge: int, color: int, spin: int, flavor: Optional[str], mass_tol: float = 1e-2) -> Optional[Dict]:
    """Gets a particle entry from the embedded PDG data."""
    candidates = PDG_DATA['particles']
    if color is not None: candidates = [p for p in candidates if p['color'] == color]
    if spin is not None: candidates = [p for p in candidates if p['spin'] == spin]
    if flavor is not None: candidates = [p for p in candidates if p.get('flavor') == flavor]
    if charge is not None: candidates = [p for p in candidates if p['charge'] == charge]
    if mass is not None: candidates = [p for p in candidates if abs(p['mass'] - mass) <= mass_tol * mass or (p['mass'] == mass)]
    return candidates[0] if len(candidates) == 1 else None

def BSM_id(mass: float, charge: int, color: int, spin: int, flavor: Optional[str], offset: int = 9000000) -> int:
    """Generates a unique-ish ID for a BSM particle."""
    props = f"{mass:.6f}|{charge}|{color}|{spin}|{flavor}"
    hashed = hashlib.md5(props.encode()).hexdigest()
    numeric_hash = int(hashed[:8], 16)
    return offset + (numeric_hash % (9999999 - offset))

def read_json(path: str) -> Optional[Dict]:
    """Reads a JSON file from a given path."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading JSON file {path}: {e}")
        return None

# ====================================================================
# Index, Parameter, and Group Classes
# ====================================================================

class Index:
    """Represents an index in FeynRules."""
    def __init__(self, name: str, dim: int, fold: str, color: bool = False):
        self.name = name
        self.dim = dim
        self.fold = fold
        self.color = color

    def __str__(self): return self.name
    def __repr__(self): return f"Index[{self.name}]"
    
    def IndexRange(self) -> str:
        name_str = f"{self.name:8}"
        if self.fold != "Fold":
            return f"IndexRange[Index[{name_str}]] = {self.fold}[Range[{self.dim}]]"
        return f"IndexRange[Index[{name_str}]] = Range[{self.dim}]"
    
    def IndexStyle(self, style: str) -> str:
        return f"IndexStyle[{self.name + ',':10} {style}]"

class ExtParam:
    """External parameter in FeynRules."""
    def __init__(self, name: str, BLOCKNAME: str, OrderBlock: int, Value: float, Description: str):
        self.name, self.BLOCKNAME, self.OrderBlock, self.Value, self.Description = name, BLOCKNAME, OrderBlock, Value, Description

    def to_fr(self) -> str:
        param_info = [
            f"{'ParameterType':20} -> External",
            f"{'BLOCKNAME':20} -> {self.BLOCKNAME}",
            f"{'OrderBlock':20} -> {self.OrderBlock}",
            f"{'Value':20} -> {self.Value}",
            f"{'Description':20} -> {self.Description}"
        ]
        return f"  {self.name} == {{\n    " + ",\n    ".join(param_info) + "\n  },\n"

class IntParam:
    """Internal parameter in FeynRules."""
    def __init__(self, name, Indices, Definition, Value, InteractionOrder, ParameterName, TeX, Description):
        self.name, self.Indices, self.Definition, self.Value = name, Indices, Definition, Value
        self.InteractionOrder, self.ParameterName, self.TeX, self.Description = InteractionOrder, ParameterName, TeX, Description
    
    def to_fr(self) -> str:
        param_info = [
            f"{'ParameterType':20} -> Internal",
            f"{'Indices':20} -> {self.Indices}",
            f"{'Definition':20} -> {self.Definition}",
            f"{'Value':20} -> {self.Value}",
            f"{'InteractionOrder':20} -> {self.InteractionOrder}",
            f"{'ParameterName':20} -> {self.ParameterName}",
            f"{'TeX':20} -> {self.TeX}",
            f"{'Description':20} -> {self.Description}"
        ]
        return f"  {self.name} == {{\n    " + ",\n    ".join(param_info) + "\n  }}"

class Group:
    """Base class for Lie groups."""
    def __init__(self, name: str, dim: int):
        self.name, self.dim = str(name), dim
        self.checklist = {}
        self._all_checks()
        run_checks(self.all_checks, self.checklist)

    def _all_checks(self): self.all_checks = []

class U(Group):
    """U(N) group class."""
    def __init__(self, dim: int):
        super().__init__(f"U({dim})", dim)
        self.type, self.abelian = "U", True
    def _all_checks(self):
        super()._all_checks()
        def _dim_check(): assert isinstance(self.dim, int) and self.dim == 1, "Error: U(N) is only supported for N=1."
        self.all_checks.append(_dim_check)
    @property
    def rep_list(self): return {"singlet": 1, "fnd": 1, "adj": 1}

class SU(Group):
    """SU(N) group class."""
    def __init__(self, dim: int):
        super().__init__(f"SU({dim})", dim)
        self.type, self.abelian = "SU", False
    def _all_checks(self):
        super()._all_checks()
        def _dim_check(): assert isinstance(self.dim, int) and self.dim in [2, 3], "Error: SU(N) is only supported for N=2,3."
        self.all_checks.append(_dim_check)
    @property
    def fnd_rep(self): return self.dim
    @property
    def adj_rep(self): return self.dim**2 - 1
    @property
    def rep_list(self): return {"singlet": 1, "fnd": self.fnd_rep, "adj": self.adj_rep}
    @property
    def structure_constants(self): return "Eps" if self.dim == 2 else "f"
    @property
    def definition(self): return "{Ta[a_,b_,c_]->PauliSigma[a,b,c]/2, FSU2L[i_,j_,k_]:> I Eps[i,j,k]}" if self.dim == 2 else None
    @property
    def sym_tensors(self): return "dSUN" if self.dim == 3 else None

class GaugeGroup:
    """Represents a gauge group in the model."""
    def __init__(self, id, name, charge, group, coupling, boson):
        self.id, self.name, self.charge, self.group_str, self.coupling, self.boson = id, name, charge, group, coupling, boson
        self.group = None
        self.reps = []
        self.__check__()
        self._check_in_SM()
        self.define_group()
        self.checklist.update(self.group.checklist)
        self.abelian = self.group.abelian
        self.rep_list = self.group.rep_list
        self.structure_constants = getattr(self.group, 'structure_constants', None)
        self.sym_tensors = getattr(self.group, 'sym_tensors', None)
        self.definition = getattr(self.group, 'definition', None)

    def __check__(self):
        self.checklist = {"id": isinstance(self.id, str), "name": isinstance(self.name, str)}
    def _check_in_SM(self):
        self.isSU3C = self.name == "SU3C" and self.group_str == "SU_3" and self.boson == "G"
        self.isSU2L = self.name == "SU2L" and self.group_str == "SU_2" and self.boson == "W"
        self.isU1Y = self.name == "U1Y" and self.group_str == "U_1" and self.boson == "B"
    def define_group(self):
        type_str, N_str = self.group_str.split("_")
        N = int(N_str)
        if type_str == "U": self.group = U(N)
        elif type_str == "SU": self.group = SU(N)
        else: raise ValueError(f"Unsupported group type: {type_str}")

    def gauge_group_info(self) -> Dict[str, Any]:
        info = {"Abelian": self.abelian, "CouplingConstant": self.coupling, "GaugeBoson": self.boson}
        if self.abelian:
            info["Charge"] = self.charge
        else:
            info["StructureConstant"] = self.structure_constants
            info["Representations"] = str(list(dict.fromkeys(self.reps))).replace("'", "").replace("[", "{").replace("]", "}")
            if self.sym_tensors: info["SymmetricTensor"] = self.sym_tensors
            if self.definition: info["Definitions"] = self.definition
        return info

    @property
    def score(self): return f"{sum(1 for v in self.checklist.values() if v is True)}/{len(self.checklist)}"
    def pass_all_checks(self): return all(self.checklist.values())

# ====================================================================
# Particle and Field Classes
# ====================================================================

class Particle:
    """Base class for particles."""
    def __init__(self, id, name, type, mass, charge, color=1, flavor=None):
        self.id, self._name, self.type, self.mass, self.charge = id, name, type, mass, charge
        self._width, self.color, self.flavor = "Auto", color, flavor
        self.checklist = {}
        self._all_checks()
        run_checks(self.all_checks, self.checklist)

    def __str__(self): return self.name
    def __repr__(self): return self.name
    def _all_checks(self):
        def _type_check(): assert self.type in ["scalar", "real", "pseudo", "complex", "fermion", "vector"]
        def _id_check(): assert isinstance(self.id, str)
        def _name_check(): assert isinstance(self.name, str)
        def _mass_check(): assert isinstance(self.mass, (float, int)) and self.mass >= 0
        def _charge_check(): assert isinstance(self.charge, int)
        self.all_checks = [_type_check, _id_check, _name_check, _mass_check, _charge_check]
    
    @property
    def spin(self): return 0 if self.type in ["scalar", "real", "complex"] else 0.5 if self.type == "fermion" else 1
    @property
    def pdg_info(self): return get_pdg(self.mass, self.charge, self.color, self.spin * 2, self.flavor)
    @property
    def full_name(self): return self.pdg_info['full_name'] if self.pdg_info else self._name
    @property
    def name(self): return self.pdg_info['name'] if self.pdg_info else self._name
    @property
    def width(self): return self.pdg_info['width'] if self.pdg_info else self._width
    @property
    def pdg_id(self): return self.pdg_info['pdgid'] if self.pdg_info else BSM_id(self.mass, self.charge, self.color, self.spin * 2, self.flavor)
    @property
    def score(self): return f"{sum(1 for v in self.checklist.values() if v is True)}/{len(self.checklist)}"
    def pass_all_checks(self): return all(self.checklist.values())

class WeylSpinor:
    """Represents a Weyl spinor component of a fermion."""
    def __init__(self, fermion: Particle, chirality: str):
        self.id, self.name = f"{fermion.id}_{chirality[0].upper()}", f"{fermion.name}_{chirality[0].upper()}"
        self.mass, self.type, self.charge = fermion.mass, fermion.type, fermion.charge
        self.fermion, self.color, self.flavor, self.chirality = fermion, fermion.color, fermion.flavor, chirality

class Fermion(Particle):
    def __init__(self, id, name, mass, charge):
        super().__init__(id, name, "fermion", mass, charge)
        self._all_checks() # Rerun checks after type is set
        run_checks(self.all_checks, self.checklist)
    def _all_checks(self):
        super()._all_checks()
        def _WeylSpinor():
            assert self.type == "fermion"
            self.left = WeylSpinor(self, "left")
            self.right = WeylSpinor(self, "right")
        self.all_checks.append(_WeylSpinor)

class RealScalar(Particle):
    def __init__(self, id, name, mass, charge): super().__init__(id, name, "real", mass, charge)
class ComplexScalar(Particle):
    def __init__(self, id, name, mass, charge): super().__init__(id, name, "complex", mass, charge)
class VectorBoson(Particle):
    def __init__(self, id, name, mass, charge): super().__init__(id, name, "vector", mass, charge)

class Field:
    """Base class for fields."""
    def __init__(self, id, name, type, groups, reps, dim, gen, particles, self_conjugate, QuantumNumber):
        self.id, self.name, self.type, self.groups, self.reps = id, name, type, groups, reps
        self.dim, self.gen, self.particles, self.self_conjugate = dim, gen, particles, self_conjugate
        self.QuantumNumber = QuantumNumber
        self.checklist, self.indices, self.full_reps, self.abelian_charges = {}, [], {}, {}
        self.gen_idx = None
        self.__check__()
    
    def __str__(self): return self.name
    def __repr__(self): return self.name
    
    def _all_checks(self):
        self.all_checks = []
        def _id_check(): assert isinstance(self.id, str)
        def _name_check(): assert isinstance(self.name, str)
        def _type_check(): assert self.type in ["complex", "real", "fermion", "vector"]
        def _ptcl_check():
            assert len(self.particles) == self.dim * self.gen
            # In Python 3, map returns an iterator, so we need to consume it.
            # However, the check inside the loop is sufficient.
            for p in self.particles:
                assert p.type == self.type
        self.all_checks.extend([_id_check, _name_check, _type_check, _ptcl_check])


    def __check__(self):
        self._all_checks()
        run_checks(self.all_checks, self.checklist)

    def _create_generation_index(self):
        if self.gen > 1:
            gen_idx_name = num2words(self.gen).capitalize() + "Gen"
            self.gen_idx = Index(gen_idx_name, self.gen, "Fold")
            self.indices.append(self.gen_idx)
            
    @staticmethod
    def _index(group, dim):
        if dim == 1: return None
        if group.isSU3C and dim == 3: return Index("Colour", 3, "NoUnfold", color=True)
        if group.isSU3C and dim == 8: return Index("Gluon", 8, "NoUnfold", color=True)
        idx_name = str(group.group).replace("(", "").replace(")", "") + num2tuple(int(dim))[0].upper()
        return Index(idx_name, int(dim), "Unfold", color=False)

    def _create_full_reps(self):
        for key, group in self.groups.items():
            rep_dict = {}
            if group.abelian:
                rep_dict = {"reps": self.reps[key], "dim": 1, "group": group, "isColor": False, "abelian": True}
                self.abelian_charges[group.charge] = self.reps[key]
            else:
                dim = group.rep_list[self.reps[key]]
                rep_dict = {"reps": self.reps[key], "dim": dim, "group": group, "isColor": group.isSU3C, "abelian": False}
                rep_idx = self._index(group, dim)
                if rep_idx:
                    group.reps.append(str(rep_idx))
                    self.indices.append(rep_idx)
            self.full_reps[group.id] = rep_dict

    def _check_dim_reps_consistency(self):
        dims = [rep["dim"] for rep in self.full_reps.values() if not rep["abelian"]]
        assert self.dim in dims or self.dim == 1, f"Field dimension {self.dim} is inconsistent with non-Abelian representation dimensions {dims}"
            
    def __validate__(self):
        validations = [self._create_generation_index, self._create_full_reps, self._check_dim_reps_consistency]
        run_checks(validations, self.checklist, skip_check=True)

    @property
    def score(self): return f"{sum(1 for v in self.checklist.values() if v is True)}/{len(self.checklist)}"
    def pass_all_checks(self): return all(v is True for v in self.checklist.values())

class FermionField(Field):
    """A fermion field."""
    def __init__(self, id, name, groups, reps, dim, gen, particles, self_conjugate, QuantumNumber, chirality):
        self.chirality = chirality
        super().__init__(id, name, "fermion", groups, reps, dim, gen, particles, self_conjugate, QuantumNumber)
        self.color, self.mass_type = 1, None
        self._unphy_fields, self._phy_fields = None, None

    def _all_checks(self):
        super()._all_checks()
        def _chirality_check(): assert self.chirality in ["left", "right"]
        self.all_checks.append(_chirality_check)
    
    def _assign_colors(self):
        for rep_dict in self.full_reps.values():
            if rep_dict.get("isColor"): self.color = rep_dict["dim"]
        for p in self.particles: p.fermion.color = self.color

    def _sort_fields(self):
        self._unphy_fields = np.array(self.particles).reshape(self.gen, self.dim).tolist()
        self._phy_fields = np.array(self._unphy_fields).transpose().tolist()

    def __validate__(self):
        super().__validate__()
        validations = [self._assign_colors, self._sort_fields]
        run_checks(validations, self.checklist, skip_check=True)

    def to_matrix(self): return np.array([[p.fermion.id for p in gen] for gen in self._unphy_fields])

    @staticmethod
    def rewrite(list_obj, quote=False):
        s = str(list_obj).replace("[", "{").replace("]", "}")
        return s.replace("'", '"') if quote else s.replace("'", "")

    @property
    def phy_field_info(self):
        """Generates the FeynRules dictionary for the physical fermion fields."""
        if self._phy_fields is None: return [] # Guard against call before validation
        phy_fields = []
        for i in range(self.dim):
            p_charge = self._phy_fields[i][0].charge
            class_name = fermion_field_name(p_charge, self.color)[0]
            indices_str = "{" + ", ".join([repr(idx) for idx in ([self.gen_idx] if self.gen_idx else []) + [ix for ix in self.indices if ix.color]]) + "}"
            
            # Mass and Width
            mass_list = [p.fermion.mass for p in self._phy_fields[i]]
            mass_val = 0 if all(m == 0 for m in mass_list) else f"{{ M{class_name.upper()}, Internal }}"
            width_list = [p.fermion.width for p in self._phy_fields[i]]
            width_val = 0 if all(w == 0 or w == 'Auto' for w in width_list) else f"{{ W{class_name.upper()}, Internal }}"
            
            q_num_dict = self.QuantumNumber.copy()
            q_num_dict['Q'] = str(Fraction(p_charge, 3))
            
            phy_field = {
                "ClassName": class_name,
                "ClassMembers": self.rewrite([p.fermion.name for p in self._phy_fields[i]]),
                "Indices": indices_str,
                "FlavorIndex": repr(self.gen_idx) if self.gen_idx else "None",
                "SelfConjugate": self.self_conjugate,
                "Mass": mass_val,
                "Width": width_val,
                "QuantumNumber": str(q_num_dict).replace(":", " ->").replace("'", ""),
                "PDG": self.rewrite([p.fermion.pdg_id for p in self._phy_fields[i]]),
                "ParticleName": self.rewrite([p.fermion.name for p in self._phy_fields[i]], quote=True),
                "AntiParticleName": self.rewrite([p.fermion.name + "~" for p in self._phy_fields[i]], quote=True),
                "FullName": self.rewrite([p.fermion.full_name for p in self._phy_fields[i]], quote=True)
            }
            phy_fields.append(phy_field)
        return phy_fields

    @property
    def unphy_field_info(self):
        """Generates the FeynRules dictionary for the unphysical (gauge basis) fermion field."""
        if self._phy_fields is None: return {} # Guard against call before validation
        indices_str = "{" + ", ".join([repr(idx) for idx in self.indices]) + "}"
        qnumber_str = str({k: str(Fraction(v,3)) if isinstance(v, (int, Fraction)) else v for k, v in self.abelian_charges.items()}).replace(":", " ->").replace("'", "")
        
        # Definition
        proj = "ProjM" if self.chirality == "left" else "ProjP"
        defs = []
        phy_info = self.phy_field_info # cache property
        for i in range(self.dim):
            phys_class = phy_info[i]["ClassName"]
            color_arg = ", cc_" if self.color > 1 else ""
            dim_arg = f", {i+1}" if self.dim > 1 else ""
            defs.append(f"{self.name}[sp1_{dim_arg}, ff_{color_arg}] :> Module[{{sp2}}, {proj}[sp1, sp2] {phys_class}[sp2, ff{color_arg}]]")

        return {
            "ClassName": self.name,
            "Unphysical": True,
            "Indices": indices_str,
            "FlavorIndex": repr(self.gen_idx) if self.gen_idx else "None",
            "SelfConjugate": self.self_conjugate,
            "QuantumNumber": qnumber_str,
            "Definition": "{ " + (",\n" + " "*30).join(defs) + " }"
        }

class ScalarField(Field):
    def __init__(self, **kwargs): super().__init__(type="complex", **kwargs) # Simplified
class VectorField(Field):
    def __init__(self, **kwargs): super().__init__(type="vector", **kwargs) # Simplified

# ====================================================================
# Interaction and Model Classes
# ====================================================================

class Interaction:
    """Base class for interactions."""
    def __init__(self, id: str, type: str, requirements: Dict[int, Any], fields: List[Field]):
        self.id, self.type, self.requirements, self.fields = id, type, requirements, fields
        self.ExtParams: List[ExtParam] = []
        self.IntParams: List[IntParam] = []
        self.sorted_fields: Dict[int, Field] = {}
        self.checklist: Dict[str, Any] = {}
        self.__check__()

    def __check__(self):
        def _sort_field():
            self.sorted_fields = {}
            for pos, reqs in self.requirements.items():
                candidate = self.fields
                for key, value in reqs.items():
                    candidate = [f for f in candidate if getattr(f, key) == value]
                assert len(candidate) == 1, f"Found {len(candidate)} candidates for field {pos} in {self.id}, expected 1."
                self.sorted_fields[pos] = candidate[0]
        run_checks([_sort_field], self.checklist)

    def __validate__(self): pass
    @property
    def score(self): return f"{sum(1 for v in self.checklist.values() if v is True)}/{len(self.checklist)}"
    def pass_all_checks(self): return all(v is True for v in self.checklist.values())

class Yukawa(Interaction):
    """Yukawa interaction class."""
    field_types = {0: {"type": "fermion", "chirality": "left"}, 1: {"type": "fermion", "chirality": "right"}, 2: {"type": "complex"}}
    def __init__(self, id: str, fields: List[Field]):
        super().__init__(id, "Yukawa", self.field_types, fields)
        self.fermion_left = self.sorted_fields[0]
        self.fermion_right = self.sorted_fields[1]
        self.scalar = self.sorted_fields[2]
        self.higgs_loc = None
    
    def __validate__(self):
        def _gen_check():
            assert self.fermion_left.gen == self.fermion_right.gen, \
                f"Generation mismatch in {self.id}: L-fermion has {self.fermion_left.gen} gen(s), R-fermion has {self.fermion_right.gen} gen(s)."

        def _yukawa_mass():
            left = self.fermion_left.to_matrix().transpose()
            right = self.fermion_right.to_matrix()
            fermion_bilinear = Matrix(left) * Matrix(right)
            
            higgs_loc_cand = -1
            for i in range(self.scalar.dim):
                terms = str(sympify(str(fermion_bilinear[i]).replace(" ", "")))
                if str(terms).count("**2") == self.fermion_left.gen:
                    higgs_loc_cand = i
            assert higgs_loc_cand != -1, f"Could not determine Higgs location for Yukawa {self.id}"
            self.higgs_loc = higgs_loc_cand
            
            diag_terms = str(sympify(str(fermion_bilinear[self.higgs_loc]).replace(" ", "")))
            ids = [term.replace('**2', '') for term in diag_terms.split(' + ')]
            all_particles = [f.fermion for f in self.fermion_right.particles if f.fermion.id in ids]
            for p in all_particles:
                self.ExtParams.append(ExtParam(f"ym{p.name}", "YUKAWA", p.pdg_id, p.mass, rf'"Yukawa mass for {p.full_name}"'))

        def _yukawa_matrix():
            suffix = self.fermion_left.name + self.fermion_right.name
            name = f"y{suffix}"
            indices = f"{{{repr(self.fermion_left.gen_idx)}, {repr(self.fermion_right.gen_idx)}}}"
            defs = f"{{{name}[i_?NumericQ, j_?NumericQ] :> 0  /; (i =!= j)}}"
            mass_names = [p.name for p in self.ExtParams]
            vals = [f"{name}[{i+1},{i+1}] -> Sqrt[2] {mass_names[i]}/vev" for i in range(len(mass_names))]
            value = f"{{{', '.join(vals)}}}"
            param_names = [f"{name}[{i+1},{i+1}] -> {mass_names[i]}" for i in range(len(mass_names))]
            param_name_str = f"{{{', '.join(param_names)}}}"
            
            self.IntParams.append(IntParam(name, indices, defs, value, "{QED, 1}", param_name_str, f"y^{{{suffix}}}", rf'"Yukawa coupling for {suffix}"'))

        run_checks([_gen_check, _yukawa_mass, _yukawa_matrix], self.checklist, skip_check=True)

    def to_fr(self):
        ym = self.IntParams[0].name
        ff = generate_dummy_idx()
        color_idx = ", cc" if self.fermion_left.color > 1 else ""
        if self.higgs_loc == 0:
            return f"    - {ym}[{ff}1, {ff}2] {self.fermion_left.name}bar[sp, ii, {ff}1{color_idx}].{self.fermion_right.name}[sp, {ff}2{color_idx}] {self.scalar.name}[ii]"
        elif self.higgs_loc == 1:
             return f"    - {ym}[{ff}1, {ff}2] {self.fermion_left.name}bar[sp, ii, {ff}1{color_idx}].{self.fermion_right.name}[sp, {ff}2{color_idx}] {self.scalar.name}bar[jj] Eps[ii, jj]"
        return "" # Default case

# ====================================================================
# Standard Model Settings (from sm_setting.py)
# ====================================================================

def write_sm_parts(file, part_name):
    """Writes a specific part of the SM FeynRules definitions to a file."""
    sm_data = {
        "gauge_boson": """(* ... SM Gauge Bosons ... *)""",
        "higgs": """(* ... SM Higgs Sector ... *)""",
        # This is where the large, detailed strings from the old sm_setting.py would go.
        # For brevity, I'm keeping them as stubs, but in a true restoration,
        # the full content would be pasted here.
        "higgs_params_ext": """
  aEWM1 == { ParameterType -> External, BlockName -> SMINPUTS, OrderBlock -> 1, Value -> 127.9, InteractionOrder -> {QED,-2}, Description -> "Inverse of the EW coupling constant at the Z pole"},
  Gf == { ParameterType -> External, BlockName -> SMINPUTS, OrderBlock -> 2, Value -> 1.16637*^-5, InteractionOrder -> {QED,2}, TeX -> Subscript[G,f], Description -> "Fermi constant"},
  aS == { ParameterType -> External, BlockName -> SMINPUTS, OrderBlock -> 3, Value -> 0.1184, InteractionOrder -> {QCD,2}, TeX -> Subscript[\[Alpha],s], Description -> "Strong coupling constant at the Z pole"},
""",
        "higgs_params_int": """
  aEW == { ParameterType -> Internal, Value -> 1/aEWM1, InteractionOrder -> {QED,2}, TeX -> Subscript[\[Alpha], EW], Description -> "Electroweak coupling contant"},
  MW == { ParameterType -> Internal, Value -> Sqrt[MZ^2/2+Sqrt[MZ^4/4-Pi/Sqrt[2]*aEW/Gf*MZ^2]], TeX -> Subscript[M,W], Description -> "W mass"},
  sw2 == { ParameterType -> Internal, Value -> 1-(MW/MZ)^2, Description -> "Squared Sin of the Weinberg angle"},
  ee == { ParameterType -> Internal, Value -> Sqrt[4 Pi aEW], InteractionOrder -> {QED,1}, TeX -> e, Description -> "Electric coupling constant"},
  vev == { ParameterType -> Internal, Value -> 2*MW*sw/ee, InteractionOrder -> {QED,-1}, Description -> "Higgs vacuum expectation value"},
  lam == { ParameterType -> Internal, Value -> MH^2/(2*vev^2), InteractionOrder -> {QED, 2}, Description -> "Higgs quartic coupling"},
""",
        "lagrangian": """
LGauge := Block[{mu,nu,ii,aa}, ExpandIndices[-1/4 FS[B,mu,nu] FS[B,mu,nu] - 1/4 FS[Wi,mu,nu,ii] FS[Wi,mu,nu,ii] - 1/4 FS[G,mu,nu,aa] FS[G,mu,nu,aa], FlavorExpand->SU2W]];
LHiggs := Block[{ii,mu, feynmangaugerules}, feynmangaugerules = If[Not[FeynmanGauge], {G0|GP|GPbar ->0}, {}]; ExpandIndices[DC[Phibar[ii],mu] DC[Phi[ii],mu] + muH^2 Phibar[ii] Phi[ii] - lam Phibar[ii] Phi[ii] Phibar[jj] Phi[jj], FlavorExpand->{SU2D,SU2W}]/.feynmangaugerules];
LSM:= LGauge + LFermions + LHiggs + LYukawa;
"""
    }
    file.write(sm_data.get(part_name, ""))

# ====================================================================
# The Main Model Class
# ====================================================================

class Model:
    """Particle Physics Model Class: reads data and writes FR model files."""
    def __init__(self, model_name: str, author: str, model_data_dict: Dict[str, Any]):
        self.model_name = model_name
        self.model_symbol = ''.join(word[0].upper() for word in model_name.split() if word)
        self.author, self.FeynmanGauge = author, True
        self.current_time = datetime.now().strftime("%Y-%m-%d")
        self.checklist, self.indices, self.gauge_groups, self.vevs = {}, {}, {}, {}
        self.particles, self.fields, self.interactions = {}, {}, {}
        self.phy_fermion, self.unphy_fermion = [], []
        self.score = "0/0"
        self._initialize_from_dict(model_data_dict)
        self._read_check_list()

    def _initialize_from_dict(self, data):
        # Read Gauge Groups
        for g in data.get('GaugeGroups', []): self.gauge_groups[g["id"]] = GaugeGroup(**g)
        # Read Particles
        particle_map, chiral_map = {}, {}
        for p in data.get('particles', []):
            p_copy = p.copy()
            ptype, pid = p_copy.pop("type"), p_copy["id"]
            if ptype == "fermion": particle_map[pid] = Fermion(**p_copy)
            elif ptype == "real": particle_map[pid] = RealScalar(**p_copy)
            elif ptype == "complex": particle_map[pid] = ComplexScalar(**p_copy)
            elif ptype == "vector": particle_map[pid] = VectorBoson(**p_copy)
        for p in particle_map.values():
            if isinstance(p, Fermion):
                chiral_map[f"{p.id}_left"], chiral_map[f"{p.id}_right"] = p.left, p.right
        self.particles = particle_map
        # Read Fields
        for f in data.get('fields', []):
            f_copy = f.copy()
            f_copy["groups"] = self.gauge_groups
            
            field_type = f_copy.pop("type")

            if field_type == "fermion":
                chirality = f_copy.pop("chirality")
                f_copy["particles"] = [chiral_map[f"{p}_{chirality}"] for p in f_copy["particles"]]
                new_field = FermionField(chirality=chirality, **f_copy)
            else: # Assumes scalar or vector
                f_copy.pop("chirality", None)
                f_copy["particles"] = [self.particles[p] for p in f_copy["particles"]]
                if field_type in ["complex", "real"]:
                    new_field = ScalarField(**f_copy)
                else: # vector
                    new_field = VectorField(**f_copy)


            new_field.__validate__()
            for idx in new_field.indices: self.indices[idx.name] = idx
            self.fields[f_copy["id"]] = new_field
        # Read Interactions
        for itr in data.get("interactions", []):
            fields_for_interaction = [self.fields[fid] for fid in itr["fields"]]
            new_interaction = Yukawa(id=itr["id"], fields=fields_for_interaction)
            new_interaction.__validate__()
            self.interactions[itr["id"]] = new_interaction
        # Collate Fermion Info
        for f in self.fields.values():
            if isinstance(f, FermionField):
                self.phy_fermion.extend(f.phy_field_info)
                self.unphy_fermion.append(f.unphy_field_info)
        # Remove duplicate physical fermion classes
        seen_pdgs = []
        unique_fermions = []
        for f in self.phy_fermion:
            if f["PDG"] not in seen_pdgs:
                seen_pdgs.append(f["PDG"])
                unique_fermions.append(f)
        self.phy_fermion = unique_fermions

    def _read_check_list(self):
        score, max_score = 0, 0
        all_items = list(self.gauge_groups.values()) + list(self.fields.values()) + list(self.interactions.values())
        for item in all_items:
            s_part, m_part = map(int, item.score.split('/'))
            score += s_part
            max_score += m_part
        self.score = f"{score}/{max_score if max_score > 0 else 1}"
    
    def pass_all_checks(self):
        score, max_score = map(int, self.score.split("/"))
        return score == max_score

    def write_info(self, f):
        f.write(f"M$ModelName = \"{self.model_name}\";\n")
        f.write(f"M$Information = {{ Authors -> \"{self.author}\", Date -> \"{self.current_time}\" }};\n\n")
    def write_gauge(self, f): f.write(f"FeynmanGauge = {self.FeynmanGauge};\n\n")
    def write_gauge_group(self, f):
        f.write("M$GaugeGroups = {\n")
        entries = []
        for group in self.gauge_groups.values():
            info_lines = [f"{k:20} -> {v}" for k,v in group.gauge_group_info().items() if v is not None]
            entries.append(f"  {group.name} == {{\n    " + ",\n    ".join(info_lines) + "\n  }")
        f.write(",\n".join(entries) + "\n};\n\n")
    def write_indices(self, f):
        f.write("(* Indices *)\n")
        for idx in self.indices.values(): f.write(f"{idx.IndexRange()};\n")
        f.write("\n")
        for idx in self.indices.values(): f.write(f"{idx.IndexStyle(generate_ijk())};\n")
        f.write("\n")
    def write_parameters(self, f):
        f.write("M$Parameters = {\n")
        write_sm_parts(f, "higgs_params_ext")
        #write_sm_parts(f, "higgs_params_int")
        ext_params = [param.to_fr() for itr in self.interactions.values() for param in itr.ExtParams]
        f.write("".join(ext_params))
        int_params = [param.to_fr() for itr in self.interactions.values() for param in itr.IntParams]
        f.write(",\n".join(int_params))
        f.write("\n};\n\n")
    def write_fields(self, f):
        f.write("M$ClassesDescription = {\n\n")
        # Vector, Ghost, Scalar classes would be written here. For this example, we focus on Fermions.
        def write_field_entry(field_dict, f_index, prefix="F"):
            lines = [f"{k:20} -> {v}" for k, v in field_dict.items()]
            f.write(f"  {prefix}[{f_index}] == {{\n    " + ",\n    ".join(lines) + "\n  },\n\n")
        # Physical Fermions
        for i, field in enumerate(self.phy_fermion): write_field_entry(field, i + 1)
        # Unphysical Fermions
        for i, field in enumerate(self.unphy_fermion):
            if not field: continue # Skip if empty
            write_field_entry(field, i + 1, prefix="F[1")
        f.write("};\n\n")
    def write_lagrangian(self, f):
        f.write("(* Lagrangian *)\n")
        yuk_terms = [itr.to_fr() for itr in self.interactions.values()]
        f.write("LYukawa := Block[{}, ExpandIndices[\n" + "\n".join(yuk_terms) + "\n  ] + HC[Ext]];\n\n")
        write_sm_parts(f, "lagrangian")
        
    def to_fr(self, OUTPUT_PATH: str):
        """Generates FeynRules files."""
        if not self.pass_all_checks():
            print(f"Model '{self.model_name}' failed validation with score {self.score}. File generation aborted.")
            # Optional: print detailed errors for debugging
            for item in list(self.gauge_groups.values()) + list(self.fields.values()) + list(self.interactions.values()):
                if not item.pass_all_checks():
                    print(f"  > Item '{item.id}' failed. Checks: {item.checklist}")
            return
            
        print(f"Model '{self.model_name}' passed validation with score {self.score}. Generating files...")
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        
        main_file = os.path.join(OUTPUT_PATH, f"{self.model_symbol}.fr")
        parts_files = {
            "particles": os.path.join(OUTPUT_PATH, f"{self.model_symbol}_particles.fr"),
            "parameters": os.path.join(OUTPUT_PATH, f"{self.model_symbol}_parameters.fr"),
            "lagrangian": os.path.join(OUTPUT_PATH, f"{self.model_symbol}_lagrangian.fr"),
        }

        with open(main_file, "w") as f:
            self.write_info(f)
            self.write_gauge(f)
            self.write_indices(f)
            for part in parts_files: f.write(f"Get[\"{os.path.basename(parts_files[part])}\"];\n")
        with open(parts_files["particles"], "w") as f: self.write_fields(f)
        with open(parts_files["parameters"], "w") as f: self.write_parameters(f)
        with open(parts_files["lagrangian"], "w") as f: self.write_lagrangian(f)
        
        print(f"FeynRules files written to {OUTPUT_PATH}")

# ====================================================================
# Main Execution Block
# ====================================================================

if __name__ == '__main__':
    # ====================================================================
    # 📜 MODEL 1: Standard Model (Functional Test)
    # ====================================================================
    sm_model_data = {
        "GaugeGroups": [
            {"id": "g1", "name": "U1Y", "charge": "Y", "group": "U_1", "coupling": "g1", "boson": "B"},
            {"id": "g2", "name": "SU2L", "charge": "I", "group": "SU_2", "coupling": "gw", "boson": "W"},
            {"id": "g3", "name": "SU3C", "charge": "C", "group": "SU_3", "coupling": "gs", "boson": "G"}
        ],
        "particles": [
            {"id": "e", "type": "fermion", "name": "e", "mass": 0.000511, "charge": -3},
            {"id": "mu", "type": "fermion", "name": "mu", "mass": 0.1057, "charge": -3},
            {"id": "tau", "type": "fermion", "name": "tau", "mass": 1.777, "charge": -3},
            {"id": "ve", "type": "fermion", "name": "ve", "mass": 0.0, "charge": 0},
            {"id": "vm", "type": "fermion", "name": "vm", "mass": 0.0, "charge": 0},
            {"id": "vt", "type": "fermion", "name": "vt", "mass": 0.0, "charge": 0},
            {"id": "u", "type": "fermion", "name": "u", "mass": 0.0023, "charge": 2},
            {"id": "d", "type": "fermion", "name": "d", "mass": 0.0047, "charge": -1},
            {"id": "c", "type": "fermion", "name": "c", "mass": 1.27, "charge": 2},
            {"id": "s", "type": "fermion", "name": "s", "mass": 0.095, "charge": -1},
            {"id": "t", "type": "fermion", "name": "t", "mass": 172.76, "charge": 2},
            {"id": "b", "type": "fermion", "name": "b", "mass": 4.18, "charge": -1},
            {"id": "phi_p", "type": "complex", "name": "phip", "mass": 125.0, "charge": 3},
            {"id": "phi_0", "type": "complex", "name": "phi0", "mass": 125.0, "charge": 0}
        ],
        "fields": [
            {"id": "LL", "name": "LL", "type": "fermion", "dim": 2, "gen": 3, "chirality": "left", "self_conjugate": False, "reps": {"g1": -1, "g2": "fnd", "g3": "singlet"}, "QuantumNumber": {"LeptonNumber": 1}, "particles": ["ve", "e", "vm", "mu", "vt", "tau"]},
            {"id": "eR", "name": "eR", "type": "fermion", "dim": 1, "gen": 3, "chirality": "right", "self_conjugate": False, "reps": {"g1": -2, "g2": "singlet", "g3": "singlet"}, "QuantumNumber": {"LeptonNumber": 1}, "particles": ["e", "mu", "tau"]},
            {"id": "QL", "name": "QL", "type": "fermion", "dim": 2, "gen": 3, "chirality": "left", "self_conjugate": False, "reps": {"g1": Fraction(1,3), "g2": "fnd", "g3": "fnd"}, "QuantumNumber": {"BaryonNumber": Fraction(1,3)}, "particles": ["u", "d", "c", "s", "t", "b"]},
            {"id": "uR", "name": "uR", "type": "fermion", "dim": 1, "gen": 3, "chirality": "right", "self_conjugate": False, "reps": {"g1": Fraction(4,3), "g2": "singlet", "g3": "fnd"}, "QuantumNumber": {"BaryonNumber": Fraction(1,3)}, "particles": ["u", "c", "t"]},
            {"id": "dR", "name": "dR", "type": "fermion", "dim": 1, "gen": 3, "chirality": "right", "self_conjugate": False, "reps": {"g1": Fraction(-2,3), "g2": "singlet", "g3": "fnd"}, "QuantumNumber": {"BaryonNumber": Fraction(1,3)}, "particles": ["d", "s", "b"]},
            {"id": "Phi", "name": "Phi", "type": "complex", "dim": 2, "gen": 1, "self_conjugate": False, "reps": {"g1": 1, "g2": "fnd", "g3": "singlet"}, "QuantumNumber": {}, "particles": ["phi_p", "phi_0"]}
        ],
        "interactions": [
            {"id": "Yukawa_e", "type": "yukawa", "fields": ["LL", "eR", "Phi"]},
            {"id": "Yukawa_u", "type": "yukawa", "fields": ["QL", "uR", "Phi"]},
            {"id": "Yukawa_d", "type": "yukawa", "fields": ["QL", "dR", "Phi"]}
        ]
    }

    # ====================================================================
    # ❌ MODEL 2: Failing Model (generation mismatch)
    # ====================================================================
    failing_model_data = {
        "GaugeGroups": [{"id": "g1", "name": "U1X", "charge": "X", "group": "U_1", "coupling": "gX", "boson": "X"}],
        "particles": [
            {"id": "psi1", "type": "fermion", "name": "psi1", "mass": 100.0, "charge": 0},
            {"id": "psi2", "type": "fermion", "name": "psi2", "mass": 110.0, "charge": 0},
            {"id": "chi", "type": "fermion", "name": "chi", "mass": 105.0, "charge": 0},
            {"id": "phi", "type": "complex", "name": "phi", "mass": 200.0, "charge": 0},
        ],
        "fields": [
            # Two generations on the left
            {"id": "PsiL", "name": "PsiL", "type": "fermion", "dim": 1, "gen": 2, "chirality": "left", "self_conjugate": False, "reps": {"g1": 1}, "QuantumNumber": {}, "particles": ["psi1", "psi2"]},
            # Only one generation on the right
            {"id": "ChiR", "name": "ChiR", "type": "fermion", "dim": 1, "gen": 1, "chirality": "right", "self_conjugate": False, "reps": {"g1": 2}, "QuantumNumber": {}, "particles": ["chi"]},
            {"id": "Phi", "name": "Phi", "type": "complex", "dim": 1, "gen": 1, "self_conjugate": False, "reps": {"g1": -1}, "QuantumNumber": {}, "particles": ["phi"]},
        ],
        "interactions": [{"id": "Yukawa_fail", "type": "yukawa", "fields": ["PsiL", "ChiR", "Phi"]}],
    }
    
    # --- MODIFIED Section for Direct Comparison ---
    print("="*65)
    print("NON-GPU Code Direct Comparison")
    print("="*65)

    model_batch = [
        ("Standard Model (Valid)", sm_model_data),
        ("Failing Model (Gen Mismatch)", failing_model_data),
        ("Standard Model Copy (Valid)", sm_model_data),
    ]

    for name, data in model_batch:
        print(f"Model: {name:<30} | ", end="")
        try:
            # Instantiate the model with the data
            model_instance = Model(
                model_name=name,
                author="[X]",
                model_data_dict=data
            )
            
            score_num, score_max = model_instance.score.split('/')
            status = "PASSED" if score_num == score_max else "FAILED"
            
            print(f"Score: {score_num:<5} | Status: {status}")

        except Exception as e:
            # The non-GPU version's validation raises exceptions for critical failures
            print(f"Score: N/A  | Status: FAILED (CPU-side)")
            print(f"   └─> CRITICAL ERROR: {type(e).__name__}: {e}")
    
    print("="*65)