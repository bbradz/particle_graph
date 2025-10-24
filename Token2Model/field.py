### ========================================================================== ###
###                                                                            ###
###                           Particles Classes                                ###
###                                                                            ###
### ========================================================================== ###

import numpy as np
from fractions import Fraction
from . import name
from .particle import Particle, WeylSpinor
#from .utility import run_checks
from .check import run_checks

# ====================================================================
#                              Field
# ====================================================================
class Field:
    """
    Base class for fields.
    id: str
    name: str
    type: str ["complex", "real", "fermion", "vector"]
    reps: dict of str (non-abelian group reps) or int (abelian group charge)
    groups: dict of group
    dim: int > 0
    gen: int > 0
    particles: list of Particle
    self_conjugate: bool
    """
    def __init__(self, id, name, type, groups, reps, dim, gen, particles, self_conjugate):
        self.id = id
        self.name = name
        self.type = type
        self.groups = groups
        self.reps = reps
        self.dim = dim
        self.gen = gen
        self.particles = particles
        self.self_conjugate = self_conjugate
        self.full_reps = {}
        self.is_massive = False
        self.mass_term = []
        self.__check__()
        
        
    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __dict__(self):
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "reps": self.reps,
            "groups": self.groups,
            "dim": self.dim,
            "gen": self.gen,
            "particles": self.particles,
            "self_conjugate": self.self_conjugate
        }
    
    def _all_checks(self):
        """ All checks for the initial INPUTs of the 'Field' class. """
        self.all_checks = []
        
        from functools import wraps

        def _finalize_result(result: dict) -> dict:
            good_var = result.get('good_var', []) or []
            error_var = result.get('error_var', []) or []
            mattered = list({*good_var, *error_var})
            result['mattered_vars'] = mattered
            # If neither good nor error vars were specified, treat as block-level
            result['level'] = 'field' if mattered else 'block'
            return result
        
        def _wrap(fn):
            @wraps(fn)
            def wrapped():
                return _finalize_result(fn())
            return wrapped

        # ------------------------- Simple Checks ------------------------------
        # check if the name is a string
        def _name_check():
            result = {"score": 1, 
                      "error_var": [], 
                      "good_var": [f"fields.{self.id}.name"], 
                      "message": "Passed", 
                      "max_score": 1,
                      }
            if not isinstance(self.name, str):
                result.update({"score": 0, 
                               "error_var": [f"fields.{self.id}.name"], 
                               "good_var": [], 
                               "message": f"'name' must be a string."
                               })
            return result
        
        # check if the type is a string and one of complex/real/fermion/vector
        def _type_check():
            result = {"score": 1, 
                      "error_var": [], 
                      "good_var": [f"fields.{self.id}.type"], 
                      "message": "Passed", 
                      "max_score": 1,
                      }
            if not isinstance(self.type, str) or self.type not in ["complex", "real", "fermion", "vector"]:
                result.update({"score": 0, 
                               "error_var": [f"fields.{self.id}.type"], 
                               "good_var": [], 
                               "message": f"'type' must be a string and one of complex/real/fermion/vector"
                               })
            return result
        
        # check if the groups are a dictionary
        def _groups_check():
            result = {"score": 1, 
                      "error_var": [], 
                      "good_var": [f"fields.{self.id}.groups"], 
                      "message": "Passed", 
                      "max_score": 1,
                      }
            if not isinstance(self.groups, dict):
                result.update({"score": 0, 
                               "error_var": [f"fields.groups"], 
                               "good_var": [], 
                               "message": f"'groups' must be a dictionary"
                               })
            return result

        # check if the reps are a dictionary of int or str
        def _reps_check():
            result = {"score": 3, 
                      "error_var": [], 
                      "good_var": [f"fields.{self.id}.reps"], 
                      "message": "Passed", 
                      "max_score": 3,
                      }
            if not isinstance(self.reps, dict):
                result.update({"score": 0, 
                               "error_var": [f"fields.{self.id}.reps"], 
                               "good_var": [], 
                               "message": f"'reps' must be a dictionary"
                               })
                return result
            
            if len(self.reps) != len(self.groups):
                result.update({"score": 1, 
                               "error_var": [f"fields.{self.id}.reps"], 
                               "good_var": [], 
                               "message": f"'reps' and 'groups' must be the same length"
                               })
                return result
            
            error_var = [f"fields.{self.id}.reps.{key}" for key, value in self.reps.items() if not isinstance(value, (int, str))]
            
            if error_var:
                good_var = [f"fields.{self.id}.reps.{key}" for key, value in self.reps.items() if isinstance(value, (int, str))]
                result.update({"score": 2, 
                               "error_var": error_var, 
                               "good_var": good_var, 
                               "message": f"'reps' must be a dictionary of int or str"
                               })
                return result
            return result
        
        # check if the dimension is a positive integer
        def _dim_check():
            result = {"score": 1, 
                      "error_var": [], 
                      "good_var": [f"fields.{self.id}.dim"], 
                      "message": "Passed", 
                      "max_score": 1,
                      }
            if not isinstance(self.dim, int) or self.dim <= 0:
                result.update({"score": 0, 
                               "error_var": [f"fields.{self.id}.dim"], 
                               "good_var": [], 
                               "message": f"'dim' must be a positive integer"
                               })
            return result
        
        # check if the generation is a positive integer
        def _gen_check():
            result = {"score": 1, 
                      "error_var": [], 
                      "good_var": [f"fields.{self.id}.gen"], 
                      "message": "Passed", 
                      "max_score": 1,
                      }
            if not isinstance(self.gen, int) or self.gen <= 0:
                result.update({"score": 0, 
                               "error_var": [f"fields.{self.id}.gen"], 
                               "good_var": [], 
                               "message": f"'gen' must be a positive integer"
                               })
            return result
        
        # check if self-conjugate is a bool
        def _self_conjugate_check():
            result = {"score": 1, 
                      "error_var": [], 
                      "good_var": [f"fields.{self.id}.self_conjugate"], 
                      "message": "Passed", 
                      "max_score": 1,
                      }
            if not isinstance(self.self_conjugate, bool):
                result.update({"score": 0, 
                               "error_var": [f"fields.{self.id}.self_conjugate"], 
                               "good_var": [], 
                               "message": f"'self_conjugate' must be a bool"
                               })
            return result
        
        # ------------------------- Necessary Checks ------------------------------
        
        # check if the particles are a list of Particle
        def _particles_check():
            result = {"score": 2, 
                      "error_var": [], 
                      "good_var": [], 
                      "message": "Passed", 
                      "max_score": 2,
                      }
            if not isinstance(self.particles, list):
                result.update({"score": 0, 
                               "error_var": [f"fields.{self.id}.particles"], 
                               "good_var": [], 
                               "message": f"'particles' must be a list"
                               })
                return result
            
            if self.type == "fermion":
                error_var = [f"fields.{self.id}.particles.{p}" for p in self.particles if not isinstance(p, WeylSpinor)]
                if error_var:
                    good_var = [f"fields.{self.id}.particles.{p.fermion.id}" for p in self.particles if isinstance(p, WeylSpinor)]
                    result.update({"score": 1, 
                                   "error_var": error_var, 
                                   "good_var": good_var, 
                                   "message": f"'particles' in fermion fields must be a list of WeylSpinor"
                                   })
                    return result
            else:
                error_var = [f"fields.{self.id}.particles.{p}" for p in self.particles if not isinstance(p, Particle)]
                if error_var:
                    good_var = [f"fields.{self.id}.particles.{p}" for p in self.particles if isinstance(p, Particle)]
                    result.update({"score": 1, 
                                   "error_var": error_var, 
                                   "good_var": good_var, 
                                   "message": f"'particles' must be a list of Particle"
                                   })
                    return result
            
            return result

        # check if the field is consistent with the type
        def _gen_type_consistency():
            result = {"score": 1, 
                      "error_var": [], 
                      "good_var": [f"fields.{self.id}.gen"], 
                      "message": "Passed", 
                      "max_score": 1,
                      }
            if self.type != "fermion" and self.gen != 1:
                result.update({"score": 0, 
                               "error_var": [f"fields.{self.id}.gen"], 
                               "good_var": [], 
                               "message": f"Non-fermion fields can have only one generation"
                               })
            return result

        # Sort reps
        def _sort_reps():
            result = {"score": 3, 
                      "error_var": [], 
                      "good_var": [f"fields.{self.id}.reps"], 
                      "message": "Passed", 
                      "max_score": 3,
                      }
            rep_dict = {}
            self.allow_dim = [1]
            score = 0
            error_var = []
            error_msg = []
            for key, group in self.groups.items():
                try:
                    if group.abelian:
                        rep_dict[key] = Fraction(self.reps[key], 6)
                    else:
                        rep_dict[key] = group.dim(self.reps[key])
                        if group.isSU3C:
                            self.color = rep_dict[key]
                        else:
                            self.allow_dim.append(rep_dict[key])
                    score += 1
                except Exception as e:
                    error_var.append(f"fields.{self.id}.reps.{key}")
                    error_msg.append(f"{key}: {e}")
            if error_var:
                result.update({"score": score, 
                                "error_var": error_var, 
                                "good_var": [], 
                                "message": error_msg
                                })
                return result
            else:
                self.full_reps = rep_dict
                self.allow_dim = sorted(list(set(self.allow_dim)))
                return result

        # Check reps and dim consistency
        def _reps_dim_consistency():    
            # g2 is SU(2) group and is directly related to the dim in SU3xSU2xU1      
            result = {"score": 1, 
                      "error_var": [], 
                      "good_var": [f"fields.{self.id}.dim", f"fields.{self.id}.reps.g2"], 
                      "message": "Passed", 
                      "max_score": 1,
                      }
            if self.dim == 1 and self.allow_dim != [1]:
                result.update({"score": 0, 
                               "error_var": [f"fields.{self.id}.dim", f"fields.{self.id}.reps.g2"] , 
                               "good_var": [], 
                               "message": f"reps and dim are not consistent"
                               })
                return result
            elif self.dim != 1 and self.allow_dim != [1, self.dim]:
                result.update({"score": 0, 
                               "error_var": [f"fields.{self.id}.dim", f"fields.{self.id}.reps.g2"], 
                               "good_var": [], 
                               "message": f"reps and dim are not consistent"
                               })
                return result
            return result

        # compute allowed charges
        def _allowed_charges():
            # assume SU3xSU2xU1
            result = {"score": 1, 
                      "error_var": [], 
                      "good_var": [f"fields.{self.id}.reps.g2", f"fields.{self.id}.reps.g1"], 
                      "message": "Passed", 
                      "max_score": 1,
                      }
            
            def _compute_allowed_charges():
                if self.reps["g2"] == "singlet":
                    T3 = [0]
                elif self.reps["g2"] == "fnd":
                    T3 = [1/2, -1/2]
                elif self.reps["g2"] == "adj":
                    T3 = [1, 0, -1]
                Y = self.reps["g1"]/6
                Q = np.array(T3) + Y
                Q = Q * 3            # times 3 to get integer charges
                self.allowed_Q = Q
            try:
                _compute_allowed_charges()
            except Exception as e:
                result.update({"score": 0, 
                               "error_var": [f"fields.{self.id}.reps.g2", f"fields.{self.id}.reps.g1"], 
                               "good_var": [], 
                               "message": f"Error: {e}"
                               })
                return result
            return result
        
        # check if there are duplicate particles
        def _deplicate_particles():
            result = {"score": 1, 
                      "error_var": [], 
                      "good_var": [f"fields.{self.id}.particles"], 
                      "message": "Passed", 
                      "max_score": 1,
                      }
            if len(self.particles) != len(set(self.particles)):
                result.update({"score": 0, 
                               "error_var": [f"fields.{self.id}.particles"], 
                               "good_var": [], 
                               "message": f"Duplicate particles found"
                               })
                return result
            return result

        # check if the number of particles is consistent with the dim and gen
        def _particle_numbers():
            good_var = [f"fields.{self.id}.particles", f"fields.{self.id}.dim", f"fields.{self.id}.gen"]
            result = {"score": 1, 
                      "error_var": [], 
                      "good_var": good_var, 
                      "message": "Passed", 
                      "max_score": 1,
                      }
            if len(self.particles) != self.dim * self.gen:
                result.update({"score": 0, 
                               "error_var": good_var, 
                               "good_var": [], 
                               "message": f"Number of particles ({len(self.particles)}) does not match dim ({self.dim}) * gen ({self.gen})"
                               })
                return result
            return result
        

        # check if the particles are consistent with the type
        def _particle_types():
            result = {"score": 1, 
                      "error_var": [], 
                      "good_var": [f"fields.{self.id}.particles"], 
                      "message": "Passed", 
                      "max_score": 1,
                      }
            # Optimized: use generator expression and avoid unnecessary loop after first mismatch
            mismatched_particle = [p for p in self.particles if p.type != self.type]

            if mismatched_particle:
                error_var = [f"fields.{self.id}.particles"] + [f"particles.{p.id}.type" for p in mismatched_particle]
                good_var = [f"particles.{p.id}.type" for p in self.particles if p.type == self.type]
                message = f"{mismatched_particle[0].name} is a {mismatched_particle[0].type} particle, but this field is a {self.type} field."
                result.update({"score": 0, 
                               "error_var": error_var, 
                               "good_var": good_var, 
                               "message": message
                               })
                return result
            return result
        
        # check if self-conjugate is consistent with the charges
        def _particle_charges():
            if self.type == "fermion":
                good_var = [f"particles.{p.fermion.id}.charge" for p in self.particles] + [f"fields.{self.id}.self_conjugate"]
            else:
                good_var = [f"particles.{p.id}.charge" for p in self.particles] + [f"fields.{self.id}.self_conjugate"]
            result = {"score": 1, 
                      "error_var": [], 
                      "good_var": good_var, 
                      "message": "Passed", 
                      "max_score": 1,
                      }
            error_var = []
            if self.self_conjugate:
                if self.type == "fermion":
                    error_var = [f"particles.{p.fermion.id}.charge" for p in self.particles if p.fermion.charge != 0]
                else:
                    error_var = [f"particles.{p.id}.charge" for p in self.particles if p.charge != 0]
            if error_var:
                if self.type == "fermion":
                    good_var = [f"particles.{p.fermion.id}.charge" for p in self.particles if p.fermion.charge == 0]
                else:
                    good_var = [f"particles.{p.id}.charge" for p in self.particles if p.charge == 0]
                message = f"this field is self-conjugate, but the particles have non-zero charges."
                result.update({"score": 0, 
                               "error_var": error_var, 
                               "good_var": good_var, 
                               "message": message
                               })
            return result
        
        # check if all particles pass all checks
        def _all_particle_pass():
            if self.type == "fermion":
                good_var = [f"particles.{p.fermion.id}.charge" for p in self.particles] + [f"fields.{self.id}.self_conjugate"]
            else:
                good_var = [f"particles.{p.id}.charge" for p in self.particles] + [f"fields.{self.id}.self_conjugate"]
            result = {"score": 1, 
                      "error_var": [], 
                      "good_var": good_var, 
                      "message": "Passed", 
                      "max_score": 1,
                      }
            if self.type == "fermion":

                error_var = [f"particles.{p.fermion.id}" for p in self.particles if not p.fermion.pass_all_checks()]
                if error_var:
                    good_var = [f"particles.{p.fermion.id}" for p in self.particles if p.fermion.pass_all_checks()]
                    message = f"Some particles do NOT pass ALL checks"
                    result.update({"score": 0, 
                                   "error_var": error_var, 
                                   "good_var": good_var, 
                                   "message": message
                                   })
                    return result
            else: 
                error_var = [f"particles.{p.id}" for p in self.particles if not p.pass_all_checks()]
                if error_var:
                    good_var = [f"particles.{p.id}" for p in self.particles if p.pass_all_checks()]
                    message = f"Some particles do NOT pass ALL checks"
                    result.update({"score": 0, 
                                   "error_var": error_var, 
                                   "good_var": good_var, 
                                   "message": message
                                   })
                    return result
            return result

        # Sort particles ordering
        def _sort_particles():
            if self.type == "fermion":
                good_var = [f"particles.{p.fermion.id}" for p in self.particles]
            else:
                good_var = [f"particles.{p.id}" for p in self.particles]
            result = {"score": 2, 
                      "error_var": [], 
                      "good_var": good_var, 
                      "message": "Passed", 
                      "max_score": 2,
                      }
            error_var = []

            charge_eigenvec = {q: [] for q in self.allowed_Q}

            if self.type == "fermion":
                error_var = [f"particles.{p.fermion.id}.charge" for p in self.particles if p.fermion.charge not in self.allowed_Q]
            else:
                error_var = [f"particles.{p.id}.charge" for p in self.particles if p.charge not in self.allowed_Q]
            
            if error_var:
                # More precise attribution: blame the specific reps that define allowed_Q and the first invalid charge.
                first_invalid_particle = next((p for p in self.particles if (p.fermion.charge if self.type == "fermion" else p.charge) not in self.allowed_Q), None)
                
                message = "A particle's charge is not in the allowed charges for this field."
                precise_error_var = [f"fields.{self.id}.reps.g1", f"fields.{self.id}.reps.g2"] # The root cause
                if first_invalid_particle:
                    p_charge = first_invalid_particle.fermion.charge if self.type == "fermion" else first_invalid_particle.charge
                    p_id = first_invalid_particle.fermion.id if self.type == "fermion" else first_invalid_particle.id
                    precise_error_var.append(f"particles.{p_id}.charge") # The symptom
                    message = f"Charge {p_charge/3.0} for particle '{p_id}' is not in allowed charges {np.array(self.allowed_Q)/3.0}, derived from reps."

                good_var = [f"particles.{p.id}.charge" for p in self.particles if (p.fermion.charge if self.type == "fermion" else p.charge) in self.allowed_Q]
                
                result.update({"score": 0, 
                               "error_var": precise_error_var, 
                               "good_var": good_var, 
                               "message": message
                               })
                return result
            
            for p in self.particles:
                if self.type == "fermion":
                    charge_eigenvec[p.fermion.charge].append(p)
                else:
                    charge_eigenvec[p.charge].append(p)
            
            charge_eigenval = sorted(charge_eigenvec.keys(), reverse=True)
            for _, ptcls in charge_eigenvec.items():
                ptcls.sort(key=lambda p: p.mass)

            if len(charge_eigenval) != self.dim:
                if self.type == "fermion":
                    error_var = [f"particles.{p.fermion.id}.charge" for p in self.particles]
                else:
                    error_var = [f"particles.{p.id}.charge" for p in self.particles]
                result.update({"score": 0, 
                               "error_var": error_var, 
                               "good_var": [f"fields.{self.id}.reps.g1", f"fields.{self.id}.reps.g2"], 
                               "message": f"Particles number is not consistent with the dim"
                               })
                return result
        
            self._unphy_fields = np.column_stack([charge_eigenvec[charge] for charge in charge_eigenval])
            self._phy_fields = np.array(self._unphy_fields).transpose().tolist()

            return result
    

        self.all_checks = [(_wrap(_name_check), 1),    
                           (_wrap(_type_check), 1), 
                           (_wrap(_groups_check), 1), 
                           (_wrap(_reps_check), 3), 
                           (_wrap(_dim_check), 1), 
                           (_wrap(_gen_check), 1), 
                           (_wrap(_particles_check), 1), 
                           (_wrap(_self_conjugate_check), 1), 
                           (_wrap(_sort_reps), 3),
                           (_wrap(_reps_dim_consistency), 1),
                           (_wrap(_gen_type_consistency), 1),
                           (_wrap(_allowed_charges), 1),
                           (_wrap(_deplicate_particles), 1),
                           (_wrap(_particle_numbers), 1), 
                           (_wrap(_particle_types), 1),
                           (_wrap(_particle_charges), 1),
                           (_wrap(_all_particle_pass), 1),
                           (_wrap(_sort_particles), 2)
                           ]

    def __check__(self):
        """ Input checks for the field class. """
        self.checklist = {}
        self._all_checks()
        run_checks(self.all_checks, self.checklist, skip_results = True)

    def _all_validations(self):
        """ All validations for the 'Field' class. """

        def _mass_term_check():
            result = {"score": 10, "max_score": 10, "error_var": [], "good_var": [], "message": "Passed"}
            if self.type == "fermion":
                self.is_massive = any(p.fermion.mass != 0 for p in self.particles)
            else:
                self.is_massive = any(p.mass != 0 for p in self.particles)
            
            if self.is_massive and self.mass_term == []:
                error_var = [f"fields.{self.id}"]
                if self.type == "fermion":
                    error_var.extend([f"particles.{p.fermion.id}.mass" for p in self.particles if p.fermion.mass > 0])
                else:
                    error_var.extend([f"particles.{p.id}.mass" for p in self.particles if p.mass > 0])
                result.update({"score": 0, "error_var": error_var, "message": "this field is massive, but no mass term is defined"})
            elif not self.is_massive and self.mass_term != []:
                error_var = [f"interactions.{m_term}" for m_term in self.mass_term]
                result.update({"score": 0, "error_var": error_var, "message": "this field is massless, but a mass term is defined"})
            else:
                # On success, credit the particles and any relevant interactions
                good_var = [f"fields.{self.id}.particles"]
                if self.mass_term:
                    good_var.extend([f"interactions.{m_term}" for m_term in self.mass_term])
                result["good_var"] = good_var
            return result
        
        def _finalize_result(result: dict) -> dict:
            good_var = result.get('good_var', []) or []
            error_var = result.get('error_var', []) or []
            mattered = list({*good_var, *error_var})
            result['mattered_vars'] = mattered
            # If neither good nor error vars were specified, treat as block-level
            result['level'] = 'field' if mattered else 'block'
            return result
        
        # Preserve name for validation wrapper
        def _mass_term_check_wrapped():
            return _finalize_result(_mass_term_check())
        _mass_term_check_wrapped.__name__ = '_mass_term_check'

        self.all_validations = [(_mass_term_check_wrapped, 10)]

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
#                            Fermion Field
# ====================================================================
class FermionField(Field):
    """
    Class for fermion fields.
    groups: list of str
    reps: list of int
    dim: int > 0
    gen: int > 0
    particles: list of Particle
    self_conjugate: bool
    chirality: str
    """
    def __init__(self, id, name, groups, reps, dim, gen, particles, self_conjugate, chirality):
        self.chirality = chirality
        self._unphy_fields = None
        self._phy_fields = None
        self.color = 1
        super().__init__(id, name, "fermion", groups, reps, dim, gen, particles, self_conjugate)
        

    def __dict__(self):
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "reps": self.reps,
            "groups": self.groups,
            "dim": self.dim,
            "gen": self.gen,
            "particles": self.particles,
            "self_conjugate": self.self_conjugate,
            "chirality": self.chirality
        }

    def _all_checks(self):
        super()._all_checks()

        def _chirality_check():
            result = {"score": 1, 
                      "error_var": [], 
                      "good_var": [], 
                      "message": "Passed", 
                      "max_score": 1,
                      }
            if self.chirality not in ["left", "right"]:
                result.update({"score": 0, 
                               "error_var": [f"fields.{self.id}.chirality"], 
                               "message": "chirality must be left or right"
                               })
            return result
        
        # ------------------------------------------------------------------
        
        # Assign colors to the particles
        def _assign_colors():
            result = {"score": 1, 
                      "error_var": [], 
                      "good_var": [], 
                      "message": "Passed", 
                      "max_score": 1,
                      }
            error_var = []
            good_var = []
            error_msg = []
            for p in self.particles:
                try:
                    if self.color is not None:
                        p.fermion.color = self.color
                        good_var.append(f"particles.{p.id}")
                except Exception as e:
                    error_var.append(f"particles.{p.id}")
                    error_msg.append(f"Error: {e}")
            if error_var:
                result.update({"score": 0, 
                               "error_var": error_var, 
                               "good_var": good_var, 
                               "message": error_msg
                               })
            return result

        self.all_checks.extend([(_chirality_check, 1),
                                (_assign_colors, 1)
                                ])

        
    # ------------------------------------------------------------------
    #                        Write Model File
    # ------------------------------------------------------------------

    def _class_name(self, idx):
            charge = self._phy_fields[idx][0].charge
            color = self.color
            class_name, description = name.fermion_field_name(charge, color)
            if self.chirality == "left":
                return f"{class_name}L", description
            elif self.chirality == "right":
                return f"{class_name}R", description

    @property
    def phy_field_info(self):
        # Check if the field passes all checks
        if not self.pass_all_checks():
            return None
        
        def _class_members(idx):
            member_names = [p.fermion.name for p in self._phy_fields[idx]]
            return member_names

        def _PDG(idx):
            PDG_list = [p.fermion.pdg_id for p in self._phy_fields[idx]]
            return PDG_list

        def _mass(idx):
            mass = [p.fermion.mass for p in self._phy_fields[idx]]
            return mass

        def _width(idx):
            particle_list = self._phy_fields[idx]
            width = [p.fermion.width for p in particle_list]
            if all(w == "Automatic" for w in width):
                width = "Automatic"
            return width

        def _charge(idx):
            charge = self._phy_fields[idx][0].charge
            charge = Fraction(charge, 3)
            return charge

        phy_fields = {}
        for i in range(self.dim):
            class_name = self._class_name(i)
            phy_field = {
                "Description": class_name[1],
                "LaTeX": class_name[0],
                "PDG": _PDG(i),
                "Mass": _mass(i),
                "OutputName": class_name[0][:-1],
                "ElectricCharge": _charge(i),
                "Width": _width(i),
                # additional info
                "chirality": self.chirality,
                "class_members": _class_members(i),
                "location": i
            }
            phy_fields[class_name[0]] = phy_field
        return phy_fields
    
    @property
    def unphy_field_info(self):
        if not self.pass_all_checks():
            print(f"Error: {self.name} has failed the checks")
            return None

        if self.chirality == "left":
            element = [f"{self._class_name(i)[0]}" for i in range(self.dim)]
        elif self.chirality == "right":
            element = [f"conj[{self._class_name(i)[0]}]" for i in range(self.dim)]
            if self.full_reps["g3"] == 1:
                pass
            else:
                self.full_reps["g3"] = - self.full_reps["g3"]

            self.full_reps["g1"] = - self.full_reps["g1"]
    
        if self.dim == 1:
            multiplet = element[0]
        else:
            multiplet = "{" + ", ".join(element) + "}"
        reps_info = ", ".join([str(value) for value in self.full_reps.values()])

        return f"{self.name}, {self.gen}, {multiplet}, {reps_info}"
        

# ====================================================================
#                            Scalar Field
# ====================================================================
class ScalarField(Field):
    def __init__(self, id, name, type, groups, reps, dim, gen, particles, self_conjugate):
        self.chirality = None
        super().__init__(id, name, type, groups, reps, dim, gen, particles, self_conjugate)
        self.potential = None
        self.vev = None
        self.get_vev = False
        self.potential = []
        

    def _all_validations(self):
        """ All validations for the 'Field' class. """
        super()._all_validations()
        
        def _potential_term_check():
            result = {"score": 1, "max_score": 1, "error_var": [], "good_var": [], "message": "Passed"}
            if self.potential == []:
                result.update({"score": 0, "error_var": [f"fields.{self.id}"], "message": "scalar field must have a potential term"})
            else:
                # FIX: Ensure good_var is populated on success
                result.update({"score": 1, "good_var": [f"interactions.{itr}" for itr in self.potential], "message": "Passed"})
            return result
        
        self.all_validations.extend([(_potential_term_check, 1)])

    def phy_field_info(self):
        pass

    @property
    def unphy_field_info(self):
        reps_info = ", ".join([str(value) for value in self.full_reps.values()])
        if self.dim == 1:
            multiplet = f"{self.particles[0].name}"
        else:
            multiplet = "{" + ", ".join([f"{p.name}" for p in self.particles]) + "}"
        
        return f"{self.name}, {self.gen}, {multiplet}, {reps_info}"



# ====================================================================
#                            Vector Fields
# ====================================================================
class VectorField(Field):
    def __init__(self, id, name, groups, reps, dim, gen, particles, self_conjugate):
        super().__init__(id, name, "vector", groups, reps, dim, gen, particles, self_conjugate)
        self.__check__()

    def __check__(self):
        super().__check__()

# ------------------------------------------------------------------
if __name__ == "__main__":
    pass