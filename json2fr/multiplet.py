import naming
from fractions import Fraction
import numpy as np
class UnphysicalMultiplet:  
    """
    Base class for multiplets representing gauge group representations.
    
    Attributes:
        id: Unique identifier
        name: 
        dim: Dimension of the multiplet
        gen: Number of generations
        reps: List of representations under the gauge groups
        gauge_groups: List of gauge groups
        particles: List of particle objects contained in the multiplet
        QuantumNumber: Dictionary of quantum numbers
    """
    def __init__(self, id, name, dim, gen, reps, gauge_groups, particles=[], QuantumNumber=None):
        self.id = id
        self.name = name
        self.dim = dim
        self.gen = gen
        self.reps = reps
        self.gauge_groups = gauge_groups
        self.particles = particles
        self.QuantumNumber = QuantumNumber
        self.physical = False

        # Apply multiplet quantum numbers to each particle
        for g_idx, gen_particles in enumerate(self.particles):
            for p_idx, particle in enumerate(gen_particles):
                if particle == 0.0:
                    raise ValueError(f"No particle is assigned to {self.name} at generation {g_idx} and position {p_idx}")

                if self.color:
                    particle.color = True
                else:
                    particle.color = False
                    
                if self.QuantumNumber is not None:
                    if particle.QuantumNumber is None:
                        particle.QuantumNumber = self.QuantumNumber
                    else:
                        # Merge quantum numbers if particle already has some
                        particle.QuantumNumber.update(self.QuantumNumber)

        self.particle_full_names = self.get_particle_info("full_name")
        self.particle_names = self.get_particle_info("name")

    @property
    def GenerationIndex(self):
        if self.gen > 1:
            return naming.number_to_english(self.gen) + "Gen"
        else:
            return None
        
    @property
    def color(self):
        return self.full_reps['SU3C']['confinement'] and self.full_reps['SU3C']["rep"] == "fnd"
    
    @property
    def full_reps(self):
        """Sort the reps of the multiplet."""
        assert len(self.reps) == len(self.gauge_groups)

        _full_reps = {}
        for key, group in self.gauge_groups.items():
            rep_info = {
                'rep': self.reps[key],
                'id': group.id,
                'name': group.name,
                'type': group.group_type,
                'dim': group.dim,
                'abelian': group.abelian,
                'confinement': group.confinement
            }   
            _full_reps[group.name] = rep_info
        return _full_reps
    
    @property
    def particle_type(self):
        type = self.particles[0][0].type
        for gen in range(self.gen):
            for particle in self.particles[gen]:
                if particle.type != type:
                    raise ValueError("All particles must have the same type in the multiplet.")
        return type

    @property
    def indices(self):
        """Write Indices for the unphysical multiplet."""
        
        indices = []
        for key, value in self.full_reps.items():
            if not value['abelian']:
                color = value['confinement'] and value['rep'] == "fnd" and value['type'] == "SU" and value['dim'] == 3
                if value['rep'] != "singlet":
                    if color:
                        indices.append(("NoUnfold", "Colour", 3, value['id']))
                    elif self.dim > 1:
                        idx_name = value['type'] + str(value['dim']) + naming.number_to_plural(self.dim)[0].upper()
                        indices.append(("Unfold", idx_name, self.dim, value['id']))

        if self.GenerationIndex:
            indices.append(("fold", self.GenerationIndex, self.gen, None))
        indices = list(set(indices))

        return indices

    def __str__(self):
        return f"Multiplet(name={self.name}, dim={self.dim}, gen={self.gen}, reps={self.reps}, type = {self.particle_type}, particles={self.particle_names})"
    
    def get_particle_info(self, info_type):
        """
        Extract specific information from particles.
        """
        particle_info = []
        for gen in range(self.gen):
            one_gen_of_particles = []
            for particle in self.particles[gen]:
                if info_type == "name":
                    one_gen_of_particles.append(particle.name)
                elif info_type == "full_name":
                    one_gen_of_particles.append(particle.full_name)
                elif info_type == "id":
                    one_gen_of_particles.append(particle.id)
                elif info_type == "mass":
                    one_gen_of_particles.append(particle.mass)
                else:
                    raise ValueError(f"Invalid info type: {info_type}")
            particle_info.append(one_gen_of_particles)
        return particle_info
    
    @property
    def self_conjugate(self):
        """
        Check if all particles in the multiplet are self-conjugate.
        
        Returns:
            Boolean indicating if the multiplet is self-conjugate
        """
        self_conjugate = self.particles[0][0].self_conjugate
        for gen in range(self.gen):
            for particle in self.particles[gen]:
                if particle.self_conjugate != self_conjugate:
                    raise ValueError("All particles must have the same self-conjugate property in the multiplet.")
        return self_conjugate

    def write_QuantumNumbers(self):
        """Write QuantumNumbers for the unphysical multiplet."""
        qnumbers = {}
        for key, value in self.full_reps.items():
            if value['abelian']:
                charge = Fraction(value['rep'], 6)
                qnumbers[value['name']] = str(charge)
        qnumbers = str(qnumbers).replace(":", " ->").replace("'", "")
        return qnumbers    
        
    def write_Indices(self):
        """Write Indices for the unphysical multiplet."""
        indices = str(np.array(self.indices)[:,1]).replace("'", "").replace("[", "{").replace("]", "}").replace(" ", ", ")
        return indices
    
    def write_Definition(self):
        """Write FeynRules definition for the multiplet."""
        definition = []
        if self.dim > 1:
            for i in range(self.dim):
                definition.append(f"{self.name}[sp1_, {i}, ff_, cc] :> Module[{{sp2}}, {self.name}[sp2, cc, ff]]")
        else:
            definition.append(f"{self.name}[sp1_, ff_, cc] :> Module[{{sp2}}, Proj{self.name}[sp2, cc, ff]]")

        return str(definition).replace("'", "")
    
    def to_dict(self):
        """
        Convert multiplet to dictionary representation.
        
        Returns:
            Dictionary with multiplet attributes
        """
        return {
            "id": self.id,
            "name": self.name,
            "dim": self.dim,
            "gen": self.gen,
            "reps": self.reps,
            "gauge_groups": self.group_names,
            "QuantumNumber": self.QuantumNumber
        }




class PhysicalMultiplet:
    """
    Class for physical multiplets that represent observable particles.
    
    Attributes:
        id: Unique identifier
        full name: 
        name: 
        dim: Dimension of the multiplet
        gen: Number of generations
        particles: List of particle objects
        QuantumNumber: Dictionary of quantum numbers
    """
    def __init__(self, id, full_name, name, dim, gen, particles=[], QuantumNumber=None):
        self.id = id
        self.full_name = full_name
        self.name = name
        self.dim = dim
        self.gen = gen
        self.particles = particles
        self.QuantumNumber = QuantumNumber
        self.physical = True
        self.GenerationIndex = naming.number_to_english(gen) + "Gen"
        self.particle_names = self.get_particle_info("name")
        self.particle_full_names = self.get_particle_info("full_name")

        self.indices = []

    @property
    def self_conjugate(self):
        """
        Check if all particles in the multiplet are self-conjugate.
        
        Returns:
            Boolean indicating if the multiplet is self-conjugate
        """
        self_conjugate = self.particles[0].self_conjugate
        for particle in self.particles:
            if particle.self_conjugate != self_conjugate:
                raise ValueError("All particles must have the same self-conjugate property in the multiplet.")
        return self_conjugate


    def __str__(self):
        return f"PhysicalMultiplet(id={self.id}, full_name={self.full_name}, name={self.name}, dim={self.dim}, gen={self.gen}, particles={self.particle_names}, QuantumNumber={self.QuantumNumber})"

    def get_particle_info(self, info_type):
        """
        Extract specific information from particles.
        
        Args:
            info_type: Type of information to extract ("name", "full_name", "id", or "mass")
            
        Returns:
            List containing the requested information for each particle
        """
        particle_info = []
        for particle in self.particles:
            if info_type == "full_name":
                particle_info.append(particle.full_name)
            elif info_type == "name":
                particle_info.append(particle.name)
            elif info_type == "id":
                particle_info.append(particle.id)
            elif info_type == "mass":
                particle_info.append(particle.mass)
            else:
                raise ValueError(f"Invalid info type: {info_type}")
        return particle_info



    def write_ClassMembers(self):
        """Format particle names for FeynRules ClassMembers."""
        return str(self.particle_names).replace("'", "").replace("[", "{").replace("]", "}")

    def write_ParticleName(self):
        """Format particle names for FeynRules ParticleName."""
        return str(self.particle_names).replace("'", "").replace("[", "{").replace("]", "}")

    def write_Mass(self):
        """Format particle masses for FeynRules Mass."""
        Mass = ["M"+str(self.name)]
        for particle in self.particles:
            mass_name = "M" + particle.name.upper()
            Mass.append((mass_name, particle.mass))
        Mass = str(Mass).replace("'", "")
        Mass = Mass.replace("[", "{").replace("]", "}")
        Mass = Mass.replace("(", "{").replace(")", "}")
        return Mass

    def write_FullName(self):
        """Format particle names for FeynRules FullName."""
        return str(self.particle_names).replace("[", "{").replace("]", "}")
    
    def write_AntiParticleName(self):
        """Format antiparticle names for FeynRules AntiParticleName."""
        AntiParticleName = [name + "~" for name in self.particle_names]
        return str(AntiParticleName).replace("[", "{").replace("]", "}")

    def write_QuantumNumbers(self):
        """Format quantum numbers for FeynRules."""
        for key, value in self.QuantumNumber.items():
            self.QuantumNumber[key] = str(Fraction(value, 3))

        return str(self.QuantumNumber).replace(":", "->").replace("'", "")

    def write_Indices(self):
        """Format indices for FeynRules Indices."""
        idx_list = []
        if self.GenerationIndex:
            idx_list.append(f"Index[{self.GenerationIndex}]")
        idx_list = str(idx_list).replace("'", "").replace("[", "{").replace("]", "}")
        return idx_list

    def write_PropagatorLabel(self):
        """Format propagator label for FeynRules PropagatorLabel."""
        propagator_label = [self.name, *self.particle_names]
        return str(propagator_label).replace("[", "{").replace("]", "}")

    def to_dict(self):
        """
        Convert multiplet to dictionary representation.
        
        Returns:
            Dictionary with multiplet attributes
        """
        return {
            "id": self.id,
            "name": self.name,
            "full_name": self.full_name,
            "dim": self.dim,
            "gen": self.gen,
            "QuantumNumber": self.QuantumNumber
        }



class HiggsSector:
    """
    Higgs sector of the Standard Model.
    """
    def __init__(self, Higgs_fields, vev):
        self.Higgs_fields = Higgs_fields
        self.vev = vev

    