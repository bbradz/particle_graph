import naming
from fractions import Fraction

class UnphysicalMultiplet:  
    """
    Base class for multiplets representing gauge group representations.
    
    Attributes:
        id: Unique identifier
        label: LaTeX label
        dim: Dimension of the multiplet
        gen: Number of generations
        reps: List of representations under the gauge groups
        gauge_groups: List of gauge groups
        particles: List of particle objects contained in the multiplet
        QuantumNumber: Dictionary of quantum numbers
    """
    def __init__(self, id, label, dim, gen, reps, gauge_groups, particles=[], QuantumNumber=None):
        self.id = id
        self.label = label
        self.dim = dim
        self.gen = gen
        self.reps = reps
        self.gauge_groups = gauge_groups
        self.sort_reps()
        self.particles = particles
        self.QuantumNumber = QuantumNumber
        self.physical = False
        self.FlavorIndex = naming.number_to_english(gen) + "Gen"

        # Apply multiplet quantum numbers to each particle
        if self.full_reps['SU3C']['confinement'] and self.full_reps['SU3C']["rep"] == "fnd":
            self.color = True
        else:
            self.color = False

        for gen_particles in self.particles:
            for particle in gen_particles:
                if QuantumNumber is not None:
                    if particle.QuantumNumber is None:
                        particle.QuantumNumber = QuantumNumber
                    else:
                        # Merge quantum numbers if particle already has some
                        for key, value in QuantumNumber.items():
                            particle.QuantumNumber[key] = value

                if self.full_reps['SU3C']['confinement'] and self.full_reps['SU3C']["rep"] == "fnd":
                    particle.color = True
                else:
                    particle.color = False

        self.num_groups = len(self.gauge_groups)
        self.num_particles = len(self.particles)
        self.particle_type = self.get_particle_type()
        self.particle_names = self.get_particle_info("name")
        self.particle_labels = self.get_particle_info("label")
        self.particle_ids = self.get_particle_info("id")
        self.particle_masses = self.get_particle_info("mass")
        self.group_names = self.get_group_names()
        self.self_conjugate = self.is_self_conjugate()

        self.indices = self.get_Indices()

        #self.sort_reps() # sort the reps
        self._validate() # validate 

    def __str__(self):
        return f"Multiplet(name={self.name}, dim={self.dim}, gen={self.gen}, reps={self.reps}, gauge_groups={self.group_names}, type = {self.particle_type}, particles={self.particle_labels})"
    
    def get_group_names(self):
        """Return a list of gauge group names."""
        return [group.name for group in self.gauge_groups]
    
    def get_particle_info(self, info_type):
        """
        Extract specific information from particles.
        
        Args:
            info_type: Type of information to extract ("name", "label", "id", or "mass")
            
        Returns:
            List of lists containing the requested information for each generation
        """
        particle_info = []
        for gen in range(self.gen):
            one_gen_of_particles = []
            for particle in self.particles[gen]:
                if info_type == "name":
                    one_gen_of_particles.append(particle.name)
                elif info_type == "label":
                    one_gen_of_particles.append(particle.label)
                elif info_type == "id":
                    one_gen_of_particles.append(particle.id)
                elif info_type == "mass":
                    one_gen_of_particles.append(particle.mass)
                else:
                    raise ValueError(f"Invalid info type: {info_type}")
            particle_info.append(one_gen_of_particles)
        return particle_info
    
    def get_particle_type(self):
        """
        Get the particle type and verify all particles have the same type.
        
        Returns:
            The common particle type
        """
        type = self.particles[0][0].type
        for gen in range(self.gen):
            for particle in self.particles[gen]:
                if particle.type != type:
                    raise ValueError("All particles must have the same type in the multiplet.")
        return type
    
    def is_self_conjugate(self):
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

    def sort_reps(self):
        """Sort the reps of the multiplet."""
        assert len(self.reps) == len(self.gauge_groups)
        
        self.full_reps = {}
        for idx, group in enumerate(self.gauge_groups):
            rep_info = {
                'rep': self.reps[idx],
                'label': group.label,
                'type': group.type,
                'dim': group.dim,
                'abelian': group.abelian,
                'confinement': group.confinement
            }

            self.full_reps[group.name] = rep_info

    def write_QuantumNumbers(self):
        """Write QuantumNumbers for the unphysical multiplet."""
        qnumbers = {}
        for key, value in self.full_reps.items():
            if value['abelian']:
                charge = Fraction(value['rep'], 6)
                qnumbers[value['label']] = str(charge)
        qnumbers = str(qnumbers).replace(":", " ->").replace("'", "")
        return qnumbers    

    def get_Indices(self):
        """Write Indices for the unphysical multiplet."""
        
        indices = []
        for key, value in self.full_reps.items():
            if not value['abelian']:
                isColor = value['confinement'] and value['rep'] == "fnd" and value['type'] == "SU3"
                if value['rep'] != "singlet":
                    if isColor:
                        indices.append("Index{Colour}")
                    elif self.dim > 1:
                        idx_name = value['type'] + naming.number_to_plural(self.dim)[0].upper()
                        indices.append(f"Index[{idx_name}]")

        indices.append(f"Index[{self.FlavorIndex}]")
        indices = list(set(indices))

        return indices
        
    def write_Indices(self):
        """Write Indices for the unphysical multiplet."""
        indices = str(self.indices).replace("'", "").replace("[", "{").replace("]", "}")
        return indices
    
    def write_Definition(self):
        """Write FeynRules definition for the multiplet."""
        
        return "{}"

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
            "particles": self.particle_ids,
            "QuantumNumber": self.QuantumNumber
        }

    def _validate(self):
        """Validate multiplet properties."""
        pass





class PhysicalMultiplet:
    """
    Class for physical multiplets that represent observable particles.
    
    Attributes:
        id: Unique identifier
        name: Name of the multiplet
        label: LaTeX label
        dim: Dimension of the multiplet
        gen: Number of generations
        particles: List of particle objects
        QuantumNumber: Dictionary of quantum numbers
    """
    def __init__(self, id, name, label, dim, gen, particles=[], QuantumNumber=None):
        self.id = id
        self.name = name
        self.label = label
        self.dim = dim
        self.gen = gen
        self.particles = particles
        self.QuantumNumber = QuantumNumber
        self.physical = True
        self.self_conjugate = self.is_self_conjugate()

        self.FlavorIndex = naming.number_to_english(gen) + "Gen"
        self.particle_names = self.get_particle_info("name")
        self.particle_labels = self.get_particle_info("label")
        self.particle_ids = self.get_particle_info("id")
        self.particle_masses = self.get_particle_info("mass")

        self.indices = []

    def __str__(self):
        return f"PhysicalMultiplet(id={self.id}, name={self.name}, label={self.label}, dim={self.dim}, gen={self.gen}, particles={self.particle_labels}, QuantumNumber={self.QuantumNumber})"

    def get_particle_info(self, info_type):
        """
        Extract specific information from particles.
        
        Args:
            info_type: Type of information to extract ("name", "label", "id", or "mass")
            
        Returns:
            List containing the requested information for each particle
        """
        particle_info = []
        for particle in self.particles:
            if info_type == "name":
                particle_info.append(particle.name)
            elif info_type == "label":
                particle_info.append(particle.label)
            elif info_type == "id":
                particle_info.append(particle.id)
            elif info_type == "mass":
                particle_info.append(particle.mass)
            else:
                raise ValueError(f"Invalid info type: {info_type}")
        return particle_info

    def is_self_conjugate(self):
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

    def write_ClassMembers(self):
        """Format particle labels for FeynRules ClassMembers."""
        return str(self.particle_labels).replace("'", "").replace("[", "{").replace("]", "}")

    def write_ParticleName(self):
        """Format particle labels for FeynRules ParticleName."""
        return str(self.particle_labels).replace("'", "").replace("[", "{").replace("]", "}")

    def write_Mass(self):
        """Format particle masses for FeynRules Mass."""
        Mass = ["M"+str(self.label)]
        for particle in self.particles:
            mass_label = "M" + particle.label.upper()
            Mass.append((mass_label, particle.mass))
        Mass = str(Mass).replace("'", "")
        Mass = Mass.replace("[", "{").replace("]", "}")
        Mass = Mass.replace("(", "{").replace(")", "}")
        return Mass

    def write_FullName(self):
        """Format particle names for FeynRules FullName."""
        return str(self.particle_names).replace("[", "{").replace("]", "}")
    
    def write_AntiParticleName(self):
        """Format antiparticle names for FeynRules AntiParticleName."""
        AntiParticleName = [name + "~" for name in self.particle_labels]
        return str(AntiParticleName).replace("[", "{").replace("]", "}")

    def write_QuantumNumbers(self):
        """Format quantum numbers for FeynRules."""
        for key, value in self.QuantumNumber.items():
            self.QuantumNumber[key] = str(Fraction(value, 3))

        return str(self.QuantumNumber).replace(":", "->").replace("'", "")

    def write_Indices(self):
        """Format indices for FeynRules Indices."""
        idx_list = []
        if self.FlavorIndex:
            idx_list.append(f"Index[{self.FlavorIndex}]")
        idx_list = str(idx_list).replace("'", "").replace("[", "{").replace("]", "}")
        return idx_list

    def write_PropagatorLabel(self):
        """Format propagator label for FeynRules PropagatorLabel."""
        propagator_label = [self.label, *self.particle_labels]
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
            "label": self.label,
            "dim": self.dim,
            "gen": self.gen,
            "particles": self.particle_ids,
            "QuantumNumber": self.QuantumNumber
        }


class HiggsSector:
    """
    Higgs sector of the Standard Model.
    
    Attributes:
        Higgs_fields: List of Higgs fields
        vev: Vacuum expectation value
        num_Higgs: Number of Higgs fields
        Higgs_names: Names of Higgs fields
    """
    def __init__(self, Higgs_fields, vev):
        self.Higgs_fields = Higgs_fields
        self.vev = vev
        self.num_Higgs = len(self.Higgs_fields)
        self.Higgs_names = self.get_Higgs_names()
    


