class Interaction:
    """
    Interaction class
    """
    def __init__(self, id, num_pts, name, mplt_types, mplt_list):
        self.id = id
        self.num_pts = num_pts
        self.name = name
        self.mplt_types = mplt_types
        self.mplt_list = mplt_list

        self.check_mplt()

    def check_mplt(self):
        """
        Check if the particles in the interaction are allowed
        """
        mplt_type_list = self.mplt_types.copy()
        for mplt in self.mplt_list:
            if mplt.type not in mplt_type_list["type"]:
                raise ValueError(f"Particle type {mplt.type} not in allowed particle types {mplt_type_list}")
            mplt_type_list.remove(mplt.type)

        if len(mplt_type_list) > 0:
            raise ValueError(f"Allowed particle types {mplt_type_list} not found in interaction {self.name}")

DC_ptcl_types = [{"type": "gauge"}, {"type": "fermion"}, {"type": "scalar"}]

class CovariantDerivative(Interaction):
    """
    Covariant derivative interaction
    """
    def __init__(self, id, name, particles):
        super().__init__(id, 2, name, DC_ptcl_types, particles)

        self.DC_particle = self.get_particle()

    def get_particle(self):
        for particle in self.particles:
            if particle.particle_types == "fermion" or particle.particle_types == "scalar":
                return particle

    def __str__(self):
        return f"Covariant Derivative"

    def to_fr(self):
        if self.DC_particle.particle_types == "fermion":
            return f"I * {self.DC_particle.name}bar.Ga[mu].DC[{self.DC_particle.name},mu]"
        elif self.DC_particle.particle_types == "scalar":
            return f"DC[{self.DC_particle.name}bar[ii], mu] DC[{self.DC_particle.name},mu]"
    
yukawa_particle_types = [{"type": "fermion", "chirality": "left"},
                        {"type": "fermion", "chirality": "right"},
                        {"type": "scalar"}]

class Yukawa(Interaction):
    """
    Yukawa interaction
    """
    def __init__(self, id, name, particles):
        super().__init__(id, 3, name, yukawa_particle_types, particles)

    def __str__(self):
        return f"{self.name}"
