import numpy as np
from multiplet import UnphysicalMultiplet, PhysicalMultiplet
from particles import VectorBoson

class HiggsGaugeSector:
    """
    Higgs sector of the model
    """
    def __init__(self, 
                 unphy_scalars, 
                 gauge_groups,
                 vev):
        
        self.unphy_scalars = unphy_scalars
        self.gauge_groups = gauge_groups
        self.vev = vev

        self.vacuum = np.array(self.vev['vacuum'])
        self.vev_values = self.vev['value']
        self._validate()
        self._before_SSB()
        self._fake_SSB_main()
        
    def _validate(self):
        """
        Validate the Higgs sector
        """
        if len(self.vev['vacuum']) != self.unphy_scalars.dim:
            raise ValueError("vev and Higgs must have the same dimension")
    
    @property
    def broken_groups(self):
        """
        Symmetry breaking pattern
        return broken and residual groups
        """

        broken_groups = []
        for key, value in self.unphy_scalars.full_reps.items():
            if value['abelian']:
                if value['rep'] != 0:                
                    broken_groups.append(key)
            elif value['rep'] != "singlet":
                broken_groups.append(key)

        return broken_groups

    def _before_SSB(self):
        """
        Before symmetry breaking
        """
        
        initial_reps = {key: "singlet" for key, group in self.gauge_groups.items() if not group.abelian}
        initial_reps.update({key: 0 for key, group in self.gauge_groups.items() if group.abelian})
        self.gauge_bosons = {}

        for key, group in self.gauge_groups.items():
            reps = initial_reps.copy()
            if not group.abelian:
                reps[key] = "adj"

            boson_list = []
            for i in range(group.adjoint_rep):
                new_vector = VectorBoson(id = f"{key}_{i + 1}", 
                                         full_name = f"{group.boson}_{i + 1} Boson",
                                         name = f"{group.boson}_{i + 1}",
                                         mass = 0,
                                         group = group,
                                         self_conjugate = True,
                                         QuantumNumber = {"Q": 0}
                                         )
                boson_list.append(new_vector)

            new_gauge_field = UnphysicalMultiplet(id = key, 
                                                  name = f"{group.boson} Field",
                                                  dim = group.adjoint_rep,
                                                  gen = 1,
                                                  reps = reps,
                                                  gauge_groups = self.gauge_groups,
                                                  particles = [boson_list],
                                                  QuantumNumber= {"LeptonNumber": 0, "BaryonNumber": 0}
                                                  )
            if key in self.broken_groups:
                new_gauge_field.physical = False
            else:
                new_gauge_field.physical = True
            self.gauge_bosons[key] = new_gauge_field

    def _fake_SSB_main(self):
        """
        Symmetry breaking of the Higgs sector
        """
        self.nonzero_idx = np.nonzero(self.vacuum)[0]

    def write_vev(self):
        """
        Write the VEV to a FeynRules file
        """
        vev_defs = []
        for idx in range(len(self.nonzero_idx)):
            value = self.vacuum[self.nonzero_idx[idx]]
            if value == 1: 
                vev_defs.append(f"{self.unphy_scalars.name}[{self.nonzero_idx[idx] + 1}], {self.vev['name']}")
            elif value > 1:
                vev_defs.append(f"{self.unphy_scalars.name}[{self.nonzero_idx[idx] + 1}], {value} * {self.vev['name']}")
        vev_defs = " {" + str(vev_defs).replace("'", "")[1:-1] + "} "

        return vev_defs
    

    def get_phys_gauge_bosons(self):
        """
        Get the physical gauge bosons
        """
        return self.phys_gauge_bosons
    

    def get_unphys_gauge_boson(self):
        """
        Get the unphysical gauge boson
        """
        return self.unphys_gauge_boson


    def get_phys_scalars(self):
        """
        Get the physical scalars
        """
        return self.phys_scalars
    

    def get_unphys_scalars(self):
        """
        Get the unphysical scalars
        """
        return self.unphys_scalars
    
    