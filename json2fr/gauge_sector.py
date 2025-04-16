class GaugeSector:
    """
    Gauge sector of the SM
    """
    def __init__(self, 
                 gauge_groups, 
                 higgs, 
                 vev):
        
        self.gauge_groups = gauge_groups
        self.higgs = higgs
        self.vev = vev
        self.num_groups = len(self.gauge_groups)

        self.OrderHierarchy = []

    def symmetry_breaking(self):
        """
        Symmetry breaking of the gauge groups
        """
        
        
    def __str__(self):
        return f"Gauge Sector: {', '.join([group.group_name for group in self.gauge_groups.values()])}"
