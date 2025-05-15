TOKEN_MAP = {
    # control tokens
    0: "<PAD>",
    1: "<START>",
    2: "<END>",

    # Entity creation tokens
    10: "CREATE_GAUGE_GROUP",
    11: "CREATE_PARTICLE",
    12: "CREATE_FIELD",
    13: "CREATE_INTERACTION",

    # Entity type tokens
    20: "GAUGE_GROUP",
    21: "PARTICLE",
    22: "FIELD",
    23: "INTERACTION",

    # Gauge group tokens
    30: "U",
    31: "SU",
    32: "SO",
    33: "G",
    34: "GL",

    # Particle tokens
    40: "RealScalar",
    41: "ComplexScalar",
    42: "Fermion",
    43: "VectorBoson",
    
    # Field tokens
    50: "SCALAR_FIELD",
    51: "FERMION_FIELD",
    52: "VECTOR_FIELD",
    53: "HIGGS_FIELD",
    
    # Interaction tokens
    60: "YUKAWA",
    61: "COVARIANT_DERIVATIVE",
    62: "SCALAR_MASS",
    63: "SCALAR4",
    64: "GAUGE_KINETIC",
    
    # Representation tokens
    70: "FUNDAMENTAL",
    71: "ADJOINT",
    72: "SYMMETRIC",
    73: "ANTISYMMETRIC",
    74: "SINGLET",
    
    # Chirality tokens
    80: "LEFT",
    81: "RIGHT",
    
    # Quantum number tokens
    90: "CHARGE",
    91: "LEPTON_NUMBER",
    92: "BARYON_NUMBER",
    
    # Multiplet tokens
    100: "MULTIPLET",
    101: "GENERATION",
    102: "DIMENSION",
    
}
