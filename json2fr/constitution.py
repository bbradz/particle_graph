# Particle Constitution is a rulebook for constructing a particle physics model.

# ======================= #
# == Sec.I Gauge Group == #
# ======================= #
## gauge symmetry ==>> adjoint representation ==>> vector bosons 
### Abellian / non-Abelian
### Spontaneous Symmetry Breaking (Higgs Sector)
### masses of the vector bosons

# ====================== #
# == Sec.II Multiplet == #
# ====================== #
## boolean list of whether the multiplet gauges to which group
## e.g. {G1: True, G2: False, G3: True, ...}
## compute allowed dimensions of the multiplet
## choose dimensions 
## ==>> automatically get representations(non-abelian) and charges (abelian)
## define particle 

# ======================= #
# == Sec. III Particle == #
# ======================= #

## non-Higgs scalar
### real scalar
### complex scalar
### psuedo scalar 

## fermion
### self_conjugate: True (Majorana) / False (Dirac)
### ==>> Dirac fermions requires Yukawa coupling
### ==>> 

## (vector bosons are fixed by gauge groups)

# ========================= #
# == Sec. IV Interaction == #
# ========================= #

# Yukawa Coupling
# require three elements: 1) Higgs (m_h, n_h) 2) LH fermion (dim_L, gen_L) 2) RH fermion (dim_R, gen_R)
# (m_h, n_h) . (dim_L, gen) . (dim_R, gen) = (1, gen)

# Majorana Coupling



# QUESTION: How does one know about a certain interaction can break the group into a subgroup? If we know this answer, we can know the global symmetry structure.

# =============================== #
# == Sec. V Consistency Checks == #
# =============================== #

# Gauge Symmetry 

# Lorentz Symmetry

# Helicity