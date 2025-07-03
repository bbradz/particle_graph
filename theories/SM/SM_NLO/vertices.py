# This file was automatically created by FeynRules 2.3.49
# Mathematica version: 13.2.0 for Linux x86 (64-bit) (December 7, 2022)
# Date: Tue 24 Jun 2025 14:22:37


from object_library import all_vertices, Vertex
import particles as P
import couplings as C
import lorentz as L


V_1 = Vertex(name = 'V_1',
             particles = [ P.<>CreateObjectParticleName[PartNameMG[anti[MakeIdenticalFermions[#1]] & ]], P.<>CreateObjectParticleName[PartNameMG[anti[MakeIdenticalFermions[#1]] & ]], P.<>CreateObjectParticleName[PartNameMG[anti[MakeIdenticalFermions[#1]] & ]], P.<>CreateObjectParticleName[PartNameMG[3]], P.<>CreateObjectParticleName[PartNameMG[anti[MakeIdenticalFermions[#1]] & ]], P.<>CreateObjectParticleName[PartNameMG[3]] ],
             color = [ '1' ],
             lorentz = [ L.<>PRIVATE`ConvertSpinToString[0]<>PRIVATE`ConvertSpinToString[0]<>PRIVATE`ConvertSpinToString[0]<>PRIVATE`ConvertSpinToString[0]<>PRIVATE`ConvertSpinToString[0]<>PRIVATE`ConvertSpinToString[0]<>1 ],
             couplings = {(0,-1 + PRIVATE`ConvertSpinToString[0]<>PRIVATE`ConvertSpinToString[0]<>PRIVATE`ConvertSpinToString[0]<>PRIVATE`ConvertSpinToString[0]<>PRIVATE`ConvertSpinToString[0]<>PRIVATE`ConvertSpinToString[0]<>1):C.GC_2})

V_2 = Vertex(name = 'V_2',
             particles = [],
             color = [ '1' ],
             lorentz = [ L.1 ],
             couplings = {(0,0):C.GC_1})

