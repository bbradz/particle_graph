def write_sm_gauge_boson(file):
    """
    Write the SM gauge bosons to a file
    """
    file.write("""(* Gauge bosons: physical vector fields *)        
  V[1] == { 
    ClassName       -> A, 
    SelfConjugate   -> True,  
    Mass            -> 0,  
    Width           -> 0,  
    ParticleName    -> "a", 
    PDG             -> 22, 
    PropagatorLabel -> "a", 
    PropagatorType  -> W, 
    PropagatorArrow -> None,
    FullName        -> "Photon"
  },
  V[2] == { 
    ClassName       -> Z, 
    SelfConjugate   -> True,
    Mass            -> {MZ, 91.1876},
    Width           -> {WZ, 2.4952},
    ParticleName    -> "Z", 
    PDG             -> 23, 
    PropagatorLabel -> "Z",
    PropagatorType  -> Sine,
    PropagatorArrow -> None,
    FullName        -> "Z"
  },
  V[3] == {
    ClassName        -> W,
    SelfConjugate    -> False,
    Mass             -> {MW, Internal},
    Width            -> {WW, 2.085},
    ParticleName     -> "W+",
    AntiParticleName -> "W-",
    QuantumNumbers   -> {Q -> 1},
    PDG              -> 24, 
    PropagatorLabel  -> "W",
    PropagatorType   -> Sine,
    PropagatorArrow  -> Forward,
    FullName         -> "W"
  },
  V[4] == {
    ClassName        -> G,
    SelfConjugate    -> True,
    Indices          -> {Index[Gluon]},
    Mass             -> 0,
    Width            -> 0,
    ParticleName     -> "g", 
    PDG              -> 21,
    PropagatorLabel  -> "G",
    PropagatorType   -> C,
    PropagatorArrow  -> None,
    FullName         -> "G"
  },

(* Ghosts: related to physical gauge bosons *)
  U[1] == { 
    ClassName       -> ghA, 
    SelfConjugate   -> False,
    Ghost           -> A,
    QuantumNumbers  -> {GhostNumber -> 1},
    Mass            -> 0,
    Width	    -> 0,
    PropagatorLabel -> "uA",
    PropagatorType  -> GhostDash,
    PropagatorArrow -> Forward
  },
  U[2] == {
    ClassName       -> ghZ,
    SelfConjugate   -> False,
    Ghost           -> Z,
    QuantumNumbers  -> {GhostNumber -> 1},
    Mass            -> {MZ,91.1876},  
    Width	    -> {WZ, 2.4952},
    PropagatorLabel -> "uZ",
    PropagatorType  -> GhostDash,
    PropagatorArrow -> Forward
  },
  U[31] == { 
    ClassName       -> ghWp,
    SelfConjugate   -> False, 
    Ghost           -> W,
    QuantumNumbers  -> {GhostNumber -> 1, Q -> 1},
    Mass            -> {MW,Internal}, 
    Width           -> {WW, 2.085}, 
    PropagatorLabel -> "uWp",
    PropagatorType  -> GhostDash, 
    PropagatorArrow -> Forward
  },
  U[32] == { 
    ClassName       -> ghWm,
    SelfConjugate   -> False, 
    Ghost           -> Wbar,
    QuantumNumbers  -> {GhostNumber -> 1, Q -> -1},
    Mass            -> {MW,Internal}, 
    Width           -> {WW, 2.085},
    PropagatorLabel -> "uWm",
    PropagatorType  -> GhostDash, 
    PropagatorArrow -> Forward
  },
  U[4] == { 
    ClassName       -> ghG, 
    SelfConjugate   -> False,
    Indices         -> {Index[Gluon]},
    Ghost           -> G,
    PDG             -> 82,
    QuantumNumbers  ->{GhostNumber -> 1}, 
    Mass            -> 0,
    Width	        -> 0,
    PropagatorLabel -> "uG",
    PropagatorType  -> GhostDash,
    PropagatorArrow -> Forward
  },

(* Gauge bosons: unphysical vector fields *)
  V[11] == { 
    ClassName     -> B, 
    Unphysical    -> True, 
    SelfConjugate -> True, 
    Definitions   -> { B[mu_] -> -sw Z[mu]+cw A[mu]} 
  },
  V[12] == { 
    ClassName     -> Wi,
    Unphysical    -> True,
    SelfConjugate -> True, 
    Indices       -> {Index[SU2W]},
    FlavorIndex   -> SU2W,
    Definitions   -> { Wi[mu_,1] -> (Wbar[mu]+W[mu])/Sqrt[2], Wi[mu_,2] -> (Wbar[mu]-W[mu])/(I*Sqrt[2]), Wi[mu_,3] -> cw Z[mu] + sw A[mu]}
  },

(* Ghosts: related to unphysical gauge bosons *)
  U[11] == {
    ClassName     -> ghB, 
    Unphysical    -> True,
    SelfConjugate -> False,
    Ghost         -> B, 
    Definitions   -> { ghB -> -sw ghZ + cw ghA}
  },
  U[12] == {
    ClassName     -> ghWi,
    Unphysical    -> True,
    SelfConjugate -> False,
    Ghost         -> Wi,
    Indices       -> {Index[SU2W]},
    FlavorIndex   -> SU2W,
    Definitions   -> { ghWi[1] -> (ghWp+ghWm)/Sqrt[2], ghWi[2] -> (ghWm-ghWp)/(I*Sqrt[2]), ghWi[3] -> cw ghZ+sw ghA}
  } ,\n
""")
    
def write_sm_higgs(file):
    """
    Write the SM Higgs to a file
    """
    file.write("""
(* Higgs: physical scalars  *)
  S[1] == {
    ClassName       -> H,
    SelfConjugate   -> True,
    Mass            -> {MH,125},
    Width           -> {WH,0.00407},
    PropagatorLabel -> "H",
    PropagatorType  -> D,
    PropagatorArrow -> None,
    PDG             -> 25,
    ParticleName    -> "H",
    FullName        -> "H"
  },
  S[2] == {
    ClassName       -> G0,
    SelfConjugate   -> True,
    Goldstone       -> Z,
    Mass            -> {MZ, 91.1876},
    Width           -> {WZ, 2.4952},
    PropagatorLabel -> "Go",
    PropagatorType  -> D,
    PropagatorArrow -> None,
    PDG             -> 250,
    ParticleName    -> "G0",
    FullName        -> "G0"
  },
  S[3] == {
    ClassName        -> GP,
    SelfConjugate    -> False,
    Goldstone        -> W,
    Mass             -> {MW, Internal},
    QuantumNumbers   -> {Q -> 1},
    Width            -> {WW, 2.085},
    PropagatorLabel  -> "GP",
    PropagatorType   -> D,
    PropagatorArrow  -> None,
    PDG              -> 251,
    ParticleName     -> "G+",
    AntiParticleName -> "G-",
    FullName         -> "GP"
  },

(* Higgs: unphysical scalars  *)
  S[11] == { 
    ClassName      -> Phi, 
    Unphysical     -> True, 
    Indices        -> {Index[SU2D]},
    FlavorIndex    -> SU2D,
    SelfConjugate  -> False,
    QuantumNumbers -> {Y -> 1/2},
    Definitions    -> { Phi[1] -> -I GP, Phi[2] -> (vev + H + I G0)/Sqrt[2]  }
  }
    """)


def write_sm_higgs_parameters_external(file):
    """
    Write the SM Higgs parameters to a file
    """
    file.write("""
  aEWM1 == { 
    ParameterType    -> External, 
    BlockName        -> SMINPUTS, 
    OrderBlock       -> 1, 
    Value            -> 127.9,
    InteractionOrder -> {QED,-2},
    Description      -> "Inverse of the EW coupling constant at the Z pole"
  },
  Gf == {
    ParameterType    -> External,
    BlockName        -> SMINPUTS,
    OrderBlock       -> 2,
    Value            -> 1.16637*^-5, 
    InteractionOrder -> {QED,2},
    TeX              -> Subscript[G,f],
    Description      -> "Fermi constant"
  },
  aS    == { 
    ParameterType    -> External,
    BlockName        -> SMINPUTS,
    OrderBlock       -> 3,
    Value            -> 0.1184, 
    InteractionOrder -> {QCD,2},
    TeX              -> Subscript[\[Alpha],s],
    Description      -> "Strong coupling constant at the Z pole"
  },
    """)


def write_sm_higgs_parameters_internal(file):
    """
    Write the SM Higgs parameters to a file
    """
    file.write("""
  aEW == {
    ParameterType    -> Internal,
    Value            -> 1/aEWM1,
    InteractionOrder -> {QED,2},
    TeX              -> Subscript[\[Alpha], EW],
    Description      -> "Electroweak coupling contant"
  },
  MW == { 
    ParameterType -> Internal, 
    Value         -> Sqrt[MZ^2/2+Sqrt[MZ^4/4-Pi/Sqrt[2]*aEW/Gf*MZ^2]], 
    TeX           -> Subscript[M,W], 
    Description   -> "W mass"
  },
  sw2 == { 
    ParameterType -> Internal, 
    Value         -> 1-(MW/MZ)^2, 
    Description   -> "Squared Sin of the Weinberg angle"
  },
  ee == { 
    ParameterType    -> Internal, 
    Value            -> Sqrt[4 Pi aEW], 
    InteractionOrder -> {QED,1}, 
    TeX              -> e,  
    Description      -> "Electric coupling constant"
  },
  cw == { 
    ParameterType -> Internal, 
    Value         -> Sqrt[1-sw2], 
    TeX           -> Subscript[c,w], 
    Description   -> "Cosine of the Weinberg angle"
  },
  sw == { 
    ParameterType -> Internal, 
    Value         -> Sqrt[sw2], 
    TeX           -> Subscript[s,w], 
    Description   -> "Sine of the Weinberg angle"
  },
  gw == { 
    ParameterType    -> Internal, 
    Definitions      -> {gw->ee/sw}, 
    InteractionOrder -> {QED,1},  
    TeX              -> Subscript[g,w], 
    Description      -> "Weak coupling constant at the Z pole"
  },
  g1 == { 
    ParameterType    -> Internal, 
    Definitions      -> {g1->ee/cw}, 
    InteractionOrder -> {QED,1},  
    TeX              -> Subscript[g,1], 
    Description      -> "U(1)Y coupling constant at the Z pole"
  },
  gs == { 
    ParameterType    -> Internal, 
    Value            -> Sqrt[4 Pi aS],
    InteractionOrder -> {QCD,1},  
    TeX              -> Subscript[g,s], 
    ParameterName    -> G,
    Description      -> "Strong coupling constant at the Z pole"
  },
  vev == {
    ParameterType    -> Internal,
    Value            -> 2*MW*sw/ee, 
    InteractionOrder -> {QED,-1},
    Description      -> "Higgs vacuum expectation value"
  },
  lam == {
    ParameterType    -> Internal,
    Value            -> MH^2/(2*vev^2),
    InteractionOrder -> {QED, 2},
    Description      -> "Higgs quartic coupling"
  },
  muH == {
    ParameterType -> Internal,
    Value         -> Sqrt[vev^2 lam],
    TeX           -> \[Mu],
    Description   -> "Coefficient of the quadratic piece of the Higgs potential"
  },\n""")

def write_sm_lagrangian(file):
    """
    Write the SM Lagrangian to a file
    """
    file.write("""
LGauge := Block[{mu,nu,ii,aa}, 
  ExpandIndices[-1/4 FS[B,mu,nu] FS[B,mu,nu] - 1/4 FS[Wi,mu,nu,ii] FS[Wi,mu,nu,ii] - 1/4 FS[G,mu,nu,aa] FS[G,mu,nu,aa], FlavorExpand->SU2W]];
               
LHiggs := Block[{ii,mu, feynmangaugerules},
  feynmangaugerules = If[Not[FeynmanGauge], {G0|GP|GPbar ->0}, {}];
 
  ExpandIndices[DC[Phibar[ii],mu] DC[Phi[ii],mu] + muH^2 Phibar[ii] Phi[ii] - lam Phibar[ii] Phi[ii] Phibar[jj] Phi[jj], FlavorExpand->{SU2D,SU2W}]/.feynmangaugerules
 ];  

LGhost := Block[{LGh1,LGhw,LGhs,LGhphi,mu, generators,gh,ghbar,Vectorize,phi1,phi2,togoldstones,doublet,doublet0},
  (* Pure gauge piece *) 	
  LGh1 = -ghBbar.del[DC[ghB,mu],mu];
  LGhw = -ghWibar[ii].del[DC[ghWi[ii],mu],mu];
  LGhs = -ghGbar[ii].del[DC[ghG[ii],mu],mu];

  (* Scalar pieces: see Peskin pages 739-742 *)
  (* phi1 and phi2 are the real degrees of freedom of GP *)
  (* Vectorize transforms a doublet in a vector in the phi-basis, i.e. the basis of real degrees of freedom *)
  gh    = {ghB, ghWi[1], ghWi[2], ghWi[3]};
  ghbar = {ghBbar, ghWibar[1], ghWibar[2], ghWibar[3]};
  generators = {-I/2 g1 IdentityMatrix[2], -I/2 gw PauliSigma[1], -I/2 gw PauliSigma[2], -I/2 gw PauliSigma[3]};
  doublet = Expand[{(-I phi1 - phi2)/Sqrt[2], Phi[2]} /. MR$Definitions /. vev -> 0]; 
  doublet0 = {0, vev/Sqrt[2]};
  Vectorize[{a_, b_}]:= Simplify[{Sqrt[2] Re[Expand[a]], Sqrt[2] Im[Expand[a]], Sqrt[2] Re[Expand[b]], Sqrt[2] Im[Expand[b]]}/.{Im[_]->0, Re[num_]->num}];
  togoldstones := {phi1 -> (GP + GPbar)/Sqrt[2], phi2 -> (-GP + GPbar)/(I Sqrt[2])};
  LGhphi=Plus@@Flatten[Table[-ghbar[[kkk]].gh[[lll]] Vectorize[generators[[kkk]].doublet0].Vectorize[generators[[lll]].(doublet+doublet0)],{kkk,4},{lll,4}]] /.togoldstones;

ExpandIndices[ LGhs + If[FeynmanGauge, LGh1 + LGhw + LGhphi,0], FlavorExpand->SU2W]];\n
    """)

def write_sm_gauge_parameters(file):
    """
    Write the SM gauge parameters to a file
    """
    file.write("""
  (* ************************** *)
  (* *****     Gauge      ***** *)
  (* *****   Parameters   ***** *)
  (* *****   (FeynArts)   ***** *)
  (* ************************** *)
  (* *)
  GaugeXi[ V[1]  ] = GaugeXi[A];
  GaugeXi[ V[2]  ] = GaugeXi[Z];
  GaugeXi[ V[3]  ] = GaugeXi[W];
  GaugeXi[ V[4]  ] = GaugeXi[G];
  GaugeXi[ S[1]  ] = 1;
  GaugeXi[ S[2]  ] = GaugeXi[Z];  
  GaugeXi[ S[3]  ] = GaugeXi[W];
  GaugeXi[ U[1]  ] = GaugeXi[A];
  GaugeXi[ U[2]  ] = GaugeXi[Z];
  GaugeXi[ U[31] ] = GaugeXi[W];
  GaugeXi[ U[32] ] = GaugeXi[W];
  GaugeXi[ U[4]  ] = GaugeXi[G];
    """)

def write_sm_interaction_orders(file):
    """
    Write the SM interaction orders to a file
    """
    file.write("(* ************************** *)\n")
    file.write("(* *** Interaction orders *** *)\n")
    file.write("(* ***  (as used by mg5)  *** *)\n")
    file.write("(* ************************** *)\n")
    file.write("\n")
    file.write("M$InteractionOrderHierarchy = {\n")
    file.write("  {QCD, 1},\n")
    file.write("  {QED, 2}\n")
    file.write("};\n")