Off[General::spell];

Model`Name = "BNM";
Model`NameLaTeX = "Brand New Model";
Model`Authors = "Cooper";
Model`Date = "2025-08-25-07-32-46";

(*-------------------------------------------*)
(*   Particle Content*)
(*-------------------------------------------*)

(* Gauge Groups *)
Gauge[[1]]={B,   U[1], hypercharge, g1,False};
Gauge[[2]]={WB, SU[2], left,        g2,True};
Gauge[[3]]={G,  SU[3], color,       g3,False};

(* Matter Fields *)
FermionFields[[1]] = { l, 3, {vL, eL}, -1/2, 2, 1 };
FermionFields[[2]] = { e, 3, conj[eR], 1, 1, 1 };
FermionFields[[3]] = { None };
FermionFields[[4]] = { None };
FermionFields[[5]] = { d, 3, conj[dR], 1/3, 1, -3 };

ScalarFields[[1]] = { H, 1, {Hp, H0}, 1/2, 2, 1 };

(*----------------------------------------------*)
(*   DEFINITION                                 *)
(*----------------------------------------------*)
NameOfStates={GaugeES, EWSB};

(* ----- Before EWSB ----- *)

DEFINITION[GaugeES][LagrangianInput]= 
{
   {LagHC, {AddHC->True}},
   {LagNoHC,{AddHC->False}}
};

LagHC = - Ye conj[H].e.l;
LagNoHC = - mu2 conj[H].H - 1/2 \[Lambda] conj[H].H.conj[H].H;

(* Gauge Sector *)

DEFINITION[EWSB][GaugeSector] = 
{   {{VB,VWB[3]},{VP,VZ},ZZ},
   {{VWB[1],VWB[2]},{VWp,conj[VWp]},ZW}
};

(* ----- VEVs ---- *)

DEFINITION[EWSB][VEVs]= 
{
    {H0, {v, 1/Sqrt[2]}, {Ah, \[ImaginaryI]/Sqrt[2]},{hh, 1/Sqrt[2]}}
};

DEFINITION[EWSB][MatterSector]= {
    {{{eL}, {conj[eR]}}, {{EL, Ve}, {ER, Ue}}}
};

(*------------------------------------------------------*)
(* Dirac-Spinors *)
(*------------------------------------------------------*)

DEFINITION[EWSB][DiracSpinors]= {
    Fv -> {vL, 0},
    Fe -> {EL, conj[ER]}
};

DEFINITION[EWSB][GaugeES]= {
    Fe1 -> {FeL, 0},
    Fe2 -> {0, FeR}
};

