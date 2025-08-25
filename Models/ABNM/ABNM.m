Off[General::spell];

Model`Name = "ABNM";
Model`NameLaTeX = "A Brand New Model";
Model`Authors = "Cooper";
Model`Date = "2025-08-25-08-39-49";

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
FermionFields[[3]] = { q, 3, {uL, dL}, 1/6, 2, 3 };
FermionFields[[4]] = { u, 3, conj[uR], -2/3, 1, -3 };
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

LagHC = - Ye conj[H].e.l - Yu u.q.H - Yd conj[H].d.q;
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
    {{{eL}, {conj[eR]}}, {{EL, Ve}, {ER, Ue}}},
    {{{uL}, {conj[uR]}}, {{UL, Vu}, {UR, Uu}}},
    {{{dL}, {conj[dR]}}, {{DL, Vd}, {DR, Ud}}}
};

(*------------------------------------------------------*)
(* Dirac-Spinors *)
(*------------------------------------------------------*)

DEFINITION[EWSB][DiracSpinors]= {
    Fv -> {vL, 0},
    Fe -> {EL, conj[ER]},
    Fu -> {UL, conj[UR]},
    Fd -> {DL, conj[DR]}
};

DEFINITION[EWSB][GaugeES]= {
    Fd1 -> {FdL, 0},
    Fd2 -> {0, FdR},
    Fe1 -> {FeL, 0},
    Fe2 -> {0, FeR},
    Fu1 -> {FuL, 0},
    Fu2 -> {0, FuR}
};

