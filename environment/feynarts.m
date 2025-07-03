(*********************************************)
(*****     Run FeynArts Computations     *****)
(*********************************************)

(* ========== Get the arguments ========== *)
args = Rest[$CommandLine];

PathToFeynRules = args[[3]];
PathToFeynArts = args[[4]];
PathToModel = args[[5]];
ModelName = args[[6]];

SetDirectory[PathToFeynArts]; 
<< FeynArts`
SetDirectory[PathToFeynRules];
<< NLOCT`
WriteCT[ModelName <> "_QCDrenoL", "Lorentz", 
  Output -> ModelName <> "_QCDrenoL", 
  LabelInternal -> True, 
  QCDOnly -> True, 
  KeptIndices -> {}, 
  ZeroMom -> {{aS, {F[7], V[4], -F[7]}, 0}}, 
  ComplexMass -> False] // Timing 

Quit[];