(***********************************************)
(*****     Run SARAH Model Computation     *****)
(***********************************************)

(* ========== Get the arguments ========== *)
args = Rest[$CommandLine];
PathToSARAH = args[[3]];
PathToModel = args[[4]];
ModelName = args[[5]];


SetDirectory[PathToSARAH];
<< SARAH.m
SARAH[InputDirectories]={PathToModel};
SARAH[OutputDirectory] = PathToModel;

Start[ModelName];

MakeSPheno[IncludeFlavorKit -> False, TwoLoop -> False, IncludeLoopDecays -> False, Include2loopMasses -> False] // Timing;







