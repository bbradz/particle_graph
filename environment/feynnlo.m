(****************************************)
(*****     Output the NLO model     *****)
(****************************************)

(* ========== Get the arguments ========== *)
args = Rest[$CommandLine];
PathToFeynRules = args[[3]];
PathToFeynArts = args[[4]];
PathToModel = args[[5]];
ModelName = args[[6]];

(* ========== Initialize ========== *)
$FeynRulesPath = SetDirectory[PathToFeynRules];
<< FeynRules`
$ModelPath = SetDirectory[PathToModel];
LoadModel[ModelName <> ".fr"];

$SetDirectory[PathToFeynRules];
Get[ModelName <> "_QCDrenoL.nlo"];

$ModelPath = SetDirectory[PathToModel];
WriteUFO[LSM, 
    UVCounterterms -> UV$vertlist, 
    R2Vertices -> R2$vertlist, 
    Output -> ModelName <> "_NLO"
    ];

Quit[];