(****************************************)
(*****     Run FeynRules checks     *****)
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

(* ========== Run the checks ========== *)
RunChecks[] := (
    result = <|"MassCheck" -> False, "HermiticityCheck" -> False|>;
    checkMassSpectrum = Check[CheckMassSpectrum[LSM], $Failed];
    If[checkMassSpectrum === $Failed, 
        result["MassCheck"] = False;
        Return[{result, False}],
        result["MassCheck"] = True;
    ];
    checkHermiticity = Check[CheckHermiticity[LSM, FlavorExpand -> True], $Failed];
    If[checkHermiticity === $Failed, 
        result["HermiticityCheck"] = False;
        Return[{result, False}],
        result["HermiticityCheck"] = True;
    ];
    (*checkKineticTerm = Check[CheckKineticTermNormalisation[LSM, FlavorExpand -> True], $Failed];*)
    Return[{result, True}];
)

{result, pass} = RunChecks[];
Print["result: ", result];
Print["pass: ", pass];

(* Export results to text file *)
(* Convert boolean values to strings for better readability *)
statusText[status_] := If[status, "PASS", "FAIL"];

FRchecks = {
    {"Check","Status"},
    {"MassCheck",statusText[result["MassCheck"]]},
    {"HermiticityCheck",statusText[result["HermiticityCheck"]]}
};

(* Add error handling for export *)
exportSuccess = Check[
    Export[FileNameJoin[{PathToModel, "feynrules_checks.txt"}], FRchecks, "Table"],
    $Failed
];

If[exportSuccess === $Failed,
    Print["Warning: Failed to export check results to file"],
    Print["Check results exported to: ", FileNameJoin[{PathToModel, "feynrules_checks.txt"}]]
];

If[pass,
    Print["Passed all checks!"],
    Quit[]
];

(* ========== Compute the renormalization ========== *)
Print["Computing On-Shell renormalization..."];
Lren = OnShellRenormalization[LSM, QCDOnly -> True, FlavorMixing -> False];
SetDirectory[PathToFeynArts];
FAModelName = ModelName <> "_QCDrenoL";
WriteFeynArtsOutput[Lren, Output -> FAModelName, GenericFile -> False, FlavorExpand -> True];
Print["Renormalization done!"];
Quit[];
