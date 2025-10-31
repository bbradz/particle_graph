(*-------------------------------------------*)
(*                   SPheno                  *)
(*-------------------------------------------*)

OnlyLowEnergySPheno = True;

AddTreeLevelUnitarityLimits=True;

MINPAR={
{1, MtINPUT},
{2, LambdaVari4INPUT}
};

ParametersToSolveTadpoles={mu2};

BoundaryLowScaleInput={
{Mt, MtINPUT},
{\[Lambda], LambdaVari4INPUT}
};

DEFINITION[MatchingConditions]= {
{v, vSM},
{g1, g1SM},
{g2, g2SM},
{g3, g3SM},
{Ye[1, 1], YeSM[1,1]},
{Ye[2, 2], YeSM[2,2]},
{Ye[3, 3], YeSM[3,3]},
{Yu[1, 1], YuSM[1,1]},
{Yu[2, 2], YuSM[2,2]},
{Yu[3, 3], Sqrt[2]/vSM*Mt},
{Yd[1, 1], YdSM[1,1]},
{Yd[2, 2], YdSM[2,2]},
{Yd[3, 3], YdSM[3,3]}
};

ListDecayParticles = {hh,Fv,Fe,Fu,Fd};

DefaultInputValues={
MtINPUT -> 198.53056835647058,
LambdaVari4INPUT -> 0.8386222920299667
};

RenConditionsDecays={
{dCosTW, 1/2*Cos[ThetaW] * (PiVWp/(MVWp^2) - PiVZ/(mVZ^2)) },
{dSinTW, -dCosTW/Tan[ThetaW]},
{dg2, 1/2*g2*(derPiVPheavy0 + PiVPlightMZ/MVZ^2 - (-(PiVWp/MVWp^2) + PiVZ/MVZ^2)/Tan[ThetaW]^2 + (2*PiVZVP*Tan[ThetaW])/MVZ^2)  },
{dg1, dg2*Tan[ThetaW]+g2*dSinTW/Cos[ThetaW]- dCosTW*g2*Tan[ThetaW]/Cos[ThetaW]}
};

