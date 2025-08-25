(*-------------------------------------------*)
(*                   SPheno                  *)
(*-------------------------------------------*)

OnlyLowEnergySPheno = True;

AddTreeLevelUnitarityLimits=True;

MINPAR={
{1, LamINPUT}
};

ParametersToSolveTadpoles={mu2};

BoundaryLowScaleInput={
{\[Lambda], LamINPUT}
};

DEFINITION[MatchingConditions]= {
{v, vSM},
{g1, g1SM},
{g2, g2SM},
{g3, g3SM},
{Ye[1, 1], YeSM[1,1]},
{Ye[2, 2], YeSM[2,2]},
{Ye[3, 3], YeSM[3,3]}
};

ListDecayParticles = {hh,Fv,Fe};

DefaultInputValues={
LamINPUT -> [0.1, 1]
};

RenConditionsDecays={
{dCosTW, 1/2*Cos[ThetaW] * (PiVWp/(MVWp^2) - PiVZ/(mVZ^2)) },
{dSinTW, -dCosTW/Tan[ThetaW]},
{dg2, 1/2*g2*(derPiVPheavy0 + PiVPlightMZ/MVZ^2 - (-(PiVWp/MVWp^2) + PiVZ/MVZ^2)/Tan[ThetaW]^2 + (2*PiVZVP*Tan[ThetaW])/MVZ^2)  },
{dg1, dg2*Tan[ThetaW]+g2*dSinTW/Cos[ThetaW]- dCosTW*g2*Tan[ThetaW]/Cos[ThetaW]}
};

