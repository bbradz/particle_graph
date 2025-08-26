{{hh, -1/2*(A0[Mass2[Ah]]*Cp[hh, Ah, Ah]) - (A0[Mass2[hh]]*Cp[hh, hh, hh])/
    2 + A0[Mass2[gWp]]*Cp[hh, bar[gWp], gWp] + 
   A0[Mass2[gWpC]]*Cp[hh, bar[gWpC], gWpC] + 
   A0[Mass2[gZ]]*Cp[hh, bar[gZ], gZ] - A0[Mass2[Hp]]*Cp[hh, conj[Hp], Hp] + 
   4*Cp[hh, conj[VWp], VWp]*(A0[Mass2[VWp]] - (rMS*Mass2[VWp])/2) + 
   2*Cp[hh, VZ, VZ]*(A0[Mass2[VZ]] - (rMS*Mass2[VZ])/2) + 
   6*sum[gI1, 1, 3, A0[Mass2[Fd[{gI1}]]]*Mass[Fd[{gI1}]]*
      (Cp[hh, bar[Fd[{gI1}]], Fd[{gI1}]][PL] + 
       Cp[hh, bar[Fd[{gI1}]], Fd[{gI1}]][PR])] + 
   2*sum[gI1, 1, 3, A0[Mass2[Fe[{gI1}]]]*Mass[Fe[{gI1}]]*
      (Cp[hh, bar[Fe[{gI1}]], Fe[{gI1}]][PL] + 
       Cp[hh, bar[Fe[{gI1}]], Fe[{gI1}]][PR])] + 
   6*sum[gI1, 1, 3, A0[Mass2[Fu[{gI1}]]]*Mass[Fu[{gI1}]]*
      (Cp[hh, bar[Fu[{gI1}]], Fu[{gI1}]][PL] + 
       Cp[hh, bar[Fu[{gI1}]], Fu[{gI1}]][PR])]}}
