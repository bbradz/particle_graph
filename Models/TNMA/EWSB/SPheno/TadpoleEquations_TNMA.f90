! ------------------------------------------------------------------------------  
! This file was automatically created by SARAH version 4.15.4 
! SARAH References: arXiv:0806.0538, 0909.2863, 1002.0840, 1207.0906, 1309.7223,
!           1405.1434, 1411.0675, 1503.03098, 1703.09237, 1706.05372, 1805.07306  
! (c) Florian Staub, Mark Goodsell and Werner Porod 2020  
! ------------------------------------------------------------------------------  
! File created at 13:50 on 27.8.2025   
! ----------------------------------------------------------------------  
 
 
Module Tadpoles_TNMA 
 
Use Model_Data_TNMA 
Use TreeLevelMasses_TNMA 
Use RGEs_TNMA 
Use Control 
Use Settings 
Use Mathematics 

Contains 


Subroutine SolveTadpoleEquations(g1,g2,g3,LambdaVari4,Yu,Yd,Ye,mu2,vvSM,Tad1Loop)

Implicit None
Real(dp),Intent(inout) :: g1,g2,g3,LambdaVari4,vvSM

Complex(dp),Intent(inout) :: Yu(3,3),Yd(3,3),Ye(3,3),mu2

Complex(dp), Intent(in) :: Tad1Loop(1)

! For numerical routines 
Real(dp) :: gC(61)
logical :: broycheck 
Real(dp) :: broyx(1)

If (HighScaleModel.Eq."LOW") Then 
mu2 = -1._dp/2._dp*(vvSM**2*LambdaVari4) + Tad1Loop(1)/vvSM

 ! ----------- Check solutions for consistency  -------- 

 ! Check for NaNs 
If (Real(mu2,dp).ne.Real(mu2,dp)) Then 
   Write(*,*) "NaN appearing in solution of tadpole equations for mu2" 
   Call TerminateProgram  
 End If 
 If (Abs(AImag(mu2)).gt.1.0E-04_dp) Then 
   Write(*,*) "No real solution of tadpole equations for mu2" 
   !Call TerminateProgram  
   mu2 = Real(mu2,dp) 
  SignOfMuChanged= .True. 
End If 
 Else 
mu2 = -1._dp/2._dp*(vvSM**2*LambdaVari4) + Tad1Loop(1)/vvSM

 ! ----------- Check solutions for consistency  -------- 

 ! Check for NaNs 
If (Real(mu2,dp).ne.Real(mu2,dp)) Then 
   Write(*,*) "NaN appearing in solution of tadpole equations for mu2" 
   Call TerminateProgram  
 End If 
 If (Abs(AImag(mu2)).gt.1.0E-04_dp) Then 
   Write(*,*) "No real solution of tadpole equations for mu2" 
   !Call TerminateProgram  
   mu2 = Real(mu2,dp) 
  SignOfMuChanged= .True. 
End If 
 End if 
End Subroutine SolveTadpoleEquations

Subroutine CalculateTadpoles(g1,g2,g3,LambdaVari4,Yu,Yd,Ye,mu2,vvSM,Tad1Loop,         & 
& TadpoleValues)

Real(dp),Intent(in) :: g1,g2,g3,LambdaVari4,vvSM

Complex(dp),Intent(in) :: Yu(3,3),Yd(3,3),Ye(3,3),mu2

Complex(dp), Intent(in) :: Tad1Loop(1)

Real(dp), Intent(out) :: TadpoleValues(1)

TadpoleValues(1) = Real(mu2*vvSM + (vvSM**3*LambdaVari4)/2._dp - Tad1Loop(1),dp) 
End Subroutine CalculateTadpoles 

End Module Tadpoles_TNMA 
 
