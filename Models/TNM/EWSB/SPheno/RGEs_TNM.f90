! ------------------------------------------------------------------------------  
! This file was automatically created by SARAH version 4.15.4 
! SARAH References: arXiv:0806.0538, 0909.2863, 1002.0840, 1207.0906, 1309.7223,
!           1405.1434, 1411.0675, 1503.03098, 1703.09237, 1706.05372, 1805.07306  
! (c) Florian Staub, Mark Goodsell and Werner Porod 2020  
! ------------------------------------------------------------------------------  
! File created at 21:36 on 25.8.2025   
! ----------------------------------------------------------------------  
 
 
Module RGEs_TNM 
 
Use Control 
Use Settings 
Use Model_Data_TNM 
Use Mathematics 
 
Logical,Private,Save::OnlyDiagonal

Real(dp),Parameter::id3R(3,3)=& 
   & Reshape(Source=(/& 
   & 1,0,0,& 
 &0,1,0,& 
 &0,0,1& 
 &/),shape=(/3,3/)) 
Contains 


Subroutine GToParameters60(g,g1,g2,g3,LambdaVari4,Yu,Yd,Ye,m2SM)

Implicit None 
Real(dp), Intent(in) :: g(60) 
Real(dp),Intent(out) :: g1,g2,g3,LambdaVari4

Complex(dp),Intent(out) :: Yu(3,3),Yd(3,3),Ye(3,3),m2SM

Integer i1, i2, i3, i4, SumI 
 
Iname = Iname +1 
NameOfUnit(Iname) = 'GToParameters60' 
 
g1= g(1) 
g2= g(2) 
g3= g(3) 
LambdaVari4= g(4) 
Do i1 = 1,3
Do i2 = 1,3
SumI = (i2-1) + (i1-1)*3
SumI = SumI*2 
Yu(i1,i2) = Cmplx( g(SumI+5), g(SumI+6), dp) 
End Do 
 End Do 
 
Do i1 = 1,3
Do i2 = 1,3
SumI = (i2-1) + (i1-1)*3
SumI = SumI*2 
Yd(i1,i2) = Cmplx( g(SumI+23), g(SumI+24), dp) 
End Do 
 End Do 
 
Do i1 = 1,3
Do i2 = 1,3
SumI = (i2-1) + (i1-1)*3
SumI = SumI*2 
Ye(i1,i2) = Cmplx( g(SumI+41), g(SumI+42), dp) 
End Do 
 End Do 
 
m2SM= Cmplx(g(59),g(60),dp) 
Do i1=1,60 
If (g(i1).ne.g(i1)) Then 
 Write(*,*) "NaN appearing in ",NameOfUnit(Iname) 
 Write(*,*) "At position ", i1 
 Call TerminateProgram 
End if 
End do 
Iname = Iname - 1 
 
End Subroutine GToParameters60

Subroutine ParametersToG60(g1,g2,g3,LambdaVari4,Yu,Yd,Ye,m2SM,g)

Implicit None 
Real(dp), Intent(out) :: g(60) 
Real(dp), Intent(in) :: g1,g2,g3,LambdaVari4

Complex(dp), Intent(in) :: Yu(3,3),Yd(3,3),Ye(3,3),m2SM

Integer i1, i2, i3, i4, SumI 
 
Iname = Iname +1 
NameOfUnit(Iname) = 'ParametersToG60' 
 
g(1) = g1  
g(2) = g2  
g(3) = g3  
g(4) = LambdaVari4  
Do i1 = 1,3
Do i2 = 1,3
SumI = (i2-1) + (i1-1)*3
SumI = SumI*2 
g(SumI+5) = Real(Yu(i1,i2), dp) 
g(SumI+6) = Aimag(Yu(i1,i2)) 
End Do 
End Do 

Do i1 = 1,3
Do i2 = 1,3
SumI = (i2-1) + (i1-1)*3
SumI = SumI*2 
g(SumI+23) = Real(Yd(i1,i2), dp) 
g(SumI+24) = Aimag(Yd(i1,i2)) 
End Do 
End Do 

Do i1 = 1,3
Do i2 = 1,3
SumI = (i2-1) + (i1-1)*3
SumI = SumI*2 
g(SumI+41) = Real(Ye(i1,i2), dp) 
g(SumI+42) = Aimag(Ye(i1,i2)) 
End Do 
End Do 

g(59) = Real(m2SM,dp)  
g(60) = Aimag(m2SM)  
Iname = Iname - 1 
 
End Subroutine ParametersToG60

Subroutine rge60(len, T, GY, F) 
Implicit None 
Integer, Intent(in) :: len 
Real(dp), Intent(in) :: T, GY(len) 
Real(dp), Intent(out) :: F(len) 
Integer :: i1,i2,i3,i4 
Integer :: j1,j2,j3,j4,j5,j6,j7 
Real(dp) :: q 
Real(dp) :: g1,betag11,betag12,Dg1,g2,betag21,betag22,Dg2,g3,betag31,betag32,         & 
& Dg3,LambdaVari4,betaLambdaVari41,betaLambdaVari42,DLambdaVari4
Complex(dp) :: Yu(3,3),betaYu1(3,3),betaYu2(3,3),DYu(3,3),adjYu(3,3),Yd(3,3)          & 
& ,betaYd1(3,3),betaYd2(3,3),DYd(3,3),adjYd(3,3),Ye(3,3),betaYe1(3,3),betaYe2(3,3)       & 
& ,DYe(3,3),adjYe(3,3),m2SM,betam2SM1,betam2SM2,Dm2SM
Iname = Iname +1 
NameOfUnit(Iname) = 'rge60' 
 
OnlyDiagonal = .Not.GenerationMixing 
q = t 
 
Call GToParameters60(gy,g1,g2,g3,LambdaVari4,Yu,Yd,Ye,m2SM)

Call Adjungate(Yu,adjYu)
Call Adjungate(Yd,adjYd)
Call Adjungate(Ye,adjYe)


If (TwoLoopRGE) Then 
End If 
 
 
!-------------------- 
! g1 
!-------------------- 
 
betag11  = 0

 
 
If (TwoLoopRGE) Then 
betag12 = 0

 
Dg1 = oo16pi2*( betag11 + oo16pi2 * betag12 ) 

 
Else 
Dg1 = oo16pi2* betag11 
End If 
 
 
!-------------------- 
! g2 
!-------------------- 
 
betag21  = 0

 
 
If (TwoLoopRGE) Then 
betag22 = 0

 
Dg2 = oo16pi2*( betag21 + oo16pi2 * betag22 ) 

 
Else 
Dg2 = oo16pi2* betag21 
End If 
 
 
!-------------------- 
! g3 
!-------------------- 
 
betag31  = 0

 
 
If (TwoLoopRGE) Then 
betag32 = 0

 
Dg3 = oo16pi2*( betag31 + oo16pi2 * betag32 ) 

 
Else 
Dg3 = oo16pi2* betag31 
End If 
 
 
!-------------------- 
! LambdaVari4 
!-------------------- 
 
betaLambdaVari41  = 0

 
 
If (TwoLoopRGE) Then 
betaLambdaVari42 = 0

 
DLambdaVari4 = oo16pi2*( betaLambdaVari41 + oo16pi2 * betaLambdaVari42 ) 

 
Else 
DLambdaVari4 = oo16pi2* betaLambdaVari41 
End If 
 
 
!-------------------- 
! Yu 
!-------------------- 
 
betaYu1  = 0

 
 
If (TwoLoopRGE) Then 
betaYu2 = 0

 
DYu = oo16pi2*( betaYu1 + oo16pi2 * betaYu2 ) 

 
Else 
DYu = oo16pi2* betaYu1 
End If 
 
 
Call Chop(DYu) 

!-------------------- 
! Yd 
!-------------------- 
 
betaYd1  = 0

 
 
If (TwoLoopRGE) Then 
betaYd2 = 0

 
DYd = oo16pi2*( betaYd1 + oo16pi2 * betaYd2 ) 

 
Else 
DYd = oo16pi2* betaYd1 
End If 
 
 
Call Chop(DYd) 

!-------------------- 
! Ye 
!-------------------- 
 
betaYe1  = 0

 
 
If (TwoLoopRGE) Then 
betaYe2 = 0

 
DYe = oo16pi2*( betaYe1 + oo16pi2 * betaYe2 ) 

 
Else 
DYe = oo16pi2* betaYe1 
End If 
 
 
Call Chop(DYe) 

!-------------------- 
! m2SM 
!-------------------- 
 
betam2SM1  = 0

 
 
If (TwoLoopRGE) Then 
betam2SM2 = 0

 
Dm2SM = oo16pi2*( betam2SM1 + oo16pi2 * betam2SM2 ) 

 
Else 
Dm2SM = oo16pi2* betam2SM1 
End If 
 
 
Call Chop(Dm2SM) 

Call ParametersToG60(Dg1,Dg2,Dg3,DLambdaVari4,DYu,DYd,DYe,Dm2SM,f)

Iname = Iname - 1 
 
End Subroutine rge60  

Subroutine GToParameters61(g,g1,g2,g3,LambdaVari4,Yu,Yd,Ye,m2SM,vvSM)

Implicit None 
Real(dp), Intent(in) :: g(61) 
Real(dp),Intent(out) :: g1,g2,g3,LambdaVari4,vvSM

Complex(dp),Intent(out) :: Yu(3,3),Yd(3,3),Ye(3,3),m2SM

Integer i1, i2, i3, i4, SumI 
 
Iname = Iname +1 
NameOfUnit(Iname) = 'GToParameters61' 
 
g1= g(1) 
g2= g(2) 
g3= g(3) 
LambdaVari4= g(4) 
Do i1 = 1,3
Do i2 = 1,3
SumI = (i2-1) + (i1-1)*3
SumI = SumI*2 
Yu(i1,i2) = Cmplx( g(SumI+5), g(SumI+6), dp) 
End Do 
 End Do 
 
Do i1 = 1,3
Do i2 = 1,3
SumI = (i2-1) + (i1-1)*3
SumI = SumI*2 
Yd(i1,i2) = Cmplx( g(SumI+23), g(SumI+24), dp) 
End Do 
 End Do 
 
Do i1 = 1,3
Do i2 = 1,3
SumI = (i2-1) + (i1-1)*3
SumI = SumI*2 
Ye(i1,i2) = Cmplx( g(SumI+41), g(SumI+42), dp) 
End Do 
 End Do 
 
m2SM= Cmplx(g(59),g(60),dp) 
vvSM= g(61) 
Do i1=1,61 
If (g(i1).ne.g(i1)) Then 
 Write(*,*) "NaN appearing in ",NameOfUnit(Iname) 
 Write(*,*) "At position ", i1 
 Call TerminateProgram 
End if 
End do 
Iname = Iname - 1 
 
End Subroutine GToParameters61

Subroutine ParametersToG61(g1,g2,g3,LambdaVari4,Yu,Yd,Ye,m2SM,vvSM,g)

Implicit None 
Real(dp), Intent(out) :: g(61) 
Real(dp), Intent(in) :: g1,g2,g3,LambdaVari4,vvSM

Complex(dp), Intent(in) :: Yu(3,3),Yd(3,3),Ye(3,3),m2SM

Integer i1, i2, i3, i4, SumI 
 
Iname = Iname +1 
NameOfUnit(Iname) = 'ParametersToG61' 
 
g(1) = g1  
g(2) = g2  
g(3) = g3  
g(4) = LambdaVari4  
Do i1 = 1,3
Do i2 = 1,3
SumI = (i2-1) + (i1-1)*3
SumI = SumI*2 
g(SumI+5) = Real(Yu(i1,i2), dp) 
g(SumI+6) = Aimag(Yu(i1,i2)) 
End Do 
End Do 

Do i1 = 1,3
Do i2 = 1,3
SumI = (i2-1) + (i1-1)*3
SumI = SumI*2 
g(SumI+23) = Real(Yd(i1,i2), dp) 
g(SumI+24) = Aimag(Yd(i1,i2)) 
End Do 
End Do 

Do i1 = 1,3
Do i2 = 1,3
SumI = (i2-1) + (i1-1)*3
SumI = SumI*2 
g(SumI+41) = Real(Ye(i1,i2), dp) 
g(SumI+42) = Aimag(Ye(i1,i2)) 
End Do 
End Do 

g(59) = Real(m2SM,dp)  
g(60) = Aimag(m2SM)  
g(61) = vvSM  
Iname = Iname - 1 
 
End Subroutine ParametersToG61

Subroutine rge61(len, T, GY, F) 
Implicit None 
Integer, Intent(in) :: len 
Real(dp), Intent(in) :: T, GY(len) 
Real(dp), Intent(out) :: F(len) 
Integer :: i1,i2,i3,i4 
Integer :: j1,j2,j3,j4,j5,j6,j7 
Real(dp) :: q 
Real(dp) :: g1,betag11,betag12,Dg1,g2,betag21,betag22,Dg2,g3,betag31,betag32,         & 
& Dg3,LambdaVari4,betaLambdaVari41,betaLambdaVari42,DLambdaVari4,vvSM,betavvSM1,         & 
& betavvSM2,DvvSM
Complex(dp) :: Yu(3,3),betaYu1(3,3),betaYu2(3,3),DYu(3,3),adjYu(3,3),Yd(3,3)          & 
& ,betaYd1(3,3),betaYd2(3,3),DYd(3,3),adjYd(3,3),Ye(3,3),betaYe1(3,3),betaYe2(3,3)       & 
& ,DYe(3,3),adjYe(3,3),m2SM,betam2SM1,betam2SM2,Dm2SM
Iname = Iname +1 
NameOfUnit(Iname) = 'rge61' 
 
OnlyDiagonal = .Not.GenerationMixing 
q = t 
 
Call GToParameters61(gy,g1,g2,g3,LambdaVari4,Yu,Yd,Ye,m2SM,vvSM)

Call Adjungate(Yu,adjYu)
Call Adjungate(Yd,adjYd)
Call Adjungate(Ye,adjYe)


If (TwoLoopRGE) Then 
End If 
 
 
!-------------------- 
! g1 
!-------------------- 
 
betag11  = 0

 
 
If (TwoLoopRGE) Then 
betag12 = 0

 
Dg1 = oo16pi2*( betag11 + oo16pi2 * betag12 ) 

 
Else 
Dg1 = oo16pi2* betag11 
End If 
 
 
!-------------------- 
! g2 
!-------------------- 
 
betag21  = 0

 
 
If (TwoLoopRGE) Then 
betag22 = 0

 
Dg2 = oo16pi2*( betag21 + oo16pi2 * betag22 ) 

 
Else 
Dg2 = oo16pi2* betag21 
End If 
 
 
!-------------------- 
! g3 
!-------------------- 
 
betag31  = 0

 
 
If (TwoLoopRGE) Then 
betag32 = 0

 
Dg3 = oo16pi2*( betag31 + oo16pi2 * betag32 ) 

 
Else 
Dg3 = oo16pi2* betag31 
End If 
 
 
!-------------------- 
! LambdaVari4 
!-------------------- 
 
betaLambdaVari41  = 0

 
 
If (TwoLoopRGE) Then 
betaLambdaVari42 = 0

 
DLambdaVari4 = oo16pi2*( betaLambdaVari41 + oo16pi2 * betaLambdaVari42 ) 

 
Else 
DLambdaVari4 = oo16pi2* betaLambdaVari41 
End If 
 
 
!-------------------- 
! Yu 
!-------------------- 
 
betaYu1  = 0

 
 
If (TwoLoopRGE) Then 
betaYu2 = 0

 
DYu = oo16pi2*( betaYu1 + oo16pi2 * betaYu2 ) 

 
Else 
DYu = oo16pi2* betaYu1 
End If 
 
 
Call Chop(DYu) 

!-------------------- 
! Yd 
!-------------------- 
 
betaYd1  = 0

 
 
If (TwoLoopRGE) Then 
betaYd2 = 0

 
DYd = oo16pi2*( betaYd1 + oo16pi2 * betaYd2 ) 

 
Else 
DYd = oo16pi2* betaYd1 
End If 
 
 
Call Chop(DYd) 

!-------------------- 
! Ye 
!-------------------- 
 
betaYe1  = 0

 
 
If (TwoLoopRGE) Then 
betaYe2 = 0

 
DYe = oo16pi2*( betaYe1 + oo16pi2 * betaYe2 ) 

 
Else 
DYe = oo16pi2* betaYe1 
End If 
 
 
Call Chop(DYe) 

!-------------------- 
! m2SM 
!-------------------- 
 
betam2SM1  = 0

 
 
If (TwoLoopRGE) Then 
betam2SM2 = 0

 
Dm2SM = oo16pi2*( betam2SM1 + oo16pi2 * betam2SM2 ) 

 
Else 
Dm2SM = oo16pi2* betam2SM1 
End If 
 
 
Call Chop(Dm2SM) 

!-------------------- 
! vvSM 
!-------------------- 
 
betavvSM1  = 0

 
 
If (TwoLoopRGE) Then 
betavvSM2 = 0

 
DvvSM = oo16pi2*( betavvSM1 + oo16pi2 * betavvSM2 ) 

 
Else 
DvvSM = oo16pi2* betavvSM1 
End If 
 
 
Call ParametersToG61(Dg1,Dg2,Dg3,DLambdaVari4,DYu,DYd,DYe,Dm2SM,DvvSM,f)

Iname = Iname - 1 
 
End Subroutine rge61  

End Module RGEs_TNM 
 
