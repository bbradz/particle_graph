! ------------------------------------------------------------------------------  
! This file was automatically created by SARAH version 4.15.4 
! SARAH References: arXiv:0806.0538, 0909.2863, 1002.0840, 1207.0906, 1309.7223,
!           1405.1434, 1411.0675, 1503.03098, 1703.09237, 1706.05372, 1805.07306  
! (c) Florian Staub, Mark Goodsell and Werner Porod 2020  
! ------------------------------------------------------------------------------  
! File created at 13:50 on 27.8.2025   
! ----------------------------------------------------------------------  
 
 
Module LowEnergy_TNMA 
Use Control 
Use Settings 
Use Couplings_TNMA 
Use LoopCouplings_TNMA 
Use LoopMasses_SM_HC 
Use LoopFunctions 
Use LoopMasses_TNMA 
Use StandardModel 
Use RunSM_TNMA
Private::F1,F2,F3,F4,F3Gamma
!private variables
 

Contains


Subroutine Gminus2(Ifermion,MAh,MAh2,MFe,MFe2,Mhh,Mhh2,MHp,MHp2,cplcFeFeAhL,          & 
& cplcFeFeAhR,cplcFeFehhL,cplcFeFehhR,cplcFvFeHpL,cplcFvFeHpR,cplcFeFeVPL,               & 
& cplcFeFeVPR,cplcFeFvcHpL,cplcFeFvcHpR,cplHpcHpVP,a_mu)

Real(dp),Intent(in)  :: MAh,MAh2,MFe(3),MFe2(3),Mhh,Mhh2,MHp,MHp2

Complex(dp),Intent(in)  :: cplcFeFeAhL(3,3),cplcFeFeAhR(3,3),cplcFeFehhL(3,3),cplcFeFehhR(3,3),cplcFvFeHpL(3,3), & 
& cplcFvFeHpR(3,3),cplcFeFeVPL(3,3),cplcFeFeVPR(3,3),cplcFeFvcHpL(3,3),cplcFeFvcHpR(3,3),& 
& cplHpcHpVP

Real(dp), Intent(out) :: a_mu 
Integer, Intent(in) :: Ifermion 
Real(dp) :: ratio, chargefactor 
Integer :: i1, i2, i3, gt1, gt2 
Complex(dp) :: coup1L,coup1R,coup2L,coup2R 
 
Iname = Iname + 1 
NameOfUnit(Iname) = "Gminus2" 
 
 
a_mu = 0._dp 
gt1 = Ifermion 
gt2 = Ifermion 
 
chargefactor = 1 
Do i1= 2,1
  Do i2= 1,3
   i3 = i2
  If ((MAh2.gt.mz2).Or.(MFe2(i2).gt.mz2).Or.(MFe2(i3).gt.mz2)) Then
coup1L = cplcFeFeAhL(gt1,i2)
coup1R = cplcFeFeAhR(gt1,i2)
coup2L = cplcFeFeAhL(i3,gt2)
coup2R = cplcFeFeAhR(i3,gt2)
ratio = MAh2/MFe2(i2)
 chargefactor = chargefactor*(1._dp)
 If ((ratio.eq.ratio).and.(ratio.lt.1.0E+30_dp).and.(ratio.gt.1.0E-30_dp)) Then 
a_mu = a_mu - chargefactor*(Real(coup1L*Conjg(coup1R),dp)*F3gamma(ratio)/MFe(i2)& 
      & - 2._dp*MFe(Ifermion)*(Abs(coup1L)**2 + Abs(coup1R)**2)*F2(ratio)/MFe2(i2)) 
End if 
 
End if 
   End Do
  End Do


chargefactor = 1 
Do i1= 1,3
  Do i2= 2,1
   i3 = i2
  If ((0._dp.gt.mz2).Or.(MHp2.gt.mz2).Or.(MHp2.gt.mz2)) Then
coup1L = cplcFeFvcHpL(gt1,i1)
coup1R = cplcFeFvcHpR(gt1,i1)
coup2L = cplcFvFeHpL(i1,gt2)
coup2R = cplcFvFeHpR(i1,gt2)
ratio = MHp2/0._dp
 chargefactor = chargefactor*(1._dp)
 If ((ratio.eq.ratio).and.(ratio.lt.1.0E+30_dp).and.(ratio.gt.1.0E-30_dp)) Then 
a_mu = a_mu - chargefactor*(2._dp*Real(coup1L*Conjg(coup1R),dp)*F4(ratio)/0._dp& 
      & +2._dp*MFe(Ifermion)*(Abs(coup1L)**2 + Abs(coup1R)**2)*F1(ratio)/0._dp) 
End if 
 
End if 
   End Do
  End Do


chargefactor = 1 
  Do i2= 1,3
   i3 = i2
  If ((Mhh2.gt.mz2).Or.(MFe2(i2).gt.mz2).Or.(MFe2(i3).gt.mz2)) Then
coup1L = cplcFeFehhL(gt1,i2)
coup1R = cplcFeFehhR(gt1,i2)
coup2L = cplcFeFehhL(i3,gt2)
coup2R = cplcFeFehhR(i3,gt2)
ratio = Mhh2/MFe2(i2)
 chargefactor = chargefactor*(1._dp)
 If ((ratio.eq.ratio).and.(ratio.lt.1.0E+30_dp).and.(ratio.gt.1.0E-30_dp)) Then 
a_mu = a_mu - chargefactor*(Real(coup1L*Conjg(coup1R),dp)*F3gamma(ratio)/MFe(i2)& 
      & - 2._dp*MFe(Ifermion)*(Abs(coup1L)**2 + Abs(coup1R)**2)*F2(ratio)/MFe2(i2)) 
End if 
 
End if 
  End Do


a_mu = a_mu*MFe(Ifermion)*oo16pi2 
Iname = Iname -1 
 
End Subroutine Gminus2 
 
 
Subroutine LeptonEDM(Ifermion,MAh,MAh2,MFe,MFe2,Mhh,Mhh2,MHp,MHp2,MVWp,               & 
& MVWp2,MVZ,MVZ2,cplcFeFeAhL,cplcFeFeAhR,cplcFeFehhL,cplcFeFehhR,cplcFvFeHpL,            & 
& cplcFvFeHpR,cplcFeFeVPL,cplcFeFeVPR,cplcFvFeVWpL,cplcFvFeVWpR,cplcFeFeVZL,             & 
& cplcFeFeVZR,cplcFeFvcHpL,cplcFeFvcHpR,cplcFeFvcVWpL,cplcFeFvcVWpR,cplHpcHpVP,          & 
& cplcVWpVPVWp,EDM)

Implicit None
Real(dp),Intent(in)  :: MAh,MAh2,MFe(3),MFe2(3),Mhh,Mhh2,MHp,MHp2,MVWp,MVWp2,MVZ,MVZ2

Complex(dp),Intent(in)  :: cplcFeFeAhL(3,3),cplcFeFeAhR(3,3),cplcFeFehhL(3,3),cplcFeFehhR(3,3),cplcFvFeHpL(3,3), & 
& cplcFvFeHpR(3,3),cplcFeFeVPL(3,3),cplcFeFeVPR(3,3),cplcFvFeVWpL(3,3),cplcFvFeVWpR(3,3),& 
& cplcFeFeVZL(3,3),cplcFeFeVZR(3,3),cplcFeFvcHpL(3,3),cplcFeFvcHpR(3,3),cplcFeFvcVWpL(3,3),& 
& cplcFeFvcVWpR(3,3),cplHpcHpVP,cplcVWpVPVWp

Real(dp), Intent(out) :: EDM 
Real(dp) :: ratio, chargefactor 
Integer, Intent(in) :: Ifermion 
Integer :: i1, i2, i3, gt1, gt2 
Complex(dp) :: coup1L,coup1R,coup2L,coup2R 
 
Iname = Iname + 1 
NameOfUnit(Iname) = "Gminus2" 
 
 
EDM = 0._dp 
gt1 = Ifermion 
gt2 = Ifermion 
 
chargefactor = 1 
Do i1= 2,1
  Do i2= 1,3
   i3 = i2
  If ((MAh2.gt.mz2).Or.(MFe2(i2).gt.mz2).Or.(MFe2(i3).gt.mz2)) Then
coup1L = cplcFeFeAhL(gt1,i2)
coup1R = cplcFeFeAhR(gt1,i2)
coup2L = cplcFeFeAhL(i3,gt2)
coup2R = cplcFeFeAhR(i3,gt2)
!ratio = MFe2(i2)/MAh2
 !If ((ratio.eq.ratio).and.(ratio.lt.1.0E+30_dp).and.(ratio.gt.1.0E-30_dp)) Then 
!EDM = EDM -(-1)* chargefactor*Aimag(coup1R*Conjg(coup1L))*FeynFunctionA(ratio)*MFe(i2)/MAh2 
!End if 
 
ratio = MAh2/MFe2(i2)
 chargefactor = chargefactor*(1._dp)
 If ((ratio.eq.ratio).and.(ratio.lt.1.0E+30_dp).and.(ratio.gt.1.0E-30_dp)) Then 
EDM = EDM - 0.5_dp*chargefactor*(Aimag(coup1L*Conjg(coup1R))*F3gamma(ratio)/MFe(i2)) 
End if 
 
End if 
   End Do
  End Do


chargefactor = 1 
Do i1= 1,3
  Do i2= 2,1
   i3 = i2
  If ((0._dp.gt.mz2).Or.(MHp2.gt.mz2).Or.(MHp2.gt.mz2)) Then
coup1L = cplcFeFvcHpL(gt1,i1)
coup1R = cplcFeFvcHpR(gt1,i1)
coup2L = cplcFvFeHpL(i1,gt2)
coup2R = cplcFvFeHpR(i1,gt2)
!ratio = 0._dp/MHp2
 !If ((ratio.eq.ratio).and.(ratio.lt.1.0E+30_dp).and.(ratio.gt.1.0E-30_dp)) Then 
!EDM = EDM +(1)* chargefactor*Aimag(coup1L*Conjg(coup1R))*FeynFunctionB(ratio)*0._dp/MHp2 
!End if 
 
ratio = MHp2/0._dp
 chargefactor = chargefactor*(1._dp)
 If ((ratio.eq.ratio).and.(ratio.lt.1.0E+30_dp).and.(ratio.gt.1.0E-30_dp)) Then 
EDM = EDM - chargefactor*(Aimag(coup1L*Conjg(coup1R))*F4(ratio)/0._dp) 
End if 
 
End if 
   End Do
  End Do


chargefactor = 1 
Do i1= 1,3
  i2= 1
  If ((0._dp.gt.mz2).Or.(MVWp2.gt.mz2).Or.(MVWp2.gt.mz2)) Then
coup1L = cplcFeFvcVWpL(gt1,i1)
coup1R = cplcFeFvcVWpR(gt1,i1)
coup2L = cplcFvFeVWpL(i1,gt2)
coup2R = cplcFvFeVWpR(i1,gt2)
ratio = MVWp2/0._dp
 chargefactor = chargefactor*(1._dp)
 ratio = 1._dp/ratio ! conventions in 1402.7065 are different 
 If ((ratio.eq.ratio).and.(ratio.lt.1.0E+30_dp).and.(ratio.gt.1.0E-30_dp)) Then 
EDM = EDM - 2._dp*chargefactor*(Aimag(coup1L*Conjg(coup1R))*gVVF(ratio)/0._dp) 
End if 
 
End if 
   End Do


chargefactor = 1 
  Do i2= 1,3
   i3 = i2
  If ((Mhh2.gt.mz2).Or.(MFe2(i2).gt.mz2).Or.(MFe2(i3).gt.mz2)) Then
coup1L = cplcFeFehhL(gt1,i2)
coup1R = cplcFeFehhR(gt1,i2)
coup2L = cplcFeFehhL(i3,gt2)
coup2R = cplcFeFehhR(i3,gt2)
!ratio = MFe2(i2)/Mhh2
 !If ((ratio.eq.ratio).and.(ratio.lt.1.0E+30_dp).and.(ratio.gt.1.0E-30_dp)) Then 
!EDM = EDM -(-1)* chargefactor*Aimag(coup1R*Conjg(coup1L))*FeynFunctionA(ratio)*MFe(i2)/Mhh2 
!End if 
 
ratio = Mhh2/MFe2(i2)
 chargefactor = chargefactor*(1._dp)
 If ((ratio.eq.ratio).and.(ratio.lt.1.0E+30_dp).and.(ratio.gt.1.0E-30_dp)) Then 
EDM = EDM - 0.5_dp*chargefactor*(Aimag(coup1L*Conjg(coup1R))*F3gamma(ratio)/MFe(i2)) 
End if 
 
End if 
  End Do


chargefactor = 1 
  Do i2= 1,3
   i3 = i2
  If ((0._dp.gt.mz2).Or.(MFe2(i2).gt.mz2).Or.(MFe2(i3).gt.mz2)) Then
coup1L = cplcFeFeVPL(gt1,i2)
coup1R = cplcFeFeVPR(gt1,i2)
coup2L = cplcFeFeVPL(i3,gt2)
coup2R = cplcFeFeVPR(i3,gt2)
ratio = 0._dp/MFe2(i2)
 ratio = 1._dp/ratio ! conventions in 1402.7065 are different 
 chargefactor = chargefactor*(1._dp)
 If ((ratio.eq.ratio).and.(ratio.lt.1.0E+30_dp).and.(ratio.gt.1.0E-30_dp)) Then 
EDM = EDM - 4._dp*chargefactor*(Aimag(coup1L*Conjg(coup1R))*gFFV(ratio)/MFe(i2)) 
End if 
 
End if 
  End Do


chargefactor = 1 
  Do i2= 1,3
   i3 = i2
  If ((MVZ2.gt.mz2).Or.(MFe2(i2).gt.mz2).Or.(MFe2(i3).gt.mz2)) Then
coup1L = cplcFeFeVZL(gt1,i2)
coup1R = cplcFeFeVZR(gt1,i2)
coup2L = cplcFeFeVZL(i3,gt2)
coup2R = cplcFeFeVZR(i3,gt2)
ratio = MVZ2/MFe2(i2)
 ratio = 1._dp/ratio ! conventions in 1402.7065 are different 
 chargefactor = chargefactor*(1._dp)
 If ((ratio.eq.ratio).and.(ratio.lt.1.0E+30_dp).and.(ratio.gt.1.0E-30_dp)) Then 
EDM = EDM - 4._dp*chargefactor*(Aimag(coup1L*Conjg(coup1R))*gFFV(ratio)/MFe(i2)) 
End if 
 
End if 
  End Do


EDM = ecmfactor*EDM*oo16pi2 
Iname = Iname -1 
 
End Subroutine LeptonEDM 
 
 
Subroutine DeltaRho(MAh,MAh2,MFd,MFd2,MFe,MFe2,MFu,MFu2,Mhh,Mhh2,MHp,MHp2,            & 
& MVWp,MVWp2,MVZ,MVZ2,cplAhAhcVWpVWp,cplAhAhVZVZ,cplAhhhVZ,cplAhHpcVWp,cplcFdFdVZL,      & 
& cplcFdFdVZR,cplcFdFucVWpL,cplcFdFucVWpR,cplcFeFeVZL,cplcFeFeVZR,cplcFeFvcVWpL,         & 
& cplcFeFvcVWpR,cplcFuFuVZL,cplcFuFuVZR,cplcFvFvVZL,cplcFvFvVZR,cplcgAgWpcVWp,           & 
& cplcgWCgAcVWp,cplcgWCgWCVZ,cplcgWCgZcVWp,cplcgWpgWpVZ,cplcgZgWpcVWp,cplcVWpcVWpVWpVWp1,& 
& cplcVWpcVWpVWpVWp2,cplcVWpcVWpVWpVWp3,cplcVWpVPVPVWp1,cplcVWpVPVPVWp2,cplcVWpVPVPVWp3, & 
& cplcVWpVPVWp,cplcVWpVWpVZ,cplcVWpVWpVZVZ1,cplcVWpVWpVZVZ2,cplcVWpVWpVZVZ3,             & 
& cplhhcVWpVWp,cplhhhhcVWpVWp,cplhhhhVZVZ,cplhhHpcVWp,cplhhVZVZ,cplHpcHpcVWpVWp,         & 
& cplHpcHpVZ,cplHpcHpVZVZ,cplHpcVWpVP,cplHpcVWpVZ,rho)

Implicit None
Real(dp),Intent(in)  :: MAh,MAh2,MFd(3),MFd2(3),MFe(3),MFe2(3),MFu(3),MFu2(3),Mhh,Mhh2,MHp,MHp2,              & 
& MVWp,MVWp2,MVZ,MVZ2

Complex(dp),Intent(in)  :: cplAhAhcVWpVWp,cplAhAhVZVZ,cplAhhhVZ,cplAhHpcVWp,cplcFdFdVZL(3,3),cplcFdFdVZR(3,3),   & 
& cplcFdFucVWpL(3,3),cplcFdFucVWpR(3,3),cplcFeFeVZL(3,3),cplcFeFeVZR(3,3),               & 
& cplcFeFvcVWpL(3,3),cplcFeFvcVWpR(3,3),cplcFuFuVZL(3,3),cplcFuFuVZR(3,3),               & 
& cplcFvFvVZL(3,3),cplcFvFvVZR(3,3),cplcgAgWpcVWp,cplcgWCgAcVWp,cplcgWCgWCVZ,            & 
& cplcgWCgZcVWp,cplcgWpgWpVZ,cplcgZgWpcVWp,cplcVWpcVWpVWpVWp1,cplcVWpcVWpVWpVWp2,        & 
& cplcVWpcVWpVWpVWp3,cplcVWpVPVPVWp1,cplcVWpVPVPVWp2,cplcVWpVPVPVWp3,cplcVWpVPVWp,       & 
& cplcVWpVWpVZ,cplcVWpVWpVZVZ1,cplcVWpVWpVZVZ2,cplcVWpVWpVZVZ3,cplhhcVWpVWp,             & 
& cplhhhhcVWpVWp,cplhhhhVZVZ,cplhhHpcVWp,cplhhVZVZ,cplHpcHpcVWpVWp,cplHpcHpVZ,           & 
& cplHpcHpVZVZ,cplHpcVWpVP,cplHpcVWpVZ

Real(dp), Intent(out) :: rho 
Integer :: i1, i2, i3, kont 
Real(dp) ::  delta_rho, delta_rho0, Drho_top, mu_old 
Complex(dp) ::  dmW2, dmz2 
mu_old = SetRenormalizationScale(mZ2) 
 
Call Pi1LoopVZ(0._dp,Mhh,Mhh2,MAh,MAh2,MFd,MFd2,MFe,MFe2,MFu,MFu2,MVZ,MVZ2,           & 
& MHp,MHp2,MVWp,MVWp2,cplAhhhVZ,cplcFdFdVZL,cplcFdFdVZR,cplcFeFeVZL,cplcFeFeVZR,         & 
& cplcFuFuVZL,cplcFuFuVZR,cplcFvFvVZL,cplcFvFvVZR,cplcgWpgWpVZ,cplcgWCgWCVZ,             & 
& cplhhVZVZ,cplHpcHpVZ,cplHpcVWpVZ,cplcVWpVWpVZ,cplAhAhVZVZ,cplhhhhVZVZ,cplHpcHpVZVZ,    & 
& cplcVWpVWpVZVZ1,cplcVWpVWpVZVZ2,cplcVWpVWpVZVZ3,kont,dmZ2)

Call Pi1LoopVWp(0._dp,MHp,MHp2,MAh,MAh2,MFd,MFd2,MFu,MFu2,MFe,MFe2,Mhh,               & 
& Mhh2,MVWp,MVWp2,MVZ,MVZ2,cplAhHpcVWp,cplcFdFucVWpL,cplcFdFucVWpR,cplcFeFvcVWpL,        & 
& cplcFeFvcVWpR,cplcgWCgAcVWp,cplcgAgWpcVWp,cplcgZgWpcVWp,cplcgWCgZcVWp,cplhhHpcVWp,     & 
& cplhhcVWpVWp,cplHpcVWpVP,cplHpcVWpVZ,cplcVWpVPVWp,cplcVWpVWpVZ,cplAhAhcVWpVWp,         & 
& cplhhhhcVWpVWp,cplHpcHpcVWpVWp,cplcVWpVPVPVWp3,cplcVWpVPVPVWp1,cplcVWpVPVPVWp2,        & 
& cplcVWpcVWpVWpVWp2,cplcVWpcVWpVWpVWp3,cplcVWpcVWpVWpVWp1,cplcVWpVWpVZVZ1,              & 
& cplcVWpVWpVZVZ2,cplcVWpVWpVZVZ3,kont,dmW2)

Drho_top = 3*G_F*mf_u(3)**2*oosqrt2*oo8pi2 
 
mu_old = SetRenormalizationScale(mu_old) 
 
! Tree Level 
delta_rho0 =  0 
 
! 1-Loop Level 
delta_rho =  dmZ2/mz2 - dMW2/mW2
 Rho= delta_rho + delta_rho0
End subroutine DeltaRho 
 
 
Subroutine STUparameter(vSM,g1SM,g2SM,g3SM,YuSM,YdSM,YeSM,MAh,MAh2,MFd,               & 
& MFd2,MFe,MFe2,MFu,MFu2,Mhh,Mhh2,MHp,MHp2,MVWp,MVWp2,MVZ,MVZ2,cplAhcHpVWp,              & 
& cplhhcHpVWp,cplHpcHpVP,cplHpcHpVZ,cplcHpVPVWp,cplcHpVWpVZ,cplHpcHpVPVP,cplHpcHpcVWpVWp,& 
& cplHpcHpVZVZ,cplAhhhVZ,cplAhHpcVWp,cplAhAhcVWpVWp,cplAhAhVZVZ,cplhhHpcVWp,             & 
& cplhhcVWpVWp,cplhhVZVZ,cplhhhhcVWpVWp,cplhhhhVZVZ,cplcFdFdVPL,cplcFdFdVPR,             & 
& cplcFeFeVPL,cplcFeFeVPR,cplcFuFuVPL,cplcFuFuVPR,cplcgWpgWpVP,cplcgWCgWCVP,             & 
& cplHpcVWpVP,cplcVWpVPVWp,cplcVWpVPVPVWp1,cplcVWpVPVPVWp2,cplcVWpVPVPVWp3,              & 
& cplcFdFdVZL,cplcFdFdVZR,cplcFeFeVZL,cplcFeFeVZR,cplcFuFuVZL,cplcFuFuVZR,               & 
& cplcFvFvVZL,cplcFvFvVZR,cplcgWpgWpVZ,cplcgWCgWCVZ,cplHpcVWpVZ,cplcVWpVWpVZ,            & 
& cplcVWpVWpVZVZ1,cplcVWpVWpVZVZ2,cplcVWpVWpVZVZ3,cplcFdFucVWpL,cplcFdFucVWpR,           & 
& cplcFeFvcVWpL,cplcFeFvcVWpR,cplcgWCgAcVWp,cplcgAgWpcVWp,cplcgZgWpcVWp,cplcgWCgZcVWp,   & 
& cplcVWpcVWpVWpVWp1,cplcVWpcVWpVWpVWp2,cplcVWpcVWpVWpVWp3,cplHpcHpVPVZ,cplcVWpVPVWpVZ1, & 
& cplcVWpVPVWpVZ2,cplcVWpVPVWpVZ3,Spar,Tpar,Upar)

Implicit None
Real(dp),Intent(in)  :: MAh,MAh2,MFd(3),MFd2(3),MFe(3),MFe2(3),MFu(3),MFu2(3),Mhh,Mhh2,MHp,MHp2,              & 
& MVWp,MVWp2,MVZ,MVZ2

Complex(dp),Intent(in)  :: cplAhcHpVWp,cplhhcHpVWp,cplHpcHpVP,cplHpcHpVZ,cplcHpVPVWp,cplcHpVWpVZ,cplHpcHpVPVP,   & 
& cplHpcHpcVWpVWp,cplHpcHpVZVZ,cplAhhhVZ,cplAhHpcVWp,cplAhAhcVWpVWp,cplAhAhVZVZ,         & 
& cplhhHpcVWp,cplhhcVWpVWp,cplhhVZVZ,cplhhhhcVWpVWp,cplhhhhVZVZ,cplcFdFdVPL(3,3),        & 
& cplcFdFdVPR(3,3),cplcFeFeVPL(3,3),cplcFeFeVPR(3,3),cplcFuFuVPL(3,3),cplcFuFuVPR(3,3),  & 
& cplcgWpgWpVP,cplcgWCgWCVP,cplHpcVWpVP,cplcVWpVPVWp,cplcVWpVPVPVWp1,cplcVWpVPVPVWp2,    & 
& cplcVWpVPVPVWp3,cplcFdFdVZL(3,3),cplcFdFdVZR(3,3),cplcFeFeVZL(3,3),cplcFeFeVZR(3,3),   & 
& cplcFuFuVZL(3,3),cplcFuFuVZR(3,3),cplcFvFvVZL(3,3),cplcFvFvVZR(3,3),cplcgWpgWpVZ,      & 
& cplcgWCgWCVZ,cplHpcVWpVZ,cplcVWpVWpVZ,cplcVWpVWpVZVZ1,cplcVWpVWpVZVZ2,cplcVWpVWpVZVZ3, & 
& cplcFdFucVWpL(3,3),cplcFdFucVWpR(3,3),cplcFeFvcVWpL(3,3),cplcFeFvcVWpR(3,3),           & 
& cplcgWCgAcVWp,cplcgAgWpcVWp,cplcgZgWpcVWp,cplcgWCgZcVWp,cplcVWpcVWpVWpVWp1,            & 
& cplcVWpcVWpVWpVWp2,cplcVWpcVWpVWpVWp3,cplHpcHpVPVZ,cplcVWpVPVWpVZ1,cplcVWpVPVWpVZ2,    & 
& cplcVWpVPVWpVZ3

Real(dp), Intent(out) :: Spar,Tpar,Upar 
Integer :: i1, i2, i3, kont 
Real(dp) ::  mu_old, cw2, sw2, rMS_SM_save 
Real(dp) ::  delta_T_SM, delta_S_SM, delta_U_SM 
Complex(dp) :: PiZZ, PiZZ_mz2, PiWW,PiWW_mw2,PiZg_mz2, Pigg_mz2 
Complex(dp) :: PiZZ_SM, PiZZ_mz2_SM, PiWW_SM,PiWW_mw2_SM,PiZg_mz2_SM, Pigg_mz2_SM 
Complex(dp) :: LamSM 
Complex(dp), Intent(in) :: YdSM(3,3), YuSM(3,3), YeSM(3,3) 
Real(dp), Intent(in) :: g1SM,g2SM,g3SM,vSM 
mu_old = SetRenormalizationScale(mZ2) 
 
LamSM = Mhh2/vSM**2
 
Call OneLoop_Z_W_Gamma_SM(vSM,g1SM,g2SM,g3SM,LamSM,YuSM,YdSM,YeSM,kont, & 
 & PiZZ_SM,PiZZ_mz2_SM,PiWW_SM,PiWW_mw2_SM,PiZg_mz2_SM,Pigg_mz2_SM) 
Call Pi1LoopVZ(0._dp,Mhh,Mhh2,MAh,MAh2,MFd,MFd2,MFe,MFe2,MFu,MFu2,MVZ,MVZ2,           & 
& MHp,MHp2,MVWp,MVWp2,cplAhhhVZ,cplcFdFdVZL,cplcFdFdVZR,cplcFeFeVZL,cplcFeFeVZR,         & 
& cplcFuFuVZL,cplcFuFuVZR,cplcFvFvVZL,cplcFvFvVZR,cplcgWpgWpVZ,cplcgWCgWCVZ,             & 
& cplhhVZVZ,cplHpcHpVZ,cplHpcVWpVZ,cplcVWpVWpVZ,cplAhAhVZVZ,cplhhhhVZVZ,cplHpcHpVZVZ,    & 
& cplcVWpVWpVZVZ1,cplcVWpVWpVZVZ2,cplcVWpVWpVZVZ3,kont,PiZZ)

Call Pi1LoopVZ(MVZ2,Mhh,Mhh2,MAh,MAh2,MFd,MFd2,MFe,MFe2,MFu,MFu2,MVZ,MVZ2,            & 
& MHp,MHp2,MVWp,MVWp2,cplAhhhVZ,cplcFdFdVZL,cplcFdFdVZR,cplcFeFeVZL,cplcFeFeVZR,         & 
& cplcFuFuVZL,cplcFuFuVZR,cplcFvFvVZL,cplcFvFvVZR,cplcgWpgWpVZ,cplcgWCgWCVZ,             & 
& cplhhVZVZ,cplHpcHpVZ,cplHpcVWpVZ,cplcVWpVWpVZ,cplAhAhVZVZ,cplhhhhVZVZ,cplHpcHpVZVZ,    & 
& cplcVWpVWpVZVZ1,cplcVWpVWpVZVZ2,cplcVWpVWpVZVZ3,kont,PiZZ_mz2)

Call Pi1LoopVWp(0._dp,MHp,MHp2,MAh,MAh2,MFd,MFd2,MFu,MFu2,MFe,MFe2,Mhh,               & 
& Mhh2,MVWp,MVWp2,MVZ,MVZ2,cplAhHpcVWp,cplcFdFucVWpL,cplcFdFucVWpR,cplcFeFvcVWpL,        & 
& cplcFeFvcVWpR,cplcgWCgAcVWp,cplcgAgWpcVWp,cplcgZgWpcVWp,cplcgWCgZcVWp,cplhhHpcVWp,     & 
& cplhhcVWpVWp,cplHpcVWpVP,cplHpcVWpVZ,cplcVWpVPVWp,cplcVWpVWpVZ,cplAhAhcVWpVWp,         & 
& cplhhhhcVWpVWp,cplHpcHpcVWpVWp,cplcVWpVPVPVWp3,cplcVWpVPVPVWp1,cplcVWpVPVPVWp2,        & 
& cplcVWpcVWpVWpVWp2,cplcVWpcVWpVWpVWp3,cplcVWpcVWpVWpVWp1,cplcVWpVWpVZVZ1,              & 
& cplcVWpVWpVZVZ2,cplcVWpVWpVZVZ3,kont,PiWW)

Call Pi1LoopVWp(MVWp2,MHp,MHp2,MAh,MAh2,MFd,MFd2,MFu,MFu2,MFe,MFe2,Mhh,               & 
& Mhh2,MVWp,MVWp2,MVZ,MVZ2,cplAhHpcVWp,cplcFdFucVWpL,cplcFdFucVWpR,cplcFeFvcVWpL,        & 
& cplcFeFvcVWpR,cplcgWCgAcVWp,cplcgAgWpcVWp,cplcgZgWpcVWp,cplcgWCgZcVWp,cplhhHpcVWp,     & 
& cplhhcVWpVWp,cplHpcVWpVP,cplHpcVWpVZ,cplcVWpVPVWp,cplcVWpVWpVZ,cplAhAhcVWpVWp,         & 
& cplhhhhcVWpVWp,cplHpcHpcVWpVWp,cplcVWpVPVPVWp3,cplcVWpVPVPVWp1,cplcVWpVPVPVWp2,        & 
& cplcVWpcVWpVWpVWp2,cplcVWpcVWpVWpVWp3,cplcVWpcVWpVWpVWp1,cplcVWpVWpVZVZ1,              & 
& cplcVWpVWpVZVZ2,cplcVWpVWpVZVZ3,kont,PiWW_mw2)

Call Pi1LoopVPVZ(MVZ2,MFd,MFd2,MFe,MFe2,MFu,MFu2,MHp,MHp2,MVWp,MVWp2,cplcFdFdVPL,     & 
& cplcFdFdVPR,cplcFdFdVZL,cplcFdFdVZR,cplcFeFeVPL,cplcFeFeVPR,cplcFeFeVZL,               & 
& cplcFeFeVZR,cplcFuFuVPL,cplcFuFuVPR,cplcFuFuVZL,cplcFuFuVZR,cplcgWCgWCVP,              & 
& cplcgWCgWCVZ,cplcgWpgWpVP,cplcgWpgWpVZ,cplcHpVPVWp,cplcHpVWpVZ,cplcVWpVPVWp,           & 
& cplcVWpVPVWpVZ1,cplcVWpVPVWpVZ2,cplcVWpVPVWpVZ3,cplcVWpVWpVZ,cplHpcHpVP,               & 
& cplHpcHpVPVZ,cplHpcHpVZ,cplHpcVWpVP,cplHpcVWpVZ,kont,PiZg_mz2)

Call Pi1LoopVP(MVZ2,MFd,MFd2,MFe,MFe2,MFu,MFu2,MHp,MHp2,MVWp,MVWp2,cplcFdFdVPL,       & 
& cplcFdFdVPR,cplcFeFeVPL,cplcFeFeVPR,cplcFuFuVPL,cplcFuFuVPR,cplcgWpgWpVP,              & 
& cplcgWCgWCVP,cplHpcHpVP,cplHpcVWpVP,cplcVWpVPVWp,cplHpcHpVPVP,cplcVWpVPVPVWp3,         & 
& cplcVWpVPVPVWp1,cplcVWpVPVPVWp2,kont,Pigg_mz2)

PiZZ = PiZZ - PiZZ_SM 
PiZZ_mz2 = PiZZ_mz2 - PiZZ_mz2_SM 
PiWW = PiWW - PiWW_SM 
PiWW_mw2 = PiWW_mw2 - PiWW_mw2_SM 
PiZg_mz2 = PiZg_mz2 - PiZg_mz2_SM 
Pigg_mz2 = Pigg_mz2 - Pigg_mz2_SM 
sw2 = 0.22290_dp 
cw2 = 1._dp - sw2 
 
mu_old = SetRenormalizationScale(mu_old) 
 
! T-parameter 
Tpar= PiWW/mW2 - PiZZ/mz2   
 Tpar= -Tpar/alpha 


! S-parameter 
Spar= (PiZZ_mz2-PiZZ)/mz2 - (cw2-sw2)/(sqrt(cw2*sw2))*PiZg_mz2/mz2 - Pigg_mz2/mz2
Spar= -4._dp*sw2*cw2/alpha*Spar 


! U-parameter 
Upar= (PiWW_mw2-PiWW)/mw2 -cw2*(PiZZ_mz2-PiZZ)/mz2 - 2._dp*(sqrt(cw2*sw2))*PiZg_mz2/mz2 - sw2*Pigg_mz2/mz2
Upar= -4._dp*sw2/alpha*Upar 


End subroutine STUparameter 
 
 
Subroutine CalculateLowEnergyConstraints(g1input,g2input,g3input,LambdaVari4input,    & 
& Yuinput,Ydinput,Yeinput,mu2input,vvSMinput,Tpar,Spar,Upar,ae,amu,atau,EDMe,            & 
& EDMmu,EDMtau,dRho)

Real(dp),Intent(inout) :: g1input,g2input,g3input,LambdaVari4input,vvSMinput

Complex(dp),Intent(inout) :: Yuinput(3,3),Ydinput(3,3),Yeinput(3,3),mu2input

Real(dp) :: MAh,MAh2,MFd(3),MFd2(3),MFe(3),MFe2(3),MFu(3),MFu2(3),Mhh,Mhh2,MHp,MHp2,              & 
& MVWp,MVWp2,MVZ,MVZ2,TW,ZZ(2,2)

Complex(dp) :: ZDR(3,3),ZER(3,3),ZUR(3,3),ZDL(3,3),ZEL(3,3),ZUL(3,3),ZW(2,2)

Real(dp) :: g1,g2,g3,LambdaVari4,vvSM

Complex(dp) :: Yu(3,3),Yd(3,3),Ye(3,3),mu2

Complex(dp) :: cplAhAhcVWpVWp,cplAhAhhh,cplAhAhVZVZ,cplAhcHpVWp,cplAhhhVZ,cplAhHpcVWp,               & 
& cplcFdFdAhL(3,3),cplcFdFdAhR(3,3),cplcFdFdhhL(3,3),cplcFdFdhhR(3,3),cplcFdFdVGL(3,3),  & 
& cplcFdFdVGR(3,3),cplcFdFdVPL(3,3),cplcFdFdVPR(3,3),cplcFdFdVZL(3,3),cplcFdFdVZR(3,3),  & 
& cplcFdFucHpL(3,3),cplcFdFucHpR(3,3),cplcFdFucVWpL(3,3),cplcFdFucVWpR(3,3),             & 
& cplcFeFeAhL(3,3),cplcFeFeAhR(3,3),cplcFeFehhL(3,3),cplcFeFehhR(3,3),cplcFeFeVPL(3,3),  & 
& cplcFeFeVPR(3,3),cplcFeFeVZL(3,3),cplcFeFeVZR(3,3),cplcFeFvcHpL(3,3),cplcFeFvcHpR(3,3),& 
& cplcFeFvcVWpL(3,3),cplcFeFvcVWpR(3,3),cplcFuFdHpL(3,3),cplcFuFdHpR(3,3),               & 
& cplcFuFdVWpL(3,3),cplcFuFdVWpR(3,3),cplcFuFuAhL(3,3),cplcFuFuAhR(3,3),cplcFuFuhhL(3,3),& 
& cplcFuFuhhR(3,3),cplcFuFuVGL(3,3),cplcFuFuVGR(3,3),cplcFuFuVPL(3,3),cplcFuFuVPR(3,3),  & 
& cplcFuFuVZL(3,3),cplcFuFuVZR(3,3),cplcFvFeHpL(3,3),cplcFvFeHpR(3,3),cplcFvFeVWpL(3,3), & 
& cplcFvFeVWpR(3,3),cplcFvFvVZL(3,3),cplcFvFvVZR(3,3),cplcgAgWpcVWp,cplcgWCgAcVWp,       & 
& cplcgWCgWCVP,cplcgWCgWCVZ,cplcgWCgZcVWp,cplcgWpgWpVP,cplcgWpgWpVZ,cplcgZgWpcVWp,       & 
& cplcHpVPVWp,cplcHpVWpVZ,cplcVWpcVWpVWpVWp1,cplcVWpcVWpVWpVWp2,cplcVWpcVWpVWpVWp3,      & 
& cplcVWpVPVPVWp1,cplcVWpVPVPVWp2,cplcVWpVPVPVWp3,cplcVWpVPVWp,cplcVWpVPVWpVZ1,          & 
& cplcVWpVPVWpVZ2,cplcVWpVPVWpVZ3,cplcVWpVWpVZ,cplcVWpVWpVZVZ1,cplcVWpVWpVZVZ2,          & 
& cplcVWpVWpVZVZ3,cplhhcHpVWp,cplhhcVWpVWp,cplhhhhcVWpVWp,cplhhhhhh,cplhhhhVZVZ,         & 
& cplhhHpcHp,cplhhHpcVWp,cplhhVZVZ,cplHpcHpcVWpVWp,cplHpcHpVP,cplHpcHpVPVP,              & 
& cplHpcHpVPVZ,cplHpcHpVZ,cplHpcHpVZVZ,cplHpcVWpVP,cplHpcVWpVZ,cplVGVGVG

Real(dp),Intent(out) :: Tpar,Spar,Upar,ae,amu,atau,EDMe,EDMmu,EDMtau,dRho

Complex(dp) :: c7,c7p,c8,c8p 
Real(dp) :: ResultMuE(6), ResultTauMeson(3), ResultTemp(99) 
Real(dp) :: epsTree=1.0E-20_dp 
Complex(dp), Dimension(3,3) :: Yu_save, Yd_save, Ye_save, CKMsave 
Complex(dp) :: YuSM(3,3),YeSM(3,3),YdSM(3,3) 
Real(dp)::g1SM, g2SM, g3SM, vSM 
Real(dp) ::Qin,vev2,sinw2, mzsave, scalein, scale_save, gSM(11),Qinsave, maxdiff =0._dp 
Integer :: i1, i2, i3, gt1, gt2, gt3, gt4,iQTEST, iQFinal 
Integer :: IndexArray4(99,4), IndexArray3(99,3), IndexArray2(99,2)   
Write(*,*) "Calculating low energy constraints" 
If (MatchingOrder.gt.-1) Then 


End if 
!-------------------------------------
! running to 160 GeV for b -> so gamma
!-------------------------------------

Qin=sqrt(getRenormalizationScale()) 
scale_save = Qin 
Call RunSM_and_SUSY_RGEs(160._dp,g1input,g2input,g3input,LambdaVari4input,            & 
& Yuinput,Ydinput,Yeinput,mu2input,vvSMinput,g1,g2,g3,LambdaVari4,Yu,Yd,Ye,              & 
& mu2,vvSM,CKM_160,sinW2_160,Alpha_160,AlphaS_160,.false.)

If (MatchingOrder.eq.-1) Then 
g1=g1input 
g2=g2input 
g3=g3input 
LambdaVari4=LambdaVari4input 
Yu=Yuinput 
Yd=Ydinput 
Ye=Yeinput 
mu2=mu2input 
vvSM=vvSMinput 
End if 

! ## All contributions ## 

Call SolveTadpoleEquations(g1,g2,g3,LambdaVari4,Yu,Yd,Ye,mu2,vvSM,(/ ZeroC /))

Call TreeMasses(MAh,MAh2,MFd,MFd2,MFe,MFe2,MFu,MFu2,Mhh,Mhh2,MHp,MHp2,MVWp,           & 
& MVWp2,MVZ,MVZ2,TW,ZDR,ZER,ZUR,ZDL,ZEL,ZUL,ZW,ZZ,vvSM,g1,g2,g3,LambdaVari4,             & 
& Yu,Yd,Ye,mu2,GenerationMixing,kont)

 mf_d_160 = MFd(1:3) 
 mf_d2_160 = MFd(1:3)**2 
 mf_u_160 = MFu(1:3) 
 mf_u2_160 = MFu(1:3)**2 
 mf_l_160 = MFe(1:3) 
 mf_l2_160 = MFe(1:3)**2 
If (WriteParametersAtQ) Then 
! Write running parameters at Q=160 GeV in output file 
g1input = g1
g2input = g2
g3input = g3
LambdaVari4input = LambdaVari4
Yuinput = Yu
Ydinput = Yd
Yeinput = Ye
mu2input = mu2
vvSMinput = vvSM
End If 
 
Mhh= MhhL 
Mhh2 = Mhh2L 
MAh= MAhL 
MAh2 = MAh2L 
MAh=MVZ
MAh2=MVZ2
MHp=MVWp
MHp2=MVWp2
Call AllCouplings(LambdaVari4,vvSM,g1,g2,TW,g3,Yd,ZDL,ZDR,Ye,ZEL,ZER,Yu,              & 
& ZUL,ZUR,cplAhAhhh,cplhhhhhh,cplhhHpcHp,cplAhhhVZ,cplAhHpcVWp,cplAhcHpVWp,              & 
& cplhhHpcVWp,cplhhcHpVWp,cplHpcHpVP,cplHpcHpVZ,cplhhcVWpVWp,cplhhVZVZ,cplHpcVWpVP,      & 
& cplHpcVWpVZ,cplcHpVPVWp,cplcHpVWpVZ,cplVGVGVG,cplcVWpVPVWp,cplcVWpVWpVZ,               & 
& cplcFdFdAhL,cplcFdFdAhR,cplcFeFeAhL,cplcFeFeAhR,cplcFuFuAhL,cplcFuFuAhR,               & 
& cplcFdFdhhL,cplcFdFdhhR,cplcFuFdHpL,cplcFuFdHpR,cplcFeFehhL,cplcFeFehhR,               & 
& cplcFvFeHpL,cplcFvFeHpR,cplcFuFuhhL,cplcFuFuhhR,cplcFdFucHpL,cplcFdFucHpR,             & 
& cplcFeFvcHpL,cplcFeFvcHpR,cplcFdFdVGL,cplcFdFdVGR,cplcFdFdVPL,cplcFdFdVPR,             & 
& cplcFuFdVWpL,cplcFuFdVWpR,cplcFdFdVZL,cplcFdFdVZR,cplcFeFeVPL,cplcFeFeVPR,             & 
& cplcFvFeVWpL,cplcFvFeVWpR,cplcFeFeVZL,cplcFeFeVZR,cplcFuFuVGL,cplcFuFuVGR,             & 
& cplcFuFuVPL,cplcFuFuVPR,cplcFuFuVZL,cplcFuFuVZR,cplcFdFucVWpL,cplcFdFucVWpR,           & 
& cplcFvFvVZL,cplcFvFvVZR,cplcFeFvcVWpL,cplcFeFvcVWpR)


! ## SM only ##

CKM = CKMsave 
!-------------------------------------
! running to M_Z 
!-------------------------------------

scalein=SetRenormalizationScale(scale_save**2) 
If (MatchingOrder.gt.-1) Then 
Call RunSM_and_SUSY_RGEs(mz,g1input,g2input,g3input,LambdaVari4input,Yuinput,         & 
& Ydinput,Yeinput,mu2input,vvSMinput,g1,g2,g3,LambdaVari4,Yu,Yd,Ye,mu2,vvSM,             & 
& CKM_MZ,sinW2_MZ,Alpha_MZ,AlphaS_MZ,.true.)

Else 
g1=g1input 
g2=g2input 
g3=g3input 
LambdaVari4=LambdaVari4input 
Yu=Yuinput 
Yd=Ydinput 
Ye=Yeinput 
mu2=mu2input 
vvSM=vvSMinput 
End if 
Call SolveTadpoleEquations(g1,g2,g3,LambdaVari4,Yu,Yd,Ye,mu2,vvSM,(/ ZeroC /))

Call TreeMasses(MAh,MAh2,MFd,MFd2,MFe,MFe2,MFu,MFu2,Mhh,Mhh2,MHp,MHp2,MVWp,           & 
& MVWp2,MVZ,MVZ2,TW,ZDR,ZER,ZUR,ZDL,ZEL,ZUL,ZW,ZZ,vvSM,g1,g2,g3,LambdaVari4,             & 
& Yu,Yd,Ye,mu2,GenerationMixing,kont)

mzsave  = sqrt(mz2) 
 mf_d_mz = MFd(1:3) 
 mf_d2_mz = MFd(1:3)**2 
 mf_u_mz = MFu(1:3) 
 mf_u2_mz = MFu(1:3)**2 
 mf_l_MZ = MFe(1:3) 
 mf_l2_MZ = MFe(1:3)**2 
Call AllCouplings(LambdaVari4,vvSM,g1,g2,TW,g3,Yd,ZDL,ZDR,Ye,ZEL,ZER,Yu,              & 
& ZUL,ZUR,cplAhAhhh,cplhhhhhh,cplhhHpcHp,cplAhhhVZ,cplAhHpcVWp,cplAhcHpVWp,              & 
& cplhhHpcVWp,cplhhcHpVWp,cplHpcHpVP,cplHpcHpVZ,cplhhcVWpVWp,cplhhVZVZ,cplHpcVWpVP,      & 
& cplHpcVWpVZ,cplcHpVPVWp,cplcHpVWpVZ,cplVGVGVG,cplcVWpVPVWp,cplcVWpVWpVZ,               & 
& cplcFdFdAhL,cplcFdFdAhR,cplcFeFeAhL,cplcFeFeAhR,cplcFuFuAhL,cplcFuFuAhR,               & 
& cplcFdFdhhL,cplcFdFdhhR,cplcFuFdHpL,cplcFuFdHpR,cplcFeFehhL,cplcFeFehhR,               & 
& cplcFvFeHpL,cplcFvFeHpR,cplcFuFuhhL,cplcFuFuhhR,cplcFdFucHpL,cplcFdFucHpR,             & 
& cplcFeFvcHpL,cplcFeFvcHpR,cplcFdFdVGL,cplcFdFdVGR,cplcFdFdVPL,cplcFdFdVPR,             & 
& cplcFuFdVWpL,cplcFuFdVWpR,cplcFdFdVZL,cplcFdFdVZR,cplcFeFeVPL,cplcFeFeVPR,             & 
& cplcFvFeVWpL,cplcFvFeVWpR,cplcFeFeVZL,cplcFeFeVZR,cplcFuFuVGL,cplcFuFuVGR,             & 
& cplcFuFuVPL,cplcFuFuVPR,cplcFuFuVZL,cplcFuFuVZR,cplcFdFucVWpL,cplcFdFucVWpR,           & 
& cplcFvFvVZL,cplcFvFvVZR,cplcFeFvcVWpL,cplcFeFvcVWpR)

Mhh_s = Mhh 
Mhh2_s  = Mhh2   
MAh_s = MAh 
MAh2_s  = MAh2   
Mhh= MhhL 
Mhh2 = Mhh2L 
MAh= MAhL 
MAh2 = MAh2L 
Mhh= Mhh_s 
Mhh2 = Mhh2_s 
MAh= MAh_s 
MAh2 = MAh2_s 

! *****  G minus 2 ***** 

Call Gminus2(1,MAh,MAh2,MFe,MFe2,Mhh,Mhh2,MHp,MHp2,cplcFeFeAhL,cplcFeFeAhR,           & 
& cplcFeFehhL,cplcFeFehhR,cplcFvFeHpL,cplcFvFeHpR,cplcFeFeVPL,cplcFeFeVPR,               & 
& cplcFeFvcHpL,cplcFeFvcHpR,cplHpcHpVP,ae)

Call Gminus2(2,MAh,MAh2,MFe,MFe2,Mhh,Mhh2,MHp,MHp2,cplcFeFeAhL,cplcFeFeAhR,           & 
& cplcFeFehhL,cplcFeFehhR,cplcFvFeHpL,cplcFvFeHpR,cplcFeFeVPL,cplcFeFeVPR,               & 
& cplcFeFvcHpL,cplcFeFvcHpR,cplHpcHpVP,amu)

Call Gminus2(3,MAh,MAh2,MFe,MFe2,Mhh,Mhh2,MHp,MHp2,cplcFeFeAhL,cplcFeFeAhR,           & 
& cplcFeFehhL,cplcFeFehhR,cplcFvFeHpL,cplcFvFeHpR,cplcFeFeVPL,cplcFeFeVPR,               & 
& cplcFeFvcHpL,cplcFeFvcHpR,cplHpcHpVP,atau)


! *****  Lepton EDM ***** 

Call LeptonEDM(1,MAh,MAh2,MFe,MFe2,Mhh,Mhh2,MHp,MHp2,MVWp,MVWp2,MVZ,MVZ2,             & 
& cplcFeFeAhL,cplcFeFeAhR,cplcFeFehhL,cplcFeFehhR,cplcFvFeHpL,cplcFvFeHpR,               & 
& cplcFeFeVPL,cplcFeFeVPR,cplcFvFeVWpL,cplcFvFeVWpR,cplcFeFeVZL,cplcFeFeVZR,             & 
& cplcFeFvcHpL,cplcFeFvcHpR,cplcFeFvcVWpL,cplcFeFvcVWpR,cplHpcHpVP,cplcVWpVPVWp,EDMe)

Call LeptonEDM(2,MAh,MAh2,MFe,MFe2,Mhh,Mhh2,MHp,MHp2,MVWp,MVWp2,MVZ,MVZ2,             & 
& cplcFeFeAhL,cplcFeFeAhR,cplcFeFehhL,cplcFeFehhR,cplcFvFeHpL,cplcFvFeHpR,               & 
& cplcFeFeVPL,cplcFeFeVPR,cplcFvFeVWpL,cplcFvFeVWpR,cplcFeFeVZL,cplcFeFeVZR,             & 
& cplcFeFvcHpL,cplcFeFvcHpR,cplcFeFvcVWpL,cplcFeFvcVWpR,cplHpcHpVP,cplcVWpVPVWp,EDMmu)

Call LeptonEDM(3,MAh,MAh2,MFe,MFe2,Mhh,Mhh2,MHp,MHp2,MVWp,MVWp2,MVZ,MVZ2,             & 
& cplcFeFeAhL,cplcFeFeAhR,cplcFeFehhL,cplcFeFehhR,cplcFvFeHpL,cplcFvFeHpR,               & 
& cplcFeFeVPL,cplcFeFeVPR,cplcFvFeVWpL,cplcFvFeVWpR,cplcFeFeVZL,cplcFeFeVZR,             & 
& cplcFeFvcHpL,cplcFeFvcHpR,cplcFeFvcVWpL,cplcFeFvcVWpR,cplHpcHpVP,cplcVWpVPVWp,EDMtau)


! *****  delta Rho ***** 

sinW2=0.22290_dp 
TW = asin(sqrt(sinW2)) 
mW2=(1._dp-sinW2)*mz2 + 0
g2SM=2._dp*Sqrt(alpha*pi/sinW2) 
g1SM=g2SM*Sqrt(sinW2/(1._dp-sinW2)) 
If (MatchingOrder.gt.-1) Then 
   vSM = Sqrt(mZ2*(1._dp-sinW2)*SinW2/(pi*alpha)) 
 Else
   vSM=1/Sqrt((G_F*Sqrt(2._dp)))
End if
YuSM=0._dp
YdSM=0._dp
YeSM=0._dp
   Do i1=1,3 
      YuSM(i1,i1)=sqrt(2._dp)*mf_u(i1)/vSM 
      YeSM(i1,i1)=sqrt(2._dp)*mf_l(i1)/vSM 
      YdSM(i1,i1)=sqrt(2._dp)*mf_d(i1)/vSM 
    End Do 
Call SetMatchingConditions(g1SM,g2SM,g3SM,YuSM,YdSM,YeSM,vSM,vvSM,g1,g2,              & 
& g3,LambdaVari4,Yu,Yd,Ye,mu2,.False.)

Call SolveTadpoleEquations(g1,g2,g3,LambdaVari4,Yu,Yd,Ye,mu2,vvSM,(/ ZeroC /))

Call TreeMasses(MAh,MAh2,MFd,MFd2,MFe,MFe2,MFu,MFu2,Mhh,Mhh2,MHp,MHp2,MVWp,           & 
& MVWp2,MVZ,MVZ2,TW,ZDR,ZER,ZUR,ZDL,ZEL,ZUL,ZW,ZZ,vvSM,g1,g2,g3,LambdaVari4,             & 
& Yu,Yd,Ye,mu2,GenerationMixing,kont)

Call CouplingsForVectorBosons(g2,g1,TW,vvSM,ZUL,ZDL,ZEL,cplAhcHpVWp,cplhhcHpVWp,      & 
& cplHpcHpVP,cplHpcHpVZ,cplcHpVPVWp,cplcHpVWpVZ,cplHpcHpVPVP,cplHpcHpcVWpVWp,            & 
& cplHpcHpVZVZ,cplAhhhVZ,cplAhHpcVWp,cplAhAhcVWpVWp,cplAhAhVZVZ,cplhhHpcVWp,             & 
& cplhhcVWpVWp,cplhhVZVZ,cplhhhhcVWpVWp,cplhhhhVZVZ,cplcFdFdVPL,cplcFdFdVPR,             & 
& cplcFeFeVPL,cplcFeFeVPR,cplcFuFuVPL,cplcFuFuVPR,cplcgWpgWpVP,cplcgWCgWCVP,             & 
& cplHpcVWpVP,cplcVWpVPVWp,cplcVWpVPVPVWp1,cplcVWpVPVPVWp2,cplcVWpVPVPVWp3,              & 
& cplcFdFdVZL,cplcFdFdVZR,cplcFeFeVZL,cplcFeFeVZR,cplcFuFuVZL,cplcFuFuVZR,               & 
& cplcFvFvVZL,cplcFvFvVZR,cplcgWpgWpVZ,cplcgWCgWCVZ,cplHpcVWpVZ,cplcVWpVWpVZ,            & 
& cplcVWpVWpVZVZ1,cplcVWpVWpVZVZ2,cplcVWpVWpVZVZ3,cplcFdFucVWpL,cplcFdFucVWpR,           & 
& cplcFeFvcVWpL,cplcFeFvcVWpR,cplcgWCgAcVWp,cplcgAgWpcVWp,cplcgZgWpcVWp,cplcgWCgZcVWp,   & 
& cplcVWpcVWpVWpVWp1,cplcVWpcVWpVWpVWp2,cplcVWpcVWpVWpVWp3,cplHpcHpVPVZ,cplcVWpVPVWpVZ1, & 
& cplcVWpVPVWpVZ2,cplcVWpVPVWpVZ3)

Call DeltaRho(MAh,MAh2,MFd,MFd2,MFe,MFe2,MFu,MFu2,Mhh,Mhh2,MHp,MHp2,MVWp,             & 
& MVWp2,MVZ,MVZ2,cplAhAhcVWpVWp,cplAhAhVZVZ,cplAhhhVZ,cplAhHpcVWp,cplcFdFdVZL,           & 
& cplcFdFdVZR,cplcFdFucVWpL,cplcFdFucVWpR,cplcFeFeVZL,cplcFeFeVZR,cplcFeFvcVWpL,         & 
& cplcFeFvcVWpR,cplcFuFuVZL,cplcFuFuVZR,cplcFvFvVZL,cplcFvFvVZR,cplcgAgWpcVWp,           & 
& cplcgWCgAcVWp,cplcgWCgWCVZ,cplcgWCgZcVWp,cplcgWpgWpVZ,cplcgZgWpcVWp,cplcVWpcVWpVWpVWp1,& 
& cplcVWpcVWpVWpVWp2,cplcVWpcVWpVWpVWp3,cplcVWpVPVPVWp1,cplcVWpVPVPVWp2,cplcVWpVPVPVWp3, & 
& cplcVWpVPVWp,cplcVWpVWpVZ,cplcVWpVWpVZVZ1,cplcVWpVWpVZVZ2,cplcVWpVWpVZVZ3,             & 
& cplhhcVWpVWp,cplhhhhcVWpVWp,cplhhhhVZVZ,cplhhHpcVWp,cplhhVZVZ,cplHpcHpcVWpVWp,         & 
& cplHpcHpVZ,cplHpcHpVZVZ,cplHpcVWpVP,cplHpcVWpVZ,dRho)

Call STUparameter(vSM,g1SM,g2SM,g3SM,YuSM,YdSM,YeSM,MAh,MAh2,MFd,MFd2,MFe,            & 
& MFe2,MFu,MFu2,Mhh,Mhh2,MHp,MHp2,MVWp,MVWp2,MVZ,MVZ2,cplAhcHpVWp,cplhhcHpVWp,           & 
& cplHpcHpVP,cplHpcHpVZ,cplcHpVPVWp,cplcHpVWpVZ,cplHpcHpVPVP,cplHpcHpcVWpVWp,            & 
& cplHpcHpVZVZ,cplAhhhVZ,cplAhHpcVWp,cplAhAhcVWpVWp,cplAhAhVZVZ,cplhhHpcVWp,             & 
& cplhhcVWpVWp,cplhhVZVZ,cplhhhhcVWpVWp,cplhhhhVZVZ,cplcFdFdVPL,cplcFdFdVPR,             & 
& cplcFeFeVPL,cplcFeFeVPR,cplcFuFuVPL,cplcFuFuVPR,cplcgWpgWpVP,cplcgWCgWCVP,             & 
& cplHpcVWpVP,cplcVWpVPVWp,cplcVWpVPVPVWp1,cplcVWpVPVPVWp2,cplcVWpVPVPVWp3,              & 
& cplcFdFdVZL,cplcFdFdVZR,cplcFeFeVZL,cplcFeFeVZR,cplcFuFuVZL,cplcFuFuVZR,               & 
& cplcFvFvVZL,cplcFvFvVZR,cplcgWpgWpVZ,cplcgWCgWCVZ,cplHpcVWpVZ,cplcVWpVWpVZ,            & 
& cplcVWpVWpVZVZ1,cplcVWpVWpVZVZ2,cplcVWpVWpVZVZ3,cplcFdFucVWpL,cplcFdFucVWpR,           & 
& cplcFeFvcVWpL,cplcFeFvcVWpR,cplcgWCgAcVWp,cplcgAgWpcVWp,cplcgZgWpcVWp,cplcgWCgZcVWp,   & 
& cplcVWpcVWpVWpVWp1,cplcVWpcVWpVWpVWp2,cplcVWpcVWpVWpVWp3,cplHpcHpVPVZ,cplcVWpVPVWpVZ1, & 
& cplcVWpVPVWpVZ2,cplcVWpVPVWpVZ3,Spar,Tpar,Upar)

If (WriteParametersAtQ) Then 
scalein = SetRenormalizationScale(160._dp**2) 
Else 
scalein = SetRenormalizationScale(scale_save**2) 
End if 
mz2 = mzsave**2 
mz = mzsave 
End subroutine CalculateLowEnergyConstraints 
 
 
End Module LowEnergy_TNMA 
