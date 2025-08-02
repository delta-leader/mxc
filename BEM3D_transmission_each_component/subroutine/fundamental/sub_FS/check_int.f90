subroutine check_int(xi,zeta,eta,Ixz)
   implicit none
   integer::i,j,k,ng,integ
   real(kind(0d0))::lR,rr,rr3,rr5,area
   real(kind(0d0)),dimension(3)::y1,y2,y3,yco,nvec
   real(kind(0d0)),intent(in)::eta,xi(4),zeta(4)
   real(kind(0d0)),intent(out)::Ixz(5,10)
   real(kind(0d0)),dimension(:),allocatable::gzi1,gzi2,gzi3,wi
!=======================================================================
interface
subroutine Gauss_tri(n,gzi1,gzi2,gzi3,wi)
   implicit none
   integer,parameter::n_ava=7
   integer,dimension(n_ava),parameter::ni_ava=(/3,4,7,13,27,48,79/)
   integer,intent(in)::n
   real(kind(0d0)),dimension(:),allocatable,intent(inout)::gzi1,gzi2,gzi3,wi
   end subroutine Gauss_tri
end interface
!=======================================================================
   y1(1)=xi(1)
   y1(2)=zeta(1)
   y1(3)=eta
   y2(1)=xi(2)
   y2(2)=zeta(2)
   y2(3)=eta
   y3(1)=xi(3)
   y3(2)=zeta(3)
   y3(3)=eta
   call out_product(nvec,y2-y1,y3-y1)
   area=0.5d0*sqrt(dot_product(nvec,nvec))
   integ=13
   call Gauss_tri(integ,gzi1,gzi2,gzi3,wi)
   Ixz=0.d0
   do ng=1,integ
      yco(:)=y1(:)*gzi1(ng)+y2(:)*gzi2(ng)+y3(:)*gzi3(ng)
      lR=sqrt(yco(1)**2+yco(2)**2)
      rr=sqrt(lR**2+eta**2)
      rr3=rr*rr*rr
      rr5=rr3*rr*rr
      Ixz(1,1)=Ixz(1,1)+(1.d0/rr)&
         &*area*wi(ng)
      Ixz(1,2)=Ixz(1,2)+(yco(1)/rr)&
         &*area*wi(ng)
      Ixz(1,3)=Ixz(1,3)+(yco(2)/rr)&
         &*area*wi(ng)
      !---
      Ixz(3,1)=Ixz(3,1)+(1.d0/rr3)*yco(3)&
         &*area*wi(ng)
      Ixz(3,2)=Ixz(3,2)+(yco(1)/rr3)&
         &*area*wi(ng)
      Ixz(3,3)=Ixz(3,3)+(yco(2)/rr3)&
         &*area*wi(ng)
      Ixz(3,4)=Ixz(3,4)+(yco(1)*yco(1)/rr3)&
         &*area*wi(ng)
      Ixz(3,5)=Ixz(3,5)+(yco(1)*yco(2)/rr3)&
         &*area*wi(ng)
      Ixz(3,6)=Ixz(3,6)+(yco(2)*yco(2)/rr3)&
         &*area*wi(ng)
      Ixz(3,7)=Ixz(3,7)+(yco(1)*yco(1)*yco(1)/rr3)&
         &*area*wi(ng)
      Ixz(3,8)=Ixz(3,8)+(yco(1)*yco(1)*yco(2)/rr3)&
         &*area*wi(ng)
      Ixz(3,9)=Ixz(3,9)+(yco(1)*yco(2)*yco(2)/rr3)&
         &*area*wi(ng)
      Ixz(3,10)=Ixz(3,10)+(yco(2)*yco(2)*yco(2)/rr3)&
         &*area*wi(ng)
      !---
      Ixz(5,1)=Ixz(5,1)+(1.d0/rr5)*(yco(3)**3)&
         &*area*wi(ng)
      Ixz(5,2)=Ixz(5,2)+(yco(1)/rr5)&
         &*area*wi(ng)
      Ixz(5,3)=Ixz(5,3)+(yco(2)/rr5)&
         &*area*wi(ng)
      Ixz(5,4)=Ixz(5,4)+(yco(1)*yco(1)/rr5)*yco(3)&
         &*area*wi(ng)
      Ixz(5,5)=Ixz(5,5)+(yco(1)*yco(2)/rr5)&
         &*area*wi(ng)
      Ixz(5,6)=Ixz(5,6)+(yco(2)*yco(2)/rr5)*yco(3)&
         &*area*wi(ng)
      Ixz(5,7)=Ixz(5,7)+(yco(1)*yco(1)*yco(1)/rr5)&
         &*area*wi(ng)
      Ixz(5,8)=Ixz(5,8)+(yco(1)*yco(1)*yco(2)/rr5)&
         &*area*wi(ng)
      Ixz(5,9)=Ixz(5,9)+(yco(1)*yco(2)*yco(2)/rr5)&
         &*area*wi(ng)
      Ixz(5,10)=Ixz(5,10)+(yco(2)*yco(2)*yco(2)/rr5)&
         &*area*wi(ng)
   end do
   end subroutine check_int
