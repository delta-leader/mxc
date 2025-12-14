subroutine freq_inc_traction(elx,om,tout)
   use BEM3d
   use math_cst
   use elast_parameter
   use struct_type
   implicit none
   integer::i,j,k,ip,ng,integ
   real(kind(0d0)),intent(in)::om
   real(kind(0d0))::kk,dia_norm,del,xi_x,zeta_x
   real(kind(0d0)),dimension(3)::x1,x2,x3,xco,phix,nvec,pvec,dvec
   real(kind(0d0)),dimension(:),allocatable::gzi1,gzi2,gzi3,wi
   complex(kind(0d0)),dimension(3,3),intent(out)::tout
   complex(kind(0d0)),dimension(3,3)::ud
   type(element),intent(in)::elx
!================================================================
!================================================================
interface
subroutine cal_phiy(y1,y2,y3,yco,phiy)
   implicit none
   real(kind(0d0)),dimension(3),intent(in)::y1,y2,y3,yco
   real(kind(0d0)),dimension(3),intent(out)::phiy
   end subroutine cal_phiy
end interface
!================================================================
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
   integ=3
   call Gauss_tri(integ,gzi1,gzi2,gzi3,wi)
   pvec(1)=cos(theta_in)
   pvec(2)=0.d0
   pvec(3)=sin(theta_in)
   select case(id_inc)
   case(0)
      dvec(:)=pvec(:)
      kk=om/cl(1)
   case(1)
      dvec(1)=-sin(theta_in)
      dvec(2)=0.d0
      dvec(3)=cos(theta_in)
      kk=om/ct(1)
   case(2)
      dvec(:)=0.d0
      dvec(2)=1.d0
      kk=om/ct(1)
   end select
   x1(:)=node(elx%ind(1))%xc(:)
   x2(:)=node(elx%ind(2))%xc(:)
   x3(:)=node(elx%ind(3))%xc(:)
   nvec(:)=elx%nvec(:)
   tout(:,:)=0.d0
   do ng=1,integ
      xco(:)=x1(:)*gzi1(ng)+x2(:)*gzi2(ng)+x3(:)*gzi3(ng)
      call cal_phiy(x1,x2,x3,xco,phix)
      do i=1,3
         do j=1,3
            ud(i,j)=ii*u0*kk*dvec(i)*pvec(j)*cdexp(ii*kk*(dot_product(pvec,xco)))
         end do
      end do
      do i=1,3
         do k=1,3
            do ip=1,3
               tout(i,ip)=tout(i,ip)+(lam*ud(k,k)*nvec(i)+mu*(ud(i,k)+ud(k,i))*nvec(k))*phix(ip)*wi(ng)*elx%Jgg
            end do
         end do
      end do
   end do
   end subroutine freq_inc_traction
!=======================================================================
!=======================================================================
!=======================================================================
subroutine freq_inc_traction_constant_x(elx,om,tout)
   use BEM3d
   use math_cst
   use elast_parameter
   use struct_type
   implicit none
   integer::i,j,k,ng,integ
   real(kind(0d0)),intent(in)::om
   real(kind(0d0))::kk,dia_norm,del,xi_x,zeta_x
   real(kind(0d0)),dimension(3)::x1,x2,x3,xco,nvec,pvec,dvec
   real(kind(0d0)),dimension(:),allocatable::gzi1,gzi2,gzi3,wi
   complex(kind(0d0)),dimension(3),intent(out)::tout
   complex(kind(0d0)),dimension(3,3)::ud
   type(element),intent(in)::elx
!================================================================
!================================================================
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
   integ=3
   call Gauss_tri(integ,gzi1,gzi2,gzi3,wi)
   pvec(1)=cos(theta_in)
   pvec(2)=0.d0
   pvec(3)=sin(theta_in)
   select case(id_inc)
   case(0)
      dvec(:)=pvec(:)
      kk=om/cl(1)
   case(1)
      dvec(1)=-sin(theta_in)
      dvec(2)=0.d0
      dvec(3)=cos(theta_in)
      kk=om/ct(1)
   case(2)
      dvec(:)=0.d0
      dvec(2)=1.d0
      kk=om/ct(1)
   end select
   x1(:)=node(elx%ind(1))%xc(:)
   x2(:)=node(elx%ind(2))%xc(:)
   x3(:)=node(elx%ind(3))%xc(:)
   nvec(:)=elx%nvec(:)
   tout(:)=0.d0
   do ng=1,integ
      xco(:)=x1(:)*gzi1(ng)+x2(:)*gzi2(ng)+x3(:)*gzi3(ng)
      do i=1,3
         do j=1,3
            ud(i,j)=ii*u0*kk*dvec(i)*pvec(j)*cdexp(ii*kk*(dot_product(pvec,xco)))
         end do
      end do
      do i=1,3
         do k=1,3
            tout(i)=tout(i)+(lam*ud(k,k)*nvec(i)+mu*(ud(i,k)+ud(k,i))*nvec(k))*wi(ng)*elx%Jgg
         end do
      end do
   end do
   end subroutine freq_inc_traction_constant_x
!=======================================================================
!=======================================================================
!=======================================================================
subroutine freq_inc_traction_noint(x,nvec,om,tout)
   use BEM3d
   use math_cst
   use elast_parameter
   use struct_type
   implicit none
   integer::i,j,k
   real(kind(0d0)),intent(in)::om,x(3),nvec(3)
   real(kind(0d0))::kk
   real(kind(0d0)),dimension(3)::pvec,dvec
   complex(kind(0d0)),dimension(3),intent(out)::tout
   complex(kind(0d0)),dimension(3,3)::ud
!=======================================================================
   pvec(1)=cos(theta_in)
   pvec(2)=0.d0
   pvec(3)=sin(theta_in)
   select case(id_inc)
   case(0)
      dvec(:)=pvec(:)
      kk=om/cl(1)
   case(1)
      dvec(1)=-sin(theta_in)
      dvec(2)=0.d0
      dvec(3)=cos(theta_in)
      kk=om/ct(1)
   case(2)
      dvec(:)=0.d0
      dvec(2)=1.d0
      kk=om/ct(1)
   end select
   tout(:)=0.d0
   do i=1,3
      do j=1,3
         ud(i,j)=ii*u0*kk*dvec(i)*pvec(j)*exp(ii*kk*(dot_product(pvec,x)))
      end do
   end do
   do i=1,3
      do k=1,3
         tout(i)=tout(i)+(lam*ud(k,k)*nvec(i)+mu*(ud(i,k)+ud(k,i))*nvec(k))
      end do
   end do
   end subroutine freq_inc_traction_noint