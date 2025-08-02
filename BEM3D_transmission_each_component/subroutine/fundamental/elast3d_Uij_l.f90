subroutine elast3d_Uij_l(om,zUij,xco,ely,sing)
   !zUij(i,j)=\int mu*Uij dSy
   !\phi_{iq} is piecewise linear shape function.
   !sing=1:xco is on ely; 0:no
   use BEM3D
   use elast_parameter
   use struct_type
   use math_cst
   implicit none
   integer::i,j,ng,integ
   real(kind(0d0))::cst
   real(kind(0d0)),dimension(3)::yco,y1,y2,y3,phiy
   real(kind(0d0)),dimension(3,3)::ust
   real(kind(0d0)),dimension(3,3,3)::tst
   real(kind(0d0)),dimension(:),allocatable::gzi1,gzi2,gzi3,wi
   complex(kind(0d0))::zs
   complex(kind(0d0)),dimension(3,3)::zu
   !--------------------------------
   integer,intent(in)::sing
   real(kind(0d0)),intent(in)::om,xco(3)
   complex(kind(0d0)),intent(out)::zUij(3,3)
   type(element),intent(in)::ely
!=======================================================================
!=======================================================================
interface
subroutine elastUij_dynamic(x,y,nvec,zs,duij)
   use math_cst, only: pi_4
   use BEM3D
   implicit none
   integer::i,j
   real(kind(0d0)),dimension(3),intent(in)::x,y,nvec
   complex(kind(0d0)),intent(in)::zs
   complex(kind(0d0)),dimension(3,3),intent(out)::duij
   end subroutine elastUij_dynamic
end interface
!=======================================================================
interface
subroutine elast3d_UTij_static(xx,y1,y2,y3,on_id,nvec,ust,tst)
   !ust(i,j)= \mu \int Uij^{st} (x,y) dS_y
   !tst(i,j,iq)=\int Tij^{st} (x,y) phi_{iq}(y) dS_y
   use math_cst
   use elast_parameter
   implicit none
   integer,intent(in)::on_id  !1:x is on the element, 0:not
   real(kind(0d0)),dimension(3),intent(in)::xx,y1,y2,y3,nvec
   real(kind(0d0)),dimension(3,3),intent(out)::ust
   real(kind(0d0)),dimension(3,3,3),intent(out)::tst
   end subroutine elast3d_UTij_static
end interface
!=======================================================================
interface
subroutine cal_phiy(y1,y2,y3,yco,phiy)
   implicit none
   real(kind=8),dimension(3),intent(in)::y1,y2,y3,yco
   real(kind=8),dimension(3),intent(out)::phiy
   end subroutine cal_phiy
end interface
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
   zs=-(0.0d0,1.0d0)*om
!-------------------------------------------
   y1(:)=node(ely%ind(1))%xc(:)
   y2(:)=node(ely%ind(2))%xc(:)
   y3(:)=node(ely%ind(3))%xc(:)
   !--- static part ---
   call elast3d_UTij_static(xco,y1,y2,y3,sing,ely%nvec,ust,tst)
   zUij(:,:)=ust(:,:)
   !--- dynamic part ---
   integ=4
   call Gauss_tri(integ,gzi1,gzi2,gzi3,wi)
   do ng=1,integ
      yco(:)=y1(:)*gzi1(ng)+y2(:)*gzi2(ng)+y3(:)*gzi3(ng)
      call cal_phiy(y1,y2,y3,yco,phiy)
      call elastUij_dynamic(xco,yco,ely%nvec,zs,zu)
      cst=ely%Jgg*wi(ng)
      do i=1,3
         do j=1,3
            zUij(i,j)=zUij(i,j)+zu(i,j)*cst
         end do
      end do
   end do
   end subroutine elast3d_Uij_l
!=============================================================================
!=============================================================================
!=============================================================================
subroutine elastUij_dynamic(x,y,nvec,zs,duij)
   use math_cst, only: pi_4
   use BEM3D
   use elast_parameter
   implicit none
   integer::i,j
   real(kind(0d0))::rr(4),r_y(3)
   complex(kind(0d0))::zsl(4),zst(4),zcst
   complex(kind(0d0))::zu_part(2),zal_L,zal_T,zslr,zstr,zbe_L,zbe_T
   real(kind(0d0)),dimension(3),intent(in)::x,y,nvec
   complex(kind(0d0)),intent(in)::zs
   complex(kind(0d0)),dimension(3,3),intent(out)::duij
!=============================================================================
interface
subroutine Exp_sr_series_BEM(zal_L,zal_T,zbe_L,zbe_T,zslr,zstr)
   ! calculate ( e^{-sr} -1 +sr -(sr)^2/2 ) as zal
   ! calculate ( e^{-sr} -1 +sr -(sr)^2/2 +(sr)^3/6 ) as zbe
   implicit none
   complex(kind(0d0)),intent(in)::zslr,zstr
   complex(kind(0d0)),intent(out)::zal_L,zal_T,zbe_L,zbe_T
   end subroutine Exp_sr_series_BEM
end interface
!=============================================================================
   rr(1)=sqrt(dot_product(x-y,x-y))
   r_y(:)=(y(:)-x(:))/rr(1)
   zsl(1)=zs/cl(im)
   zst(1)=zs/ct(im)
   do i=1,3
      rr(i+1)=rr(i)*rr(1)
      zsl(i+1)=zsl(i)*zsl(1)
      zst(i+1)=zst(i)*zst(1)
   end do
   zslr=zsl(1)*rr(1)
   zstr=zst(1)*rr(1)
   call Exp_sr_series_BEM(zal_L,zal_T,zbe_L,zbe_T,zslr,zstr)
   zal_L=zal_L/rr(3)
   zal_T=zal_T/rr(3)
   zu_part(1)=(zsl(3)-zst(3))/2.d0+rr(1)*(zsl(4)-zst(4))/2.d0&
      &+(3.d0+3.d0*zslr+zsl(2)*rr(2))*zal_L&
      &-(3.d0+3.d0*zstr+zst(2)*rr(2))*zal_T
   zu_part(2)=(zsl(3)+zst(3))/2.d0-zst(4)*rr(1)/2.d0&
      &+(1.d0+zslr)*zal_L&
      &-(1.d0+zstr+zst(2)*rr(2))*zal_T
   zcst=1.d0/(zst(2)*pi_4)
   do i=1,3
      do j=1,3
         duij(i,j)=zcst*( zu_part(1)*r_y(i)*r_y(j)-zu_part(2)*delta(i,j) )
      end do
   end do
   end subroutine elastUij_dynamic
