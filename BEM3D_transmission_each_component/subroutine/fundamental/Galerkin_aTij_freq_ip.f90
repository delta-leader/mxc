subroutine linear_x_aTij_freq_ip(om,zaTij,elx,ely,sing,ip)
   !\int \phi_{ip} \int aTij dSy dSx
   use BEM3d
   use math_cst
   use elast_parameter
   use struct_type
   implicit none
   integer::i,j,ng,integ
   real(kind(0d0))::cst
   real(kind(0d0)),dimension(3)::phix,xco,x1,x2,x3
   real(kind(0d0)),dimension(:),allocatable::gzi1,gzi2,gzi3,wi
   complex(kind(0d0)),dimension(3,3)::ten2
   !--------------------------------
   integer,intent(in)::sing,ip
   real(kind(0d0)),intent(in)::om
   complex(kind(0d0)),dimension(3,3),intent(out)::zaTij
   type(element),intent(in)::elx,ely
!=======================================================================
!=======================================================================
!=======================================================================
interface
subroutine elast3d_aTij_l(om,zaTij,xco,nx,ely,sing)
   !\int aTij dSy
   use BEM3d
   use elast_parameter
   use struct_type
   use math_cst
   implicit none
   integer,intent(in)::sing
   real(kind(0d0)),intent(in)::om,xco(3),nx(3)
   complex(kind(0d0)),intent(out)::zaTij(3,3)
   type(element),intent(in)::ely
   end subroutine elast3d_aTij_l
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
   integ=3
   call Gauss_tri(integ,gzi1,gzi2,gzi3,wi)
   x1(:)=node(elx%ind(1))%xc(:)
   x2(:)=node(elx%ind(2))%xc(:)
   x3(:)=node(elx%ind(3))%xc(:)
   zaTij=0.d0
   do ng=1,integ
      xco(:)=x1(:)*gzi1(ng)+x2(:)*gzi2(ng)+x3(:)*gzi3(ng)
      call cal_phiy(x1,x2,x3,xco,phix)
      call elast3d_aTij_l(om,ten2,xco,elx%nvec,ely,sing)
      cst=wi(ng)*elx%Jgg*phix(ip)
      zaTij(:,:)=zaTij(:,:)+ten2(:,:)*cst
   end do !ng
   end subroutine linear_x_aTij_freq_ip
!=======================================================================
!=======================================================================
!=======================================================================
subroutine constant_x_aTij_freq(om,zaTij,elx,ely,sing)
   !\int \phi_{ip} \int aTij dSy dSx
   use BEM3d
   use math_cst
   use elast_parameter
   use struct_type
   implicit none
   integer::i,j,ng,integ
   real(kind(0d0))::cst
   real(kind(0d0)),dimension(3)::xco,x1,x2,x3
   real(kind(0d0)),dimension(:),allocatable::gzi1,gzi2,gzi3,wi
   complex(kind(0d0)),dimension(3,3)::ten2
   !--------------------------------
   integer,intent(in)::sing
   real(kind(0d0)),intent(in)::om
   complex(kind(0d0)),dimension(3,3),intent(out)::zaTij
   type(element),intent(in)::elx,ely
!=======================================================================
!=======================================================================
interface
subroutine elast3d_aTij_l(om,zaTij,xco,nx,ely,sing)
   !\int aTij dSy
   use BEM3d
   use elast_parameter
   use struct_type
   use math_cst
   implicit none
   integer,intent(in)::sing
   real(kind(0d0)),intent(in)::om,xco(3),nx(3)
   complex(kind(0d0)),intent(out)::zaTij(3,3)
   type(element),intent(in)::ely
   end subroutine elast3d_aTij_l
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
   integ=3
   call Gauss_tri(integ,gzi1,gzi2,gzi3,wi)
   x1(:)=node(elx%ind(1))%xc(:)
   x2(:)=node(elx%ind(2))%xc(:)
   x3(:)=node(elx%ind(3))%xc(:)
   zaTij=0.d0
   do ng=1,integ
      xco(:)=x1(:)*gzi1(ng)+x2(:)*gzi2(ng)+x3(:)*gzi3(ng)
      call elast3d_aTij_l(om,ten2,xco,elx%nvec,ely,sing)
      cst=wi(ng)*elx%Jgg
      zaTij(:,:)=zaTij(:,:)+ten2(:,:)*cst
   end do !ng
   end subroutine constant_x_aTij_freq
