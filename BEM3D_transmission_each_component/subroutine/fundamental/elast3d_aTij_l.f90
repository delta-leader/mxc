subroutine elast3d_aTij_l(om,zaTij,xco,nx,ely,sing)
   !\int aTij dSy
   use BEM3d
   use elast_parameter
   use struct_type
   use math_cst
   implicit none
   integer::i,j,k,ngy
   integer::integ
   real(kind(0d0)),dimension(3)::yco,y1,y2,y3
   real(kind(0d0)),dimension(3,3,3)::aaa,bbb
   real(kind(0d0)),dimension(3,3,3,3,3)::sigma_ikjd
   real(kind(0d0)),dimension(:),allocatable::gzi1,gzi2,gzi3,wi
   complex(kind(0d0))::zs
   complex(kind(0d0)),dimension(3,3,3)::dsigma,zsigma_ijk
   !--------------------------------
   integer,intent(in)::sing
   real(kind(0d0)),intent(in)::om,xco(3),nx(3)
   complex(kind(0d0)),intent(out)::zaTij(3,3)
   type(element),intent(in)::ely
!=======================================================================
!=======================================================================
interface
subroutine static_Sigma_d_Uij(xx,y1,y2,y3,on_id,sigma_ijk,sigma_ijk_d,Uij)
   use math_cst
   use elast_parameter
   implicit none
   integer,intent(in)::on_id  !1:x is on the element, 0:not
   real(kind=8),dimension(3),intent(in)::xx,y1,y2,y3
   real(kind=8),dimension(3,3,3),intent(out)::Uij,sigma_ijk
   real(kind=8),dimension(3,3,3,3,3),intent(out)::sigma_ijk_d
   end subroutine static_Sigma_d_Uij
end interface
!=======================================================================
interface
subroutine elastSigma_dynamic(x,y,cl,ct,zs,dsigma)
   use math_cst, only: pi_4
   use elast_parameter
   implicit none
   real(kind(0d0)),intent(in)::cl,ct
   real(kind(0d0)),dimension(3),intent(in)::x,y
   complex(kind(0d0)),intent(in)::zs
   complex(kind(0d0)),dimension(3,3,3),intent(out)::dsigma
   end subroutine elastSigma_dynamic
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
   integ=4
   call Gauss_tri(integ,gzi1,gzi2,gzi3,wi)
   zs=-(0.0d0,1.0d0)*om
!-------------------------------------------
   y1(:)=node(ely%ind(1))%xc(:)
   y2(:)=node(ely%ind(2))%xc(:)
   y3(:)=node(ely%ind(3))%xc(:)
   !initialize
   zsigma_ijk=0.d0
   do ngy=1,integ
      do i=1,3
         yco(i)=y1(i)*gzi1(ngy)+y2(i)*gzi2(ngy)+y3(i)*gzi3(ngy)
      end do
      call elastSigma_dynamic(xco,yco,cl(im),ct(im),zs,dsigma)
      zsigma_ijk(:,:,:)=zsigma_ijk(:,:,:)+dsigma(:,:,:)*wi(ngy)*ely%Jgg
   end do !ngy
   !--- static ---
   call static_Sigma_d_Uij(xco,y1,y2,y3,sing,bbb,sigma_ikjd,aaa)
   zsigma_ijk(:,:,:)=zsigma_ijk(:,:,:)+bbb(:,:,:)
   !------- calculate zaTij -------
   zaTij=0.d0
   do i=1,3
      do j=1,3
         do k=1,3
            zaTij(i,j)=zaTij(i,j)-nx(k)*zsigma_ijk(j,k,i)
         end do
      end do
   end do
   end subroutine elast3d_aTij_l