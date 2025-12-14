subroutine mass_mat_ll(elx,mass_mat)
   use BEM3d
   use struct_type
   implicit none
   integer::i,ng,integ,ip,iq
   real(kind(0d0)),dimension(3)::xco,x1,x2,x3,phix
   real(kind(0d0)),dimension(:),allocatable::gzi1,gzi2,gzi3,wi
   real(kind(0d0)),dimension(3,3),intent(out)::mass_mat
   type(element),intent(in)::elx
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
   x1(:)=node(elx%ind(1))%xc(:)
   x2(:)=node(elx%ind(2))%xc(:)
   x3(:)=node(elx%ind(3))%xc(:)
   mass_mat=0.d0
   do ng=1,integ
      do i=1,3
         xco(i)=x1(i)*gzi1(ng)+x2(i)*gzi2(ng)+x3(i)*gzi3(ng)
      end do
      call cal_phiy(x1,x2,x3,xco,phix)
      do ip=1,3
         do iq=1,3
            mass_mat(ip,iq)=mass_mat(ip,iq)+phix(ip)*phix(iq)*wi(ng)*elx%Jgg
         end do
      end do
   end do
   end subroutine mass_mat_ll
!=======================================================================
!=======================================================================
!=======================================================================
subroutine mass_mat_lc(elx,mass_mat)
   use BEM3d
   use struct_type
   implicit none
   integer::i,ng,integ,ip,iq
   real(kind(0d0)),dimension(3)::xco,x1,x2,x3,phix
   real(kind(0d0)),dimension(:),allocatable::gzi1,gzi2,gzi3,wi
   real(kind(0d0)),dimension(3),intent(out)::mass_mat
   type(element),intent(in)::elx
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
   x1(:)=node(elx%ind(1))%xc(:)
   x2(:)=node(elx%ind(2))%xc(:)
   x3(:)=node(elx%ind(3))%xc(:)
   mass_mat=0.d0
   do ng=1,integ
      do i=1,3
         xco(i)=x1(i)*gzi1(ng)+x2(i)*gzi2(ng)+x3(i)*gzi3(ng)
      end do
      call cal_phiy(x1,x2,x3,xco,phix)
      do ip=1,3
         mass_mat(ip)=mass_mat(ip)+phix(ip)*wi(ng)*elx%Jgg
      end do
   end do
   end subroutine mass_mat_lc