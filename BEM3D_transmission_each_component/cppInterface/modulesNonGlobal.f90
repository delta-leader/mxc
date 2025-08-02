module struct_type_fixed_len_node_mod
  use iso_c_binding
  implicit none
  !==================
  !=== for 3D BEM ===
  !==================
  type, bind(c) :: element
     integer(c_int) :: id_e, nedge, nnear
     integer(c_int),dimension(3) ::ind,id !id=1:edge
     real(c_double) :: Jgg
     real(c_double),dimension(3) :: xc, nvec, mvec, svec
     complex(c_double_complex),dimension(3) :: t
  end type element

  type, bind(c) :: nodal_point
     integer(c_int) :: ident, nel
     !integer,dimension(:,:),allocatable::iel
     integer(c_int) :: iel(10, 2)
     real(c_double),dimension(3) :: xc,nvec,svec
     complex(c_double_complex),dimension(3) :: u
  end type nodal_point

  type infield
     real(kind(0d0)),dimension(3)::xc
     complex(kind(0d0)),dimension(3)::uin,usc
  end type infield
end module struct_type_fixed_len_node_mod
!-------------------------------------------
module bem3d_small_mod
  use struct_type_fixed_len_node_mod
  implicit none

  integer::nthread,id_bie,im
  !integer::n_mat,nnode,nel,ninf,id_inc,ix1_min,ix1_0,ix1_max,nel3,nnode3
  integer::n_mat,ninf,id_inc,ix1_min,ix1_0,ix1_max,nel3,nnode3
  real(kind(0d0))::u0,theta_in,rad
  !   type(element),dimension(:),allocatable::el
  type(infield),dimension(:),allocatable::xinf
  !   type(nodal_point),dimension(:),allocatable::node
  integer::icheck

  interface
     subroutine out_product(xx,aa,bb)!   xx=aa cross bb
       implicit none
       real(kind(0d0)),dimension(3),intent(in)::aa,bb
       real(kind(0d0)),dimension(3),intent(out)::xx
       !    xx(1)=aa(2)*bb(3)-aa(3)*bb(2)
       !    xx(2)=aa(3)*bb(1)-aa(1)*bb(3)
       !    xx(3)=aa(1)*bb(2)-aa(2)*bb(1)
     end subroutine out_product
  end interface
end module BEM3d_Small_mod
!-------------------------------------------
module elast_parameter_struct_mod
  type :: elast_parameter_struct
     real(kind(0d0))::mu,lam,nu
     real(kind(0d0)),dimension(:,:)::delta(3,3)
     real(kind(0d0)),dimension(:,:,:)::permut(3,3,3)
     real(kind(0d0)),dimension(3,3,3,3)::Cijkl,pet,delta2 !Cijkl/mu
     real(kind(0d0)),dimension(3,3,3,3)::const2
     real(kind(0d0)),dimension(3,3,3,3,3,3)::const1,const_f
     real(kind(0d0)),dimension(3,3,3,3,3,3)::delta3
     real(kind(0d0)) :: cl,ct,rho
   contains
     procedure :: set_elast_para
  end type elast_parameter_struct

contains
  subroutine set_elast_para(self, cl, ct, rho)
    use math_cst, only: pi

    class(elast_parameter_struct), intent(inout) :: self
    real(kind(0d0)),intent(in)::cl,ct,rho

    integer::i,j,k,l,p,q,r,a,b,c,d,s,t,n,m
    real(kind(0d0))::clct2,cst
    real(kind(0d0)),dimension(3,3,3,3)::ccc
    real(kind(0d0)),dimension(3,3,3,3,3)::ddd

    self%mu = rho*(ct**2)
    self%lam = rho*(cl**2)-2.d0*self%mu
    self%nu = self%lam*0.5d0/(self%lam + self%mu)
    self%permut(:,:,:) = 0.d0
    self%permut(1,2,3) = 1.d0
    self%permut(3,1,2) = 1.d0
    self%permut(2,3,1) = 1.d0
    self%permut(2,1,3) = -1.d0
    self%permut(3,2,1) = -1.d0
    self%permut(1,3,2) = -1.d0
    self%delta(:,:) = 0.d0
    self%delta(1,1) = 1.d0
    self%delta(2,2) = 1.d0
    self%delta(3,3) = 1.d0

    clct2 = (cl/ct)**2 - 2.d0

    do k=1,3
       do j=1,3
          do p=1,3
             do q=1,3
                self%Cijkl(k,j,p,q) = clct2*self%delta(k,j)*self%delta(p,q) + self%delta(k,p)*self%delta(j,q) + self%delta(k,q)*self%delta(j,p)
             end do
          end do
       end do
    end do
    self%pet(:,:,:,:) = 0.d0
    do i=1,3
       do j=1,3
          do k=1,3
             do l=1,3
                do r=1,3
                   self%pet(i,j,k,l) = self%pet(i,j,k,l) + self%permut(r,i,j)*self%permut(r,k,l)
                end do
             end do
          end do
       end do
    end do
    do i=1,3
       do j=1,3
          do k=1,3
             do l=1,3
                self%delta2(i,j,k,l) = self%delta(i,j)*self%delta(k,l)
                do p=1,3
                   do q=1,3
                      self%delta3(i,j,k,l,p,q) = self%delta(i,j)*self%delta(k,l)*self%delta(p,q)
                   end do
                end do
             end do
          end do
       end do
    end do

    cst = 0.25d0/(pi*(self%lam + self%mu)*self%mu) !normalized by mu

    do a=1,3
       do b=1,3
          do j=1,3
             do k=1,3
                ccc(a,b,j,k)= -cst*self%mu*(2.0d0*self%lam*self%delta2(a,b,j,k)&
                     & + (self%lam + 2.0d0*self%mu)*(self%delta2(a,k,b,j) + self%delta2(b,k,a,j)))
                self%const2(a,b,j,k) = cst*self%mu*(self%lam*self%delta2(a,b,j,k) + self%mu*(self%delta2(a,j,b,k) + self%delta2(a,k,b,j)))
                do p=1,3
                   do q=1,3
                      self%const1(a,b,j,k,p,q) = -cst*((self%lam**2)*self%delta3(a,b,j,k,p,q)&
                           & + 2.0d0*self%lam*self%mu*(self%delta3(a,b,j,p,k,q) + self%delta3(j,k,a,p,b,q))&
                           & + (self%mu**2)*(self%delta3(a,p,j,q,b,k) + self%delta3(b,p,k,q,a,j) + self%delta3(a,p,k,q,b,j) + self%delta3(b,p,j,q,a,k)))
                   end do
                end do
             end do
          end do
       end do
    end do
    ddd = 0.0d0; self%const_f = 0.0d0
    do b=1,3
       do d=1,3
          do s=1,3
             do m=1,3
                do n=1,3
                   do t=1,3
                      ddd(b,d,m,n,t) = ddd(b,d,m,n,t) + self%permut(b,d,s)*ccc(m,s,n,t)
                   end do
                end do
             end do
          end do
       end do
    end do
    do b=1,3
       do d=1,3
          do m=1,3
             do n=1,3
                do t=1,3
                   do j=1,3
                      do p=1,3
                         self%const_f(b,d,m,t,j,p) = self%const_f(b,d,m,t,j,p) + self%permut(j,p,n)*ddd(b,d,m,n,t)
                      end do
                   end do
                end do
             end do
          end do
       end do
    end do
  end subroutine set_elast_para
end module elast_parameter_struct_mod
!-----------------------------------------------------
module elast_parameter_struct_mod_global
  use elast_parameter_struct_mod
  implicit none

  type(elast_parameter_struct), save :: elout
  type(elast_parameter_struct), save :: elin(10)
end module elast_parameter_struct_mod_global
