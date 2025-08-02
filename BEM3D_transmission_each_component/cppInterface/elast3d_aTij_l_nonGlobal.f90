module elast3d_atij_entrywise_mod
  use iso_c_binding
  implicit none
contains
  !-------------------------------------
  !subroutine elast3d_aTij_l(om,zaTij,xco,nx,ely,sing)
  subroutine elast3d_aTij_l_nonGlobal(xco, nx, y_nodals, yNumNodeBasis, ely, om, elastp, sing, zaTij)
    !\int aTij dSy
    use BEM3d_small_mod
    use elast_parameter_struct_mod
    use struct_type_fixed_len_node_mod
    use math_cst
    use Dyn_Sigma_Uij_nonGlobal_mod
    use static_Sigma_d_Uij_nonGlobal_mod

    implicit none

    real(kind(0d0)), intent(in) :: xco(3)
    real(kind(0d0)), intent(in) :: nx(3)
    type(nodal_point), intent(in) :: y_nodals(yNumNodeBasis)
    integer, intent(in) :: yNumNodeBasis
    type(element), intent(in) :: ely
    real(kind(0d0)), intent(in) :: om
    type(elast_parameter_struct), intent(in) :: elastp
    integer, intent(in) :: sing
    complex(kind(0d0)), intent(out) :: zaTij(3,3)

    integer::i,j,k,ngy
    integer::integ
    real(kind(0d0)),dimension(3)::yco,y1,y2,y3
    real(kind(0d0)),dimension(3,3,3)::aaa,bbb
    real(kind(0d0)),dimension(3,3,3,3,3)::sigma_ikjd
    real(kind(0d0)),dimension(:),allocatable::gzi1,gzi2,gzi3,wi
    complex(kind(0d0))::zs
    complex(kind(0d0)),dimension(3,3,3)::dsigma,zsigma_ijk

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

    interface
       subroutine Gauss_tri(n,gzi1,gzi2,gzi3,wi)
         implicit none
         integer,parameter::n_ava=7
         integer,dimension(n_ava),parameter::ni_ava=(/3,4,7,13,27,48,79/)
         integer,intent(in)::n
         real(kind(0d0)),dimension(:),allocatable,intent(inout)::gzi1,gzi2,gzi3,wi
       end subroutine Gauss_tri
    end interface

    integ=4
    call Gauss_tri(integ,gzi1,gzi2,gzi3,wi)
    zs=-(0.0d0,1.0d0)*om

    y1(:) = y_nodals(ely%ind(1))%xc(:)
    y2(:) = y_nodals(ely%ind(2))%xc(:)
    y3(:) = y_nodals(ely%ind(3))%xc(:)
    !initialize
    zsigma_ijk=0.d0
    do ngy=1,integ
       do i=1,3
          yco(i)=y1(i)*gzi1(ngy)+y2(i)*gzi2(ngy)+y3(i)*gzi3(ngy)
       end do
       call elastSigma_dynamic_nonGlobal(xco,yco,elastp%cl,elastp%ct,zs,dsigma,elastp)
       zsigma_ijk(:,:,:)=zsigma_ijk(:,:,:)+dsigma(:,:,:)*wi(ngy)*ely%Jgg
    end do !ngy
    !--- static ---
    call static_Sigma_d_Uij_nonGlobal(xco,y1,y2,y3,sing,bbb,sigma_ikjd,aaa,elastp)
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
  end subroutine elast3d_aTij_l_nonGlobal
  !-------------------------------------
end module elast3d_atij_entrywise_mod
