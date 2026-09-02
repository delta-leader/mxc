module galerkin_uij_3d_entrywise_mod
  use iso_c_binding
  implicit none

  interface
     subroutine Gauss_tri(n,gzi1,gzi2,gzi3,wi)
       implicit none
       integer,parameter::n_ava=7
       integer,dimension(n_ava),parameter::ni_ava=(/3,4,7,13,27,48,79/)
       integer,intent(in)::n
       real(kind(0d0)),dimension(:),allocatable,intent(inout)::gzi1,gzi2,gzi3,wi
     end subroutine Gauss_tri
  end interface

contains
  subroutine constant_x_Uij_freq_nonGlobal(x_nodals, xNumNodeBasis, elx, y_nodals, yNumNodeBasis, ely, omega, elastp, sing, zUij)
    !\int \phi_{ip} \int mu*Uij dSy dSx
    use struct_type_fixed_len_node_mod
    use bem3d_small_mod
    use elast_parameter_struct_mod
    use math_cst
    use elast3d_uij_entrywise_mod
    implicit none

    type(nodal_point), intent(in) :: x_nodals(xNumNodeBasis)
    integer, intent(in) :: xNumNodeBasis
    type(element), intent(in) :: elx
    type(nodal_point), intent(in) :: y_nodals(yNumNodeBasis)
    integer, intent(in) :: yNumNodeBasis
    type(element), intent(in) :: ely
    real(kind(0d0)), intent(in) :: omega
    type(elast_parameter_struct), intent(in) :: elastp
    integer, intent(in) :: sing
    complex(kind(0d0)), intent(out):: zUij(3, 3)

    integer::i,j,ng,integ
    real(kind(0d0))::cst
    real(kind(0d0)),dimension(3)::xco,x1,x2,x3
    real(kind(0d0)),dimension(:),allocatable::gzi1,gzi2,gzi3,wi
    complex(kind(0d0)),dimension(3,3)::ten2

    integ=3
    call Gauss_tri(integ,gzi1,gzi2,gzi3,wi)
    x1(:) = x_nodals(elx%ind(1))%xc(:)
    x2(:) = x_nodals(elx%ind(2))%xc(:)
    x3(:) = x_nodals(elx%ind(3))%xc(:)
    zUij=0.d0
    do ng=1,integ
       xco(:)=x1(:)*gzi1(ng)+x2(:)*gzi2(ng)+x3(:)*gzi3(ng)
       ten2(:, :) = 0.0d0
       call elast3d_Uij_l_nonGlobal(xco, elx%nvec, y_nodals, yNumNodeBasis, ely, omega, elastp, sing, ten2)
       cst=wi(ng)*elx%Jgg
       zUij(:,:)=zUij(:,:)+ten2(:,:)*cst
    end do !ng
  end subroutine constant_x_Uij_freq_nonGlobal
  !--------------------------------------------------
  subroutine constant_x_Uij_freq_reg_nonGlobal(x_nodals, xNumNodeBasis, elx, y_nodals, yNumNodeBasis, ely, omega, elastp, sing, zUij)
    !\int \phi_{ip} \int mu*Uij dSy dSx
    use struct_type_fixed_len_node_mod
    use bem3d_small_mod
    use elast_parameter_struct_mod
    use math_cst
    use elast3d_uij_entrywise_mod
    implicit none

    type(nodal_point), intent(in) :: x_nodals(xNumNodeBasis)
    integer, intent(in) :: xNumNodeBasis
    type(element), intent(in) :: elx
    type(nodal_point), intent(in) :: y_nodals(yNumNodeBasis)
    integer, intent(in) :: yNumNodeBasis
    type(element), intent(in) :: ely
    real(kind(0d0)), intent(in) :: omega
    type(elast_parameter_struct), intent(in) :: elastp
    integer, intent(in) :: sing
    complex(kind(0d0)), intent(out):: zUij(3, 3)

    integer::i,j,ng,integ
    real(kind(0d0))::cst
    real(kind(0d0)),dimension(3)::xco,x1,x2,x3
    real(kind(0d0)),dimension(:),allocatable::gzi1,gzi2,gzi3,wi
    complex(kind(0d0)),dimension(3,3)::ten2

    integ = ngauss_x
    call Gauss_tri(integ,gzi1,gzi2,gzi3,wi)
    x1(:) = x_nodals(elx%ind(1))%xc(:)
    x2(:) = x_nodals(elx%ind(2))%xc(:)
    x3(:) = x_nodals(elx%ind(3))%xc(:)
    zUij=0.d0
    do ng=1,integ
       xco(:)=x1(:)*gzi1(ng)+x2(:)*gzi2(ng)+x3(:)*gzi3(ng)
       call elast3d_Uij_l_reg_nonGlobal(xco, elx%nvec, y_nodals, yNumNodeBasis, ely, omega, elastp, sing, ten2)
       cst=wi(ng)*elx%Jgg
       zUij(:,:)=zUij(:,:)+ten2(:,:)*cst
    end do !ng
  end subroutine constant_x_Uij_freq_reg_nonGlobal
  !--------------------------------------------------
end module galerkin_uij_3d_entrywise_mod