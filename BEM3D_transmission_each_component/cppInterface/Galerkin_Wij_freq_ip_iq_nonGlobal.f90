module galerkin_wij_3d_entrywise_mod
  implicit none
contains
  !----------------------------------------
  subroutine linear_x_Wij_freq_ip_iq_nonGlobal(x_nodals, xNumNodeBasis, elx, y_nodals, yNumNodeBasis, ely, omega, elastp, sing, ip, iq, zWij)
    !Wij/mu if Cijkl/mu
    !\int \phi_{ip} \int \phi_{iq} Wij/mu dSy dSx
    use struct_type_fixed_len_node_mod
    use BEM3d_small_mod
    use math_cst
    use elast_parameter_struct_mod
    use elast3d_wij_entrywise_mod
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
    integer, intent(in) :: ip
    integer, intent(in) :: iq
    complex(kind(0d0)), intent(out):: zWij(3, 3)

    integer :: i, j
    integer :: ng, integ
    real(kind(0d0))::cst
    real(kind(0d0)),dimension(3)::phix,xco,x1,x2,x3
    real(kind(0d0)),dimension(:),allocatable::gzi1,gzi2,gzi3,wi
    complex(kind(0d0)),dimension(3,3)::ten2

    interface
       subroutine cal_phiy(y1,y2,y3,yco,phiy)
         implicit none
         real(kind=8),dimension(3),intent(in)::y1,y2,y3,yco
         real(kind=8),dimension(3),intent(out)::phiy
       end subroutine cal_phiy
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

    integ = 3
    call Gauss_tri(integ,gzi1,gzi2,gzi3,wi)
    x1(:) = x_nodals(elx%ind(1))%xc(:)
    x2(:) = x_nodals(elx%ind(2))%xc(:)
    x3(:) = x_nodals(elx%ind(3))%xc(:)
    zWij(:,:) = 0.d0
    do ng = 1, integ
       ten2(:,:) = 0.d0
       xco(:) = x1(:)*gzi1(ng) + x2(:)*gzi2(ng) + x3(:)*gzi3(ng)
       call cal_phiy(x1,x2,x3,xco,phix)
       !call elast3d_Wij_l_iq(om,ten2,xco,elx%nvec,ely,sing,iq)
       call elast3d_Wij_l_iq_nonGlobal(xco, elx%nvec, y_nodals, yNumNodeBasis, ely, omega, elastp, sing, iq, ten2)
       cst = wi(ng)*elx%Jgg*phix(ip)
       zWij(:,:) = zWij(:,:) + ten2(:,:)*cst
    end do
  end subroutine linear_x_Wij_freq_ip_iq_nonGlobal
  !----------------------------------------
end module galerkin_wij_3d_entrywise_mod