module galerkin_tij_3d_entrywise_mod
  use iso_c_binding
  implicit none
contains
  !--------------------------------------------------
  !  subroutine linear_x_Tij_freq_ip_iq(om,zTij,elx,ely,sing,ip,iq)
  !    !\int \phi_{ip} \int \phi_{iq} Tij dSy dSx
  !    use BEM3D
  !    use math_cst
  !    use elast_parameter
  !    use struct_type
  !    implicit none
  !    integer::i,j,ng,integ
  !    real(kind(0d0))::cst
  !    real(kind(0d0)),dimension(3)::phix,xco,x1,x2,x3
  !    real(kind(0d0)),dimension(:),allocatable::gzi1,gzi2,gzi3,wi
  !    complex(kind(0d0)),dimension(3,3)::ten2
  !    !--------------------------------
  !    integer,intent(in)::sing,ip,iq
  !    real(kind(0d0)),intent(in)::om
  !    complex(kind(0d0)),dimension(3,3),intent(out)::zTij
  !    type(element),intent(in)::elx,ely
  !    !=======================================================================
  !    !=======================================================================
  !    !=======================================================================
  !    interface
  !       subroutine elast3d_Tij_l_iq(om,zTij,xco,ely,sing,iq)
  !         !zTij(i,j,iq)=\int Tij \phi_{iq} dSy
  !         !\phi_{iq} is piecewise linear shape function.
  !         !sing=1:xco is on ely; 0:no
  !         use BEM3D
  !         use elast_parameter
  !         use struct_type
  !         use math_cst
  !         implicit none
  !         integer,intent(in)::sing,iq
  !         real(kind(0d0)),intent(in)::om,xco(3)
  !         complex(kind(0d0)),intent(out)::zTij(3,3)
  !         type(element),intent(in)::ely
  !       end subroutine elast3d_Tij_l_iq
  !    end interface
  !    !=======================================================================
  !    interface
  !       subroutine cal_phiy(y1,y2,y3,yco,phiy)
  !         implicit none
  !         real(kind=8),dimension(3),intent(in)::y1,y2,y3,yco
  !         real(kind=8),dimension(3),intent(out)::phiy
  !       end subroutine cal_phiy
  !    end interface
  !    !=======================================================================
  !    interface
  !       subroutine Gauss_tri(n,gzi1,gzi2,gzi3,wi)
  !         implicit none
  !         integer,parameter::n_ava=7
  !         integer,dimension(n_ava),parameter::ni_ava=(/3,4,7,13,27,48,79/)
  !         integer,intent(in)::n
  !         real(kind(0d0)),dimension(:),allocatable,intent(inout)::gzi1,gzi2,gzi3,wi
  !       end subroutine Gauss_tri
  !    end interface
  !    !=======================================================================
  !    integ=3
  !    call Gauss_tri(integ,gzi1,gzi2,gzi3,wi)
  !    x1(:)=node(elx%ind(1))%xc(:)
  !    x2(:)=node(elx%ind(2))%xc(:)
  !    x3(:)=node(elx%ind(3))%xc(:)
  !    zTij=0.d0
  !    do ng=1,integ
  !       xco(:)=x1(:)*gzi1(ng)+x2(:)*gzi2(ng)+x3(:)*gzi3(ng)
  !       call cal_phiy(x1,x2,x3,xco,phix)
  !       call elast3d_Tij_l_iq(om,ten2,xco,ely,sing,iq)
  !       cst=wi(ng)*elx%Jgg*phix(ip)
  !       zTij(:,:)=zTij(:,:)+ten2(:,:)*cst
  !    end do !ng
  !  end subroutine linear_x_Tij_freq_ip_iq
  !=======================================================================
  !=======================================================================
  !=======================================================================
  subroutine constant_x_Tij_freq_iq_nonGlobal(x_nodals, xNumNodeBasis, elx, y_nodals, yNumNodeBasis, ely, om, elastp, sing, iq, zTij)
    !\int \phi_{ip} \int \phi_{iq} Tij dSy dSx
    use BEM3D_SMALL_MOD
    use math_cst
    use elast_parameter_struct_mod
    use struct_type_fixed_len_node_mod
    use elast3d_Tij_entrywise_mod
    implicit none

    type(nodal_point), intent(in) :: x_nodals(xNumNodeBasis)
    integer, intent(in) :: xNumNodeBasis
    type(element), intent(in) :: elx
    type(nodal_point), intent(in) :: y_nodals(yNumNodeBasis)
    integer, intent(in) :: yNumNodeBasis
    type(element), intent(in) :: ely
    real(kind(0d0)), intent(in) :: om
    type(elast_parameter_struct), intent(in) :: elastp
    integer, intent(in) :: sing
    integer, intent(in) :: iq
    complex(kind(0d0)), intent(out):: zTij(3, 3)

    integer::i,j,ng,integ
    real(kind(0d0))::cst
    real(kind(0d0)),dimension(3)::xco,x1,x2,x3
    real(kind(0d0)),dimension(:),allocatable::gzi1,gzi2,gzi3,wi
    complex(kind(0d0)),dimension(3,3)::ten2

    interface
       subroutine Gauss_tri(n,gzi1,gzi2,gzi3,wi)
         implicit none
         integer,parameter::n_ava=7
         integer,dimension(n_ava),parameter::ni_ava=(/3,4,7,13,27,48,79/)
         integer,intent(in)::n
         real(kind(0d0)),dimension(:),allocatable,intent(inout)::gzi1,gzi2,gzi3,wi
       end subroutine Gauss_tri
    end interface

    integ=3
    call Gauss_tri(integ,gzi1,gzi2,gzi3,wi)
    x1(:) = x_nodals(elx%ind(1))%xc(:)
    x2(:) = x_nodals(elx%ind(2))%xc(:)
    x3(:) = x_nodals(elx%ind(3))%xc(:)
    zTij=0.d0
    do ng=1,integ
       xco(:)=x1(:)*gzi1(ng)+x2(:)*gzi2(ng)+x3(:)*gzi3(ng)
       !call elast3d_Tij_l_iq(om,ten2,xco,ely,sing,iq)
       ten2(:, :) = 0.0d0
       call elast3d_Tij_l_iq_nonGlobal(xco, elx%nvec, y_nodals, yNumNodeBasis, ely, om, elastp, sing, iq, ten2)
       cst=wi(ng)*elx%Jgg
       zTij(:,:)=zTij(:,:)+ten2(:,:)*cst
    end do !ng
  end subroutine constant_x_Tij_freq_iq_nonGlobal
  !--------------------------------------------------
end module galerkin_tij_3d_entrywise_mod