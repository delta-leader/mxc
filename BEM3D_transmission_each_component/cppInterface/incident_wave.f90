module elast3d_incident_wave_mod
  implicit none
contains
!-------------------------------------------------
  pure complex(c_double_complex) function set_alpha(omega) result(res) bind(c)
    use iso_c_binding
    use math_cst
    use elast_parameter_struct_mod_global
    implicit none

    real(c_double), intent(in) :: omega

    res = ii/(omega/elout%ct)

  end function set_alpha
!-------------------------------------------------
  real(c_double) function get_mu(in_out, index) result(res) bind(c)
    use iso_c_binding
    use elast_parameter_struct_mod_global
    implicit none

    integer(c_int), intent(in) :: in_out
    integer(c_int), intent(in) :: index

    select case (in_out)
    case(0)
       res = elout%mu
    case(1)
       res = elin(index)%mu
    case default
       write(*,*) "ERROR at line", __LINE__, "in file", __FILE__
       stop
    end select

  end function get_mu
!-------------------------------------------------
  subroutine set_theta(theta) bind(c)
    use iso_c_binding
    use math_cst
    use BEM3d_small_mod
    implicit none

    real(c_double), intent(in) :: theta
    theta_in = theta*pi/180.0d0

  end subroutine set_theta
!-------------------------------------------------
  subroutine inc_disp(nodals, nnode, ix, elems, nel, om, uout) bind(c)
    use BEM3d_small_mod
    use math_cst
    use elast_parameter_struct_mod_global
    use struct_type_fixed_len_node_mod
    implicit none

    type(nodal_point), intent(in) :: nodals(nnode)
    integer(c_int), intent(in) :: nnode
    integer(c_int), intent(in) :: ix
    type(element), intent(in) :: elems(nel)
    integer(c_int), intent(in) :: nel
    real(c_double), intent(in) :: om
    complex(c_double_complex), intent(out) :: uout(3)

    integer::i,j,k,ng,integ
    integer :: ip, ex, nip
    real(kind(0d0))::kk
    real(kind(0d0)),dimension(3)::x1,x2,x3,xco,phix,pvec,dvec
    real(kind(0d0)),dimension(:),allocatable::gzi1,gzi2,gzi3,wi

    interface
       subroutine cal_phiy(y1,y2,y3,yco,phiy)
         implicit none
         real(kind(0d0)),dimension(3),intent(in)::y1,y2,y3,yco
         real(kind(0d0)),dimension(3),intent(out)::phiy
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

    integ=3
    call Gauss_tri(integ,gzi1,gzi2,gzi3,wi)
    pvec(1)=cos(theta_in)
    pvec(2)=0.d0
    pvec(3)=sin(theta_in)
    select case(id_inc)
    case(0)
       dvec(:)=pvec(:)
       kk = om/elout%cl
    case(1)
       dvec(1)=-sin(theta_in)
       dvec(2)=0.d0
       dvec(3)=cos(theta_in)
       kk = om/elout%ct
    case(2)
       dvec(:)=0.d0
       dvec(2)=1.d0
       kk = om/elout%ct
    end select

    uout(:) = 0.0d0

    do nip = 1, nodals(ix)%nel
       ex = nodals(ix)%iel(nip, 1)
       ip = nodals(ix)%iel(nip, 2)
       x1(:) = nodals(elems(ex)%ind(1))%xc(:)
       x2(:) = nodals(elems(ex)%ind(2))%xc(:)
       x3(:) = nodals(elems(ex)%ind(3))%xc(:)
       do ng = 1, integ
          xco(:) = x1(:)*gzi1(ng) + x2(:)*gzi2(ng) + x3(:)*gzi3(ng)
          call cal_phiy(x1,x2,x3,xco,phix)
          do i = 1, 3
             uout(i) = uout(i) + u0*dvec(i)*exp(ii*kk*(dot_product(pvec,xco)))&
                  &*phix(ip)*wi(ng)*elems(ex)%Jgg
          end do
       end do
    end do

  end subroutine inc_disp
!---------------------------------------
  subroutine inc_disp_const_x(nodals, nnode, elx, om, uout) bind(c)
    use BEM3d_small_mod
    use math_cst
    use elast_parameter_struct_mod_global
    use struct_type_fixed_len_node_mod
    implicit none

    type(nodal_point), intent(in) :: nodals(nnode)
    type(element), intent(in) :: elx
    integer(c_int), intent(in) :: nnode
    real(c_double), intent(in) :: om
    complex(c_double_complex), intent(out) :: uout(3)

    integer::i,j,k,ip,ng,integ
    real(kind(0d0))::kk
    real(kind(0d0)),dimension(3)::x1,x2,x3,xco,pvec,dvec
    real(kind(0d0)),dimension(:),allocatable::gzi1,gzi2,gzi3,wi

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
    pvec(1)=cos(theta_in)
    pvec(2)=0.d0
    pvec(3)=sin(theta_in)
    select case(id_inc)
    case(0)
       dvec(:)=pvec(:)
       kk = om/elout%cl
    case(1)
       dvec(1)=-sin(theta_in)
       dvec(2)=0.d0
       dvec(3)=cos(theta_in)
       kk = om/elout%ct
    case(2)
       dvec(:)=0.d0
       dvec(2)=1.d0
       kk = om/elout%ct
    end select
    x1(:) = nodals(elx%ind(1))%xc(:)
    x2(:) = nodals(elx%ind(2))%xc(:)
    x3(:) = nodals(elx%ind(3))%xc(:)
    uout(:)=0.d0

    do ng=1,integ
       xco(:)=x1(:)*gzi1(ng)+x2(:)*gzi2(ng)+x3(:)*gzi3(ng)
       !$omp simd
       do i=1,3
          uout(i)=uout(i)+u0*dvec(i)*exp(ii*kk*(dot_product(pvec,xco)))&
               &*wi(ng)*elx%Jgg
       end do
    end do

  end subroutine inc_disp_const_x
!--------------------------------------------
  subroutine inc_trac(nodals, nnode, ix, elems, nel, om, tout) bind(c)
    use BEM3d_small_mod
    use math_cst
    use elast_parameter_struct_mod_global
    use struct_type_fixed_len_node_mod
    implicit none

    type(nodal_point), intent(in) :: nodals(nnode)
    integer(c_int), intent(in) :: nnode
    integer(c_int), intent(in) :: ix
    type(element), intent(in) :: elems(nel)
    integer(c_int), intent(in) :: nel
    real(c_double), intent(in) :: om
    complex(c_double_complex), intent(out) :: tout(3)

    integer :: i, j, k, ng, integ
    integer :: ip, ex, nip
    real(kind(0d0))::kk,dia_norm,del,xi_x,zeta_x
    real(kind(0d0)),dimension(3)::x1,x2,x3,xco,phix,nvec,pvec,dvec
    real(kind(0d0)),dimension(:),allocatable::gzi1,gzi2,gzi3,wi
    complex(kind(0d0)),dimension(3,3)::ud

    interface
       subroutine cal_phiy(y1,y2,y3,yco,phiy)
         implicit none
         real(kind(0d0)),dimension(3),intent(in)::y1,y2,y3,yco
         real(kind(0d0)),dimension(3),intent(out)::phiy
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

    integ=3
    call Gauss_tri(integ,gzi1,gzi2,gzi3,wi)
    pvec(1)=cos(theta_in)
    pvec(2)=0.d0
    pvec(3)=sin(theta_in)
    select case(id_inc)
    case(0)
       dvec(:)=pvec(:)
       kk = om/elout%cl
    case(1)
       dvec(1)=-sin(theta_in)
       dvec(2)=0.d0
       dvec(3)=cos(theta_in)
       kk = om/elout%ct
    case(2)
       dvec(:)=0.d0
       dvec(2)=1.d0
       kk = om/elout%ct
    end select

    tout(:) = 0.d0

    do nip = 1, nodals(ix)%nel
       ex = nodals(ix)%iel(nip, 1)
       ip = nodals(ix)%iel(nip, 2)
       x1(:) = nodals(elems(ex)%ind(1))%xc(:)
       x2(:) = nodals(elems(ex)%ind(2))%xc(:)
       x3(:) = nodals(elems(ex)%ind(3))%xc(:)
       nvec(:) = elems(ex)%nvec(:)
       do ng = 1, integ
          xco(:) = x1(:)*gzi1(ng) + x2(:)*gzi2(ng) + x3(:)*gzi3(ng)
          call cal_phiy(x1,x2,x3,xco,phix)
          do i = 1, 3
             do j = 1, 3
                ud(i, j) = ii*u0*kk*dvec(i)*pvec(j)*cdexp(ii*kk*(dot_product(pvec,xco)))
             end do
          end do
          do i = 1, 3
             do k = 1, 3
                tout(i) = tout(i) + (elout%lam*ud(k, k)*nvec(i) + elout%mu*(ud(i, k) + ud(k, i))*nvec(k))*phix(ip)*wi(ng)*elems(ex)%Jgg
             end do
          end do
       end do
    end do

    tout(:) = tout(:)/elout%mu

  end subroutine inc_trac
  !-----------------------------------
end module elast3d_incident_wave_mod