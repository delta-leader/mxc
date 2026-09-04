module elast3d_incident_wave_mod
  implicit none
contains
!-------------------------------------------------
  subroutine inc_disp_const_x(nodals, nnode, elx, omega, theta, uout) bind(c)
    use BEM3d_small_mod
    use math_cst
    use elast_parameter_struct_mod_global
    use struct_type_fixed_len_node_mod
    implicit none

    type(nodal_point), intent(in) :: nodals(nnode)
    type(element), intent(in) :: elx
    integer(c_int), intent(in) :: nnode
    real(c_double), intent(in) :: omega, theta
    complex(c_double_complex), intent(out) :: uout(3)

    integer::i,j,k,ip,ng,integ
    real(kind(0d0))::kk, theta_in
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
    
    ! convert angle
    theta_in = theta*pi/180.0d0
    
    integ=3
    call Gauss_tri(integ,gzi1,gzi2,gzi3,wi)
    pvec(1)=cos(theta_in)
    pvec(2)=0.d0
    pvec(3)=sin(theta_in)
    ! P-wave
    dvec(:)=pvec(:)
    kk = omega/elout%cl
    
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
  !-----------------------------------
end module elast3d_incident_wave_mod