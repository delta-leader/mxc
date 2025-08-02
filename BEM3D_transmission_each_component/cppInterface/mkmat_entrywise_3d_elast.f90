module mkmat_inclusion_entrywise_3d_mod
  use iso_c_binding
  implicit none
contains
!--------------------------------------------------
  subroutine mkmat_entrywise_3d_elast(x_nodals, xNumNodeBasis, x_elems, xNumElemBasis, xindex, y_nodals, yNumNodeBasis, y_elems, yNumElemBasis, yindex, omega, out_in, slp_or_dlp, linear_or_const, dummat) bind(c)
    use struct_type_fixed_len_node_mod
    use elast_parameter_struct_mod_global
    use galerkin_uij_3d_entrywise_mod
    use galerkin_tij_3d_entrywise_mod
    use galerkin_atij_3d_entrywise_mod
    use galerkin_wij_3d_entrywise_mod
    implicit none

    type(nodal_point), intent(in) :: x_nodals(xNumNodeBasis)
    integer(c_int), intent(in) :: xNumNodeBasis
    type(element), intent(in) :: x_elems(xNumElemBasis)
    integer(c_int), intent(in) :: xNumElemBasis
    integer(c_int), intent(in) :: xindex
    type(nodal_point), intent(in) :: y_nodals(yNumNodeBasis)
    integer(c_int), intent(in) :: yNumNodeBasis
    type(element), intent(in) :: y_elems(yNumElemBasis)
    integer(c_int), intent(in) :: yNumElemBasis
    integer(c_int), intent(in) :: yindex
    real(c_double), intent(in) :: omega
    integer(c_int), intent(in) :: out_in ! 0==out, 1==in
    integer(c_int), intent(in) :: slp_or_dlp ! 1==slp, 2==dlp, 3==d_slp, 4==d_dlp
    integer(c_int), intent(in) :: linear_or_const ! 0==linear basis Galerkin, 1==const basis Galerkin
    complex(c_double_complex), intent(out) :: dummat(3, 3)

    type(elast_parameter_struct) :: elastp
    complex(c_double_complex) :: zten2(3, 3)
    !integer :: i, j
    integer :: nip, ex, ip
    integer :: niq, ey, iq
    integer :: sing

    dummat(:, :) = 0.0d0
    zten2(:, :) = 0.0d0

    select case (out_in)
    case(0)
       elastp = elout
    case(1)
       elastp = elin(1)
    case default
       write(*,*) "ERROR at line", __LINE__, "in file", __FILE__
       stop
    end select

    select case(linear_or_const)
    case(0) ! Galerkin method with linear test function
       select case(slp_or_dlp)
       case(3)
          do nip = 1, x_nodals(xindex)%nel
             ex = x_nodals(xindex)%iel(nip, 1)
             ip = x_nodals(xindex)%iel(nip, 2)

             ey = yindex
             if(ex == ey)then
                sing = 1
             else
                sing = 0
             end if
             !--- aTij ---
             !call linear_x_aTij_freq_ip(omega,zten2,el(ex),el(ey),sing,ip)
             zten2(:, :) = 0.0d0
             call linear_x_aTij_freq_ip_nonGlobal(x_nodals, xNumNodeBasis, x_elems(ex), y_nodals, yNumNodeBasis, y_elems(ey), omega, elastp, sing, ip, zten2)
             dummat(:, :) = dummat(:, :) + zten2(:, :)
          end do
       case(4)
          do nip = 1, x_nodals(xindex)%nel
             ex = x_nodals(xindex)%iel(nip, 1)
             ip = x_nodals(xindex)%iel(nip, 2)
             do niq = 1, y_nodals(yindex)%nel
                ey = y_nodals(yindex)%iel(niq, 1)
                iq = y_nodals(yindex)%iel(niq, 2)
                if(ex == ey)then
                   sing = 1
                else
                   sing = 0
                end if
                !--- Wij ---
                !call linear_x_Wij_freq_ip_iq(omega, zten2, x_elems(ex), y_elems(ey), sing, ip, iq)
                zten2(:, :) = 0.0d0
                call linear_x_Wij_freq_ip_iq_nonGlobal(x_nodals, xNumNodeBasis, x_elems(ex), y_nodals, yNumNodeBasis, y_elems(ey), omega, elastp, sing, ip, iq, zten2)
                dummat(:, :) = dummat(:, :) + zten2(:, :)
             end do
          end do
       case default
          write(*,*) "ERROR at line", __LINE__, "in file", __FILE__
          stop
       end select
    case(1) !
       select case(slp_or_dlp)
       case(1)
          ex = xindex
          ey = yindex
          if(ex == ey)then
             sing = 1
          else
             sing = 0
          end if
          !--- Uij ---
          zten2(:, :) = 0.0d0
          call constant_x_Uij_freq_nonGlobal(x_nodals, xNumNodeBasis, x_elems(ex), y_nodals, yNumNodeBasis, y_elems(ey), omega, elastp, sing, zten2)
          dummat(:, :) = dummat(:, :) + zten2(:, :)
       case(2)
          ex = xindex
          do niq = 1, y_nodals(yindex)%nel
             ey = y_nodals(yindex)%iel(niq, 1)
             iq = y_nodals(yindex)%iel(niq, 2)
             if(ex == ey)then
                sing = 1
             else
                sing = 0
             end if
             !--- Tij ---
             zten2(:, :) = 0.0d0
             call constant_x_Tij_freq_iq_nonGlobal(x_nodals, xNumNodeBasis, x_elems(ex), y_nodals, yNumNodeBasis, y_elems(ey), omega, elastp, sing, iq, zten2)
             dummat(:, :) = dummat(:, :) + zten2(:, :)
          end do
       case default
          write(*,*) "ERROR at line", __LINE__, "in file", __FILE__
          stop
       end select
    case default
       write(*,*) "ERROR at line", __LINE__, "in file", __FILE__
       stop
    end select
!    if(slp_or_dlp .eq. 4) then
!    do ix=1,nnode
!       do nip=1,node(ix)%nel
!          ex=node(ix)%iel(nip,1)
!          ip=node(ix)%iel(nip,2)
!          do iy=1,nnode
!             do niq=1,node(iy)%nel
!                ey=node(iy)%iel(niq,1)
!                iq=node(iy)%iel(niq,2)
!                if(ex == ey)then
!                   sing=1
!                else
!                   sing=0
!                end if
!                !--- Wij ---
!                call linear_x_Wij_freq_ip_iq(omega,zten2,el(ex),el(ey),sing,ip,iq)
!                do i=1,3
!                   do j=1,3
!                      Cmat(3*(ix-1)+i,3*(iy-1)+j)&
!                           &=Cmat(3*(ix-1)+i,3*(iy-1)+j)&
!                           &+zten2(i,j)
!                   end do
!                end do
!             end do
!          end do
!       end do
!    end do
  end subroutine mkmat_entrywise_3d_elast

end module mkmat_inclusion_entrywise_3d_mod
