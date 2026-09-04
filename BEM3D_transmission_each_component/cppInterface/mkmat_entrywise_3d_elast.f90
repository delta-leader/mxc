module mkmat_inclusion_entrywise_3d_mod
  use iso_c_binding
  implicit none
contains
!--------------------------------------------------
  subroutine mkmat_entrywise_3d_elast(x_nodals, xNumNodeBasis, x_elems, xNumElemBasis, xindex, y_nodals, yNumNodeBasis, y_elems, yNumElemBasis, yindex, omega, dummat) bind(c)
    use struct_type_fixed_len_node_mod
    use elast_parameter_struct_mod_global
    use galerkin_uij_3d_entrywise_mod
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
    complex(c_double_complex), intent(out) :: dummat(3, 3)

    type(elast_parameter_struct) :: elastp
    complex(c_double_complex) :: zten2(3, 3)
    !integer :: i, j
    integer :: nip, ex, ip
    integer :: niq, ey, iq
    integer :: sing

    dummat(:, :) = 0.0d0
    zten2(:, :) = 0.0d0

    elastp = elout
    ex = xindex
    ey = yindex
    if(ex == ey)then
       sing = 1
    else
       sing = 0
    end if
    !--- Uij ---
    zten2(:, :) = 0.0d0
    call constant_x_Uij_freq_reg_nonGlobal(x_nodals, xNumNodeBasis, x_elems(ex), y_nodals, yNumNodeBasis, y_elems(ey), omega, elastp, sing, zten2)
    dummat(:, :) = dummat(:, :) + zten2(:, :)
  end subroutine mkmat_entrywise_3d_elast

end module mkmat_inclusion_entrywise_3d_mod