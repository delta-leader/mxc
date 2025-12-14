program GBEM3D
   !$ use omp_lib
  !use lapack95, only: gesv
  use f95_lapack, only: la_gesv
   use BEM3d
   use struct_type
   use elast_parameter
   use math_cst
   implicit none
   !id_bie=0:displacement formulation
   !id_bie=1:PMCHWT formulation
   !id_bie=2:Burton-Miller formulation
   integer::i,j,ix,iy,ex,ey,nip,niq,ip,iq,sing,freq
   real(kind(0d0))::omega,vec(3),dten2(3,3)
   complex(kind(0d0))::alpha
   complex(kind(0d0)),dimension(:),allocatable::uout,x0,rhs
   complex(kind(0d0)),dimension(:,:),allocatable::Cmat
   complex(kind(0d0))::zten2(3,3)
   character(len=72)::flname_vtu(3)
!==============================================================
interface
subroutine mkmat_transmission_disp(Cmat,rhs,omega)
   use BEM3d
   use elast_parameter
   use struct_type
   implicit none
   real(kind(0d0)),intent(in)::omega
   complex(kind(0d0)),dimension(n_mat),intent(out)::rhs
   complex(kind(0d0)),dimension(n_mat,n_mat),intent(out)::Cmat
   end subroutine mkmat_transmission_disp
end interface
!==============================================================
interface
subroutine mkmat_transmission_PMCHWT(Cmat,rhs,omega)
   use BEM3d
   use elast_parameter
   use struct_type
   implicit none
   real(kind(0d0)),intent(in)::omega
   complex(kind(0d0)),dimension(n_mat),intent(out)::rhs
   complex(kind(0d0)),dimension(n_mat,n_mat),intent(out)::Cmat
   end subroutine mkmat_transmission_PMCHWT
end interface
!==============================================================
interface
subroutine mkmat_transmission_BM(Cmat,rhs,omega,alpha)
   use BEM3d
   use elast_parameter
   use struct_type
   implicit none
   real(kind(0d0)),intent(in)::omega
   complex(kind(0d0)),intent(in)::alpha
   complex(kind(0d0)),dimension(n_mat),intent(out)::rhs
   complex(kind(0d0)),dimension(n_mat,n_mat),intent(out)::Cmat
   end subroutine mkmat_transmission_BM
end interface
!==============================================================
interface
subroutine zgmresk_BEM(n,a,x,b,id_ini,x0,id)
   !This subroutine solves complex matrix equation "a(n,n) x(n) = b(n)".
   !This subroutine outputs the residual norm normalized by |b| on the file "fort."id"".
   !If you use intial vector, "id_ini" must be equal to 1.
   !"x0" is initial vector.
   implicit none
   integer,intent(in)::n,id,id_ini
   complex(kind=8),dimension(n),intent(in)::b,x0
   complex(kind=8),dimension(n),intent(out)::x
   complex(kind=8),dimension(n,n),intent(in)::a
   end subroutine zgmresk_BEM
end interface
!==============================================================
!==============================================================
!$ call omp_set_num_threads(20)
!-------------------------------------------
   call input
   open(unit=222,file="disp.out")
   open(unit=223,file="disp_check.out")
   open(unit=224,file="trac_check.out")
   nnode3=3*nnode
   nel3=3*nel
   n_mat=nnode3+nel3
   allocate(x0(n_mat)); x0=0.d0
   !do freq=1,100
   !   write(*,*)"frequency step=",freq
   !   omega=dble(freq)*0.05d0
   do freq = 1, 1
      omega = 2.0d0
      allocate(rhs(n_mat),uout(n_mat),Cmat(n_mat,n_mat))
      select case(id_bie)
      case(0)! displacement formulation
         call mkmat_transmission_disp(Cmat,rhs,omega)
      case(1)! PMCHWT formulation
         call mkmat_transmission_PMCHWT(Cmat,rhs,omega)
      case(2)! Burton-Miller formulation
         alpha=ii/(omega/ct(1))
         call mkmat_transmission_BM(Cmat,rhs,omega,alpha)
      end select

!      !-------------------------
!      do i = 1, n_mat
!         !do i = 1, 3*nnode
!         write(*,*) 'i, rhs(i)', i, rhs(i)
!      end do
!      do j = 1, n_mat
!         do i = 1, n_mat
!            write(*,*) 'i, j, cmat(i, j)', i, j, cmat(i, j)
!         end do
!      end do
!      !-------------------------

!      call zgmresk_BEM(n_mat,Cmat,uout,rhs,1,x0,11)
      uout(:)=rhs(:)
      !call gesv(a=Cmat,b=uout)
      call la_gesv(a=Cmat,b=uout)
      deallocate(Cmat,rhs)
      !-------------------------
      block
        open(11, file='solvec_fortran.dat', status='replace')
        do i = 1, n_mat
           write(11,*) i, dble(uout(i)), dimag(uout(i))
        end do
        close(11)
      end block
      !-------------------------
      call output(omega,uout)
      do i=1,nnode
         do j=1,3
            node(i)%u(j)=uout(3*(i-1)+j)
         end do
      end do
      do i=1,nel
         do j=1,3
            el(i)%t(j)=uout(nnode3+3*(i-1)+j)
         end do
      end do
      x0(:)=uout(:)
      deallocate(uout)
      !----dbg----------------
      block
        open(unit=225,file="soluRef.dat")
        write(225, *) 'i, x, y, z, Re(u(x)), Im(u(x)), Re(u(y)), Im(u(y)), Re(u(z)), Im(u(z))'
        do i = 1, nnode
           write(225, *) i, node(i)%xc(1), node(i)%xc(2), node(i)%xc(3), real(node(i)%u(1)), imag(node(i)%u(1)), real(node(i)%u(2)), imag(node(i)%u(2)), real(node(i)%u(3)), imag(node(i)%u(3))
        end do
        close(225)
        open(unit=226,file="soltRef.dat")
        write(226, *) 'i, x, y, z, Re(t(x)), Im(t(x)), Re(t(y)), Im(t(y)), Re(t(z)), Im(t(z))'
        do i=1,nel
           write(226, *) i, el(i)%xc(1), el(i)%xc(2), el(i)%xc(3), real(el(i)%t(1)), imag(el(i)%t(1)), real(el(i)%t(2)), imag(el(i)%t(2)), real(el(i)%t(3)), imag(el(i)%t(3))
        end do
        close(226)
      end block
      !----dbg----------------
!-------------------------------------------
!      write(flname_vtu(1),'("u_real",i4.4,".vtu")')freq
!      write(flname_vtu(2),'("u_imag",i4.4,".vtu")')freq
!      write(flname_vtu(3),'("u_abs",i4.4,".vtu")')freq
!      call output_vtu_binary_file
!-------------------------------------------
   end do
   deallocate(x0)
   close(222)
!-------------------------------------------
contains
!=================================================================
subroutine output_vtu_binary_file
   implicit none
!------------------------------[Local parameters]------------------------------!
   integer, parameter           :: IP = 8         ! integer precision
   integer, parameter           :: RP = 8         ! real precision
   integer, parameter           :: SP = 4         ! single precision
   character(len= 1), parameter :: LF = char(10)  ! line feed
   character(len= 0), parameter :: IN_0 = ''      ! indentation levels
   character(len= 2), parameter :: IN_1 = '  '
   character(len= 4), parameter :: IN_2 = '    '
   character(len= 6), parameter :: IN_3 = '      '
   character(len= 8), parameter :: IN_4 = '        '
   character(len=10), parameter :: IN_5 = '          '
   integer(IP),       parameter :: VTK_HEXAHEDRON = 12
!-----------------------------------[Locals]-----------------------------------!
   integer(IP),parameter::id_tetrahedron=10    !ID for tetrahedron element in vtk format
   integer(IP),parameter::id_triangle=5    !ID for tetrahedron element in vtk format
   integer(SP)       :: data_size              ! should be SP no matter what
   integer::ifl,i,j
   integer(IP)::fu,nnd_el,offset_cell,offset_data
   integer(IP),dimension(:),allocatable::ind
   character(len=80) :: str1, str2
!==============================================================================!
   nnd_el=3!number of nodes in one element
   
   do ifl=1,3
      select case(ifl)
      case(1)
         open(newunit = fu,                     &
              file    = flname_vtu(1),    &
              status  = 'replace',              &
              form    = 'unformatted',          &
              access  = 'stream')
      case(2)
         open(newunit = fu,                     &
              file    = flname_vtu(2),    &
              status  = 'replace',              &
              form    = 'unformatted',          &
              access  = 'stream')
      case(3)
         open(newunit = fu,                     &
              file    = flname_vtu(3),     &
              status  = 'replace',              &
              form    = 'unformatted',          &
              access  = 'stream')
      end select

      !------------!
      !   Header   !
      !------------!
      write(fu) IN_0 // '<?xml version="1.0"?>' // LF
      write(fu) IN_0 // '<VTKFile type="UnstructuredGrid" version="0.1" ' //  &
                        'byte_order="LittleEndian">'                      // LF
      write(fu) IN_1 // '<UnstructuredGrid>' // LF
      write(str1, '(i0.0)') nnode
      write(str2, '(i0.0)') nel
      write(fu) IN_2 // '<Piece NumberOfPoints="' // trim(str1) //  &
                             '" NumberOfCells ="' // trim(str2) // '">' // LF

      !----------------!
      !   Point info   !
      !----------------!
      write(fu) IN_3 // '<Points>' // LF
      offset_data=0
      write(str1, '(i1)')   offset_data                ! data_offset
      write(str2, '(i0.0)') RP * 8                     ! real precision
      write(fu) IN_4 // '<DataArray type="Float' // trim(str2) // '"' //  &
                        ' NumberOfComponents="3"'                     //  &
                        ' format="appended"'                          //  &
                        ' offset="' // trim(str1) // '">' // LF
      write(fu) IN_4 // '</DataArray>' // LF
      write(fu) IN_3 // '</Points>'    // LF

      !---------------!
      !   Cell info   !
      !---------------!
      write(fu) IN_3 // '<Cells>' // LF

      ! Connectivity
      offset_data=offset_data + SP + nnode * 3 * RP
      write(str1, '(i0.0)') offset_data                 ! data_offset
      write(str2, '(i0.0)') IP * 8                      ! integer precision
      write(fu) IN_4 // '<DataArray type="Int' // trim(str2) // '"' //  &
                        ' Name="connectivity"'                      //  &
                        ' format="appended"'                        //  &
                        ' offset="' // trim(str1) // '">' // LF
      write(fu) IN_4 // '</DataArray>' // LF

      ! Offsets
      offset_data=offset_data + SP + nel * nnd_el * IP
      write(str1, '(i0.0)') offset_data               ! data_offset
      write(str2, '(i0.0)') IP * 8                    ! integer precision
      write(fu) IN_4 // '<DataArray type="Int' // trim(str2) // '"' //  &
                        ' Name="offsets"'                           //  &
                        ' format="appended"'                        //  &
                        ' offset="' // trim(str1) // '">' // LF
      write(fu) IN_4 // '</DataArray>' // LF

      ! Types
      offset_data=offset_data + SP + nel * IP
      write(str1, '(i0.0)') offset_data               ! data_offset
      write(str2, '(i0.0)') IP * 8                    ! integer precision
      write(fu) IN_4 // '<DataArray type="Int' // trim(str2) // '"' //  &
                        ' Name="types"'                             //  &
                        ' format="appended"'                        //  &
                        ' offset="' // trim(str1) // '">' // LF
      write(fu) IN_4 // '</DataArray>' // LF

      write(fu) IN_3 // '</Cells>' // LF

      !---------------------!
      !   Point data info   !
      !---------------------!
      write(fu) IN_3 // '<PointData>' // LF
      offset_data=offset_data + SP + nel * IP
      write(str1, '(i0.0)') offset_data               ! data_offset
      write(str2, '(i0.0)') RP * 8                    ! real precision
      write(fu) IN_4 // '<DataArray type="Float' // trim(str2) // '"' //  &
                        ' Name="point_vectors"'                       //  &
                        ' NumberOfComponents="3"'                     //  &
                        ' format="appended"'                          //  &
                        ' offset="' // trim(str1) // '">' // LF
      write(fu) IN_4 // '</DataArray>' // LF
      write(fu) IN_3 // '</PointData>'    // LF
      write(fu) IN_2 // '</Piece>'             // LF
      write(fu) IN_1 // '</UnstructuredGrid>'  // LF

      !---------------------!
      !   Append all data   !
      !---------------------!
      write(fu) IN_0 // '<AppendedData encoding="raw">' // LF
      write(fu) '_'

      !-----------!
      !   Point   !
      !-----------!
      data_size = nnode * 3 * RP         ! three coordinates for each node
      write(fu) data_size
      do i = 1, nnode 
        write(fu) (node(i)%xc(j),j=1,3)
      end do

      !----------!
      !   Cell   !
      !----------!

      ! Connectivity
      allocate(ind(nnd_el))
      data_size = nel * nnd_el * IP         ! eight nodes for cell
      write(fu) data_size
      do i = 1, nel
         ind(:)=el(i)%ind(:)-1
         write(fu) (ind(j),j=1,nnd_el)
      end do
      deallocate(ind)

      ! Offsets
      data_size = nel * IP         ! one offset for each cell
      write(fu) data_size
      offset_cell = 0
      do i = 1, nel
        offset_cell = offset_cell + nnd_el
        write(fu) offset_cell
      end do

      ! Types
      data_size = nel * IP     ! one type for each cell
      write(fu) data_size
      do i = 1, nel
        write(fu) id_triangle
      end do

      !----------------!
      !   Point data   !
      !----------------!
      data_size = nnode * 3 * RP         ! three coordinates for each node
      write(fu) data_size
      select case(ifl)
      case(1)
         do i = 1, nnode 
           write(fu) (real(node(i)%u(j)),j=1,3)
         end do
      case(2)
         do i = 1, nnode 
           write(fu) (aimag(node(i)%u(j)),j=1,3)
         end do
      case(3)
         do i = 1, nnode 
           write(fu) (cdabs(node(i)%u(j)),j=1,3)
         end do
      end select

      !------------!
      !            !
      !   Footer   !
      !            !
      !------------!
      write(fu) LF // IN_0 // '</AppendedData>' // LF
      write(fu)       IN_0 // '</VTKFile>'      // LF

      close(fu)
   end do

end subroutine output_vtu_binary_file
!=================================================================
end program GBEM3D
!==============================================================
!==============================================================
!==============================================================
subroutine output(om,uout)
   use BEM3d
   use elast_parameter
   use math_cst
   implicit none
   integer::i,j,ix
   real(kind(0d0))::uabs(3),norm
   real(kind(0d0)),intent(in)::om
   complex(kind(0d0)),dimension(n_mat),intent(in)::uout
   complex(kind(0d0))::zvec(3)
!------------------------------------------
   uabs=0.d0
   do j=1,3
      uabs(1)=uabs(1)+cdabs(uout(3*(ix1_min-1)+j))**2
      uabs(2)=uabs(2)+cdabs(uout(3*(ix1_0-1)+j))**2
      uabs(3)=uabs(3)+cdabs(uout(3*(ix1_max-1)+j))**2
   end do
   do i=1,3
      uabs(i)=sqrt(uabs(i))
   end do
   write(222,300)rad*om/ct(1),(uabs(i)/u0,i=1,3)
   !---
   norm=0.d0
   call freq_inc_displacement_noint(node(ix1_min)%xc,om,zvec)
   do i=1,3
      zvec(i)=zvec(i)-uout(3*(ix1_min-1)+i)
   end do
   norm=norm+real(dot_product(zvec,zvec))
   call freq_inc_displacement_noint(node(ix1_0)%xc,om,zvec)
   do i=1,3
      zvec(i)=zvec(i)-uout(3*(ix1_0-1)+i)
   end do
   norm=norm+real(dot_product(zvec,zvec))
   call freq_inc_displacement_noint(node(ix1_max)%xc,om,zvec)
   do i=1,3
      zvec(i)=zvec(i)-uout(3*(ix1_max-1)+i)
   end do
   norm=norm+real(dot_product(zvec,zvec))
   norm=sqrt(norm)
   write(223,300)rad*om/ct(1),norm
   !---
   norm=0.d0
   do ix=100,300,100
      call freq_inc_traction_noint(el(ix)%xc,el(ix)%nvec,om,zvec)
      zvec(:)=zvec(:)/mu
      do i=1,3
         zvec(i)=zvec(i)-uout(nnode3+3*(ix-1)+i)
      end do
      norm=norm+real(dot_product(zvec,zvec))
   end do
   norm=sqrt(norm)
   write(224,300)rad*om/ct(1),norm

300 format(10e20.10)
   end subroutine output
!==============================================================
!==============================================================