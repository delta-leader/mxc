module input_non_global_mod
  implicit none
contains
  !--------------------------------------------
  subroutine input_non_global(nodals, nnode, elems, nel, sphere_num, mat_num) bind(c)
    use iso_c_binding
    use struct_type_fixed_len_node_mod
    use BEM3d_small_mod
    use math_cst
    use elast_parameter_struct_mod_global
    implicit none

    type(nodal_point), intent(inout) :: nodals(nnode)
    integer(c_int), intent(inout) :: nnode
    type(element), intent(inout) :: elems(nel)
    integer(c_int), intent(inout) :: nel
    integer(c_int), intent(in) :: sphere_num
    integer(c_int), intent(in) :: mat_num

    integer::i,j,k
    real(kind(0d0)),dimension(3)::alpha, beta, outpro

    character(len=10) :: mat_num_char     ! use your maximum expected len
    character(len=50) :: filename
    write(mat_num_char , '(I10)') mat_num        ! convert integer to char
    if (sphere_num == 8) then
      write(filename, '("../input/new/eight_spheres_", A, ".inp")') trim(adjustl(mat_num_char))
    else if (sphere_num == 5) then
      write(filename, '("../input/torus/torus_", A, ".inp")') trim(adjustl(mat_num_char))  
    else if (sphere_num == 4) then
      write(filename, '("../input/new/four_spheres_", A, ".inp")') trim(adjustl(mat_num_char))
    else if (sphere_num == 3) then
      write(filename, '("../input/salt/salt_", A, "k.inp")') trim(adjustl(mat_num_char)) 
    else if (sphere_num == 2) then
      write(filename, '("../input/new/two_spheres_", A, ".inp")') trim(adjustl(mat_num_char))
    else if (sphere_num == 1) then
      write(filename, '("../input/new/sphere_", A, ".inp")') trim(adjustl(mat_num_char))
    else
      !write(filename, '("../input/mesh_sphere_", A, "nodes.inp")') trim(adjustl(mat_num_char))
      write(filename, '("../input/mesh_two_sphere_", A, "nodes.inp")') trim(adjustl(mat_num_char)) 
   end if
    !write(*,*) filename

    write(*,*) 'inputNonGlobal, !dbg'

    ! file read
    !---------------------------------------------
    !   open(unit=10,file='mesh_sphere486.inp')
    !   open(unit=10,file='mesh_box970.inp')
    !open(unit=10,file='../input/mesh_sphere_160nodes.inp')
    open(unit=10,file=filename)
    !---------------------------------------------
    read(10,*)
    read(10,*) nel
    read(10,*) nnode
    !      allocate(el(nel),node(nnode))
    read(10,*)
    do i = 1, nnode
       read(10,*) (nodals(i)%xc(j), j = 1, 3)
    end do
    read(10,*)
    do i=1,nel
       read(10,*) (elems(i)%ind(j), j = 1, 3)
    end do
    close(10)
    call remove_dn_sort_nn_non_global(nodals, elems, nnode, nel)

    nnode3 = 3*nnode
    nel3 = 3*nel
    n_mat = nnode3 + nel3

    !---------------------------------------------
    open(unit=10,file='../input/analysis_condition.inp')
    !---------------------------------------------
    read(10,*)
    !id_bie=0:displacement formulation
    !id_bie=1:PMCHWT formulation
    !id_bie=2:Burton-Miller formulation
    read(10,*) id_bie
    select case(id_bie)
    case(0)
       write(*,*) "displacement formulation"
    case(1)
       write(*,*) "PMCHWT formulation"
    case(2)
       write(*,*) "Burton-Miller formulation"
    end select
    read(10,*)
    read(10,*) id_inc
    read(10,*) theta_in
    theta_in = theta_in*pi/180.0d0
    read(10,*) elout%cl, elout%ct, elout%rho
    read(10,*) elin(1)%cl, elin(1)%ct, elin(1)%rho
    read(10,*) ngauss_x, ngauss_y, ngauss_l
    rad=1.d0; u0=1.d0
    read(10,*)
    read(10,*)
    read(10,*)ninf
    allocate(xinf(ninf))
    do i=1,ninf
       read(10,*)k,(xinf(i)%xc(j),j=1,3)
    end do
    close(10)

    im = -1 ! dbg
    call elout%set_elast_para(elout%cl, elout%ct, elout%rho)
    call elin(1)%set_elast_para(elin(1)%cl, elin(1)%ct, elin(1)%rho)

    do i = 1, nel
       elems(i)%xc(:) = (nodals(elems(i)%ind(1))%xc(:)&
            & + nodals(elems(i)%ind(2))%xc(:)&
            & + nodals(elems(i)%ind(3))%xc(:))/3.0d0
       do j = 1, 3
          elems(i)%id(j) = 0
       end do
    end do
    do i = 1, nel
       do j = 1, 3
          alpha(j) = nodals(elems(i)%ind(1))%xc(j) - nodals(elems(i)%ind(3))%xc(j)
          beta(j)  = nodals(elems(i)%ind(2))%xc(j) - nodals(elems(i)%ind(3))%xc(j)
       end do
       call out_product(outpro, alpha, beta)
       elems(i)%Jgg     = sqrt(dot_product(outpro, outpro))
       elems(i)%nvec(:) = outpro(:)/elems(i)%Jgg
       elems(i)%Jgg     = elems(i)%Jgg*0.5d0

       if(dot_product(elems(i)%nvec, elems(i)%xc) > 0.d0) then
          elems(i)%nvec(:) = -elems(i)%nvec(:)
          k = elems(i)%ind(2)
          elems(i)%ind(2) = elems(i)%ind(3)
          elems(i)%ind(3) = k
       end if
       !--- set iel of node (triangular element) ---
       do j = 1, 3
          call append_iel_fixed_len(nodals(elems(i)%ind(j)), j, i)
       end do
    end do
!   !---------------------------
!    do i = 1, nnode
!       !nodals(i)%nel = size(node(i)%iel,1) ! it is already finished in append_iel
!       write(*,*) 'i, nodals(i)%nel', i, nodals(i)%nel
!    end do
!   !---------------------------

    ix1_min = 1; ix1_0 = 1; ix1_max = 1
    do i = 2, nnode
       if(nodals(i)%xc(1) < nodals(ix1_min)%xc(1)) then
          ix1_min = i
       else if(nodals(i)%xc(1) == nodals(ix1_min)%xc(1)&
            &.and. nodals(i)%xc(2)**2 + nodals(i)%xc(3)**2 < nodals(ix1_min)%xc(2)**2 + nodals(ix1_min)%xc(3)**2) then
          ix1_min = i
       end if
       if(nodals(i)%xc(1) > nodals(ix1_max)%xc(1))then
          ix1_max = i
       else if(nodals(i)%xc(1) == nodals(ix1_max)%xc(1)&
            &.and. nodals(i)%xc(2)**2+nodals(i)%xc(3)**2 < nodals(ix1_max)%xc(2)**2+nodals(ix1_max)%xc(3)**2)then
          ix1_max = i
       end if
       !      if(nodals(i)%xc(3) > nodals(ix1_0)%xc(3))then
       if(abs(nodals(i)%xc(1)) < abs(nodals(ix1_0)%xc(1)))then
          ix1_0 = i
       end if
    end do
    write(*,*) 'Min x1', nodals(ix1_min)%xc
    write(*,*) 'Most near origin', nodals(ix1_0)%xc
    write(*,*) 'Max x1', nodals(ix1_max)%xc

  end subroutine input_non_global
  !--------------------------------------------
  subroutine remove_dn_sort_nn_non_global(nodals, elems, nnode, nel)
    !remove double nodes and sort as {node_0, node_l, node_r}
    use BEM3d_small_mod
    use struct_type_fixed_len_node_mod
    implicit none

    type(nodal_point), intent(inout) :: nodals(nnode)
    type(element), intent(inout) :: elems(nel)
    integer(c_int), intent(inout) :: nnode
    integer(c_int), intent(inout) :: nel

    integer::i,j,icount,i1,i2,i3,nnode_0,nnode_l,nnode_r
    real(kind(0d0))::norm,x_left,x_right
    integer,allocatable::nd(:),ed(:),nd_el(:)
    type(nodal_point),allocatable::dnode(:)
    !=======================================================================
    !--- remove double node ---
    allocate(nd(nnode))
    icount=0
    do i=nnode,1,-1
       nd(i)=i
       do j=i+1,nnode
          norm=sqrt(dot_product(nodals(i)%xc-nodals(j)%xc,nodals(i)%xc-nodals(j)%xc))
          if(norm < 1.d-5)then
             icount=icount+1
             nd(i)=nd(j)
             exit
          end if
       end do
    end do

    if(icount .ne. 0) then
       write(*,*) "ERROR at line", __LINE__, "in file", __FILE__
       stop
    end if

!    allocate(dnode(nnode))
!    dnode=node
!    deallocate(node)
!    allocate(node(nnode-icount),nd_el(nnode))
!    nd_el=nd
!    j=0
!    do i=1,nnode
!       if(nd(i) == i)then
!          j=j+1
!          node(j)=dnode(i)
!          nd_el(i)=j
!       end if
!    end do
!    do i=1,nnode
!       if(nd(i) /= i)then
!          nd_el(i)=nd_el(nd(i))
!       end if
!    end do
!    do i=1,nel
!       do j=1,3
!          el(i)%ind(j)=nd_el(el(i)%ind(j))
!       end do
!    end do
!    nnode=nnode-icount
!    deallocate(nd_el,nd,dnode)
!    !--- remove double node ---
! 
!    !--- sort ---
!    x_left=-2.d0
!    x_right=2.d0
!    allocate(nd(nnode))
!    nnode_0=0
!    nnode_l=0
!    nnode_r=0
!    do i=1,nnode
!       if(abs(node(i)%xc(1)-x_left) < 1.d-10)then
!          nnode_l=nnode_l+1
!       else if(abs(node(i)%xc(1)-x_right) < 1.d-10)then
!          nnode_r=nnode_r+1
!       else
!          nnode_0=nnode_0+1
!       end if
!    end do
! 
!    i1=0
!    i2=0
!    i3=0
!    do i=1,nnode
!       if(abs(node(i)%xc(1)-x_left) < 1.d-10)then
!          i1=i1+1
!          nd(i)=nnode_0+i1
!       else if(abs(node(i)%xc(1)-x_right) < 1.d-10)then
!          i2=i2+1
!          nd(i)=nnode_0+nnode_l+i2
!       else
!          i3=i3+1
!          nd(i)=i3
!       end if
!    end do
! 
!    allocate(dnode(nnode))
!    dnode=node
!    do i=1,nnode
!       node(nd(i))=dnode(i)
!    end do
!    deallocate(dnode)
! 
!    do i=1,nel
!       do j=1,3
!          el(i)%ind(j)=nd(el(i)%ind(j))
!       end do
!    end do
!    deallocate(nd)
!    !--- sort ---
  end subroutine remove_dn_sort_nn_non_global
  !--------------------------------------------
  subroutine append_iel_fixed_len(singleNode, ip, indexElem)
    use BEM3d_small_mod
    use struct_type_fixed_len_node_mod
    implicit none

    type(nodal_point), intent(inout) :: singleNode
    integer, intent(in) :: ip
    integer, intent(in) :: indexElem

    ! nodals%iel is already allocated in initialization
    if(singleNode%nel == -1) then
       singleNode%nel = 1
    else
       singleNode%nel = singleNode%nel + 1
    end if

    singleNode%iel(singleNode%nel, 1) = indexElem
    singleNode%iel(singleNode%nel, 2) = ip

    !    if(allocated(node(nd)%iel))then
    !       i=size(node(nd)%iel,1)
    !       allocate(ievec(i,2))
    !       ievec(:,:)=node(nd)%iel(:,:)
    !       deallocate(node(nd)%iel)
    !       allocate(node(nd)%iel(i+1,2))
    !       node(nd)%iel(1:i,:)=ievec(:,:)
    !       deallocate(ievec)
    !    else
    !       i=0
    !       allocate(node(nd)%iel(i+1,2))
    !    end if
    !    node(nd)%iel(i+1,1)=ne
    !    node(nd)%iel(i+1,2)=ip
  end subroutine append_iel_fixed_len
  !--------------------------------------------
end module input_non_global_mod
