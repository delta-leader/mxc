subroutine input
   use BEM3d
   use struct_type
   use math_cst
   use elast_parameter
   implicit none
   integer::i,j,k
   real(kind(0d0)),dimension(3)::alpha,beta,outpro
!==============================================================
!==============================================================
interface
subroutine append_iel(nd,ip,ne)
   use BEM3d
   use struct_type
   implicit none
   integer,intent(in)::nd,ne,ip
   end subroutine append_iel
end interface
!==============================================================
!---------------------------------------------
   open(unit=10,file='mesh_sphere486.inp')
   !open(unit=10,file='mesh_box970.inp')
!   open(unit=10,file='mesh_two_sphere_204nodes.inp')
!---------------------------------------------
   read(10,*)
   read(10,*)nel
   read(10,*)nnode
   allocate(el(nel),node(nnode))
   read(10,*)
   do i=1,nnode
      read(10,*)(node(i)%xc(j),j=1,3)
   end do
   read(10,*)
   do i=1,nel
      read(10,*)(el(i)%ind(j),j=1,3)
   end do
   close(10)
   call remove_dn__sort_nn
!---------------------------------------------
   open(unit=10,file='analysis_condition.inp')
!---------------------------------------------
   read(10,*)
   !id_bie=0:displacement formulation
   !id_bie=1:PMCHWT formulation
   !id_bie=2:Burton-Miller formulation
   read(10,*)id_bie
   select case(id_bie)
   case(0)
      write(*,*)"displacement formulation"
   case(1)
      write(*,*)"PMCHWT formulation"
   case(2)
      write(*,*)"Burton-Miller formulation"
   end select
   read(10,*)
   read(10,*)id_inc
   read(10,*)theta_in
   theta_in=theta_in*pi/180.0d0
   read(10,*)cl(1),ct(1),rho(1)
   read(10,*)cl(2),ct(2),rho(2)
   rad=1.d0; u0=1.d0
   read(10,*)
   read(10,*)
   read(10,*)ninf
   allocate(xinf(ninf))
   do i=1,ninf
      read(10,*)k,(xinf(i)%xc(j),j=1,3)
   end do
   close(10)
   im=1
   call elast_para(cl(1),ct(1),rho(1))
   !--------------------------------------
   do i=1,nel
      el(i)%xc(:)=( node(el(i)%ind(1))%xc(:)&
         &+node(el(i)%ind(2))%xc(:)&
         &+node(el(i)%ind(3))%xc(:) )/3.0d0
      do j=1,3
         el(i)%id(j)=0
      end do
   end do
   do i=1,nel
      do j=1,3
         alpha(j)=node(el(i)%ind(1))%xc(j)-node(el(i)%ind(3))%xc(j)
         beta(j)=node(el(i)%ind(2))%xc(j)-node(el(i)%ind(3))%xc(j)
      end do
      call out_product(outpro,alpha,beta)
      el(i)%Jgg=sqrt(dot_product(outpro,outpro))
      el(i)%nvec(:)=outpro(:)/el(i)%Jgg
      el(i)%Jgg=el(i)%Jgg*0.5d0

      if(dot_product(el(i)%nvec,el(i)%xc) > 0.d0)then
         el(i)%nvec(:)=-el(i)%nvec(:)
         k=el(i)%ind(2)
         el(i)%ind(2)=el(i)%ind(3)
         el(i)%ind(3)=k
      end if
      !--- set iel of node (triangular element) ---
      do j=1,3
         call append_iel(el(i)%ind(j),j,i)
      end do
   end do
   do i=1,nnode
      node(i)%nel=size(node(i)%iel,1)
   end do

!   !---------------------------
!   do i = 1, nnode
!      write(*,*) 'i, node(i)%nel', i, node(i)%nel
!   end do
!   !---------------------------

   ix1_min=1; ix1_0=1; ix1_max=1
   do i=2,nnode
      if(node(i)%xc(1) < node(ix1_min)%xc(1))then
         ix1_min=i
      else if(node(i)%xc(1) == node(ix1_min)%xc(1)&
         &.and. node(i)%xc(2)**2+node(i)%xc(3)**2 < node(ix1_min)%xc(2)**2+node(ix1_min)%xc(3)**2)then
         ix1_min=i
      end if
      if(node(i)%xc(1) > node(ix1_max)%xc(1))then
         ix1_max=i
      else if(node(i)%xc(1) == node(ix1_max)%xc(1)&
         &.and. node(i)%xc(2)**2+node(i)%xc(3)**2 < node(ix1_max)%xc(2)**2+node(ix1_max)%xc(3)**2)then
         ix1_max=i
      end if
!      if(node(i)%xc(3) > node(ix1_0)%xc(3))then
      if(abs(node(i)%xc(1)) < abs(node(ix1_0)%xc(1)))then
         ix1_0=i
      end if
   end do
   write(*,*)node(ix1_min)%xc
   write(*,*)node(ix1_0)%xc
   write(*,*)node(ix1_max)%xc
end subroutine input
!=======================================================================
!=======================================================================
!=======================================================================
subroutine remove_dn__sort_nn
   !remove double nodes and sort as {node_0, node_l, node_r}
   use BEM3d
   use struct_type
   implicit none
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
         norm=sqrt(dot_product(node(i)%xc-node(j)%xc,node(i)%xc-node(j)%xc))
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

   if(.false.) then ! no sort is required by c++
      allocate(dnode(nnode))
      dnode=node
      deallocate(node)
      allocate(node(nnode-icount),nd_el(nnode))
      nd_el=nd
      j=0
      do i=1,nnode
         if(nd(i) == i)then
            j=j+1
            node(j)=dnode(i)
            nd_el(i)=j
         end if
      end do
      do i=1,nnode
         if(nd(i) /= i)then
            nd_el(i)=nd_el(nd(i))
         end if
      end do
      do i=1,nel
         do j=1,3
            el(i)%ind(j)=nd_el(el(i)%ind(j))
         end do
      end do
      nnode=nnode-icount
      deallocate(nd_el,nd,dnode)
      !--- remove double node ---

      !--- sort ---
      x_left=-2.d0
      x_right=2.d0
      allocate(nd(nnode))
      nnode_0=0
      nnode_l=0
      nnode_r=0
      do i=1,nnode
         if(abs(node(i)%xc(1)-x_left) < 1.d-10)then
            nnode_l=nnode_l+1
         else if(abs(node(i)%xc(1)-x_right) < 1.d-10)then
            nnode_r=nnode_r+1
         else
            nnode_0=nnode_0+1
         end if
      end do

      i1=0
      i2=0
      i3=0
      do i=1,nnode
         if(abs(node(i)%xc(1)-x_left) < 1.d-10)then
            i1=i1+1
            nd(i)=nnode_0+i1
         else if(abs(node(i)%xc(1)-x_right) < 1.d-10)then
            i2=i2+1
            nd(i)=nnode_0+nnode_l+i2
         else
            i3=i3+1
            nd(i)=i3
         end if
      end do

      allocate(dnode(nnode))
      dnode=node
      do i=1,nnode
         node(nd(i))=dnode(i)
      end do
      deallocate(dnode)

      do i=1,nel
         do j=1,3
            el(i)%ind(j)=nd(el(i)%ind(j))
         end do
      end do
      deallocate(nd)
      !--- sort ---
   end if
 end subroutine remove_dn__sort_nn