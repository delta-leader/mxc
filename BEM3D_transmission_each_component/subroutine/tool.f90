subroutine append_integer(a,i)
   implicit none
   integer,intent(in)::i
   integer,dimension(:),allocatable,intent(inout)::a
   integer,dimension(:),allocatable::b
!----------------------------------------------------
   allocate(b(size(a)))
   b=a
   deallocate(a)
   allocate(a(size(b)+1))
   a(1:size(b))=b
   a(size(a))=i
   deallocate(b)
   end subroutine append_integer
!=======================================================================
!=======================================================================
!=======================================================================
subroutine append_iel(nd,ip,ne)
   use BEM3d
   use struct_type
   implicit none
   integer::i
   integer,intent(in)::nd,ne,ip
   integer,dimension(:,:),allocatable::ievec
!----------------------------------------------------
   if(allocated(node(nd)%iel))then
      i=size(node(nd)%iel,1)
      allocate(ievec(i,2))
      ievec(:,:)=node(nd)%iel(:,:)
      deallocate(node(nd)%iel)
      allocate(node(nd)%iel(i+1,2))
      node(nd)%iel(1:i,:)=ievec(:,:)
      deallocate(ievec)
   else
      i=0
      allocate(node(nd)%iel(i+1,2))
   end if
   node(nd)%iel(i+1,1)=ne
   node(nd)%iel(i+1,2)=ip
   end subroutine append_iel
!=======================================================================
!=======================================================================
!=======================================================================
subroutine exchange_node(n1,n2)
   use BEM3d
   use struct_type
   implicit none
   integer::i,j
   integer,intent(in)::n1,n2
   type(nodal_point)::dnode
!----------------------------------------------------
   !--- exchange node data ---
   dnode=node(n1)
   node(n1)=node(n2)
   node(n2)=dnode
   !--- exchange node index in element ---
   do i=1,size(node(n1)%iel,1)
      do j=1,3
         !prevent overwriting
         if(el(node(n1)%iel(i,1))%ind(j) == n2) el(node(n1)%iel(i,1))%ind(j)=-100
      end do
   end do
   do i=1,size(node(n2)%iel,1)
      do j=1,3
         if(el(node(n2)%iel(i,1))%ind(j) == n1) el(node(n2)%iel(i,1))%ind(j)=n2
      end do
   end do
   do i=1,size(node(n1)%iel,1)
      do j=1,3
         if(el(node(n1)%iel(i,1))%ind(j) == -100) el(node(n1)%iel(i,1))%ind(j)=n1
      end do
   end do
   end subroutine exchange_node
!=======================================================================
!=======================================================================
!=======================================================================