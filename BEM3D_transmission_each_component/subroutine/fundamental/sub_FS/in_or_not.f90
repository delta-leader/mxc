subroutine in_or_not(xi,zeta,eta,in_id,pm)
   implicit none
   integer::i,j
   integer,intent(out)::in_id
   real(kind=8)::mdum,check
   real(kind=8),intent(in)::eta
   real(kind=8),dimension(3)::xx,dum,alpha,beta,mid
   real(kind=8),dimension(:,:)::y(4,3)
   real(kind=8),dimension(3),intent(out)::pm
   real(kind=8),dimension(4),intent(in)::xi,zeta
!---------------------------------------------------------------------------
   xx(:)=0.0d0

   y(1,1)=xi(1)
   y(1,2)=zeta(1)
   y(1,3)=eta
   y(2,1)=xi(2)
   y(2,2)=zeta(2)
   y(2,3)=eta
   y(3,1)=xi(3)
   y(3,2)=zeta(3)
   y(3,3)=eta
   y(4,:)=y(1,:)
   mid(:)=(y(1,:)+y(2,:)+y(3,:))/3.0d0
   do i=1,3
      alpha(:)=y(i,:)-xx(:)
      beta(:)=y(i+1,:)-xx(:)
      dum(i)=alpha(1)*beta(2)-alpha(2)*beta(1)
      alpha(:)=y(i,:)-mid(:)
      beta(:)=y(i+1,:)-mid(:)
      mdum=alpha(1)*beta(2)-alpha(2)*beta(1)
      check=dum(i)*mdum
      if(check >= 0.0d0)then
         pm(i)=1.0d0
      else
         pm(i)=-1.0d0
      end if
   end do
   in_id=0
   if(dum(1) > 0.0d0 .and.&
      &dum(2) > 0.0d0 .and.&
      &dum(3) > 0.0d0) in_id=1
   if(dum(1) < 0.0d0 .and.&
      &dum(2) < 0.0d0 .and.&
      &dum(3) < 0.0d0) in_id=1
!   if(dabs(dum(1)) < 1.0d-10 .or.&
!      &dabs(dum(2)) < 1.0d-10 .or.&
!      &dabs(dum(3)) < 1.0d-10) in_id=2
!
!   !on the vertex
!   if(dabs(dum(1)) < 1.0d-10 .and. dabs(dum(2)) < 1.0d-10)then
!      in_id=3
!   else if(dabs(dum(2)) < 1.0d-10 .and. dabs(dum(3)) < 1.0d-10)then
!      in_id=4
!   else if(dabs(dum(3)) < 1.0d-10 .and. dabs(dum(1)) < 1.0d-10)then
!      in_id=5
!   end if
!
!   !on the line
!   if(dabs(dum(1)) < 1.0d-10 .and.&
!      &dum(2) > 0.0d0 .and.&
!      &dum(3) > 0.0d0)then
!      in_id=2
!   else if(dabs(dum(1)) < 1.0d-10 .and.&
!      &dum(2) < 0.0d0 .and.&
!      &dum(3) < 0.0d0)then
!      in_id=2
!
!   else if(dabs(dum(2)) < 1.0d-10 .and.&
!      &dum(3) > 0.0d0 .and.&
!      &dum(1) > 0.0d0)then
!      in_id=2
!   else if(dabs(dum(2)) < 1.0d-10 .and.&
!      &dum(3) < 0.0d0 .and.&
!      &dum(1) < 0.0d0)then
!      in_id=2
!
!   else if(dabs(dum(3)) < 1.0d-10 .and.&
!      &dum(1) > 0.0d0 .and.&
!      &dum(2) > 0.0d0)then
!      in_id=2
!   else if(dabs(dum(3)) < 1.0d-10 .and.&
!      &dum(1) < 0.0d0 .and.&
!      &dum(2) < 0.0d0)then
!      in_id=2
!
!   end if

   end subroutine in_or_not
!===========================================
