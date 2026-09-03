subroutine check_norm(Ixz1,Ixz2,norm)
  implicit none
  integer::i
  real(kind(0d0))::check
  real(kind(0d0)),intent(out)::norm
  real(kind(0d0)),intent(in)::Ixz1(5,10),Ixz2(5,10)
  !================================================================
  check=0.d0; norm=0.d0
  do i=1,3
     norm=norm+Ixz1(1,i)**2
  end do
  do i=1,10
     norm=norm+Ixz1(3,i)**2+Ixz1(5,i)**2
  end do
  norm=sqrt(norm)
  do i=1,3
     check=check+(Ixz1(1,i)-Ixz2(1,i))**2
  end do
  do i=1,10
     check=check+(Ixz1(3,i)-Ixz2(3,i))**2+(Ixz1(5,i)-Ixz2(5,i))**2
  end do
  check=sqrt(check)
  norm=check/norm
end subroutine check_norm
