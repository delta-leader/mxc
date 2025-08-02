subroutine zgmresk_BEM(n,a,x,b,id_ini,x0,id)
   !This subroutine solves complex matrix equation "a(n,n) x(n) = b(n)".
   !This subroutine outputs the residual norm normalized by |b| on the file "fort."id"".
   !If you use intial vector, "id_ini" must be equal to 1.
   !"x0" is initial vector.
   implicit none
   integer::m,i,j,itr,l
   integer,intent(in)::n,id,id_ini
   integer,parameter::k=1500
   integer,parameter::itrmax=1
   real(kind(0d0)),parameter::eps=1.0e-10
   real(kind(0d0))::bnorm,csb
   real(kind(0d0)),dimension(:)::c(n)
   complex(kind(0d0))::r0,tmp
   complex(kind(0d0)),dimension(:),allocatable::r,y,s,e,jacb,dum
   complex(kind(0d0)),dimension(:,:),allocatable::v,h,t
   complex(kind(0d0)),dimension(n),intent(in)::b,x0
   complex(kind(0d0)),dimension(n),intent(out)::x
   complex(kind(0d0)),dimension(n,n),intent(in)::a
!====================================================================
   allocate(r(n),y(n),s(n),e(n+1),jacb(n),dum(n),v(n,k+1),h(n+1,k),t(n,k))
   !$OMP workshare
   r(:)=b(:)
   !$OMP end workshare
!--- calculate |b| (i.e. \sqrt{\sum_{i=1}^{n} |b_i|^2}) ---
   bnorm=0.0d0
   !$OMP parallel do reduction(+:bnorm)
   do i=1,n
      bnorm=bnorm+dreal(b(i))**2+dimag(b(i))**2
   end do
   !$OMP end parallel do
   bnorm=dsqrt(bnorm)
!----------------------------------------------------------
   if (id_ini == 1)then
      !$OMP workshare
      x(:)=x0(:)
      !$OMP end workshare
!      r(:)=r(:)-matmul(a(:,:),x0(:))
      call zgemv('N',n,n,(-1.0d0,0.0d0),a,n,x0,1,(1.0d0,0.0d0),r,1)
   else
      !$OMP workshare
      x(:)=0.0d0
      !$OMP end workshare
   end if
!--- preconditioning matrix of point Jacobi ---
   !$OMP parallel do
   do i=1,n
      jacb(i)=1.0d0/a(i,i)
   end do
   !$OMP end parallel do
!----------------------------------------------
   do itr=1,itrmax
      !$OMP workshare
      e(:)=0.0d0
      r(:)=r(:)*jacb(:) !point Jacobi
      !$OMP end workshare
      call omp_dot_product(n,r(:),r(:),tmp)
      !$OMP single
      r0=cdsqrt(tmp)
      !$OMP end single
      if(r0 == 0.0d0) go to 10
      !$OMP workshare
      v(:,1)=r(:)/r0
      !$OMP end workshare
      !$OMP single
      e(1)=r0
      !$OMP end single
      do j=1,k
!--- calculate matrix-vector product ("zgemv" is Blas routine and faster than "matmul") ---
!         call omp_matmul(n,a(:,:),v(:,j),v(:,j+1))  !\tilde{v}_{j+1} = A v_j
!         v(:,j+1)=matmul(a,v(:,j))
         call zgemv('N',n,n,(1.0d0,0.0d0),a,n,v(:,j),1,(0.0d0,0.0d0),dum,1)
!         call h_matrix_vector(v(:,j),dum)
!         do i=1,n
!            write(33,*)dum(i)
!         end do
!         stop
!------------------------------------------------------------------------------------------
         do i=1,n
            v(i,j+1)=dum(i)*jacb(i) !point Jacobi
         end do
         do i=1,j
            call omp_dot_product(n,v(:,j+1),v(:,i),h(i,j))
            !$OMP do
            do l=1,n
               v(l,j+1)=v(l,j+1)-h(i,j)*v(l,i)
            end do
            !$OMP end do
         end do
         call omp_dot_product(n,v(:,j+1),v(:,j+1),tmp)
         h(j+1,j)=cdsqrt(tmp)
         !$OMP workshare
         v(:,j+1)=v(:,j+1)/h(j+1,j)
         !$OMP end workshare
         !$OMP single
!--------- for complex ------
         do i=1,j-1
            tmp=h(i,j)
            h(i,j)=c(i)*tmp-dconjg(s(i))*h(i+1,j)
            h(i+1,j)=s(i)*tmp+c(i)*h(i+1,j)
         end do
         csb=dsqrt(cdabs(h(j,j))**2+cdabs(h(j+1,j))**2)
         if(csb /= 0.0d0)then
            c(j)=cdabs(h(j,j))/csb
            s(j)=-h(j+1,j)/h(j,j)*c(j)
            h(j,j)=c(j)*h(j,j)-dconjg(s(j))*h(j+1,j)
            e(j+1)=s(j)*e(j)
            e(j)=c(j)*e(j)
            h(j+1,j)=0.0d0
         end if
!----------------------------
         !$OMP end single
         write(id,*)'itr/step/err=',itr,j,cdabs(e(j+1))/bnorm
         if(j==k .or. cdabs(e(j+1)) <= eps*bnorm)then
            y(j)=e(j)/h(j,j)
            !$OMP single
            do i=j-1,1,-1
               y(i)=e(i)
               do m=i+1,j
                  y(i)=y(i)-h(i,m)*y(m)
               end do
               y(i)=y(i)/h(i,i)
            end do
            !$OMP end single
            !$OMP do
            do i=1,j
               x=x+y(i)*v(:,i)
            end do
            !$OMP end do
            if(cdabs(e(j+1)) <= eps*bnorm) write(555,*)j
            if(cdabs(e(j+1)) <= eps*bnorm) go to 10
         end if
      end do
   end do
10 continue
   deallocate(r,y,s,e,jacb,dum,v,h,t)
   end subroutine zgmresk_BEM
!====================================================================
subroutine omp_dot_product(n,a,b,s)
   implicit none
   integer::i,n
   complex(kind=8)::s
   complex(kind=8),dimension(:)::a(n),b(n)
!====================================================================
   s=0.0d0
   !$OMP do reduction(+:s)
   do i=1,n
      s=s+a(i)*b(i)
   end do
   !$OMP end do
   end subroutine omp_dot_product
!====================================================================
subroutine omp_matmul(n,a,x,b)
   implicit none
   integer::n,i,j
   complex(kind=8),dimension(:)::x(n),b(n)
   complex(kind=8),dimension(:,:)::a(n,n)
!====================================================================
   !$OMP workshare
   b(:)=0.0d0
   !$OMP end workshare
   !$OMP parallel do
   do i=1,n
      do j=1,n
         b(i)=b(i)+a(i,j)*x(j)
      end do
   end do
   !$OMP end parallel do
   end subroutine omp_matmul
