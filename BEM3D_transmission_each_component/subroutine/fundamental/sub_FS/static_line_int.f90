subroutine static_line_int(xx,y1,y2,id,Sigma_ijk)
   use elast_parameter
   use math_cst, only: pi
   implicit none
   integer::i,j,k,n,m,is,iq
   integer,intent(in)::id
   real(kind=8)::RR,leng,nnn,nne,nee,eee,cst
   real(kind=8),dimension(2)::ss,pm
   real(kind=8),dimension(3)::nn,ee,mm,xy1,ri,ris
   real(kind=8),dimension(3),intent(in)::xx,y1,y2
   real(kind=8),dimension(3,3)::ri_ip
   real(kind=8),dimension(3,3,3)::rijk,rijks
   real(kind=8),dimension(3,3,3,3)::rijk_ip
   real(kind=8),dimension(3,3,3,3),intent(out)::Sigma_ijk
!============================================
   ee(:)=y2(:)-y1(:)
   leng=dsqrt(ee(1)**2+ee(2)**2+ee(3)**2)
   ee(:)=ee(:)/leng
   ss(1)=dot_product(ee,y1-xx)
   ss(2)=dot_product(ee,y2-xx)
   xy1(:)=xx(:)-y1(:)
   call out_product(mm,ee,xy1)
   leng=dsqrt(mm(1)**2+mm(2)**2+mm(3)**2)
   mm(:)=mm(:)/leng
   call out_product(nn,ee,mm)
   RR=dot_product(nn,y1-xx)
   pm(1)=-1.0d0; pm(2)=1.0d0
   ri(:)=0.0d0; ris(:)=0.0d0; rijk(:,:,:)=0.0d0; rijks(:,:,:)=0.0d0
   do is=1,2
      do i=1,3
         ri(i)=ri(i)+pm(is)*(-RR*ee(i)+ss(is)*nn(i))/(RR*dsqrt(RR**2+ss(is)**2))
         ris(i)=ris(i)+pm(is)*((-RR*nn(i)-ss(is)*ee(i))/dsqrt(RR**2+ss(is)**2)+ee(i)*dlog(ss(is)+dsqrt(RR**2+ss(is)**2)))
      end do
      do i=1,3
         do j=1,3
            do k=1,3
               nnn=nn(i)*nn(j)*nn(k)
               nne=nn(i)*nn(j)*ee(k)+nn(j)*nn(k)*ee(i)+nn(k)*nn(i)*ee(j)
               nee=nn(i)*ee(j)*ee(k)+nn(j)*ee(k)*ee(i)+nn(k)*ee(i)*ee(j)
               eee=ee(i)*ee(j)*ee(k)
               rijk(i,j,k)=rijk(i,j,k)+pm(is)*( ((nee+2.0d0*nnn)*(ss(is)**3)-3.0d0*eee*RR*(ss(is)**2)&
                  &+3.0d0*nnn*(RR**2)*ss(is)-(2.0d0*eee+nne)*(RR**3))/(3.0d0*RR*((RR**2+ss(is)**2)**1.5d0)) )
               rijks(i,j,k)=rijks(i,j,k)+pm(is)*( dsqrt(RR**2+ss(is)**2)&
                  &*(((nee-nnn)*(RR**3)+(eee-nne)*(RR**2)*ss(is))/(3.0d0*((RR**2+ss(is)**2)**2))&
                  &+(-3.0d0*nee*RR+(nne-4.0d0*eee)*ss(is))/(3.0d0*(RR**2+ss(is)**2)))&
                  &+eee*dlog(ss(is)+dsqrt(RR**2+ss(is)**2)) )
            end do
         end do
      end do
   end do
!----------------------------
   select case(id)
   case(1)
      ri_ip(:,1)=(ris(:)-ss(2)*ri(:))/(ss(1)-ss(2))
      ri_ip(:,2)=(ris(:)-ss(1)*ri(:))/(ss(2)-ss(1))
      ri_ip(:,3)=0.0d0
      rijk_ip(:,:,:,1)=(rijks(:,:,:)-ss(2)*rijk(:,:,:))/(ss(1)-ss(2))
      rijk_ip(:,:,:,2)=(rijks(:,:,:)-ss(1)*rijk(:,:,:))/(ss(2)-ss(1))
      rijk_ip(:,:,:,3)=0.0d0
   case(2)
      ri_ip(:,1)=0.0d0
      ri_ip(:,2)=(ris(:)-ss(2)*ri(:))/(ss(1)-ss(2))
      ri_ip(:,3)=(ris(:)-ss(1)*ri(:))/(ss(2)-ss(1))
      rijk_ip(:,:,:,1)=0.0d0
      rijk_ip(:,:,:,2)=(rijks(:,:,:)-ss(2)*rijk(:,:,:))/(ss(1)-ss(2))
      rijk_ip(:,:,:,3)=(rijks(:,:,:)-ss(1)*rijk(:,:,:))/(ss(2)-ss(1))
   case(3)
      ri_ip(:,1)=(ris(:)-ss(1)*ri(:))/(ss(2)-ss(1))
      ri_ip(:,2)=0.0d0
      ri_ip(:,3)=(ris(:)-ss(2)*ri(:))/(ss(1)-ss(2))
      rijk_ip(:,:,:,1)=(rijks(:,:,:)-ss(1)*rijk(:,:,:))/(ss(2)-ss(1))
      rijk_ip(:,:,:,2)=0.0d0
      rijk_ip(:,:,:,3)=(rijks(:,:,:)-ss(2)*rijk(:,:,:))/(ss(1)-ss(2))
   end select
!----------------------------
   cst=-1.0d0/(8.0d0*pi*(1.0d0-nu))
   do m=1,3
      do i=1,3
         do j=1,3
            do iq=1,3
               Sigma_ijk(m,i,j,iq)=cst*((1.0d0-2.0d0*nu)*&
                  &(delta(m,i)*ri_ip(j,iq)+delta(m,j)*ri_ip(i,iq)-delta(i,j)*ri_ip(m,iq))&
                  &+3.0d0*rijk_ip(m,i,j,iq))
            end do
         end do
      end do
   end do
   end subroutine static_line_int
