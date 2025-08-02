subroutine elastUij_Sigma_dynamic(x,y,cl,ct,zs,duij,dsigma)
   use math_cst, only: pi_4
   use elast_parameter
   implicit none
   integer::i,j,k
   real(kind(0d0))::rr(4),r_y(3)
   complex(kind(0d0))::zsl(4),zst(4),zcst
   complex(kind(0d0))::zu_part(2),zs_part(3),zal_L,zal_T,zslr,zstr,zbe_L,zbe_T
   !---
   real(kind(0d0)),intent(in)::cl,ct
   real(kind(0d0)),dimension(3),intent(in)::x,y
   complex(kind(0d0)),intent(in)::zs
   complex(kind(0d0)),dimension(3,3),intent(out)::duij
   complex(kind(0d0)),dimension(3,3,3),intent(out)::dsigma
!=============================================================================
interface
subroutine Exp_sr_series_BEM(zal_L,zal_T,zbe_L,zbe_T,zslr,zstr)
   ! calculate ( e^{-sr} -1 +sr -(sr)^2/2 ) as zal
   ! calculate ( e^{-sr} -1 +sr -(sr)^2/2 +(sr)^3/6 ) as zbe
   implicit none
   complex(kind(0d0)),intent(in)::zslr,zstr
   complex(kind(0d0)),intent(out)::zal_L,zal_T,zbe_L,zbe_T
   end subroutine Exp_sr_series_BEM
end interface
!=============================================================================
   rr(1)=sqrt(dot_product(x-y,x-y))
   r_y(:)=(y(:)-x(:))/rr(1)
   zsl(1)=zs/cl
   zst(1)=zs/ct
   do i=1,3
      rr(i+1)=rr(i)*rr(1)
      zsl(i+1)=zsl(i)*zsl(1)
      zst(i+1)=zst(i)*zst(1)
   end do
   zslr=zsl(1)*rr(1)
   zstr=zst(1)*rr(1)
   call Exp_sr_series_BEM(zal_L,zal_T,zbe_L,zbe_T,zslr,zstr)
   zal_L=zal_L/rr(3)
   zal_T=zal_T/rr(3)
   zbe_L=zbe_L/rr(4)
   zbe_T=zbe_T/rr(4)
   !---
   zu_part(1)=(zsl(3)-zst(3))/2.d0+rr(1)*(zsl(4)-zst(4))/2.d0&
      &+(3.d0+3.d0*zslr+zsl(2)*rr(2))*zal_L&
      &-(3.d0+3.d0*zstr+zst(2)*rr(2))*zal_T
   zu_part(2)=(zsl(3)+zst(3))/2.d0-zst(4)*rr(1)/2.d0&
      &+(1.d0+zslr)*zal_L&
      &-(1.d0+zstr+zst(2)*rr(2))*zal_T
   !---
   zs_part(1)=-zslr*zsl(4)/3.d0+(6.d0+6.d0*zslr+2.d0*zsl(2)*rr(2))*zbe_L&
      &+zst(4)/2.d0+zstr*zstr*zst(4)/6.d0-(6.d0+6.d0*zstr+3.d0*zst(2)*rr(2)+zst(3)*rr(3))*zbe_T
   zs_part(2)=zsl(2)*zst(2)/2.d0*(1.d0-zslr)-zstr*zst(1)*(1.d0+zslr)*zal_L&
      &-zsl(4)+zslr*zsl(4)/3.d0-zslr*zslr*zsl(4)/3.d0+(6.d0+6.d0*zslr+4.d0*zsl(2)*rr(2)+2.d0*zsl(3)*rr(3))*zbe_L&
      &+zstr*zst(4)/3.d0-(6.d0+6.d0*zstr+2.d0*zst(2)*rr(2))*zbe_T
   zs_part(3)=zsl(4)+zslr*zsl(4)+zslr*zslr*zsl(4)/3.d0-2.d0*(15.d0+15.d0*zslr+6.d0*zsl(2)*rr(2)+zsl(3)*rr(3))*zbe_L&
      &-zst(4)-zstr*zst(4)-zstr*zstr*zst(4)/3.d0+2.d0*(15.d0+15.d0*zstr+6.d0*zst(2)*rr(2)+zst(3)*rr(3))*zbe_T
   !---
   zcst=1.d0/(zst(2)*pi_4)
   do i=1,3
      do j=1,3
         duij(i,j)=zcst*( zu_part(1)*r_y(i)*r_y(j)-zu_part(2)*delta(i,j) )/mu!not non-dimensionalization
         do k=1,3
            dsigma(i,j,k)=zcst*( zs_part(1)*(delta(i,j)*r_y(k)+delta(i,k)*r_y(j))&
               &+zs_part(2)*r_y(i)*delta(j,k)+zs_part(3)*r_y(i)*r_y(j)*r_y(k) )
         end do
      end do
   end do
   end subroutine elastUij_Sigma_dynamic
!=============================================================================
!=============================================================================
!=============================================================================
subroutine elastSigma_dynamic(x,y,cl,ct,zs,dsigma)
   use math_cst, only: pi_4
   use elast_parameter
   implicit none
   integer::i,j,k
   real(kind(0d0))::rr(4),r_y(3)
   complex(kind(0d0))::zsl(4),zst(4),zcst
   complex(kind(0d0))::zs_part(3),zal_L,zal_T,zslr,zstr,zbe_L,zbe_T
   !---
   real(kind(0d0)),intent(in)::cl,ct
   real(kind(0d0)),dimension(3),intent(in)::x,y
   complex(kind(0d0)),intent(in)::zs
   complex(kind(0d0)),dimension(3,3,3),intent(out)::dsigma
!=============================================================================
interface
subroutine Exp_sr_series_BEM(zal_L,zal_T,zbe_L,zbe_T,zslr,zstr)
   ! calculate ( e^{-sr} -1 +sr -(sr)^2/2 ) as zal
   ! calculate ( e^{-sr} -1 +sr -(sr)^2/2 +(sr)^3/6 ) as zbe
   implicit none
   complex(kind(0d0)),intent(in)::zslr,zstr
   complex(kind(0d0)),intent(out)::zal_L,zal_T,zbe_L,zbe_T
   end subroutine Exp_sr_series_BEM
end interface
!=============================================================================
   rr(1)=sqrt(dot_product(x-y,x-y))
   r_y(:)=(y(:)-x(:))/rr(1)
   zsl(1)=zs/cl
   zst(1)=zs/ct
   do i=1,3
      rr(i+1)=rr(i)*rr(1)
      zsl(i+1)=zsl(i)*zsl(1)
      zst(i+1)=zst(i)*zst(1)
   end do
   zslr=zsl(1)*rr(1)
   zstr=zst(1)*rr(1)
   call Exp_sr_series_BEM(zal_L,zal_T,zbe_L,zbe_T,zslr,zstr)
   zal_L=zal_L/rr(3)
   zal_T=zal_T/rr(3)
   zbe_L=zbe_L/rr(4)
   zbe_T=zbe_T/rr(4)
   !---
   zs_part(1)=-zslr*zsl(4)/3.d0+(6.d0+6.d0*zslr+2.d0*zsl(2)*rr(2))*zbe_L&
      &+zst(4)/2.d0+zstr*zstr*zst(4)/6.d0-(6.d0+6.d0*zstr+3.d0*zst(2)*rr(2)+zst(3)*rr(3))*zbe_T
   zs_part(2)=zsl(2)*zst(2)/2.d0*(1.d0-zslr)-zstr*zst(1)*(1.d0+zslr)*zal_L&
      &-zsl(4)+zslr*zsl(4)/3.d0-zslr*zslr*zsl(4)/3.d0+(6.d0+6.d0*zslr+4.d0*zsl(2)*rr(2)+2.d0*zsl(3)*rr(3))*zbe_L&
      &+zstr*zst(4)/3.d0-(6.d0+6.d0*zstr+2.d0*zst(2)*rr(2))*zbe_T
   zs_part(3)=zsl(4)+zslr*zsl(4)+zslr*zslr*zsl(4)/3.d0-2.d0*(15.d0+15.d0*zslr+6.d0*zsl(2)*rr(2)+zsl(3)*rr(3))*zbe_L&
      &-zst(4)-zstr*zst(4)-zstr*zstr*zst(4)/3.d0+2.d0*(15.d0+15.d0*zstr+6.d0*zst(2)*rr(2)+zst(3)*rr(3))*zbe_T
   !---
   zcst=1.d0/(zst(2)*pi_4)
   do i=1,3
      do j=1,3
         do k=1,3
            dsigma(i,j,k)=zcst*( zs_part(1)*(delta(i,j)*r_y(k)+delta(i,k)*r_y(j))&
               &+zs_part(2)*r_y(i)*delta(j,k)+zs_part(3)*r_y(i)*r_y(j)*r_y(k) )
         end do
      end do
   end do
   end subroutine elastSigma_dynamic
!=============================================================================
!===========================================================================
!===========================================================================
subroutine Uij_Sigma_lap(x,y,cl,ct,zs,zUij,zSigma)
   use math_cst
   use elast_parameter
   implicit none
   integer::i,j,k
   real(kind=8)::rr,rr2,rr3,rr4,check,gamma2,gamma4
   real(kind=8),intent(in)::cl,ct
   real(kind=8),dimension(3)::r_y
   real(kind=8),dimension(3),intent(in)::x,y
   complex(kind=8),intent(in)::zs
   complex(kind=8)::zsl,zsl2,zsl3,zsl4,zst,zst2,zst3,zst4
   complex(kind=8)::zeslr,zestr,sum2,sum4l,sum4t,zL1,zL2,zL3,zT1,zT2,zT3
   complex(kind=8),dimension(2)::zDyn
   complex(kind=8),dimension(3,3),intent(out)::zUij
   complex(kind=8),dimension(3,3,3),intent(out)::zSigma
!===========================================================================
   rr=dsqrt((y(1)-x(1))**2+(y(2)-x(2))**2+(y(3)-x(3))**2)
   r_y(:)=(y(:)-x(:))/rr
   rr2=rr*rr
   rr3=rr2*rr
   rr4=rr2*rr2
   zsl=zs/cl
   zst=zs/ct
   zsl2=zsl*zsl
   zsl3=zsl2*zsl
   zsl4=zsl2*zsl2
   zst2=zst*zst
   zst3=zst2*zst
   zst4=zst2*zst2
   check=cdabs(zst*rr)
   zeslr=cdexp(-zsl*rr)
   zestr=cdexp(-zst*rr)
   zL1=(3.0d0/(rr4)+3.0d0*zsl/(rr3)+zsl2/(rr2))*zeslr
   zT1=(3.0d0/(rr4)+3.0d0*zst/(rr3)+zst2/(rr2))*zestr
   zL2=(15.0d0/(rr4)+15.0d0*zsl/(rr3)+6.0d0*zsl2/(rr2)+zsl3/rr)*zeslr
   zT2=(15.0d0/(rr4)+15.0d0*zst/(rr3)+6.0d0*zst2/(rr2)+zst3/rr)*zestr
   zL3=(zsl2/(rr2)+zsl3/rr)*zeslr
   zT3=(zst2/(rr2)+zst3/rr)*zestr
   do i=1,3
      do j=1,3
         zUij(i,j)=( (zL1*rr-zT1*rr)*r_y(i)*r_y(j)&
            & -(zL3/zsl2/rr-zT3/zst-1.0d0/rr3*zestr)*delta(i,j))/zst2
      end do
   end do
   do i=1,3
      do j=1,3
         do k=1,3
            zSigma(i,j,k)=-zL3/zsl2*r_y(i)*delta(j,k) +( &
               & (2.0d0*zL1-(2.0d0*zT1+zT3))*(delta(i,j)*r_y(k)+delta(i,k)*r_y(j)) &
               & +(2.0d0*zL1+2.0d0*zL3-2.0d0*zT1)*r_y(i)*delta(j,k) &
               & -2.0d0*(zL2-zT2)*r_y(i)*r_y(j)*r_y(k)&
!               & -zsl2/rr2*(r_y(i)*delta(j,k)-r_y(j)*delta(i,k)-r_y(k)*delta(i,j))&
!               & +3.0d0/rr2*(zst2-zsl2)*r_y(i)*r_y(j)*r_y(k)&
               & )/zst2
         end do
      end do
   end do
   zUij(:,:)=zUij(:,:)/(pi_4*mu)
   zSigma(:,:,:)=zSigma(:,:,:)/pi_4
   end subroutine Uij_Sigma_lap
!===========================================================================
!===========================================================================
!===========================================================================
subroutine Sigma_lap(x,y,cl,ct,zs,zSigma)
   use math_cst
   use elast_parameter
   implicit none
   integer::i,j,k
   real(kind=8)::rr,rr2,rr3,rr4,check,gamma2,gamma4
   real(kind=8),intent(in)::cl,ct
   real(kind=8),dimension(3)::r_y
   real(kind=8),dimension(3),intent(in)::x,y
   complex(kind=8),intent(in)::zs
   complex(kind=8)::zsl,zsl2,zsl3,zsl4,zst,zst2,zst3,zst4
   complex(kind=8)::zeslr,zestr,sum2,sum4l,sum4t,zL1,zL2,zL3,zT1,zT2,zT3
   complex(kind=8),dimension(2)::zDyn
   complex(kind=8),dimension(3,3,3),intent(out)::zSigma
!===========================================================================
   rr=dsqrt((y(1)-x(1))**2+(y(2)-x(2))**2+(y(3)-x(3))**2)
   r_y(:)=(y(:)-x(:))/rr
   rr2=rr*rr
   rr3=rr2*rr
   rr4=rr2*rr2
   zsl=zs/cl
   zst=zs/ct
   zsl2=zsl*zsl
   zsl3=zsl2*zsl
   zsl4=zsl2*zsl2
   zst2=zst*zst
   zst3=zst2*zst
   zst4=zst2*zst2
   check=cdabs(zst*rr)
   zeslr=cdexp(-zsl*rr)
   zestr=cdexp(-zst*rr)
   zL1=(3.0d0/(rr4)+3.0d0*zsl/(rr3)+zsl2/(rr2))*zeslr
   zT1=(3.0d0/(rr4)+3.0d0*zst/(rr3)+zst2/(rr2))*zestr
   zL2=(15.0d0/(rr4)+15.0d0*zsl/(rr3)+6.0d0*zsl2/(rr2)+zsl3/rr)*zeslr
   zT2=(15.0d0/(rr4)+15.0d0*zst/(rr3)+6.0d0*zst2/(rr2)+zst3/rr)*zestr
   zL3=(zsl2/(rr2)+zsl3/rr)*zeslr
   zT3=(zst2/(rr2)+zst3/rr)*zestr
   do i=1,3
      do j=1,3
         do k=1,3
            zSigma(i,j,k)=-zL3/zsl2*r_y(i)*delta(j,k) +( &
               & (2.0d0*zL1-(2.0d0*zT1+zT3))*(delta(i,j)*r_y(k)+delta(i,k)*r_y(j)) &
               & +(2.0d0*zL1+2.0d0*zL3-2.0d0*zT1)*r_y(i)*delta(j,k) &
               & -2.0d0*(zL2-zT2)*r_y(i)*r_y(j)*r_y(k)&
               & )/zst2
         end do
      end do
   end do
   zSigma(:,:,:)=zSigma(:,:,:)/pi_4
   end subroutine Sigma_lap
