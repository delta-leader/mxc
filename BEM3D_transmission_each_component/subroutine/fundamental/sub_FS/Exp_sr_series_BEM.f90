subroutine Exp_sr_series_BEM(zal_L,zal_T,zbe_L,zbe_T,zslr,zstr)
   ! calculate ( e^{-sr} -1 +sr -(sr)^2/2 ) as zal
   ! calculate ( e^{-sr} -1 +sr -(sr)^2/2 +(sr)^3/6 ) as zbe
   implicit none
   integer::i,iw
   real(kind(0d0))::dsr,dom
   complex(kind(0d0))::z,z0,zres,zsr,zres2
   complex(kind(0d0)),intent(in)::zslr,zstr
   complex(kind(0d0)),intent(out)::zal_L,zal_T,zbe_L,zbe_T
!=================================================
   do iw=1,2
      select case(iw)
      case(1)
         zsr=zslr
      case(2)
         zsr=zstr
      end select
      dsr=cdabs(zsr)
      if(dsr > 1.0d-3)then
         zres=exp(-zsr)-(1.d0-zsr+(zsr**2)/2.d0)
         zres2=exp(-zsr)-(1.d0-zsr+(zsr**2)/2.d0-(zsr**3)/6.d0)
      else
         zres=0.d0
         dom=2.d0
         do i=3,10
            dom=dom*dble(i)
            z=((-zsr)**i)/dom
            if(i == 3) z0=z
            zres=zres+z
            if(cdabs(z/z0) < 1.d-16) exit
         end do
         zres2=0.d0
         dom=6.d0
         do i=4,10
            dom=dom*dble(i)
            z=((-zsr)**i)/dom
            if(i == 3) z0=z
            zres2=zres2+z
            if(cdabs(z/z0) < 1.d-16) exit
         end do
      end if
      select case(iw)
      case(1)
         zal_L=zres
         zbe_L=zres2
      case(2)
         zal_T=zres
         zbe_T=zres2
      end select
   end do
   end subroutine Exp_sr_series_BEM
!=================================================
