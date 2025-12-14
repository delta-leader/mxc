subroutine elast3d_UTij_static(xx_in,y1,y2,y3,on_id,nvec,ust,tst)
   !ust(i,j)= \mu \int Uij^{st} (x,y) dS_y
   !tst(i,j,iq)=\int Tij^{st} (x,y) phi_{iq}(y) dS_y
   use math_cst
   use elast_parameter
   implicit none
   integer::i,j,k,l,iq,in_id
   real(kind(0d0))::eta,area,gam1,gam2,gam3
   real(kind(0d0))::dtheta,pme,const,nu2,del,cst1,cst2,int3,rirj,rinjq(3,3,3)
   real(kind(0d0))::prho(2)
   real(kind(0d0)),dimension(:)::xi(4),zeta(4),rho(4),gd(2),lcal_sub(2),del_sub(2)
   real(kind(0d0)),dimension(3)::gami,chii,trho,Lcal,deltai,pm,dx,ay,by,cy
   real(kind(0d0)),dimension(3)::alpha,vec12,vec23,vec31,qx,ee1,ee2,ee3,a_phi,b_phi
   real(kind(0d0)),dimension(:,:)::pp(3,4),qq(3,4),h_phi(3,3),ph(3,3)
   real(kind(0d0)),dimension(:,:)::Ixz(5,10)
   integer,intent(in)::on_id  !1:x is on the element, 0:not
   real(kind(0d0)),dimension(3),intent(in)::xx_in,y1,y2,y3,nvec
   real(kind(0d0)),dimension(3,3),intent(out)::ust
   real(kind(0d0)),dimension(3,3,3),intent(out)::tst
   !---
   integer::ic,icheck
   real(kind(0d0))::norm,check,ep_ixz,cvec(3),yc(3),dd
   real(kind(0d0))::Ixz_num(5,10),xx(3),ep_shift,Ixz_store(3,5,10),val(3)
!===================================================
interface
subroutine check_norm(Ixz1,Ixz2,norm)
   implicit none
   real(kind(0d0)),intent(out)::norm
   real(kind(0d0)),intent(in)::Ixz1(5,10),Ixz2(5,10)
   end subroutine check_norm
end interface
!===================================================
   vec12(:)=y2(:)-y1(:)
   vec23(:)=y3(:)-y2(:)
   vec31(:)=y1(:)-y3(:)
!-----coordinate {e1,e2,e3}-----------------
   ee1(:)=vec12(:)/sqrt(vec12(1)**2+vec12(2)**2+vec12(3)**2)
   call out_product(ee3,vec12,-vec31)
   area=sqrt(ee3(1)**2+ee3(2)**2+ee3(3)**2)
   ee3(:)=ee3(:)/area
   area=0.5d0*area
   call out_product(ee2,ee3,ee1)
!-------------------------------------------
!--- cvec is perturbation vector to check in_id ---
   yc(:)=(y1(:)+y2(:)+y3(:))/3.d0-xx_in(:)
   norm=dot_product(yc,nvec)
   cvec(:)=yc(:)-norm*nvec(:)
   norm=sqrt(dot_product(cvec,cvec))
   if(norm > 1.d-10)then
      cvec(:)=cvec(:)/norm
   else
      yc(:)=y1(:)-xx_in(:)
      norm=dot_product(yc,nvec)
      cvec(:)=yc(:)-norm*nvec(:)
      norm=sqrt(dot_product(cvec,cvec))
      cvec(:)=cvec(:)/norm
   end if
!-----------------------------------
!--- for check with perturbation ---
!-----------------------------------
   icheck=0
   ep_shift=sqrt(area)*1.d-6
   do ic=1,3
      select case(ic)
      case(1)
         xx(:)=xx_in(:)
      case(2)
         xx(:)=xx_in(:)+ep_shift*cvec(:)
      case(3)
         xx(:)=xx_in(:)-ep_shift*cvec(:)
      end select
!-----------------------------------
!-----------------------------------
!-----------------------------------
   Ixz(:,:)=0.0d0
!-------------------------------------------
!---calculate xi,zeta,eta-------------------
!-------------------------------------------
   xi(1)=dot_product(y1(:)-xx(:),ee1(:))
   xi(2)=dot_product(y2(:)-xx(:),ee1(:))
   xi(3)=dot_product(y3(:)-xx(:),ee1(:))
   xi(4)=xi(1)
   zeta(1)=dot_product(y1(:)-xx(:),ee2(:))
   zeta(2)=dot_product(y2(:)-xx(:),ee2(:))
   zeta(3)=dot_product(y3(:)-xx(:),ee2(:))
   zeta(4)=zeta(1)
   eta=dot_product(y1(:)-xx(:),ee3(:))
   if(eta >= 0.0d0)then
      pme=1.0d0
   else
      pme=-1.0d0
   end if
!-------------------------------------------
   if(abs(eta) < 1.0d-10)then
      if(on_id == 1)then
         in_id=1
      else
         in_id=0
      end if
   end if
   call in_or_not(xi,zeta,eta,in_id,pm)
!   if(in_id == 2)in_id=0
!-------------------------------------------
   del=(xi(2)-xi(1))*(zeta(3)-zeta(1))
   ay(1)=-1.0d0/(xi(2)-xi(1))
   ay(2)=1.0d0/(xi(2)-xi(1))
   ay(3)=0.0d0
   by(1)=(xi(3)-xi(2))/del
   by(2)=(xi(1)-xi(3))/del
   by(3)=1.0d0/(zeta(3)-zeta(1))
   cy(1)=(xi(2)*zeta(3)-xi(3)*zeta(2))/del
   cy(2)=(xi(3)*zeta(1)-xi(1)*zeta(3))/del
   cy(3)=-zeta(1)/(zeta(3)-zeta(1))
!-------------------------------------------
   select case(in_id)
   case(0)  !out of the element
      dtheta=0.0d0
   case(1)  !in the element
      dtheta=2.0d0*pi
   case(2)  !on the line of element
      dtheta=pi
   case(3)  !on the vertex y2
      dtheta=dacos((vec12(1)*(-vec23(1))+vec12(2)*(-vec23(2))+vec12(3)*(-vec23(3)))/&
         &(dsqrt(vec12(1)**2+vec12(2)**2+vec12(3)**2)*dsqrt(vec23(1)**2+vec23(2)**2+vec23(3)**2)))
   case(4)  !on the vertex y3
      dtheta=dacos((vec23(1)*(-vec31(1))+vec23(2)*(-vec31(2))+vec23(3)*(-vec31(3)))/&
         &(dsqrt(vec23(1)**2+vec23(2)**2+vec23(3)**2)*dsqrt(vec31(1)**2+vec31(2)**2+vec31(3)**2)))
   case(5)  !on the vertex y1
      dtheta=dacos((vec31(1)*(-vec12(1))+vec31(2)*(-vec12(2))+vec31(3)*(-vec12(3)))/&
         &(dsqrt(vec31(1)**2+vec31(2)**2+vec31(3)**2)*dsqrt(vec12(1)**2+vec12(2)**2+vec12(3)**2)))
   end select
!==================================
!==================================
!--- p.v. integral ----------------
   if(on_id == 1)dtheta=0.0d0
!==================================
!==================================
   alpha(1)=0.0d0
   alpha(2)=dacos((vec12(1)*(-vec23(1))+vec12(2)*(-vec23(2))+vec12(3)*(-vec23(3)))/&
      &(dsqrt(vec12(1)**2+vec12(2)**2+vec12(3)**2)*dsqrt(vec23(1)**2+vec23(2)**2+vec23(3)**2)))
   alpha(2)=pi-alpha(2)
   alpha(3)=dacos((vec12(1)*(-vec31(1))+vec12(2)*(-vec31(2))+vec12(3)*(-vec31(3)))/&
      &(dsqrt(vec12(1)**2+vec12(2)**2+vec12(3)**2)*dsqrt(vec31(1)**2+vec31(2)**2+vec31(3)**2)))
   alpha(3)=pi+alpha(3)
   do i=1,3
      do j=1,3
         pp(i,j)=xi(j)*dcos(alpha(i))+zeta(j)*dsin(alpha(i))
         qq(i,j)=-xi(j)*dsin(alpha(i))+zeta(j)*dcos(alpha(i))
      end do
   end do
   pp(3,4)=pp(3,1)
   do i=1,3
      rho(i)=dsqrt(pp(1,i)**2+qq(1,i)**2+eta**2)
      qx(i)=qq(i,i)
      dx(i)=qx(i)**2+eta**2
   end do
   qq(3,4)=qq(3,1)
   rho(4)=rho(1)
!--------parameter-----------------------------------
   do i=1,3
      gd(1)=atan(eta*pp(i,i+1)/(qx(i)*rho(i+1)))
      gd(2)=atan(eta*pp(i,i)/(qx(i)*rho(i)))
      gami(i)=gd(1)-gd(2)
      gami(i)=gami(i)*2.d0
      !--- series expansion for chii ---
      prho(1)=pp(i,i)+rho(i)
      if(abs(prho(1))/(abs(pp(i,i))+abs(rho(i))) < 1.d-5)then
         dd=qq(i,i)**2+eta**2
         prho(1)=dd/(2.d0*abs(pp(i,i)))-(dd**2)/(8.d0*(abs(pp(i,i))**3))+(dd**3)/(16.d0*(abs(pp(i,i))**5))
      end if
      prho(2)=pp(i,i+1)+rho(i+1)
      if(abs(prho(2))/(abs(pp(i,i+1))+abs(rho(i+1))) < 1.d-5)then
         dd=qq(i,i+1)**2+eta**2
         prho(2)=dd/(2.d0*abs(pp(i,i+1)))-(dd**2)/(8.d0*(abs(pp(i,i+1))**3))+(dd**3)/(16.d0*(abs(pp(i,i+1))**5))
      end if
      chii(i)=log(prho(1)/prho(2))
      !---------------------------------
      trho(i)=rho(i)-rho(i+1)
      dd=sqrt(qx(i)**2+eta**2)
      if(dd < 1.d-5)then
         Lcal(i)=(1.d0/abs(pp(i,i))-1.d0/abs(pp(i,i+1)))&
            &+(-0.5d0/abs(pp(i,i)**3)+0.5d0/abs(pp(i,i+1)**3))*(dd**2)
         deltai(i)=(pp(i,i)/abs(pp(i,i))-pp(i,i+1)/abs(pp(i,i+1)))&
            &+(-0.5d0*pp(i,i)/abs(pp(i,i)**3)+0.5d0*pp(i,i+1)/abs(pp(i,i+1)**3))*(dd**2)
      else
         Lcal(i)=(rho(i+1)-rho(i))/(rho(i)*rho(i+1))
         deltai(i)=(rho(i+1)*pp(i,i)-rho(i)*pp(i,i+1))/(rho(i)*rho(i+1))
      end if
   end do
!----------------------------------------------------
!--------I1-OK---------------------------------------
   if(on_id == 1)then
      do i=1,3
         Ixz(1,1)=Ixz(1,1)+qx(i)*chii(i)-0.5d0*eta*gami(i)
      end do
   else
      do i=1,3
         Ixz(1,1)=Ixz(1,1)+qx(i)*chii(i)-0.5d0*eta*gami(i)
      end do
      Ixz(1,1)=Ixz(1,1)-dabs(eta)*dtheta
   end if
!----------------------------------------------------
!--------I1^xi-OK------------------------------------
   do i=1,3
      gam1=qx(i)*trho(i)*dcos(alpha(i))
      Ixz(1,2)=Ixz(1,2)+gam1-dx(i)*dsin(alpha(i))*chii(i)
   end do
   Ixz(1,2)=0.5d0*Ixz(1,2)
!----------------------------------------------------
!--------I1^zeta-OK------------------------------------
   do i=1,3
      gam1=qx(i)*trho(i)*dsin(alpha(i))
      Ixz(1,3)=Ixz(1,3)+gam1+dx(i)*dcos(alpha(i))*chii(i)
   end do
   Ixz(1,3)=0.5d0*Ixz(1,3)
!----------------------------------------------------
!--------eta*I3(regular)-OK--------------------------
   if(on_id == 1)then
      do i=1,3
         Ixz(3,1)=Ixz(3,1)+0.5d0*gami(i)
      end do
   else
      do i=1,3
         Ixz(3,1)=Ixz(3,1)+0.5d0*gami(i)
      end do
      Ixz(3,1)=Ixz(3,1)+pme*dtheta
   end if
!----------------------------------------------------
!--------I3^xi-OK------------------------------------
   do i=1,3
      Ixz(3,2)=Ixz(3,2)+chii(i)*dsin(alpha(i))
   end do
!----------------------------------------------------
!--------I3^zeta-OK----------------------------------
   do i=1,3
      Ixz(3,3)=Ixz(3,3)-chii(i)*dcos(alpha(i))
   end do
!----------------------------------------------------
!--------I3^(xi*xi)-OK-------------------------------
   do i=1,3
      gam1=trho(i)*dsin(alpha(i))*dcos(alpha(i))
      gam2=qx(i)*(dcos(alpha(i))**2)*chii(i)
      Ixz(3,4)=Ixz(3,4)+gam1+gam2
   end do
   Ixz(3,4)=Ixz(3,4)-eta*Ixz(3,1)
!----------------------------------------------------
!--------I3^(xi*zeta)-OK-----------------------------
   do i=1,3
      gam1=trho(i)*(dsin(alpha(i))**2)
      gam2=qx(i)*dsin(alpha(i))*dcos(alpha(i))*chii(i)
      Ixz(3,5)=Ixz(3,5)+gam1+gam2
   end do
!----------------------------------------------------
!--------I3^(zeta*zeta)-OK---------------------------
   do i=1,3
      gam1=-trho(i)*dsin(alpha(i))*dcos(alpha(i))
      gam2=qx(i)*(dsin(alpha(i))**2)*chii(i)
      Ixz(3,6)=Ixz(3,6)+gam1+gam2
   end do
   Ixz(3,6)=Ixz(3,6)-eta*Ixz(3,1)
!----------------------------------------------------
!--------I3^(xi*xi*xi)-OK----------------------------
   do i=1,3
      gam1=((qx(i)**2)*(dsin(alpha(i))**3)-dx(i)*dsin(alpha(i))*(1.5d0-0.5d0*(dsin(alpha(i))**2)))&
         &*chii(i)
      gam2=0.5d0*dsin(alpha(i))*(dcos(alpha(i))**2)*(pp(i,i)*rho(i)-pp(i,i+1)*rho(i+1))
      gam3=qx(i)*dcos(alpha(i))*(1.0d0-2.0d0*(dsin(alpha(i))**2))*trho(i)
      Ixz(3,7)=Ixz(3,7)+gam1+gam2+gam3
   end do
!----------------------------------------------------
!--------I3^(xi*xi*zeta)-OK------------------------
   do i=1,3
      gam1=(0.5d0*dx(i)*(dcos(alpha(i))**3)-(qx(i)**2)*(dsin(alpha(i))**2)*dcos(alpha(i)))&
         &*chii(i)
      gam2=(dcos(alpha(i))**2)*(-0.5d0*pp(i,i)*rho(i)*dcos(alpha(i))+2.0d0*qx(i)*rho(i)*dsin(alpha(i))&
         &+0.5d0*pp(i,i+1)*rho(i+1)*dcos(alpha(i))-2.0d0*qx(i)*rho(i+1)*dsin(alpha(i)))
      Ixz(3,8)=Ixz(3,8)+gam1+gam2
   end do
!----------------------------------------------------
!--------I3^(xi*zeta*zeta)-OK------------------------
   do i=1,3
      gam1=(-0.5d0*dx(i)*(dsin(alpha(i))**3)+(qx(i)**2)*dsin(alpha(i))*(dcos(alpha(i))**2))&
         &*chii(i)
      gam2=(dsin(alpha(i))**2)*(0.5d0*pp(i,i)*rho(i)*dsin(alpha(i))+2.0d0*qx(i)*rho(i)*dcos(alpha(i))&
         &-0.5d0*pp(i,i+1)*rho(i+1)*dsin(alpha(i))-2.0d0*qx(i)*rho(i+1)*dcos(alpha(i)))
      Ixz(3,9)=Ixz(3,9)+gam1+gam2
   end do
!----------------------------------------------------
!--------I3^(zeta*zeta*zeta)-OK----------------------
   do i=1,3
      gam1=(-(qx(i)**2)*(dcos(alpha(i))**3)+dx(i)*dcos(alpha(i))*(1.5d0-0.5d0*(dcos(alpha(i))**2)))&
         &*chii(i)
      gam2=-0.5d0*dcos(alpha(i))*(dsin(alpha(i))**2)*(pp(i,i)*rho(i)-pp(i,i+1)*rho(i+1))
      gam3=qx(i)*dsin(alpha(i))*(1.0d0-2.0d0*(dcos(alpha(i))**2))*trho(i)
      Ixz(3,10)=Ixz(3,10)+gam1+gam2+gam3
   end do
!----------------------------------------------------
!--------(eta^3)*I5(regular)-OK----------------------
   if(on_id == 1)then
      do i=1,3
         gam1=qx(i)/dx(i)*(pp(i,i+1)/rho(i+1)-pp(i,i)/rho(i))
         Ixz(5,1)=Ixz(5,1)+0.5d0*gami(i)-1.0d0*eta*gam1
      end do
      Ixz(5,1)=Ixz(5,1)/3.0d0
   else
      do i=1,3
         gam1=qx(i)/dx(i)*(pp(i,i+1)/rho(i+1)-pp(i,i)/rho(i))
         Ixz(5,1)=Ixz(5,1)+0.5d0*gami(i)-1.0d0*eta*gam1
      end do
      Ixz(5,1)=Ixz(5,1)+pme*dtheta
      Ixz(5,1)=Ixz(5,1)/3.0d0
   end if
!----------------------------------------------------
!--------I5^xi-OK------------------------------------
   do i=1,3
      gam1=pp(i,i+1)/rho(i+1)-pp(i,i)/rho(i)
      if(abs(gam1)/(abs(pp(i,i+1)/rho(i+1))+abs(pp(i,i)/rho(i))) < 1.d-5)then
         dd=qq(i,i)**2+eta**2
         prho(1)=-pp(i,i)*dd/(2.d0*(abs(pp(i,i))**3))&
            &+3.d0*(dd**2)/(8.d0*(pp(i,i)**3)*abs(pp(i,i)))&
            &-5.d0*(dd**3)/(16.d0*(pp(i,i)**5)*abs(pp(i,i)))
         dd=qq(i,i+1)**2+eta**2
         prho(2)=-pp(i,i+1)*dd/(2.d0*(abs(pp(i,i+1))**3))&
            &+3.d0*(dd**2)/(8.d0*(pp(i,i+1)**3)*abs(pp(i,i+1)))&
            &-5.d0*(dd**3)/(16.d0*(pp(i,i+1)**5)*abs(pp(i,i+1)))
         gam1=prho(2)-prho(1)
      end if
      Ixz(5,2)=Ixz(5,2)-gam1*dsin(alpha(i))/dx(i)
   end do
   Ixz(5,2)=Ixz(5,2)/3.0d0
!----------------------------------------------------
!--------I5^zeta-OK------------------------------------
   do i=1,3
      gam1=pp(i,i+1)/rho(i+1)-pp(i,i)/rho(i)
      if(abs(gam1)/(abs(pp(i,i+1)/rho(i+1))+abs(pp(i,i)/rho(i))) < 1.d-5)then
         dd=qq(i,i)**2+eta**2
         prho(1)=-pp(i,i)*dd/(2.d0*(abs(pp(i,i))**3))&
            &+3.d0*(dd**2)/(8.d0*(pp(i,i)**3)*abs(pp(i,i)))&
            &-5.d0*(dd**3)/(16.d0*(pp(i,i)**5)*abs(pp(i,i)))
         dd=qq(i,i+1)**2+eta**2
         prho(2)=-pp(i,i+1)*dd/(2.d0*(abs(pp(i,i+1))**3))&
            &+3.d0*(dd**2)/(8.d0*(pp(i,i+1)**3)*abs(pp(i,i+1)))&
            &-5.d0*(dd**3)/(16.d0*(pp(i,i+1)**5)*abs(pp(i,i+1)))
         gam1=prho(2)-prho(1)
      end if
      Ixz(5,3)=Ixz(5,3)+gam1*dcos(alpha(i))/dx(i)
   end do
   Ixz(5,3)=Ixz(5,3)/3.0d0
!----------------------------------------------------
!--------eta*I5^(xi*xi)-OK---------------------------
   do i=1,3
      gam1=(Lcal(i)*dcos(alpha(i))+qx(i)/dx(i)*deltai(i)*dsin(alpha(i)))*dsin(alpha(i))
      Ixz(5,4)=Ixz(5,4)-gam1*eta
   end do
   Ixz(5,4)=Ixz(5,4)+Ixz(3,1)
   Ixz(5,4)=Ixz(5,4)/3.0d0
!----------------------------------------------------
!--------I5^(xi*zeta)-OK-----------------------------
   do i=1,3
      gam1=(Lcal(i)*dsin(alpha(i))-qx(i)/dx(i)*deltai(i)*dcos(alpha(i)))*dsin(alpha(i))
      Ixz(5,5)=Ixz(5,5)-gam1
   end do
   Ixz(5,5)=Ixz(5,5)/3.0d0
!----------------------------------------------------
!--------eta*I5^(zeta*zeta)-OK-----------------------
   do i=1,3
      gam1=(Lcal(i)*dsin(alpha(i))-qx(i)/dx(i)*deltai(i)*dcos(alpha(i)))*dcos(alpha(i))
      Ixz(5,6)=Ixz(5,6)+gam1*eta
   end do
   Ixz(5,6)=Ixz(5,6)+Ixz(3,1)
   Ixz(5,6)=Ixz(5,6)/3.0d0
!----------------------------------------------------
!--------I5^(xi*xi*xi)-OK----------------------------
   do i=1,3
      gam1=(chii(i)*dcos(alpha(i))+2.0d0*qx(i)*Lcal(i)*dsin(alpha(i)))*dsin(alpha(i))*dcos(alpha(i))
      gam2=((qx(i)**2)*(2.0d0*(dcos(alpha(i))**2)-1.0d0)+(eta**2)*(dcos(alpha(i))**2))*deltai(i)/dx(i)*dsin(alpha(i))
      Ixz(5,7)=Ixz(5,7)+2.0d0*chii(i)*dsin(alpha(i))+gam1-gam2
   end do
   Ixz(5,7)=Ixz(5,7)/3.0d0
!----------------------------------------------------
!--------I5^(xi*xi*zeta)-OK--------------------------
   do i=1,3
      gam1=chii(i)*(dsin(alpha(i))**2)*dcos(alpha(i))
      gam2=Lcal(i)*(dsin(alpha(i))**2-dcos(alpha(i))**2)*qx(i)*dsin(alpha(i))
      gam3=deltai(i)/dx(i)*(2.0d0*(qx(i)**2)+eta**2)*(dsin(alpha(i))**2)*dcos(alpha(i))
      Ixz(5,8)=Ixz(5,8)-chii(i)*dcos(alpha(i))+gam1+gam2-gam3
   end do
   Ixz(5,8)=Ixz(5,8)/3.0d0
!----------------------------------------------------
!--------I5^(xi*zeta*zeta)-OK------------------------
   do i=1,3
      gam1=chii(i)*dsin(alpha(i))*(dcos(alpha(i))**2)
      gam2=Lcal(i)*(dsin(alpha(i))**2-dcos(alpha(i))**2)*qx(i)*dcos(alpha(i))
      gam3=deltai(i)/dx(i)*(2.0d0*(qx(i)**2)+eta**2)*dsin(alpha(i))*(dcos(alpha(i))**2)
      Ixz(5,9)=Ixz(5,9)-chii(i)*dsin(alpha(i))+gam1+gam2-gam3
   end do
   Ixz(5,9)=Ixz(5,9)/3.0d0
   Ixz(5,9)=-Ixz(5,9)
!----------------------------------------------------
!--------I5^(zeta*zeta*zeta)-OK----------------------
   do i=1,3
      gam1=(chii(i)*dsin(alpha(i))-2.0d0*qx(i)*Lcal(i)*dcos(alpha(i)))*dsin(alpha(i))*dcos(alpha(i))
      gam2=((qx(i)**2)*(2.0d0*(dsin(alpha(i))**2)-1.0d0)+(eta**2)*(dsin(alpha(i))**2))*deltai(i)/dx(i)*dcos(alpha(i))
      Ixz(5,10)=Ixz(5,10)-2.0d0*chii(i)*dcos(alpha(i))-gam1+gam2
   end do
   Ixz(5,10)=Ixz(5,10)/3.0d0
!----------------------------------------------------
!----------------------------------------------------
!================================================================
!================================================================
!-----------------------------------
   if(on_id == 1) exit
!-----------------------------------
!--- for check with perturbation ---
!-----------------------------------
      Ixz_store(ic,:,:)=Ixz(:,:)
   end do
   if(on_id /= 1)then
      ep_ixz=1.d-3
      call check_norm(Ixz_store(1,:,:),Ixz_store(2,:,:),val(1))
      call check_norm(Ixz_store(2,:,:),Ixz_store(3,:,:),val(2))
      call check_norm(Ixz_store(3,:,:),Ixz_store(1,:,:),val(3))
      if(val(1) < ep_ixz .and. val(2) < ep_ixz .and. val(3) < ep_ixz)then
         Ixz(:,:)=Ixz_store(1,:,:)
      else if(val(1) < ep_ixz .or. val(3) < ep_ixz)then
         Ixz(:,:)=Ixz_store(1,:,:)
      else if(val(2) < ep_ixz)then
         Ixz(:,:)=0.5d0*(Ixz_store(2,:,:)+Ixz_store(3,:,:))
      else
         write(1111,*)"maybe error in static"
         write(1111,*)"val=",val
         write(1111,*)"xx=",xx_in
         write(1111,*)"y1=",y1
         write(1111,*)"y2=",y2
         write(1111,*)"y3=",y3
         write(1111,*)
         icheck=1
         if(val(1) < val(2) .or. val(3) < val(2))then
            Ixz(:,:)=Ixz_store(1,:,:)
         else
            Ixz(:,:)=0.5d0*(Ixz_store(2,:,:)+Ixz_store(3,:,:))
         end if
      end if
   end if
!================================================================
!================================================================
!================================================================
if(icheck == 1)then
   !--- replace Ixz with numerical one ---
   xx(:)=xx_in(:)
   xi(1)=dot_product(y1(:)-xx(:),ee1(:))
   xi(2)=dot_product(y2(:)-xx(:),ee1(:))
   xi(3)=dot_product(y3(:)-xx(:),ee1(:))
   xi(4)=xi(1)
   zeta(1)=dot_product(y1(:)-xx(:),ee2(:))
   zeta(2)=dot_product(y2(:)-xx(:),ee2(:))
   zeta(3)=dot_product(y3(:)-xx(:),ee2(:))
   zeta(4)=zeta(1)
   eta=dot_product(y1(:)-xx(:),ee3(:))
   write(1111,*)"eta=",eta
   write(1111,*)"xi,zeta"
   do i=1,3
      write(1111,*)xi(i),zeta(i)
   end do
   write(1111,*)"cevec",dot_product(cvec,ee1),dot_product(cvec,ee2)
   call check_int(xi,zeta,eta,Ixz_num)
   norm=0.d0
   do i=1,3
      norm=norm+Ixz(1,i)**2
   end do
   do i=1,10
      norm=norm+Ixz(3,i)**2+Ixz(5,i)**2
   end do
   norm=sqrt(norm)
   do i=1,3
      check=check+(Ixz(1,i)-Ixz_num(1,i))**2
   end do
   do i=1,10
      check=check+(Ixz(3,i)-Ixz_num(3,i))**2+(Ixz(5,i)-Ixz_num(5,i))**2
   end do
   check=sqrt(check)
   write(1111,*)
   write(1111,*)"on_id, in_id=",on_id,in_id
   write(1111,*)
   write(1111,*)"Ixz",check,norm
   write(1111,*)"on_id,in_id=",on_id,in_id
   do i=1,3
      write(1111,*)"Ixz(1,:)=",i,Ixz(1,i),Ixz_num(1,i)
   end do
   do i=1,10
      write(1111,*)"Ixz(3,:)=",i,Ixz(3,i),Ixz_num(3,i)
   end do
   do i=1,10
      write(1111,*)"Ixz(5,:)=",i,Ixz(5,i),Ixz_num(5,i)
   end do
   if(check/norm > 1.d-3)then
      write(1111,*)"numerical norm",check/norm
      Ixz(:,:)=Ixz_num(:,:)
   end if
end if
!================================================================
!================================================================
!----------------------------------------------------
   call trans_dia_vector_h_phi(ee1,ee2,ee3,h_phi)
!----------------------------------------------------
!----------------------------------------------------
   cst1=1.d0/(8.d0*pi*(1.d0-nu))
   cst2=1.d0-2.d0*nu
   !--- tst ---
   do i=1,3
      do j=1,3
         do iq=1,3
            rinjq(i,j,iq)=nvec(j)*(&
               &ay(iq)*h_phi(i,1)*Ixz(3,4)+by(iq)*h_phi(i,2)*Ixz(3,6)+cy(iq)*h_phi(i,3)*Ixz(3,1)&
               &+(ay(iq)*h_phi(i,2)+by(iq)*h_phi(i,1))*Ixz(3,5)&
               &+(ay(iq)*h_phi(i,3)*eta+cy(iq)*h_phi(i,1))*Ixz(3,2)&
               &+(by(iq)*h_phi(i,3)*eta+cy(iq)*h_phi(i,2))*Ixz(3,3)   )
         end do
      end do
   end do
   do i=1,3
      do j=1,3
         do k=1,3
            a_phi(k)=h_phi(i,k)*h_phi(j,k)
         end do
         b_phi(1)=h_phi(i,2)*h_phi(j,3)+h_phi(i,3)*h_phi(j,2)
         b_phi(2)=h_phi(i,3)*h_phi(j,1)+h_phi(i,1)*h_phi(j,3)
         b_phi(3)=h_phi(i,1)*h_phi(j,2)+h_phi(i,2)*h_phi(j,1)
         do iq=1,3
            tst(i,j,iq)=-cst1*(&
               &cst2*( delta(i,j)*(ay(iq)*eta*Ixz(3,2)+by(iq)*eta*Ixz(3,3)+cy(iq)*Ixz(3,1))-rinjq(i,j,iq)+rinjq(j,i,iq) )&
               &+3.d0*( ay(iq)*a_phi(1)*eta*Ixz(5,7)+by(iq)*a_phi(2)*eta*Ixz(5,10)+cy(iq)*a_phi(3)*Ixz(5,1)&
                        &+(by(iq)*a_phi(1)+ay(iq)*b_phi(3))*eta*Ixz(5,8)+(ay(iq)*a_phi(2)+by(iq)*b_phi(3))*eta*Ixz(5,9)&
                        &+(cy(iq)*a_phi(1)+ay(iq)*eta*b_phi(2))*Ixz(5,4)+(cy(iq)*a_phi(2)+by(iq)*eta*b_phi(1))*Ixz(5,6)&
                        &+(cy(iq)*b_phi(3)+ay(iq)*eta*b_phi(1)+by(iq)*eta*b_phi(2))*eta*Ixz(5,5)&
                        &+(ay(iq)*eta*a_phi(3)+cy(iq)*b_phi(2))*(eta**2)*Ixz(5,2)&
                        &+(by(iq)*eta*a_phi(3)+cy(iq)*b_phi(1))*(eta**2)*Ixz(5,3) )&
               &)
         end do
      end do
   end do
   !--- ust ---
   cst1=1.d0/(16.d0*pi*(1.d0-nu))
   cst2=3.d0-4.d0*nu
   do i=1,3
      do j=1,3
         rirj=h_phi(i,1)*h_phi(j,1)*Ixz(3,4)+h_phi(i,2)*h_phi(j,2)*Ixz(3,6)+h_phi(i,3)*h_phi(j,3)*Ixz(3,1)*eta&
            &+(h_phi(i,1)*h_phi(j,2)+h_phi(i,2)*h_phi(j,1))*Ixz(3,5)&
            &+(h_phi(i,2)*h_phi(j,3)+h_phi(i,3)*h_phi(j,2))*Ixz(3,3)*eta&
            &+(h_phi(i,3)*h_phi(j,1)+h_phi(i,1)*h_phi(j,3))*Ixz(3,2)*eta
         ust(i,j)=cst1*(cst2*Ixz(1,1)*delta(i,j)+rirj)
      end do
   end do
   end subroutine elast3d_UTij_static
!================================================================
!================================================================
!================================================================
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