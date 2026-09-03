subroutine cal_phiy_phiyd(y1,y2,y3,yco,phiy,phiyd)
   ! calculate the drivative of shape function (\frac{\partial \phi_iq}{\partial y_d})
   implicit none
   integer::iq,d
   real(kind(0d0))::dia_norm,del,s1,s3,s_y,m_y,ra,xi_y,zeta_y
   real(kind(0d0)),dimension(3)::vec12,vec23,vec31,xi,zeta,ee1,ee2,ee3,ay,by,cy
   real(kind(0d0)),dimension(3),intent(in)::y1,y2,y3,yco
   real(kind(0d0)),dimension(3),intent(out)::phiy
   real(kind(0d0)),dimension(3,3),intent(out)::phiyd
!---------------------------------------------
   vec12(:)=y2(:)-y1(:)
   vec23(:)=y3(:)-y2(:)
   vec31(:)=y1(:)-y3(:)
   !-----coordinate {e1,e2,e3}-----------------
   ee1(:)=vec12(:)/dsqrt(vec12(1)**2+vec12(2)**2+vec12(3)**2)
   call out_product(ee3,vec12,-vec31)
   dia_norm=dsqrt(ee3(1)**2+ee3(2)**2+ee3(3)**2)
   ee3(:)=ee3(:)/dia_norm
   call out_product(ee2,ee3,ee1)
   !-------------------------------------------
   !---calculate xi,zeta,eta-------------------
   xi(1)=dot_product(y1(:),ee1(:))
   xi(2)=dot_product(y2(:),ee1(:))
   xi(3)=dot_product(y3(:),ee1(:))
   zeta(1)=dot_product(y1(:),ee2(:))
   zeta(2)=dot_product(y2(:),ee2(:))
   zeta(3)=dot_product(y3(:),ee2(:))
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
   xi_y=dot_product(yco,ee1)
   zeta_y=dot_product(yco,ee2)
   do iq=1,3
      do d=1,3
         phiyd(iq,d)=ay(iq)*ee1(d)+by(iq)*ee2(d)
      end do
      phiy(iq)=ay(iq)*xi_y+by(iq)*zeta_y+cy(iq)
   end do
   end subroutine cal_phiy_phiyd
!==============================================================
!==============================================================
!==============================================================
subroutine cal_phiy(y1,y2,y3,yco,phiy)
   implicit none
   integer::iq,d
   real(kind(0d0))::dia_norm,del,s1,s3,s_y,m_y,ra,xi_y,zeta_y
   real(kind(0d0)),dimension(3)::vec12,vec23,vec31,xi,zeta,ee1,ee2,ee3,ay,by,cy
   real(kind(0d0)),dimension(3),intent(in)::y1,y2,y3,yco
   real(kind(0d0)),dimension(3),intent(out)::phiy
!---------------------------------------------
   vec12(:)=y2(:)-y1(:)
   vec23(:)=y3(:)-y2(:)
   vec31(:)=y1(:)-y3(:)
   !-----coordinate {e1,e2,e3}-----------------
   ee1(:)=vec12(:)/dsqrt(vec12(1)**2+vec12(2)**2+vec12(3)**2)
   call out_product(ee3,vec12,-vec31)
   dia_norm=dsqrt(ee3(1)**2+ee3(2)**2+ee3(3)**2)
   ee3(:)=ee3(:)/dia_norm
   call out_product(ee2,ee3,ee1)
   !-------------------------------------------
   !---calculate xi,zeta,eta-------------------
   xi(1)=dot_product(y1(:),ee1(:))
   xi(2)=dot_product(y2(:),ee1(:))
   xi(3)=dot_product(y3(:),ee1(:))
   zeta(1)=dot_product(y1(:),ee2(:))
   zeta(2)=dot_product(y2(:),ee2(:))
   zeta(3)=dot_product(y3(:),ee2(:))
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
   xi_y=dot_product(yco,ee1)
   zeta_y=dot_product(yco,ee2)
   do iq=1,3
      phiy(iq)=ay(iq)*xi_y+by(iq)*zeta_y+cy(iq)
   end do
   end subroutine cal_phiy