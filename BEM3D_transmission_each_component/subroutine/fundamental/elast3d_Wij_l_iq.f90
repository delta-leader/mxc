subroutine elast3d_Wij_l_iq(om,zWij,xco,nx,ely,sing,iq)
   !Wij/mu if Cijkl/mu
   !\int Wij/mu \phi_{iq} dSy
   use BEM3d
   use elast_parameter
   use struct_type
   use math_cst
   implicit none
   integer,parameter::id_line=0
   integer::i,j,p,q,k,l,ngy,ngl,a,b,c,d,e,r,is
   integer::integ,linteg
   real(kind(0d0)),dimension(3)::yco,y1,y2,y3,dd,dist
   real(kind(0d0)),dimension(3)::ny,s1,s2,phiy,ay,by
   real(kind(0d0)),dimension(3,3)::phiyd
   real(kind(0d0)),dimension(4,3)::seg
   real(kind(0d0)),dimension(3,3,3)::aaa,bbb
   real(kind(0d0)),dimension(3,3,3,3)::ccc
   real(kind(0d0)),dimension(3,3,3,3,3)::sigma_ikjd
   real(kind(0d0)),dimension(:),allocatable::lgzi,lwi
   real(kind(0d0)),dimension(:),allocatable::gzi1,gzi2,gzi3,wi
   complex(kind(0d0))::zs
   complex(kind(0d0)),dimension(3,3)::dUij,zUij
   complex(kind(0d0)),dimension(3,3,3)::dsigma,zsigma_ijk
   complex(kind(0d0)),dimension(3,3,3,3)::zsigma_ijk_d,zsigma_line
   !--------------------------------
   integer,intent(in)::sing,iq
   real(kind(0d0)),intent(in)::om,xco(3),nx(3)
   complex(kind(0d0)),intent(out)::zWij(3,3)
   type(element),intent(in)::ely
!=======================================================================
!=======================================================================
interface
subroutine static_Sigma_d_Uij(xx,y1,y2,y3,on_id,sigma_ijk,sigma_ijk_d,Uij)
   use math_cst
   use elast_parameter
   implicit none
   integer,intent(in)::on_id  !1:x is on the element, 0:not
   real(kind=8),dimension(3),intent(in)::xx,y1,y2,y3
   real(kind=8),dimension(3,3,3),intent(out)::Uij,sigma_ijk
   real(kind=8),dimension(3,3,3,3,3),intent(out)::sigma_ijk_d
   end subroutine static_Sigma_d_Uij
end interface
!=======================================================================
interface
subroutine static_line_int(xx,y1,y2,id,Sigma_ijk)
   use elast_parameter
   use math_cst, only: pi
   implicit none
   integer,intent(in)::id
   real(kind=8),dimension(3),intent(in)::xx,y1,y2
   real(kind=8),dimension(3,3,3,3),intent(out)::Sigma_ijk
   end subroutine static_line_int
end interface
!=======================================================================
interface
subroutine Uij_Sigma_lap(x,y,cl,ct,zs,zUij,zSigma)
   use math_cst
   use elast_parameter
   implicit none
   real(kind=8),intent(in)::cl,ct
   real(kind=8),dimension(3),intent(in)::x,y
   complex(kind=8),intent(in)::zs
   complex(kind=8),dimension(3,3),intent(out)::zUij
   complex(kind=8),dimension(3,3,3),intent(out)::zSigma
   end subroutine Uij_Sigma_lap
end interface
!=======================================================================
interface
subroutine elastUij_Sigma_dynamic(x,y,cl,ct,zs,duij,dsigma)
   use math_cst, only: pi_4
   use elast_parameter
   implicit none
   real(kind(0d0)),intent(in)::cl,ct
   real(kind(0d0)),dimension(3),intent(in)::x,y
   complex(kind(0d0)),intent(in)::zs
   complex(kind(0d0)),dimension(3,3),intent(out)::duij
   complex(kind(0d0)),dimension(3,3,3),intent(out)::dsigma
   end subroutine elastUij_Sigma_dynamic
end interface
!=======================================================================
interface
subroutine elastSigma_dynamic(x,y,cl,ct,zs,dsigma)
   use math_cst, only: pi_4
   use elast_parameter
   implicit none
   real(kind(0d0)),intent(in)::cl,ct
   real(kind(0d0)),dimension(3),intent(in)::x,y
   complex(kind(0d0)),intent(in)::zs
   complex(kind(0d0)),dimension(3,3,3),intent(out)::dsigma
   end subroutine elastSigma_dynamic
end interface
!=======================================================================
interface
subroutine cal_phiy(y1,y2,y3,yco,phiy)
   implicit none
   real(kind=8),dimension(3),intent(in)::y1,y2,y3,yco
   real(kind=8),dimension(3),intent(out)::phiy
   end subroutine cal_phiy
end interface
!=======================================================================
interface
subroutine cal_phiy_phiyd(y1,y2,y3,yco,phiy,phiyd)
   ! calculate the drivative of shape function (\frac{\partial \phi_iq}{\partial y_d})
   implicit none
   real(kind=8),dimension(3),intent(in)::y1,y2,y3,yco
   real(kind=8),dimension(3),intent(out)::phiy
   real(kind=8),dimension(3,3),intent(out)::phiyd
   end subroutine cal_phiy_phiyd
end interface
!=======================================================================
interface
subroutine Gauss_tri(n,gzi1,gzi2,gzi3,wi)
   implicit none
   integer,parameter::n_ava=7
   integer,dimension(n_ava),parameter::ni_ava=(/3,4,7,13,27,48,79/)
   integer,intent(in)::n
   real(kind(0d0)),dimension(:),allocatable,intent(inout)::gzi1,gzi2,gzi3,wi
   end subroutine Gauss_tri
end interface
!=======================================================================
interface
subroutine Gauss_line(n,gzi,wi)
   implicit none
   integer,intent(in)::n
   real(kind(0d0)),dimension(:),allocatable,intent(inout)::gzi,wi
   end subroutine Gauss_line
end interface
!=======================================================================
   integ=4
   call Gauss_tri(integ,gzi1,gzi2,gzi3,wi)
   zs=-(0.0d0,1.0d0)*om
!-------------------------------------------
   y1(:)=node(ely%ind(1))%xc(:)
   y2(:)=node(ely%ind(2))%xc(:)
   y3(:)=node(ely%ind(3))%xc(:)
   ny(:)=ely%nvec(:)
   !initialize
   zUij=0.d0; zsigma_ijk_d=0.d0; zsigma_line=0.d0; zsigma_ijk=0.d0
   do ngy=1,integ
      do i=1,3
         yco(i)=y1(i)*gzi1(ngy)+y2(i)*gzi2(ngy)+y3(i)*gzi3(ngy)
      end do
      call cal_phiy_phiyd(y1,y2,y3,yco,phiy,phiyd)
      call elastUij_Sigma_dynamic(xco,yco,cl(im),ct(im),zs,dUij,dsigma)
      zsigma_ijk(:,:,:)=zsigma_ijk(:,:,:)+dsigma(:,:,:)*wi(ngy)*ely%Jgg
      do i=1,3
         do j=1,3
            zUij(i,j)=zUij(i,j)+dUij(i,j)*phiy(iq)*wi(ngy)*ely%Jgg
            do k=1,3
               do d=1,3
                  zsigma_ijk_d(i,j,k,d)=zsigma_ijk_d(i,j,k,d)&
                     &+dsigma(i,j,k)*phiyd(iq,d)*wi(ngy)*ely%Jgg
               end do
            end do
         end do
      end do
   end do !ngy
   !--- static ---
   call static_Sigma_d_Uij(xco,y1,y2,y3,sing,bbb,sigma_ikjd,aaa)
   zUij(:,:)=zUij(:,:)+aaa(:,:,iq)
   zsigma_ijk_d(:,:,:,:)=zsigma_ijk_d(:,:,:,:)+sigma_ikjd(:,:,:,:,iq)
   zsigma_ijk(:,:,:)=zsigma_ijk(:,:,:)+bbb(:,:,:)
   if(id_line == 1)then
      linteg=3
      call Gauss_line(linteg,lgzi,lwi)
      ! Constant interpolation because of differentiation (linear case)
      seg(1,:)=y1(:); seg(2,:)=y2(:); seg(3,:)=y3(:); seg(4,:)=y1(:) !segment for line integral
      dist(1)=dsqrt((y1(1)-y2(1))**2+(y1(2)-y2(2))**2+(y1(3)-y2(3))**2)
      dist(2)=dsqrt((y2(1)-y3(1))**2+(y2(2)-y3(2))**2+(y2(3)-y3(3))**2)
      dist(3)=dsqrt((y3(1)-y1(1))**2+(y3(2)-y1(2))**2+(y3(3)-y1(3))**2)
      do is=1,3
         s1(:)=seg(is,:)
         s2(:)=seg(is+1,:)
         dd(:)=0.5d0*(s2(:)-s1(:))
         do ngl=1,linteg
            ay(:)=(s2(:)-s1(:))/2.0d0
            by(:)=(s1(:)+s2(:))/2.0d0
            yco(:)=ay(:)*lgzi(ngl)+by(:)
            call cal_phiy(y1,y2,y3,yco,phiy)
            call elastSigma_dynamic(xco,yco,cl(im),ct(im),zs,dsigma)
            do i=1,3
               do j=1,3
                  do k=1,3
                     do r=1,3
                        zsigma_line(i,j,k,r)=zsigma_line(i,j,k,r)&
                           &+dsigma(i,j,k)*phiy(iq)*lwi(ngl)*dd(r)
                     end do
                  end do
               end do
            end do
         end do !ngl
      end do !is
      do is=1,3
         s1(:)=seg(is,:)
         s2(:)=seg(is+1,:)
         call Static_line_int(xco,s1,s2,is,ccc)
         dd(:)=s2(:)-s1(:)
         do i=1,3
            do j=1,3
               do k=1,3
                  do r=1,3
                     zsigma_line(i,j,k,r)=zsigma_line(i,j,k,r)+ccc(i,j,k,iq)/dist(is)*dd(r)
                  end do
               end do
            end do
         end do
      end do !is
   end if
   aaa=0.d0; bbb=0.d0; ccc=0.d0
   do d=1,3
      do e=1,3
         do c=1,3
            do k=1,3
               aaa(d,c,k)=aaa(d,c,k)+pet(d,e,c,k)*ny(e)
            end do
         end do
      end do
   end do
   do b=1,3
      do a=1,3
         do i=1,3
            do c=1,3
               bbb(b,i,c)=bbb(b,i,c)+Cijkl(b,a,i,c)*nx(a)
            end do
         end do
      end do
   end do
   do b=1,3
      do i=1,3
         do c=1,3
            do d=1,3
               do k=1,3
                  ccc(b,i,d,k)=ccc(b,i,d,k)+aaa(d,c,k)*bbb(b,i,c)
               end do
            end do
         end do
      end do
   end do
   zWij=0.d0
   do b=1,3
      do i=1,3
         do d=1,3
            do k=1,3
               do j=1,3
                  zWij(b,j)=zWij(b,j)+ccc(b,i,d,k)*zsigma_ijk_d(i,k,j,d)
               end do
            end do
         end do
      end do
   end do
   ! uij term
   do a=1,3
      do b=1,3
         do i=1,3
            do c=1,3
               do j=1,3
                  zWij(b,j)=zWij(b,j)-mu*((zs/ct(im))**2)*nx(a)*Cijkl(b,a,i,c)&
                     &*ny(c)*zUij(i,j)
               end do
            end do
         end do
      end do
   end do
   ! line integral term
   if(id_line == 1)then
      ccc=0.0d0
      do r=1,3
         do c=1,3
            do k=1,3
               do b=1,3
                  do i=1,3
                     ccc(r,k,b,i)=ccc(r,k,b,i)+permut(r,c,k)*bbb(b,i,c)
                  end do
               end do
            end do
         end do
      end do
      do r=1,3
         do k=1,3
            do b=1,3
               do i=1,3
                  do j=1,3
                     zWij(b,j)=zWij(b,j)-ccc(r,k,b,i)*zsigma_line(i,k,j,r)
                  end do
               end do
            end do
         end do
      end do
   end if
   end subroutine elast3d_Wij_l_iq
