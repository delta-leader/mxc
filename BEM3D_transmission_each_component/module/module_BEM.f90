module BEM3d
   use struct_type
   implicit none
   integer::nthread,id_bie,im
   integer::n_mat,nnode,nel,ninf,id_inc,ix1_min,ix1_0,ix1_max,nel3,nnode3
   real(kind(0d0))::u0,theta_in,rad
   real(kind(0d0)),dimension(2)::cl,ct,rho
   type(element),dimension(:),allocatable::el
   type(infield),dimension(:),allocatable::xinf
   type(nodal_point),dimension(:),allocatable::node
   integer::icheck
   end module BEM3d
!=========================================================================
module elast_parameter
   real(kind(0d0))::mu,lam,nu
   real(kind(0d0)),dimension(:,:)::delta(3,3)
   real(kind(0d0)),dimension(:,:,:)::permut(3,3,3)
   real(kind(0d0)),dimension(3,3,3,3)::Cijkl,pet,delta2 !Cijkl/mu
   real(kind(0d0)),dimension(3,3,3,3)::const2
   real(kind(0d0)),dimension(3,3,3,3,3,3)::const1,const_f
   real(kind(0d0)),dimension(3,3,3,3,3,3)::delta3
   contains
      subroutine elast_para(cl,ct,rho)
         use math_cst, only: pi
         integer::i,j,k,l,p,q,r,a,b,c,d,s,t,n,m
         real(kind(0d0))::clct2,cst
         real(kind(0d0)),intent(in)::cl,ct,rho
         real(kind(0d0)),dimension(3,3,3,3)::ccc
         real(kind(0d0)),dimension(3,3,3,3,3)::ddd
      !-----------------------------
         mu=rho*(ct**2)
         lam=rho*(cl**2)-2.d0*mu
         nu=lam*0.5d0/(lam+mu)
         permut(:,:,:)=0.d0
         permut(1,2,3)=1.d0
         permut(3,1,2)=1.d0
         permut(2,3,1)=1.d0
         permut(2,1,3)=-1.d0
         permut(3,2,1)=-1.d0
         permut(1,3,2)=-1.d0
         delta(:,:)=0.d0
         delta(1,1)=1.d0
         delta(2,2)=1.d0
         delta(3,3)=1.d0
         clct2=(cl/ct)**2-2.d0
         do k=1,3
            do j=1,3
               do p=1,3
                  do q=1,3
                     Cijkl(k,j,p,q)=clct2*delta(k,j)*delta(p,q)+delta(k,p)*delta(j,q)+delta(k,q)*delta(j,p)
                  end do
               end do
            end do
         end do
         pet(:,:,:,:)=0.d0
         do i=1,3
            do j=1,3
               do k=1,3
                  do l=1,3
                     do r=1,3
                        pet(i,j,k,l)=pet(i,j,k,l)+permut(r,i,j)*permut(r,k,l)
                     end do
                  end do
               end do
            end do
         end do
         do i=1,3
            do j=1,3
               do k=1,3
                  do l=1,3
                     delta2(i,j,k,l)=delta(i,j)*delta(k,l)
                     do p=1,3
                        do q=1,3
                           delta3(i,j,k,l,p,q)=delta(i,j)*delta(k,l)*delta(p,q)
                        end do
                     end do
                  end do
               end do
            end do
         end do
         !--------------------
         cst=0.25d0/(pi*(lam+mu)*mu) !normalized by mu
         do a=1,3
            do b=1,3
               do j=1,3
                  do k=1,3
                     ccc(a,b,j,k)=-cst*mu*(2.0d0*lam*delta2(a,b,j,k)&
                        &+(lam+2.0d0*mu)*(delta2(a,k,b,j)+delta2(b,k,a,j)))
                     const2(a,b,j,k)=cst*mu*(lam*delta2(a,b,j,k)+mu*(delta2(a,j,b,k)+delta2(a,k,b,j)))
                     do p=1,3
                        do q=1,3
                           const1(a,b,j,k,p,q)=-cst*((lam**2)*delta3(a,b,j,k,p,q)&
                              &+2.0d0*lam*mu*(delta3(a,b,j,p,k,q)+delta3(j,k,a,p,b,q))&
                              &+(mu**2)*(delta3(a,p,j,q,b,k)+delta3(b,p,k,q,a,j)+delta3(a,p,k,q,b,j)+delta3(b,p,j,q,a,k)))
                        end do
                     end do
                  end do
               end do
            end do
         end do
         ddd=0.0d0; const_f=0.0d0
         do b=1,3
            do d=1,3
               do s=1,3
                  do m=1,3
                     do n=1,3
                        do t=1,3
                           ddd(b,d,m,n,t)=ddd(b,d,m,n,t)+permut(b,d,s)*ccc(m,s,n,t)
                        end do
                     end do
                  end do
               end do
            end do
         end do
         do b=1,3
            do d=1,3
               do m=1,3
                  do n=1,3
                     do t=1,3
                        do j=1,3
                           do p=1,3
                              const_f(b,d,m,t,j,p)=const_f(b,d,m,t,j,p)+permut(j,p,n)*ddd(b,d,m,n,t)
                           end do
                        end do
                     end do
                  end do
               end do
            end do
         end do
         !--------------------
      end subroutine elast_para
   end module elast_parameter
!======================================
!======================================
subroutine out_product(xx,aa,bb)!   xx=aa cross bb
   implicit none
   real(kind(0d0)),dimension(3),intent(in)::aa,bb
   real(kind(0d0)),dimension(3),intent(out)::xx
!---------------------
   xx(1)=aa(2)*bb(3)-aa(3)*bb(2)
   xx(2)=aa(3)*bb(1)-aa(1)*bb(3)
   xx(3)=aa(1)*bb(2)-aa(2)*bb(1)
   end subroutine out_product
