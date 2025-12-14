subroutine mkmat_transmission_PMCHWT(Cmat,rhs,omega)
   !test function is piecewise linear for hypersingular BIE 1.
   !test function is piecewise constant for displacement BIE 2.
   !displacement (unknown) is interpolated by piecewise linear.
   !traction (unknown) is interpolated by piecewise constant.
   use BEM3d
   use elast_parameter
   use struct_type
   implicit none
   integer::i,j,ix,iy,ex,ey,nip,niq,ip,iq,sing
   real(kind(0d0))::mu1,mu2
   complex(kind(0d0))::zvec(3),zten2(3,3)
   !---
   real(kind(0d0)),intent(in)::omega
   complex(kind(0d0)),dimension(n_mat),intent(out)::rhs
   complex(kind(0d0)),dimension(n_mat,n_mat),intent(out)::Cmat
!==============================================================
!==============================================================
interface
subroutine freq_inc_displacement_constant_x(elx,om,uout)
   use BEM3d
   use math_cst
   use elast_parameter
   use struct_type
   implicit none
   real(kind(0d0)),intent(in)::om
   complex(kind(0d0)),dimension(3),intent(out)::uout
   type(element),intent(in)::elx
   end subroutine freq_inc_displacement_constant_x
end interface
!==============================================================
interface
subroutine freq_inc_traction(elx,om,tout)
   use BEM3d
   use math_cst
   use elast_parameter
   use struct_type
   implicit none
   real(kind(0d0)),intent(in)::om
   complex(kind(0d0)),dimension(3,3),intent(out)::tout
   type(element),intent(in)::elx
   end subroutine freq_inc_traction
end interface
!==============================================================
interface
subroutine linear_x_aTij_freq_ip(om,zaTij,elx,ely,sing,ip)
   !\int \phi_{ip} \int aTij dSy dSx
   use BEM3d
   use math_cst
   use elast_parameter
   use struct_type
   implicit none
   integer,intent(in)::sing,ip
   real(kind(0d0)),intent(in)::om
   complex(kind(0d0)),dimension(3,3),intent(out)::zaTij
   type(element),intent(in)::elx,ely
   end subroutine linear_x_aTij_freq_ip
end interface
!==============================================================
interface
subroutine linear_x_Wij_freq_ip_iq(om,zWij,elx,ely,sing,ip,iq)
   !Wij/mu if Cijkl/mu
   !\int \phi_{ip} \int \phi_{iq} Wij/mu dSy dSx
   use BEM3d
   use math_cst
   use elast_parameter
   use struct_type
   implicit none
   integer,intent(in)::sing,ip,iq
   real(kind(0d0)),intent(in)::om
   complex(kind(0d0)),dimension(3,3),intent(out)::zWij
   type(element),intent(in)::elx,ely
   end subroutine linear_x_Wij_freq_ip_iq
end interface
!==============================================================
interface
subroutine constant_x_Uij_freq(om,zUij,elx,ely,sing)
   !\int \phi_{ip} \int mu*Uij dSy dSx
   use BEM3D
   use math_cst
   use elast_parameter
   use struct_type
   implicit none
   integer,intent(in)::sing
   real(kind(0d0)),intent(in)::om
   complex(kind(0d0)),dimension(3,3),intent(out)::zUij
   type(element),intent(in)::elx,ely
   end subroutine constant_x_Uij_freq
end interface
!==============================================================
interface
subroutine constant_x_Tij_freq_iq(om,zTij,elx,ely,sing,iq)
   !\int \phi_{ip} \int \phi_{iq} Tij dSy dSx
   use BEM3D
   use math_cst
   use elast_parameter
   use struct_type
   implicit none
   integer,intent(in)::sing,iq
   real(kind(0d0)),intent(in)::om
   complex(kind(0d0)),dimension(3,3),intent(out)::zTij
   type(element),intent(in)::elx,ely
   end subroutine constant_x_Tij_freq_iq
end interface
!==============================================================
   rhs=0.d0; Cmat=0.d0
   im=1
   call elast_para(cl(1),ct(1),rho(1))
   mu1=mu
   !--- incident wave ---
   do ex=1,nel
      call freq_inc_traction(el(ex),omega,zten2)
      do ip=1,3
         do i=1,3
         rhs(3*(el(ex)%ind(ip)-1)+i)=&
            &rhs(3*(el(ex)%ind(ip)-1)+i)+zten2(i,ip)/mu1
         end do
      end do
   end do
   do ex=1,nel
      call freq_inc_displacement_constant_x(el(ex),omega,zvec)
      do i=1,3
         rhs(nnode3+3*(ex-1)+i)=&
            &rhs(nnode3+3*(ex-1)+i)+zvec(i)
      end do
   end do
   !--- influence functions ---
   !--------------------
   !--- for domain 1 ---
   !--------------------
   mu1=mu
   !$OMP parallel do private(sing,ex,ey,ip,iq,zten2)
   do ix=1,nnode
      do nip=1,node(ix)%nel
         ex=node(ix)%iel(nip,1)
         ip=node(ix)%iel(nip,2)
         do iy=1,nnode
            do niq=1,node(iy)%nel
               ey=node(iy)%iel(niq,1)
               iq=node(iy)%iel(niq,2)
               if(ex == ey)then
                  sing=1
               else
                  sing=0
               end if
               !--- Wij ---
               call linear_x_Wij_freq_ip_iq(omega,zten2,el(ex),el(ey),sing,ip,iq)
               do i=1,3
                  do j=1,3
                     Cmat(3*(ix-1)+i,3*(iy-1)+j)&
                        &=Cmat(3*(ix-1)+i,3*(iy-1)+j)&
                        &+zten2(i,j)
                  end do
               end do
            end do
         end do
         do ey=1,nel
            if(ex == ey)then
               sing=1
            else
               sing=0
            end if
            !--- aTij ---
            call linear_x_aTij_freq_ip(omega,zten2,el(ex),el(ey),sing,ip)
            do i=1,3
               do j=1,3
                  Cmat(3*(ix-1)+i,nnode3+3*(ey-1)+j)&
                     &=Cmat(3*(ix-1)+i,nnode3+3*(ey-1)+j)&
                     &-zten2(i,j)
               end do
            end do
         end do
      end do
   end do
   !$OMP end parallel do
   !$OMP parallel do private(sing,ey,iq,zten2)
   do ex=1,nel
      do iy=1,nnode
         do niq=1,node(iy)%nel
            ey=node(iy)%iel(niq,1)
            iq=node(iy)%iel(niq,2)
            if(ex == ey)then
               sing=1
            else
               sing=0
            end if
            !--- Tij ---
            call constant_x_Tij_freq_iq(omega,zten2,el(ex),el(ey),sing,iq)
            do i=1,3
               do j=1,3
                  Cmat(nnode3+3*(ex-1)+i,3*(iy-1)+j)&
                     &=Cmat(nnode3+3*(ex-1)+i,3*(iy-1)+j)&
                     &+zten2(i,j)
               end do
            end do
         end do
      end do
      do ey=1,nel
         if(ex == ey)then
            sing=1
         else
            sing=0
         end if
         !--- Uij ---
         call constant_x_Uij_freq(omega,zten2,el(ex),el(ey),sing)
         do i=1,3
            do j=1,3
               Cmat(nnode3+3*(ex-1)+i,nnode3+3*(ey-1)+j)&
                  &=Cmat(nnode3+3*(ex-1)+i,nnode3+3*(ey-1)+j)&
                  &-zten2(i,j)
            end do
         end do
      end do
   end do
   !$OMP end parallel do
   !--------------------
   !--- for domain 2 ---
   !--------------------
   im=2
   call elast_para(cl(2),ct(2),rho(2))
   mu2=mu
   !$OMP parallel do private(sing,ex,ey,ip,iq,zten2)
   do ix=1,nnode
      do nip=1,node(ix)%nel
         ex=node(ix)%iel(nip,1)
         ip=node(ix)%iel(nip,2)
         do iy=1,nnode
            do niq=1,node(iy)%nel
               ey=node(iy)%iel(niq,1)
               iq=node(iy)%iel(niq,2)
               if(ex == ey)then
                  sing=1
               else
                  sing=0
               end if
               !--- Wij ---
               call linear_x_Wij_freq_ip_iq(omega,zten2,el(ex),el(ey),sing,ip,iq)
               do i=1,3
                  do j=1,3
                     Cmat(3*(ix-1)+i,3*(iy-1)+j)&
                        &=Cmat(3*(ix-1)+i,3*(iy-1)+j)&
                        &+(mu2/mu1)*zten2(i,j)
                  end do
               end do
            end do
         end do
         do ey=1,nel
            if(ex == ey)then
               sing=1
            else
               sing=0
            end if
            !--- aTij ---
            call linear_x_aTij_freq_ip(omega,zten2,el(ex),el(ey),sing,ip)
            do i=1,3
               do j=1,3
                  Cmat(3*(ix-1)+i,nnode3+3*(ey-1)+j)&
                     &=Cmat(3*(ix-1)+i,nnode3+3*(ey-1)+j)&
                     &-zten2(i,j)
               end do
            end do
         end do
      end do
   end do
   !$OMP end parallel do
   !$OMP parallel do private(sing,ey,iq,zten2)
   do ex=1,nel
      do iy=1,nnode
         do niq=1,node(iy)%nel
            ey=node(iy)%iel(niq,1)
            iq=node(iy)%iel(niq,2)
            if(ex == ey)then
               sing=1
            else
               sing=0
            end if
            !--- Tij ---
            call constant_x_Tij_freq_iq(omega,zten2,el(ex),el(ey),sing,iq)
            do i=1,3
               do j=1,3
                  Cmat(nnode3+3*(ex-1)+i,3*(iy-1)+j)&
                     &=Cmat(nnode3+3*(ex-1)+i,3*(iy-1)+j)&
                     &+zten2(i,j)
               end do
            end do
         end do
      end do
      do ey=1,nel
         if(ex == ey)then
            sing=1
         else
            sing=0
         end if
         !--- Uij ---
         call constant_x_Uij_freq(omega,zten2,el(ex),el(ey),sing)
         do i=1,3
            do j=1,3
               Cmat(nnode3+3*(ex-1)+i,nnode3+3*(ey-1)+j)&
                  &=Cmat(nnode3+3*(ex-1)+i,nnode3+3*(ey-1)+j)&
                  &-(mu1/mu2)*zten2(i,j)
            end do
         end do
      end do
   end do
   !$OMP end parallel do
   !reset
   im=1
   call elast_para(cl(1),ct(1),rho(1))
   end subroutine mkmat_transmission_PMCHWT