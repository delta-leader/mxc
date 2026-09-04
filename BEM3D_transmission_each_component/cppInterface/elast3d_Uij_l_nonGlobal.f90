module elast3d_uij_entrywise_mod
  implicit none

  private
  public :: elast3d_Uij_l_reg_nonGlobal

  interface
     subroutine Gauss_tri(n,gzi1,gzi2,gzi3,wi)
       implicit none
       integer,parameter::n_ava=7
       integer,dimension(n_ava),parameter::ni_ava=(/3,4,7,13,27,48,79/)
       integer,intent(in)::n
       real(kind(0d0)),dimension(:),allocatable,intent(inout)::gzi1,gzi2,gzi3,wi
     end subroutine Gauss_tri
  end interface

contains
  !----------------------------------------
  subroutine elast3d_Uij_l_reg_nonGlobal(xco, nx, y_nodals, yNumNodeBasis, ely, om, elastp, sing, zUij)
    !parts of \int Uij dSy
    !sing=1:xco is on ely; 0:no
    use bem3d_small_mod
    use elast_parameter_struct_mod
    use struct_type_fixed_len_node_mod
    use math_cst
    !use elast3d_UTij_static_nonGlobal_mod
    implicit none

    real(kind(0d0)), intent(in) :: xco(3)
    real(kind(0d0)), intent(in) :: nx(3)
    type(nodal_point), intent(in) :: y_nodals(yNumNodeBasis)
    integer, intent(in) :: yNumNodeBasis
    type(element), intent(in) :: ely
    real(kind(0d0)), intent(in) :: om
    type(elast_parameter_struct), intent(in) :: elastp
    integer, intent(in) :: sing
    complex(kind(0d0)), intent(out) :: zUij(3,3)

    integer::i,j,k,ing,jng,ndiv,ngy,integ,linteg
    real(kind(0d0))::Jgg,coff
    real(kind(0d0)),dimension(3)::alpha,beta,outpro,y1,y2,y3
    real(kind(0d0)),dimension(3)::phi,yco,phi_zeta2
    real(kind(0d0)),dimension(3,3)::y1d,y2d,y3d
    real(kind(0d0)),dimension(:),allocatable::gzi1,gzi2,gzi3,wi
    real(kind(0d0)),dimension(:),allocatable::lgzi,lwi
    complex(kind(0d0)),dimension(3,3)::uij

    interface
       subroutine Gauss_line(n,gzi,wi)
         implicit none
         integer,intent(in)::n
         real(kind(0d0)),dimension(:),allocatable,intent(inout)::gzi,wi
       end subroutine Gauss_line
    end interface

   integ = ngauss_y
   call Gauss_tri(integ,gzi1,gzi2,gzi3,wi)
   linteg = ngauss_l
   call Gauss_line(linteg,lgzi,lwi)
!-------------------------------------------
    y1(:) = y_nodals(ely%ind(1))%xc(:)
    y2(:) = y_nodals(ely%ind(2))%xc(:)
    y3(:) = y_nodals(ely%ind(3))%xc(:)
!-------------------------------------------
   zUij=0.d0
   if(sing == 1)then
!---------------------
      y1d(1,:)=xco(:)
      y1d(2,:)=xco(:)
      y1d(3,:)=xco(:)

      y2d(1,:)=y1(:)
      y2d(2,:)=y2(:)
      y2d(3,:)=y3(:)

      y3d(1,:)=y2(:)
      y3d(2,:)=y3(:)
      y3d(3,:)=y1(:)
!---------------------
      do ndiv=1,3
         do ing=1,linteg
            do jng=1,linteg
               phi(1)=0.5d0*(1.0d0-lgzi(jng))
               phi(2)=0.25d0*(1.0d0+lgzi(ing))*(1.0d0+lgzi(jng))
               phi(3)=0.25d0*(1.0d0-lgzi(ing))*(1.0d0+lgzi(jng))
               yco(:)=y1d(ndiv,:)*phi(1)+y2d(ndiv,:)*phi(2)+y3d(ndiv,:)*phi(3)
               phi_zeta2(1)=-0.5d0
               phi_zeta2(2)=0.25d0*(1.0d0+lgzi(ing))
               phi_zeta2(3)=0.25d0*(1.0d0-lgzi(ing))
               alpha(:)=0.5d0*(y2d(ndiv,:)-y3d(ndiv,:))
               beta(:)=phi_zeta2(1)*y1d(ndiv,:)+phi_zeta2(2)*y2d(ndiv,:)+phi_zeta2(3)*y3d(ndiv,:)
               call out_product(outpro,alpha,beta)
               Jgg=dsqrt(dot_product(outpro,outpro))
               Jgg=Jgg*dabs(0.5d0*(1.0d0+lgzi(jng)))
               coff=lwi(ing)*lwi(jng)*Jgg
               !---
               call elastUij(xco, yco, elastp%cl, elastp%ct, om, uij)
               zUij(:,:)=zUij(:,:)+uij(:,:)*coff
            end do
         end do
      end do
   else
      do ngy=1,integ
         yco(:)=y1(:)*gzi1(ngy)+y2(:)*gzi2(ngy)+y3(:)*gzi3(ngy)
         coff=wi(ngy)*ely%Jgg
         !---
         call elastUij(xco, yco, elastp%cl, elastp%ct, om, uij)
         zUij(:,:)=zUij(:,:)+uij(:,:)*coff
      end do
   end if
  end subroutine elast3d_Uij_l_reg_nonGlobal
  !----------------------------------------
  subroutine elastUij(x,y,cl,ct,om,duij)
    !normalized by *mu
    use math_cst, only: pi_4, ii
    implicit none
    integer::i,j
    real(kind(0d0))::rr2,rr3,rr4,rr,r_y(3),delta(3,3)
    complex(kind(0d0))::zs,zst,zst2,zst3,zsl,zsl2,zsl3,zestr,zeslr
    complex(kind(0d0))::zL1,zT1,zL2,zT2,zL3,zT3
    real(kind(0d0)),intent(in)::cl,ct,om
    real(kind(0d0)),dimension(3),intent(in)::x,y
    complex(kind(0d0)),dimension(3,3),intent(out)::duij
    !=============================================================================
    !=============================================================================
    delta(:,:)=0.d0
    do i=1,3
       delta(i,i)=1.d0
    end do
    zs=-ii*om
    rr=sqrt(dot_product(x-y,x-y))
    r_y(:)=(y(:)-x(:))/rr
    zsl=zs/cl
    zsl2=zsl*zsl
    zsl3=zsl2*zsl
    zst=zs/ct
    zst2=zst*zst
    zst3=zst2*zst
    rr2=rr*rr
    rr3=rr2*rr
    rr4=rr2*rr2
    zeslr=exp(-zsl*rr)
    zestr=exp(-zst*rr)
    zL1=(3.d0/rr4+3.d0*zsl/rr3+zsl2/rr2)*zeslr
    zT1=(3.d0/rr4+3.d0*zst/rr3+zst2/rr2)*zestr
    zL3=(zsl2/rr2+zsl3/rr)*zeslr
    zT3=(zst2/rr2+zst3/rr)*zestr
    do i=1,3
       do j=1,3
          duij(i,j)=( (zL1*rr-zT1*rr)*r_y(i)*r_y(j)&
               & -(zL3/zsl2/rr-zT3/zst-1.0d0/rr3*zestr)*delta(i,j)&
               &) /(zst2*pi_4)
       end do
    end do
  end subroutine elastUij
  !--------------------------------------
end module elast3d_uij_entrywise_mod