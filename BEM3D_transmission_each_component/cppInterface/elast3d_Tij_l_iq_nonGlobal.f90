module elast3d_Tij_entrywise_mod
  use iso_c_binding
  implicit none
contains
  !--------------------------------------------------
  subroutine elast3d_Tij_l_iq_nonGlobal(xco, nx, y_nodals, yNumNodeBasis, ely, om, elastp, sing, iq, zTij)
    !zUij(i,j)=\int mu*Uij dSy
    !\phi_{iq} is piecewise linear shape function.
    !sing=1:xco is on ely; 0:no
    use BEM3D_SMALL_MOD
    use elast_parameter_struct_mod
    use struct_type_fixed_len_node_mod
    use math_cst
    use elast3d_UTij_static_nonGlobal_mod
    implicit none

    real(kind(0d0)), intent(in) :: xco(3)
    real(kind(0d0)), intent(in) :: nx(3)
    type(nodal_point), intent(in) :: y_nodals(yNumNodeBasis)
    integer, intent(in) :: yNumNodeBasis
    type(element), intent(in) :: ely
    real(kind(0d0)), intent(in) :: om
    type(elast_parameter_struct), intent(in) :: elastp
    integer, intent(in) :: sing
    integer, intent(in) :: iq
    complex(kind(0d0)), intent(out) :: zTij(3,3)

    integer::i,j,ng,integ
    real(kind(0d0))::cst
    real(kind(0d0)),dimension(3)::yco,y1,y2,y3,phiy
    real(kind(0d0)),dimension(3,3)::ust
    real(kind(0d0)),dimension(3,3,3)::tst
    real(kind(0d0)),dimension(:),allocatable::gzi1,gzi2,gzi3,wi
    complex(kind(0d0))::zs
    complex(kind(0d0)),dimension(3,3)::zt

    interface
       subroutine cal_phiy(y1,y2,y3,yco,phiy)
         implicit none
         real(kind=8),dimension(3),intent(in)::y1,y2,y3,yco
         real(kind=8),dimension(3),intent(out)::phiy
       end subroutine cal_phiy
    end interface

    interface
       subroutine Gauss_tri(n,gzi1,gzi2,gzi3,wi)
         implicit none
         integer,parameter::n_ava=7
         integer,dimension(n_ava),parameter::ni_ava=(/3,4,7,13,27,48,79/)
         integer,intent(in)::n
         real(kind(0d0)),dimension(:),allocatable,intent(inout)::gzi1,gzi2,gzi3,wi
       end subroutine Gauss_tri
    end interface

    zs=-(0.0d0,1.0d0)*om

    y1(:) = y_nodals(ely%ind(1))%xc(:)
    y2(:) = y_nodals(ely%ind(2))%xc(:)
    y3(:) = y_nodals(ely%ind(3))%xc(:)
    !--- static part ---
    call elast3d_UTij_static_nonGlobal(xco,y1,y2,y3,sing,ely%nvec,ust,tst,elastp)
    zTij(:,:)=tst(:,:,iq)
    !--- dynamic part ---
    integ=4
    call Gauss_tri(integ,gzi1,gzi2,gzi3,wi)
    do ng=1,integ
       yco(:)=y1(:)*gzi1(ng)+y2(:)*gzi2(ng)+y3(:)*gzi3(ng)
       call cal_phiy(y1,y2,y3,yco,phiy)
       call elastTij_dynamic_nonGlobal(xco,yco,ely%nvec,zs,zt,elastp)
       cst=ely%Jgg*wi(ng)*phiy(iq)
       do i=1,3
          do j=1,3
             zTij(i,j)=zTij(i,j)+zt(i,j)*cst
          end do
       end do
    end do
  end subroutine elast3d_Tij_l_iq_nonGlobal
  !=============================================================================
  !=============================================================================
  !=============================================================================
  subroutine elastTij_dynamic_nonGlobal(x,y,nvec,zs,dtij,elastp)
    use math_cst, only: pi_4
    use BEM3D_SMALL_MOD
    use elast_parameter_struct_mod
    implicit none
    integer::i,j
    real(kind(0d0))::rknk,rr(4),r_y(3)
    complex(kind(0d0))::zst(4),zsl(4),zcst
    complex(kind(0d0))::zt_part(3),zal_L,zal_T,zslr,zstr,zbe_L,zbe_T
    real(kind(0d0)),dimension(3),intent(in)::x,y,nvec
    complex(kind(0d0)),intent(in)::zs
    complex(kind(0d0)),dimension(3,3),intent(out)::dtij
    type(elast_parameter_struct), intent(in) :: elastp
    !=============================================================================
    interface
       subroutine Exp_sr_series_BEM(zal_L,zal_T,zbe_L,zbe_T,zslr,zstr)
         implicit none
         complex(kind(0d0)),intent(in)::zslr,zstr
         complex(kind(0d0)),intent(out)::zal_L,zal_T,zbe_L,zbe_T
       end subroutine Exp_sr_series_BEM
    end interface
    !=============================================================================
    rr(1)=sqrt(dot_product(x-y,x-y))
    r_y(:)=(y(:)-x(:))/rr(1)
    rknk=dot_product(r_y,nvec)
    zsl(1)=zs/elastp%cl
    zst(1)=zs/elastp%ct
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
    zt_part(1)=-zsl(4)+zsl(4)*zslr/3.d0*(1.d0-zslr)&
         &+(6.d0+6.d0*zslr+4.d0*zsl(2)*rr(2)+2.d0*zsl(3)*rr(3))*zbe_L&
         &+zst(4)*zstr/3.d0-(6.d0+6.d0*zstr+2.d0*zst(2)*rr(2))*zbe_T&
         &+zst(2)*zsl(2)-zst(2)*(1.d0+zslr)*(rr(1)*zal_L+zsl(2)/2.d0)
    zt_part(2)=-zsl(4)*zslr/3.d0&
         &+(6.d0+6.d0*zslr+2.d0*zsl(2)*rr(2))*zbe_L&
         &+zst(4)/2.d0+zst(4)*zst(2)*rr(2)/6.d0&
         &-(6.d0+6.d0*zstr+3.d0*zst(2)*rr(2)+zst(3)*rr(3))*zbe_T
    zt_part(3)=zsl(4)*(1.d0+zslr+zsl(2)*rr(2)/3.d0)&
         &-2.d0*(15.d0+15.d0*zslr+6.d0*zsl(2)*rr(2)+zsl(3)*rr(3))*zbe_L&
         &-zst(4)*(1.d0+zstr+zst(2)*rr(2)/3.d0)&
         &+2.d0*(15.d0+15.d0*zstr+6.d0*zst(2)*rr(2)+zst(3)*rr(3))*zbe_T
    zcst=1.d0/(zst(2)*pi_4)
    do i=1,3
       do j=1,3
          dtij(i,j)=zcst*( zt_part(1)*r_y(i)*nvec(j)&
               &+zt_part(2)*(r_y(j)*nvec(i)+elastp%delta(i,j)*rknk)&
               &+zt_part(3)*r_y(i)*r_y(j)*rknk )
       end do
    end do
  end subroutine elastTij_dynamic_nonGlobal
  !--------------------------------------------------
end module elast3d_Tij_entrywise_mod