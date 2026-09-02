module elast3d_wij_entrywise_mod
  implicit none
contains
  !------------------------------------------
  subroutine elast3d_Wij_l_iq_nonGlobal(xco, nx, y_nodals, yNumNodeBasis, ely, om, elastp, sing, iq, zWij)
    !Wij/mu if Cijkl/mu
    !\int Wij/mu \phi_{iq} dSy
    use BEM3d_small_mod
    use elast_parameter_struct_mod
    use struct_type_fixed_len_node_mod
    use math_cst
    use Dyn_Sigma_Uij_nonGlobal_mod
    use static_Sigma_d_Uij_nonGlobal_mod
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
    complex(kind(0d0)), intent(out) :: zWij(3,3)

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
    y1(:) = y_nodals(ely%ind(1))%xc(:)
    y2(:) = y_nodals(ely%ind(2))%xc(:)
    y3(:) = y_nodals(ely%ind(3))%xc(:)
    ny(:) = ely%nvec(:)
    !initialize
    zUij=0.d0; zsigma_ijk_d=0.d0; zsigma_line=0.d0; zsigma_ijk=0.d0
    do ngy=1,integ
       do i=1,3
          yco(i)=y1(i)*gzi1(ngy)+y2(i)*gzi2(ngy)+y3(i)*gzi3(ngy)
       end do
       call cal_phiy_phiyd(y1,y2,y3,yco,phiy,phiyd)
       call elastUij_Sigma_dynamic_nonGlobal(xco,yco,elastp%cl,elastp%ct,zs,dUij,dsigma,elastp)
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
    call static_Sigma_d_Uij_nonGlobal(xco,y1,y2,y3,sing,bbb,sigma_ikjd,aaa,elastp)
    zUij(:,:)=zUij(:,:)+aaa(:,:,iq)
    zsigma_ijk_d(:,:,:,:)=zsigma_ijk_d(:,:,:,:)+sigma_ikjd(:,:,:,:,iq)
    zsigma_ijk(:,:,:)=zsigma_ijk(:,:,:)+bbb(:,:,:)
    aaa=0.d0; bbb=0.d0; ccc=0.d0
    do d=1,3
       do e=1,3
          do c=1,3
             do k=1,3
                aaa(d,c,k)=aaa(d,c,k)+elastp%pet(d,e,c,k)*ny(e)
             end do
          end do
       end do
    end do
    do b=1,3
       do a=1,3
          do i=1,3
             do c=1,3
                bbb(b,i,c)=bbb(b,i,c)+elastp%Cijkl(b,a,i,c)*nx(a)
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
                   zWij(b,j)=zWij(b,j)-elastp%mu*((zs/elastp%ct)**2)*nx(a)*elastp%Cijkl(b,a,i,c)&
                        &*ny(c)*zUij(i,j)
                end do
             end do
          end do
       end do
    end do
  end subroutine elast3d_Wij_l_iq_nonGlobal
!-------------------------------------
end module elast3d_wij_entrywise_mod