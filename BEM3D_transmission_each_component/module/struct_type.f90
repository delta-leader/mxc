module struct_type
  use iso_c_binding
  implicit none
  !==================
  !=== for 3D BEM ===
  !==================
  type element
     integer::id_e,nedge,nnear
     integer,dimension(3)::ind,id !id=1:edge
     real(kind(0d0))::Jgg
     real(kind(0d0)),dimension(3)::xc,nvec,mvec,svec
     complex(kind(0d0)),dimension(3)::t
  end type element
  !-----
  type nodal_point
     integer::ident,nel
     integer,dimension(:,:),allocatable::iel
     real(kind(0d0)),dimension(3)::xc,nvec,svec
     complex(kind(0d0)),dimension(3)::u
  end type nodal_point
  !-----
  type infield
     real(kind(0d0)),dimension(3)::xc
     complex(kind(0d0)),dimension(3)::uin,usc
  end type infield
end module struct_type