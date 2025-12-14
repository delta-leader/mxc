module math_cst
   real(kind=8),parameter::pi=3.141592653589793d0
   real(kind=8),parameter::pi_2=6.28318530717958d0
   real(kind=8),parameter::pi_4=12.5663706143592d0
   real(kind=8),parameter::egamma=0.5772156649015329d0
   complex(kind=8),parameter::ii=(0.0d0,1.0d0)
   
   type amatrix
      integer::ir
      integer::ic
      real(kind=8)::cof
   end type amatrix
   
   type amatrixPSV
      integer::ir
      integer::ic
      real(kind=8),dimension(2,2)::cof
   end type amatrixPSV

end module math_cst