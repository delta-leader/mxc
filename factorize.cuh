#pragma once

#include <complex>
#include <cuComplex.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

class H2Factorize {
private:
  cudaStream_t stream;
  cublasHandle_t cublasH;

  long long maxA, maxQ;
  long long lenD, lenA, bdim, rank, offsetD;
  cuDoubleComplex* Adata, *Bdata, *Udata, *Vdata;
  cuDoubleComplex** A_SS, **A_SR, **A_RS, **A_RR;
  cuDoubleComplex** B, **U, **V, **V_R;
  cuDoubleComplex* hostA, **hostP;
  int* ipiv, *info;
  
public:
  H2Factorize(long long LD, long long lenA, long long lenQ, cudaStream_t stream);
  ~H2Factorize();

  void compute(long long bdim, long long rank, long long D, long long M, long long N, const long long ARows[], const long long ACols[], std::complex<double>* A, std::complex<double>* R, const std::complex<double>* Q);
};
