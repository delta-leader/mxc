#pragma once

#include <data_container.hpp>
#include <complex>
#include <cuComplex.h>
#include <cuda.h>
#include <cublas_v2.h>

class H2Factorize {
private:
  cudaStream_t stream;
  cublasHandle_t cublasH;

  long long maxA, maxQ, bdim;
  long long lenD, lenA, rank;
  cuDoubleComplex* Adata, *Bdata, *Udata, *Vdata;
  cuDoubleComplex** A_SS, **A_SR, **A_RS, **A_RR;
  cuDoubleComplex** B, **U, **V, **V_R;
  int* ipiv, *info;
  
public:
  H2Factorize() {}
  H2Factorize(long long LD, long long lenA, long long lenQ, cudaStream_t stream);
  ~H2Factorize();

  void setData(long long rank, long long D, long long M, const long long ARows[], const long long ACols[], const long long Dims[], const MatrixDataContainer<std::complex<double>>& A, const MatrixDataContainer<std::complex<double>>& Q);
  void compute();
  void getResults(long long D, long long M, const long long ARows[], const long long ACols[], const long long Dims[], MatrixDataContainer<std::complex<double>>& A, int* ipvts);
};
