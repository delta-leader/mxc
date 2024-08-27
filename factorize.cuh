#pragma once

#include <data_container.hpp>
#include <complex>
#include <cuComplex.h>
#include <cuda.h>
#include <cublas_v2.h>

template <typename DT = cuDoubleComplex>
class H2Factorize {
private:
  cudaStream_t stream;
  cublasHandle_t cublasH;

  long long maxA, maxQ, bdim;
  long long lenD, lenA, rank;
  DT* Adata, *Bdata, *Udata, *Vdata;
  DT** A_SS, **A_SR, **A_RS, **A_RR;
  DT** B, **U, **V, **V_R;
  int* ipiv, *info;
  
public:
  H2Factorize() {}
  H2Factorize(long long LD, long long lenA, long long lenQ, cudaStream_t stream);
  ~H2Factorize();

  template <typename OT>
  void setData(long long rank, long long D, long long M, const long long ARows[], const long long ACols[], const long long Dims[], const MatrixDataContainer<OT>& A, const MatrixDataContainer<OT>& Q);
  void compute();
  template <typename OT>
  void getResults(long long D, long long M, const long long ARows[], const long long ACols[], const long long Dims[], MatrixDataContainer<OT>& A, int* ipvts);
};
