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
  void compute(const cublasComputeType_t COMP);
  template <typename OT>
  void getResults(long long D, long long M, const long long ARows[], const long long ACols[], const long long Dims[], MatrixDataContainer<OT>& A, int* ipvts);
};

template <>
class H2Factorize<__half> {
private:
  cudaStream_t stream;
  cublasHandle_t cublasH;

  long long maxA, maxQ, bdim;
  long long lenD, lenA, rank;
  __half* Adata, *Bdata, *Udata, *Vdata;
  float* Adata_float, *Vdata_float;
  __half** A_SS, **A_SR, **A_RS, **A_RR;
  __half** B, **U, **V, **V_R;
  float** A_RR_float, **A_RS_float, **V_R_float;
  int* ipiv, *info;
  
public:
  H2Factorize() {}
  H2Factorize(long long LD, long long lenA, long long lenQ, cudaStream_t stream);
  ~H2Factorize();

  void setData(long long rank, long long D, long long M, const long long ARows[], const long long ACols[], const long long Dims[], const MatrixDataContainer<float>& A, const MatrixDataContainer<float>& Q);
  void compute();
  void getResults(long long D, long long M, const long long ARows[], const long long ACols[], const long long Dims[], MatrixDataContainer<float>& A, int* ipvts);
};