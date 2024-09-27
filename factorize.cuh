#pragma once

#include <complex>
#include <cuComplex.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cublasLt.h>

/*template <typename DT = cuDoubleComplex>
class H2Factorize {
private:
  cudaStream_t stream;
  cublasHandle_t cublasH;
  
  long long maxA, maxQ;
  long long lenD, lenA, bdim, rank, offsetD;
  DT* Adata, *Bdata, *Udata, *Vdata;
  DT** A_SS, **A_SR, **A_RS, **A_RR;
  DT** B, **U, **V, **V_R;
  DT* hostA, **hostP;
  int* ipiv, *info;

  void factorize(const long long lenA, const long long bdim, const long long rank, const long long block, const long long M, const long long D,
    DT** A_SS, DT** A_SR, DT** A_RS, DT** A_RR, DT** U, DT** V, DT** V_R, DT** B);
  
public:
  H2Factorize() {}
  H2Factorize(long long LD, long long lenA, long long lenQ, cudaStream_t stream);
  ~H2Factorize();

  //template <typename OT>
  //void setData(long long rank, long long D, long long M, const long long ARows[], const long long ACols[], const long long Dims[], const MatrixDataContainer<OT>& A, const MatrixDataContainer<OT>& Q);
  //void compute();
  //void compute(const cublasComputeType_t COMP);
  //template <typename OT>
  //void getResults(long long D, long long M, const long long ARows[], const long long ACols[], const long long Dims[], MatrixDataContainer<OT>& A, int* ipvts);
  template <typename OT>
  void compute(const long long bdim, const long long rank, const long long D, const long long M, const long long N, const long long ARows[], const long long ACols[], OT* const A, OT* const R, const OT* const Q);
};
*/
class ColCommMPI;

void compute_factorize(cublasHandle_t cublasH, long long bdim, long long rank, long long D, long long M, long long N, const long long ARows[], const long long ACols[], std::complex<double>* A, std::complex<double>* R, const std::complex<double>* Q, const ColCommMPI& comm);
