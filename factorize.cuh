#pragma once

#include <complex>
#include <map>
#include <vector>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <cuComplex.h>
#include <mpi.h>
#include <nccl.h>

#define STD_CTYPE std::complex<double>
#define THRUST_CTYPE thrust::complex<double>
#define CUDA_CTYPE cuDoubleComplex

class ColCommMPI;
class deviceMatrixDesc_t {
public:
  long long bdim = 0;
  long long rank = 0;
  long long diag_offset = 0;
  long long lower_offset = 0;
  long long reducLen = 0;

  CUDA_CTYPE** A_ss = nullptr;
  CUDA_CTYPE** A_sr = nullptr;
  CUDA_CTYPE** A_rs = nullptr;
  CUDA_CTYPE** A_rr = nullptr;
  CUDA_CTYPE** A_sr_rows = nullptr;
  CUDA_CTYPE** A_dst = nullptr;
  const CUDA_CTYPE** A_unsort = nullptr;

  CUDA_CTYPE** U_cols = nullptr;
  CUDA_CTYPE** U_R = nullptr;
  CUDA_CTYPE** V_rows = nullptr;
  CUDA_CTYPE** V_R = nullptr;

  CUDA_CTYPE** B_ind = nullptr;
  CUDA_CTYPE** B_cols = nullptr;
  CUDA_CTYPE** B_R = nullptr;

  CUDA_CTYPE** X_cols = nullptr;
  CUDA_CTYPE** Y_R_cols = nullptr;

  CUDA_CTYPE** AC_X = nullptr;
  CUDA_CTYPE** AC_X_R = nullptr;
  CUDA_CTYPE** AC_ind = nullptr;

  CUDA_CTYPE* Adata = nullptr;
  CUDA_CTYPE* Udata = nullptr;
  CUDA_CTYPE* Vdata = nullptr;
  CUDA_CTYPE* Bdata = nullptr;

  CUDA_CTYPE* ACdata = nullptr;
  CUDA_CTYPE* Xdata = nullptr;
  CUDA_CTYPE* Ydata = nullptr;
  CUDA_CTYPE* ONEdata = nullptr;
  
  int* Ipiv = nullptr;
  int* Info = nullptr;
};

class CsrMatVecDesc_t {
public:
  long long lenX = 0;
  long long lenZ = 0;
  long long xbegin = 0;
  long long zbegin = 0;
  long long lowerZ = 0;
  
  std::complex<double>* X = nullptr;
  std::complex<double>* Y = nullptr;
  std::complex<double>* Z = nullptr;
  std::complex<double>* W = nullptr;
  long long* NeighborX = nullptr;
  long long* NeighborZ = nullptr;

  int* RowOffsetsU = nullptr;
  int* ColIndU = nullptr;
  std::complex<double>* ValuesU = nullptr;

  int* RowOffsetsC = nullptr;
  int* ColIndC = nullptr;
  std::complex<double>* ValuesC = nullptr;

  int* RowOffsetsA = nullptr;
  int* ColIndA = nullptr;
  std::complex<double>* ValuesA = nullptr;

  void* descV = nullptr;
  void* descU = nullptr;
  void* descC = nullptr;
  void* descA = nullptr;
};

constexpr int hint_number = 512;

class hostMatrix_t {
public:
  long long lenA;
  CUDA_CTYPE* Adata;
};

void initGpuEnvs(cudaStream_t* memory_stream, cudaStream_t* compute_stream, cublasHandle_t* cublasH, cusparseHandle_t* cusparseH, cusolverDnHandle_t* cusolverH, std::map<const MPI_Comm, ncclComm_t>& nccl_comms, const std::vector<MPI_Comm>& comms, MPI_Comm world = MPI_COMM_WORLD);
void finalizeGpuEnvs(cudaStream_t memory_stream, cudaStream_t compute_stream, cublasHandle_t cublasH, cusparseHandle_t cusparseH, cusolverDnHandle_t cusolverH, std::map<const MPI_Comm, ncclComm_t>& nccl_comms);

void createMatrixDesc(deviceMatrixDesc_t* desc, long long bdim, long long rank, deviceMatrixDesc_t lower, const ColCommMPI& comm);
void destroyMatrixDesc(deviceMatrixDesc_t desc);

long long computeCooNNZ(long long Mb, const long long RowDims[], const long long ColDims[], const long long ARows[], const long long ACols[]);
void genCsrEntries(long long csrM, long long devRowIndx[], long long devColIndx[], std::complex<double> devVals[], long long Mb, long long Nb, const long long RowDims[], const long long ColDims[], const long long ARows[], const long long ACols[]);

void createHostMatrix(hostMatrix_t* h, long long bdim, long long lenA);
void destroyHostMatrix(hostMatrix_t h);

void copyDataInMatrixDesc(deviceMatrixDesc_t desc, long long lenA, const STD_CTYPE* A, long long lenU, const STD_CTYPE* U, cudaStream_t stream);
void copyDataOutMatrixDesc(deviceMatrixDesc_t desc, long long lenA, STD_CTYPE* A, long long lenV, STD_CTYPE* V, cudaStream_t stream);

void compute_factorize(deviceMatrixDesc_t A, deviceMatrixDesc_t Al, cudaStream_t stream, cublasHandle_t cublasH, const ColCommMPI& comm, const std::map<const MPI_Comm, ncclComm_t>& nccl_comms);
void compute_forward_substitution(deviceMatrixDesc_t A, const CUDA_CTYPE* X, cudaStream_t stream, cublasHandle_t cublasH, const ColCommMPI& comm, const std::map<const MPI_Comm, ncclComm_t>& nccl_comms);
void compute_backward_substitution(deviceMatrixDesc_t A, CUDA_CTYPE* X, cudaStream_t stream, cublasHandle_t cublasH, const ColCommMPI& comm, const std::map<const MPI_Comm, ncclComm_t>& nccl_comms);

int check_info(deviceMatrixDesc_t A, const ColCommMPI& comm);

void createSpMatrixDesc(CsrMatVecDesc_t* desc, bool is_leaf, long long lowerZ, const long long Dims[], const long long Ranks[], const std::complex<double> U[], const std::complex<double> C[], const std::complex<double> A[], const ColCommMPI& comm);
void destroySpMatrixDesc(CsrMatVecDesc_t desc);

void matVecUpwardPass(CsrMatVecDesc_t desc, const std::complex<double>* X_in, const ColCommMPI& comm);
void matVecHorizontalandDownwardPass(CsrMatVecDesc_t desc, std::complex<double>* Y_out);
void matVecLeafHorizontalPass(CsrMatVecDesc_t desc, std::complex<double>* X_io, const ColCommMPI& comm);
