#pragma once

#include <complex>
#include <map>
#include <vector>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <mpi.h>
#include <nccl.h>

#define STD_CTYPE std::complex<double>
#define THRUST_CTYPE thrust::complex<double>
#define CUDA_CTYPE cuDoubleComplex

class ColCommMPI;
class deviceMatrixDesc_t {
public:
  long long bdim;
  long long rank;
  long long reducLen;

  CUDA_CTYPE** A_ss;
  CUDA_CTYPE** A_sr;
  CUDA_CTYPE** A_rs;
  CUDA_CTYPE** A_rr;
  CUDA_CTYPE** A_sr_rows;
  const CUDA_CTYPE** A_unsort;

  CUDA_CTYPE** U_cols;
  CUDA_CTYPE** U_R;
  CUDA_CTYPE** V_rows;
  CUDA_CTYPE** V_R;

  CUDA_CTYPE** B_ind;
  CUDA_CTYPE** B_cols;
  CUDA_CTYPE** B_R;
  CUDA_CTYPE** AC_ind;
  CUDA_CTYPE** L_dst;

  CUDA_CTYPE* Adata;
  CUDA_CTYPE* Udata;
  CUDA_CTYPE* Vdata;
  CUDA_CTYPE* Bdata;
  CUDA_CTYPE* ACdata;
  
  int* Ipiv;
  int* Info;
};

void initGpuEnvs(cudaStream_t* memory_stream, cudaStream_t* compute_stream, cublasHandle_t* cublasH, std::map<const MPI_Comm, ncclComm_t>& nccl_comms, const std::vector<MPI_Comm>& comms, MPI_Comm world = MPI_COMM_WORLD);
void finalizeGpuEnvs(cudaStream_t memory_stream, cudaStream_t compute_stream, cublasHandle_t cublasH, std::map<const MPI_Comm, ncclComm_t>& nccl_comms);

void createMatrixDesc(deviceMatrixDesc_t* desc, long long bdim, long long rank, long long lower_rank, const ColCommMPI& comm);
void destroyMatrixDesc(deviceMatrixDesc_t desc);

void copyDataInMatrixDesc(deviceMatrixDesc_t desc, long long lenA, const STD_CTYPE* A, long long lenU, const STD_CTYPE* U, cudaStream_t stream);
void copyDataOutMatrixDesc(deviceMatrixDesc_t desc, long long lenA, STD_CTYPE* A, long long lenV, STD_CTYPE* V, cudaStream_t stream);

void compute_factorize(deviceMatrixDesc_t A, deviceMatrixDesc_t Al, cudaStream_t stream, cublasHandle_t cublasH, const ColCommMPI& comm, const std::map<const MPI_Comm, ncclComm_t>& nccl_comms);
