#pragma once

#include <gpu_handles.cuh>
#include <complex>
#include <cuComplex.h>

#define STD_CTYPE std::complex<double>
#define THRUST_CTYPE thrust::complex<double>
#define CUDA_CTYPE cuDoubleComplex

class ColCommMPI;
struct deviceMatrixDesc_t {
  long long bdim = 0;
  long long rank = 0;
  long long lenM = 0;
  long long lenN = 0;
  long long diag_offset = 0;
  long long lenA = 0;
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

  long long LenComms = 0;
  long long* Neighbor = nullptr;
  long long* NeighborRoots = nullptr;
  ncclComm_t* NeighborComms = nullptr;
  ncclComm_t MergeComm = nullptr;
  ncclComm_t DupComm = nullptr;
};

void createMatrixDesc(deviceMatrixDesc_t* desc, long long bdim, long long rank, deviceMatrixDesc_t lower, const ColCommMPI& comm, const ncclComms nccl_comms);
void destroyMatrixDesc(deviceMatrixDesc_t desc);

void copyDataInMatrixDesc(deviceMatrixDesc_t desc, const STD_CTYPE* A, const STD_CTYPE* U, cudaStream_t stream);
void copyDataOutMatrixDesc(deviceMatrixDesc_t desc, STD_CTYPE* A, STD_CTYPE* V, cudaStream_t stream);

void compute_factorize(deviceHandle_t handle, deviceMatrixDesc_t A, deviceMatrixDesc_t Al);
void compute_forward_substitution(deviceHandle_t handle, deviceMatrixDesc_t A, const std::complex<double>* X);
void compute_backward_substitution(deviceHandle_t handle, deviceMatrixDesc_t A, std::complex<double>* X);
void matSolvePreconditionDeviceH2(deviceHandle_t handle, long long levels, deviceMatrixDesc_t A[], std::complex<double>* devX);

int check_info(deviceMatrixDesc_t A, const ColCommMPI& comm);

