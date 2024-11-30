#pragma once

#include <gpu_handles.cuh>
#include <complex>
#include <cuComplex.h>

//#define STD_CTYPE std::complex<double>
//#define THRUST_CTYPE thrust::complex<double>
//#define CUDA_CTYPE cuDoubleComplex

class ColCommMPI;
template <typename DT>
struct deviceMatrixDesc_t {
  typedef DT CT;

  long long bdim = 0;
  long long rank = 0;
  long long lenM = 0;
  long long lenN = 0;
  long long diag_offset = 0;
  long long lenA = 0;
  long long lower_offset = 0;
  long long reducLen = 0;

  DT** A_ss = nullptr;
  DT** A_sr = nullptr;
  DT** A_rs = nullptr;
  DT** A_rr = nullptr;
  DT** A_sr_rows = nullptr;
  DT** A_dst = nullptr;
  const DT** A_unsort = nullptr;

  DT** U_cols = nullptr;
  DT** U_R = nullptr;
  DT** V_rows = nullptr;
  DT** V_R = nullptr;

  DT** B_ind = nullptr;
  DT** B_cols = nullptr;
  DT** B_R = nullptr;

  DT** X_cols = nullptr;
  DT** Y_R_cols = nullptr;

  DT** AC_X = nullptr;
  DT** AC_X_R = nullptr;
  DT** AC_ind = nullptr;

  DT* Adata = nullptr;
  DT* Udata = nullptr;
  DT* Vdata = nullptr;
  DT* Bdata = nullptr;

  DT* ACdata = nullptr;
  DT* Xdata = nullptr;
  DT* Ydata = nullptr;
  DT* ONEdata = nullptr;
  
  int* Ipiv = nullptr;
  int* Info = nullptr;

  long long LenComms = 0;
  long long* Neighbor = nullptr;
  long long* NeighborRoots = nullptr;
  ncclComm_t* NeighborComms = nullptr;
  ncclComm_t MergeComm = nullptr;
  ncclComm_t DupComm = nullptr;
};

template <>
struct deviceMatrixDesc_t<std::complex<double>> {
  typedef cuDoubleComplex CT;

  long long bdim = 0;
  long long rank = 0;
  long long lenM = 0;
  long long lenN = 0;
  long long diag_offset = 0;
  long long lenA = 0;
  long long lower_offset = 0;
  long long reducLen = 0;

  cuDoubleComplex** A_ss = nullptr;
  cuDoubleComplex** A_sr = nullptr;
  cuDoubleComplex** A_rs = nullptr;
  cuDoubleComplex** A_rr = nullptr;
  cuDoubleComplex** A_sr_rows = nullptr;
  cuDoubleComplex** A_dst = nullptr;
  const cuDoubleComplex** A_unsort = nullptr;

  cuDoubleComplex** U_cols = nullptr;
  cuDoubleComplex** U_R = nullptr;
  cuDoubleComplex** V_rows = nullptr;
  cuDoubleComplex** V_R = nullptr;

  cuDoubleComplex** B_ind = nullptr;
  cuDoubleComplex** B_cols = nullptr;
  cuDoubleComplex** B_R = nullptr;

  cuDoubleComplex** X_cols = nullptr;
  cuDoubleComplex** Y_R_cols = nullptr;

  cuDoubleComplex** AC_X = nullptr;
  cuDoubleComplex** AC_X_R = nullptr;
  cuDoubleComplex** AC_ind = nullptr;

  cuDoubleComplex* Adata = nullptr;
  cuDoubleComplex* Udata = nullptr;
  cuDoubleComplex* Vdata = nullptr;
  cuDoubleComplex* Bdata = nullptr;

  cuDoubleComplex* ACdata = nullptr;
  cuDoubleComplex* Xdata = nullptr;
  cuDoubleComplex* Ydata = nullptr;
  cuDoubleComplex* ONEdata = nullptr;
  
  int* Ipiv = nullptr;
  int* Info = nullptr;

  long long LenComms = 0;
  long long* Neighbor = nullptr;
  long long* NeighborRoots = nullptr;
  ncclComm_t* NeighborComms = nullptr;
  ncclComm_t MergeComm = nullptr;
  ncclComm_t DupComm = nullptr;
};

template <>
struct deviceMatrixDesc_t<std::complex<float>> {
  typedef cuComplex CT;

  long long bdim = 0;
  long long rank = 0;
  long long lenM = 0;
  long long lenN = 0;
  long long diag_offset = 0;
  long long lenA = 0;
  long long lower_offset = 0;
  long long reducLen = 0;

  cuComplex** A_ss = nullptr;
  cuComplex** A_sr = nullptr;
  cuComplex** A_rs = nullptr;
  cuComplex** A_rr = nullptr;
  cuComplex** A_sr_rows = nullptr;
  cuComplex** A_dst = nullptr;
  const cuComplex** A_unsort = nullptr;

  cuComplex** U_cols = nullptr;
  cuComplex** U_R = nullptr;
  cuComplex** V_rows = nullptr;
  cuComplex** V_R = nullptr;

  cuComplex** B_ind = nullptr;
  cuComplex** B_cols = nullptr;
  cuComplex** B_R = nullptr;

  cuComplex** X_cols = nullptr;
  cuComplex** Y_R_cols = nullptr;

  cuComplex** AC_X = nullptr;
  cuComplex** AC_X_R = nullptr;
  cuComplex** AC_ind = nullptr;

  cuComplex* Adata = nullptr;
  cuComplex* Udata = nullptr;
  cuComplex* Vdata = nullptr;
  cuComplex* Bdata = nullptr;

  cuComplex* ACdata = nullptr;
  cuComplex* Xdata = nullptr;
  cuComplex* Ydata = nullptr;
  cuComplex* ONEdata = nullptr;
  
  int* Ipiv = nullptr;
  int* Info = nullptr;

  long long LenComms = 0;
  long long* Neighbor = nullptr;
  long long* NeighborRoots = nullptr;
  ncclComm_t* NeighborComms = nullptr;
  ncclComm_t MergeComm = nullptr;
  ncclComm_t DupComm = nullptr;
};

// TODO why is lower not a reference?
template <typename DT>
void createMatrixDesc(deviceMatrixDesc_t<DT>* desc, long long bdim, long long rank, deviceMatrixDesc_t<DT> lower, const ColCommMPI& comm, const ncclComms nccl_comms);
template <typename DT>
void destroyMatrixDesc(deviceMatrixDesc_t<DT> desc);

template <typename DT>
void copyDataInMatrixDesc(deviceMatrixDesc_t<DT> desc, const DT* A, const DT* U, cudaStream_t stream);
template <typename DT>
void copyDataOutMatrixDesc(deviceMatrixDesc_t<DT> desc, DT* A, DT* V, cudaStream_t stream);

template <typename DT>
void compute_factorize(deviceHandle_t handle, deviceMatrixDesc_t<DT> A, deviceMatrixDesc_t<DT> Al);
template <typename DT>
void compute_factorize(deviceHandle_t handle, deviceMatrixDesc_t<DT> A, deviceMatrixDesc_t<DT> Al, const cublasComputeType_t COMP);
template <typename DT>
void compute_forward_substitution(deviceHandle_t handle, deviceMatrixDesc_t<DT> A, const DT* X);
template <typename DT>
void compute_backward_substitution(deviceHandle_t handle, deviceMatrixDesc_t<DT> A, DT* X);
template <typename DT>
void matSolvePreconditionDeviceH2(deviceHandle_t handle, long long levels, deviceMatrixDesc_t<DT> A[], DT* devX);

template <typename DT>
int check_info(deviceMatrixDesc_t<DT> A, const long long M);

