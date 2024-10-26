#pragma once

#include <gpu_handles.cuh>
#include <complex>

class ColCommMPI;
template <typename DT>
class deviceMatrixDesc_t;

struct CsrContainer {
  long long M = 0;
  long long N = 0;
  long long NNZ = 0;
  long long* RowOffsets = nullptr;
  long long* ColInd = nullptr;
  std::complex<double>* Vals = nullptr;
};

struct VecDnContainer {
  long long N = 0;
  long long Xbegin = 0;
  long long lenX = 0;
  std::complex<double>* Vals = nullptr;

  long long LenComms = 0;
  long long* Neighbor = nullptr;
  long long* NeighborRoots = nullptr;
  ncclComm_t* NeighborComms = nullptr;
  ncclComm_t DupComm = nullptr;
};

typedef struct CsrContainer* CsrContainer_t;
typedef struct VecDnContainer* VecDnContainer_t;

struct CsrMatVecDesc {
  long long lowerZ = 0;
  long long buffer_size = 0;
  
  VecDnContainer_t X = nullptr;
  VecDnContainer_t Y = nullptr;
  VecDnContainer_t Z = nullptr;
  VecDnContainer_t W = nullptr;

  CsrContainer_t U = nullptr;
  CsrContainer_t C = nullptr;
  CsrContainer_t A = nullptr;

  cusparseDnVecDescr_t descX = nullptr;
  cusparseDnVecDescr_t descXi = nullptr;
  cusparseDnVecDescr_t descYi = nullptr;

  cusparseDnVecDescr_t descZ = nullptr;
  cusparseDnVecDescr_t descZi = nullptr;
  cusparseDnVecDescr_t descWi = nullptr;

  cusparseConstSpMatDescr_t descU = nullptr;
  cusparseConstSpMatDescr_t descV = nullptr;
  cusparseConstSpMatDescr_t descC = nullptr;
  cusparseConstSpMatDescr_t descA = nullptr;

  void* buffer = nullptr;
};

typedef struct CsrMatVecDesc* CsrMatVecDesc_t;

//void createDeviceCsr(CsrContainer_t* A, long long Mb, long long Nb, const long long RowDims[], const long long ColDims[], const long long ARows[], const long long ACols[], const std::complex<double> data[]);
//void createDeviceVec(VecDnContainer_t* X, const long long RowDims[], const ColCommMPI& comm, const ncclComms nccl_comms);
//void destroyDeviceVec(VecDnContainer_t X);

//void createSpMatrixDesc(deviceHandle_t handle, CsrMatVecDesc_t* desc, bool is_leaf, long long lowerZ, const long long Dims[], const long long Ranks[], const std::complex<double> U[], const std::complex<double> C[], const std::complex<double> A[], const ColCommMPI& comm, const ncclComms nccl_comms);
//void destroySpMatrixDesc(CsrMatVecDesc_t desc);

//void matVecUpwardPass(deviceHandle_t handle, CsrMatVecDesc_t desc, const std::complex<double>* X_in);
//void matVecHorizontalandDownwardPass(deviceHandle_t handle, CsrMatVecDesc_t desc, std::complex<double>* Y_out);
//void matVecLeafHorizontalPass(deviceHandle_t handle, CsrMatVecDesc_t desc, std::complex<double>* X_io);

//void matVecDeviceH2(deviceHandle_t handle, long long levels, CsrMatVecDesc_t desc[], std::complex<double>* devX);
//long long solveDeviceGMRES(deviceHandle_t handle, long long levels, CsrMatVecDesc_t desc[], long long mlevels, deviceMatrixDesc_t desc_m[], double tol, std::complex<double>* X, const std::complex<double>* B, long long inner_iters, long long outer_iters, double resid[], const ColCommMPI& comm, const ncclComms nccl_comms);
