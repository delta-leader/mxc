#pragma once

#include <gpu_handles.cuh>
#include <complex>
#include <h2matrix.hpp>

class ColCommMPI;
template <typename DT>
class deviceMatrixDesc_t;

template <typename DT>
struct CsrContainer {
  long long M = 0;
  long long N = 0;
  long long NNZ = 0;
  long long* RowOffsets = nullptr;
  long long* ColInd = nullptr;
  DT* Vals = nullptr;
};

template <typename DT>
struct VecDnContainer {
  long long N = 0;
  long long Xbegin = 0;
  long long lenX = 0;
  DT* Vals = nullptr;

  long long LenComms = 0;
  long long* Neighbor = nullptr;
  long long* NeighborRoots = nullptr;
  ncclComm_t* NeighborComms = nullptr;
  ncclComm_t DupComm = nullptr;
};

template <typename DT>
using CsrContainer_t = CsrContainer<DT>*;
template <typename DT>
using VecDnContainer_t = struct VecDnContainer<DT>*;

template <typename DT>
struct CsrMatVecDesc {
  long long lowerZ = 0;
  long long buffer_size = 0;
  
  VecDnContainer_t<DT> X = nullptr;
  VecDnContainer_t<DT> Y = nullptr;
  VecDnContainer_t<DT> Z = nullptr;
  VecDnContainer_t<DT> W = nullptr;

  CsrContainer_t<DT> U = nullptr;
  CsrContainer_t<DT> C = nullptr;
  CsrContainer_t<DT> A = nullptr;

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

template <typename DT>
using CsrMatVecDesc_t = struct CsrMatVecDesc<DT>*;

template <typename DT>
void createDeviceCsr(CsrContainer_t<DT>* A, long long Mb, long long Nb, const long long RowDims[], const long long ColDims[], const long long ARows[], const long long ACols[], const DT data[]);
template <typename DT>
void createDeviceVec(VecDnContainer_t<DT>* X, const long long RowDims[], const long long nodes);
template <typename DT>
void destroyDeviceVec(VecDnContainer_t<DT> X);

template <typename DT>
void createSpMatrixDesc(deviceHandle_t handle, CsrMatVecDesc_t<DT>* desc, bool is_leaf, long long lowerZ, const long long Dims[], const long long Ranks[], const DT U[], const DT C[], const DT A[], const H2Matrix<DT>& matrix);
template <typename DT>
void destroySpMatrixDesc(CsrMatVecDesc_t<DT> desc);

template <typename DT>
void matVecUpwardPass(deviceHandle_t handle, CsrMatVecDesc_t<DT> desc, const DT* X_in);
template <typename DT>
void matVecHorizontalandDownwardPass(deviceHandle_t handle, CsrMatVecDesc_t<DT> desc, DT* Y_out);
template <typename DT>
void matVecLeafHorizontalPass(deviceHandle_t handle, CsrMatVecDesc_t<DT> desc, DT* X_io);

template <typename DT>
void matVecDeviceH2(deviceHandle_t handle, long long levels, CsrMatVecDesc_t<DT> desc[], DT* devX);
template <typename DT>
long long solveDeviceGMRES(deviceHandle_t handle, long long levels, CsrMatVecDesc_t<DT> desc[], long long mlevels, deviceMatrixDesc_t<DT> desc_m[], double tol, DT* X, const DT* B, long long inner_iters, long long outer_iters, double resid[]);
template <typename DT, typename OT>
long long solveDeviceGMRES(deviceHandle_t handle, long long levels, CsrMatVecDesc_t<DT> desc[], long long mlevels, deviceMatrixDesc_t<OT> desc_m[], double tol, DT* X, const DT* B, long long inner_iters, long long outer_iters, double resid[]);