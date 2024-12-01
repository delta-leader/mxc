
#include <device_csr_matrix.cuh>
#include <comm-mpi.hpp>

#include <numeric>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <thrust/sort.h>
#include <thrust/iterator/constant_iterator.h>


/* explicit template instantiation */
// complex double
template void createDeviceCsr(CsrContainer_t<std::complex<double>>* A, long long Mb, long long Nb, const long long RowDims[], const long long ColDims[], const long long ARows[], const long long ACols[], const std::complex<double> data[]);
template void createDeviceVec(VecDnContainer_t<std::complex<double>>* X, const long long RowDims[], const long long nodes);
template void destroyDeviceVec(VecDnContainer_t<std::complex<double>> X);
template void createSpMatrixDesc(deviceHandle_t handle, CsrMatVecDesc_t<std::complex<double>>* desc, bool is_leaf, long long lowerZ, const long long Dims[], const long long Ranks[], const std::complex<double> U[], const std::complex<double> C[], const std::complex<double> A[], const H2Matrix<std::complex<double>>& matrix);
template void destroySpMatrixDesc(CsrMatVecDesc_t<std::complex<double>> desc);
template void matVecUpwardPass(deviceHandle_t handle, CsrMatVecDesc_t<std::complex<double>> desc, const std::complex<double>* X_in);
template void matVecHorizontalandDownwardPass(deviceHandle_t handle, CsrMatVecDesc_t<std::complex<double>> desc, std::complex<double>* Y_out);
template void matVecLeafHorizontalPass(deviceHandle_t handle, CsrMatVecDesc_t<std::complex<double>> desc, std::complex<double>* X_io);
template void matVecDeviceH2(deviceHandle_t handle, long long levels, CsrMatVecDesc_t<std::complex<double>> desc[], std::complex<double>* devX);
// complex float
template void createDeviceCsr(CsrContainer_t<std::complex<float>>* A, long long Mb, long long Nb, const long long RowDims[], const long long ColDims[], const long long ARows[], const long long ACols[], const std::complex<float> data[]);
template void createDeviceVec(VecDnContainer_t<std::complex<float>>* X, const long long RowDims[], const long long nodes);
template void destroyDeviceVec(VecDnContainer_t<std::complex<float>> X);
template void createSpMatrixDesc(deviceHandle_t handle, CsrMatVecDesc_t<std::complex<float>>* desc, bool is_leaf, long long lowerZ, const long long Dims[], const long long Ranks[], const std::complex<float> U[], const std::complex<float> C[], const std::complex<float> A[], const H2Matrix<std::complex<float>>& matrix);
template void destroySpMatrixDesc(CsrMatVecDesc_t<std::complex<float>> desc);
template void matVecUpwardPass(deviceHandle_t handle, CsrMatVecDesc_t<std::complex<float>> desc, const std::complex<float>* X_in);
template void matVecHorizontalandDownwardPass(deviceHandle_t handle, CsrMatVecDesc_t<std::complex<float>> desc, std::complex<float>* Y_out);
template void matVecLeafHorizontalPass(deviceHandle_t handle, CsrMatVecDesc_t<std::complex<float>> desc, std::complex<float>* X_io);
template void matVecDeviceH2(deviceHandle_t handle, long long levels, CsrMatVecDesc_t<std::complex<float>> desc[], std::complex<float>* devX);
// double
template void createDeviceCsr(CsrContainer_t<double>* A, long long Mb, long long Nb, const long long RowDims[], const long long ColDims[], const long long ARows[], const long long ACols[], const double data[]);
template void createDeviceVec(VecDnContainer_t<double>* X, const long long RowDims[], const long long nodes);
template void destroyDeviceVec(VecDnContainer_t<double> X);
template void createSpMatrixDesc(deviceHandle_t handle, CsrMatVecDesc_t<double>* desc, bool is_leaf, long long lowerZ, const long long Dims[], const long long Ranks[], const double U[], const double C[], const double A[], const H2Matrix<double>& matrix);
template void destroySpMatrixDesc(CsrMatVecDesc_t<double> desc);
template void matVecUpwardPass(deviceHandle_t handle, CsrMatVecDesc_t<double> desc, const double* X_in);
template void matVecHorizontalandDownwardPass(deviceHandle_t handle, CsrMatVecDesc_t<double> desc, double* Y_out);
template void matVecLeafHorizontalPass(deviceHandle_t handle, CsrMatVecDesc_t<double> desc, double* X_io);
template void matVecDeviceH2(deviceHandle_t handle, long long levels, CsrMatVecDesc_t<double> desc[], double* devX);
// float
template void createDeviceCsr(CsrContainer_t<float>* A, long long Mb, long long Nb, const long long RowDims[], const long long ColDims[], const long long ARows[], const long long ACols[], const float data[]);
template void createDeviceVec(VecDnContainer_t<float>* X, const long long RowDims[], const long long nodes);
template void destroyDeviceVec(VecDnContainer_t<float> X);
template void createSpMatrixDesc(deviceHandle_t handle, CsrMatVecDesc_t<float>* desc, bool is_leaf, long long lowerZ, const long long Dims[], const long long Ranks[], const float U[], const float C[], const float A[], const H2Matrix<float>& matrix);
template void destroySpMatrixDesc(CsrMatVecDesc_t<float> desc);
template void matVecUpwardPass(deviceHandle_t handle, CsrMatVecDesc_t<float> desc, const float* X_in);
template void matVecHorizontalandDownwardPass(deviceHandle_t handle, CsrMatVecDesc_t<float> desc, float* Y_out);
template void matVecLeafHorizontalPass(deviceHandle_t handle, CsrMatVecDesc_t<float> desc, float* X_io);
template void matVecDeviceH2(deviceHandle_t handle, long long levels, CsrMatVecDesc_t<float> desc[], float* devX);

struct genXY {
  const long long* M, *A, *Y, *X;
  genXY(const long long* M, const long long* A, const long long* Y, const long long* X) : M(M), A(A), Y(Y), X(X) {}
  __host__ __device__ thrust::tuple<long long, long long> operator()(long long i, thrust::tuple<long long, long long> c) const {
    long long b = thrust::get<0>(c);
    long long m = M[b]; long long id = i - A[b];
    long long x = id / m; long long y = id - x * m;
    return thrust::make_tuple(Y[b] + y, X[b] + x);
  }
};

template <typename DT>
struct cmpXY {
  __host__ __device__ bool operator()(const thrust::tuple<long long, long long, DT>& l, const thrust::tuple<long long, long long, DT>& r) const {
    return thrust::get<0>(l) == thrust::get<0>(r) ? thrust::get<1>(l) < thrust::get<1>(r) : thrust::get<0>(l) < thrust::get<0>(r);
  }
};

long long computeCooNNZ(long long Mb, const long long RowDims[], const long long ColDims[], const long long ARows[], const long long ACols[]) {
  return std::transform_reduce(ARows, &ARows[Mb], RowDims, 0ll, std::plus<long long>(), [&](const long long& begin, long long rows) { 
    return rows * std::transform_reduce(&ACols[begin], &ACols[(&begin)[1]], 0ll, std::plus<long long>(), [&](long long col) { return ColDims[col]; }); });
}

template <typename DT>
void sort(thrust::device_vector<long long>& Rows, thrust::device_ptr<long long>& ColsPtr, DT devVals[], const long long NNZ) {
  thrust::device_ptr<DT> Vals(devVals);
  auto sort_iter = thrust::make_zip_iterator(Rows.begin(), ColsPtr, Vals);
  thrust::sort(sort_iter, sort_iter + NNZ, cmpXY<DT>());
}

template <>
void sort(thrust::device_vector<long long>& Rows, thrust::device_ptr<long long>& ColsPtr, std::complex<double> devVals[], const long long NNZ) {
  thrust::device_ptr<thrust::complex<double>> Vals(reinterpret_cast<thrust::complex<double>*>(devVals));
  auto sort_iter = thrust::make_zip_iterator(Rows.begin(), ColsPtr, Vals);
  thrust::sort(sort_iter, sort_iter + NNZ, cmpXY<thrust::complex<double>>());
}

template <>
void sort(thrust::device_vector<long long>& Rows, thrust::device_ptr<long long>& ColsPtr, std::complex<float> devVals[], const long long NNZ) {
  thrust::device_ptr<thrust::complex<float>> Vals(reinterpret_cast<thrust::complex<float>*>(devVals));
  auto sort_iter = thrust::make_zip_iterator(Rows.begin(), ColsPtr, Vals);
  thrust::sort(sort_iter, sort_iter + NNZ, cmpXY<thrust::complex<float>>());
}

template <typename DT>
void genCsrEntries(long long CsrM, long long devRowIndx[], long long devColIndx[], DT devVals[], long long Mb, long long Nb, const long long RowDims[], const long long ColDims[], const long long ARows[], const long long ACols[]) {
  long long lenA = ARows[Mb];
  thrust::device_vector<long long> ARowOffset(ARows, &ARows[Mb + 1]);
  thrust::device_vector<long long> ARowIndx(lenA, 0ll);
  thrust::device_vector<long long> AColIndx(ACols, &ACols[lenA]);
  thrust::device_vector<long long> AOffsets(lenA + 1, 0ll);

  thrust::device_vector<long long> devRowOffsets(Mb + 1);
  thrust::device_vector<long long> devColOffsets(Nb + 1);
  thrust::device_vector<long long> devADimM(lenA);
  thrust::device_vector<long long> devAIndY(lenA);
  thrust::device_vector<long long> devAIndX(lenA);

  long long keys_len = std::max(CsrM, std::max(lenA, Mb));
  thrust::device_vector<long long> keys(keys_len);
  thrust::device_vector<long long> counts(keys_len);
  
  auto one_iter = thrust::make_constant_iterator(1ll);
  auto ydim_iter = thrust::make_permutation_iterator(devRowOffsets.begin(), ARowIndx.begin());
  auto xdim_iter = thrust::make_permutation_iterator(devColOffsets.begin(), AColIndx.begin());

  thrust::copy(RowDims, &RowDims[Mb], devRowOffsets.begin());
  thrust::copy(ColDims, &ColDims[Nb], devColOffsets.begin());

  auto counts_end = thrust::reduce_by_key(ARowOffset.begin() + 1, ARowOffset.begin() + Mb, one_iter, keys.begin(), counts.begin()).second;
  thrust::scatter(counts.begin(), counts_end, keys.begin(), ARowIndx.begin()); 
  thrust::inclusive_scan(ARowIndx.begin(), ARowIndx.end(), ARowIndx.begin());
  thrust::transform(ydim_iter, ydim_iter + lenA, xdim_iter, AOffsets.begin(), thrust::multiplies<long long>());
  thrust::exclusive_scan(AOffsets.begin(), AOffsets.end(), AOffsets.begin(), 0ll);

  thrust::copy(ydim_iter, ydim_iter + lenA, devADimM.begin());
  thrust::exclusive_scan(devRowOffsets.begin(), devRowOffsets.end(), devRowOffsets.begin(), 0ll);
  thrust::exclusive_scan(devColOffsets.begin(), devColOffsets.end(), devColOffsets.begin(), 0ll);
  thrust::copy(ydim_iter, ydim_iter + lenA, devAIndY.begin());
  thrust::copy(xdim_iter, xdim_iter + lenA, devAIndX.begin());

  long long NNZ = AOffsets.back();
  thrust::device_vector<long long> Rows(NNZ, 0ll);
  thrust::device_ptr<long long> RowsPtr(devRowIndx);
  thrust::device_ptr<long long> ColsPtr(devColIndx);
  //thrust::device_ptr<thrust::complex<double>> Vals(reinterpret_cast<thrust::complex<double>*>(devVals));
  
  auto ind_iter = thrust::make_zip_iterator(Rows.begin(), ColsPtr);
  auto inc_iter = thrust::make_counting_iterator(0ll);
  //auto sort_iter = thrust::make_zip_iterator(Rows.begin(), ColsPtr, Vals);

  const long long* Mptr = thrust::raw_pointer_cast(devADimM.data());
  const long long* Aptr = thrust::raw_pointer_cast(AOffsets.data());
  const long long* Yptr = thrust::raw_pointer_cast(devAIndY.data());
  const long long* Xptr = thrust::raw_pointer_cast(devAIndX.data());

  counts_end = thrust::reduce_by_key(AOffsets.begin() + 1, AOffsets.begin() + lenA, one_iter, keys.begin(), counts.begin()).second;
  thrust::scatter(counts.begin(), counts_end, keys.begin(), Rows.begin());
  thrust::inclusive_scan(Rows.begin(), Rows.end(), Rows.begin());
  thrust::transform(inc_iter, inc_iter + NNZ, ind_iter, ind_iter, genXY(Mptr, Aptr, Yptr, Xptr));
  //thrust::sort(sort_iter, sort_iter + NNZ, cmpXY());
  sort(Rows, ColsPtr, devVals, NNZ);

  counts_end = thrust::reduce_by_key(Rows.begin(), Rows.end(), one_iter, keys.begin(), counts.begin()).second;
  thrust::fill(RowsPtr, &RowsPtr[CsrM + 1], 0ll);
  thrust::scatter(counts.begin(), counts_end, keys.begin(), RowsPtr);
  thrust::exclusive_scan(RowsPtr, &RowsPtr[CsrM + 1], RowsPtr, 0ll);
}

template <typename DT>
void createDeviceCsr(CsrContainer_t<DT>* A, long long Mb, long long Nb, const long long RowDims[], const long long ColDims[], const long long ARows[], const long long ACols[], const DT data[]) {
  *A = new struct CsrContainer<DT>();
  long long CsrM = (*A)->M = std::reduce(RowDims, &RowDims[Mb]);
  long long NNZ = (*A)->NNZ = computeCooNNZ(Mb, RowDims, ColDims, ARows, ACols);
  (*A)->N = std::reduce(ColDims, &ColDims[Nb]);

  if (0 < NNZ) {
    cudaMalloc(reinterpret_cast<void**>(&((*A)->RowOffsets)), sizeof(long long) * (CsrM + 1));
    cudaMalloc(reinterpret_cast<void**>(&((*A)->ColInd)), sizeof(long long) * NNZ);
    cudaMalloc(reinterpret_cast<void**>(&((*A)->Vals)), sizeof(DT) * NNZ);
    thrust::copy(data, &data[NNZ], thrust::device_ptr<DT>((*A)->Vals));
    genCsrEntries(CsrM, (*A)->RowOffsets, (*A)->ColInd, (*A)->Vals, Mb, Nb, RowDims, ColDims, ARows, ACols);
  }
}

template <typename DT>
void destroyDeviceCsr(CsrContainer_t<DT> A) {
  if (0 < A->NNZ) {
    cudaFree(A->RowOffsets);
    cudaFree(A->ColInd);
    cudaFree(A->Vals);
  }
  delete A;
}

template <typename DT>
void createDeviceVec(VecDnContainer_t<DT>* X, const long long RowDims[], const long long nodes) {
  *X = new struct VecDnContainer<DT>();
  //long long Mb = comm.lenNeighbors();
  long long Mb = nodes;
  std::vector<long long> DimsOffsets(Mb + 1);
  std::inclusive_scan(RowDims, &RowDims[Mb], DimsOffsets.begin() + 1);
  DimsOffsets[0] = 0;

  long long N = (*X)->N = DimsOffsets[Mb];
  if (0 < N) {
    //(*X)->Xbegin = DimsOffsets[comm.oLocal()];
    (*X)->Xbegin = DimsOffsets[0];
    //(*X)->lenX = DimsOffsets[comm.oLocal() + comm.lenLocal()] - (*X)->Xbegin;
    (*X)->lenX = DimsOffsets[nodes] - (*X)->Xbegin;
    cudaMalloc(reinterpret_cast<void**>(&((*X)->Vals)), sizeof(DT) * N);

    /*(*X)->Neighbor = reinterpret_cast<long long*>(std::malloc(comm.BoxOffsets.size() * sizeof(long long)));
    std::transform(comm.BoxOffsets.begin(), comm.BoxOffsets.end(), (*X)->Neighbor, [&](long long i) { return DimsOffsets[i]; });

    (*X)->LenComms = comm.NeighborComm.size();
    if ((*X)->LenComms) {
      (*X)->NeighborRoots = reinterpret_cast<long long*>(std::malloc((*X)->LenComms * sizeof(long long)));
      (*X)->NeighborComms = reinterpret_cast<ncclComm_t*>(std::malloc((*X)->LenComms * sizeof(ncclComm_t)));

      std::transform(comm.NeighborComm.begin(), comm.NeighborComm.end(), (*X)->NeighborRoots, 
        [](const std::pair<int, MPI_Comm>& comm) { return static_cast<long long>(comm.first); });
      std::transform(comm.NeighborComm.begin(), comm.NeighborComm.end(), (*X)->NeighborComms, 
        [=](const std::pair<int, MPI_Comm>& comm) { return findNcclComm(comm.second, nccl_comms); });
    }

    (*X)->DupComm = findNcclComm(comm.DupComm, nccl_comms);*/
  }
}

template <typename DT>
void destroyDeviceVec(VecDnContainer_t<DT> X) {
  if (X->N) {
    if (X->Neighbor)
      std::free(X->Neighbor);
    cudaFree(X->Vals);
    if (X->LenComms) {
      std::free(X->NeighborRoots);
      std::free(X->NeighborComms);
    }
  }
  delete X;
}

inline void createDnVec(cusparseDnVecDescr_t* dnVecDescr, int64_t size, std::complex<double>* values) {
  cusparseCreateDnVec(dnVecDescr, size, values, CUDA_C_64F);
}

inline void createDnVec(cusparseDnVecDescr_t* dnVecDescr, int64_t size, std::complex<float>* values) {
  cusparseCreateDnVec(dnVecDescr, size, values, CUDA_C_32F);
}

inline void createDnVec(cusparseDnVecDescr_t* dnVecDescr, int64_t size, double* values) {
  cusparseCreateDnVec(dnVecDescr, size, values, CUDA_R_64F);
}

inline void createDnVec(cusparseDnVecDescr_t* dnVecDescr, int64_t size, float* values) {
  cusparseCreateDnVec(dnVecDescr, size, values, CUDA_R_32F);
}

inline void createConstCsr(cusparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
  const long long* csrRowOffsets, const long long* csrColInd, const std::complex<double>* csrValues, cusparseIndexType_t csrRowOffsetsType,
  cusparseIndexType_t csrColIndType, cusparseIndexBase_t idxBase) {
    cusparseCreateConstCsr(spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, idxBase, CUDA_C_64F);
}

inline void createConstCsr(cusparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
  const long long* csrRowOffsets, const long long* csrColInd, const std::complex<float>* csrValues, cusparseIndexType_t csrRowOffsetsType,
  cusparseIndexType_t csrColIndType, cusparseIndexBase_t idxBase) {
    cusparseCreateConstCsr(spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, idxBase, CUDA_C_32F);
}

inline void createConstCsr(cusparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
  const long long* csrRowOffsets, const long long* csrColInd, const double* csrValues, cusparseIndexType_t csrRowOffsetsType,
  cusparseIndexType_t csrColIndType, cusparseIndexBase_t idxBase) {
    cusparseCreateConstCsr(spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, idxBase, CUDA_R_64F);
}

inline void createConstCsr(cusparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
  const long long* csrRowOffsets, const long long* csrColInd, const float* csrValues, cusparseIndexType_t csrRowOffsetsType,
  cusparseIndexType_t csrColIndType, cusparseIndexBase_t idxBase) {
    cusparseCreateConstCsr(spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, idxBase, CUDA_R_32F);
}

inline void createConstCsc(cusparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
  const long long* cscRowOffsets, const long long* cscColInd, const std::complex<double>* cscValues, cusparseIndexType_t cscRowOffsetsType,
  cusparseIndexType_t cscColIndType, cusparseIndexBase_t idxBase) {
    cusparseCreateConstCsc(spMatDescr, rows, cols, nnz, cscRowOffsets, cscColInd, cscValues, cscRowOffsetsType, cscColIndType, idxBase, CUDA_C_64F);
}

inline void createConstCsc(cusparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
  const long long* cscRowOffsets, const long long* cscColInd, const std::complex<float>* cscValues, cusparseIndexType_t cscRowOffsetsType,
  cusparseIndexType_t cscColIndType, cusparseIndexBase_t idxBase) {
    cusparseCreateConstCsc(spMatDescr, rows, cols, nnz, cscRowOffsets, cscColInd, cscValues, cscRowOffsetsType, cscColIndType, idxBase, CUDA_C_32F);
}

inline void createConstCsc(cusparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
  const long long* cscRowOffsets, const long long* cscColInd, const double* cscValues, cusparseIndexType_t cscRowOffsetsType,
  cusparseIndexType_t cscColIndType, cusparseIndexBase_t idxBase) {
    cusparseCreateConstCsc(spMatDescr, rows, cols, nnz, cscRowOffsets, cscColInd, cscValues, cscRowOffsetsType, cscColIndType, idxBase, CUDA_R_64F);
}

inline void createConstCsc(cusparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
  const long long* cscRowOffsets, const long long* cscColInd, const float* cscValues, cusparseIndexType_t cscRowOffsetsType,
  cusparseIndexType_t cscColIndType, cusparseIndexBase_t idxBase) {
    cusparseCreateConstCsc(spMatDescr, rows, cols, nnz, cscRowOffsets, cscColInd, cscValues, cscRowOffsetsType, cscColIndType, idxBase, CUDA_R_32F);
}

inline cusparseStatus_t spMV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, const std::complex<double>* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX,
  const std::complex<double>* beta, cusparseDnVecDescr_t vecY, cusparseSpMVAlg_t alg, size_t* bufferSize) {
  return cusparseSpMV_bufferSize(handle, opA, alpha, matA,  vecX, beta, vecY, CUDA_C_64F, alg, bufferSize);
}

inline cusparseStatus_t spMV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, const std::complex<float>* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX,
  const std::complex<float>* beta, cusparseDnVecDescr_t vecY, cusparseSpMVAlg_t alg, size_t* bufferSize) {
  return cusparseSpMV_bufferSize(handle, opA, alpha, matA, vecX, beta, vecY, CUDA_C_32F, alg, bufferSize);
}

inline cusparseStatus_t spMV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, const double* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX,
  const double* beta, cusparseDnVecDescr_t vecY, cusparseSpMVAlg_t alg, size_t* bufferSize) {
  return cusparseSpMV_bufferSize(handle, opA, alpha, matA,  vecX, beta, vecY, CUDA_R_64F, alg, bufferSize);
}

inline cusparseStatus_t spMV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, const float* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX,
  const float* beta, cusparseDnVecDescr_t vecY, cusparseSpMVAlg_t alg, size_t* bufferSize) {
  return cusparseSpMV_bufferSize(handle, opA, alpha, matA,  vecX, beta, vecY, CUDA_R_32F, alg, bufferSize);
}

template <typename DT>
void createSpMatrixDesc(deviceHandle_t handle, CsrMatVecDesc_t<DT>* desc, bool is_leaf, long long lowerZ, const long long Dims[], const long long Ranks[], const DT U[], const DT C[], const DT A[], const H2Matrix<DT>& matrix) {
  *desc = new struct CsrMatVecDesc<DT>();
  (*desc)->lowerZ = lowerZ;
  
  //long long ibegin = comm.oLocal();
  //long long nodes = comm.lenLocal();
  //long long xlen = comm.lenNeighbors();
  long long nodes = matrix.nodes;
  long long xlen = nodes;
  createDeviceVec(&((*desc)->X), Dims, nodes);
  createDeviceVec(&((*desc)->Y), Dims, nodes);
  createDeviceVec(&((*desc)->Z), Ranks, nodes);
  createDeviceVec(&((*desc)->W), Ranks, nodes);
  std::vector<long long> seq(nodes + 1);
  std::iota(seq.begin(), seq.end(), 0ll);
  createDeviceCsr(&((*desc)->U), nodes, nodes, &Dims[0], &Ranks[0], &seq[0], &seq[0], U);
  //createDeviceCsr(&((*desc)->C), nodes, xlen, &Ranks[0], Ranks, comm.CRowOffsets.data(), comm.CColumns.data(), C);
  createDeviceCsr(&((*desc)->C), nodes, xlen, &Ranks[0], Ranks, matrix.CRows.data(), matrix.CCols.data(), C);
  if (is_leaf) {
    //createDeviceCsr(&((*desc)->A), nodes, xlen, &Dims[0], Dims, comm.ARowOffsets.data(), comm.AColumns.data(), A);
    createDeviceCsr(&((*desc)->A), nodes, xlen, &Dims[0], Dims, matrix.ARows.data(), matrix.ACols.data(), A);
  } else
    (*desc)->A = new struct CsrContainer<DT>();
  if ((*desc)->X->N) {
    createDnVec(&((*desc)->descX), (*desc)->X->N, (*desc)->X->Vals);
    createDnVec(&((*desc)->descXi), (*desc)->X->lenX, (*desc)->X->Vals + (*desc)->X->Xbegin);
    createDnVec(&((*desc)->descYi), (*desc)->Y->lenX, (*desc)->Y->Vals + (*desc)->Y->Xbegin);
  }

  if ((*desc)->Z->N) {
    createDnVec(&((*desc)->descZ), (*desc)->Z->N, (*desc)->Z->Vals);
    createDnVec(&((*desc)->descZi), (*desc)->Z->lenX, (*desc)->Z->Vals + (*desc)->Z->Xbegin);
    createDnVec(&((*desc)->descWi), (*desc)->W->lenX, (*desc)->W->Vals + (*desc)->W->Xbegin);
  }
  size_t buffer_size = 0, buffer;
  DT one{1.}, zero{0.};

  if ((*desc)->U->NNZ) {
    createConstCsr(&((*desc)->descU), (*desc)->U->M, (*desc)->U->N, (*desc)->U->NNZ, (*desc)->U->RowOffsets, (*desc)->U->ColInd, (*desc)->U->Vals, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO);
    createConstCsc(&((*desc)->descV), (*desc)->U->N, (*desc)->U->M, (*desc)->U->NNZ, (*desc)->U->RowOffsets, (*desc)->U->ColInd, (*desc)->U->Vals, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO);

    spMV_bufferSize(handle->cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, (*desc)->descV, (*desc)->descXi, &zero, (*desc)->descZi, CUSPARSE_SPMV_ALG_DEFAULT, &buffer);
    buffer_size = std::max(buffer, buffer_size);
    spMV_bufferSize(handle->cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, (*desc)->descU, (*desc)->descWi, &one, (*desc)->descYi, CUSPARSE_SPMV_ALG_DEFAULT, &buffer);
    buffer_size = std::max(buffer, buffer_size);
  }
  if ((*desc)->C->NNZ) {
    createConstCsr(&((*desc)->descC), (*desc)->C->M, (*desc)->C->N, (*desc)->C->NNZ, (*desc)->C->RowOffsets, (*desc)->C->ColInd, (*desc)->C->Vals, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO);
    spMV_bufferSize(handle->cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, (*desc)->descC, (*desc)->descZ, &one, (*desc)->descWi, CUSPARSE_SPMV_ALG_DEFAULT, &buffer);
    buffer_size = std::max(buffer, buffer_size);
  }

  if ((*desc)->A->NNZ) {
    createConstCsr(&((*desc)->descA), (*desc)->A->M, (*desc)->A->N, (*desc)->A->NNZ, (*desc)->A->RowOffsets, (*desc)->A->ColInd, (*desc)->A->Vals, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO);
    spMV_bufferSize(handle->cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, (*desc)->descA, (*desc)->descX, &one, (*desc)->descYi, CUSPARSE_SPMV_ALG_DEFAULT, &buffer);
    buffer_size = std::max(buffer, buffer_size);
  }

  if (buffer_size) {
    (*desc)->buffer_size = buffer_size;
    cudaMalloc(reinterpret_cast<void**>(&((*desc)->buffer)), buffer_size);
  }
}

template <typename DT>
void destroySpMatrixDesc(CsrMatVecDesc_t<DT> desc) {
  if (desc->X->N) {
    cusparseDestroyDnVec(desc->descX);
    cusparseDestroyDnVec(desc->descXi);
    cusparseDestroyDnVec(desc->descYi);
  }

  if (desc->Z->N) {
    cusparseDestroyDnVec(desc->descZ);
    cusparseDestroyDnVec(desc->descZi);
    cusparseDestroyDnVec(desc->descWi);
  }

  if (desc->U->NNZ) {
    cusparseDestroySpMat(desc->descU);
    cusparseDestroySpMat(desc->descV);
  }
  if (desc->C->NNZ)
    cusparseDestroySpMat(desc->descC);
  if (desc->A->NNZ)
    cusparseDestroySpMat(desc->descA);

  destroyDeviceVec(desc->X);
  destroyDeviceVec(desc->Y);
  destroyDeviceVec(desc->Z);
  destroyDeviceVec(desc->W);

  destroyDeviceCsr(desc->U);
  destroyDeviceCsr(desc->C);
  destroyDeviceCsr(desc->A);

  if (desc->buffer_size)
    cudaFree(desc->buffer);
  delete desc;
}
/*
void deviceVecNeighborBcast(cudaStream_t stream, VecDnContainer_t<std::complex<double>> X) {
  if (X->N) {
    ncclGroupStart();
    for (long long p = 0; p < X->LenComms; p++) {
      long long start = (X->Neighbor)[p];
      long long len = (X->Neighbor)[p + 1] - start;
      ncclBroadcast(const_cast<const std::complex<double>*>(&(X->Vals)[start]), &(X->Vals)[start], len * 2, ncclDouble, X->NeighborRoots[p], X->NeighborComms[p], stream);
    }

    if (X->DupComm)
      ncclBroadcast(const_cast<const std::complex<double>*>(X->Vals), X->Vals, X->N * 2, ncclDouble, 0, X->DupComm, stream);
    ncclGroupEnd();
  }
}

void deviceVecNeighborBcast(cudaStream_t stream, VecDnContainer_t<std::complex<float>> X) {
  if (X->N) {
    ncclGroupStart();
    for (long long p = 0; p < X->LenComms; p++) {
      long long start = (X->Neighbor)[p];
      long long len = (X->Neighbor)[p + 1] - start;
      ncclBroadcast(const_cast<const std::complex<float>*>(&(X->Vals)[start]), &(X->Vals)[start], len * 2, ncclFloat, X->NeighborRoots[p], X->NeighborComms[p], stream);
    }

    if (X->DupComm)
      ncclBroadcast(const_cast<const std::complex<float>*>(X->Vals), X->Vals, X->N * 2, ncclFloat, 0, X->DupComm, stream);
    ncclGroupEnd();
  }
}

void deviceVecNeighborBcast(cudaStream_t stream, VecDnContainer_t<double> X) {
  if (X->N) {
    ncclGroupStart();
    for (long long p = 0; p < X->LenComms; p++) {
      long long start = (X->Neighbor)[p];
      long long len = (X->Neighbor)[p + 1] - start;
      ncclBroadcast(const_cast<const double*>(&(X->Vals)[start]), &(X->Vals)[start], len, ncclDouble, X->NeighborRoots[p], X->NeighborComms[p], stream);
    }

    if (X->DupComm)
      ncclBroadcast(const_cast<const double*>(X->Vals), X->Vals, X->N, ncclDouble, 0, X->DupComm, stream);
    ncclGroupEnd();
  }
}

void deviceVecNeighborBcast(cudaStream_t stream, VecDnContainer_t<float> X) {
  if (X->N) {
    ncclGroupStart();
    for (long long p = 0; p < X->LenComms; p++) {
      long long start = (X->Neighbor)[p];
      long long len = (X->Neighbor)[p + 1] - start;
      ncclBroadcast(const_cast<const float*>(&(X->Vals)[start]), &(X->Vals)[start], len, ncclFloat, X->NeighborRoots[p], X->NeighborComms[p], stream);
    }

    if (X->DupComm)
      ncclBroadcast(const_cast<const float*>(X->Vals), X->Vals, X->N, ncclFloat, 0, X->DupComm, stream);
    ncclGroupEnd();
  }
}*/

inline void spMV(cusparseHandle_t handle, cusparseOperation_t opA, const std::complex<double>* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX,
  const std::complex<double>* beta, cusparseDnVecDescr_t vecY, cusparseSpMVAlg_t alg, void* externalBuffer) {
    cusparseSpMV(handle, opA, alpha, matA, vecX, beta, vecY, CUDA_C_64F, alg, externalBuffer);
}

inline void spMV(cusparseHandle_t handle, cusparseOperation_t opA, const std::complex<float>* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX,
  const std::complex<float>* beta, cusparseDnVecDescr_t vecY, cusparseSpMVAlg_t alg, void* externalBuffer) {
    cusparseSpMV(handle, opA, alpha, matA, vecX, beta, vecY, CUDA_C_32F, alg, externalBuffer);
}

inline void spMV(cusparseHandle_t handle, cusparseOperation_t opA, const double* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX,
  const double* beta, cusparseDnVecDescr_t vecY, cusparseSpMVAlg_t alg, void* externalBuffer) {
    cusparseSpMV(handle, opA, alpha, matA, vecX, beta, vecY, CUDA_R_64F, alg, externalBuffer);
}

inline void spMV(cusparseHandle_t handle, cusparseOperation_t opA, const float* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX,
  const float* beta, cusparseDnVecDescr_t vecY, cusparseSpMVAlg_t alg, void* externalBuffer) {
    cusparseSpMV(handle, opA, alpha, matA, vecX, beta, vecY, CUDA_R_32F, alg, externalBuffer);
}

template <typename DT>
void matVecUpwardPass(deviceHandle_t handle, CsrMatVecDesc_t<DT> desc, const DT* X_in) {
  long long lenX = desc->X->lenX;
  cudaStream_t stream = handle->compute_stream;
  cusparseHandle_t cusparseH = handle->cusparseH;
  if (lenX) {
    cudaMemcpyAsync(desc->X->Vals + desc->X->Xbegin, &X_in[desc->lowerZ], sizeof(DT) * lenX, cudaMemcpyDeviceToDevice, stream);
    cudaMemsetAsync(desc->Y->Vals + desc->Y->Xbegin, 0, sizeof(DT) * lenX, stream);
  }

  DT one{1.}, zero{0.};
  if (desc->U->NNZ)
    spMV(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, desc->descV, desc->descXi, &zero, desc->descZi, CUSPARSE_SPMV_ALG_DEFAULT, desc->buffer);
  //deviceVecNeighborBcast(stream, desc->Z);
}

template <typename DT>
void matVecHorizontalandDownwardPass(deviceHandle_t handle, CsrMatVecDesc_t<DT> desc, DT* Y_out) {
  long long lenX = desc->X->lenX;
  cudaStream_t stream = handle->compute_stream;
  cusparseHandle_t cusparseH = handle->cusparseH;

  DT one{1.};
  if (desc->C->NNZ)
    spMV(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, desc->descC, desc->descZ, &one, desc->descWi, CUSPARSE_SPMV_ALG_DEFAULT, desc->buffer);
  if (desc->U->NNZ)
    spMV(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, desc->descU, desc->descWi, &one, desc->descYi, CUSPARSE_SPMV_ALG_DEFAULT, desc->buffer);

  if (lenX)
    cudaMemcpyAsync(&Y_out[desc->lowerZ], desc->Y->Vals + desc->Y->Xbegin, sizeof(DT) * lenX, cudaMemcpyDeviceToDevice, stream);
}


template <typename DT>
void matVecLeafHorizontalPass(deviceHandle_t handle, CsrMatVecDesc_t<DT> desc, DT* X_io) {
  long long lenX = desc->X->lenX;
  cudaStream_t stream = handle->compute_stream;
  cusparseHandle_t cusparseH = handle->cusparseH;
  if (lenX)
    cudaMemcpyAsync(desc->X->Vals + desc->X->Xbegin, X_io, sizeof(DT) * lenX, cudaMemcpyDeviceToDevice, stream);
  //deviceVecNeighborBcast(stream, desc->X);

  DT one{1.};
  if (desc->A->NNZ)
    spMV(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, desc->descA, desc->descX, &one, desc->descYi, CUSPARSE_SPMV_ALG_DEFAULT, desc->buffer);
  if (desc->C->NNZ)
    spMV(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, desc->descC, desc->descZ, &one, desc->descWi, CUSPARSE_SPMV_ALG_DEFAULT, desc->buffer);
  if (desc->U->NNZ)
    spMV(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, desc->descU, desc->descWi, &one, desc->descYi, CUSPARSE_SPMV_ALG_DEFAULT, desc->buffer);

  if (lenX)
    cudaMemcpyAsync(X_io, desc->Y->Vals + desc->Y->Xbegin, sizeof(DT) * lenX, cudaMemcpyDeviceToDevice, stream);
}

template <typename DT>
void matVecDeviceH2(deviceHandle_t handle, long long levels, CsrMatVecDesc_t<DT> desc[], DT* devX) {
  if (0 <= levels) {
    matVecUpwardPass(handle, desc[levels], devX);
    for (long long l = levels - 1; l >= 0; l--)
    matVecUpwardPass(handle, desc[l], desc[l + 1]->Z->Vals);

    for (long long l = 0; l < levels; l++)
      matVecHorizontalandDownwardPass(handle, desc[l], desc[l + 1]->W->Vals);
    matVecLeafHorizontalPass(handle, desc[levels], devX);
  }
}
