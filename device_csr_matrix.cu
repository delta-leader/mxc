
#include <device_csr_matrix.cuh>
#include <comm-mpi.hpp>

#include <numeric>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <thrust/sort.h>
#include <thrust/iterator/constant_iterator.h>

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

struct cmpXY {
  __host__ __device__ bool operator()(const thrust::tuple<long long, long long, thrust::complex<double>>& l, const thrust::tuple<long long, long long, thrust::complex<double>>& r) const {
    return thrust::get<0>(l) == thrust::get<0>(r) ? thrust::get<1>(l) < thrust::get<1>(r) : thrust::get<0>(l) < thrust::get<0>(r);
  }
};

long long computeCooNNZ(long long Mb, const long long RowDims[], const long long ColDims[], const long long ARows[], const long long ACols[]) {
  return std::transform_reduce(ARows, &ARows[Mb], RowDims, 0ll, std::plus<long long>(), [&](const long long& begin, long long rows) { 
    return rows * std::transform_reduce(&ACols[begin], &ACols[(&begin)[1]], 0ll, std::plus<long long>(), [&](long long col) { return ColDims[col]; }); });
}

void genCsrEntries(long long CsrM, long long devRowIndx[], long long devColIndx[], std::complex<double> devVals[], long long Mb, long long Nb, const long long RowDims[], const long long ColDims[], const long long ARows[], const long long ACols[]) {
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
  thrust::device_ptr<thrust::complex<double>> Vals(reinterpret_cast<thrust::complex<double>*>(devVals));

  auto ind_iter = thrust::make_zip_iterator(Rows.begin(), ColsPtr);
  auto inc_iter = thrust::make_counting_iterator(0ll);
  auto sort_iter = thrust::make_zip_iterator(Rows.begin(), ColsPtr, Vals);

  const long long* Mptr = thrust::raw_pointer_cast(devADimM.data());
  const long long* Aptr = thrust::raw_pointer_cast(AOffsets.data());
  const long long* Yptr = thrust::raw_pointer_cast(devAIndY.data());
  const long long* Xptr = thrust::raw_pointer_cast(devAIndX.data());

  counts_end = thrust::reduce_by_key(AOffsets.begin() + 1, AOffsets.begin() + lenA, one_iter, keys.begin(), counts.begin()).second;
  thrust::scatter(counts.begin(), counts_end, keys.begin(), Rows.begin());
  thrust::inclusive_scan(Rows.begin(), Rows.end(), Rows.begin());
  thrust::transform(inc_iter, inc_iter + NNZ, ind_iter, ind_iter, genXY(Mptr, Aptr, Yptr, Xptr));
  thrust::sort(sort_iter, sort_iter + NNZ, cmpXY());

  counts_end = thrust::reduce_by_key(Rows.begin(), Rows.end(), one_iter, keys.begin(), counts.begin()).second;
  thrust::fill(RowsPtr, &RowsPtr[CsrM + 1], 0ll);
  thrust::scatter(counts.begin(), counts_end, keys.begin(), RowsPtr);
  thrust::exclusive_scan(RowsPtr, &RowsPtr[CsrM + 1], RowsPtr, 0ll);
}

void createDeviceCsr(CsrContainer_t* A, long long Mb, long long Nb, const long long RowDims[], const long long ColDims[], const long long ARows[], const long long ACols[], const std::complex<double> data[]) {
  *A = new struct CsrContainer();
  long long CsrM = (*A)->M = std::reduce(RowDims, &RowDims[Mb]);
  long long NNZ = (*A)->NNZ = computeCooNNZ(Mb, RowDims, ColDims, ARows, ACols);
  (*A)->N = std::reduce(ColDims, &ColDims[Nb]);

  if (0 < NNZ) {
    cudaMalloc(reinterpret_cast<void**>(&((*A)->RowOffsets)), sizeof(long long) * (CsrM + 1));
    cudaMalloc(reinterpret_cast<void**>(&((*A)->ColInd)), sizeof(long long) * NNZ);
    cudaMalloc(reinterpret_cast<void**>(&((*A)->Vals)), sizeof(std::complex<double>) * NNZ);
    thrust::copy(data, &data[NNZ], thrust::device_ptr<std::complex<double>>((*A)->Vals));
    genCsrEntries(CsrM, (*A)->RowOffsets, (*A)->ColInd, (*A)->Vals, Mb, Nb, RowDims, ColDims, ARows, ACols);
  }
}

void destroyDeviceCsr(CsrContainer_t A) {
  if (0 < A->NNZ) {
    cudaFree(A->RowOffsets);
    cudaFree(A->ColInd);
    cudaFree(A->Vals);
  }
  delete A;
}

void createDeviceVec(VecDnContainer_t* X, const long long RowDims[], const ColCommMPI& comm) {
  *X = new struct VecDnContainer();
  long long Mb = comm.lenNeighbors();
  std::vector<long long> DimsOffsets(Mb + 1);
  std::inclusive_scan(RowDims, &RowDims[Mb], DimsOffsets.begin() + 1);
  DimsOffsets[0] = 0;

  long long N = (*X)->N = DimsOffsets[Mb];
  if (0 < N) {
    (*X)->Xbegin = DimsOffsets[comm.oLocal()];
    (*X)->lenX = DimsOffsets[comm.oLocal() + comm.lenLocal()] - (*X)->Xbegin;
    (*X)->Neighbor = reinterpret_cast<long long*>(std::malloc(comm.BoxOffsets.size() * sizeof(long long)));
    cudaMalloc(reinterpret_cast<void**>(&((*X)->Vals)), sizeof(std::complex<double>) * N);
    std::transform(comm.BoxOffsets.begin(), comm.BoxOffsets.end(), (*X)->Neighbor, [&](long long i) { return DimsOffsets[i]; });
  }
}

void destroyDeviceVec(VecDnContainer_t X) {
  if (0 < X->N) {
    std::free(X->Neighbor);
    cudaFree(X->Vals);
  }
  delete X;
}

void createSpMatrixDesc(deviceHandle_t handle, CsrMatVecDesc_t* desc, bool is_leaf, long long lowerZ, const long long Dims[], const long long Ranks[], const std::complex<double> U[], const std::complex<double> C[], const std::complex<double> A[], const ColCommMPI& comm) {
  *desc = new struct CsrMatVecDesc();
  (*desc)->lowerZ = lowerZ;
  createDeviceVec(&((*desc)->X), Dims, comm);
  createDeviceVec(&((*desc)->Y), Dims, comm);
  createDeviceVec(&((*desc)->Z), Ranks, comm);
  createDeviceVec(&((*desc)->W), Ranks, comm);

  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  long long xlen = comm.lenNeighbors();
  std::vector<long long> seq(nodes + 1);
  std::iota(seq.begin(), seq.end(), 0ll);
  createDeviceCsr(&((*desc)->U), nodes, nodes, &Dims[ibegin], &Ranks[ibegin], &seq[0], &seq[0], U);
  createDeviceCsr(&((*desc)->C), nodes, xlen, &Ranks[ibegin], Ranks, comm.CRowOffsets.data(), comm.CColumns.data(), C);
  if (is_leaf)
    createDeviceCsr(&((*desc)->A), nodes, xlen, &Dims[ibegin], Dims, comm.ARowOffsets.data(), comm.AColumns.data(), A);
  else
    (*desc)->A = new struct CsrContainer();

  if ((*desc)->X->N) {
    cusparseCreateDnVec(&((*desc)->descX), (*desc)->X->N, (*desc)->X->Vals, CUDA_C_64F);
    cusparseCreateDnVec(&((*desc)->descXi), (*desc)->X->lenX, (*desc)->X->Vals + (*desc)->X->Xbegin, CUDA_C_64F);
    cusparseCreateDnVec(&((*desc)->descYi), (*desc)->Y->lenX, (*desc)->Y->Vals + (*desc)->Y->Xbegin, CUDA_C_64F);
  }

  if ((*desc)->Z->N) {
    cusparseCreateDnVec(&((*desc)->descZ), (*desc)->Z->N, (*desc)->Z->Vals, CUDA_C_64F);
    cusparseCreateDnVec(&((*desc)->descZi), (*desc)->Z->lenX, (*desc)->Z->Vals + (*desc)->Z->Xbegin, CUDA_C_64F);
    cusparseCreateDnVec(&((*desc)->descWi), (*desc)->W->lenX, (*desc)->W->Vals + (*desc)->W->Xbegin, CUDA_C_64F);
  }

  size_t buffer_size = 0, buffer;
  std::complex<double> one(1., 0.), zero(0., 0.);

  if ((*desc)->U->NNZ) {
    cusparseCreateConstCsr(&((*desc)->descU), (*desc)->U->M, (*desc)->U->N, (*desc)->U->NNZ, (*desc)->U->RowOffsets, (*desc)->U->ColInd, (*desc)->U->Vals, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F);
    cusparseCreateConstCsc(&((*desc)->descV), (*desc)->U->N, (*desc)->U->M, (*desc)->U->NNZ, (*desc)->U->RowOffsets, (*desc)->U->ColInd, (*desc)->U->Vals, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F);

    cusparseSpMV_bufferSize(handle->cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, (*desc)->descV, (*desc)->descXi, &zero, (*desc)->descZi, CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer);
    buffer_size = std::max(buffer, buffer_size);
    cusparseSpMV_bufferSize(handle->cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, (*desc)->descU, (*desc)->descWi, &one, (*desc)->descYi, CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer);
    buffer_size = std::max(buffer, buffer_size);
  }

  if ((*desc)->C->NNZ) {
    cusparseCreateConstCsr(&((*desc)->descC), (*desc)->C->M, (*desc)->C->N, (*desc)->C->NNZ, (*desc)->C->RowOffsets, (*desc)->C->ColInd, (*desc)->C->Vals, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F);
    cusparseSpMV_bufferSize(handle->cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, (*desc)->descC, (*desc)->descZ, &one, (*desc)->descWi, CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer);
    buffer_size = std::max(buffer, buffer_size);
  }

  if ((*desc)->A->NNZ) {
    cusparseCreateConstCsr(&((*desc)->descA), (*desc)->A->M, (*desc)->A->N, (*desc)->A->NNZ, (*desc)->A->RowOffsets, (*desc)->A->ColInd, (*desc)->A->Vals, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F);
    cusparseSpMV_bufferSize(handle->cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, (*desc)->descA, (*desc)->descX, &one, (*desc)->descYi, CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer);
    buffer_size = std::max(buffer, buffer_size);
  }

  if (buffer_size) {
    (*desc)->buffer_size = buffer_size;
    cudaMalloc(reinterpret_cast<void**>(&((*desc)->buffer)), buffer_size);
  }
}

void destroySpMatrixDesc(CsrMatVecDesc_t desc) {
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

void matVecUpwardPass(deviceHandle_t handle, CsrMatVecDesc_t desc, const std::complex<double>* X_in, const ColCommMPI& comm, const ncclComms nccl_comms) {
  long long lenX = desc->X->lenX;
  cudaStream_t stream = handle->compute_stream;
  cusparseHandle_t cusparseH = handle->cusparseH;
  if (lenX) {
    cudaMemcpyAsync(desc->X->Vals + desc->X->Xbegin, &X_in[desc->lowerZ], sizeof(std::complex<double>) * lenX, cudaMemcpyDeviceToDevice, stream);
    cudaMemsetAsync(desc->Y->Vals + desc->Y->Xbegin, 0, sizeof(std::complex<double>) * lenX, stream);
  }

  std::complex<double> one(1., 0.), zero(0., 0.);
  if (desc->U->NNZ)
    cusparseSpMV(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, desc->descV, desc->descXi, &zero, desc->descZi, CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, desc->buffer);

  if (1 < comm.lenNeighbors() && desc->Z->N) {
    ncclGroupStart();
    for (long long p = 0; p < (long long)comm.NeighborComm.size(); p++) {
      long long start = (desc->Z->Neighbor)[p];
      long long len = (desc->Z->Neighbor)[p + 1] - start;
      auto neighbor = nccl_comms->find(comm.NeighborComm[p].second);
      ncclBroadcast(const_cast<const std::complex<double>*>(&(desc->Z->Vals)[start]), &(desc->Z->Vals)[start], len * 2, ncclDouble, comm.NeighborComm[p].first, (*neighbor).second, stream);
    }

    auto dup = nccl_comms->find(comm.DupComm);
    if (comm.DupComm != MPI_COMM_NULL)
      ncclBroadcast(const_cast<const std::complex<double>*>(desc->Z->Vals), desc->Z->Vals, desc->Z->N * 2, ncclDouble, 0, (*dup).second, stream);
    ncclGroupEnd();
  }
}

void matVecHorizontalandDownwardPass(deviceHandle_t handle, CsrMatVecDesc_t desc, std::complex<double>* Y_out) {
  long long lenX = desc->X->lenX;
  cudaStream_t stream = handle->compute_stream;
  cusparseHandle_t cusparseH = handle->cusparseH;

  std::complex<double> one(1., 0.);
  if (desc->C->NNZ)
    cusparseSpMV(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, desc->descC, desc->descZ, &one, desc->descWi, CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, desc->buffer);
  if (desc->U->NNZ)
    cusparseSpMV(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, desc->descU, desc->descWi, &one, desc->descYi, CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, desc->buffer);

  if (lenX)
    cudaMemcpyAsync(&Y_out[desc->lowerZ], desc->Y->Vals + desc->Y->Xbegin, sizeof(std::complex<double>) * lenX, cudaMemcpyDeviceToDevice, stream);
}

void matVecLeafHorizontalPass(deviceHandle_t handle, CsrMatVecDesc_t desc, std::complex<double>* X_io, const ColCommMPI& comm, const ncclComms nccl_comms) {
  long long lenX = desc->X->lenX;
  cudaStream_t stream = handle->compute_stream;
  cusparseHandle_t cusparseH = handle->cusparseH;
  if (lenX)
    cudaMemcpyAsync(desc->X->Vals + desc->X->Xbegin, X_io, sizeof(std::complex<double>) * lenX, cudaMemcpyDeviceToDevice, stream);

  if (1 < comm.lenNeighbors() && desc->X->N) {
    ncclGroupStart();
    for (long long p = 0; p < (long long)comm.NeighborComm.size(); p++) {
      long long start = (desc->X->Neighbor)[p];
      long long len = (desc->X->Neighbor)[p + 1] - start;
      auto neighbor = nccl_comms->find(comm.NeighborComm[p].second);
      ncclBroadcast(const_cast<const std::complex<double>*>(&(desc->X->Vals)[start]), &(desc->X->Vals)[start], len * 2, ncclDouble, comm.NeighborComm[p].first, (*neighbor).second, stream);
    }

    auto dup = nccl_comms->find(comm.DupComm);
    if (comm.DupComm != MPI_COMM_NULL)
      ncclBroadcast(const_cast<const std::complex<double>*>(desc->X->Vals), desc->X->Vals, desc->X->N * 2, ncclDouble, 0, (*dup).second, stream);
    ncclGroupEnd();
  }

  std::complex<double> one(1., 0.);
  if (desc->A->NNZ)
    cusparseSpMV(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, desc->descA, desc->descX, &one, desc->descYi, CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, desc->buffer);
  if (desc->C->NNZ)
    cusparseSpMV(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, desc->descC, desc->descZ, &one, desc->descWi, CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, desc->buffer);
  if (desc->U->NNZ)
    cusparseSpMV(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, desc->descU, desc->descWi, &one, desc->descYi, CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, desc->buffer);

  if (lenX)
    cudaMemcpyAsync(X_io, desc->Y->Vals + desc->Y->Xbegin, sizeof(std::complex<double>) * lenX, cudaMemcpyDeviceToDevice, stream);
}
