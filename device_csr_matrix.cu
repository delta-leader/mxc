
#include <device_csr_matrix.cuh>
#include <comm-mpi.hpp>

#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <thrust/sort.h>
#include <thrust/iterator/constant_iterator.h>

#include <mkl_spblas.h>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <iostream>

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

struct lltoi {
  __host__ __device__ int operator()(long long i) { return static_cast<int>(i); }
};

void createDeviceCsr(CsrContainer_t* A, long long Mb, long long Nb, const long long RowDims[], const long long ColDims[], const long long ARows[], const long long ACols[], const std::complex<double> data[]) {
  *A = new struct CsrContainer();
  long long CsrM = (*A)->M = std::reduce(RowDims, &RowDims[Mb]);
  long long NNZ = (*A)->NNZ = computeCooNNZ(Mb, RowDims, ColDims, ARows, ACols);
  (*A)->N = std::reduce(ColDims, &ColDims[Nb]);

  if (0 < NNZ) {
    cudaMalloc(reinterpret_cast<void**>(&((*A)->RowOffsets)), sizeof(int) * (CsrM + 1));
    cudaMalloc(reinterpret_cast<void**>(&((*A)->ColInd)), sizeof(int) * NNZ);
    cudaMalloc(reinterpret_cast<void**>(&((*A)->Vals)), sizeof(std::complex<double>) * NNZ);
    thrust::copy(data, &data[NNZ], thrust::device_ptr<std::complex<double>>((*A)->Vals));

    thrust::device_vector<long long> rows(CsrM + 1);
    thrust::device_vector<long long> cols(NNZ);
    genCsrEntries(CsrM, thrust::raw_pointer_cast(rows.data()), thrust::raw_pointer_cast(cols.data()), (*A)->Vals, Mb, Nb, RowDims, ColDims, ARows, ACols);

    thrust::transform(rows.begin(), rows.end(), thrust::device_ptr<int>((*A)->RowOffsets), lltoi());
    thrust::transform(cols.begin(), cols.end(), thrust::device_ptr<int>((*A)->ColInd), lltoi());
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

void convertCsrEntries(int RowOffsets[], int Columns[], std::complex<double> Values[], long long Mb, long long Nb, const long long RowDims[], const long long ColDims[], const long long ARows[], const long long ACols[], const std::complex<double> data[]) {
  CsrContainer_t a;
  createDeviceCsr(&a, Mb, Nb, RowDims, ColDims, ARows, ACols, data);

  cudaMemcpy(RowOffsets, a->RowOffsets, sizeof(int) * (1 + a->M), cudaMemcpyDeviceToHost);
  cudaMemcpy(Columns, a->ColInd, sizeof(int) * a->NNZ, cudaMemcpyDeviceToHost);
  cudaMemcpy(Values, a->Vals, sizeof(std::complex<double>) * a->NNZ, cudaMemcpyDeviceToHost);

  destroyDeviceCsr(a);
}

void createSpMatrixDesc(CsrMatVecDesc_t* desc, bool is_leaf, long long lowerZ, const long long Dims[], const long long Ranks[], const std::complex<double> U[], const std::complex<double> C[], const std::complex<double> A[], const ColCommMPI& comm) {
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  long long xlen = comm.lenNeighbors();

  std::vector<long long> DimsOffsets(xlen + 1);
  std::vector<long long> RanksOffsets(xlen + 1);
  std::inclusive_scan(Dims, Dims + xlen, DimsOffsets.begin() + 1);
  std::inclusive_scan(Ranks, Ranks + xlen, RanksOffsets.begin() + 1);
  DimsOffsets[0] = RanksOffsets[0] = 0;

  desc->lenX = DimsOffsets[ibegin + nodes] - DimsOffsets[ibegin];
  desc->lenZ = RanksOffsets[ibegin + nodes] - RanksOffsets[ibegin];
  desc->xbegin = DimsOffsets[ibegin];
  desc->zbegin = RanksOffsets[ibegin];
  desc->lowerZ = lowerZ;

  long long lenX_all = DimsOffsets.back();
  long long lenZ_all = RanksOffsets.back();
  if (0 < lenX_all) {
    desc->X = reinterpret_cast<std::complex<double>*>(std::malloc(lenX_all * sizeof(std::complex<double>)));
    desc->Y = reinterpret_cast<std::complex<double>*>(std::malloc(lenX_all * sizeof(std::complex<double>)));
    std::fill(desc->X, desc->X + lenX_all, std::complex<double>(0., 0.));
    std::fill(desc->Y, desc->Y + lenX_all, std::complex<double>(0., 0.));
  }

  if (0 < lenZ_all) {
    desc->Z = reinterpret_cast<std::complex<double>*>(std::malloc(lenZ_all * sizeof(std::complex<double>)));
    desc->W = reinterpret_cast<std::complex<double>*>(std::malloc(lenZ_all * sizeof(std::complex<double>)));
    std::fill(desc->Z, desc->Z + lenZ_all, std::complex<double>(0., 0.));
    std::fill(desc->W, desc->W + lenZ_all, std::complex<double>(0., 0.));
  }

  long long nranks = comm.BoxOffsets.size();
  desc->NeighborX = reinterpret_cast<long long*>(std::malloc(nranks * sizeof(long long)));;
  desc->NeighborZ = reinterpret_cast<long long*>(std::malloc(nranks * sizeof(long long)));;
  for (long long i = 0; i < nranks; i++) {
    desc->NeighborX[i] = DimsOffsets[comm.BoxOffsets[i]];
    desc->NeighborZ[i] = RanksOffsets[comm.BoxOffsets[i]];
  }

  long long nnzU = std::transform_reduce(&Dims[ibegin], &Dims[ibegin + nodes], &Ranks[ibegin], 0ll, std::plus<long long>(), std::multiplies<long long>());
  long long nnzC = 0, nnzA = 0;
  for (long long i = 0; i < nodes; i++) {
    long long row_c = Ranks[i + ibegin];
    const long long* CCols = &comm.CColumns[comm.CRowOffsets[i]];
    const long long* CCols_end = &comm.CColumns[comm.CRowOffsets[i + 1]];
    nnzC += std::transform_reduce(CCols, CCols_end, 0ll, std::plus<long long>(), [&](long long col) { return row_c * Ranks[col]; });

    if (is_leaf) {
      long long row_a = Dims[i + ibegin];
      const long long* ACols = &comm.AColumns[comm.ARowOffsets[i]];
      const long long* ACols_end = &comm.AColumns[comm.ARowOffsets[i + 1]];
      nnzA += std::transform_reduce(ACols, ACols_end, 0ll, std::plus<long long>(), [&](long long col) { return row_a * Dims[col]; });
    }
  }

  matrix_descr type_desc;
  type_desc.type = SPARSE_MATRIX_TYPE_GENERAL;

  if (0 < nnzU) {
    desc->RowOffsetsU = reinterpret_cast<int*>(std::malloc((desc->lenX + 1) * sizeof(int)));
    desc->ColIndU = reinterpret_cast<int*>(std::malloc(nnzU * sizeof(int)));
    desc->ValuesU = reinterpret_cast<std::complex<double>*>(std::malloc(nnzU * sizeof(std::complex<double>)));

    std::vector<long long> seq(nodes + 1);
    std::iota(seq.begin(), seq.end(), 0ll);
    convertCsrEntries(desc->RowOffsetsU, desc->ColIndU, desc->ValuesU, nodes, nodes, &Dims[ibegin], &Ranks[ibegin], &seq[0], &seq[0], U);

    sparse_matrix_t descV, descU;
    mkl_sparse_z_create_csc(&descV, SPARSE_INDEX_BASE_ZERO, desc->lenZ, desc->lenX, desc->RowOffsetsU, &(desc->RowOffsetsU)[1], desc->ColIndU, desc->ValuesU);
    mkl_sparse_z_create_csr(&descU, SPARSE_INDEX_BASE_ZERO, desc->lenX, desc->lenZ, desc->RowOffsetsU, &(desc->RowOffsetsU)[1], desc->ColIndU, desc->ValuesU);
    
    mkl_sparse_set_mv_hint(descV, SPARSE_OPERATION_NON_TRANSPOSE, type_desc, hint_number);
    mkl_sparse_set_mv_hint(descU, SPARSE_OPERATION_NON_TRANSPOSE, type_desc, hint_number);
    mkl_sparse_optimize(descV);
    mkl_sparse_optimize(descU);
    desc->descV = descV;
    desc->descU = descU;
  }

  if (0 < nnzC) {
    desc->RowOffsetsC = reinterpret_cast<int*>(std::malloc((desc->lenZ + 1) * sizeof(int)));
    desc->ColIndC = reinterpret_cast<int*>(std::malloc(nnzC * sizeof(int)));
    desc->ValuesC = reinterpret_cast<std::complex<double>*>(std::malloc(nnzC * sizeof(std::complex<double>)));
    convertCsrEntries(desc->RowOffsetsC, desc->ColIndC, desc->ValuesC, nodes, xlen, &Ranks[ibegin], Ranks, comm.CRowOffsets.data(), comm.CColumns.data(), C);

    sparse_matrix_t descC;
    mkl_sparse_z_create_csr(&descC, SPARSE_INDEX_BASE_ZERO, desc->lenZ, lenZ_all, desc->RowOffsetsC, &(desc->RowOffsetsC)[1], desc->ColIndC, desc->ValuesC);
    mkl_sparse_set_mv_hint(descC, SPARSE_OPERATION_NON_TRANSPOSE, type_desc, hint_number);
    mkl_sparse_optimize(descC);
    desc->descC = descC;
  }

  if (0 < nnzA) {
    desc->RowOffsetsA = reinterpret_cast<int*>(std::malloc((desc->lenX + 1) * sizeof(int)));
    desc->ColIndA = reinterpret_cast<int*>(std::malloc(nnzA * sizeof(int)));
    desc->ValuesA = reinterpret_cast<std::complex<double>*>(std::malloc(nnzA * sizeof(std::complex<double>)));
    convertCsrEntries(desc->RowOffsetsA, desc->ColIndA, desc->ValuesA, nodes, xlen, &Dims[ibegin], Dims, comm.ARowOffsets.data(), comm.AColumns.data(), A);

    sparse_matrix_t descA;
    mkl_sparse_z_create_csr(&descA, SPARSE_INDEX_BASE_ZERO, desc->lenX, lenX_all, desc->RowOffsetsA, &(desc->RowOffsetsA)[1], desc->ColIndA, desc->ValuesA);
    mkl_sparse_set_mv_hint(descA, SPARSE_OPERATION_NON_TRANSPOSE, type_desc, hint_number);
    mkl_sparse_optimize(descA);
    desc->descA = descA;
  }
}

void destroySpMatrixDesc(CsrMatVecDesc_t desc) {
  if (desc.X) std::free(desc.X);
  if (desc.Y) std::free(desc.Y);
  if (desc.Z) std::free(desc.Z);
  if (desc.W) std::free(desc.W);
  if (desc.NeighborZ) std::free(desc.NeighborZ);

  if (desc.RowOffsetsU) std::free(desc.RowOffsetsU);
  if (desc.ColIndU) std::free(desc.ColIndU);
  if (desc.ValuesU) std::free(desc.ValuesU);

  if (desc.RowOffsetsC) std::free(desc.RowOffsetsC);
  if (desc.ColIndC) std::free(desc.ColIndC);
  if (desc.ValuesC) std::free(desc.ValuesC);

  if (desc.RowOffsetsA) std::free(desc.RowOffsetsA);
  if (desc.ColIndA) std::free(desc.ColIndA);
  if (desc.ValuesA) std::free(desc.ValuesA);

  if (desc.descV) mkl_sparse_destroy(reinterpret_cast<sparse_matrix_t>(desc.descV));
  if (desc.descU) mkl_sparse_destroy(reinterpret_cast<sparse_matrix_t>(desc.descU));
  if (desc.descC) mkl_sparse_destroy(reinterpret_cast<sparse_matrix_t>(desc.descC));
  if (desc.descA) mkl_sparse_destroy(reinterpret_cast<sparse_matrix_t>(desc.descA));
}

void matVecUpwardPass(CsrMatVecDesc_t desc, const std::complex<double>* X_in, const ColCommMPI& comm) {
  long long lenX = desc.lenX;
  long long lenZ = desc.lenZ;
  std::copy(&X_in[desc.lowerZ], &X_in[desc.lowerZ + lenX], &(desc.X)[desc.xbegin]);

  if (0 < lenZ) {
    matrix_descr type_desc;
    type_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
    sparse_matrix_t descV = reinterpret_cast<sparse_matrix_t>(desc.descV);
    mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, std::complex<double>(1., 0.), descV, type_desc, &(desc.X)[desc.xbegin], std::complex<double>(0., 0.), &(desc.Z)[desc.zbegin]);
  }

  comm.neighbor_bcast(desc.Z, &desc.NeighborZ[1]);
}

void matVecHorizontalandDownwardPass(CsrMatVecDesc_t desc, std::complex<double>* Y_out) {
  long long lenX = desc.lenX;
  long long lenZ = desc.lenZ;
  matrix_descr type_desc;
  type_desc.type = SPARSE_MATRIX_TYPE_GENERAL;

  if (0 < lenZ) {
    sparse_matrix_t descC = reinterpret_cast<sparse_matrix_t>(desc.descC);
    sparse_matrix_t descU = reinterpret_cast<sparse_matrix_t>(desc.descU);

    mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, std::complex<double>(1., 0.), descC, type_desc, desc.Z, std::complex<double>(1., 0.), &(desc.W)[desc.zbegin]);
    mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, std::complex<double>(1., 0.), descU, type_desc, &(desc.W)[desc.zbegin], std::complex<double>(0., 0.), &(desc.Y)[desc.xbegin]);
  }

  std::copy(&(desc.Y)[desc.xbegin], &(desc.Y)[desc.xbegin + lenX], &Y_out[desc.lowerZ]);
}

void matVecLeafHorizontalPass(CsrMatVecDesc_t desc, std::complex<double>* X_io, const ColCommMPI& comm) {
  long long lenX = desc.lenX;
  long long lenZ = desc.lenZ;
  std::copy(X_io, &X_io[lenX], &(desc.X)[desc.xbegin]);
  comm.neighbor_bcast(desc.X, &desc.NeighborX[1]);

  matrix_descr type_desc;
  type_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
  sparse_matrix_t descA = reinterpret_cast<sparse_matrix_t>(desc.descA);

  mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, std::complex<double>(1., 0.), descA, type_desc, desc.X, std::complex<double>(0., 0.), &(desc.Y)[desc.xbegin]);

  if (0 < lenZ) {
    sparse_matrix_t descC = reinterpret_cast<sparse_matrix_t>(desc.descC);
    sparse_matrix_t descU = reinterpret_cast<sparse_matrix_t>(desc.descU);

    mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, std::complex<double>(1., 0.), descC, type_desc, desc.Z, std::complex<double>(1., 0.), &(desc.W)[desc.zbegin]);
    mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, std::complex<double>(1., 0.), descU, type_desc, &(desc.W)[desc.zbegin], std::complex<double>(1., 0.), &(desc.Y)[desc.xbegin]);
  }

  std::copy(&(desc.Y)[desc.xbegin], &(desc.Y)[desc.xbegin + lenX], X_io);
}
