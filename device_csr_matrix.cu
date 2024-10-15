
#include <factorize.cuh>
#include <comm-mpi.hpp>

#include <mkl_spblas.h>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <thrust/device_vector.h>
#include <iostream>

void convertCsrEntries(int RowOffsets[], int Columns[], std::complex<double> Values[], long long Mb, long long Nb, const long long RowDims[], const long long ColDims[], const long long ARows[], const long long ACols[], const std::complex<double> data[]) {
  long long CsrM = std::reduce(RowDims, &RowDims[Mb]);
  long long NNZ = computeCooNNZ(Mb, RowDims, ColDims, ARows, ACols);

  thrust::device_vector<long long> row(CsrM + 1), col(NNZ);
  thrust::device_vector<std::complex<double>> vals(data, &data[NNZ]);
  std::vector<long long> row_h(CsrM + 1), col_h(NNZ);
  genCsrEntries(CsrM, thrust::raw_pointer_cast(row.data()), thrust::raw_pointer_cast(col.data()), thrust::raw_pointer_cast(vals.data()), Mb, Nb, RowDims, ColDims, ARows, ACols);

  thrust::copy(row.begin(), row.end(), row_h.begin());
  thrust::copy(col.begin(), col.end(), col_h.begin());
  thrust::copy(vals.begin(), vals.end(), Values);

  std::transform(row_h.begin(), row_h.end(), RowOffsets, [](long long a) { return (int)a; });
  std::transform(col_h.begin(), col_h.end(), Columns, [](long long a) { return (int)a; });
}

/*void convertCsrEntries(int RowOffsets[], int Columns[], std::complex<double> Values[], long long Mb, long long Nb, const long long RowDims[], const long long ColDims[], const long long ARows[], const long long ACols[], const std::complex<double> data[]) {
  long long NNZ = computeCooNNZ(Mb, RowDims, ColDims, ARows, ACols);
  std::vector<std::tuple<int, int, std::complex<double>>> A;
  A.reserve(NNZ);

  std::vector<long long> y_offsets(Mb);
  std::vector<long long> x_offsets(Nb);
  std::exclusive_scan(RowDims, &RowDims[Mb], y_offsets.begin(), 0ll);
  std::exclusive_scan(ColDims, &ColDims[Nb], x_offsets.begin(), 0ll);

  for (long long i = 0; i < Mb; i++) {
    long long m = RowDims[i];
    long long ybegin = y_offsets[i];

    for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
      long long j = ACols[ij];
      long long n = ColDims[j];
      long long xbegin = x_offsets[j];

      for (long long x = 0; x < n; x++)
        for (long long y = 0; y < m; y++)
          A.emplace_back(ybegin + y, xbegin + x, std::complex<double>(0., 0.));
    }
  }

  std::transform(data, &data[NNZ], A.begin(), A.begin(), [](std::complex<double> c, const auto& a) { return std::make_tuple(std::get<0>(a), std::get<1>(a), c); });
  std::sort(A.begin(), A.end(), [](const auto& a, const auto& b) { 
    return std::get<0>(a) == std::get<0>(b) ? std::get<1>(a) < std::get<1>(b) : std::get<0>(a) < std::get<0>(b); });

  long long M = std::reduce(RowDims, &RowDims[Mb]);
  RowOffsets[0] = 0;
  for (long long i = 1; i <= M; i++)
    RowOffsets[i] = std::distance(A.begin(), std::find_if(A.begin() + RowOffsets[i - 1], A.end(), [=](const auto& a) { return i <= std::get<0>(a); }));
  std::transform(A.begin(), A.end(), Columns, [](const auto& a) { return std::get<1>(a); });
  std::transform(A.begin(), A.end(), Values, [](const auto& a) { return std::get<2>(a); });

  std::vector<int> test_r(M + 1), test_c(NNZ);
  std::vector<std::complex<double>> test_v(NNZ);
  convertCsrEntries_test(test_r.data(), test_c.data(), test_v.data(), Mb, Nb, RowDims, ColDims, ARows, ACols, data);
  
  for (long long i = 0; i <= M; i++)
    if (test_r[i] != RowOffsets[i])
      std::cout << i << ", " << test_r[i] << ", " << RowOffsets[i] << std::endl;
}*/

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
