
#include <csr_matrix.hpp>

#include <algorithm>
#include <numeric>
#include <tuple>
#include <iostream>

CSRMatrix::CSRMatrix() : M(0), N(0), NNZ(0), handle(nullptr) {}

void CSRMatrix::construct(long long Mb, long long Nb, const long long RowDims[], const long long ColDims[], const long long ARows[], const long long ACols[], const std::complex<double>* DataPtrs[], const long long LDs[]) {

  M = std::reduce(RowDims, &RowDims[Mb]);
  N = std::reduce(ColDims, &ColDims[Nb]);
  NNZ = 0;
  
  for (long long i = 0; i < Mb; i++) {
    long long rows = RowDims[i];
    NNZ += std::transform_reduce(&ACols[ARows[i]], &ACols[ARows[i + 1]], 0ll, std::plus<long long>(), [&](long long col) { return rows * ColDims[col]; });
  }

  RowOffsets.resize(M + 1);
  Columns.resize(NNZ);
  Values.resize(NNZ);

  std::vector<std::tuple<MKL_INT, MKL_INT, std::complex<double>>> A;
  A.reserve(NNZ);

  std::vector<long long> y_offsets(Mb);
  std::vector<long long> x_offsets(Nb);
  std::exclusive_scan(RowDims, &RowDims[Mb], y_offsets.begin(), 0ll);
  std::exclusive_scan(ColDims, &ColDims[Nb], x_offsets.begin(), 0ll);

  for (long long i = 0; i < Mb; i++) {
    long long m = RowDims[i];
    long long ld = LDs == nullptr ? m : LDs[i];
    long long ybegin = y_offsets[i];

    for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
      long long j = ACols[ij];
      long long n = ColDims[j];
      long long xbegin = x_offsets[j];
      const std::complex<double>* dp = DataPtrs[ij];

      for (long long x = 0; x < n; x++)
        for (long long y = 0; y < m; y++)
          A.emplace_back(ybegin + y, xbegin + x, dp[y + x * ld]);
    }
  }

  std::sort(A.begin(), A.end(), [](const auto& a, const auto& b) { 
    return std::get<0>(a) == std::get<0>(b) ? std::get<1>(a) < std::get<1>(b) : std::get<0>(a) < std::get<0>(b); });

  RowOffsets[0] = 0;
  for (long long i = 1; i <= M; i++)
    RowOffsets[i] = std::distance(A.begin(), std::find_if(A.begin() + RowOffsets[i - 1], A.end(), [=](const auto& a) { return i <= std::get<0>(a); }));
  std::transform(A.begin(), A.end(), Columns.begin(), [](const auto& a) { return std::get<1>(a); });
  std::transform(A.begin(), A.end(), Values.begin(), [](const auto& a) { return std::get<2>(a); });

  mkl_sparse_z_create_csr(&handle, SPARSE_INDEX_BASE_ZERO, M, N, &RowOffsets[0], &RowOffsets[1], &Columns[0], &Values[0]);
  mkl_sparse_optimize(handle);
}

