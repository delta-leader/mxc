#pragma once

#include <complex>
#include <vector>
#include <mkl_spblas.h>

class CSRMatrix {
public:
  MKL_INT M;
  MKL_INT N;
  MKL_INT NNZ;
  std::vector<MKL_INT> RowOffsets;
  std::vector<MKL_INT> Columns;
  std::vector<std::complex<double>> Values;
  sparse_matrix_t handle;

  CSRMatrix();
  void construct(long long Mb, long long Nb, const long long RowDims[], const long long ColDims[], const long long ARows[], const long long ACols[], const std::complex<double>* DataPtrs[], const long long LDs[] = nullptr);
};
