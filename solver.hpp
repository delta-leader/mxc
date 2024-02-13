#pragma once

#include <vector>
#include <cstdint>
#include <complex>

class CSR;
class CellComm;

class BlockSparseMatrix {
private:
  std::vector<std::complex<double>> Adata;
  std::vector<std::complex<double>> Fdata;
  std::vector<int64_t> blocksOnRow;
  std::vector<int64_t> elementsOnRow;

public:
  std::vector<int64_t> M;
  std::vector<int64_t> N;
  std::vector<const std::complex<double>*> A;
  std::vector<int64_t> ARows;
  std::vector<int64_t> ACols;

  std::vector<int64_t> FM;
  std::vector<int64_t> FN;
  std::vector<const std::complex<double>*> F;
  std::vector<int64_t> FRows;
  std::vector<int64_t> FCols;

  BlockSparseMatrix(const int64_t Dims[], const CSR& csr, const CellComm& comm);
};

