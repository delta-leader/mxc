#pragma once

#include <vector>
#include <cstdint>
#include <complex>

class CSR;
class CellComm;

class BlockSparseMatrix {
private:
  std::vector<std::complex<double>> Adata;
  std::vector<int64_t> blocksOnRow;
  std::vector<int64_t> elementsOnRow;

public:
  std::vector<int64_t> M;
  std::vector<int64_t> N;
  std::vector<const std::complex<double>*> A;

  BlockSparseMatrix(const int64_t Dims[], const CSR& csr, const CellComm& comm);
};

