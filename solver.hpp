#pragma once

#include <vector>
#include <cstdint>
#include <complex>

class CSR;
class Cell;
class CellComm;
class ClusterBasis;
class MatrixAccessor;

class BlockSparseMatrix {
public:
  std::vector<int64_t> RowIndex;
  std::vector<int64_t> ColIndex;
  std::vector<int64_t> blocksOnRow;
  std::vector<int64_t> elementsOnRow;

  std::vector<std::complex<double>> Data;
  std::vector<int64_t> DataOffsets;
  std::vector<int64_t> M;
  std::vector<int64_t> N;
  std::vector<int64_t> RankM;
  std::vector<int64_t> RankN;

  BlockSparseMatrix() {};
  BlockSparseMatrix(int64_t len, const std::pair<int64_t, int64_t> lil[], const std::pair<int64_t, int64_t> dim[], const CellComm& comm);
  const std::complex<double>* operator[](int64_t i) const;
  const std::complex<double>* operator()(int64_t y, int64_t x) const;
  std::complex<double>* operator[](int64_t i);
  std::complex<double>* operator()(int64_t y, int64_t x);
};

class UlvSolver {
private:
  BlockSparseMatrix A;
  BlockSparseMatrix C;
  std::vector<int64_t> Ck;

public:

  UlvSolver(const int64_t Dims[], const CSR& csr, const CellComm& comm);

  void loadDataLeaf(const MatrixAccessor& eval, const Cell cells[], const double bodies[], const CellComm& comm);

  void preCompressA2(double epi, ClusterBasis& basis, const CellComm& comm);
};

