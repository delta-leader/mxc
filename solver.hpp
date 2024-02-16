#pragma once

#include <vector>
#include <cstdint>
#include <complex>

class CSR;
class Cell;
class CellComm;
class ClusterBasis;
class MatrixAccessor;

class UlvSolver {
private:
  std::vector<std::complex<double>> Adata;
  std::vector<std::complex<double>> Cdata;

  std::vector<int64_t> blocksOnRow;
  std::vector<int64_t> elementsOnRow;

  std::vector<int64_t> CM;
  std::vector<int64_t> CN;
  std::vector<int64_t> CRankM;
  std::vector<int64_t> CRankN;
  std::vector<int64_t> CRows;
  std::vector<int64_t> CCols;
  std::vector<std::complex<double>*> C;

public:
  std::vector<int64_t> M;
  std::vector<int64_t> N;
  std::vector<int64_t> RankM;
  std::vector<int64_t> RankN;
  std::vector<int64_t> ARows;
  std::vector<int64_t> ACols;
  std::vector<const std::complex<double>*> A;

  UlvSolver(const int64_t Dims[], const CSR& csr, const CellComm& comm);

  void loadDataLeaf(const MatrixAccessor& eval, const Cell cells[], const double bodies[], const CellComm& comm);

  void preCompressA2(ClusterBasis& basis, const CellComm& comm);
};

