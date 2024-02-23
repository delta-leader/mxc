#pragma once

#include <vector>
#include <array>
#include <complex>

class CSR;
class Cell;
class CellComm;
class ClusterBasis;
class MatrixAccessor;

class BlockSparseMatrix {
public:
  std::vector<long long> RowIndex;
  std::vector<long long> ColIndex;
  std::vector<long long> ColIndexLocal;
  std::vector<long long> blocksOnRow;
  std::vector<long long> elementsOnRow;

  std::vector<std::complex<double>> Data;
  std::vector<long long> DataOffsets;
  std::vector<std::pair<long long, long long>> Dims;
  std::vector<std::array<long long, 4>> DimsLr;

  BlockSparseMatrix() {};
  BlockSparseMatrix(long long len, const std::pair<long long, long long> lil[], const std::pair<long long, long long> dim[], const CellComm& comm);

  const std::complex<double>* operator[](long long i) const;
  std::complex<double>* operator[](long long i);
  long long operator()(long long y, long long x) const;
};

class UlvSolver {
private:
  BlockSparseMatrix A;
  BlockSparseMatrix C;
  std::vector<long long> Ck;
  std::vector<long long> Ad;
  std::vector<std::pair<long long, long long>> ALocalCol;
  std::vector<std::vector<long long>> ALocalElements;
  std::vector<std::vector<long long>> Apiv;

public:
  UlvSolver() {};
  UlvSolver(const long long Dims[], const CSR& Near, const CSR& Far, const CellComm& comm);

  void loadDataLeaf(const MatrixAccessor& eval, const Cell cells[], const double bodies[], const CellComm& comm);
  void loadDataInterNode(const Cell cells[], const UlvSolver& prev_matrix, const CellComm& prev_comm, const CellComm& comm);
  void preCompressA2(double epi, ClusterBasis& basis, const CellComm& comm);
  void factorizeA(const ClusterBasis& basis, const CellComm& comm);
  void forwardSubstitute(long long nrhs, std::complex<double> X[], std::complex<double> Y[], const ClusterBasis& basis, const CellComm& comm) const;
  void backwardSubstitute(long long nrhs, const std::complex<double> Y[], std::complex<double> Z[], const ClusterBasis& basis, const CellComm& comm) const;
};

