#pragma once

#include <data_container.hpp>
#include <complex>

class MatrixAccessor;
class CSR;
class Cell;
class ColCommMPI;

/*
Basically, this class is used to sample the far field of a node
in the cluster tree.
Let us take the example of a leaf level cell in an HSS cluster tree.
The far field would be of dimension leaf x N-leaf.
In order to reduce this to leaf x max_rank (or less if a desired
accuracy is already satisfied), we can sample the far field.
This sampling is done for each level seperately and lower level nodes
can reuse the samples from higher levels if necessary.
*/
class WellSeparatedApproximation {
private:
  // first index in the cell array for the current level
  long long lbegin;
  // last index in the cell array for the current level
  long long lend;
  // the sampled bodies for each cell in the level
  // local index (i.e. starts from 0 for each level)
  std::vector<std::vector<double>> M;

public:
  WellSeparatedApproximation() : lbegin(0), lend(0) {}
  /*
  In:
    kernel: kernel function
    epsilon: epsilon
    max_rank: maximum rank
    cell_begin: first index in the cell array on the current level
    ncells: number of cells on the current level
    cells: the cell array
    Far: Far field in CSR format
    bodies: the points
    upper_level: the approximation from the upper level
  */
  WellSeparatedApproximation(const MatrixAccessor& kernel, const double epsilon, const long long max_rank, const long long cell_begin, const long long ncells, const Cell cells[], const CSR& Far, const double bodies[], const WellSeparatedApproximation& upper_level);

  long long fbodies_size_at_i(long long i) const;
  const double* fbodies_at_i(long long i) const;
};

class H2Matrix {
private:
  std::vector<long long> DimsLr;
  std::vector<long long> UpperStride;
  MatrixDataContainer<std::complex<double>> Q;
  MatrixDataContainer<std::complex<double>> R;
  MatrixDataContainer<double> S;

  std::vector<long long> CRows;
  std::vector<long long> CCols;
  std::vector<std::complex<double>*> C;

  std::vector<long long> ARows;
  std::vector<long long> ACols;
  MatrixDataContainer<std::complex<double>> A;
  std::vector<std::complex<double>*> NA;
  std::vector<int> Ipivots;

  std::vector<std::complex<double>*> NX;
  std::vector<std::complex<double>*> NY;

public:
  std::vector<long long> Dims;
  MatrixDataContainer<std::complex<double>> X;
  MatrixDataContainer<std::complex<double>> Y;
  
  H2Matrix() {}
  H2Matrix(const MatrixAccessor& eval, double epi, const Cell cells[], const CSR& Near, const CSR& Far, const double bodies[], const WellSeparatedApproximation& wsa, const ColCommMPI& comm, H2Matrix& lowerA, const ColCommMPI& lowerComm, bool use_near_bodies = false);

  void matVecUpwardPass(const ColCommMPI& comm);
  void matVecHorizontalandDownwardPass(const ColCommMPI& comm);
  void matVecLeafHorizontalPass(const ColCommMPI& comm);
  void resetX();

  void factorize(const ColCommMPI& comm);
  void forwardSubstitute(const ColCommMPI& comm);
  void backwardSubstitute(const ColCommMPI& comm);
};

