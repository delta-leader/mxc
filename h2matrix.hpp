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


  /*
  Returns the number of sampled bodies for the cell with index idx.
  */
  long long fbodies_size_at_i(const long long i) const;
  /*
  Returns a pointer to the sampled bodies for the cell with index idx.
  */
  const double* fbodies_at_i(const long long i) const;
};

template <typename DT = std::complex<double>>
class H2Matrix {
private:
  // stores the rank for each cell
  std::vector<long long> DimsLr;
  // the dimension (Dim) of the parent of each cell
  // i.e. the number of points in the parent
  std::vector<long long> UpperStride;
  // the basis matrices
  // the basis is computed as a row ID such that A = X A(rows)
  // where X = QR
  MatrixDataContainer<DT> Q;
  // the corresponding R matrices (see above)
  MatrixDataContainer<DT> R;
  // stores the points corresponding to each cell
  MatrixDataContainer<double> S;

  // Far field rows and columns in CSR format
  std::vector<long long> CRows;
  std::vector<long long> CCols;
  // the actual far field skeleton matrices are stored in A
  // at the upper level
  // this pointer provides a convenient way of accessing them
  // from this level
  std::vector<DT*> C;

  // Near field rows and columns in CSR format
  std::vector<long long> ARows;
  std::vector<long long> ACols;
  // stores the dense matrices at the leaf level
  // at the upper levels it stores the skeleton matrices
  // TODO what about the skeleton matrices on the leaf level?
  MatrixDataContainer<DT> A;
  // Pointer to the upper level skeleton/dense matrices for the near field
  // 0 initialized as far as I can tell
  // TODO not sure how this is used?
  std::vector<DT*> NA;
  // length is equal to the total number of points at this level
  // TODO not used yet
  std::vector<int> Ipivots;

  // pointers to X of the parent
  std::vector<DT*> NX;
  // pointers ot Y of the parent
  std::vector<DT*> NY;

public:
  // the number of points contained in each cell for this level
  std::vector<long long> Dims;
  // Used for storing the input/output vector
  // and intermediate results during matrix-vector multiplication
  MatrixDataContainer<DT> X;
  MatrixDataContainer<DT> Y;
  
  H2Matrix() {}
  /*
  creates an H2 matrix for a certain level
  kernel: kernel function
  epsilon: accuracy, if fixed rank is used it contains the max_rank
  cells: the cell array (nodes in the cluster tree)
  Near: Near field in CSR format
  Far: Far field in CSR format
  bodies: the points
  wsa: the sampled far field points
  comm: MPI communicator for this level
  lowerA: the H2Matrix for the level one below the current one
  lowerComm: communicator for this level
  use_near_bodies: not exactly sure what this does, default: false
  */
  H2Matrix(const MatrixAccessor& kernel, const double epsilon, const Cell cells[], const CSR& Near, const CSR& Far, const double bodies[], const WellSeparatedApproximation& wsa, const ColCommMPI& comm, H2Matrix& h2_lower, const ColCommMPI& lowerComm, const bool use_near_bodies = false);

  /*
  multiplies the Q matrices for all cells on this level with a vector
  the vector needs to be stored in X beforehand
  the result is stored in X of the parent node
  comm: the communicator for this level
  */
  void matVecUpwardPass(const ColCommMPI& comm);
  /*
  multiplies the C matrices and the Q matrices for all cell on this 
  level with a vector (stored in X)
  the result is stored in Y
  comm: the communicator for this level
  */
  void matVecHorizontalandDownwardPass(const ColCommMPI& comm);
  /*
  multiplies the dense matrices for the leaf level
  Y and X must be calculated beforehand
  the result is stored in Y
  comm: the communicator for this level
  */
  void matVecLeafHorizontalPass(const ColCommMPI& comm);
  // initializes X and Y to zero
  void resetX();

  /*
  factorize the matrix on this level
  comm: the communicator for this level
  */
  void factorize(const ColCommMPI& comm);
  /*
  forward substitution for this level
  comm: the communicator for this level
  */
  void forwardSubstitute(const ColCommMPI& comm);
  /*
  backward substitution for this level
  comm: the communicator for this level
  */
  void backwardSubstitute(const ColCommMPI& comm);
};

