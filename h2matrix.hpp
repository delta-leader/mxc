#pragma once

#include <vector>
#include <numeric>
#include <complex>

template <typename DT>
class MatrixAccessor;
class CSR;
class Cell;
class ColCommMPI;
class MatrixDesc;

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
template <typename DT = std::complex<double>>
class WellSeparatedApproximation {
private:
  // first index in the cell array for the current level
  long long lbegin = 0;
  // last index in the cell array for the current level
  long long lend = 0;
  // the sampled bodies for each cell in the level
  // local index (i.e. starts from 0 for each level)
  std::vector<std::vector<double>> M;

public:
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
  void construct(const MatrixAccessor<DT>& kernel, const double epsilon, long long max_rank, const long long cell_begin, const long long ncells, const Cell cells[], const CSR& Far, const double bodies[], const WellSeparatedApproximation& upper_level, const bool fix_rank);

  /*
  Returns the number of sampled bodies for the cell with index idx.
  */
  long long fbodies_size_at_i(const long long i) const;
  /*
  Returns a pointer to the sampled bodies for the cell with index idx.
  */
  const double* fbodies_at_i(const long long i) const;
};

template<class T> class MatrixDataContainer {
private:
  std::vector<long long> offsets;
  // TODO isn't that data never freed?
  T* data = nullptr;

public:
  template <class U> friend class MatrixDataContainer;
  MatrixDataContainer() = default;
  template <class U>
  MatrixDataContainer(const MatrixDataContainer<U>& container);

  void alloc(long long len, const long long* dims);
  T* operator[](long long index);
  const T* operator[](long long index) const;
  long long size() const;
  void reset();
};

template <typename DT = std::complex<double>>
class H2Matrix {
private:
  // the dimension (Dim) of the parent of each cell
  // i.e. the number of points in the parent
  std::vector<long long> UpperStride;
  // stores the points corresponding to each cell
  MatrixDataContainer<double> S;

  // Far field rows and columns in CSR format
  std::vector<long long> CRows;
  std::vector<long long> CCols;
  // the actual far field skeleton matrices are stored in A
  // at the upper level
  // this pointer provides a convenient way of accessing them
  // from this level
  std::vector<long long> C;

  // pointer to the uper level skeleton matrices
  std::vector<long long> NA;
  std::vector<long long> LowerX;
  std::vector<long long> NbXoffsets;

public:
  template <typename OT> friend class H2Matrix;
  template <typename OT> friend class H2MatrixSolver;
  // the number of points contained in each cell for this level
  std::vector<long long> Dims;
  // stores the rank for each cell
  std::vector<long long> DimsLr;

  // Near field rows and columns in CSR format
  std::vector<long long> ARows;
  std::vector<long long> ACols;
  // the basis matrices
  // the basis is computed as a row ID such that A = X A(rows)
  // where X = QR
  MatrixDataContainer<DT> Q;
  // the corresponding R matrices (see above)
  MatrixDataContainer<DT> R;
  MatrixDataContainer<DT> A;

  MatrixDataContainer<DT> X;
  MatrixDataContainer<DT> Y;

  H2Matrix() = default;
  
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
  void construct(const MatrixAccessor<DT>& kernel, const double epsilon, const Cell cells[], const CSR& Near, const CSR& Far, const double bodies[], const WellSeparatedApproximation<DT>& wsa, const ColCommMPI& comm, H2Matrix& h2_lower, const ColCommMPI& lowerComm, const bool use_near_bodies = false);

  /*
  Copy constructor to convert a H2matrix to a different datatype
  */
  template <typename OT>
  H2Matrix(const H2Matrix<OT>& h2matrix);
  
  void upwardCopyNext(char src, char dst, const ColCommMPI& comm, const H2Matrix& lowerA);
  void downwardCopyNext(char src, char dst, const H2Matrix& upperA, const ColCommMPI& upperComm);

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
  void matVecHorizontalandDownwardPass(const H2Matrix& upperA, const ColCommMPI& comm);
  /*
  multiplies the dense matrices for the leaf level
  Y and X must be calculated beforehand
  the result is stored in Y
  comm: the communicator for this level
  */
  void matVecLeafHorizontalPass(const ColCommMPI& comm);

  /*
  factorize the matrix on this level
  comm: the communicator for this level
  */
  void factorize(const ColCommMPI& comm);
  //void factorize(const ColCommMPI& comm, const cublasComputeType_t COMP);
  void factorizeCopyNext(const ColCommMPI& comm, const H2Matrix& lowerA, const ColCommMPI& lowerComm);
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
