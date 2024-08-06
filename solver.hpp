#pragma once

#include <build_tree.hpp>
#include <comm-mpi.hpp>
#include <h2matrix.hpp>
#include <kernel.hpp>

template <typename DT = std::complex<double>>
class H2MatrixSolver {
private:
  // the depth of the tree
  long long levels;
  // vector of H2 matrices for each level, levels + 1
  std::vector<H2Matrix<DT>> A;
  // communicator for each level, levels + 1
  std::vector<ColCommMPI> comm;
  // empty
  std::vector<MPI_Comm> allocedComm;

public:
  // Contains the starting and ending index of the bodies
  // local for each process
  std::pair<long long, long long> local_bodies;
  std::vector<double> resid;
  long long iters;
  
  H2MatrixSolver();
  /*
  kernel: kernel function
  epsilon: accuracy
  max_rank: the maximum rank if fixed rank is used
  cells: cell array containing the nodes of the index tree
  theta: admisibility
  bodies: the points
  max_level: the maximum level
  fixed_rank: if true, use max_rank, use epsilon otherwise default; false
  world: MPI communicator, default: all processes
  */
  H2MatrixSolver(const MatrixAccessor& kernel, double epsilon, const long long max_rank, const std::vector<Cell>& cells, const double theta, const double bodies[], const long long max_level, const bool fix_rank = false, MPI_Comm world = MPI_COMM_WORLD);

  /*
  Matrix vector multiplication
  Inout:
    X: the vector to be multiplied 
       (overwritten with the result)
  */
  void matVecMul(DT X[]);
  // factorize the matrix
  void factorizeM();
  /*
  solve the system LUx = b
  after the matrix has been factorized
  Inout:
    X: the vector b
       (overwritten with x as the output)
  */
  void solvePrecondition(DT X[]);
  void solveGMRES(double tol, H2MatrixSolver& M, DT X[], const DT B[], long long inner_iters, long long outer_iters);

  void free_all_comms();
};

/*
calculates the relative error between 2 vectors
In:
  lenX: the number of rows in the vector
  X: the first vector
  ref: the second vector
  world: MPI communicator, default: all processes
Returns:
  the relative error
*/
template <typename DT>
double computeRelErr(const long long lenX, const DT X[], const DT ref[], MPI_Comm world = MPI_COMM_WORLD);
template <typename DT>
double computeRelErr(const long long lenX, const std::complex<DT> X[], const std::complex<DT> ref[], MPI_Comm world = MPI_COMM_WORLD);
