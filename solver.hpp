#pragma once

#include <build_tree.hpp>
#include <comm-mpi.hpp>
#include <h2matrix.hpp>
#include <kernel.hpp>

class H2MatrixSolver {
private:
  // the depth of the tree
  long long levels;
  // vector of H2 matrices for each level, levels + 1
  std::vector<H2Matrix> A;
  // communicator for each level, levels + 1
  std::vector<ColCommMPI> comm;
  // empty
  std::vector<MPI_Comm> allocedComm;

public:
  // (0, 0)
  std::pair<long long, long long> local_bodies;
  std::vector<double> resid;
  long long iters;
  
  H2MatrixSolver();
  /*
  eval: kernel function
  epi: epsilon, accuracy
  rank
  cells: array with the nodes of the index tree
  theta: admisibility
  bodies: the points
  levels: the depth of the tree;
  fixed_rank: if true all bases use the same rank, default; false
  world; MPI communicator, default: all processes
  */
  H2MatrixSolver(const MatrixAccessor& eval, double epi, long long rank, const std::vector<Cell>& cells, double theta, const double bodies[], long long levels, bool fix_rank = false, MPI_Comm world = MPI_COMM_WORLD);

  void matVecMul(std::complex<double> X[]);
  void factorizeM();
  void solvePrecondition(std::complex<double> X[]);
  void solveGMRES(double tol, H2MatrixSolver& M, std::complex<double> X[], const std::complex<double> B[], long long inner_iters, long long outer_iters);

  void free_all_comms();
  static double solveRelErr(long long lenX, const std::complex<double> X[], const std::complex<double> ref[], MPI_Comm world = MPI_COMM_WORLD);
};
