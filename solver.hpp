#pragma once

#include <build_tree.hpp>
#include <comm-mpi.hpp>
#include <h2matrix.hpp>
#include <kernel.hpp>

class H2MatrixSolver {
private:
  long long levels;
  std::vector<H2Matrix> A;
  std::vector<H2Matrix> M;
  std::vector<ColCommMPI> comm;
  std::vector<MPI_Comm> allocedComm;

public:
  std::pair<double, double> timer;
  std::pair<long long, long long> local_bodies;
  
  H2MatrixSolver(const MatrixAccessor& eval, double epi, long long rank, const Cell cells[], long long ncells, const CSR& Near, const CSR& Far, const double bodies[], long long levels, MPI_Comm world = MPI_COMM_WORLD);

  void matVecMul(std::complex<double> X[]);
  void solvePrecondition(std::complex<double> X[]);
  std::pair<double, long long> solveGMRES(double tol, std::complex<double> X[], const std::complex<double> B[], long long inner_iters, long long outer_iters);
  
  void free_all_comms();
};
