#pragma once

#include <build_tree.hpp>
#include <comm-mpi.hpp>
#include <h2matrix.hpp>
#include <kernel.hpp>

class H2MatrixSolver {
private:
  long long levels;
  std::vector<H2Matrix> A;
  std::vector<ColCommMPI> comm;
  std::vector<MPI_Comm> allocedComm;

public:
  std::pair<long long, long long> local_bodies;
  std::vector<double> resid;
  long long iters;
  
  H2MatrixSolver();
  H2MatrixSolver(const MatrixAccessor& eval, double epi, long long rank, long long leveled_rank, const std::vector<Cell>& cells, double theta, const double bodies[], long long levels, MPI_Comm world = MPI_COMM_WORLD);

  void matVecMul(std::complex<double> X[]);
  void factorizeM();
  void factorizeDeviceM(int device);
  void solvePrecondition(std::complex<double> X[]);
  void solveGMRES(double tol, H2MatrixSolver& M, std::complex<double> X[], const std::complex<double> B[], long long inner_iters, long long outer_iters);

  void free_all_comms();
  static double solveRelErr(long long lenX, const std::complex<double> X[], const std::complex<double> ref[], MPI_Comm world = MPI_COMM_WORLD);
};
