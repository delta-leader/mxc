#pragma once

#include <build_tree.hpp>
#include <comm-mpi.hpp>
#include <h2matrix.hpp>
#include <h-matrix.hpp>
#include <kernel.hpp>
#include <Eigen/Dense>
#include <string>

template <typename DT>
class H2MatrixSolver {
public:
  long long levels;
  std::vector<H2Matrix<DT>> A;
  std::vector<ColCommMPI> comm;
  std::vector<MPI_Comm> allocedComm;

  std::pair<long long, long long> local_bodies;
  std::vector<double> resid;
  long long iters;
  
  H2MatrixSolver();
  H2MatrixSolver(const MatrixGenerator& matgen, double epi, long long rank, long long leveled_rank, const std::vector<Cell>& cells, double theta, long long levels, double omega, std::string filename = "", bool verbose = false, MPI_Comm world = MPI_COMM_WORLD);
  H2MatrixSolver(const MatrixGenerator& matgen, double epi, long long rank, long long leveled_rank, const std::vector<Cell>& cells, double theta, long long levels, double omega, const std::vector<elastWave3d::element>& elems, long long s1, long long s2, bool verbose = false, MPI_Comm world = MPI_COMM_WORLD);
  H2MatrixSolver(const H2MatrixSolver& solver);

  void matVecMul(DT X[]);
  void factorizeM();
  void solvePrecondition(DT X[]);
  void solveGMRES(double tol, H2MatrixSolver& M, DT X[], const DT B[], long long inner_iters, long long outer_iters);
  void solveGMRES(double tol, DT X[], const DT B[], long long inner_iters, long long outer_iters);
  template <typename OT>
  void solveGMRES(double tol, H2MatrixSolver<OT>& M, DT X[], const DT B[], long long inner_iters, long long outer_iters);

  void free_all_comms();
};

template<typename DT>
double solveRelErr(long long lenX, const DT X[], const DT ref[], MPI_Comm world = MPI_COMM_WORLD);
