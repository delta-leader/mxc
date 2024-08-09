#pragma once

#include <Eigen/Dense>

#include <build_tree.hpp>
#include <comm-mpi.hpp>
#include <h2matrix.hpp>
#include <kernel.hpp>

#include <iostream>

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
  template <typename OT> friend class H2MatrixSolver;
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
  H2MatrixSolver(const MatrixAccessor<DT>& kernel, double epsilon, const long long max_rank, const std::vector<Cell>& cells, const double theta, const double bodies[], const long long max_level, const bool fix_rank = false, const bool factorization_basis = false, MPI_Comm world = MPI_COMM_WORLD);

  /* creates an exact copy in a different datatype
  In:
    solver: the solver to copy
  */
  template <typename OT>
  H2MatrixSolver(const H2MatrixSolver<OT>& solver);

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
  void solveGMRES(double tol, H2MatrixSolver& M, DT X[], const DT B[], long long inner_iters, long long outer_iters=1);
  long long solveMyGMRES(double tol, H2MatrixSolver& M, DT X[], const DT B[], long long inner_iters, long long outer_iters=1);

  void free_all_comms();

  template <typename OT>
  long long solveIR(double tol, H2MatrixSolver<OT>& M, DT X[], const DT B[], long long max_iters) {
    typedef Eigen::Matrix<DT, Eigen::Dynamic, 1> Vector_dt;
    typedef Eigen::Matrix<OT, Eigen::Dynamic, 1> Vector_ot;
    
    long long lbegin = comm[levels].oLocal();
    long long llen = comm[levels].lenLocal();
    long long N = std::reduce(A[levels].Dims.begin() + lbegin, A[levels].Dims.begin() + (lbegin + llen));
    resid = std::vector<double>(max_iters + 1, 0.);

    Eigen::Map<const Vector_dt> b(B, N);
    Eigen::Map<Vector_dt> x(X, N);
    Vector_ot x_ot = b.template cast<OT>();
    M.solvePrecondition(x_ot.data());
    x = x_ot.template cast<DT>();
    Vector_dt r;

    DT norm_local = b.squaredNorm();
    comm[levels].level_sum(&norm_local, 1);
    double norm, normb = std::sqrt(std::real(norm_local));
    if (normb == 0.)
      normb = 1.;

    for (long long iter = 0; iter<max_iters; ++iter) {
      r = -x;
      matVecMul(r.data());
      r += b;
      norm_local = r.squaredNorm();
      comm[levels].level_sum(&norm_local, 1);
      norm = std::sqrt(std::real(norm_local));
      resid[iter] = norm / normb;
      if (resid[iter]<tol) {
        return iter;
      }
      x_ot = r.template cast<OT>();
      M.solvePrecondition(x_ot.data());
      x = x + x_ot.template cast<DT>();    
    }
    r = -x;
    matVecMul(r.data());
    r += b;
    norm_local = r.squaredNorm();
    comm[levels].level_sum(&norm_local, 1);
    norm = std::sqrt(std::real(norm_local));
    resid[max_iters] = norm / normb;
    return max_iters;
  }

  long long solveIR(double tol, H2MatrixSolver<DT>& M, DT X[], const DT B[], long long max_iters) {
    typedef Eigen::Matrix<DT, Eigen::Dynamic, 1> Vector_dt;
    
    long long lbegin = comm[levels].oLocal();
    long long llen = comm[levels].lenLocal();
    long long N = std::reduce(A[levels].Dims.begin() + lbegin, A[levels].Dims.begin() + (lbegin + llen));
    resid = std::vector<double>(max_iters + 1, 0.);

    Eigen::Map<const Vector_dt> b(B, N);
    Eigen::Map<Vector_dt> x(X, N);
    x = b;
    M.solvePrecondition(x.data());
    Vector_dt r;

    DT norm_local = b.squaredNorm();
    comm[levels].level_sum(&norm_local, 1);
    double norm, normb = std::sqrt(std::real(norm_local));
    if (normb == 0.)
      normb = 1.;

    for (long long iter = 0; iter<max_iters; ++iter) {
      r = -x;
      matVecMul(r.data());
      r += b;
      norm_local = r.squaredNorm();
      comm[levels].level_sum(&norm_local, 1);
      norm = std::sqrt(std::real(norm_local));
      resid[iter] = norm / normb;
      if (resid[iter]<tol) {
        return iter;
      }
      M.solvePrecondition(r.data());
      x += r;    
    }
    r = -x;
    matVecMul(r.data());
    r += b;
    norm_local = r.squaredNorm();
    comm[levels].level_sum(&norm_local, 1);
    norm = std::sqrt(std::real(norm_local));
    resid[max_iters] = norm / normb;
    return max_iters;
  }

  long long solveGMRESIR(double tol, H2MatrixSolver<DT>& M, DT X[], const DT B[], long long max_iters, long long inner_iters, long long outer_iters=1) {
    typedef Eigen::Matrix<DT, Eigen::Dynamic, 1> Vector_dt;
    
    long long lbegin = comm[levels].oLocal();
    long long llen = comm[levels].lenLocal();
    long long N = std::reduce(A[levels].Dims.begin() + lbegin, A[levels].Dims.begin() + (lbegin + llen));
    resid = std::vector<double>(max_iters + 1, 0.);

    Eigen::Map<const Vector_dt> b(B, N);
    Eigen::Map<Vector_dt> x(X, N);
    x = b;
    M.solvePrecondition(x.data());
    Vector_dt r;
    Vector_dt d = Vector_dt::Zero(N);

    DT norm_local = b.squaredNorm();
    comm[levels].level_sum(&norm_local, 1);
    double norm, normb = std::sqrt(std::real(norm_local));
    if (normb == 0.)
      normb = 1.;

    for (long long iter = 0; iter<max_iters; ++iter) {
      r = -x;
      matVecMul(r.data());
      r += b;
      norm_local = r.squaredNorm();
      comm[levels].level_sum(&norm_local, 1);
      norm = std::sqrt(std::real(norm_local));
      resid[iter] = norm / normb;
      if (resid[iter]<tol) {
        return iter;
      }
      // solve with gmres for x=d and r=b
      solveMyGMRES(1e-6, M, d.data(), r.data(), inner_iters, outer_iters);
      x += d;
    }
    r = x;
    matVecMul(r.data());
    r -= b;
    norm_local = r.squaredNorm();
    comm[levels].level_sum(&norm_local, 1);
    norm = std::sqrt(std::real(norm_local));
    resid[max_iters] = norm / normb;
    return max_iters;
  }
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
