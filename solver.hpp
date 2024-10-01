#pragma once

#include <complex>
#include <mkl.h>
#include <Eigen/Dense>
#include <cublas_v2.h>

#include <build_tree.hpp>
#include <comm-mpi.hpp>
#include <h2matrix.hpp>
#include <kernel.hpp>
#include <utils.hpp>

#include <iostream>

// GMRES ported from MixFits, not sure, if this works for multiple processes
template <typename T>
std::vector<T> rotmat(T a, T b) {
  std::vector<T> result(2);
  if (!b){
    result[0] = 1;
    result[1] = 0;
    return result;
  }
  if (std::abs(b) > a){
    T temp = a / b;
    result[1] = 1.0/std::sqrt(1 + temp*temp);
    result [0] = temp * result[1];
    return result;
  }
  T temp = b / a;
  result[0] = 1.0/std::sqrt(1 + temp*temp);
  result [1] = temp * result[0];
  return result;
}

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
  H2MatrixSolver(const MatrixAccessor<DT>& kernel, double epsilon, const long long rank, const long long leveled_rank, const std::vector<Cell>& cells, const double theta, const double bodies[], const long long max_level, const bool fix_rank = false, const bool factorization_basis = false, MPI_Comm world = MPI_COMM_WORLD);

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
  //void factorizeM(const cublasComputeType_t COMP);
  void factorizeDeviceM(int device);
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
    Vector_dt test = b;
    Vector_ot x_ot = test.template cast<OT>();
    M.solvePrecondition(x_ot.data());
    x = x_ot.template cast<DT>();
    Vector_dt r;

    DT norm_local = b.squaredNorm();
    comm[levels].level_sum(&norm_local, 1);
    double norm, normb = std::sqrt(get_real(norm_local));
    if (normb == 0.)
      normb = 1.;

    for (long long iter = 0; iter<max_iters; ++iter) {
      r = -x;
      matVecMul(r.data());
      r += b;
      norm_local = r.squaredNorm();
      comm[levels].level_sum(&norm_local, 1);
      norm = std::sqrt(get_real(norm_local));
      resid[iter] = norm / normb;
      if (iter && resid[iter] > resid[iter-1]) {
        //std::cout << "Divergence detected (" << resid[iter-1] << " followed by " << resid[iter] << ")" << std::endl;
        return iter-1;
      }
      if (resid[iter]<tol) {
        return iter;
      }
      test = r;
      x_ot = test.template cast<OT>();
      M.solvePrecondition(x_ot.data());
      x = x + x_ot.template cast<DT>();    
    }
    r = -x;
    matVecMul(r.data());
    r += b;
    norm_local = r.squaredNorm();
    comm[levels].level_sum(&norm_local, 1);
    norm = std::sqrt(get_real(norm_local));
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
    double norm, normb = std::sqrt(get_real(norm_local));
    if (normb == 0.)
      normb = 1.;

    for (long long iter = 0; iter<max_iters; ++iter) {
      r = -x;
      matVecMul(r.data());
      r += b;
      norm_local = r.squaredNorm();
      comm[levels].level_sum(&norm_local, 1);
      norm = std::sqrt(get_real(norm_local));
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
    norm = std::sqrt(get_real(norm_local));
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
    double norm, normb = std::sqrt(get_real(norm_local));
    if (normb == 0.)
      normb = 1.;

    for (long long iter = 0; iter<max_iters; ++iter) {
      r = -x;
      matVecMul(r.data());
      r += b;
      norm_local = r.squaredNorm();
      comm[levels].level_sum(&norm_local, 1);
      norm = std::sqrt(get_real(norm_local));
      resid[iter] = norm / normb;
      if (resid[iter]<tol) {
        return iter;
      }
      // solve with gmres for x=d and r=b
      //long long gmres_iters = solveMyGMRES(1e-6, M, d.data(), r.data(), inner_iters, outer_iters);
      long long gmres_iters = GMRES_no_restart(1e-6, M, d.data(), r.data(), inner_iters);
      std::cout<<"Gmres iterations: "<< gmres_iters<<std::endl;
      x += d;
    }
    r = x;
    matVecMul(r.data());
    r -= b;
    norm_local = r.squaredNorm();
    comm[levels].level_sum(&norm_local, 1);
    norm = std::sqrt(get_real(norm_local));
    resid[max_iters] = norm / normb;
    return max_iters;
  }

  // GMRES ported from MixFits, not sure, if this really works for multiple processes
  long long GMRES_no_restart(double tol, H2MatrixSolver& M, DT X[], const DT B[], long long max_iters) {
    typedef Eigen::Matrix<DT, Eigen::Dynamic, 1> Vector_dt;
    typedef Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic> Matrix_dt;

    long long lbegin = comm[levels].oLocal();
    long long llen = comm[levels].lenLocal();
    long long N = std::reduce(A[levels].Dims.begin() + lbegin, A[levels].Dims.begin() + (lbegin + llen));
    long long ld = max_iters + 1;
  
    Matrix_dt Q(N, ld);
    Eigen::Map<const Vector_dt> b(B, N);
    Eigen::Map<Vector_dt> x(X, N);

    Q.col(0) = b;
    Matrix_dt H(ld, max_iters);
    std::vector<DT> cs(max_iters);
    std::vector<DT> sn(max_iters);
    Vector_dt s(ld);
    Vector_dt w(N);

    DT norm_local = b.squaredNorm();
    comm[levels].level_sum(&norm_local, 1);
    double norm_r, norm_b = std::sqrt(get_real(norm_local));
  
    // calculate residual
    Q.col(0) = -x;
    matVecMul(Q.col(0).data());
    Q.col(0) += b;
    M.solvePrecondition(Q.col(0).data());
    norm_local = Q.col(0).squaredNorm();
    comm[levels].level_sum(&norm_local, 1);
    norm_r = std::sqrt(get_real(norm_local));
    s(0) = norm_r;
    // basically switch to preconditioner only if possible
    if (norm_r / norm_b <= tol){
      x = Q.col(0);
      return 0;
    }
    Q.col(0) = Q.col(0) * 1 /  norm_r;

    for (int iter=0; iter<max_iters; ++iter) {
      w = Q.col(iter);
      matVecMul(w.data());
      M.solvePrecondition(w.data());

      // ortohogonalize
      for (int i=0; i<=iter; ++i){
        H(i, iter) = w.dot(Q.col(i));
        w = w - Q.col(i) * H(i,iter);
      }

      norm_local = w.squaredNorm();
      comm[levels].level_sum(&norm_local, 1);
      H(iter+1,iter) = std::sqrt(get_real(norm_local));
      w = w * 1 / H(iter+1,iter);
      Q.col(iter+1) = w;

      // apply Givens rotation
      for (int64_t i=0; i<iter; ++i){
        DT temp = cs[i]*H(i,iter) + sn[i]*H(i+1,iter);
        H(i+1,iter) = -sn[i]*H(i,iter) + cs[i]*H(i+1,iter);
        H(i,iter) = temp;
      }

      //form rotation matrix
      std::vector<DT> rotation = rotmat(H(iter,iter), H(iter+1, iter));
      cs[iter]= rotation[0];
      sn[iter] = rotation[1];
      DT temp = cs[iter] * s(iter);
      s(iter+1) = -sn[iter] * s(iter);
      s(iter) = temp;

      H(iter, iter) = cs[iter] * H(iter,iter) + sn[iter] * H(iter+1, iter);
      H(iter+1,iter) = 0;
      double error = std::abs(s(iter+1)) / norm_r;
      
      if (error < tol){
        H.topLeftCorner(iter+1, iter+1).template triangularView<Eigen::Upper>().solveInPlace(s.topRows(iter+1));
        x = Q.topLeftCorner(N, iter+1) * s.topRows(iter+1);      
        return iter + 1;
      }
    }
    
    H.topLeftCorner(max_iters, max_iters).template triangularView<Eigen::Upper>().solveInPlace(s.topRows(max_iters));
    x = Q.topLeftCorner(N, max_iters) * s.topRows(max_iters);
    return max_iters + 1;
  }

    long long GMRES_no_restart_direct(double tol, H2MatrixSolver& M, DT X[], const DT B[], long long max_iters) {
    typedef Eigen::Matrix<DT, Eigen::Dynamic, 1> Vector_dt;
    typedef Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic> Matrix_dt;

    long long lbegin = comm[levels].oLocal();
    long long llen = comm[levels].lenLocal();
    long long N = std::reduce(A[levels].Dims.begin() + lbegin, A[levels].Dims.begin() + (lbegin + llen));
    long long ld = max_iters + 1;
  
    Matrix_dt Q(N, ld);
    Eigen::Map<const Vector_dt> b(B, N);
    Eigen::Map<Vector_dt> x(X, N);
    x = Vector_dt::Zero(N);
    resid = std::vector<double>(max_iters);

    Q.col(0) = b;
    Matrix_dt H(ld, max_iters);
    std::vector<DT> cs(max_iters);
    std::vector<DT> sn(max_iters);
    Vector_dt s(ld);
    Vector_dt w(N);

    DT norm_local = b.squaredNorm();
    comm[levels].level_sum(&norm_local, 1);
    double norm_r, norm_b = std::sqrt(get_real(norm_local));
  
    // calculate residual
    Q.col(0) = -x;
    matVecMul(Q.col(0).data());
    Q.col(0) += b;
    M.solvePrecondition(Q.col(0).data());
    norm_local = Q.col(0).squaredNorm();
    comm[levels].level_sum(&norm_local, 1);
    norm_r = std::sqrt(get_real(norm_local));
    s(0) = norm_r;
    // basically switch to preconditioner only if possible
    if (norm_r / norm_b <= tol){
      x = Q.col(0);
      return 0;
    }
    Q.col(0) = Q.col(0) * 1 /  norm_r;

    for (iters=0; iters<max_iters; ++iters) {
      w = Q.col(iters);
      matVecMul(w.data());
      M.solvePrecondition(w.data());

      // ortohogonalize
      for (int i=0; i<=iters; ++i){
        H(i, iters) = w.dot(Q.col(i));
        w = w - Q.col(i) * H(i,iters);
      }

      norm_local = w.squaredNorm();
      comm[levels].level_sum(&norm_local, 1);
      H(iters+1,iters) = std::sqrt(get_real(norm_local));
      w = w * 1 / H(iters+1,iters);
      Q.col(iters+1) = w;

      // apply Givens rotation
      for (int64_t i=0; i<iters; ++i){
        DT temp = cs[i]*H(i,iters) + sn[i]*H(i+1,iters);
        H(i+1,iters) = -sn[i]*H(i,iters) + cs[i]*H(i+1,iters);
        H(i,iters) = temp;
      }

      //form rotation matrix
      std::vector<DT> rotation = rotmat(H(iters,iters), H(iters+1, iters));
      cs[iters]= rotation[0];
      sn[iters] = rotation[1];
      DT temp = cs[iters] * s(iters);
      s(iters+1) = -sn[iters] * s(iters);
      s(iters) = temp;

      H(iters, iters) = cs[iters] * H(iters,iters) + sn[iters] * H(iters+1, iters);
      H(iters+1,iters) = 0;
      double error = std::abs(s(iters+1)) / norm_r;
      resid[iters] = error;
      
      if (error < tol){
        H.topLeftCorner(iters+1, iters+1).template triangularView<Eigen::Upper>().solveInPlace(s.topRows(iters+1));
        x = Q.topLeftCorner(N, iters+1) * s.topRows(iters+1);      
        return iters + 1;
      }
    }
    
    H.topLeftCorner(max_iters, max_iters).template triangularView<Eigen::Upper>().solveInPlace(s.topRows(max_iters));
    x = Q.topLeftCorner(N, max_iters) * s.topRows(max_iters);
    return max_iters + 1;
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
