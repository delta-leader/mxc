
#include <solver.hpp>
#include <build_tree.hpp>
#include <comm-mpi.hpp>
#include <kernel.hpp>
#include <h2matrix.hpp>

#include <mkl.h>
#include <eigen3/Eigen/Dense>
#include <algorithm>
#include <cmath>

H2MatrixSolver::H2MatrixSolver(H2Matrix A[], const ColCommMPI comm[], long long levels) :
  levels(levels), A(A), Comm(comm) {
}

void H2MatrixSolver::matVecMul(std::complex<double> X[]) {
  typedef Eigen::Map<Eigen::VectorXcd> Vector_t;
  long long lbegin = Comm[levels].oLocal();
  long long llen = Comm[levels].lenLocal();
  long long lenX = std::reduce(A[levels].Dims.begin() + lbegin, A[levels].Dims.begin() + (lbegin + llen));
  
  Vector_t X_in(X, lenX);
  Vector_t X_leaf(A[levels].X[lbegin], lenX);
  Vector_t Y_leaf(A[levels].Y[lbegin], lenX);

  for (long long l = levels; l >= 0; l--)
    A[l].resetX();
  X_leaf = X_in;

  for (long long l = levels; l >= 0; l--)
    A[l].matVecUpwardPass(Comm[l]);
  for (long long l = 1; l <= levels; l++)
    A[l].matVecHorizontalandDownwardPass(Comm[l]);
  A[levels].matVecLeafHorizontalPass(Comm[levels]);
  X_in = Y_leaf;
}

void H2MatrixSolver::solvePrecondition(std::complex<double>[]) {
  // Default preconditioner = I
}

std::pair<double, long long> H2MatrixSolver::solveGMRES(double tol, std::complex<double> x[], const std::complex<double> b[], long long inner_iters, long long outer_iters) {
  using Eigen::VectorXcd, Eigen::MatrixXcd;

  long long lbegin = Comm[levels].oLocal();
  long long llen = Comm[levels].lenLocal();
  long long N = std::reduce(A[levels].Dims.begin() + lbegin, A[levels].Dims.begin() + (lbegin + llen));
  long long ld = inner_iters + 1;

  Eigen::Map<const Eigen::VectorXcd> B(b, N);
  Eigen::Map<Eigen::VectorXcd> X(x, N);
  VectorXcd R = B;
  solvePrecondition(R.data());

  std::complex<double> normr = R.adjoint() * R;
  Comm[levels].level_sum(&normr, 1);
  double normb = std::sqrt(normr.real());
  if (normb == 0.)
    normb = 1.;

  for (long long j = 0; j < outer_iters; j++) {
    R = -X;
    matVecMul(R.data());
    R += B;
    solvePrecondition(R.data());

    normr = R.adjoint() * R;
    Comm[levels].level_sum(&normr, 1);
    double beta = std::sqrt(normr.real());
    double resid = beta / normb;
    if (resid < tol)
      return std::make_pair(resid, j);

    MatrixXcd H = MatrixXcd::Zero(ld, inner_iters);
    MatrixXcd v = MatrixXcd::Zero(N, ld);
    v.col(0) = R * (1. / beta);
    
    for (long long i = 0; i < inner_iters; i++) {
      VectorXcd w = v.col(i);
      matVecMul(w.data());
      solvePrecondition(w.data());

      for (long long k = 0; k <= i; k++)
        H(k, i) = v.col(k).adjoint() * w;
      Comm[levels].level_sum(H.col(i).data(), i + 1);

      for (long long k = 0; k <= i; k++)
        w -= H(k, i) * v.col(k);

      std::complex<double> normw = w.adjoint() * w;
      Comm[levels].level_sum(&normw, 1);
      H(i + 1, i) = std::sqrt(normw.real());
      v.col(i + 1) = w * (1. / H(i + 1, i));
    }

    VectorXcd s = VectorXcd::Zero(ld);
    s(0) = beta;

    VectorXcd y = H.colPivHouseholderQr().solve(s);
    X += v.leftCols(inner_iters) * y;
  }

  R = -X;
  matVecMul(R.data());
  R += B;
  solvePrecondition(R.data());

  normr = R.adjoint() * R;
  Comm[levels].level_sum(&normr, 1);
  double beta = std::sqrt(normr.real());
  double resid = beta / normb;
  return std::make_pair(resid, outer_iters);
}