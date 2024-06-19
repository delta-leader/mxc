
#include <solver.hpp>
#include <build_tree.hpp>
#include <comm-mpi.hpp>
#include <kernel.hpp>
#include <h2matrix.hpp>

#include <mkl.h>
#include <eigen3/Eigen/Dense>
#include <algorithm>
#include <numeric>
#include <cmath>

H2MatrixSolver::H2MatrixSolver(const H2Matrix A[], const Cell cells[], const ColCommMPI comm[], long long levels) :
  levels(levels), offsets(levels + 1), upperIndex(levels + 1), upperOffsets(levels + 1), A(A), Comm(comm) {
  
  for (long long l = levels; l >= 0; l--) {
    long long xlen = comm[l].lenNeighbors();
    offsets[l] = std::vector<long long>(xlen + 1, 0);
    upperIndex[l] = std::vector<long long>(xlen, 0);
    upperOffsets[l] = std::vector<long long>(xlen, 0);
    std::inclusive_scan(A[l].Dims.begin(), A[l].Dims.end(), offsets[l].begin() + 1);

    if (l < levels)
      for (long long i = 0; i < xlen; i++) {
        long long ci = comm[l].iGlobal(i);
        long long child = comm[l + 1].iLocal(cells[ci].Child[0]);
        long long clen = cells[ci].Child[1] - cells[ci].Child[0];

        if (child >= 0 && clen > 0) {
          std::fill(upperIndex[l + 1].begin() + child, upperIndex[l + 1].begin() + child + clen, i);
          std::exclusive_scan(A[l + 1].DimsLr.begin() + child, A[l + 1].DimsLr.begin() + child + clen, upperOffsets[l + 1].begin() + child, 0ll);
        }
      }
  }
}

void H2MatrixSolver::matVecMul(std::complex<double> X[]) const {
  typedef Eigen::Map<Eigen::VectorXcd> Vector_t;
  typedef Eigen::Stride<Eigen::Dynamic, 1> Stride_t;
  typedef Eigen::Map<const Eigen::MatrixXcd, Eigen::Unaligned, Stride_t> Matrix_t;

  long long lbegin = Comm[levels].oLocal();
  long long llen = Comm[levels].lenLocal();
  long long lenX = offsets[levels][lbegin + llen] - offsets[levels][lbegin];

  std::vector<std::vector<std::complex<double>>> rhsX(levels + 1);
  std::vector<std::vector<std::complex<double>>> rhsY(levels + 1);

  for (long long l = levels; l >= 0; l--) {
    long long xlen = Comm[l].lenNeighbors();
    rhsX[l] = std::vector<std::complex<double>>(offsets[l][xlen], std::complex<double>(0., 0.));
    rhsY[l] = std::vector<std::complex<double>>(offsets[l][xlen], std::complex<double>(0., 0.));
  }

  Vector_t X_in(X, lenX);
  Vector_t X_leaf(rhsX[levels].data() + offsets[levels][lbegin], lenX);
  Vector_t Y_leaf(rhsY[levels].data() + offsets[levels][lbegin], lenX);

  if (X)
    X_leaf = X_in;

  for (long long l = levels; l >= 0; l--) {
    long long ibegin = Comm[l].oLocal();
    long long iboxes = Comm[l].lenLocal();
    long long xlen = Comm[l].lenNeighbors();
    Comm[l].level_merge(rhsX[l].data(), offsets[l][xlen]);
    Comm[l].neighbor_bcast(rhsX[l].data(), A[l].Dims.data());

    if (0 < l)
      for (long long y = 0; y < iboxes; y++) {
        long long M = A[l].Dims[y + ibegin];
        long long N = A[l].DimsLr[y + ibegin];
        long long U = upperIndex[l][y + ibegin];

        if (0 < N) {
          Vector_t X(rhsX[l].data() + offsets[l][ibegin + y], M);
          Vector_t Xo(rhsX[l - 1].data() + offsets[l - 1][U] + upperOffsets[l][ibegin + y], N);
          Matrix_t Q(A[l].Q[y + ibegin], M, N, Stride_t(M, 1));
          Xo = Q.transpose() * X;
        }
      }
  }

  for (long long l = 1; l <= levels; l++) {
    long long ibegin = Comm[l].oLocal();
    long long iboxes = Comm[l].lenLocal();

    for (long long y = 0; y < iboxes; y++) {
      long long M = A[l].Dims[y + ibegin];
      long long K = A[l].DimsLr[y + ibegin];
      long long UY = upperIndex[l][y + ibegin];

      if (0 < K) {
        Vector_t Y(rhsY[l].data() + offsets[l][ibegin + y], M);
        Vector_t Yo(rhsY[l - 1].data() + offsets[l - 1][UY] + upperOffsets[l][ibegin + y], K);

        for (long long yx = A[l].CRows[y]; yx < A[l].CRows[y + 1]; yx++) {
          long long x = A[l].CCols[yx];
          long long N = A[l].DimsLr[x];
          long long UX = upperIndex[l][x];

          Vector_t Xo(rhsX[l - 1].data() + offsets[l - 1][UX] + upperOffsets[l][x], N);
          Matrix_t C(A[l].C[yx], K, N, Stride_t(A[l].UpperStride[y], 1));
          Yo += C * Xo;
        }
        Matrix_t Q(A[l].Q[y + ibegin], M, K, Stride_t(M, 1));
        Y = Q * Yo;
      }
    }
  }

  for (long long y = 0; y < llen; y++) {
    long long M = A[levels].Dims[lbegin + y];
    Vector_t Y(rhsY[levels].data() + offsets[levels][lbegin + y], M);

    if (0 < M)
      for (long long yx = A[levels].ARows[y]; yx < A[levels].ARows[y + 1]; yx++) {
        long long x = A[levels].ACols[yx];
        long long N = A[levels].Dims[x];

        Vector_t X(rhsX[levels].data() + offsets[levels][x], N);
        Matrix_t C(A[levels].A[yx], M, N, Stride_t(M, 1));
        Y += C * X;
      }
  }
  if (X)
    X_in = Y_leaf;
}

void H2MatrixSolver::solvePrecondition(std::complex<double>[]) const {
  // Default preconditioner = I
}

std::pair<double, long long> H2MatrixSolver::solveGMRES(double tol, std::complex<double> x[], const std::complex<double> b[], long long inner_iters, long long outer_iters) const {
  using Eigen::VectorXcd, Eigen::MatrixXcd;

  long long lbegin = Comm[levels].oLocal();
  long long llen = Comm[levels].lenLocal();
  long long N = offsets[levels][lbegin + llen] - offsets[levels][lbegin];
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