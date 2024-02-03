
#include <kernel.hpp>

#include <Eigen/Dense>

void gen_matrix(const Eval& eval, int64_t M, int64_t N, const double* bi, const double* bj, std::complex<double> Aij[], int64_t lda) {
  Eigen::Map<const Eigen::MatrixXd> Bi(bi, 3, M);
  Eigen::Map<const Eigen::MatrixXd> Bj(bj, 3, N);

  for (int64_t ix = 0; ix < N; ix++)
    for (int64_t iy = 0; iy < M; iy++)
      Aij[iy + ix * lda] = eval((Bi.col(iy) - Bj.col(ix)).norm());
}

int64_t aca_kernel(double epi, const Eval& eval, int64_t M, int64_t N, int64_t K, const double bi[], const double bj[], std::complex<double> U[], int64_t ldu, std::complex<double> V[], int64_t ldv) {
  Eigen::Map<const Eigen::MatrixXd> Bi(bi, 3, M);
  Eigen::Map<const Eigen::MatrixXd> Bj(bj, 3, N);
  Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned, Eigen::OuterStride<>> U_(U, M, K, Eigen::OuterStride(ldu));
  Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned, Eigen::OuterStride<>> V_(V, K, N, Eigen::OuterStride(ldv));

  int64_t x = 0;
  Eigen::VectorXcd Acol(M), Arow(N);
  Eigen::VectorXd Icol(M), Irow(N);
  Eigen::VectorXd Rcol(M), Rrow(N);
  Icol.setOnes();
  Irow.setOnes();

  for (int64_t i = 0; i < M; i++)
    Acol(i) = eval((Bi.col(i) - Bj.col(x)).norm());
  
  int64_t y = 0;
  Rcol = Acol.cwiseAbs();
  Rcol.maxCoeff(&y);
  Acol *= 1. / Acol(y);
  for (int64_t i = 0; i < N; i++)
    Arow(i) = eval((Bi.col(y) - Bj.col(i)).norm());
  
  U_.leftCols(1) = Acol;
  V_.topRows(1) = Arow.transpose();

  Irow(x) = 0;
  Rrow = Arow.cwiseProduct(Irow).cwiseAbs();
  Rrow.maxCoeff(&x);
  double nrm_z = Arow.norm() * Acol.norm();
  double nrm_k = nrm_z;

  int64_t iters = 1;
  if (epi < 1.) while (iters < K && epi * nrm_z <= nrm_k) {
    for (int64_t i = 0; i < M; i++)
      Acol(i) = eval((Bi.col(i) - Bj.col(x)).norm());
    Acol -= U_.leftCols(iters) * V_.block(0, x, iters, 1);
    Icol(y) = 0;
    Rcol = Acol.cwiseProduct(Icol).cwiseAbs();
    Rcol.maxCoeff(&y);
    Acol *= 1. / Acol(y);

    for (int64_t i = 0; i < N; i++)
      Arow(i) = eval((Bi.col(y) - Bj.col(i)).norm());
    Arow -= V_.topRows(iters).transpose() * U_.block(y, 0, 1, iters).transpose();

    U_.middleCols(iters, 1) = Acol;
    V_.middleRows(iters, 1) = Arow.transpose();

    Eigen::MatrixXcd Z_k = (Acol.transpose() * U_.leftCols(iters)) * (V_.topRows(iters) * Arow);
    nrm_k = Arow.norm() * Acol.norm();
    nrm_z = std::sqrt(nrm_z + std::abs(Z_k(0, 0) * Z_k(0, 0) + nrm_k * nrm_k));
    iters++;

    Irow(x) = 0;
    Rrow = Arow.cwiseProduct(Irow).cwiseAbs();
    Rrow.maxCoeff(&x);
  }
  return K;
}


void mat_vec_reference(const Eval& eval, int64_t M, int64_t N, int64_t nrhs, std::complex<double> B[], int64_t ldB, const std::complex<double> X[], int64_t ldX, const double ibodies[], const double jbodies[]) {
  constexpr int64_t size = 64;
  Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, size, size> A(size, size);
  Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned, Eigen::OuterStride<>> B_(B, M, nrhs, Eigen::OuterStride(ldB));
  Eigen::Map<const Eigen::MatrixXcd, Eigen::Unaligned, Eigen::OuterStride<>> X_(X, N, nrhs, Eigen::OuterStride(ldX));
  
  for (int64_t i = 0; i < M; i += size) {
    int64_t m = std::min(M - i, size);
    const double* bi = &ibodies[i * 3];
    for (int64_t j = 0; j < N; j += size) {
      int64_t n = std::min(N - j, size);
      const double* bj = &jbodies[j * 3];
      gen_matrix(eval, m, n, bi, bj, A.data(), size);
      B_.middleRows(i, m) += A.topLeftCorner(m, n)*X_.middleRows(j, n);
    }
  }
}

