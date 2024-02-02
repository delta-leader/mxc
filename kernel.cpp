
#include <kernel.hpp>

#include <Eigen/Dense>

void gen_matrix(const Eval& eval, int64_t M, int64_t N, const double* bi, const double* bj, std::complex<double> Aij[], int64_t lda) {
  Eigen::Map<const Eigen::MatrixXd> Bi(bi, 3, M);
  Eigen::Map<const Eigen::MatrixXd> Bj(bj, 3, N);

  for (int64_t ix = 0; ix < N; ix++)
    for (int64_t iy = 0; iy < M; iy++)
      Aij[iy + ix * lda] = eval((Bi.col(iy) - Bj.col(ix)).norm());
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

