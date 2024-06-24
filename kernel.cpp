
#include <kernel.hpp>

#include <algorithm>
#include <numeric>
#include <vector>
#include <array>
#include <Eigen/Dense>

void gen_matrix(const MatrixAccessor& eval, long long m, long long n, const double* bi, const double* bj, std::complex<double> Aij[]) {
  const std::array<double, 3>* bi3 = reinterpret_cast<const std::array<double, 3>*>(bi);
  const std::array<double, 3>* bi3_end = reinterpret_cast<const std::array<double, 3>*>(&bi[3 * m]);
  const std::array<double, 3>* bj3 = reinterpret_cast<const std::array<double, 3>*>(bj);
  const std::array<double, 3>* bj3_end = reinterpret_cast<const std::array<double, 3>*>(&bj[3 * n]);

  std::for_each(bj3, bj3_end, [&](const std::array<double, 3>& j) -> void {
    long long ix = std::distance(bj3, &j);
    std::for_each(bi3, bi3_end, [&](const std::array<double, 3>& i) -> void {
      long long iy = std::distance(bi3, &i);
      double d = std::hypot(i[0] - j[0], i[1] - j[1], i[2] - j[2]);
      Aij[iy + ix * m] = eval(d);
    });
  });
}

long long interpolative_decomp_aca(double epi, const MatrixAccessor& eval, long long M, long long N, long long K, const double bi[], const double bj[], long long piv[], std::complex<double> U[]) {
  Eigen::MatrixXcd W(M, K), V(K, N), L(K, K);
  Eigen::VectorXcd Acol(M), Arow(N);
  Eigen::VectorXi ipiv(K), jpiv(K);
  long long x = 0, y = 0;

  gen_matrix(eval, M, 1, bi, bj, Acol.data());
  Acol.cwiseAbs().maxCoeff(&y);
  Acol *= 1. / Acol(y);
  gen_matrix(eval, 1, N, &bi[y * 3], bj, Arow.data());
  
  W.leftCols(1) = Acol;
  V.topRows(1) = Arow.transpose();
  ipiv(0) = y;
  jpiv(0) = x;

  Arow(jpiv.head(1)) = Eigen::VectorXcd::Zero(1);
  Arow.cwiseAbs().maxCoeff(&x);
  double nrm_z = Arow.norm() * Acol.norm();
  double nrm_k = nrm_z;

  long long iters = 1;
  while (iters < K && std::numeric_limits<double>::min() < nrm_z && epi * nrm_z <= nrm_k) {
    gen_matrix(eval, M, 1, bi, &bj[x * 3], &Acol[0]);
    Acol -= W.leftCols(iters) * V.block(0, x, iters, 1);
    Acol(ipiv.head(iters)) = Eigen::VectorXcd::Zero(iters);
    Acol.cwiseAbs().maxCoeff(&y);
    Acol *= 1. / Acol(y);

    gen_matrix(eval, 1, N, &bi[y * 3], bj, Arow.data());
    Arow -= (W.block(y, 0, 1, iters) * V.topRows(iters)).transpose();

    W.middleCols(iters, 1) = Acol;
    V.middleRows(iters, 1) = Arow.transpose();
    L.block(iters, 0, 1, iters) = W.block(y, 0, 1, iters);
    ipiv(iters) = y;
    jpiv(iters) = x;

    Eigen::VectorXcd Unrm = W.leftCols(iters).adjoint() * Acol;
    Eigen::VectorXcd Vnrm = V.topRows(iters).conjugate() * Arow;
    std::complex<double> Z_k = Unrm.transpose() * Vnrm;
    nrm_k = Arow.norm() * Acol.norm();
    nrm_z = std::sqrt(nrm_z * nrm_z + 2 * std::abs(Z_k) + nrm_k * nrm_k);
    iters++;

    Arow(jpiv.head(iters)) = Eigen::VectorXcd::Zero(iters);
    Arow.cwiseAbs().maxCoeff(&x);
  }

  if (U)
    Eigen::Map<Eigen::MatrixXcd>(U, M, K) = L.triangularView<Eigen::Lower>().solve<Eigen::OnTheRight>(W);
  if (piv)
    std::transform(ipiv.data(), ipiv.data() + K, piv, [](int p) { return (long long)p; });
  return iters;
}

void mat_vec_reference(const MatrixAccessor& eval, long long M, long long N, std::complex<double> B[], const std::complex<double> X[], const double ibodies[], const double jbodies[]) {
  constexpr long long size = 256;
  Eigen::Map<const Eigen::VectorXcd> x(X, N);
  Eigen::Map<Eigen::VectorXcd> b(B, M);
  
  for (long long i = 0; i < M; i += size) {
    long long m = std::min(M - i, size);
    const double* bi = &ibodies[i * 3];
    Eigen::MatrixXcd A(m, size);

    for (long long j = 0; j < N; j += size) {
      const double* bj = &jbodies[j * 3];
      long long n = std::min(N - j, size);
      gen_matrix(eval, m, n, bi, bj, A.data());
      b.segment(i, m) += A.leftCols(n) * x.segment(j, n);
    }
  }
}

