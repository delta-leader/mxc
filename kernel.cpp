
#include <kernel.hpp>

#include <algorithm>
#include <numeric>
#include <vector>
#include <array>

#include <Eigen/Dense>


/* explicit template instantiation */
// complex double
template void gen_matrix<std::complex<double>>(const MatrixAccessor<std::complex<double>>&, long long, long long, const double*, const double*, std::complex<double>[]);
template long long adaptive_cross_approximation<std::complex<double>>(double, const MatrixAccessor<std::complex<double>>&, long long, long long, long long, const double[], const double[], long long[], long long[], std::complex<double>[], std::complex<double>[]);
template void mat_vec_reference<std::complex<double>> (const MatrixAccessor<std::complex<double>>&, long long, long long, std::complex<double> B[], const std::complex<double> X[], const double[], const double[]);
template double rel_backward_error(const MatrixAccessor<std::complex<double>>&, long long, long long, const std::complex<double> B[], const std::complex<double> X[], const double[], const double[], MPI_Comm);
// complex float
template void gen_matrix<std::complex<float>>(const MatrixAccessor<std::complex<float>>&, long long, long long, const double*, const double*, std::complex<float>[]);
template long long adaptive_cross_approximation<std::complex<float>>(double, const MatrixAccessor<std::complex<float>>&, long long, long long, long long, const double[], const double[], long long[], long long[], std::complex<float>[], std::complex<float>[]);
template void mat_vec_reference<std::complex<float>> (const MatrixAccessor<std::complex<float>>&, long long, long long, std::complex<float> B[], const std::complex<float> X[], const double[], const double[]);
template double rel_backward_error(const MatrixAccessor<std::complex<float>>&, long long, long long, const std::complex<float> B[], const std::complex<float> X[], const double[], const double[], MPI_Comm);
// double
template void gen_matrix<double>(const MatrixAccessor<double>&, long long, long long, const double*, const double*, double[]);
template long long adaptive_cross_approximation<double>(double, const MatrixAccessor<double>&, long long, long long, long long, const double[], const double[], long long[], long long[], double[], double[]);
template void mat_vec_reference<double> (const MatrixAccessor<double>&, long long, long long, double B[], const double X[], const double[], const double[]);
template double rel_backward_error(const MatrixAccessor<double>&, long long, long long, const double B[], const double X[], const double[], const double[], MPI_Comm);
// float
template void gen_matrix<float>(const MatrixAccessor<float>&, long long, long long, const double*, const double*, float[]);
template long long adaptive_cross_approximation<float>(double, const MatrixAccessor<float>&, long long, long long, long long, const double[], const double[], long long[], long long[], float[], float[]);
template void mat_vec_reference<float> (const MatrixAccessor<float>&, long long, long long, float B[], const float X[], const double[], const double[]);
template double rel_backward_error(const MatrixAccessor<float>&, long long, long long, const float B[], const float X[], const double[], const double[], MPI_Comm);

template <typename DT>
void gen_matrix(const MatrixAccessor<DT>& eval, long long m, long long n, const double* bi, const double* bj, DT Aij[]) {
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

template <typename DT>
long long adaptive_cross_approximation(double epi, const MatrixAccessor<DT>& eval, long long M, long long N, long long K, const double bi[], const double bj[], long long ipiv[], long long jpiv[], DT u[], DT v[]) {
  typedef Eigen::Matrix<DT, Eigen::Dynamic, 1> Vector_dt;
  typedef Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic> Matrix_dt;

  Matrix_dt U(M, K), V(K, N);
  Vector_dt Acol(M), Arow(N);
  Eigen::VectorXi Ipiv(K), Jpiv(K);
  long long x = 0, y = 0;

  gen_matrix(eval, 1, N, bi, bj, Arow.data());
  Arow.cwiseAbs().maxCoeff(&x);
  gen_matrix(eval, M, 1, bi, &bj[x * 3], Acol.data());
  Acol.cwiseAbs().maxCoeff(&y);
  Acol *= (DT)1. / Acol(y);
  gen_matrix(eval, 1, N, &bi[y * 3], bj, Arow.data());
  
  U.leftCols(1) = Acol;
  V.topRows(1) = Arow.transpose();
  Ipiv(0) = y;
  Jpiv(0) = x;

  Arow(x) = DT{0};
  Arow.cwiseAbs().maxCoeff(&x);
  double nrm_z = Arow.norm() * Acol.norm();
  double nrm_k = nrm_z;

  long long iters = 1;
  while (iters < K && std::numeric_limits<double>::min() < nrm_z && epi * nrm_z <= nrm_k) {
    gen_matrix(eval, M, 1, bi, &bj[x * 3], &Acol[0]);
    Acol -= U.leftCols(iters) * V.block(0, x, iters, 1);
    Acol(Ipiv.head(iters)).setZero();
    Acol.cwiseAbs().maxCoeff(&y);
    Acol *= (DT)1. / Acol(y);

    gen_matrix(eval, 1, N, &bi[y * 3], bj, Arow.data());
    Arow -= (U.block(y, 0, 1, iters) * V.topRows(iters)).transpose();

    U.middleCols(iters, 1) = Acol;
    V.middleRows(iters, 1) = Arow.transpose();
    Ipiv(iters) = y;
    Jpiv(iters) = x;

    Vector_dt Unrm = U.leftCols(iters).adjoint() * Acol;
    Vector_dt Vnrm = V.topRows(iters).conjugate() * Arow;
    DT Z_k = Unrm.transpose() * Vnrm;
    nrm_k = Arow.norm() * Acol.norm();
    nrm_z = std::sqrt(nrm_z * nrm_z + 2 * std::abs(Z_k) + nrm_k * nrm_k);
    iters++;

    Arow(Jpiv.head(iters)).setZero();
    Arow.cwiseAbs().maxCoeff(&x);
  }

  if (ipiv)
    std::transform(Ipiv.data(), Ipiv.data() + K, ipiv, [](int p) { return (long long)p; });
  if (jpiv)
    std::transform(Jpiv.data(), Jpiv.data() + K, jpiv, [](int p) { return (long long)p; });
  if (u)
    Eigen::Map<Matrix_dt>(u, M, K) = U;
  if (v)
    Eigen::Map<Matrix_dt>(v, K, N) = V;

  return iters;
}

template <typename DT>
void mat_vec_reference(const MatrixAccessor<DT>& eval, long long M, long long N, DT B[], const DT X[], const double ibodies[], const double jbodies[]) {
  typedef Eigen::Matrix<DT, Eigen::Dynamic, 1> Vector_dt;
  constexpr long long size = 256;
  Eigen::Map<const Vector_dt> x(X, N);
  Eigen::Map<Vector_dt> b(B, M);
  
  for (long long i = 0; i < M; i += size) {
    long long m = std::min(M - i, size);
    const double* bi = &ibodies[i * 3];
    Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic> A(m, size);

    for (long long j = 0; j < N; j += size) {
      const double* bj = &jbodies[j * 3];
      long long n = std::min(N - j, size);
      gen_matrix(eval, m, n, bi, bj, A.data());
      b.segment(i, m) += A.leftCols(n) * x.segment(j, n);
    }
  }
}

template <typename DT>
double rel_backward_error(const MatrixAccessor<DT>& eval, long long M, long long N, const DT B[], const DT X[], const double ibodies[], const double jbodies[], MPI_Comm world) {
  typedef Eigen::Matrix<DT, Eigen::Dynamic, 1> Vector_dt;
  constexpr long long size = 256;
  Eigen::Map<const Vector_dt> x(X, N);
  Eigen::Map<const Vector_dt> b(B, M);
  Vector_dt r = b;
  double nrm[4] = {0, 0, 0, 0};
  nrm[2] = b.squaredNorm();
  nrm[3] = x.squaredNorm();
  nrm[1] = 0;

  for (long long i = 0; i < M; i += size) {
    long long m = std::min(M - i, size);
    const double* bi = &ibodies[i * 3];
    Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic> A(m, size);

    for (long long j = 0; j < N; j += size) {
      const double* bj = &jbodies[j * 3];
      long long n = std::min(N - j, size);
      gen_matrix(eval, m, n, bi, bj, A.data());
      r.segment(i, m) -= A.leftCols(n) * x.segment(j, n);
      nrm[1] += A.squaredNorm();
    }
  }
  nrm[0] = r.squaredNorm();
  MPI_Allreduce(MPI_IN_PLACE, nrm, 3, MPI_DOUBLE, MPI_SUM, world);
  return std::sqrt(nrm[0])/(std::sqrt(nrm[1]) * std::sqrt(nrm[3]) + std::sqrt(nrm[2]));
}
