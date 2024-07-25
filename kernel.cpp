
#include <kernel.hpp>

#include <algorithm>
#include <numeric>
#include <vector>
#include <array>
#include <Eigen/Dense>

void gen_matrix(const MatrixAccessor& eval, const long long M, const long long N, const double* const bi, const double* const bj, std::complex<double> Aij[]) {
  const std::array<double, 3>* bi3 = reinterpret_cast<const std::array<double, 3>*>(bi);
  const std::array<double, 3>* bi3_end = reinterpret_cast<const std::array<double, 3>*>(&bi[3 * M]);
  const std::array<double, 3>* bj3 = reinterpret_cast<const std::array<double, 3>*>(bj);
  const std::array<double, 3>* bj3_end = reinterpret_cast<const std::array<double, 3>*>(&bj[3 * N]);

  std::for_each(bj3, bj3_end, [&](const std::array<double, 3>& j) -> void {
    // row coordinate
    long long ix = std::distance(bj3, &j);
    std::for_each(bi3, bi3_end, [&](const std::array<double, 3>& i) -> void {
      // column coordinate
      long long iy = std::distance(bi3, &i);
      // square root of the sum of squares (i.e. distance)
      double d = std::hypot(i[0] - j[0], i[1] - j[1], i[2] - j[2]);
      // TODO confirm that this is row major
      Aij[iy + ix * M] = eval(d);
    });
  });
}

long long adaptive_cross_approximation(const double epi, const MatrixAccessor& eval, const long long nrows, const long long ncols, const long long max_rank, const double bi[], const double bj[], long long ipiv[], long long jpiv[]) {
  // low-rank matrices U & V
  Eigen::MatrixXcd U(nrows, max_rank), V(max_rank, ncols);
  // workspace for selected rows/columns of A
  Eigen::VectorXcd Acol(nrows), Arow(ncols);
  // pivots
  Eigen::VectorXi Ipiv(max_rank), Jpiv(max_rank);
  long long x = 0, y = 0;

  // generate the first row of A
  gen_matrix(eval, 1, ncols, bi, bj, Arow.data());
  // store the index of the maximum absolute value in x
  Arow.cwiseAbs().maxCoeff(&x);
  // generate the column containing the maximum absolute value
  gen_matrix(eval, nrows, 1, bi, &bj[x * 3], Acol.data());
  // store the index of the maximum absolute value in y
  Acol.cwiseAbs().maxCoeff(&y);
  // normalize the column by the maximum element
  Acol *= 1. / Acol(y);
  // generate the row containing the maximum absolute value
  gen_matrix(eval, 1, ncols, &bi[y * 3], bj, Arow.data());
  
  // add the selected column to U
  U.leftCols(1) = Acol;
  // add the selected row to V
  V.topRows(1) = Arow.transpose();
  // store the indices of the selected row/column
  Ipiv(0) = y;
  Jpiv(0) = x;
  
  // set the maximum row element to zero
  // TODO does this affect V?
  Arow(x) = std::complex<double>(0., 0.);
  // find the column with the containing the new maximum absolute value
  Arow.cwiseAbs().maxCoeff(&x);
  // ||Z||
  double nrm_z = Arow.norm() * Acol.norm();
  // ||u|| * ||v||
  double nrm_k = nrm_z;

  long long iters = 1;
  // iterate until either the desired accuracy or the maximum rank is reached
  // the convergence criteria for ACA is ||u|| * ||v|| <= epsilon * ||Z||
  // where u and v are the newly selected row/column and Z is the low rank approximation so far
  // the complete algorithm can be found in
  // "The Adaptive Cross Approximation Algorithm for Accelerated Method of Moments Computations of EMC Problems"
  while (iters < max_rank && std::numeric_limits<double>::min() < nrm_z && epi * nrm_z <= nrm_k) {
    // create the new column
    gen_matrix(eval, nrows, 1, bi, &bj[x * 3], &Acol[0]);
    Acol -= U.leftCols(iters) * V.block(0, x, iters, 1);
    Acol(Ipiv.head(iters)).setZero();
    Acol.cwiseAbs().maxCoeff(&y);
    Acol *= 1. / Acol(y);

    gen_matrix(eval, 1, ncols, &bi[y * 3], bj, Arow.data());
    Arow -= (U.block(y, 0, 1, iters) * V.topRows(iters)).transpose();

    U.middleCols(iters, 1) = Acol;
    V.middleRows(iters, 1) = Arow.transpose();
    Ipiv(iters) = y;
    Jpiv(iters) = x;

    Eigen::VectorXcd Unrm = U.leftCols(iters).adjoint() * Acol;
    Eigen::VectorXcd Vnrm = V.topRows(iters).conjugate() * Arow;
    std::complex<double> Z_k = Unrm.transpose() * Vnrm;
    nrm_k = Arow.norm() * Acol.norm();
    nrm_z = std::sqrt(nrm_z * nrm_z + 2 * std::abs(Z_k) + nrm_k * nrm_k);
    iters++;

    Arow(Jpiv.head(iters)).setZero();
    Arow.cwiseAbs().maxCoeff(&x);
  }

  if (ipiv)
    std::transform(Ipiv.data(), Ipiv.data() + max_rank, ipiv, [](int p) { return (long long)p; });
  if (jpiv)
    std::transform(Jpiv.data(), Jpiv.data() + max_rank, jpiv, [](int p) { return (long long)p; });
  // removed since they are always passed as nullptrs
  /*
  if (u)
    Eigen::Map<Eigen::MatrixXcd>(u, M, max_rank) = U;
  if (v)
    Eigen::Map<Eigen::MatrixXcd>(v, max_rank, N) = V;
  */
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

