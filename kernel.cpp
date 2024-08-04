
#include <kernel.hpp>

#include <algorithm>
#include <numeric>
#include <vector>
#include <array>
#include <Eigen/Dense>

void gen_matrix(const MatrixAccessor& kernel, const long long M, const long long N, const double* const bi, const double* const bj, std::complex<double> Aij[]) {
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
      Aij[iy + ix * M] = kernel(d);
    });
  });
}

long long adaptive_cross_approximation(const MatrixAccessor& kernel, const double epsilon, const long long max_rank, const long long nrows, const long long ncols, const double row_bodies[], const double col_bodies[], long long row_piv[], long long col_piv[]) {
  // low-rank matrices U & V
  Eigen::MatrixXcd U(nrows, max_rank), V(max_rank, ncols);
  // workspace for selected rows/columns of A
  Eigen::VectorXcd Acol(nrows), Arow(ncols);
  // row/column pivots
  Eigen::VectorXi Ipiv(max_rank), Jpiv(max_rank);
  long long x = 0, y = 0;

  // generate the first row of A
  gen_matrix(kernel, 1, ncols, row_bodies, col_bodies, Arow.data());
  // store the index of the maximum absolute value in x
  Arow.cwiseAbs().maxCoeff(&x);
  // generate the column containing the maximum absolute value
  gen_matrix(kernel, nrows, 1, row_bodies, &col_bodies[x * 3], Acol.data());
  // store the index of the maximum absolute value in y
  Acol.cwiseAbs().maxCoeff(&y);
  // normalize the column by the maximum element
  Acol *= 1. / Acol(y);
  // generate the row containing the maximum absolute value
  gen_matrix(kernel, 1, ncols, &row_bodies[y * 3], col_bodies, Arow.data());
  
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
  while (iters < max_rank && std::numeric_limits<double>::min() < nrm_z && epsilon * nrm_z <= nrm_k) {
    // create the new column
    gen_matrix(kernel, nrows, 1, row_bodies, &col_bodies[x * 3], &Acol[0]);
    Acol -= U.leftCols(iters) * V.block(0, x, iters, 1);
    Acol(Ipiv.head(iters)).setZero();
    Acol.cwiseAbs().maxCoeff(&y);
    Acol *= 1. / Acol(y);

    gen_matrix(kernel, 1, ncols, &row_bodies[y * 3], col_bodies, Arow.data());
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

  if (row_piv)
    std::transform(Ipiv.data(), Ipiv.data() + max_rank, row_piv, [](int p) { return (long long)p; });
  if (col_piv)
    std::transform(Jpiv.data(), Jpiv.data() + max_rank, col_piv, [](int p) { return (long long)p; });
  // removed since they are always passed as nullptrs
  /*
  if (u)
    Eigen::Map<Eigen::MatrixXcd>(u, M, max_rank) = U;
  if (v)
    Eigen::Map<Eigen::MatrixXcd>(v, max_rank, N) = V;
  */
  return iters;
}

void mat_vec_reference(const MatrixAccessor& kernel, const long long nrows, const long long ncols, std::complex<double> B[], const std::complex<double> X[], const double row_bodies[], const double col_bodies[]) {
  // calculate in blocks  (to prevent memory issues?)
  constexpr long long block_size = 256;
  Eigen::Map<const Eigen::VectorXcd> X_ref(X, ncols);
  Eigen::Map<Eigen::VectorXcd> B_ref(B, nrows);
  
  for (long long i = 0; i < nrows; i += block_size) {
    const long long brows = std::min(nrows - i, block_size);
    const double* const bi_bodies = &row_bodies[i * 3];
    Eigen::MatrixXcd A_block(brows, block_size);

    for (long long j = 0; j < ncols; j += block_size) {
      const double* const bj_bodies = &col_bodies[j * 3];
      const long long bcols = std::min(ncols - j, block_size);
      gen_matrix(kernel, brows, bcols, bi_bodies, bj_bodies, A_block.data());
      B_ref.segment(i, brows) += A_block.leftCols(bcols) * X_ref.segment(j, bcols);
    }
  }
}

