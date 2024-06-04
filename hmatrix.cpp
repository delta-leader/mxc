
#include <hmatrix.hpp>
#include <build_tree.hpp>
#include <kernel.hpp>

#include <mkl.h>
#include <algorithm>
#include <numeric>

long long adaptive_cross_approximation(double epi, const MatrixAccessor& eval, long long M, long long N, long long K, const double bi[], const double bj[], std::complex<double> U[], std::complex<double> Vh[]) {
  std::vector<std::complex<double>> Unrm(K), Vnrm(K);
  std::vector<std::complex<double>> Acol(M), Arow(N);
  std::vector<double> Rcol(M), Rrow(N);
  std::vector<long long> ipiv(K), jpiv(K);

  gen_matrix(eval, M, 1, bi, bj, &Acol[0]);
  std::transform(Acol.begin(), Acol.end(), Rcol.begin(), [](std::complex<double> c) { return std::abs(c); });
  long long x = 0;
  long long y = std::distance(Rcol.begin(), std::max_element(Rcol.begin(), Rcol.end()));
  if (std::abs(Acol[y]) < std::numeric_limits<double>::min())
    return 0;

  std::complex<double> div = 1. / Acol[y];
  std::transform(Acol.begin(), Acol.end(), Acol.begin(), [=](std::complex<double> c) { return c * div; });

  gen_matrix(eval, N, 1, bj, &bi[y * 3], &Arow[0]);
  std::copy(Acol.begin(), Acol.end(), &U[0]);
  std::copy(Arow.begin(), Arow.end(), &Vh[0]);
  ipiv[0] = y;
  jpiv[0] = 0;

  std::transform(Arow.begin(), Arow.end(), Rrow.begin(), [](std::complex<double> c) { return std::abs(c); });
  std::for_each(&jpiv[0], &jpiv[1], [&](long long piv) { Rrow[piv] = 0.; });
  x = std::distance(Rrow.begin(), std::max_element(Rrow.begin(), Rrow.end()));

  double nrm_z = cblas_dznrm2(M, &Acol[0], 1) * cblas_dznrm2(N, &Arow[0], 1);
  double nrm_k = nrm_z;

  std::complex<double> zero(0., 0.), one(1., 0.), minus_one(-1., 0.);
  long long iters = 1;
  while (iters < K && std::numeric_limits<double>::min() < nrm_z && epi * nrm_z <= nrm_k) {
    gen_matrix(eval, M, 1, bi, &bj[x * 3], &Acol[0]);
    cblas_zgemv(CblasColMajor, CblasNoTrans, M, iters, &minus_one, &U[0], M, &Vh[x], N, &one, &Acol[0], 1);

    std::transform(Acol.begin(), Acol.end(), Rcol.begin(), [](std::complex<double> c) { return std::abs(c); });
    std::for_each(&ipiv[0], &ipiv[iters], [&](long long piv) { Rcol[piv] = 0.; });
    y = std::distance(Rcol.begin(), std::max_element(Rcol.begin(), Rcol.end()));

    std::complex<double> div = 1. / Acol[y];
    std::transform(Acol.begin(), Acol.end(), Acol.begin(), [=](std::complex<double> c) { return c * div; });
    gen_matrix(eval, N, 1, bj, &bi[y * 3], &Arow[0]);
    cblas_zgemv(CblasColMajor, CblasNoTrans, N, iters, &minus_one, &Vh[0], N, &U[y], M, &one, &Arow[0], 1);

    std::copy(Acol.begin(), Acol.end(), &U[iters * M]);
    std::copy(Arow.begin(), Arow.end(), &Vh[iters * N]);
    ipiv[iters] = y;
    jpiv[iters] = x;

    cblas_zgemv(CblasColMajor, CblasConjTrans, M, iters, &one, &U[0], M, &Acol[0], 1, &zero, &Unrm[0], 1);
    cblas_zgemv(CblasColMajor, CblasConjTrans, N, iters, &one, &Vh[0], N, &Arow[0], 1, &zero, &Vnrm[0], 1);
    std::complex<double> Z_k = std::transform_reduce(&Unrm[0], &Unrm[iters], &Vnrm[0], std::complex<double>(0., 0.),
      std::plus<std::complex<double>>(), std::multiplies<std::complex<double>>());
    nrm_k = cblas_dznrm2(M, &Acol[0], 1) * cblas_dznrm2(N, &Arow[0], 1);
    nrm_z = std::sqrt(nrm_z * nrm_z + 2 * std::abs(Z_k) + nrm_k * nrm_k);
    iters++;

    std::transform(Arow.begin(), Arow.end(), Rrow.begin(), [](std::complex<double> c) { return std::abs(c); });
    std::for_each(jpiv.begin(), jpiv.begin() + iters, [&](long long piv) { Rrow[piv] = 0.; });
    x = std::distance(Rrow.begin(), std::max_element(Rrow.begin(), Rrow.end()));
  }

  return iters;
}

HMatrix::HMatrix(const MatrixAccessor& eval, double epi, long long rank, long long lbegin, long long lend, const Cell cells[], const CSR& Far, const double bodies[]) {
  long long vlen = Far.RowIndex[lend] - Far.RowIndex[lbegin];
  U = std::vector<std::vector<std::complex<double>>>(vlen);
  Vh = std::vector<std::vector<std::complex<double>>>(vlen);
  M = std::vector<long long>(vlen);
  N = std::vector<long long>(vlen);
  K = std::vector<long long>(vlen);

  for (long long y = lbegin; y < lend; y++) {
    for (long long yx = Far.RowIndex[y]; yx < Far.RowIndex[y + 1]; yx++) {
      long long x = Far.ColIndex[yx];
      long long m = cells[y].Body[1] - cells[y].Body[0];
      long long n = cells[x].Body[1] - cells[x].Body[0];
      const double* Xbodies = &bodies[3 * cells[x].Body[0]];
      const double* Ybodies = &bodies[3 * cells[y].Body[0]];

      long long k = std::min(rank, std::min(m, n));
      U[yx] = std::vector<std::complex<double>>(m * k);
      Vh[yx] = std::vector<std::complex<double>>(n * k);
      K[yx] = 0 < k ? adaptive_cross_approximation(epi, eval, m, n, k, Ybodies, Xbodies, U[yx].data(), Vh[yx].data()) : 0;
      M[yx] = m;
      N[yx] = n;
    }
  }
}
