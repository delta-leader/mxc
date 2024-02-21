
#include <kernel.hpp>

#include <cblas.h>
#include <algorithm>
#include <numeric>
#include <array>

void gen_matrix(const MatrixAccessor& eval, long long m, long long n, const double* bi, const double* bj, std::complex<double> Aij[], long long lda) {
  const std::array<double, 3>* bi3 = reinterpret_cast<const std::array<double, 3>*>(bi);
  const std::array<double, 3>* bi3_end = reinterpret_cast<const std::array<double, 3>*>(&bi[3 * m]);
  const std::array<double, 3>* bj3 = reinterpret_cast<const std::array<double, 3>*>(bj);
  const std::array<double, 3>* bj3_end = reinterpret_cast<const std::array<double, 3>*>(&bj[3 * n]);

  std::for_each(bj3, bj3_end, [&](const std::array<double, 3>& j) -> void {
    long long ix = std::distance(bj3, &j);
    std::for_each(bi3, bi3_end, [&](const std::array<double, 3>& i) -> void {
      long long iy = std::distance(bi3, &i);
      double d = std::hypot(i[0] - j[0], i[1] - j[1], i[2] - j[2]);
      Aij[iy + ix * lda] = eval(d);
    });
  });
}

long long interpolative_decomp_aca(double epi, const MatrixAccessor& eval, long long M, long long N, long long K, const double bi[], const double bj[], long long ipiv[], std::complex<double> U[], long long ldu) {
  std::vector<std::complex<double>> V(K * N), L(K * K, std::complex<double>(0., 0.)), Wnrm(K), Vnrm(K);
  std::vector<std::complex<double>> Acol(M), Arow(N);
  std::vector<double> Rcol(M), Rrow(N);
  std::vector<long long> jpiv(K);

  gen_matrix(eval, M, 1, bi, bj, &Acol[0], M);
  std::transform(Acol.begin(), Acol.end(), Rcol.begin(), [](std::complex<double> c) { return std::abs(c); });
  long long x = 0;
  long long y = std::distance(Rcol.begin(), std::max_element(Rcol.begin(), Rcol.end()));
  if (std::abs(Acol[y]) < std::numeric_limits<double>::min())
    return 0;
  
  std::complex<double> div = 1. / Acol[y];
  std::transform(Acol.begin(), Acol.end(), Acol.begin(), [=](std::complex<double> c) { return c * div; });

  gen_matrix(eval, N, 1, bj, &bi[y * 3], &Arow[0], N);
  std::copy(Acol.begin(), Acol.end(), &U[0]);
  std::copy(Arow.begin(), Arow.end(), &V[0]);
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
    gen_matrix(eval, M, 1, bi, &bj[x * 3], &Acol[0], M);
    cblas_zgemv(CblasColMajor, CblasNoTrans, M, iters, &minus_one, &U[0], ldu, &V[x], N, &one, &Acol[0], 1);

    std::transform(Acol.begin(), Acol.end(), Rcol.begin(), [](std::complex<double> c) { return std::abs(c); });
    std::for_each(&ipiv[0], &ipiv[iters], [&](long long piv) { Rcol[piv] = 0.; });
    y = std::distance(Rcol.begin(), std::max_element(Rcol.begin(), Rcol.end()));

    std::complex<double> div = 1. / Acol[y];
    std::transform(Acol.begin(), Acol.end(), Acol.begin(), [=](std::complex<double> c) { return c * div; });
    gen_matrix(eval, N, 1, bj, &bi[y * 3], &Arow[0], N);
    cblas_zgemv(CblasColMajor, CblasNoTrans, N, iters, &minus_one, &V[0], N, &U[y], ldu, &one, &Arow[0], 1);

    std::copy(Acol.begin(), Acol.end(), &U[iters * ldu]);
    std::copy(Arow.begin(), Arow.end(), &V[iters * N]);
    cblas_zcopy(iters, &U[y], ldu, &L[iters], K);
    ipiv[iters] = y;
    jpiv[iters] = x;

    cblas_zgemv(CblasColMajor, CblasTrans, M, iters, &one, &U[0], ldu, &Acol[0], 1, &zero, &Wnrm[0], 1);
    cblas_zgemv(CblasColMajor, CblasTrans, N, iters, &one, &V[0], N, &Arow[0], 1, &zero, &Vnrm[0], 1);
    std::complex<double> Z_k = std::transform_reduce(&Wnrm[0], &Wnrm[iters], &Vnrm[0], std::complex<double>(0., 0.), 
      std::plus<std::complex<double>>(), std::multiplies<std::complex<double>>());
    nrm_k = cblas_dznrm2(M, &Acol[0], 1) * cblas_dznrm2(N, &Arow[0], 1);
    nrm_z = std::sqrt(nrm_z * nrm_z + 2 * std::abs(Z_k) + nrm_k * nrm_k);
    iters++;

    std::transform(Arow.begin(), Arow.end(), Rrow.begin(), [](std::complex<double> c) { return std::abs(c); });
    std::for_each(&jpiv[0], &jpiv[iters], [&](long long piv) { Rrow[piv] = 0.; });
    x = std::distance(Rrow.begin(), std::max_element(Rrow.begin(), Rrow.end()));
  }

  cblas_ztrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit, M, iters, &one, &L[0], K, &U[0], ldu);
  return iters;
}


void mat_vec_reference(const MatrixAccessor& eval, long long M, long long N, long long nrhs, std::complex<double> B[], long long ldB, const std::complex<double> X[], long long ldX, const double ibodies[], const double jbodies[]) {
  constexpr long long size = 256;
  std::vector<std::complex<double>> A(size * size);
  std::complex<double> one(1., 0.);
  
  for (long long i = 0; i < M; i += size) {
    long long m = std::min(M - i, size);
    const double* bi = &ibodies[i * 3];
    for (long long j = 0; j < N; j += size) {
      const double* bj = &jbodies[j * 3];
      long long n = std::min(N - j, size);
      gen_matrix(eval, m, n, bi, bj, &A[0], size);
      cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, nrhs, n, &one, &A[0], size, &X[j], ldX, &one, &B[i], ldB);
    }
  }
}

