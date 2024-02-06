
#include <kernel.hpp>

#include <cblas.h>
#include <algorithm>
#include <numeric>
#include <array>

void gen_matrix(const Eval& eval, int64_t m, int64_t n, const double* bi, const double* bj, std::complex<double> Aij[], int64_t lda) {
  const std::array<double, 3>* bi3 = reinterpret_cast<const std::array<double, 3>*>(bi);
  const std::array<double, 3>* bi3_end = reinterpret_cast<const std::array<double, 3>*>(&bi[3 * m]);
  const std::array<double, 3>* bj3 = reinterpret_cast<const std::array<double, 3>*>(bj);
  const std::array<double, 3>* bj3_end = reinterpret_cast<const std::array<double, 3>*>(&bj[3 * n]);

  std::for_each(bj3, bj3_end, [&](const std::array<double, 3>& j) -> void {
    int64_t ix = std::distance(bj3, &j);
    std::for_each(bi3, bi3_end, [&](const std::array<double, 3>& i) -> void {
      int64_t iy = std::distance(bi3, &i);
      double d = std::hypot(i[0] - j[0], i[1] - j[1], i[2] - j[2]);
      Aij[iy + ix * lda] = eval(d);
    });
  });
}

int64_t interpolative_decomp_aca(double epi, const Eval& eval, int64_t M, int64_t N, int64_t K, const double bi[], const double bj[], int64_t ipiv[], std::complex<double> U[], int64_t ldu) {
  std::vector<std::complex<double>> V(K * N), L(K * K, std::complex<double>(0., 0.)), Wnrm(K), Vnrm(K);
  std::vector<std::complex<double>> Acol(M), Arow(N);
  std::vector<double> Rcol(M), Rrow(N);
  std::vector<int64_t> jpiv(K);

  gen_matrix(eval, M, 1, bi, bj, &Acol[0], M);
  std::transform(Acol.begin(), Acol.end(), Rcol.begin(), [](std::complex<double> c) { return std::abs(c); });
  int64_t x = 0;
  int64_t y = std::distance(Rcol.begin(), std::max_element(Rcol.begin(), Rcol.end()));
  std::complex<double> div = 1. / Acol[y];
  std::transform(Acol.begin(), Acol.end(), Acol.begin(), [=](std::complex<double> c) { return c * div; });

  gen_matrix(eval, N, 1, bj, &bi[y * 3], &Arow[0], N);
  std::copy(Acol.begin(), Acol.end(), &U[0]);
  std::copy(Arow.begin(), Arow.end(), &V[0]);
  ipiv[0] = y;
  jpiv[0] = 0;

  std::transform(Arow.begin(), Arow.end(), Rrow.begin(), [](std::complex<double> c) { return std::abs(c); });
  std::for_each(&jpiv[0], &jpiv[1], [&](int64_t piv) { Rrow[piv] = 0.; });
  x = std::distance(Rrow.begin(), std::max_element(Rrow.begin(), Rrow.end()));
  
  double nrm_z = cblas_dznrm2(M, &Acol[0], 1) * cblas_dznrm2(N, &Arow[0], 1);
  double nrm_k = nrm_z;

  std::complex<double> zero(0., 0.), one(1., 0.), minus_one(-1., 0.);
  int64_t iters = 1;
  while (iters < K && epi * nrm_z <= nrm_k) {
    gen_matrix(eval, M, 1, bi, &bj[x * 3], &Acol[0], M);
    cblas_zgemv(CblasColMajor, CblasNoTrans, M, iters, &minus_one, &U[0], ldu, &V[x], N, &one, &Acol[0], 1);

    std::transform(Acol.begin(), Acol.end(), Rcol.begin(), [](std::complex<double> c) { return std::abs(c); });
    std::for_each(&ipiv[0], &ipiv[iters], [&](int64_t piv) { Rcol[piv] = 0.; });
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
    std::for_each(&jpiv[0], &jpiv[iters], [&](int64_t piv) { Rrow[piv] = 0.; });
    x = std::distance(Rrow.begin(), std::max_element(Rrow.begin(), Rrow.end()));
  }

  cblas_ztrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit, M, iters, &one, &L[0], K, &U[0], ldu);
  return iters;
}


void mat_vec_reference(const Eval& eval, int64_t M, int64_t N, int64_t nrhs, std::complex<double> B[], int64_t ldB, const std::complex<double> X[], int64_t ldX, const double ibodies[], const double jbodies[]) {
  constexpr int64_t size = 256;
  std::vector<std::complex<double>> A(size * size);
  std::complex<double> one(1., 0.);
  
  for (int64_t i = 0; i < M; i += size) {
    int64_t m = std::min(M - i, size);
    const double* bi = &ibodies[i * 3];
    for (int64_t j = 0; j < N; j += size) {
      const double* bj = &jbodies[j * 3];
      int64_t n = std::min(N - j, size);
      gen_matrix(eval, m, n, bi, bj, &A[0], size);
      cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, nrhs, n, &one, &A[0], size, &X[j], ldX, &one, &B[i], ldB);
    }
  }
}

