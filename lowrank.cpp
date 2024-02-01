
#include <lowrank.hpp>

#include <cblas.h>
#include <lapacke.h>
#include <algorithm>

LowRank::LowRank(double epi, int64_t M, int64_t N, std::complex<double> A[], int64_t lda, const double* bj, int64_t incb) : N(N), Rank(0) {
  std::vector<std::complex<double>> TAU(std::min(M, N));
  std::complex<double> One(1., 0.), Zero(0., 0.);
  std::vector<int32_t> Ipiv(N, 0);

  const lapack_complex_double* One_ = reinterpret_cast<const lapack_complex_double*>(&One);
  const lapack_complex_double* Zero_ = reinterpret_cast<const lapack_complex_double*>(&Zero);
  lapack_complex_double* A_ = reinterpret_cast<lapack_complex_double*>(A);
  lapack_complex_double* TAU_ = reinterpret_cast<lapack_complex_double*>(&TAU[0]);

  LAPACKE_zgeqp3(LAPACK_COL_MAJOR, M, N, A_, lda, &Ipiv[0], TAU_);
  double s0 = epi * std::sqrt(std::norm(A[0]));
  while (Rank < std::min(M, N) && s0 <= std::sqrt(std::norm(A[Rank * (lda + 1)])))
    ++Rank;

  V = std::vector<std::complex<double>>(Rank * N);
  BodiesJ = std::vector<double>(Rank * incb);
  
  if (Rank > 0) {
    if (Rank < N)
      cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, Rank, N - Rank, &One, A, lda, &A[Rank * lda], lda);
    LAPACKE_zlaset(LAPACK_COL_MAJOR, 'F', Rank, Rank, *Zero_, *One_, A_, lda);
    std::for_each(Ipiv.begin(), Ipiv.begin() + N, [&](int32_t& piv) {
      int64_t i = std::distance(&Ipiv[0], &piv); std::copy(&A[i * lda], &A[i * lda + Rank], &V[(piv - 1) * Rank]); });
    std::for_each(Ipiv.begin(), Ipiv.begin() + Rank, [&](int32_t& piv) {
      int64_t i = std::distance(&Ipiv[0], &piv); std::copy(&bj[(piv - 1) * incb], &bj[piv * incb], &BodiesJ[i * incb]); });
  }
}
