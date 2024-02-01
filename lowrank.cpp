
#include <lowrank.hpp>

#include <cblas.h>
#include <lapacke.h>
#include <algorithm>

LowRank::LowRank(double epi, int64_t M, int64_t N, std::complex<double> A[], int64_t lda) : N(N), Rank(0) {
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
  Jpiv = std::vector<int64_t>(Rank);
  
  if (Rank > 0) {
    if (Rank < N)
      cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, Rank, N - Rank, &One, A, lda, &A[Rank * lda], lda);
    LAPACKE_zlaset(LAPACK_COL_MAJOR, 'F', Rank, Rank, *Zero_, *One_, A_, lda);
    std::for_each(Ipiv.begin(), Ipiv.begin() + N, [&](int32_t& piv) {
      int64_t i = std::distance(&Ipiv[0], &piv); std::copy(&A[i * lda], &A[i * lda + Rank], &V[(piv - 1) * Rank]); });
    std::transform(Ipiv.begin(), Ipiv.begin() + Rank, Jpiv.begin(), [](int32_t piv) { return (int64_t)piv - 1; });
  }
}

void LowRank::SelectR(void* Xout, const void* Xin, int64_t elem) const {
  if (Xin == Xout) {
    std::vector<unsigned char> Y(elem * Rank);
    SelectR(&Y[0], Xin, elem);
    std::copy(Y.begin(), Y.end(), reinterpret_cast<unsigned char*>(Xout));
  }
  else {
    const unsigned char* X = reinterpret_cast<const unsigned char*>(Xin);
    unsigned char* Y = reinterpret_cast<unsigned char*>(Xout);
    std::for_each(Jpiv.begin(), Jpiv.begin() + Rank, [&](const int64_t& piv) {
      int64_t i = std::distance(&Jpiv[0], &piv); std::copy(&X[piv * elem], &X[(piv + 1) * elem], &Y[i * elem]); });
  }
}
