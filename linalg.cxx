

#include "linalg.hxx"

#include "cblas.h"
#include "lapacke.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

using namespace nbd;

void nbd::cMatrix(Matrix& mat, int64_t m, int64_t n) {
  mat.A.resize(m * n);
  mat.M = m;
  mat.N = n;
}

void nbd::cVector(Vector& vec, int64_t n) {
  vec.X.resize(n);
  vec.N = n;
}

void nbd::cpyFromMatrix(char trans, const Matrix& A, double* V) {
  int64_t iv = A.M;
  int64_t incv = 1;
  if (trans == 'T' || trans == 't') {
    iv = 1;
    incv = A.N;
  }
  for (int64_t j = 0; j < A.N; j++)
    cblas_dcopy(A.M, &A.A[j * A.M], 1, &V[j * iv], incv);
}

void nbd::cpyFromVector(const Vector& A, double* v) {
  std::copy(A.X.data(), A.X.data() + A.N, v);
}

void nbd::maxpby(Matrix& A, const double* v, double alpha, double beta) {
  int64_t size = A.M * A.N;
  if (beta == 0.)
    std::fill(A.A.data(), A.A.data() + size, 0.);
  else if (beta != 1.)
    cblas_dscal(size, beta, A.A.data(), 1);
  cblas_daxpy(size, alpha, v, 1, A.A.data(), 1);
}

void nbd::vaxpby(Vector& A, const double* v, double alpha, double beta) {
  int64_t size = A.N;
  if (beta == 0.)
    std::fill(A.X.data(), A.X.data() + size, 0.);
  else if (beta != 1.)
    cblas_dscal(size, beta, A.X.data(), 1);
  cblas_daxpy(size, alpha, v, 1, A.X.data(), 1);
}

void nbd::cpyMatToMat(int64_t m, int64_t n, const Matrix& m1, Matrix& m2, int64_t y1, int64_t x1, int64_t y2, int64_t x2) {
  for (int64_t j = 0; j < n; j++) {
    int64_t j1 = y1 + (x1 + j) * m1.M;
    int64_t j2 = y2 + (x2 + j) * m2.M;
    std::copy(&m1.A[j1], &m1.A[j1 + m], &m2.A[j2]);
  }
}

void nbd::cpyVecToVec(int64_t n, const Vector& v1, Vector& v2, int64_t x1, int64_t x2) {
  std::copy(&v1.X[x1], &v1.X[x1 + n], &v2.X[x2]);
}

void nbd::orthoBase(double repi, Matrix& A, int64_t *rnk_out) {
  Vector S, superb;
  cVector(S, A.M);
  cVector(superb, A.M - 1);

  int info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'O', 'N', A.M, A.N, A.A.data(), A.M, S.X.data(), nullptr, A.M, nullptr, A.N, superb.X.data());
  if (info)
    fprintf(stderr, "SVD FAIL: %d on %ld %ld %e\n", info, A.M, A.N, cblas_dnrm2(A.M * A.N, A.A.data(), 1));

  int64_t rank;
  if (repi < 1.) {
    double sepi = S.X[0] * repi;
    int64_t lim = (int64_t)(A.M - 1);
    rank = std::distance(S.X.data(), std::find_if(S.X.data() + 1, S.X.data() + lim, [sepi](double& s) { return s < sepi; }));
  }
  else
    rank = (int64_t)std::floor(repi);
  *rnk_out = rank;
}

void nbd::zeroMatrix(Matrix& A) {
  std::fill(A.A.data(), A.A.data() + A.M * A.N, 0.);
}

void nbd::mmult(char ta, char tb, const Matrix& A, const Matrix& B, Matrix& C, double alpha, double beta) {
  int64_t k = (ta == 'N' || ta == 'n') ? A.N : A.M;
  auto tac = (ta == 'N' || ta == 'n') ? CblasNoTrans : CblasTrans;
  auto tbc = (tb == 'N' || tb == 'n') ? CblasNoTrans : CblasTrans;
  cblas_dgemm(CblasColMajor, tac, tbc, C.M, C.N, k, alpha, A.A.data(), A.M, B.A.data(), B.M, beta, C.A.data(), C.M);
}


void nbd::msample(char ta, int64_t lenR, const Matrix& A, const double* R, Matrix& C) {
  int64_t k = (ta == 'N' || ta == 'n') ? A.N : A.M;
  auto tac = (ta == 'N' || ta == 'n') ? CblasNoTrans : CblasTrans;
  if (lenR >= C.N * k)
    cblas_dgemm(CblasColMajor, tac, CblasNoTrans, C.M, C.N, k, 1., A.A.data(), A.M, R, k, 1., C.A.data(), C.M);
  else
    fprintf(stderr, "Insufficent random vector: %ld needed %ld provided\n", k * C.N, lenR);
}

void nbd::msample_m(char ta, const Matrix& A, const Matrix& B, Matrix& C) {
  int64_t k = (ta == 'N' || ta == 'n') ? A.N : A.M;
  int64_t nrhs = std::min(B.N, C.N);
  auto tac = (ta == 'N' || ta == 'n') ? CblasNoTrans : CblasTrans;
  cblas_dgemm(CblasColMajor, tac, CblasNoTrans, C.M, nrhs, k, 1., A.A.data(), A.M, B.A.data(), B.M, 1., C.A.data(), C.M);
}

void nbd::minvl(Matrix& A, Matrix& B) {
  Vector tau;
  cVector(tau, std::min(A.M, A.N));
  LAPACKE_dgeqrf(LAPACK_COL_MAJOR, A.M, A.N, A.A.data(), A.M, tau.X.data());
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, A.M, A.N, 1., A.A.data(), A.M, B.A.data(), B.M);
  LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', 'T', B.M, B.N, tau.N, A.A.data(), A.M, tau.X.data(), B.A.data(), B.M);
}

void nbd::chol_decomp(Matrix& A) {
  LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', A.M, A.A.data(), A.M);
}

void nbd::trsm_lowerA(Matrix& A, const Matrix& L) {
  cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, A.M, A.N, 1., L.A.data(), L.M, A.A.data(), A.M);
}

void nbd::utav(const Matrix& U, const Matrix& A, const Matrix& VT, Matrix& C) {
  Matrix work;
  cMatrix(work, C.M, A.N);
  mmult('T', 'N', U, A, work, 1., 0.);
  mmult('N', 'N', work, VT, C, 1., 0.);
}

void nbd::axat(Matrix& A, Matrix& AT) {
  for (int64_t j = 0; j < A.N; j++)
    cblas_daxpy(A.M, 1., &AT.A[j], AT.M, &A.A[j * A.M], 1);
  for (int64_t i = 0; i < A.M; i++)
    cblas_dcopy(A.N, &A.A[i], A.M, &AT.A[i * AT.M], 1);
}

void nbd::madd(Matrix& A, const Matrix& B) {
  int64_t size = A.M * A.N;
  cblas_daxpy(size, 1., &B.A[0], 1, &A.A[0], 1);
}

void nbd::chol_solve(Vector& X, const Matrix& A) {
  fw_solve(X, A);
  bk_solve(X, A);
}

void nbd::fw_solve(Vector& X, const Matrix& L) {
  cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit, X.N, L.A.data(), L.M, X.X.data(), 1);
}

void nbd::bk_solve(Vector& X, const Matrix& L) {
  cblas_dtrsv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit, X.N, L.A.data(), L.M, X.X.data(), 1);
}

void nbd::mvec(char ta, const Matrix& A, const Vector& X, Vector& B, double alpha, double beta) {
  auto tac = (ta == 'N' || ta == 'n') ? CblasNoTrans : CblasTrans;
  cblas_dgemv(CblasColMajor, tac, A.M, A.N, alpha, A.A.data(), A.M, X.X.data(), 1, beta, B.X.data(), 1);
}

void nbd::pvc_fw(const Vector& X, const Matrix& Us, const Matrix& Uc, Vector& Xs, Vector& Xc) {
  cVector(Xc, Uc.N);
  cVector(Xs, Us.N);
  mvec('T', Uc, X, Xc, 1., 0.);
  mvec('T', Us, X, Xs, 1., 0.);
}

void nbd::pvc_bk(const Vector& Xs, const Vector& Xc, const Matrix& Us, const Matrix& Uc, Vector& X) {
  mvec('N', Uc, Xc, X, 1., 0.);
  mvec('N', Us, Xs, X, 1., 1.);
}

void nbd::nrm2(const Matrix& A, double* nrm) {
  int64_t size = A.M * A.N;
  *nrm = cblas_dnrm2(size, A.A.data(), 1);
}

