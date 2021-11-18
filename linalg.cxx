

#include "linalg.hxx"

#include "cblas.h"
#include "lapacke.h"
#include <cstdlib>
#include <algorithm>

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

int64_t nbd::cpyFromMatrix(const Matrix& mat, double* m) {
  int64_t size = mat.M * mat.N;
  if (m != nullptr && size > 0)
    std::copy(&mat.A[0], &mat.A[size], m);
  return size;
}

int64_t nbd::cpyToMatrix(Matrix& mat, const double* m) {
  int64_t size = mat.M * mat.N;
  if (m != nullptr && size > 0)
    std::copy(&m[0], &m[size], &mat.A[0]);
  return size;
}

int64_t nbd::cpyFromVector(const Vector& vec, double* v) {
  int64_t size = vec.N;
  if (v != nullptr && size > 0)
    std::copy(&vec.X[0], &vec.X[size], v);
  return size;
}

int64_t nbd::cpyToVector(Vector& vec, const double* v) {
  int64_t size = vec.N;
  if (v != nullptr && size > 0)
    std::copy(&v[0], &v[size], &vec.X[0]);
  return size;
}

void nbd::cpyMatToMat(int64_t m, int64_t n, const Matrix& m1, Matrix& m2, int64_t y1, int64_t x1, int64_t y2, int64_t x2) {
  for (int64_t j = 0; j < n; j++) {
    int64_t j1 = y1 + (x1 + j) * m1.M;
    int64_t j2 = y2 + (x2 + j) * m2.M;
    std::copy(&m1.A[j1], &m1.A[j1 + m], &m2.A[j2]);
  }
}

int64_t nbd::orthoBase(double repi, Matrix& A, Matrix& Us, Matrix& Uc) {
  if (A.N < A.M)
    return -1;
  Vector S;
  Vector superb;
  cVector(S, A.M);
  cVector(superb, A.M - 1);

  LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'O', 'N', A.M, A.N, A.A.data(), A.M, S.X.data(), nullptr, A.M, nullptr, A.N, superb.X.data());
  int64_t rank;
  if (repi < 1.) {
    double sepi = S.X[0] * repi;
    rank = std::distance(S.X.data(), std::find_if(S.X.data() + 1, S.X.data() + A.M, [sepi](double& s) { return s < sepi; }));
  }
  else
    rank = (int64_t)std::floor(repi);

  cMatrix(Us, A.M, rank);
  cMatrix(Uc, A.M, A.M - rank);
  cpyMatToMat(A.M, rank, A, Us, 0, 0, 0, 0);
  cpyMatToMat(A.M, A.M - rank, A, Uc, 0, rank, 0, 0);
  return rank;
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
  int64_t colR = std::min(C.N, lenR / k);
  for (int64_t i = 0; i < C.N; i += colR) {
    int64_t nrhs = std::min(colR, C.N - i);
    cblas_dgemm(CblasColMajor, tac, CblasNoTrans, C.M, nrhs, k, 1., A.A.data(), A.M, R, k, 1., &C.A[i * C.M], C.M);
  }
}

void nbd::msample_m(char ta, const Matrix& A, const Matrix& B, Matrix& C) {
  int64_t k = (ta == 'N' || ta == 'n') ? A.N : A.M;
  int64_t nrhs = std::min(B.N, C.N);
  auto tac = (ta == 'N' || ta == 'n') ? CblasNoTrans : CblasTrans;
  cblas_dgemm(CblasColMajor, tac, CblasNoTrans, C.M, nrhs, k, 1., A.A.data(), A.M, B.A.data(), B.M, 1., C.A.data(), C.M);
}

void nbd::minv(char ta, char lr, Matrix& A, Matrix& B) {
  Vector tau;
  cVector(tau, std::min(A.M, A.N));

  LAPACKE_dgeqrf(LAPACK_COL_MAJOR, A.M, A.N, A.A.data(), A.M, tau.X.data());
  auto tac = (ta == 'N' || ta == 'n') ? CblasNoTrans : CblasTrans;
  char tact = (ta == 'N' || ta == 'n') ? 'T' : 'N';
  if (lr == 'L' || lr == 'l') {
    LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', tact, B.M, B.N, tau.N, A.A.data(), A.M, tau.X.data(), B.A.data(), B.M);
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, tac, CblasNonUnit, B.M, B.N, 1., A.A.data(), A.M, B.A.data(), B.M);
  }
  else if (lr == 'R' || lr == 'r') {
    cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, tac, CblasNonUnit, B.M, B.N, 1., A.A.data(), A.M, B.A.data(), B.M);
    LAPACKE_dormqr(LAPACK_COL_MAJOR, 'R', tact, B.M, B.N, tau.N, A.A.data(), A.M, tau.X.data(), B.A.data(), B.M);
  }
}

void nbd::lu_decomp(Matrix& A) {
  int64_t m = A.M;
  int64_t n = A.N;
  int64_t lda = A.M;
  int64_t k = std::min(m, n);
  double* a = A.A.data();

  for (int64_t i = 0; i < k; i++) {
    double p = 1. / a[i + i * lda];
    int64_t mi = m - i - 1;
    int64_t ni = n - i - 1;

    double* ax = a + i + i * lda + 1;
    double* ay = a + i + i * lda + lda;
    double* an = ay + 1;

    cblas_dscal(mi, p, ax, 1);
    cblas_dger(CblasColMajor, mi, ni, -1., ax, 1, ay, lda, an, lda);
  }
}

void nbd::trsm_lowerA(Matrix& A, const Matrix& U) {
  cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, A.M, A.N, 1., U.A.data(), U.M, A.A.data(), A.M);
}

void nbd::trsm_upperA(Matrix& A, const Matrix& L) {
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, A.M, A.N, 1., L.A.data(), L.M, A.A.data(), A.M);
}

void nbd::utav(const Matrix& U, const Matrix& A, const Matrix& VT, Matrix& C) {
  Matrix work;
  cMatrix(work, U.N, A.N);
  cMatrix(C, U.N, VT.N);

  mmult('T', 'N', U, A, work, 1., 0.);
  mmult('N', 'N', work, VT, C, 1., 0.);
}

void nbd::lu_solve(Vector& X, const Matrix& A) {
  fw_solve(X, A);
  bk_solve(X, A);
}

void nbd::fw_solve(Vector& X, const Matrix& L) {
  cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasUnit, X.N, L.A.data(), L.M, X.X.data(), 1);
}

void nbd::bk_solve(Vector& X, const Matrix& U) {
  cblas_dtrsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, X.N, U.A.data(), U.M, X.X.data(), 1);
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

void nbd::lookupIJ(int64_t& ij, const CSC& rels, int64_t i, int64_t j) {
  if (j < 0 || j >= rels.N)
  { ij = -1; return; }
  int64_t k = std::distance(rels.CSC_ROWS.data(), 
    std::find(rels.CSC_ROWS.data() + rels.CSC_COLS[j], rels.CSC_ROWS.data() + rels.CSC_COLS[j + 1], i));
  if (k < rels.CSC_COLS[j + 1])
    ij = k;
}

void nbd::toCSR(CSR& rels_csr, const CSC& rels_csc) {
  rels_csr.M = rels_csc.M;
  rels_csr.N = rels_csc.N;
  rels_csr.NNZ = rels_csc.NNZ;

  rels_csr.CSR_ROWS.resize(rels_csr.M + 1);
  rels_csr.CSR_COLS.resize(rels_csr.NNZ);

  std::vector<int64_t> counts(rels_csr.M);
  std::fill(counts.begin(), counts.end(), 0);
  for (int64_t j = 0; j < rels_csc.N; j++)
    for (int64_t ij = rels_csc.CSC_COLS[j]; ij < rels_csc.CSC_COLS[j + 1]; ij++) {
      int64_t i = rels_csc.CSC_ROWS[ij];
      counts[i] = counts[i] + 1;
    }

  rels_csr.CSR_ROWS[0] = 0;
  for (int64_t i = 1; i <= rels_csr.M; i++)
    rels_csr.CSR_ROWS[i] = rels_csr.CSR_ROWS[i - 1] + counts[i - 1];
  std::fill(counts.begin(), counts.end(), 0);

  for (int64_t j = 0; j < rels_csc.N; j++)
    for (int64_t ij = rels_csc.CSC_COLS[j]; ij < rels_csc.CSC_COLS[j + 1]; ij++) {
      int64_t i = rels_csc.CSC_ROWS[ij];
      int64_t ci = counts[i];
      int64_t loc_i = ci + rels_csr.CSR_ROWS[i];
      rels_csr.CSR_COLS[loc_i] = j;
      counts[i] = ci + 1;
    }
}


void nbd::cVectors(Vectors& Xs, int64_t n, const int64_t* dims) {
  Xs.resize(n);
  for (int64_t i = 0; i < n; i++)
    cVector(Xs[i], dims[i]);
}

int64_t nbd::ctoVectors(Vectors& Xs, const double* X) {
  const double* Xi = X;
  for (int64_t i = 0; i < Xs.size(); i++) {
    int64_t dim = Xs[i].N;
    std::copy(Xi, Xi + dim, Xs[i].X.data());
    Xi = Xi + dim;
  }
  return std::distance(X, Xi);
}

int64_t nbd::cbkVectors(double* X, const Vectors& Xs) {
  double* Xi = X;
  for (int64_t i = 0; i < Xs.size(); i++) {
    int64_t dim = Xs[i].N;
    std::copy(Xs[i].X.data(), Xs[i].X.data() + dim, Xi);
    Xi = Xi + dim;
  }
  return std::distance(X, Xi);
}

void nbd::cpsVectors(char updwn, const Vectors& Xs, Vectors& Xt) {
  if (updwn == 'U' || updwn == 'u') {
    for (int64_t i = 0; i < Xt.size(); i++) {
      const Vector& c1 = Xs[i << 1];
      const Vector& c2 = Xs[(i << 1) + 1];
      std::copy(&c1.X[0], &c1.X[c1.N], &Xt[i].X[0]);
      std::copy(&c2.X[0], &c2.X[c2.N], &Xt[i].X[c1.N]);
    }
  }
  else if (updwn == 'D' || updwn == 'd') {
    for (int64_t i = 0; i < Xs.size(); i++) {
      Vector& c1 = Xt[i << 1];
      Vector& c2 = Xt[(i << 1) + 1];
      std::copy(&Xs[i].X[0], &Xs[i].X[c1.N], &c1.X[0]);
      std::copy(&Xs[i].X[c1.N], &Xs[i].X[c1.N + c2.N], &c2.X[0]);
    }
  }
}


void nbd::cMatrices(Matrices& Ms, const CSC& rels, const int64_t* Ydims, const int64_t* Xdims) {
  Ms.resize(rels.NNZ);
  for (int64_t j = 0; j < rels.N; j++)
    for (int64_t ij = rels.CSC_COLS[j]; ij < rels.CSC_COLS[j + 1]; ij++) {
      int64_t i = rels.CSC_ROWS[ij];
      cMatrix(Ms[ij], Ydims[i], Xdims[j]);
    }
}

int64_t nbd::ctoMatrices(Matrices& Ms, const double* M) {
  const double* Mi = M;
  for (int64_t i = 0; i < Ms.size(); i++) {
    int64_t y = Ms[i].M;
    int64_t x = Ms[i].N;
    std::copy(Mi, Mi + y * x, Ms[i].A.data());
    Mi = Mi + y * x;
  }
  return std::distance(M, Mi);
}

int64_t nbd::cbkMatrices(double* M, const Matrices& Ms) {
  double* Mi = M;
  for (int64_t i = 0; i < Ms.size(); i++) {
    int64_t y = Ms[i].M;
    int64_t x = Ms[i].N;
    std::copy(Ms[i].A.data(), Ms[i].A.data() + y * x, Mi);
    Mi = Mi + y * x;
  }
  return std::distance(M, Mi);
}

void nbd::cpsMatrices(Matrices& Mup, const CSC& rels_up, const Matrices& Mlow, const CSC& rels_low) {
  for (int64_t j = 0; j < rels_up.N; j++)
    for (int64_t ij = rels_up.CSC_COLS[j]; ij < rels_up.CSC_COLS[j + 1]; ij++) {
      int64_t i = rels_up.CSC_ROWS[ij];
      zeroMatrix(Mup[ij]);
      
      int64_t i00, i01, i10, i11;
      lookupIJ(i00, rels_low, i << 1, j << 1);
      lookupIJ(i01, rels_low, i << 1, (j << 1) + 1);
      lookupIJ(i10, rels_low, (i << 1) + 1, j << 1);
      lookupIJ(i11, rels_low, (i << 1) + 1, (j << 1) + 1);

      if (i00 > 0) {
        const Matrix& m00 = Mlow[i00];
        cpyMatToMat(m00.M, m00.N, m00, Mup[ij], 0, 0, 0, 0);
      }

      if (i01 > 0) {
        const Matrix& m01 = Mlow[i01];
        cpyMatToMat(m01.M, m01.N, m01, Mup[ij], 0, 0, 0, Mup[ij].N - m01.N);
      }

      if (i10 > 0) {
        const Matrix& m10 = Mlow[i10];
        cpyMatToMat(m10.M, m10.N, m10, Mup[ij], 0, 0, Mup[ij].M - m10.M, 0);
      }

      if (i11 > 0) {
        const Matrix& m11 = Mlow[i11];
        cpyMatToMat(m11.M, m11.N, m11, Mup[ij], 0, 0, Mup[ij].M - m11.M, Mup[ij].N - m11.N);
      }
    }
}

