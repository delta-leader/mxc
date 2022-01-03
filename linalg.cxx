

#include "linalg.hxx"

#include "cblas.h"
#include "lapacke.h"

#include <algorithm>
#include <iostream>
#include <cstdlib>

using namespace nbd;

void dlra(double epi, int64_t m, int64_t n, int64_t k, double* a, int64_t lda, double* u, int64_t ldu, double* vt, int64_t ldvt, int64_t* rank) {
  double nrm = 0.;
  double epi2 = epi * epi;

  for (int64_t i = 0; i < k; i++) {
    double amax = 0.;
    int64_t ymax = 0;
    for (int64_t x = 0; x < n; x++) {
      int64_t ybegin = x * lda;
      int64_t yend = m + x * lda;
      for (int64_t y = ybegin; y < yend; y++) {
        double fa = fabs(a[y]);
        if (fa > amax) {
          amax = fa;
          ymax = y;
        }
      }
    }

    if (amax == 0.) {
      *rank = i;
      return;
    }

    int64_t xp = ymax / lda;
    int64_t yp = ymax - xp * lda;
    double ap = 1. / a[ymax];
    double* ui = u + i * ldu;
    double* vi = vt + i * ldvt;

    for (int64_t x = 0; x < n; x++) {
      double ax = a[yp + x * lda];
      vi[x] = ax;
      a[yp + x * lda] = 0.;
    }

    for (int64_t y = 0; y < m; y++) {
      double ay = a[y + xp * lda];
      ui[y] = ay * ap;
      a[y + xp * lda] = 0.;
    }
    ui[yp] = 1.;

    for (int64_t x = 0; x < n; x++) {
      if (x == xp)
        continue;
      double ri = vi[x];
      for (int64_t y = 0; y < m; y++) {
        if (y == yp)
          continue;
        double lf = ui[y];
        double e = a[y + x * lda];
        e = e - lf * ri;
        a[y + x * lda] = e;
      }
    }

    if (epi2 > 0.) {
      double nrm_v = 0.;
      double nrm_vi = 0.;
      for (int64_t x = 0; x < n; x++) {
        double vx = vi[x];
        nrm_vi = nrm_vi + vx * vx;
        for (int64_t j = 0; j < i; j++) {
          double vj = vt[x + j * ldvt];
          nrm_v = nrm_v + vx * vj;
        }
      }

      double nrm_u = 0.;
      double nrm_ui = 0.;
      for (int64_t y = 0; y < m; y++) {
        double uy = ui[y];
        nrm_ui = nrm_ui + uy * uy;
        for (int64_t j = 0; j < i; j++) {
          double uj = u[y + j * ldu];
          nrm_u = nrm_u + uy * uj;
        }
      }

      double n2 = nrm_ui * nrm_vi;
      nrm = nrm + 2. * nrm_u * nrm_v + n2;
      if (n2 <= epi2 * nrm) {
        *rank = i;
        return;
      }
    }

  }

  *rank = k;
  if (epi2 > 0.)
    fprintf(stderr, "LRA reached full iterations.\n");
}

void dorth(char ecoq, int64_t m, int64_t n, double* r, int64_t ldr, double* q, int64_t ldq) {
  int64_t k = m < n ? m : n;
  double* TAU = (double*)malloc(sizeof(double) * k);

  for (int64_t x = 0; x < k; x++) {
    double nrmx = 0.;
    for (int64_t y = x; y < m; y++) {
      double e = r[y + x * ldr];
      nrmx = nrmx + e * e;
    }
    nrmx = sqrt(nrmx);

    double rx = r[x + x * ldr];
    double s = rx > 0 ? -1. : 1.;
    double u1 = rx - s * nrmx;
    double tau = -s * u1 / nrmx;
    TAU[x] = tau;

    double iu1 = 1. / u1;
    for (int64_t y = x + 1; y < m; y++)
      r[y + x * ldr] = r[y + x * ldr] * iu1;
    r[x + x * ldr] = s * nrmx;

    for (int64_t xx = x + 1; xx < n; xx++) {
      double wdr = 0.;
      for (int64_t y = x; y < m; y++) {
        double e1 = y == x ? 1. : r[y + x * ldr];
        double e2 = r[y + xx * ldr];
        wdr = wdr + e1 * e2;
      }

      wdr = wdr * tau;
      for (int64_t y = x; y < m; y++) {
        double e1 = y == x ? 1. : r[y + x * ldr];
        double e2 = r[y + xx * ldr];
        r[y + xx * ldr] = e2 - e1 * wdr;
      }
    }
  }

  int64_t nq = (ecoq == 'F' || ecoq == 'f') ? m : k;

  for (int64_t x = 0; x < nq; x++) {
    for (int64_t y = 0; y < m; y++)
      q[y + x * ldq] = 0.;
    q[x + x * ldq] = 1.;
  }

  for (int64_t kk = k - 1; kk >= 0; kk--) {
    for (int64_t x = 0; x < nq; x++) {
      double wdq = 0.;
      for (int64_t y = kk; y < m; y++) {
        double e1 = y == kk ? 1. : r[y + kk * ldr];
        double e2 = q[y + x * ldq];
        wdq = wdq + e1 * e2;
      }

      wdq = wdq * TAU[kk];
      for (int64_t y = kk; y < m; y++) {
        double e1 = y == kk ? 1. : r[y + kk * ldr];
        double e2 = q[y + x * ldq];
        q[y + x * ldq] = e2 - e1 * wdq;
      }
    }
  }

  free(TAU);
}

void dpotrf(int64_t n, double* a, int64_t lda) {
  for (int64_t i = 0; i < n; i++) {
    double p = a[i + i * lda];
    if (p <= 0.)
    { fprintf(stderr, "A is not positive-definite.\n"); return; }
    p = sqrt(p);
    a[i + i * lda] = p;
    double invp = 1. / p;

    for (int64_t j = i + 1; j < n; j++)
      a[j + i * lda] = a[j + i * lda] * invp;
    
    for (int64_t k = i + 1; k < n; k++) {
      double c = a[k + i * lda];
      a[k + k * lda] = a[k + k * lda] - c * c;
      for (int64_t j = k + 1; j < n; j++) {
        double r = a[j + i * lda];
        a[j + k * lda] = a[j + k * lda] - r * c;
      }
    }
  }
}

void dtrsmlt_right(int64_t m, int64_t n, const double* a, int64_t lda, double* b, int64_t ldb) {
  for (int64_t i = 0; i < n; i++) {
    double p = a[i + i * lda];
    double invp = 1. / p;

    for (int64_t j = 0; j < m; j++)
      b[j + i * ldb] = b[j + i * ldb] * invp;
    
    for (int64_t k = i + 1; k < n; k++) {
      double c = a[k + i * lda];
      for (int64_t j = 0; j < m; j++) {
        double r = b[j + i * ldb];
        b[j + k * ldb] = b[j + k * ldb] - r * c;
      }
    }
  }
}

void dtrsml_left(int64_t m, int64_t n, const double* a, int64_t lda, double* b, int64_t ldb) {
  for (int64_t i = 0; i < m; i++) {
    double p = a[i + i * lda];
    double invp = 1. / p;

    for (int64_t j = 0; j < n; j++)
      b[i + j * ldb] = b[i + j * ldb] * invp;
    
    for (int64_t k = i + 1; k < m; k++) {
      double r = a[k + i * lda];
      for (int64_t j = 0; j < n; j++) {
        double c = b[i + j * ldb];
        b[k + j * ldb] = b[k + j * ldb] - r * c;
      }
    }
  }
}

void dtrsmlt_left(int64_t m, int64_t n, const double* a, int64_t lda, double* b, int64_t ldb) {
  for (int64_t i = m - 1; i >= 0; i--) {
    double p = a[i + i * lda];
    double invp = 1. / p;

    for (int64_t j = 0; j < n; j++)
      b[i + j * ldb] = b[i + j * ldb] * invp;
    
    for (int64_t k = 0; k < i; k++) {
      double r = a[i + k * lda];
      for (int64_t j = 0; j < n; j++) {
        double c = b[i + j * ldb];
        b[k + j * ldb] = b[k + j * ldb] - r * c;
      }
    }
  }
}

void dgemv(char ta, int64_t m, int64_t n, double alpha, const double* a, int64_t lda, const double* x, int64_t incx, double beta, double* y, int64_t incy) {
  auto tac = (ta == 'N' || ta == 'n') ? CblasNoTrans : CblasTrans;
  cblas_dgemv(CblasColMajor, tac, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void dgemm(char ta, char tb, int64_t m, int64_t n, int64_t k, double alpha, const double* a, int64_t lda, const double* b, int64_t ldb, double beta, double* c, int64_t ldc) {
  int64_t ma = m;
  int64_t na = k;
  if (ta == 'T' || ta == 't') {
    ma = k;
    na = m;
  }

  if (tb == 'T' || tb == 't')
    for (int64_t i = 0; i < n; i++)
      dgemv(ta, ma, na, alpha, a, lda, b + i, ldb, beta, c + i * ldc, 1);
  else if (tb == 'N' || tb == 'n')
    for (int64_t i = 0; i < n; i++)
      dgemv(ta, ma, na, alpha, a, lda, b + i * ldb, 1, beta, c + i * ldc, 1);
}

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
  bool prec = repi < 1.;
  Matrix U, V;
  int64_t rank = prec ? std::min(A.M, A.N) : (int64_t)std::floor(repi);
  cMatrix(U, A.M, rank);
  cMatrix(V, A.N, rank);

  dlra(prec ? repi : 0., A.M, A.N, rank, A.A.data(), A.M, U.A.data(), A.M, V.A.data(), A.N, rnk_out);
  rank = *rnk_out;

  if (A.N < A.M)
    cMatrix(A, A.M, A.M);
  dorth('F', A.M, rank, U.A.data(), A.M, A.A.data(), A.M);
}

void nbd::zeroMatrix(Matrix& A) {
  std::fill(A.A.data(), A.A.data() + A.M * A.N, 0.);
}

void nbd::zeroVector(Vector& A) {
  std::fill(A.X.data(), A.X.data() + A.N, 0.);
}

void nbd::mmult(char ta, char tb, const Matrix& A, const Matrix& B, Matrix& C, double alpha, double beta) {
  int64_t k = (ta == 'N' || ta == 'n') ? A.N : A.M;
  /*auto tac = (ta == 'N' || ta == 'n') ? CblasNoTrans : CblasTrans;
  auto tbc = (tb == 'N' || tb == 'n') ? CblasNoTrans : CblasTrans;
  cblas_dgemm(CblasColMajor, tac, tbc, C.M, C.N, k, alpha, A.A.data(), A.M, B.A.data(), B.M, beta, C.A.data(), C.M);*/
  dgemm(ta, tb, C.M, C.N, k, alpha, A.A.data(), A.M, B.A.data(), B.M, beta, C.A.data(), C.M);
}

void nbd::msample(char ta, int64_t lenR, const Matrix& A, const double* R, Matrix& C) {
  if (lenR < C.N * 100) { 
    std::cerr << "Insufficent random vector: " << C.N << " x 100 needed " << lenR << " provided." << std::endl;
    return;
  }
  bool noTransA = (ta == 'N' || ta == 'n');
  int64_t k = A.M;
  int64_t inca = 1;
  auto tac = CblasTrans;
  if (noTransA) {
    k = A.N;
    tac = CblasNoTrans;
    inca = A.M;
  }

  int64_t rk = lenR / C.N;
  int64_t lk = k % rk;
  if (lk > 0)
    cblas_dgemm(CblasColMajor, tac, CblasNoTrans, C.M, C.N, lk, 1., A.A.data(), A.M, R, lk, 1., C.A.data(), C.M);
  if (k > rk)
    for (int64_t i = lk; i < k; i += rk)
      cblas_dgemm(CblasColMajor, tac, CblasNoTrans, C.M, C.N, rk, 1., &A.A[i * inca], A.M, R, rk, 1., C.A.data(), C.M);
}

void nbd::msample_m(char ta, const Matrix& A, const Matrix& B, Matrix& C) {
  bool noTransA = (ta == 'N' || ta == 'n');
  int64_t k = A.M;
  auto tac = CblasTrans;
  if (noTransA) {
    k = A.N;
    tac = CblasNoTrans;
  }
  int64_t nrhs = std::min(B.N, C.N);
  cblas_dgemm(CblasColMajor, tac, CblasNoTrans, C.M, nrhs, k, 1., A.A.data(), A.M, B.A.data(), B.M, 1., C.A.data(), C.M);
}

void nbd::minvl(const Matrix& A, Matrix& B) {
  Matrix work;
  cMatrix(work, A.M, A.N);
  cpyMatToMat(A.M, A.N, A, work, 0, 0, 0, 0);
  chol_decomp(work);
  dtrsml_left(B.M, B.N, work.A.data(), A.M, B.A.data(), B.M);
  dtrsmlt_left(B.M, B.N, work.A.data(), A.M, B.A.data(), B.M);
}

void nbd::chol_decomp(Matrix& A) {
  dpotrf(A.M, A.A.data(), A.M);
}

void nbd::trsm_lowerA(Matrix& A, const Matrix& L) {
  dtrsmlt_right(A.M, A.N, L.A.data(), L.M, A.A.data(), A.M);
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
  dtrsml_left(X.N, 1, L.A.data(), L.M, X.X.data(), X.N);
}

void nbd::bk_solve(Vector& X, const Matrix& L) {
  dtrsmlt_left(X.N, 1, L.A.data(), L.M, X.X.data(), X.N);
}

void nbd::mvec(char ta, const Matrix& A, const Vector& X, Vector& B, double alpha, double beta) {
  /*auto tac = (ta == 'N' || ta == 'n') ? CblasNoTrans : CblasTrans;
  cblas_dgemv(CblasColMajor, tac, A.M, A.N, alpha, A.A.data(), A.M, X.X.data(), 1, beta, B.X.data(), 1);*/
  dgemv(ta, A.M, A.N, alpha, A.A.data(), A.M, X.X.data(), 1, beta, B.X.data(), 1);
}

void nbd::pvc_fw(const Vector& X, const Matrix& Us, const Matrix& Uc, Vector& Xs, Vector& Xc) {
  mvec('T', Uc, X, Xc, 1., 0.);
  mvec('T', Us, X, Xs, 1., 0.);
}

void nbd::pvc_bk(const Vector& Xs, const Vector& Xc, const Matrix& Us, const Matrix& Uc, Vector& X) {
  mvec('N', Uc, Xc, X, 1., 0.);
  mvec('N', Us, Xs, X, 1., 1.);
}

void nbd::vnrm2(const Vector& A, double* nrm) {
  *nrm = cblas_dnrm2(A.N, A.X.data(), 1);
}

void nbd::verr2(const Vector& A, const Vector& B, double* err) {
  Vector work;
  cVector(work, A.N);
  vaxpby(work, A.X.data(), 1., 0.);
  vaxpby(work, B.X.data(), -1., 1.);
  vnrm2(work, err);
}
