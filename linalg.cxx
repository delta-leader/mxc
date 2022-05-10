

#include "linalg.hxx"
#include "minblas.h"

#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstdlib>

void dlra(double epi, int64_t m, int64_t n, int64_t k, double* a, double* u, int64_t ldu, double* vt, int64_t ldvt, int64_t* rank, int64_t* piv) {
  double nrm = 0.;
  double epi2 = epi * epi;
  double n2 = 1.;
  double amax = 1.;

  int64_t i = 0;
  while (i < k && (n2 > epi2 * nrm) && amax > 0) {
    amax = 0.;
    int64_t ymax = 0;
    Cidamax(m * n, a, 1, &ymax);
    amax = fabs(a[ymax]);

    if (amax > 0.) {
      if (piv != NULL)
        piv[i] = ymax;
      int64_t xp = ymax / m;
      int64_t yp = ymax - xp * m;
      double ap = 1. / a[ymax];
      double* ui = &u[i * ldu];
      double* vi = &vt[i * ldvt];

      Cdcopy(n, &a[yp], m, vi, 1);
      Cdcopy(m, &a[xp * m], 1, ui, 1);
      Cdscal(m, ap, ui, 1);
      ui[yp] = 1.;

      for (int64_t x = 0; x < n; x++) {
        double ri = -vi[x];
        Cdaxpy(m, ri, ui, 1, &a[x * m], 1);
      }

      double zero = 0.;
      Cdcopy(n, &zero, 0, &a[yp], m);
      Cdcopy(m, &zero, 0, &a[xp * m], 1);

      if (epi2 > 0.) {
        double nrm_v = 0.;
        double nrm_u = 0.;
        double nrm_vi = 0.;
        double nrm_ui = 0.;

        Cddot(n, vi, 1, vi, 1, &nrm_vi);
        Cddot(m, ui, 1, ui, 1, &nrm_ui);

        for (int64_t j = 0; j < i; j++) {
          double* vj = &vt[j * ldvt];
          double* uj = &u[j * ldu];
          double nrm_vj = 0.;
          double nrm_uj = 0.;

          Cddot(n, vi, 1, vj, 1, &nrm_vj);
          Cddot(m, ui, 1, uj, 1, &nrm_uj);
          nrm_v = nrm_v + nrm_vj;
          nrm_u = nrm_u + nrm_uj;
        }

        n2 = nrm_ui * nrm_vi;
        nrm = nrm + 2. * nrm_u * nrm_v + n2;
      }
      i++;
    }
  }

  *rank = i;
}


void dorth(char ecoq, int64_t m, int64_t n, double* r, int64_t ldr, double* q, int64_t ldq, double* TAU) {
  int64_t k = m < n ? m : n;
  int64_t nq = (ecoq == 'F' || ecoq == 'f') ? m : k;

  for (int64_t x = 0; x < k; x++) {
    double nrmx = 0.;
    Cdnrm2(m - x, &r[x + x * ldr], 1, &nrmx);

    double rx = r[x + x * ldr];
    double s = rx > 0 ? -1. : 1.;
    double u1 = rx - s * nrmx;
    double tau = -s * u1 / nrmx;
    TAU[x] = tau;

    double iu1 = 1. / u1;
    Cdscal(m - x - 1, iu1, &r[x + 1 + x * ldr], 1);
    r[x + x * ldr] = s * nrmx;

    for (int64_t xx = x + 1; xx < n; xx++) {
      double wdr = 0.;
      Cddot(m - x - 1, &r[x + 1 + x * ldr], 1, &r[x + 1 + xx * ldr], 1, &wdr);
      wdr = (wdr + r[x + xx * ldr]) * tau;
      Cdaxpy(m - x - 1, -wdr, &r[x + 1 + x * ldr], 1, &r[x + 1 + xx * ldr], 1);
      r[x + xx * ldr] = r[x + xx * ldr] - wdr;
    }
  }

  double zero = 0.;
  for (int64_t x = 0; x < nq; x++) {
    Cdcopy(m, &zero, 0, &q[x * ldq], 1);
    q[x + x * ldq] = 1.;
  }

  for (int64_t kk = k - 1; kk >= 0; kk--)
    for (int64_t x = 0; x < nq; x++) {
      double wdq = 0.;
      Cddot(m - kk - 1, &r[kk + 1 + kk * ldr], 1, &q[kk + 1 + x * ldq], 1, &wdq);
      wdq = (q[kk + x * ldq] + wdq) * TAU[kk];
      Cdaxpy(m - kk - 1, -wdq, &r[kk + 1 + kk * ldr], 1, &q[kk + 1 + x * ldq], 1);
      q[kk + x * ldq] = q[kk + x * ldq] - wdq;
    }
}


void dchol(int64_t n, double* a, int64_t lda) {
  for (int64_t i = 0; i < n; i++) {
    double p = a[i + i * lda];
    if (p <= 0.)
    { fprintf(stderr, "A is not positive-definite.\n"); return; }
    p = sqrt(p);
    a[i + i * lda] = p;
    double invp = 1. / p;
    Cdscal(n - i - 1, invp, &a[i + 1 + i * lda], 1);

    for (int64_t k = i + 1; k < n; k++) {
      double c = a[k + i * lda];
      a[k + k * lda] = a[k + k * lda] - c * c;
      Cdaxpy(n - k - 1, -c, &a[k + 1 + i * lda], 1, &a[k + 1 + k * lda], 1);
    }
  }
}

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
    Cdcopy(A.M, &A.A[j * A.M], 1, &V[j * iv], incv);
}

void nbd::cpyFromVector(const Vector& A, double* v) {
  std::copy(A.X.data(), A.X.data() + A.N, v);
}

void nbd::maxpby(Matrix& A, const double* v, double alpha, double beta) {
  int64_t size = A.M * A.N;
  if (beta == 0.)
    std::fill(A.A.data(), A.A.data() + size, 0.);
  else if (beta != 1.)
    Cdscal(size, beta, A.A.data(), 1);
  Cdaxpy(size, alpha, v, 1, A.A.data(), 1);
}

void nbd::vaxpby(Vector& A, const double* v, double alpha, double beta) {
  int64_t size = A.N;
  if (beta == 0.)
    std::fill(A.X.data(), A.X.data() + size, 0.);
  else if (beta != 1.)
    Cdscal(size, beta, A.X.data(), 1);
  Cdaxpy(size, alpha, v, 1, A.X.data(), 1);
}

void nbd::cpyMatToMat(int64_t m, int64_t n, const Matrix& m1, Matrix& m2, int64_t y1, int64_t x1, int64_t y2, int64_t x2) {
  if (m > 0 && n > 0)
    for (int64_t j = 0; j < n; j++) {
      int64_t j1 = y1 + (x1 + j) * m1.M;
      int64_t j2 = y2 + (x2 + j) * m2.M;
      std::copy(&m1.A[j1], &m1.A[j1] + m, &m2.A[j2]);
    }
}

void nbd::cpyVecToVec(int64_t n, const Vector& v1, Vector& v2, int64_t x1, int64_t x2) {
  if (n > 0)
    std::copy(&v1.X[x1], &v1.X[x1] + n, &v2.X[x2]);
}

void nbd::orthoBase(double epi, Matrix& A, int64_t *rnk_out) {
  Matrix U, V;
  int64_t rank = std::min(A.M, A.N);
  cMatrix(U, A.M, rank);
  cMatrix(V, A.N, rank);

  dlra(epi, A.M, A.N, rank, A.A.data(), U.A.data(), A.M, V.A.data(), A.N, rnk_out, NULL);
  rank = *rnk_out;

  if (A.N < A.M)
    cMatrix(A, A.M, A.M);
  Vector tau;
  cVector(tau, rank);
  dorth('F', A.M, rank, U.A.data(), A.M, A.A.data(), A.M, tau.X.data());
}

void nbd::lraID(double epi, int64_t mrank, Matrix& A, Matrix& U, int64_t arows[], int64_t* rnk_out) {
  int64_t rank = mrank;
  rank = std::min(A.M, rank);
  rank = std::min(A.N, rank);

  Matrix UU, V;
  cMatrix(UU, A.M, rank);
  cMatrix(V, A.N, rank);
  dlra(epi, A.M, A.N, rank, A.A.data(), UU.A.data(), A.M, V.A.data(), A.N, rnk_out, arows);
  rank = *rnk_out;

  Matrix R, Q;
  cMatrix(R, rank, rank);
  cMatrix(Q, rank, rank);

  for (int64_t i = 0; i < rank; i++) {
    int64_t ymax = arows[i];
    int64_t xp = ymax / A.M;
    int64_t yp = ymax - xp * A.M;
    arows[i] = yp;
    Cdcopy(rank, &UU.A[yp], A.M, &R.A[i], rank);
  }
  
  Vector tau;
  cVector(tau, rank);
  dorth('F', rank, rank, R.A.data(), rank, Q.A.data(), rank, tau.X.data());
  dtrsmr_right(A.M, rank, R.A.data(), rank, UU.A.data(), A.M);
  if (U.N != rank)
    cMatrix(U, A.M, rank);
  Cdgemm('N', 'T', A.M, rank, rank, 1., UU.A.data(), A.M, Q.A.data(), rank, 0., U.A.data(), A.M);
}

void nbd::zeroMatrix(Matrix& A) {
  std::fill(A.A.data(), A.A.data() + A.M * A.N, 0.);
}

void nbd::zeroVector(Vector& A) {
  std::fill(A.X.data(), A.X.data() + A.N, 0.);
}

void nbd::mmult(char ta, char tb, const Matrix& A, const Matrix& B, Matrix& C, double alpha, double beta) {
  int64_t k = (ta == 'N' || ta == 'n') ? A.N : A.M;
  if (C.M > 0 && C.N > 0 && k > 0)
    Cdgemm(ta, tb, C.M, C.N, k, alpha, A.A.data(), A.M, B.A.data(), B.M, beta, C.A.data(), C.M);
}

void nbd::msample(char ta, int64_t lenR, const Matrix& A, const double* R, Matrix& C) {
  if (lenR < C.N * 100) { 
    std::cerr << "Insufficent random vector: " << C.N << " x 100 needed " << lenR << " provided." << std::endl;
    return;
  }
  int64_t k = A.M;
  int64_t inca = 1;
  if (ta == 'N' || ta == 'n') {
    k = A.N;
    inca = A.M;
  }

  int64_t rk = lenR / C.N;
  int64_t lk = k % rk;
  if (lk > 0)
    Cdgemm(ta, 'N', C.M, C.N, lk, 1., A.A.data(), A.M, R, lk, 1., C.A.data(), C.M);
  if (k > rk)
    for (int64_t i = lk; i < k; i += rk)
      Cdgemm(ta, 'N', C.M, C.N, rk, 1., &A.A[i * inca], A.M, R, rk, 1., C.A.data(), C.M);
}

void nbd::msample_m(char ta, const Matrix& A, const Matrix& B, Matrix& C) {
  int64_t k = A.M;
  if (ta == 'N' || ta == 'n')
    k = A.N;
  int64_t nrhs = std::min(B.N, C.N);
  Cdgemm(ta, 'N', C.M, nrhs, k, 1., A.A.data(), A.M, B.A.data(), B.M, 1., C.A.data(), C.M);
}

void nbd::minvl(const Matrix& A, Matrix& B) {
  Matrix work;
  cMatrix(work, A.M, A.N);
  cpyMatToMat(A.M, A.N, A, work, 0, 0, 0, 0);
  chol_decomp(work);
  dtrsml_left(B.M, B.N, work.A.data(), A.M, B.A.data(), B.M);
  dtrsmlt_left(B.M, B.N, work.A.data(), A.M, B.A.data(), B.M);
}

void nbd::invBasis(const Matrix& u, Matrix& uinv) {
  int64_t m = u.M;
  int64_t n = u.N;
  if (m > 0 && n > 0) {
    Matrix a;
    Matrix q;
    cMatrix(a, n, n);
    cMatrix(q, n, n);
    cMatrix(uinv, n, m);
    mmult('T', 'N', u, u, a, 1., 0.);

    Vector tau;
    cVector(tau, n);
    dorth('F', n, n, a.A.data(), n, q.A.data(), n, tau.X.data());
    mmult('T', 'T', q, u, uinv, 1., 0.);
    dtrsmr_left(n, m, a.A.data(), n, uinv.A.data(), n);
  }
}

void nbd::chol_decomp(Matrix& A) {
  if (A.M > 0)
    dchol(A.M, A.A.data(), A.M);
}

void nbd::trsm_lowerA(Matrix& A, const Matrix& L) {
  if (A.M > 0 && L.M > 0)
    dtrsmlt_right(A.M, A.N, L.A.data(), L.M, A.A.data(), A.M);
}

void nbd::utav(char tb, const Matrix& U, const Matrix& A, const Matrix& VT, Matrix& C) {
  Matrix work;
  cMatrix(work, C.M, A.N);
  if (tb == 'N' || tb == 'n') {
    mmult('T', 'N', U, A, work, 1., 0.);
    mmult('N', 'N', work, VT, C, 1., 0.);
  }
  else if (tb == 'T' || tb == 't') {
    mmult('N', 'N', U, A, work, 1., 0.);
    mmult('N', 'T', work, VT, C, 1., 0.);
  }
}

void nbd::solmv(char fwbk, Vector& X, const Matrix& A) {
  if (fwbk == 'F' || fwbk == 'f' || fwbk == 'A' || fwbk == 'a')
    if (A.M > 0 && X.N > 0)
      dtrsml_left(X.N, 1, A.A.data(), A.M, X.X.data(), X.N);
  if (fwbk == 'B' || fwbk == 'b' || fwbk == 'A' || fwbk == 'a')
    if (A.M > 0 && X.N > 0)
      dtrsmlt_left(X.N, 1, A.A.data(), A.M, X.X.data(), X.N);
}


void nbd::mvec(char ta, const Matrix& A, const Vector& X, Vector& B, double alpha, double beta) {
  if (A.M > 0 && A.N > 0)
    Cdgemv(ta, A.M, A.N, alpha, A.A.data(), A.M, X.X.data(), 1, beta, B.X.data(), 1);
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
  Cdnrm2(A.N, A.X.data(), 1, nrm);
}

void nbd::verr2(const Vector& A, const Vector& B, double* err) {
  Vector work;
  cVector(work, A.N);
  vaxpby(work, A.X.data(), 1., 0.);
  vaxpby(work, B.X.data(), -1., 1.);
  vnrm2(work, err);
}
