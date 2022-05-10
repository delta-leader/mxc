
#include "minblas.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef CBLAS
#include "mkl.h"
#endif

int64_t FLOPS = 0;

void dtrsmlt_right(int64_t m, int64_t n, const double* a, int64_t lda, double* b, int64_t ldb) {
#ifdef CBLAS
  cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, m, n, 1., a, lda, b, ldb);
#else
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
#endif
  FLOPS = FLOPS + n * m * n / 3;
}

void dtrsmr_right(int64_t m, int64_t n, const double* a, int64_t lda, double* b, int64_t ldb) {
#ifdef CBLAS
  cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, m, n, 1., a, lda, b, ldb);
#else
  for (int64_t i = 0; i < n; i++) {
    double p = a[i + i * lda];
    double invp = 1. / p;

    for (int64_t j = 0; j < m; j++)
      b[j + i * ldb] = b[j + i * ldb] * invp;

    for (int64_t k = i + 1; k < n; k++) {
      double c = a[i + k * lda];
      for (int64_t j = 0; j < m; j++) {
        double r = b[j + i * ldb];
        b[j + k * ldb] = b[j + k * ldb] - r * c;
      }
    }
  }
#endif
  FLOPS = FLOPS + n * m * n / 3;
}

void dtrsml_left(int64_t m, int64_t n, const double* a, int64_t lda, double* b, int64_t ldb) {
#ifdef CBLAS
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, m, n, 1., a, lda, b, ldb);
#else
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
#endif
  FLOPS = FLOPS + m * n * m / 3;
}

void dtrsmlt_left(int64_t m, int64_t n, const double* a, int64_t lda, double* b, int64_t ldb) {
#ifdef CBLAS
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit, m, n, 1., a, lda, b, ldb);
#else
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
#endif
  FLOPS = FLOPS + m * n * m / 3;
}

void dtrsmr_left(int64_t m, int64_t n, const double* a, int64_t lda, double* b, int64_t ldb) {
#ifdef CBLAS
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, m, n, 1., a, lda, b, ldb);
#else
  for (int64_t i = m - 1; i >= 0; i--) {
    double p = a[i + i * lda];
    double invp = 1. / p;

    for (int64_t j = 0; j < n; j++)
      b[i + j * ldb] = b[i + j * ldb] * invp;

    for (int64_t k = 0; k < i; k++) {
      double r = a[k + i * lda];
      for (int64_t j = 0; j < n; j++) {
        double c = b[i + j * ldb];
        b[k + j * ldb] = b[k + j * ldb] - r * c;
      }
    }
  }
#endif
  FLOPS = FLOPS + m * n * m / 3;
}

void Cdgemv(char ta, int64_t m, int64_t n, double alpha, const double* a, int64_t lda, const double* x, int64_t incx, double beta, double* y, int64_t incy) {
#ifdef CBLAS
  if (ta == 'T' || ta == 't')
    cblas_dgemv(CblasColMajor, CblasTrans, m, n, alpha, a, lda, x, incx, beta, y, incy);
  else if (ta == 'N' || ta == 'n')
    cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, alpha, a, lda, x, incx, beta, y, incy);
#else
  if (ta == 'T' || ta == 't') {
    int64_t lenx = m;
    int64_t leny = n;

    for (int64_t i = 0; i < leny; i++) {
      double e = 0.;
      if (beta == 1.)
        e = y[i * incy];
      else if (beta != 0.)
        e = beta * y[i * incy];
      
      double s = 0.;
      for (int64_t j = 0; j < lenx; j++) {
        double aji = a[j + i * lda];
        double xj = x[j * incx];
        s = s + aji * xj;
      }

      y[i * incy] = e + s * alpha;
    }
  }
  else if (ta == 'N' || ta == 'n') {
    int64_t lenx = n;
    int64_t leny = m;

    for (int64_t i = 0; i < leny; i++) {
      double e = 0.;
      if (beta == 1.)
        e = y[i * incy];
      else if (beta != 0.)
        e = beta * y[i * incy];
      
      double s = 0.;
      for (int64_t j = 0; j < lenx; j++) {
        double aij = a[i + j * lda];
        double xj = x[j * incx];
        s = s + aij * xj;
      }

      y[i * incy] = e + s * alpha;
    }
  }
#endif
  FLOPS = FLOPS + 2 * n * m;
}

void Cdgemm(char ta, char tb, int64_t m, int64_t n, int64_t k, double alpha, const double* a, int64_t lda, const double* b, int64_t ldb, double beta, double* c, int64_t ldc) {
#ifdef CBLAS
  if (ta == 'T' || ta == 't') {
    if (tb == 'T' || tb == 't')
      cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    else if (tb == 'N' || tb == 'n')
      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
  else if (ta == 'N' || ta == 'n') {
    if (tb == 'T' || tb == 't')
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    else if (tb == 'N' || tb == 'n')
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
#else
  int64_t ma = k;
  int64_t na = m;
  if (ta == 'N' || ta == 'n') {
    ma = m;
    na = k;
  }

  if (tb == 'T' || tb == 't')
    for (int64_t i = 0; i < n; i++)
      Cdgemv(ta, ma, na, alpha, a, lda, b + i, ldb, beta, c + i * ldc, 1);
  else if (tb == 'N' || tb == 'n')
    for (int64_t i = 0; i < n; i++)
      Cdgemv(ta, ma, na, alpha, a, lda, b + i * ldb, 1, beta, c + i * ldc, 1);
#endif
}

void Cdcopy(int64_t n, const double* x, int64_t incx, double* y, int64_t incy) {
#ifdef CBLAS
  cblas_dcopy(n, x, incx, y, incy);
#else
  for (int64_t i = 0; i < n; i++)
    y[i * incy] = x[i * incx];
#endif
}

void Cdscal(int64_t n, double alpha, double* x, int64_t incx) {
#ifdef CBLAS
  cblas_dscal(n, alpha, x, incx);
#else
  if (alpha == 0.)
    for (int64_t i = 0; i < n; i++)
      x[i * incx] = 0.;
  else if (alpha != 1.)
    for (int64_t i = 0; i < n; i++)
      x[i * incx] = alpha * x[i * incx];
#endif
  FLOPS = FLOPS + n;
}

void Cdaxpy(int64_t n, double alpha, const double* x, int64_t incx, double* y, int64_t incy) {
#ifdef CBLAS
  cblas_daxpy(n, alpha, x, incx, y, incy);
#else
  for (int64_t i = 0; i < n; i++)
    y[i * incy] = y[i * incy] + alpha * x[i * incx];
#endif
  FLOPS = FLOPS + 2 * n;
}

void Cddot(int64_t n, const double* x, int64_t incx, const double* y, int64_t incy, double* result) {
#ifdef CBLAS
  *result = cblas_ddot(n, x, incx, y, incy);
#else
  double s = 0.;
  for (int64_t i = 0; i < n; i++)
    s = s + y[i * incy] * x[i * incx];
  *result = s;
#endif
  FLOPS = FLOPS + 2 * n;
}

void Cidamax(int64_t n, const double* x, int64_t incx, int64_t* ida) {
#ifdef CBLAS
  *ida = cblas_idamax(n, x, incx);
#else
  if (n > 0) {
    double amax = x[0];
    int64_t ymax = 0;
    for (int64_t i = 1; i < n; i++) {
      double fa = fabs(x[i * incx]);
      if (fa > amax) {
        amax = fa;
        ymax = i;
      }
    }
    *ida = ymax;
  }
#endif
  FLOPS = FLOPS + n;
}

void Cdnrm2(int64_t n, const double* x, int64_t incx, double* nrm_out) {
#ifdef CBLAS
  *nrm_out = cblas_dnrm2(n, x, incx);
#else
  double nrm = 0.;
  for (int64_t i = 0; i < n; i++) {
    double e = x[i * incx];
    nrm = nrm + e * e;
  }
  nrm = sqrt(nrm);
  *nrm_out = nrm;
#endif
  FLOPS = FLOPS + 2 * n;
}

int64_t* getFLOPS() {
  return &FLOPS;
}
