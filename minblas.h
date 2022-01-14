
#include <stdint.h>

void dlra(double epi, int64_t m, int64_t n, int64_t k, double* a, int64_t lda, double* u, int64_t ldu, double* vt, int64_t ldvt, int64_t* rank);

void dorth(char ecoq, int64_t m, int64_t n, double* r, int64_t ldr, double* q, int64_t ldq);

void dpotrf(int64_t n, double* a, int64_t lda);

void dtrsmlt_right(int64_t m, int64_t n, const double* a, int64_t lda, double* b, int64_t ldb);

void dtrsml_left(int64_t m, int64_t n, const double* a, int64_t lda, double* b, int64_t ldb);

void dtrsmlt_left(int64_t m, int64_t n, const double* a, int64_t lda, double* b, int64_t ldb);

void dgemv(char ta, int64_t m, int64_t n, double alpha, const double* a, int64_t lda, const double* x, int64_t incx, double beta, double* y, int64_t incy);

void dgemm(char ta, char tb, int64_t m, int64_t n, int64_t k, double alpha, const double* a, int64_t lda, const double* b, int64_t ldb, double beta, double* c, int64_t ldc);

void dcopy(int64_t n, const double* x, int64_t incx, double* y, int64_t incy);

void dscal(int64_t n, double alpha, double* x, int64_t incx);

void daxpy(int64_t n, double alpha, const double* x, int64_t incx, double* y, int64_t incy);

void dnrm2(int64_t n, const double* x, int64_t incx, double* nrm_out);

