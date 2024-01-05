
#include <linalg.hpp>
#include <kernel.hpp>

#include "lapacke.h"
#include "cblas.h"

#include <vector>
#include <algorithm>
#include <numeric>
#include <array>

void mmult(char ta, char tb, const Matrix* A, const Matrix* B, Matrix* C, std::complex<double> alpha, std::complex<double> beta) {
  int64_t k = ta == 'N' ? A->N : A->M;
  CBLAS_TRANSPOSE tac = ta == 'N' ? CblasNoTrans : CblasConjTrans;
  CBLAS_TRANSPOSE tbc = tb == 'N' ? CblasNoTrans : CblasConjTrans;
  int64_t lda = 1 < A->LDA ? A->LDA : 1;
  int64_t ldb = 1 < B->LDA ? B->LDA : 1;
  int64_t ldc = 1 < C->LDA ? C->LDA : 1;
  cblas_zgemm(CblasColMajor, tac, tbc, C->M, C->N, k, &alpha, A->A, lda, B->A, ldb, &beta, C->A, ldc);
}

void gen_matrix(const Eval& eval, int64_t m, int64_t n, const double* bi, const double* bj, std::complex<double> Aij[], int64_t lda) {
  const std::array<double, 3>* bi3 = reinterpret_cast<const std::array<double, 3>*>(bi);
  const std::array<double, 3>* bi3_end = reinterpret_cast<const std::array<double, 3>*>(&bi[3 * m]);
  const std::array<double, 3>* bj3 = reinterpret_cast<const std::array<double, 3>*>(bj);
  const std::array<double, 3>* bj3_end = reinterpret_cast<const std::array<double, 3>*>(&bj[3 * n]);

  std::for_each(bj3, bj3_end, [&](const std::array<double, 3>& j) -> void {
    int64_t ix = std::distance(bj3, &j);
    std::for_each(bi3, bi3_end, [&](const std::array<double, 3>& i) -> void {
      int64_t iy = std::distance(bi3, &i);
      double x = i[0] - j[0];
      double y = i[1] - j[1];
      double z = i[2] - j[2];
      double d = std::sqrt(x * x + y * y + z * z);
      Aij[iy + ix * lda] = eval(d);
    });
  });
}

int64_t compute_basis(const Eval& eval, double epi, int64_t M, std::complex<double>* A, int64_t LDA, double Xbodies[], int64_t Lfar, int64_t Nfar[], const double* Fbodies[]) {

  if (M > 0 && Lfar > 0) {
    int64_t N = std::max(M, (int64_t)(1 << 12)), B2 = N + M;
    lapack_complex_double one { 1., 0. }, zero { 0., 0. };

    std::vector<std::complex<double>> B(M * B2, 0.), U(M * M, 0.);
    std::vector<double> S(M * 3);
    std::vector<int32_t> ipiv(M, 0);

    for (int64_t i = 0; i < Lfar; i++)
      for (int64_t j = 0; j < Nfar[i]; j += N) {
        int64_t len = std::min(Nfar[i] - j, N);
        gen_matrix(eval, len, M, Fbodies[i] + (j * 3), Xbodies, &B[M], B2);
        LAPACKE_zgeqrf(LAPACK_COL_MAJOR, M + len, M, reinterpret_cast<lapack_complex_double*>(&B[0]), B2, reinterpret_cast<lapack_complex_double*>(&U[0]));
        LAPACKE_zlaset(LAPACK_COL_MAJOR, 'L', M - 1, M - 1, zero, zero, reinterpret_cast<lapack_complex_double*>(&B[1]), B2);
      }

    LAPACKE_zgeqp3(LAPACK_COL_MAJOR, M, M, reinterpret_cast<lapack_complex_double*>(&B[0]), B2, &ipiv[0], reinterpret_cast<lapack_complex_double*>(&U[0]));
    LAPACKE_zlaset(LAPACK_COL_MAJOR, 'L', M - 1, M - 1, zero, zero, reinterpret_cast<lapack_complex_double*>(&B[1]), B2);
    int64_t rank = 0;
    double s0 = epi * std::sqrt(std::norm(B[0]));
    while (rank < M && s0 <= std::sqrt(std::norm(B[rank * (B2 + 1)])))
      ++rank;
    
    if (rank > 0) {
      if (rank < M)
        cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, rank, M - rank, &one, &B[0], B2, &B[rank * B2], B2);
      LAPACKE_zlaset(LAPACK_COL_MAJOR, 'F', rank, rank, zero, one, reinterpret_cast<lapack_complex_double*>(&B[0]), B2);

      for (int64_t i = 0; i < M; i++) {
        int64_t piv = (int64_t)ipiv[i] - 1;
        std::copy(&B[i * B2], &B[i * B2 + rank], &U[piv * M]);
        std::copy(&Xbodies[piv * 3], &Xbodies[piv * 3 + 3], &S[i * 3]);
      }
      std::copy(&S[0], &S[M * 3], Xbodies);

      cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, M, rank, M, &one, A, LDA, &U[0], M, &zero, &B[0], M);
      LAPACKE_zgeqrf(LAPACK_COL_MAJOR, M, rank, reinterpret_cast<lapack_complex_double*>(&B[0]), M, reinterpret_cast<lapack_complex_double*>(&U[0]));
      LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'L', M, rank, reinterpret_cast<lapack_complex_double*>(&B[0]), M, reinterpret_cast<lapack_complex_double*>(A), LDA);
      LAPACKE_zungqr(LAPACK_COL_MAJOR, M, M, rank, reinterpret_cast<lapack_complex_double*>(A), LDA, reinterpret_cast<lapack_complex_double*>(&U[0]));

      LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'U', rank, rank, reinterpret_cast<lapack_complex_double*>(&B[0]), M, reinterpret_cast<lapack_complex_double*>(&A[M * LDA]), LDA);
      LAPACKE_zlaset(LAPACK_COL_MAJOR, 'L', rank - 1, rank - 1, zero, zero, reinterpret_cast<lapack_complex_double*>(&A[M * LDA + 1]), LDA);
    }
    return rank;
  }
  return 0;
}


void mat_vec_reference(const Eval& eval, int64_t M, int64_t N, std::complex<double> B[], const std::complex<double> X[], const double ibodies[], const double jbodies[]) {
  int64_t size = 1 << 8;
  std::vector<std::complex<double>> A(size * size);
  std::complex<double> one(1., 0.);
  
  for (int64_t i = 0; i < M; i += size) {
    int64_t m = std::min(M - i, size);
    const double* bi = &ibodies[i * 3];
    for (int64_t j = 0; j < N; j += size) {
      const double* bj = &jbodies[j * 3];
      int64_t n = std::min(N - j, size);
      gen_matrix(eval, m, n, bi, bj, &A[0], size);
      cblas_zgemv(CblasColMajor, CblasNoTrans, m, n, &one, &A[0], size, &X[j], 1, &one, &B[i], 1);
    }
  }
}


