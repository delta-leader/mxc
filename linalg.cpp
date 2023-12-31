
#include <linalg.hpp>
#include <kernel.hpp>

#include "lapacke.h"
#include "cblas.h"

#include <vector>
#include <algorithm>
#include <numeric>
#include <array>

void mmult(char ta, char tb, const Matrix* A, const Matrix* B, Matrix* C, double alpha, double beta) {
  int64_t k = ta == 'N' ? A->N : A->M;
  CBLAS_TRANSPOSE tac = ta == 'N' ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE tbc = tb == 'N' ? CblasNoTrans : CblasTrans;
  int64_t lda = 1 < A->LDA ? A->LDA : 1;
  int64_t ldb = 1 < B->LDA ? B->LDA : 1;
  int64_t ldc = 1 < C->LDA ? C->LDA : 1;
  cblas_dgemm(CblasColMajor, tac, tbc, C->M, C->N, k, alpha, A->A, lda, B->A, ldb, beta, C->A, ldc);
}

void mul_AS(const Matrix* RU, const Matrix* RV, Matrix* A) {
  if (A->M > 0 && A->N > 0) {
    std::vector<double> tmp(A->M * A->N);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A->M, A->N, A->M, 1., RU->A, RU->LDA, A->A, A->LDA, 0., &tmp[0], A->M);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, A->M, A->N, A->N, 1., &tmp[0], A->M, RV->A, RV->LDA, 0., A->A, A->LDA);
  }
}

void gen_matrix(const EvalDouble& Eval, int64_t m, int64_t n, const double* bi, const double* bj, double Aij[], int64_t lda) {
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
      Aij[iy + ix * lda] = Eval(d);
    });
  });
}

int64_t compute_basis(const EvalDouble& eval, double epi, int64_t M, double* A, int64_t LDA, double Xbodies[], int64_t Lfar, int64_t Nfar[], const double* Fbodies[]) {

  if (M > 0 && Lfar > 0) {
    int64_t N = std::accumulate(Nfar, &Nfar[Lfar], 0), loc = 0;
    std::vector<double> Aall(M * std::max(M, N)), U(M * M), S(M * 3);
    std::vector<int32_t> ipiv(M, 0);
    for (int64_t i = 0; i < Lfar; i++) {
      gen_matrix(eval, Nfar[i], M, Fbodies[i], Xbodies, &Aall[loc], N);
      loc = loc + Nfar[i];
    }

    LAPACKE_dgeqp3(LAPACK_COL_MAJOR, N, M, &Aall[0], N, &ipiv[0], &S[0]);
    LAPACKE_dlaset(LAPACK_COL_MAJOR, 'L', M - 1, M - 1, 0., 0., &Aall[1], N);
    int64_t rank = 0;
    double s0 = std::abs(epi * Aall[0]);
    while (rank < M && rank < N && s0 <= std::abs(Aall[rank * (N + 1)]))
      ++rank;
    
    if (rank > 0) {
      if (rank < M)
        cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, rank, M - rank, 1., &Aall[0], N, &Aall[rank * N], N);
      LAPACKE_dlaset(LAPACK_COL_MAJOR, 'F', rank, rank, 0., 1., &Aall[0], N);

      for (int64_t i = 0; i < M; i++) {
        int64_t piv = (int64_t)ipiv[i] - 1;
        std::copy(&Aall[i * N], &Aall[i * N + rank], &U[piv * M]);
        std::copy(&Xbodies[piv * 3], &Xbodies[piv * 3 + 3], &S[i * 3]);
      }
      std::copy(&S[0], &S[M * 3], Xbodies);

      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, M, rank, M, 1., A, LDA, &U[0], M, 0., &Aall[0], M);
      LAPACKE_dgeqrf(LAPACK_COL_MAJOR, M, rank, &Aall[0], M, &S[0]);
      LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'L', M, rank, &Aall[0], M, A, LDA);
      LAPACKE_dorgqr(LAPACK_COL_MAJOR, M, M, rank, A, LDA, &S[0]);

      LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', rank, rank, &Aall[0], M, &A[M * LDA], LDA);
      LAPACKE_dlaset(LAPACK_COL_MAJOR, 'L', rank - 1, rank - 1, 0., 0., &A[M * LDA + 1], LDA);
    }
    
    return rank;
  }
  return 0;
}


void mat_vec_reference(const EvalDouble& eval, int64_t begin, int64_t end, double B[], int64_t nbodies, const double* bodies, const double Xbodies[]) {
  int64_t M = end - begin;
  int64_t N = nbodies;
  int64_t size = 1024;
  std::vector<double> A(size * size);
  
  for (int64_t i = 0; i < M; i += size) {
    int64_t y = begin + i;
    int64_t m = std::min(M - i, size);
    const double* bi = &bodies[y * 3];
    for (int64_t j = 0; j < N; j += size) {
      const double* bj = &bodies[j * 3];
      int64_t n = std::min(N - j, size);
      gen_matrix(eval, m, n, bi, bj, &A[0], size);
      cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, 1., &A[0], size, &Xbodies[j], 1, 1., &B[i], 1);
    }
  }
}


