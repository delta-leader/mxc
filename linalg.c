
#include "nbd.h"

#if defined(_MKL) || defined(_MKL_BATCH)
#include "mkl.h"
#else
#include "cblas.h"
#include "lapacke.h"
#endif

#ifndef BATCH_LEN
#define BATCH_LEN 12000
#endif

struct MatrixCopyBatch {
  int32_t rows_array[BATCH_LEN], cols_array[BATCH_LEN], lda_array[BATCH_LEN], ldb_array[BATCH_LEN];
  const double* A_array[BATCH_LEN];
  double* B_array[BATCH_LEN];
} copy_batch;
int32_t copy_batch_count = 0;

void mat_cpy_batch(int64_t m, int64_t n, const struct Matrix* m1, struct Matrix* m2, int64_t y1, int64_t x1, int64_t y2, int64_t x2) {
  if (copy_batch_count == BATCH_LEN)
    mat_cpy_flush();
  int32_t i = copy_batch_count;
  copy_batch.rows_array[i] = m;
  copy_batch.cols_array[i] = n;
  copy_batch.A_array[i] = &m1->A[y1 + x1 * m1->M];
  copy_batch.lda_array[i] = m1->M;
  copy_batch.B_array[i] = &m2->A[y2 + x2 * m2->M];
  copy_batch.ldb_array[i] = m2->M;
  copy_batch_count = i + (int)(m > 0 && n > 0);
}

void mat_cpy_flush() {
#ifdef _MKL_BATCH
  char trans_array[BATCH_LEN];
  double alpha_array[BATCH_LEN];
  size_t rows_ui64[BATCH_LEN], cols_ui64[BATCH_LEN], lda_ui64[BATCH_LEN], ldb_ui64[BATCH_LEN], group_size[BATCH_LEN];
#pragma omp parallel for
  for (int32_t i = 0; i < copy_batch_count; i++) {
    trans_array[i] = 'N';
    alpha_array[i] = 1.;
    rows_ui64[i] = copy_batch.rows_array[i];
    cols_ui64[i] = copy_batch.cols_array[i];
    lda_ui64[i] = copy_batch.lda_array[i];
    ldb_ui64[i] = copy_batch.ldb_array[i];
    group_size[i] = 1;
  }
  MKL_Domatcopy_batch('C', trans_array, rows_ui64, cols_ui64, alpha_array,
    copy_batch.A_array, lda_ui64, copy_batch.B_array, ldb_ui64, copy_batch_count, group_size);
#else
//#pragma omp parallel for
  for (int32_t i = 0; i < copy_batch_count; i++)
    LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', copy_batch.rows_array[i], copy_batch.cols_array[i],
      copy_batch.A_array[i], copy_batch.lda_array[i], copy_batch.B_array[i], copy_batch.ldb_array[i]);
#endif
  copy_batch_count = 0;
}

struct MMultBatch {
  char transa_array[BATCH_LEN], transb_array[BATCH_LEN];
  int32_t m_array[BATCH_LEN], n_array[BATCH_LEN], k_array[BATCH_LEN];
  const double* A_array[BATCH_LEN], *B_array[BATCH_LEN];
  int32_t lda_array[BATCH_LEN], ldb_array[BATCH_LEN], ldc_array[BATCH_LEN];
  double* C_array[BATCH_LEN], alpha_array[BATCH_LEN], beta_array[BATCH_LEN];
} mm_batch;
int32_t mm_batch_count = 0;

void mmult(char ta, char tb, const struct Matrix* A, const struct Matrix* B, struct Matrix* C, double alpha, double beta) {
  int64_t k = ta == 'N' ? A->N : A->M;
  CBLAS_TRANSPOSE tac = ta == 'N' ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE tbc = tb == 'N' ? CblasNoTrans : CblasTrans;
  int64_t lda = 1 < A->M ? A->M : 1;
  int64_t ldb = 1 < B->M ? B->M : 1;
  int64_t ldc = 1 < C->M ? C->M : 1;
  cblas_dgemm(CblasColMajor, tac, tbc, C->M, C->N, k, alpha, A->A, lda, B->A, ldb, beta, C->A, ldc);
}

void mmult_batch(char ta, char tb, const struct Matrix* A, const struct Matrix* B, struct Matrix* C, double alpha, double beta) {
  if (mm_batch_count == BATCH_LEN)
    mmult_flush();
  int32_t i = mm_batch_count;
  mm_batch.transa_array[i] = ta;
  mm_batch.transb_array[i] = tb;
  mm_batch.m_array[i] = C->M;
  mm_batch.n_array[i] = C->N;
  mm_batch.k_array[i] = (ta == 'N' || ta == 'n') ? A->N : A->M;
  mm_batch.alpha_array[i] = alpha;
  mm_batch.A_array[i] = A->A;
  mm_batch.lda_array[i] = 1 < A->M ? A->M : 1;
  mm_batch.B_array[i] = B->A;
  mm_batch.ldb_array[i] = 1 < B->M ? B->M : 1;
  mm_batch.beta_array[i] = beta;
  mm_batch.C_array[i] = C->A;
  mm_batch.ldc_array[i] = 1 < C->M ? C->M : 1;
  mm_batch_count = i + (int)(C->M > 0 && C->N > 0);
}

void mmult_flush() {
#ifdef _MKL_BATCH
  int32_t one_array[BATCH_LEN];
#pragma omp parallel for
  for (int32_t i = 0; i < mm_batch_count; i++)
    one_array[i] = 1;
  
  dgemm_batch(mm_batch.transa_array, mm_batch.transb_array, mm_batch.m_array, mm_batch.n_array, mm_batch.k_array,
    mm_batch.alpha_array, mm_batch.A_array, mm_batch.lda_array, mm_batch.B_array, mm_batch.ldb_array,
    mm_batch.beta_array, mm_batch.C_array, mm_batch.ldc_array, &mm_batch_count, one_array);
#else
//#pragma omp parallel for
  for (int32_t i = 0; i < mm_batch_count; i++) {
    CBLAS_TRANSPOSE tac = mm_batch.transa_array[i] == 'N' ? CblasNoTrans : CblasTrans;
    CBLAS_TRANSPOSE tbc = mm_batch.transb_array[i] == 'N' ? CblasNoTrans : CblasTrans;
    cblas_dgemm(CblasColMajor, tac, tbc, mm_batch.m_array[i], mm_batch.n_array[i], mm_batch.k_array[i], mm_batch.alpha_array[i],
      mm_batch.A_array[i], mm_batch.lda_array[i], mm_batch.B_array[i], mm_batch.ldb_array[i], mm_batch.beta_array[i], mm_batch.C_array[i], mm_batch.ldc_array[i]);
  }
#endif
  mm_batch_count = 0;
}

struct ICholBatch {
  int32_t lc_array[BATCH_LEN];
  int32_t lo_array[BATCH_LEN];
  double* A_array[BATCH_LEN];
  int32_t lda_array[BATCH_LEN];
  double* B_array[BATCH_LEN];
  int32_t ldb_array[BATCH_LEN];
  double* C_array[BATCH_LEN];
  int32_t ldc_array[BATCH_LEN];
} ichol_batch;
int32_t ichol_batch_count = 0;

void chol_decomp(struct Matrix* A) {
  int64_t lda = 1 < A->M ? A->M : 1;
  LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', A->M, A->A, lda);
}

void icmp_chol_decomp_batch(struct Matrix* A_cc, struct Matrix* A_oc, struct Matrix* A_oo) {
  if (ichol_batch_count == BATCH_LEN)
    icmp_chol_decomp_flush();
  int32_t i = ichol_batch_count;
  ichol_batch.lc_array[i] = A_cc->M;
  ichol_batch.lo_array[i] = A_oo->M;
  ichol_batch.A_array[i] = A_cc->A;
  ichol_batch.lda_array[i] = 1 < A_cc->M ? A_cc->M : 1;
  ichol_batch.B_array[i] = A_oc->A;
  ichol_batch.ldb_array[i] = 1 < A_oc->M ? A_oc->M : 1;
  ichol_batch.C_array[i] = A_oo->A;
  ichol_batch.ldc_array[i] = 1 < A_oo->M ? A_oo->M : 1;
  ichol_batch_count = i + (int)(A_cc->M > 0);
}

void icmp_chol_decomp_flush() {
#ifdef _MKL_BATCH
  char L_array[BATCH_LEN];
  char R_array[BATCH_LEN];
  char T_array[BATCH_LEN];
  char N_array[BATCH_LEN];
  double alpha_array[BATCH_LEN];
  double beta_array[BATCH_LEN];
  int32_t one_array[BATCH_LEN];
  int32_t info_array[BATCH_LEN];
#pragma omp parallel for
  for (int32_t i = 0; i < ichol_batch_count; i++) {
    L_array[i] = 'L';
    R_array[i] = 'R';
    T_array[i] = 'T';
    N_array[i] = 'N';
    alpha_array[i] = 1.;
    beta_array[i] = -1.;
    one_array[i] = 1;
  }

#pragma omp parallel for
  for (int32_t i = 0; i < ichol_batch_count; i++)
    dpotrf(&L_array[i], &ichol_batch.lc_array[i], ichol_batch.A_array[i], &ichol_batch.lda_array[i], &info_array[i]);
  
  dtrsm_batch(R_array, L_array, T_array, N_array, ichol_batch.lo_array, ichol_batch.lc_array, alpha_array,
    (const double**)ichol_batch.A_array, ichol_batch.lda_array, ichol_batch.B_array, ichol_batch.ldb_array, &ichol_batch_count, one_array);

  dgemm_batch(N_array, T_array, ichol_batch.lo_array, ichol_batch.lo_array, ichol_batch.lc_array, beta_array,
    (const double**)ichol_batch.B_array, ichol_batch.ldb_array, (const double**)ichol_batch.B_array, ichol_batch.ldb_array, alpha_array,
    ichol_batch.C_array, ichol_batch.ldc_array, &ichol_batch_count, one_array);
#else
//#pragma omp parallel for
  for (int32_t i = 0; i < ichol_batch_count; i++) {
    LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', ichol_batch.lc_array[i], ichol_batch.A_array[i], ichol_batch.lda_array[i]);
    cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, ichol_batch.lo_array[i], ichol_batch.lc_array[i], 1.,
      ichol_batch.A_array[i], ichol_batch.lda_array[i], ichol_batch.B_array[i], ichol_batch.ldb_array[i]);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, ichol_batch.lo_array[i], ichol_batch.lo_array[i], ichol_batch.lc_array[i], -1.,
      ichol_batch.B_array[i], ichol_batch.ldb_array[i], ichol_batch.B_array[i], ichol_batch.ldb_array[i], 1., ichol_batch.C_array[i], ichol_batch.ldc_array[i]);
  }
#endif
  ichol_batch_count = 0;
}

struct TrsmLBatch {
  int32_t m_array[BATCH_LEN];
  int32_t n_array[BATCH_LEN];
  const double* L_array[BATCH_LEN];
  int32_t ldl_array[BATCH_LEN];
  double* A_array[BATCH_LEN];
  int32_t lda_array[BATCH_LEN];
} trsml_batch;
int32_t trsml_batch_count = 0;

void trsm_lowerA_batch(struct Matrix* A, const struct Matrix* L) {
  if (trsml_batch_count == BATCH_LEN)
    trsm_lowerA_flush();
  int32_t i = trsml_batch_count;
  trsml_batch.m_array[i] = A->M;
  trsml_batch.n_array[i] = A->N;
  trsml_batch.L_array[i] = L->A;
  trsml_batch.ldl_array[i] = 1 < L->M ? L->M : 1;
  trsml_batch.A_array[i] = A->A;
  trsml_batch.lda_array[i] = 1 < A->M ? A->M : 1;
  trsml_batch_count = i + (int)(A->M > 0 && A->N > 0);
}

void trsm_lowerA_flush() {
#ifdef _MKL_BATCH
  char L_array[BATCH_LEN];
  char R_array[BATCH_LEN];
  char T_array[BATCH_LEN];
  char N_array[BATCH_LEN];
  double alpha_array[BATCH_LEN];
  int32_t one_array[BATCH_LEN];
#pragma omp parallel for
  for (int32_t i = 0; i < trsml_batch_count; i++) {
    L_array[i] = 'L';
    R_array[i] = 'R';
    T_array[i] = 'T';
    N_array[i] = 'N';
    alpha_array[i] = 1.;
    one_array[i] = 1;
  }

  dtrsm_batch(R_array, L_array, T_array, N_array, trsml_batch.m_array, trsml_batch.n_array, alpha_array,
    trsml_batch.L_array, trsml_batch.ldl_array, trsml_batch.A_array, trsml_batch.lda_array, &trsml_batch_count, one_array);
#else
//#pragma omp parallel for
  for (int32_t i = 0; i < trsml_batch_count; i++)
    cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, trsml_batch.m_array[i], trsml_batch.n_array[i], 1.,
      trsml_batch.L_array[i], trsml_batch.ldl_array[i], trsml_batch.A_array[i], trsml_batch.lda_array[i]);
#endif
  trsml_batch_count = 0;
}

void svd_U(struct Matrix* A, double* S) {
  int64_t rank_a = A->M < A->N ? A->M : A->N;
  int64_t lda = 1 < A->M ? A->M : 1;
  int64_t ldv = 1 < A->N ? A->N : 1;
  LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'O', 'N', A->M, A->N, A->A, lda, S, NULL, lda, NULL, ldv, &S[rank_a]);
}

struct IdRowBatch {
  int32_t rows_array[BATCH_LEN];
  int32_t cols_array[BATCH_LEN];
  double* A_array[BATCH_LEN];
  int32_t lda_array[BATCH_LEN];
  int32_t* piv_array[BATCH_LEN];
  double* work_array[BATCH_LEN];
} idrow_batch;
int32_t idrow_batch_count = 0;

void id_row_batch(struct Matrix* A, int32_t arows[], double* work) {
  if (idrow_batch_count == BATCH_LEN)
    id_row_flush();
  int32_t i = idrow_batch_count;
  idrow_batch.rows_array[i] = A->M;
  idrow_batch.cols_array[i] = A->N;
  idrow_batch.A_array[i] = A->A;
  idrow_batch.lda_array[i] = 1 < A->M ? A->M : 1;
  idrow_batch.piv_array[i] = arows;
  idrow_batch.work_array[i] = work;
  idrow_batch_count = i + (int)(A->M > 0 && A->N > 0);
}

void id_row_flush() {
#ifdef _MKL_BATCH
  char R_array[BATCH_LEN];
  char N_array[BATCH_LEN];
  char L_array[BATCH_LEN];
  char U_array[BATCH_LEN];
  double alpha_array[BATCH_LEN];
  int32_t one_array[BATCH_LEN];
  int32_t info_array[BATCH_LEN];

  size_t rows_ui64[BATCH_LEN];
  size_t cols_ui64[BATCH_LEN];
  size_t lda_ui64[BATCH_LEN];
  size_t group_size[BATCH_LEN];
#pragma omp parallel for
  for (int32_t i = 0; i < idrow_batch_count; i++) {
    R_array[i] = 'R';
    N_array[i] = 'N';
    L_array[i] = 'L';
    U_array[i] = 'U';
    alpha_array[i] = 1.;
    one_array[i] = 1;
    rows_ui64[i] = idrow_batch.rows_array[i];
    cols_ui64[i] = idrow_batch.cols_array[i];
    lda_ui64[i] = idrow_batch.lda_array[i];
    group_size[i] = 1;
  }
  
  MKL_Domatcopy_batch('C', N_array, rows_ui64, cols_ui64, alpha_array,
    (const double**)idrow_batch.A_array, lda_ui64, idrow_batch.work_array, lda_ui64, idrow_batch_count, group_size);
  
  dgetrf_batch(idrow_batch.rows_array, idrow_batch.cols_array, idrow_batch.work_array, idrow_batch.lda_array,
    idrow_batch.piv_array, &idrow_batch_count, one_array, info_array);

  dtrsm_batch(R_array, U_array, N_array, N_array, idrow_batch.rows_array, idrow_batch.cols_array, alpha_array,
    (const double**)idrow_batch.work_array, idrow_batch.lda_array, idrow_batch.A_array, idrow_batch.lda_array, &idrow_batch_count, one_array);

  dtrsm_batch(R_array, L_array, N_array, U_array, idrow_batch.rows_array, idrow_batch.cols_array, alpha_array,
    (const double**)idrow_batch.work_array, idrow_batch.lda_array, idrow_batch.A_array, idrow_batch.lda_array, &idrow_batch_count, one_array);
#else
//#pragma omp parallel for
  for (int32_t i = 0; i < idrow_batch_count; i++) {
    LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', idrow_batch.rows_array[i], idrow_batch.cols_array[i], 
      idrow_batch.A_array[i], idrow_batch.lda_array[i], idrow_batch.work_array[i], idrow_batch.lda_array[i]);
    LAPACKE_dgetrf(LAPACK_COL_MAJOR, idrow_batch.rows_array[i], idrow_batch.cols_array[i],
      idrow_batch.work_array[i], idrow_batch.lda_array[i], idrow_batch.piv_array[i]);
    cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, idrow_batch.rows_array[i], idrow_batch.cols_array[i], 1.,
      idrow_batch.work_array[i], idrow_batch.lda_array[i], idrow_batch.A_array[i], idrow_batch.lda_array[i]);
    cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit, idrow_batch.rows_array[i], idrow_batch.cols_array[i], 1.,
      idrow_batch.work_array[i], idrow_batch.lda_array[i], idrow_batch.A_array[i], idrow_batch.lda_array[i]);
  }
#endif
  idrow_batch_count = 0;
}

void upper_tri_reflec_mult(char side, const struct Matrix* R, struct Matrix* A) {
  int64_t ldr = 1 < R->M ? R->M : 1;
  int64_t lda = 1 < A->M ? A->M : 1;
  if (side == 'L' || side == 'l')
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, A->M, A->N, 1., R->A, ldr, A->A, lda);
  else if (side == 'R' || side == 'r')
    cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit, A->M, A->N, 1., R->A, ldr, A->A, lda);
}

void qr_full(struct Matrix* Q, struct Matrix* R, double* tau) {
  int64_t ldq = 1 < Q->M ? Q->M : 1;
  int64_t ldr = 1 < R->M ? R->M : 1;
  LAPACKE_dgeqrf(LAPACK_COL_MAJOR, Q->M, R->N, Q->A, ldq, tau);
  LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', R->M, R->N, Q->A, ldq, R->A, ldr);
  LAPACKE_dorgqr(LAPACK_COL_MAJOR, Q->M, Q->N, R->N, Q->A, ldq, tau);
}

void mat_solve(char type, struct Matrix* X, const struct Matrix* A) {
  int64_t lda = 1 < A->M ? A->M : 1;
  int64_t ldx = 1 < X->M ? X->M : 1;
  if (type == 'F' || type == 'f' || type == 'A' || type == 'a')
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, X->M, X->N, 1., A->A, lda, X->A, ldx);
  if (type == 'B' || type == 'b' || type == 'A' || type == 'a')
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit, X->M, X->N, 1., A->A, lda, X->A, ldx);
}

void nrm2_A(struct Matrix* A, double* nrm) {
  int64_t len_A = A->M * A->N;
  double nrm_A = cblas_dnrm2(len_A, A->A, 1);
  *nrm = nrm_A;
}

void scal_A(struct Matrix* A, double alpha) {
  int64_t len_A = A->M * A->N;
  cblas_dscal(len_A, alpha, A->A, 1);
}