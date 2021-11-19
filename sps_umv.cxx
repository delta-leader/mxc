
#include "sps_umv.hxx"
#include "sps_basis.hxx"

using namespace nbd;

void nbd::split_A(Matrices& A_out, const CSC& rels, const Matrices& A, const Matrices& U, const Matrices& V) {
  A_out.resize(rels.NNZ);

  for (int64_t x = 0; x < rels.N; x++)
    for (int64_t yx = rels.CSC_COLS[x]; yx < rels.CSC_COLS[x + 1]; yx++) {
      int64_t y = rels.CSC_ROWS[yx];
      utav(U[y], A[yx], V[x], A_out[yx]);
    }
}

void nbd::factor_Acc(Matrices& A_cc, const CSC& rels) {

  for (int64_t i = 0; i < rels.M; i++) {
    int64_t ii;
    lookupIJ(ii, rels, i, i);
    Matrix& A_ii = A_cc[ii];
    lu_decomp(A_ii);

    for (int64_t yi = rels.CSC_COLS[i]; yi < rels.CSC_COLS[i + 1]; yi++) {
      int64_t y = rels.CSC_ROWS[yi];
      if (y > i)
        trsm_lowerA(A_cc[yi], A_ii);
    }
  }
}

void nbd::factor_Alow(Matrices& Alow, const CSC& rels_low, Matrices& A_cc, const CSC& rels_cc) {
  for (int64_t x = 0; x < rels_low.N; x++) {
    int64_t xx;
    lookupIJ(xx, rels_cc, x, x);
    const Matrix& A_xx = A_cc[xx];
    for (int64_t yx = rels_low.CSC_COLS[x]; yx < rels_low.CSC_COLS[x + 1]; yx++)
      trsm_lowerA(Alow[yx], A_xx);
  }
}

void nbd::factor_Aup(Matrices& Aup, const CSC& rels_up, Matrices& A_cc, const CSC& rels_cc) {
  
}

void nbd::schur_cmplm_low(Matrices& S_oo, const Matrices& A_oc, const CSC& rels_oc, const Matrices& A_co, const CSC& rels_co) {
  for (int64_t x = 0; x < rels_oc.N; x++) {
    int64_t xx;
    lookupIJ(xx, rels_co, x, x);
    const Matrix& A_xx = A_co[xx];
    for (int64_t yx = rels_oc.CSC_COLS[x]; yx < rels_oc.CSC_COLS[x + 1]; yx++) {
      const Matrix& A_yx = A_oc[yx];
      Matrix& S_yx = S_oo[yx];
      mmult('N', 'N', A_yx, A_xx, S_yx, -1., 1.);
    }
  }
}

void nbd::schur_cmplm_up(Matrices& S_oo, const Matrices& A_oc, const CSC& rels_oc, const Matrices& A_co, const CSC& rels_co) {
  
}

void nbd::schur_cmplm_diag(Matrices& S_oo, const Matrices& A_oc, const Matrices& A_co, const CSC& rels) {
  schur_cmplm_up(S_oo, A_oc, rels, A_co, rels);

  for (int64_t x = 0; x < rels.N; x++) {
    int64_t xx;
    lookupIJ(xx, rels, x, x);
    const Matrix& A_xx = A_co[xx];
    for (int64_t yx = rels.CSC_COLS[x]; yx < rels.CSC_COLS[x + 1]; yx++) {
      int64_t y = rels.CSC_ROWS[yx];
      if (y == x)
        continue;
      const Matrix& A_yx = A_oc[yx];
      Matrix& S_yx = S_oo[yx];
      mmult('N', 'N', A_yx, A_xx, S_yx, -1., 1.);
    }
  }
}

void nbd::A_cc_fw(Vectors& Xc, const Matrices& A_cc, const CSC& rels) {
  for (int64_t i = 0; i < rels.M; i++) {
    int64_t ii;
    lookupIJ(ii, rels, i, i);
    const Matrix& A_ii = A_cc[ii];
    fw_solve(Xc[i], A_ii);

    for (int64_t yi = rels.CSC_COLS[i]; yi < rels.CSC_COLS[i + 1]; yi++) {
      int64_t y = rels.CSC_ROWS[yi];
      const Matrix& A_yi = A_cc[yi];
      if (y > i)
        mvec('N', A_yi, Xc[i], Xc[y], -1., 1.);
    }
  }
}

void nbd::A_cc_bk(Vectors& Xc, const Matrices& A_cc, const CSC& rels) {
  
}

void nbd::A_oc_fw(Vectors& Xo, const Matrices& A_oc, const CSC& rels, const Vectors& Xc) {
  for (int64_t x = 0; x < rels.N; x++)
    for (int64_t yx = rels.CSC_COLS[x]; yx < rels.CSC_COLS[x + 1]; yx++) {
      int64_t y = rels.CSC_ROWS[yx];
      const Matrix& A_yx = A_oc[yx];
      mvec('N', A_yx, Xc[x], Xo[y], -1., 1.);
    }
}


void nbd::A_co_bk(Vectors& Xc, const Matrices& A_co, const CSC& rels, const Vectors& Xo) {
  for (int64_t x = 0; x < rels.N; x++)
    for (int64_t yx = rels.CSC_COLS[x]; yx < rels.CSC_COLS[x + 1]; yx++) {
      int64_t y = rels.CSC_ROWS[yx];
      const Matrix& A_yx = A_co[yx];
      mvec('N', A_yx, Xo[x], Xc[y], -1., 1.);
    }
}
