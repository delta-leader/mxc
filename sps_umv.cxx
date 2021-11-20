
#include "sps_umv.hxx"

using namespace nbd;

void nbd::splitA(Matrices& A_out, const GlobalIndex& gi, const Matrices& A, const Matrices& U, const Matrices& V) {
  const CSC& rels = gi.RELS;
  const Matrix* vlocal = &V[gi.SELF_I * gi.BOXES];

  for (int64_t x = 0; x < rels.N; x++) {
    for (int64_t yx = rels.CSC_COLS[x]; yx < rels.CSC_COLS[x + 1]; yx++) {
      int64_t y = rels.CSC_ROWS[yx];
      int64_t box_y;
      Lookup_GlobalI(box_y, gi, y);
      utav(U[box_y], A[yx], vlocal[x], A_out[yx]);
    }
  }
}

void nbd::factorAcc(Matrices& A_cc, const GlobalIndex& gi) {
  const CSC& rels = gi.RELS;
  int64_t lbegin = gi.SELF_I * gi.BOXES;

  for (int64_t i = 0; i < rels.N; i++) {
    int64_t ii;
    lookupIJ(ii, rels, i + lbegin, i);
    Matrix& A_ii = A_cc[ii];
    chol_decomp(A_ii);

    for (int64_t yi = rels.CSC_COLS[i]; yi < rels.CSC_COLS[i + 1]; yi++) {
      int64_t y = rels.CSC_ROWS[yi];
      if (y > i + lbegin)
        trsm_lowerA(A_cc[yi], A_ii);
    }
  }
}

void factorAoc(Matrices& A_oc, const Matrices& A_cc, const GlobalIndex& gi) {
  const CSC& rels = gi.RELS;
  int64_t lbegin = gi.SELF_I * gi.BOXES;

  for (int64_t i = 0; i < rels.N; i++) {
    int64_t ii;
    lookupIJ(ii, rels, i + lbegin, i);
    const Matrix& A_ii = A_cc[ii];
    for (int64_t yi = rels.CSC_COLS[i]; yi < rels.CSC_COLS[i + 1]; yi++)
      trsm_lowerA(A_oc[yi], A_ii);
  }
}

void nbd::schurCmplm(Matrices& S, const Matrices& A_oc, const GlobalIndex& gi) {
  const CSC& rels = gi.RELS;
  int64_t lbegin = gi.SELF_I * gi.BOXES;

  for (int64_t i = 0; i < rels.N; i++) {
    int64_t ii;
    lookupIJ(ii, rels, i + lbegin, i);
    const Matrix& A_iit = A_oc[ii];
    for (int64_t yi = rels.CSC_COLS[i]; yi < rels.CSC_COLS[i + 1]; yi++) {
      const Matrix& A_yi = A_oc[yi];
      Matrix& S_yi = S[yi];
      mmult('N', 'T', A_yi, A_iit, S_yi, -1., 0.);
    }
  }
}

void nbd::axatLocal(Matrices& A, const GlobalIndex& gi) {
  const CSC& rels = gi.RELS;
  int64_t lbegin = gi.SELF_I * gi.BOXES;

  for (int64_t i = 0; i < rels.N; i++)
    for (int64_t ji = rels.CSC_COLS[i]; ji < rels.CSC_COLS[i + 1]; ji++) {
      int64_t j = rels.CSC_ROWS[ji];
      if (j > i + lbegin) {
        Matrix& A_ji = A[ji];
        int64_t ij;
        lookupIJ(ij, rels, i + lbegin, j - lbegin);
        Matrix& A_ij = A[ij];
        axat(A_ji, A_ij);
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
