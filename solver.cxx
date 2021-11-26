
#include "solver.hxx"
#include "dist.hxx"

#include <cstdio>

using namespace nbd;

void nbd::basisXoc(char fwbk, RHS& vx, const Base& basis, const GlobalIndex& gi) {
  int64_t lbegin = gi.SELF_I * gi.BOXES;
  int64_t lend = lbegin + gi.BOXES;
  Vectors& X = vx.X;
  Vectors& Xo = vx.X_o;
  Vectors& Xc = vx.X_c;

  if (fwbk == 'F' || fwbk == 'f')
    for (int64_t i = lbegin; i < lend; i++)
      pvc_fw(X[i], basis.Uo[i], basis.Uc[i], Xo[i], Xc[i]);
  else if (fwbk == 'B' || fwbk == 'b')
    for (int64_t i = lbegin; i < lend; i++)
      pvc_bk(Xo[i], Xc[i], basis.Uo[i], basis.Uc[i], X[i]);
}


void nbd::svAcc(char fwbk, Vectors& Xc, const Matrices& A_cc, const GlobalIndex& gi) {
  const CSC& rels = gi.RELS;
  int64_t lbegin = gi.GBEGIN;
  Vector* xlocal = &Xc[gi.SELF_I * gi.BOXES];

  if (fwbk == 'F' || fwbk == 'f')
    for (int64_t i = 0; i < rels.N; i++) {
      int64_t ii;
      lookupIJ(ii, rels, i + lbegin, i);
      const Matrix& A_ii = A_cc[ii];
      fw_solve(xlocal[i], A_ii);

      for (int64_t yi = rels.CSC_COLS[i]; yi < rels.CSC_COLS[i + 1]; yi++) {
        int64_t y = rels.CSC_ROWS[yi];
        const Matrix& A_yi = A_cc[yi];
        if (y > i + lbegin)
          mvec('N', A_yi, xlocal[i], xlocal[y], -1., 1.);
      }
    }
  else if (fwbk == 'B' || fwbk == 'b')
    for (int64_t i = rels.N - 1; i >= 0; i--) {
      for (int64_t yi = rels.CSC_COLS[i]; yi < rels.CSC_COLS[i + 1]; yi++) {
        int64_t y = rels.CSC_ROWS[yi];
        const Matrix& A_yi = A_cc[yi];
        if (y > i + lbegin)
          mvec('T', A_yi, xlocal[y], xlocal[i], -1., 1.);
      }

      int64_t ii;
      lookupIJ(ii, rels, i + lbegin, i);
      const Matrix& A_ii = A_cc[ii];
      bk_solve(xlocal[i], A_ii);
    }
}

void nbd::svAocFw(Vectors& Xo, const Vectors& Xc, const Matrices& A_oc, const GlobalIndex& gi) {
  const CSC& rels = gi.RELS;
  int64_t lbegin = gi.GBEGIN;
  const Vector* xlocal = &Xc[gi.SELF_I * gi.BOXES];

  for (int64_t x = 0; x < rels.N; x++)
    for (int64_t yx = rels.CSC_COLS[x]; yx < rels.CSC_COLS[x + 1]; yx++) {
      int64_t y = rels.CSC_ROWS[yx];
      int64_t box_y;
      Lookup_GlobalI(box_y, gi, y);
      const Matrix& A_yx = A_oc[yx];
      mvec('N', A_yx, xlocal[x], Xo[box_y], -1., 1.);
    }
}

void nbd::svAocBk(Vectors& Xc, const Vectors& Xo, const Matrices& A_oc, const GlobalIndex& gi) {
  const CSC& rels = gi.RELS;
  int64_t lbegin = gi.GBEGIN;
  Vector* xlocal = &Xc[gi.SELF_I * gi.BOXES];

  for (int64_t x = 0; x < rels.N; x++)
    for (int64_t yx = rels.CSC_COLS[x]; yx < rels.CSC_COLS[x + 1]; yx++) {
      int64_t y = rels.CSC_ROWS[yx];
      int64_t box_y;
      Lookup_GlobalI(box_y, gi, y);
      const Matrix& A_yx = A_oc[yx];
      mvec('T', A_yx, Xo[box_y], xlocal[x], -1., 1.);
    }
}


