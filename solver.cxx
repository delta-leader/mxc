
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

Vector* nbd::allocRightHandSides(RHSS& rhs, const Basis& base, const LocalDomain& domain) {
  rhs.resize(domain.size());
  for (int64_t i = 0; i < rhs.size(); i++) {
    int64_t nboxes = domain[i].BOXES;
    rhs[i].X.resize(nboxes);
    rhs[i].X_c.resize(nboxes);
    rhs[i].X_o.resize(nboxes);

    for (int64_t n = 0; n < nboxes; n++) {
      int64_t dim = base[i].DIMS[n];
      int64_t dim_o = base[i].DIMO[n];
      int64_t dim_c = dim - dim_o;
      cVector(rhs[i].X[n], dim);
      cVector(rhs[i].X_c[n], dim_c);
      cVector(rhs[i].X_o[n], dim_o);
    }
  }

  int64_t lbegin = domain.back().SELF_I * domain.back().BOXES;
  return &(rhs.back().X[lbegin]);
}

void nbd::permuteAndMerge(char fwbk, RHS& prev, RHS& next) {
  if (fwbk == 'F' || fwbk == 'f')
    for (int64_t i = 0; i < next.X.size(); i++) {
      int64_t c0 = i << 1;
      int64_t c1 = (i << 1) + 1;
      const Vector& x0 = prev.X_o[c0];
      const Vector& x1 = prev.X_o[c1];
      Vector& x2 = next.X[i];
      cpyVecToVec(x0.N, x0, x2, 0, 0);
      cpyVecToVec(x1.N, x1, x2, 0, x0.N);
    }
  else if (fwbk == 'B' || fwbk == 'b') 
    for (int64_t i = 0; i < next.X.size(); i++) {
      int64_t c0 = i << 1;
      int64_t c1 = (i << 1) + 1;
      Vector& x0 = prev.X_o[c0];
      Vector& x1 = prev.X_o[c1];
      const Vector& x2 = next.X[i];
      cpyVecToVec(x0.N, x2, x0, 0, 0);
      cpyVecToVec(x1.N, x2, x1, x0.N, 0);
    }
}

void nbd::solveA(RHSS& X, const Nodes& A, const Basis& B, const LocalDomain& domain) {
  int64_t lvl = domain.size();
  for (int64_t i = lvl - 1; i > 0; i--) {
    basisXoc('F', X[i], B[i], domain[i]);
    svAcc('F', X[i].X_c, A[i].A_cc, domain[i]);
    svAocFw(X[i].X_o, X[i].X_c, A[i].A_oc, domain[i]);
    permuteAndMerge('F', X[i], X[i - 1]);
  }
  chol_solve(X[0].X[0], A[0].A[0]);
  
  for (int64_t i = 1; i < lvl; i++) {
    permuteAndMerge('B', X[i], X[i - 1]);
    svAocBk(X[i].X_c, X[i].X_o, A[i].A_oc, domain[i]);
    svAcc('B', X[i].X_c, A[i].A_cc, domain[i]);
    basisXoc('B', X[i], B[i], domain[i]);
  }
}

void nbd::solveRelErr(double* err_out, const RHS& X, const Vectors& ref, const GlobalIndex& gi) {
  int64_t lbegin = gi.SELF_I * gi.BOXES;
  int64_t lend = lbegin + gi.BOXES;
  double err = 0.;
  double nrm = 0.;

  for (int64_t i = lbegin; i < lend; i++) {
    double e, n;
    verr2(X.X[i], ref[i], &e);
    vnrm2(ref[i], &n);
    err = err + e;
    nrm = nrm + n;
  }

  *err_out = err / nrm;
}
