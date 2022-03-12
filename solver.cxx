
#include "solver.hxx"
#include "dist.hxx"

using namespace nbd;

void nbd::basisXoc(char fwbk, RHS& vx, const Base& basis, const GlobalIndex& gi) {
  int64_t lbegin = gi.SELF_I * gi.BOXES;
  int64_t lend = lbegin + gi.BOXES;
  Vectors& X = vx.X;
  Vectors& Xo = vx.Xo;
  Vectors& Xc = vx.Xc;

  if (fwbk == 'F' || fwbk == 'f')
    for (int64_t i = lbegin; i < lend; i++)
      pvc_fw(X[i], basis.Uo[i], basis.Uc[i], Xo[i], Xc[i]);
  else if (fwbk == 'B' || fwbk == 'b')
    for (int64_t i = lbegin; i < lend; i++)
      pvc_bk(Xo[i], Xc[i], basis.Uo[i], basis.Uc[i], X[i]);
}


void nbd::svAccFw(Vectors& Xc, const Matrices& A_cc, const GlobalIndex& gi) {
  const CSC& rels = gi.RELS;
  int64_t lbegin = rels.CBGN;
  Vector* xlocal = &Xc[gi.SELF_I * gi.BOXES];
  recvFwSubstituted(Xc, gi.LEVEL);

  for (int64_t i = 0; i < rels.N; i++) {
    int64_t ii;
    lookupIJ(ii, rels, i + lbegin, i + lbegin);
    const Matrix& A_ii = A_cc[ii];
    fw_solve(xlocal[i], A_ii);

    for (int64_t yi = rels.CSC_COLS[i]; yi < rels.CSC_COLS[i + 1]; yi++) {
      int64_t y = rels.CSC_ROWS[yi];
      const Matrix& A_yi = A_cc[yi];
      if (y > i + lbegin) {
        int64_t box_y;
        Lookup_GlobalI(box_y, gi, y);
        mvec('N', A_yi, xlocal[i], Xc[box_y], -1., 1.);
      }
    }
  }

  sendFwSubstituted(Xc, gi.LEVEL);
}

void nbd::svAccBk(Vectors& Xc, const Matrices& A_cc, const GlobalIndex& gi) {
  const CSC& rels = gi.RELS;
  int64_t lbegin = rels.CBGN;
  Vector* xlocal = &Xc[gi.SELF_I * gi.BOXES];
  recvBkSubstituted(Xc, gi.LEVEL);

  for (int64_t i = rels.N - 1; i >= 0; i--) {
    for (int64_t yi = rels.CSC_COLS[i]; yi < rels.CSC_COLS[i + 1]; yi++) {
      int64_t y = rels.CSC_ROWS[yi];
      const Matrix& A_yi = A_cc[yi];
      if (y > i + lbegin) {
        int64_t box_y;
        Lookup_GlobalI(box_y, gi, y);
        mvec('T', A_yi, Xc[box_y], xlocal[i], -1., 1.);
      }
    }

    int64_t ii;
    lookupIJ(ii, rels, i + lbegin, i + lbegin);
    const Matrix& A_ii = A_cc[ii];
    bk_solve(xlocal[i], A_ii);
  }
  
  sendBkSubstituted(Xc, gi.LEVEL);
}

void nbd::svAocFw(Vectors& Xo, const Vectors& Xc, const Matrices& A_oc, const GlobalIndex& gi) {
  const CSC& rels = gi.RELS;
  const Vector* xlocal = &Xc[gi.SELF_I * gi.BOXES];

  for (int64_t x = 0; x < rels.N; x++)
    for (int64_t yx = rels.CSC_COLS[x]; yx < rels.CSC_COLS[x + 1]; yx++) {
      int64_t y = rels.CSC_ROWS[yx];
      int64_t box_y;
      Lookup_GlobalI(box_y, gi, y);
      const Matrix& A_yx = A_oc[yx];
      mvec('N', A_yx, xlocal[x], Xo[box_y], -1., 1.);
    }
  distributeSubstituted(Xo, gi.LEVEL);
}

void nbd::svAocBk(Vectors& Xc, const Vectors& Xo, const Matrices& A_oc, const GlobalIndex& gi) {
  const CSC& rels = gi.RELS;
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
    int64_t nboxes = base[i].DIMS.size();
    RHS& rhs_i = rhs[i];
    Vectors& ix = rhs_i.X;
    Vectors& ixc = rhs_i.Xc;
    Vectors& ixo = rhs_i.Xo;
    ix.resize(nboxes);
    ixc.resize(nboxes);
    ixo.resize(nboxes);

    for (int64_t n = 0; n < nboxes; n++) {
      int64_t dim = base[i].DIMS[n];
      int64_t dim_o = base[i].DIMO[n];
      int64_t dim_c = dim - dim_o;
      cVector(ix[n], dim);
      cVector(ixc[n], dim_c);
      cVector(ixo[n], dim_o);
      zeroVector(ix[n]);
      zeroVector(ixc[n]);
      zeroVector(ixo[n]);
    }
  }

  int64_t lbegin = domain.back().SELF_I * domain.back().BOXES;
  return &(rhs.back().X[lbegin]);
}

void nbd::permuteAndMerge(char fwbk, RHS& prev, const GlobalIndex& gprev, RHS& next, const GlobalIndex& gnext) {
  int64_t nbegin = gnext.RELS.CBGN;
  int64_t pbegin = gprev.RELS.CBGN;
  int64_t nboxes = gnext.BOXES;
  int64_t pboxes = gprev.BOXES;
  int64_t nloc = gnext.SELF_I * nboxes;
  int64_t ploc = gprev.SELF_I * pboxes;

  if (fwbk == 'F' || fwbk == 'f') {
    for (int64_t i = 0; i < nboxes; i++) {
      int64_t p = (i + nbegin) << 1;
      int64_t c0 = p - pbegin;
      int64_t c1 = p + 1 - pbegin;
      Vector& x0 = next.X[i + nloc];

      if (c0 >= 0 && c0 < pboxes) {
        const Vector& x1 = prev.Xo[c0 + ploc];
        cpyVecToVec(x1.N, x1, x0, 0, 0);
      }

      if (c1 >= 0 && c1 < pboxes) {
        const Vector& x2 = prev.Xo[c1 + ploc];
        cpyVecToVec(x2.N, x2, x0, 0, x0.N - x2.N);
      }
    }

    if (nboxes == pboxes)
      butterflySumX(next.X, gprev.LEVEL);
  }
  else if (fwbk == 'B' || fwbk == 'b') {
    for (int64_t i = 0; i < nboxes; i++) {
      int64_t p = (i + nbegin) << 1;
      int64_t c0 = p - pbegin;
      int64_t c1 = p + 1 - pbegin;
      const Vector& x0 = next.X[i + nloc];

      if (c0 >= 0 && c0 < pboxes) {
        Vector& x1 = prev.Xo[c0 + ploc];
        cpyVecToVec(x1.N, x0, x1, 0, 0);
      }

      if (c1 >= 0 && c1 < pboxes) {
        Vector& x2 = prev.Xo[c1 + ploc];
        cpyVecToVec(x2.N, x0, x2, x0.N - x2.N, 0);
      }
    }

    DistributeVectorsList(prev.Xo, gprev.LEVEL);
  }
}

void nbd::solveA(RHSS& X, const Nodes& A, const Basis& B, const LocalDomain& domain) {
  int64_t lvl = domain.size();
  for (int64_t i = lvl - 1; i > 0; i--) {
    basisXoc('F', X[i], B[i], domain[i]);
    svAccFw(X[i].Xc, A[i].A_cc, domain[i]);
    svAocFw(X[i].Xo, X[i].Xc, A[i].A_oc, domain[i]);
    permuteAndMerge('F', X[i], domain[i], X[i - 1], domain[i - 1]);
  }
  chol_solve(X[0].X[0], A[0].A[0]);
  
  for (int64_t i = 1; i < lvl; i++) {
    permuteAndMerge('B', X[i], domain[i], X[i - 1], domain[i - 1]);
    svAocBk(X[i].Xc, X[i].Xo, A[i].A_oc, domain[i]);
    svAccBk(X[i].Xc, A[i].A_cc, domain[i]);
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
