
#include "solver.hxx"
#include "h2mv.hxx"
#include "dist.hxx"

using namespace nbd;

void nbd::basisXoc(char fwbk, RHS& vx, const Base& basis, int64_t level) {
  int64_t len = basis.DIMS.size();
  int64_t lbegin = 0;
  int64_t lend = len;
  selfLocalRange(lbegin, lend, level);
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


void nbd::svAccFw(Vectors& Xc, const Matrices& A_cc, const CSC& rels, int64_t level) {
  int64_t lbegin = rels.CBGN;
  int64_t ibegin = 0, iend;
  selfLocalRange(ibegin, iend, level);
  recvFwSubstituted(Xc, level);

  Vector* xlocal = &Xc[ibegin];
  for (int64_t i = 0; i < rels.N; i++) {
    int64_t ii;
    lookupIJ(ii, rels, i + lbegin, i + lbegin);
    const Matrix& A_ii = A_cc[ii];
    fw_solve(xlocal[i], A_ii);

    for (int64_t yi = rels.CSC_COLS[i]; yi < rels.CSC_COLS[i + 1]; yi++) {
      int64_t y = rels.CSC_ROWS[yi];
      const Matrix& A_yi = A_cc[yi];
      if (y > i + lbegin) {
        int64_t box_y = y;
        neighborsILocal(box_y, y, level);
        mvec('N', A_yi, xlocal[i], Xc[box_y], -1., 1.);
      }
    }
  }

  sendFwSubstituted(Xc, level);
}

void nbd::svAccBk(Vectors& Xc, const Matrices& A_cc, const CSC& rels, int64_t level) {
  int64_t lbegin = rels.CBGN;
  int64_t ibegin = 0, iend;
  selfLocalRange(ibegin, iend, level);
  recvBkSubstituted(Xc, level);

  Vector* xlocal = &Xc[ibegin];
  for (int64_t i = rels.N - 1; i >= 0; i--) {
    for (int64_t yi = rels.CSC_COLS[i]; yi < rels.CSC_COLS[i + 1]; yi++) {
      int64_t y = rels.CSC_ROWS[yi];
      const Matrix& A_yi = A_cc[yi];
      if (y > i + lbegin) {
        int64_t box_y = y;
        neighborsILocal(box_y, y, level);
        mvec('T', A_yi, Xc[box_y], xlocal[i], -1., 1.);
      }
    }

    int64_t ii;
    lookupIJ(ii, rels, i + lbegin, i + lbegin);
    const Matrix& A_ii = A_cc[ii];
    bk_solve(xlocal[i], A_ii);
  }
  
  sendBkSubstituted(Xc, level);
}

void nbd::svAocFw(Vectors& Xo, const Vectors& Xc, const Matrices& A_oc, const CSC& rels, int64_t level) {
  int64_t ibegin = 0, iend;
  selfLocalRange(ibegin, iend, level);
  const Vector* xlocal = &Xc[ibegin];

  for (int64_t x = 0; x < rels.N; x++)
    for (int64_t yx = rels.CSC_COLS[x]; yx < rels.CSC_COLS[x + 1]; yx++) {
      int64_t y = rels.CSC_ROWS[yx];
      int64_t box_y = y;
      neighborsILocal(box_y, y, level);
      const Matrix& A_yx = A_oc[yx];
      mvec('N', A_yx, xlocal[x], Xo[box_y], -1., 1.);
    }
  distributeSubstituted(Xo, level);
}

void nbd::svAocBk(Vectors& Xc, const Vectors& Xo, const Matrices& A_oc, const CSC& rels, int64_t level) {
  int64_t ibegin = 0, iend;
  selfLocalRange(ibegin, iend, level);
  Vector* xlocal = &Xc[ibegin];

  for (int64_t x = 0; x < rels.N; x++)
    for (int64_t yx = rels.CSC_COLS[x]; yx < rels.CSC_COLS[x + 1]; yx++) {
      int64_t y = rels.CSC_ROWS[yx];
      int64_t box_y = y;
      neighborsILocal(box_y, y, level);
      const Matrix& A_yx = A_oc[yx];
      mvec('T', A_yx, Xo[box_y], xlocal[x], -1., 1.);
    }
}

void nbd::allocRightHandSides(RHSS& rhs, const Base base[], int64_t levels) {
  rhs.resize(levels + 1);
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nodes = base[i].DIMS.size();
    RHS& rhs_i = rhs[i];
    Vectors& ix = rhs_i.X;
    Vectors& ixc = rhs_i.Xc;
    Vectors& ixo = rhs_i.Xo;
    ix.resize(nodes);
    ixc.resize(nodes);
    ixo.resize(nodes);

    for (int64_t n = 0; n < nodes; n++) {
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
}

void nbd::solveA(RHS X[], const Node A[], const Base B[], const CSC rels[], int64_t levels) {
  for (int64_t i = levels; i > 0; i--) {
    basisXoc('F', X[i], B[i], i);
    svAccFw(X[i].Xc, A[i].A_cc, rels[i], i);
    svAocFw(X[i].Xo, X[i].Xc, A[i].A_oc, rels[i], i);
    permuteAndMerge('F', X[i].X, X[i - 1].Xo, i - 1);
  }
  chol_solve(X[0].X[0], A[0].A[0]);
  
  for (int64_t i = 1; i <= levels; i++) {
    permuteAndMerge('B', X[i].X, X[i - 1].Xo, i - 1);
    svAocBk(X[i].Xc, X[i].Xo, A[i].A_oc, rels[i], i);
    svAccBk(X[i].Xc, A[i].A_cc, rels[i], i);
    basisXoc('B', X[i], B[i], i);
  }
}

void nbd::solveRelErr(double* err_out, const Vectors& X, const Vectors& ref, int64_t level) {
  int64_t ibegin = 0;
  int64_t iend = ref.size();
  selfLocalRange(ibegin, iend, level);
  double err = 0.;
  double nrm = 0.;

  for (int64_t i = ibegin; i < iend; i++) {
    double e, n;
    verr2(X[i], ref[i], &e);
    vnrm2(ref[i], &n);
    err = err + e;
    nrm = nrm + n;
  }

  *err_out = err / nrm;
}
