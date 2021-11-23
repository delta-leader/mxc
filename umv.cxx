
#include "umv.hxx"
#include "dist.hxx"

#include <cstdio>

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
  int64_t lbegin = gi.GBEGIN;

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

void nbd::factorAoc(Matrices& A_oc, const Matrices& A_cc, const GlobalIndex& gi) {
  const CSC& rels = gi.RELS;
  int64_t lbegin = gi.GBEGIN;

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
  int64_t lbegin = gi.GBEGIN;

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
  int64_t lbegin = gi.GBEGIN;
  int64_t lend = lbegin + gi.BOXES;

  for (int64_t i = 0; i < rels.N; i++)
    for (int64_t ji = rels.CSC_COLS[i]; ji < rels.CSC_COLS[i + 1]; ji++) {
      int64_t j = rels.CSC_ROWS[ji];
      if (j > i + lbegin && j < lend) {
        Matrix& A_ji = A[ji];
        int64_t ij;
        lookupIJ(ij, rels, i + lbegin, j - lbegin);
        Matrix& A_ij = A[ij];
        axat(A_ji, A_ij);
      }
    }
}

Matrices* nbd::allocNodes(Nodes& nodes, const LocalDomain& domain) {
  nodes.resize(domain.size());
  for (int64_t i = 0; i < nodes.size(); i++) {
    int64_t nnz = domain[i].RELS.NNZ;
    nodes[i].A.resize(nnz);
    nodes[i].A_cc.resize(nnz);
    nodes[i].A_oc.resize(nnz);
    nodes[i].A_oo.resize(nnz);
    nodes[i].S.resize(nnz);
  }
  return &(nodes.back().A);
}

void nbd::allocA(Node& n, const GlobalIndex& gi, const int64_t* dims) {
  const CSC& rels = gi.RELS;
  int64_t lbegin = gi.SELF_I * gi.BOXES;

  for (int64_t j = 0; j < rels.N; j++) {
    int64_t box_j = lbegin + j;
    int64_t nbodies_j = dims[box_j];

    for (int64_t ij = rels.CSC_COLS[j]; ij < rels.CSC_COLS[j + 1]; ij++) {
      int64_t i = rels.CSC_ROWS[ij];
      int64_t box_i;
      Lookup_GlobalI(box_i, gi, i);
      int64_t nbodies_i = dims[box_i];

      Matrix& A_ij = n.A[ij];
      cMatrix(A_ij, nbodies_i, nbodies_j);
    }
  }
}

void nbd::allocSubMatrices(Node& n, const GlobalIndex& gi, const int64_t* dims, const int64_t* dimo) {
  const CSC& rels = gi.RELS;
  int64_t nboxes = gi.BOXES;

  for (int64_t j = 0; j < rels.N; j++) {
    int64_t box_j = gi.SELF_I * nboxes + j;
    int64_t dimo_j = dimo[box_j];
    int64_t dimc_j = dims[box_j] - dimo_j;

    for (int64_t ij = rels.CSC_COLS[j]; ij < rels.CSC_COLS[j + 1]; ij++) {
      int64_t i = rels.CSC_ROWS[ij];
      int64_t box_i;
      Lookup_GlobalI(box_i, gi, i);
      int64_t dimo_i = dimo[box_i];
      int64_t dimc_i = dims[box_i] - dimo_i;

      cMatrix(n.A_cc[ij], dimc_i, dimc_j);
      cMatrix(n.A_oc[ij], dimo_i, dimc_j);
      cMatrix(n.A_oo[ij], dimo_i, dimo_j);
      cMatrix(n.S[ij], dimo_i, dimo_j);
    }
  }
}

void nbd::factorNode(Node& n, Base& basis, const GlobalIndex& gi, double repi, const double* R, int64_t lenR) {
  sampleA(basis, repi, gi, n.A, R, lenR);
  
  allocSubMatrices(n, gi, basis.DIMS.data(), basis.DIMO.data());
  splitA(n.A_cc, gi, n.A, basis.Uc, basis.Uc);
  splitA(n.A_oc, gi, n.A, basis.Uo, basis.Uc);
  splitA(n.A_oo, gi, n.A, basis.Uo, basis.Uo);

  factorAcc(n.A_cc, gi);
  factorAoc(n.A_oc, n.A_cc, gi);
  schurCmplm(n.S, n.A_oc, gi);

  axatLocal(n.S, gi);
  axatDistribute(n.S, gi);

  for (int64_t i = 0; i < n.S.size(); i++)
    madd(n.A_oo[i], n.S[i]);
}

void nbd::nextNode(Node& Anext, Base& bsnext, const GlobalIndex& Gnext, const Node& Aprev, const Base& bsprev, const GlobalIndex& Gprev) {
  Matrices& Mup = Anext.A;
  const Matrices& Mlow = Aprev.A_oo;
  const CSC& rels_up = Gnext.RELS;
  const CSC& rels_low = Gprev.RELS;

  for (int64_t j = 0; j < rels_up.N; j++)
    for (int64_t ij = rels_up.CSC_COLS[j]; ij < rels_up.CSC_COLS[j + 1]; ij++) {
      int64_t i = rels_up.CSC_ROWS[ij];
      zeroMatrix(Mup[ij]);
      
      int64_t i00, i01, i10, i11;
      lookupIJ(i00, rels_low, i << 1, j << 1);
      lookupIJ(i01, rels_low, i << 1, (j << 1) + 1);
      lookupIJ(i10, rels_low, (i << 1) + 1, j << 1);
      lookupIJ(i11, rels_low, (i << 1) + 1, (j << 1) + 1);

      if (i00 > 0) {
        const Matrix& m00 = Mlow[i00];
        cpyMatToMat(m00.M, m00.N, m00, Mup[ij], 0, 0, 0, 0);
      }

      if (i01 > 0) {
        const Matrix& m01 = Mlow[i01];
        cpyMatToMat(m01.M, m01.N, m01, Mup[ij], 0, 0, 0, Mup[ij].N - m01.N);
      }

      if (i10 > 0) {
        const Matrix& m10 = Mlow[i10];
        cpyMatToMat(m10.M, m10.N, m10, Mup[ij], 0, 0, Mup[ij].M - m10.M, 0);
      }

      if (i11 > 0) {
        const Matrix& m11 = Mlow[i11];
        cpyMatToMat(m11.M, m11.N, m11, Mup[ij], 0, 0, Mup[ij].M - m11.M, Mup[ij].N - m11.N);
      }
    }
}

void nbd::factorA(Nodes& A, Basis& B, const LocalDomain& domain, double repi, const double* R, int64_t lenR) {
  for (int64_t i = domain.size() - 1; i > 0; i--) {
    const GlobalIndex& gi = domain[i];
    Node& Ai = A[i];
    Base& Bi = B[i];
    factorNode(Ai, Bi, gi, repi, R, lenR);

    const GlobalIndex& gn = domain[i - 1];
    Node& An = A[i - 1];
    Base& Bn = B[i - 1];
    nextNode(An, Bn, gn, Ai, Bi, gi);
  }

  chol_decomp(A[0].A[0]);
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


