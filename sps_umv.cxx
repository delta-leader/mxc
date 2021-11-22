
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

void nbd::factorAoc(Matrices& A_oc, const Matrices& A_cc, const GlobalIndex& gi) {
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
  nodes.resize(domain.MY_LEVEL + domain.LOCAL_LEVELS + 1);
  for (int64_t i = 0; i < nodes.size(); i++) {
    int64_t nnz = domain.MY_IDS[i].RELS.NNZ;
    nodes[i].A.resize(nnz);
    nodes[i].A_cc.resize(nnz);
    nodes[i].A_oc.resize(nnz);
    nodes[i].A_oo.resize(nnz);
    nodes[i].S.resize(nnz);
  }
  return &(nodes.back().A);
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

void nbd::factorNode(Node& n, const GlobalIndex& gi, const Base& basis) {
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

void nbd::nextNode(Node& Anext, const GlobalIndex& Gnext, const Node& Aprev, const GlobalIndex& Gprev) {
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
