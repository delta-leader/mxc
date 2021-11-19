
#include "sps_basis.hxx"

using namespace nbd;

void nbd::sampleC1(Matrices& C1, const GlobalIndex& gi, const CSC& rels, const Matrices& A, const double* R, int64_t lenR) {
  int64_t lbegin = gi.SELF_I * gi.BOXES;
  Matrix* C = &C1[lbegin];
  for (int64_t j = 0; j < rels.N; j++) {
    Matrix& cj = C[j];

    for (int64_t ij = rels.CSC_COLS[j]; ij < rels.CSC_COLS[j + 1]; ij++) {
      int64_t i = rels.CSC_ROWS[ij];
      if (i == j + lbegin)
        continue;
      const Matrix& Aij = A[ij];
      msample('T', lenR, Aij, R, cj);
    }

    int64_t jj;
    lookupIJ(jj, rels, lbegin + j, j);
    const Matrix& Ajj = A[jj];

    Matrix work;
    cMatrix(work, Ajj.M, Ajj.N);
    cpyMatToMat(Ajj.M, Ajj.N, Ajj, work, 0, 0, 0, 0);
    minv('N', 'L', work, cj);
  }
}


void nbd::sampleC2(Matrices& C2, const GlobalIndex& gi, const CSC& rels, const Matrices& A, const Matrices& C1) {
  int64_t lbegin = gi.SELF_I * gi.BOXES;
  Matrix* C = &C2[lbegin];
  for (int64_t j = 0; j < rels.N; j++) {
    Matrix& cj = C[j];

    for (int64_t ij = rels.CSC_COLS[j]; ij < rels.CSC_COLS[j + 1]; ij++) {
      int64_t i = rels.CSC_ROWS[ij];
      if (i == j + lbegin)
        continue;
      int64_t box_i;
      Lookup_GlobalI(box_i, gi, i);

      const Matrix& Aij = A[ij];
      msample_m('T', Aij, C1[box_i], cj);
    }
  }
}

void nbd::orth_row_basis(double repi, Matrices& Uc, Matrices& Uo, Matrices& C) {
  int64_t N = C.size();
  Uc.resize(N);
  Uo.resize(N);

  for (int64_t i = 0; i < N; i++)
    orthoBase(repi, C[i], Uo[i], Uc[i]);
}

void nbd::Alloc_basis(Basis& basis, const LocalDomain& domain) {
  basis.resize(domain.MY_LEVEL + domain.LOCAL_LEVELS);
  for (int64_t i = 0; i < basis.size(); i++) {
    int64_t nodes = domain.MY_IDS[i].BOXES * domain.MY_IDS[i].NGB_RNKS.size();
    basis[i].DIMS.resize(nodes);
    basis[i].DIMO.resize(nodes);
    basis[i].DIMC.resize(nodes);

    basis[i].Uo.resize(nodes);
    basis[i].Uc.resize(nodes);
    basis[i].C1.resize(nodes);
    basis[i].C2.resize(nodes);
  }
}

void nbd::Alloc_leaf_base(Base& leaf, const int64_t* bodies) {
  std::copy(bodies, bodies + leaf.DIMS.size(), leaf.DIMS.begin());
  for (int64_t i = 0; i < leaf.DIMS.size(); i++) {
    cMatrix(leaf.C1[i], leaf.DIMS[i], leaf.DIMS[i]);
    cMatrix(leaf.C2[i], leaf.DIMS[i], leaf.DIMS[i]);
    zeroMatrix(leaf.C1[i]);
    zeroMatrix(leaf.C2[i]);
  }
}

void nbd::sampleA(Base& basis, double repi, const CSC& rels, const Matrices& A, const double* R, int64_t lenR) {

}

void nbd::basis_fw(Vectors& Xo, Vectors& Xc, const Base& basis, const Vectors& X) {
  //for (int64_t i = 0; i < basis.N; i++)
  //  pvc_fw(X[i], basis.Uo[i], basis.Uc[i], Xo[i], Xc[i]);
}

void nbd::basis_bk(Vectors& X, const Base& basis, const Vectors& Xo, const Vectors& Xc) {
  //for (int64_t i = 0; i < basis.N; i++)
  //  pvc_bk(Xo[i], Xc[i], basis.Uo[i], basis.Uc[i], X[i]);
}
