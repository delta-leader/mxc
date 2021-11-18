
#include "sps_basis.hxx"

using namespace nbd;

void nbd::init_rows_sample(Matrices& C, int64_t M, const int64_t* DIMS) {
  C.resize(M);
  for (int64_t i = 0; i < M; i++) {
    cMatrix(C[i], DIMS[i], DIMS[i]);
    zeroMatrix(C[i]);
  }
}

void nbd::sample_rows(Matrices& C, int64_t lbegin, const CSC& rels, const Matrices& A, const double* R, int64_t lenR) {
  for (int64_t j = 0; j < rels.N; j++) {
    Matrix& cj = C[j];

    for (int64_t ij = rels.CSC_COLS[j]; ij < rels.CSC_COLS[j + 1]; ij++) {
      int64_t i = rels.CSC_ROWS[ij];
      if (i == j)
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


void nbd::sample_rows_invd(Matrices& C, const CSC& rels, const Matrices& A, const Matrices& spC) {
  for (int64_t j = 0; j < rels.N; j++) {
    Matrix& cj = C[j];

    for (int64_t ij = rels.CSC_COLS[j]; ij < rels.CSC_COLS[j + 1]; ij++) {
      int64_t i = rels.CSC_ROWS[ij];
      if (i == j)
        continue;
      const Matrix& Aij = A[ij];
      msample_m('T', Aij, spC[i], cj);
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

int64_t nbd::merge_dims(int64_t* dims, Matrices& Uo) {
  int64_t N = Uo.size() >> 1;
  for (int64_t i = 0; i < N; i++) {
    int64_t r1 = Uo[i << 1].N;
    int64_t r2 = Uo[(i << 1) + 1].N;
    dims[i] = r1 + r2;
  }
  return N;
}

void nbd::Alloc_basis(Basis& basis, const LocalDomain& domain) {
  basis.resize(domain.MY_LEVEL + domain.LOCAL_LEVELS);
  for (int64_t i = 0; i < basis.size(); i++) {
    
  }
}

/*void nbd::local_row_base(Base& basis, double repi, const Matrices& A, const double* R, const CSC& rels, int64_t lenR) {
  basis.N = rels.M;
  basis.DIMS.resize(rels.M);
  for (int64_t i = 0; i < basis.N; i++) {
    int64_t ii;
    lookupIJ(ii, rels, i, i);
    basis.DIMS[i] = A[ii].M;
  }

  Matrices C1, C2;
  init_rows_sample(C1, basis.N, basis.DIMS.data());
  init_rows_sample(C2, basis.N, basis.DIMS.data());

  sample_rows(C1, 0, rels, A, R, lenR);
  sample_rows_invd(C2, rels, A, C1);
  orth_row_basis(repi, basis.Uc, basis.Uo, C2);
}

void nbd::basis_fw(Vectors& Xo, Vectors& Xc, const Base& basis, const Vectors& X) {
  for (int64_t i = 0; i < basis.N; i++)
    pvc_fw(X[i], basis.Uo[i], basis.Uc[i], Xo[i], Xc[i]);
}

void nbd::basis_bk(Vectors& X, const Base& basis, const Vectors& Xo, const Vectors& Xc) {
  for (int64_t i = 0; i < basis.N; i++)
    pvc_bk(Xo[i], Xc[i], basis.Uo[i], basis.Uc[i], X[i]);
}*/
