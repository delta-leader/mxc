
#include "basis.hxx"
#include "dist.hxx"

using namespace nbd;

void nbd::sampleC1(Matrix* CL, const CSC& rels, const Matrices& A, const double* R, int64_t lenR) {
  int64_t lbegin = rels.CBGN;
  for (int64_t j = 0; j < rels.N; j++) {
    Matrix& cj = CL[j];

    for (int64_t ij = rels.CSC_COLS[j]; ij < rels.CSC_COLS[j + 1]; ij++) {
      int64_t i = rels.CSC_ROWS[ij];
      if (i == j + lbegin)
        continue;
      const Matrix& Aij = A[ij];
      msample('T', lenR, Aij, R, cj);
    }

    int64_t jj;
    lookupIJ(jj, rels, j + lbegin, j + lbegin);
    const Matrix& Ajj = A[jj];
    minvl(Ajj, cj);
  }
}


void nbd::sampleC2(Matrix* CL, const CSC& rels, const Matrices& A, const Matrices& C1, const int64_t ngbs[], int64_t ngb_len) {
  int64_t lbegin = rels.CBGN;
  for (int64_t j = 0; j < rels.N; j++) {
    Matrix& cj = CL[j];

    for (int64_t ij = rels.CSC_COLS[j]; ij < rels.CSC_COLS[j + 1]; ij++) {
      int64_t i = rels.CSC_ROWS[ij];
      if (i == j + lbegin)
        continue;
      int64_t box_i;
      Lookup_GlobalI(box_i, i, rels.N, ngbs, ngb_len);

      const Matrix& Aij = A[ij];
      msample_m('T', Aij, C1[box_i], cj);
    }
  }
}

void nbd::orthoBasis(double repi, int64_t N, Matrix* C, int64_t* dims_o) {
  for (int64_t i = 0; i < N; i++)
    orthoBase(repi, C[i], &dims_o[i]);
}

void nbd::allocBasis(Basis& basis, const LocalDomain& domain, const int64_t* bddims) {
  basis.resize(domain.size());
  for (int64_t i = 0; i < basis.size(); i++) {
    int64_t boxes = domain[i].BOXES;
    int64_t nodes = boxes * domain[i].NGB_RNKS.size();
    basis[i].LBOXES = boxes;
    basis[i].LBGN = domain[i].SELF_I * boxes;
    basis[i].DIMS.resize(nodes);
    basis[i].DIMO.resize(nodes);
    basis[i].Uo.resize(nodes);
    basis[i].Uc.resize(nodes);
  }

  Base& leaf = basis.back();
  int64_t nodes = leaf.DIMS.size();
  std::copy(bddims, bddims + nodes, leaf.DIMS.begin());
}

void nbd::allocUcUo(Base& basis, const Matrices& C) {
  int64_t lbegin = basis.LBGN;
  int64_t lend = lbegin + basis.LBOXES;
  for (int64_t i = 0; i < basis.DIMS.size(); i++) {
    int64_t dim = basis.DIMS[i];
    int64_t dim_o = basis.DIMO[i];
    int64_t dim_c = dim - dim_o;

    Matrix& Uo_i = basis.Uo[i];
    Matrix& Uc_i = basis.Uc[i];
    cMatrix(Uo_i, dim, dim_o);
    cMatrix(Uc_i, dim, dim_c);

    if (i >= lbegin && i < lend) {
      const Matrix& U = C[i];
      cpyMatToMat(dim, dim_o, U, Uo_i, 0, 0, 0, 0);
      cpyMatToMat(dim, dim_c, U, Uc_i, 0, dim_o, 0, 0);
    }
  }
}

void nbd::sampleA(Base& basis, double repi, const GlobalIndex& gi, const Matrices& A, const double* R, int64_t lenR) {
  Matrices C1(basis.DIMS.size());
  Matrices C2(basis.DIMS.size());

  for (int64_t i = 0; i < basis.DIMS.size(); i++) {
    int64_t dim = basis.DIMS[i];
    cMatrix(C1[i], dim, dim);
    cMatrix(C2[i], dim, dim);
    zeroMatrix(C1[i]);
    zeroMatrix(C2[i]);
  }
  double ct;

  int64_t lbegin = basis.LBGN;
  sampleC1(&C1[lbegin], gi.RELS, A, R, lenR);
  startTimer(&ct);
  DistributeMatricesList(C1, gi.LEVEL);
  stopTimer(ct, "comm1 time");
  sampleC2(&C2[lbegin], gi.RELS, A, C1, &gi.NGB_RNKS[0], gi.NGB_RNKS.size());
  orthoBasis(repi, basis.LBOXES, &C2[lbegin], &basis.DIMO[lbegin]);
  startTimer(&ct);
  DistributeDims(basis.DIMO, gi.LEVEL);
  allocUcUo(basis, C2);
  
  DistributeMatricesList(basis.Uc, gi.LEVEL);
  DistributeMatricesList(basis.Uo, gi.LEVEL);
  stopTimer(ct, "comm2 time");
}

void nbd::nextBasisDims(Base& bsnext, const GlobalIndex& gnext, const Base& bsprev, const GlobalIndex& gprev) {
  int64_t nboxes = gnext.BOXES;
  int64_t nbegin = gnext.SELF_I * nboxes;
  int64_t ngbegin = gnext.RELS.CBGN;

  std::vector<int64_t>& dims = bsnext.DIMS;
  const std::vector<int64_t>& dimo = bsprev.DIMO;

  for (int64_t i = 0; i < nboxes; i++) {
    int64_t nloc = i + nbegin;
    int64_t nrnk = i + ngbegin;
    int64_t c0rnk = nrnk << 1;
    int64_t c1rnk = (nrnk << 1) + 1;
    int64_t c0, c1;
    Lookup_GlobalI(c0, gprev, c0rnk);
    Lookup_GlobalI(c1, gprev, c1rnk);
    dims[nloc] = dimo[c0] + dimo[c1];
  }
  DistributeDims(dims, gnext.LEVEL);
}

