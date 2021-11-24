
#include "basis.hxx"
#include "dist.hxx"

#include <cstdio>
using namespace nbd;

void nbd::sampleC1(Matrices& C1, const GlobalIndex& gi, const Matrices& A, const double* R, int64_t lenR) {
  const CSC& rels = gi.RELS;
  int64_t lbegin = gi.GBEGIN;
  Matrix* C = &C1[gi.SELF_I * gi.BOXES];
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
    //minvl(work, cj);
  }
}


void nbd::sampleC2(Matrices& C2, const GlobalIndex& gi, const Matrices& A, const Matrices& C1) {
  const CSC& rels = gi.RELS;
  int64_t lbegin = gi.GBEGIN;
  Matrix* C = &C2[gi.SELF_I * gi.BOXES];
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

void nbd::orthoBasis(double repi, const GlobalIndex& gi, Matrices& C, std::vector<int64_t>& dims_o) {
  int64_t lbegin = gi.SELF_I * gi.BOXES;
  int64_t lend = lbegin + gi.BOXES;
  for (int64_t i = lbegin; i < lend; i++)
    orthoBase(repi, C[i], &dims_o[i]);
}

void nbd::allocBasis(Basis& basis, const LocalDomain& domain, const int64_t* bddims) {
  basis.resize(domain.size());
  for (int64_t i = 0; i < basis.size(); i++) {
    int64_t nodes = domain[i].BOXES * domain[i].NGB_RNKS.size();
    basis[i].DIMS.resize(nodes);
    basis[i].DIMO.resize(nodes);
    basis[i].Uo.resize(nodes);
    basis[i].Uc.resize(nodes);
  }

  Base& leaf = basis.back();
  int64_t nodes = leaf.DIMS.size();
  std::copy(bddims, bddims + nodes, leaf.DIMS.begin());
}

void nbd::allocUcUo(Base& basis, const GlobalIndex& gi, const Matrices& C) {
  int64_t lbegin = gi.SELF_I * gi.BOXES;
  int64_t lend = lbegin + gi.BOXES;
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

  sampleC1(C1, gi, A, R, lenR);
  DistributeMatricesList(C1, gi);
  sampleC2(C2, gi, A, C1);
  orthoBasis(repi, gi, C2, basis.DIMO);
  DistributeDims(basis.DIMO, gi);
  allocUcUo(basis, gi, C2);
  
  DistributeMatricesList(basis.Uc, gi);
  DistributeMatricesList(basis.Uo, gi);
}

void nbd::nextBasisDims(Base& bsnext, const GlobalIndex& gnext, const Base& bsprev, const GlobalIndex& gprev) {
  int64_t nboxes = gnext.BOXES;
  int64_t nbegin = gnext.SELF_I * nboxes;
  int64_t ngbegin = gnext.GBEGIN;

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
  DistributeDims(dims, gnext);
}

void nbd::basisFw(Vectors& Xo, Vectors& Xc, const Base& basis, const Vectors& X) {
  for (int64_t i = 0; i < X.size(); i++)
    pvc_fw(X[i], basis.Uo[i], basis.Uc[i], Xo[i], Xc[i]);
}

void nbd::basisBk(Vectors& X, const Base& basis, const Vectors& Xo, const Vectors& Xc) {
  for (int64_t i = 0; i < X.size(); i++)
    pvc_bk(Xo[i], Xc[i], basis.Uo[i], basis.Uc[i], X[i]);
}

void nbd::checkBasis(int64_t my_rank, const Base& basis) {
  int64_t tot_dim = 0;
  int64_t tot_dimo = 0;
  for (int64_t i = 0; i < basis.DIMS.size(); i++) {
    int64_t dim = basis.DIMS[i];
    int64_t dim_o = basis.DIMO[i];
    int64_t dim_c = dim - dim_o;

    tot_dim = tot_dim + dim;
    tot_dimo = tot_dimo + dim_o;

    Matrix oo, cc;
    cMatrix(oo, dim_o, dim_o);
    cMatrix(cc, dim_c, dim_c);
    mmult('T', 'N', basis.Uo[i], basis.Uo[i], oo, 1., 0.);
    mmult('T', 'N', basis.Uc[i], basis.Uc[i], cc, 1., 0.);

    for (int64_t d = 0; d < dim_o; d++)
      oo.A[d * (dim_o + 1)] = oo.A[d * (dim_o + 1)] - 1.;
    for (int64_t d = 0; d < dim_c; d++)
      cc.A[d * (dim_c + 1)] = cc.A[d * (dim_c + 1)] - 1.;
    
    double e1, e2;
    nrm2(oo, &e1);
    nrm2(cc, &e2);
    if (e1 > 1.e-10 || e2 > 1.e-10) {
      printf("%ld: FAIL at %ld: %e, %e\n", my_rank, i, e1, e2);
      return;
    }
  }

  double cmp_rate = 100 * (double)tot_dimo / tot_dim;
  printf("%ld: PASS %.3f%%\n", my_rank, cmp_rate);
}
