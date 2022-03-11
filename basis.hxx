
#pragma once

#include "domain.hxx"
#include "linalg.hxx"

namespace nbd {

  struct Base {
    int64_t LBOXES;
    int64_t LBGN;
    std::vector<int64_t> DIMS;
    std::vector<int64_t> DIMO;
    Matrices Uc;
    Matrices Uo;
  };

  typedef std::vector<Base> Basis;

  void sampleC1(Matrix* CL, const CSC& rels, const Matrices& A, const double* R, int64_t lenR);

  void sampleC2(Matrix* CL, const CSC& rels, const Matrices& A, const Matrices& C1, const int64_t ngbs[], int64_t ngb_len);

  void orthoBasis(double repi, int64_t N, Matrix* C, int64_t* dims_o);

  void allocBasis(Basis& basis, const LocalDomain& domain, const int64_t* bddims);

  void allocUcUo(Base& basis, const Matrices& C);

  void sampleA(Base& basis, double repi, const GlobalIndex& gi, const Matrices& A, const double* R, int64_t lenR);

  void nextBasisDims(Base& bsnext, const GlobalIndex& gnext, const Base& bsprev, const GlobalIndex& gprev);

};
