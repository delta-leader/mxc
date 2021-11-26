
#pragma once

#include "domain.hxx"
#include "linalg.hxx"

namespace nbd {

  struct Base {
    std::vector<int64_t> DIMS;
    std::vector<int64_t> DIMO;
    Matrices Uc;
    Matrices Uo;
  };

  typedef std::vector<Base> Basis;

  void sampleC1(Matrices& C1, const GlobalIndex& gi, const Matrices& A, const double* R, int64_t lenR);

  void sampleC2(Matrices& C2, const GlobalIndex& gi, const Matrices& A, const Matrices& C1);

  void orthoBasis(double repi, const GlobalIndex& gi, Matrices& C, std::vector<int64_t>& dims_o);

  void allocBasis(Basis& basis, const LocalDomain& domain, const int64_t* bddims);

  void allocUcUo(Base& basis, const GlobalIndex& gi, const Matrices& C);

  void sampleA(Base& basis, double repi, const GlobalIndex& gi, const Matrices& A, const double* R, int64_t lenR);

  void nextBasisDims(Base& bsnext, const GlobalIndex& gnext, const Base& bsprev, const GlobalIndex& gprev);

  void checkBasis(int64_t my_rank, const Base& basis);

};
