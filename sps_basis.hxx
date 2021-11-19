
#pragma once

#include "domain.hxx"
#include "linalg.hxx"

namespace nbd {

  struct Base {
    std::vector<int64_t> DIMS;
    std::vector<int64_t> DIMC;
    std::vector<int64_t> DIMO;
    
    Matrices Uc;
    Matrices Uo;
    Matrices C1;
    Matrices C2;
  };

  typedef std::vector<Base> Basis;

  void sampleC1(Matrices& C1, const GlobalIndex& gi, const CSC& rels, const Matrices& A, const double* R, int64_t lenR);

  void sampleC2(Matrices& C2, const GlobalIndex& gi, const CSC& rels, const Matrices& A, const Matrices& C1);

  void orth_row_basis(double repi, Matrices& Uc, Matrices& Uo, Matrices& C);

  void Alloc_basis(Basis& basis, const LocalDomain& domain);

  void Alloc_leaf_base(Base& leaf, const int64_t* bodies);

  void sampleA(Base& basis, double repi, const CSC& rels, const Matrices& A, const double* R, int64_t lenR);

  void basis_fw(Vectors& Xo, Vectors& Xc, const Base& basis, const Vectors& X);

  void basis_bk(Vectors& X, const Base& basis, const Vectors& Xo, const Vectors& Xc);

  void DistributeC1(Base& basis, const GlobalIndex& gi);

};
