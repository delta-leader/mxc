
#pragma once

#include "domain.hxx"
#include "linalg.hxx"

namespace nbd {

  struct Base {
    std::vector<int64_t> DIMS;
    std::vector<int64_t> DIMO;
    
    Matrices Uc;
    Matrices Uo;
    Matrices C1;
    Matrices C2;
  };

  typedef std::vector<Base> Basis;

  void sampleC1(Matrices& C1, const GlobalIndex& gi, const Matrices& A, const double* R, int64_t lenR);

  void sampleC2(Matrices& C2, const GlobalIndex& gi, const Matrices& A, const Matrices& C1);

  void orthoBasis(double repi, const GlobalIndex& gi, Matrices& Uc, Matrices& Uo, Matrices& C, std::vector<int64_t>& dims_o);

  void AllocBasis(Basis& basis, const LocalDomain& domain);

  void AllocLeafBase(Base& leaf, const int64_t* bodies);

  void AllocDistUcUo(Base& basis);

  void sampleA(Base& basis, double repi, const GlobalIndex& gi, const Matrices& A, const double* R, int64_t lenR);

  void basisFw(Vectors& Xo, Vectors& Xc, const Base& basis, const Vectors& X);

  void basisBk(Vectors& X, const Base& basis, const Vectors& Xo, const Vectors& Xc);

  void DistributeMatricesList(Matrices& lis, const GlobalIndex& gi);

  void DistributeDims(std::vector<int64_t>& dims, const GlobalIndex& gi);

};
