
#pragma once

#include "build_tree.hxx"

namespace nbd {

  struct Base {
    int64_t BOXES;
    int64_t SELF_I;

    std::vector<int64_t> RANKS;
    std::vector<int64_t> DIMS;
    std::vector<int64_t> DIMO;
    
    Matrices Uc;
    Matrices Uo;
    Matrices C1;
    Matrices C2;
  };

  typedef std::vector<Base> Basis;

  void init_rows_sample(Matrices& C, int64_t M, const int64_t* DIMS);

  void sample_rows(Matrices& C, int64_t lbegin, const CSC& rels, const Matrices& A, const double* R, int64_t lenR);

  void sample_rows_invd(Matrices& C, const CSC& rels, const Matrices& A, const Matrices& spC);

  void orth_row_basis(double repi, Matrices& Uc, Matrices& Uo, Matrices& C);

  void Alloc_basis(Basis& basis, const LocalDomain& domain);

  int64_t merge_dims(int64_t* dims, Matrices& Uo);

  void local_row_base(Base& basis, double repi, const Matrices& A, const double* R, const CSC& rels, int64_t lenR);

  void basis_fw(Vectors& Xo, Vectors& Xc, const Base& basis, const Vectors& X);

  void basis_bk(Vectors& X, const Base& basis, const Vectors& Xo, const Vectors& Xc);

};
