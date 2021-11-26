
#pragma once

#include "umv.hxx"

namespace nbd {

  struct RHS {
    Vectors X;
    Vectors X_c;
    Vectors X_o;
  };

  typedef std::vector<RHS> RHSS;

  void basisXoc(char fwbk, RHS& vx, const Base& basis, const GlobalIndex& gi);

  void svAcc(char fwbk, Vectors& Xc, const Matrices& A_cc, const GlobalIndex& gi);

  void svAocFw(Vectors& Xo, const Vectors& Xc, const Matrices& A_oc, const GlobalIndex& gi);

  void svAocBk(Vectors& Xc, const Vectors& Xo, const Matrices& A_oc, const GlobalIndex& gi);

  Vector* allocRightHandSides(RHSS& rhs, const Basis& base, const LocalDomain& domain);

  void permuteAndMerge(char fwbk, RHS& prev, RHS& next);

  void solveA(RHSS& X, const Nodes& A, const Basis& B, const LocalDomain& domain);

  void solveRelErr(double* err_out, const RHS& X, const Vectors& ref, const GlobalIndex& gi);


};
