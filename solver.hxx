
#pragma once

#include "umv.hxx"

namespace nbd {

  struct RHS {
    Vectors X;
    Vectors Xc;
    Vectors Xo;
  };

  typedef std::vector<RHS> RHSS;

  void basisXoc(char fwbk, RHS& vx, const Base& basis, int64_t level);

  void svAccFw(Vectors& Xc, const Matrices& A_cc, const CSC& rels, int64_t level);

  void svAccBk(Vectors& Xc, const Matrices& A_cc, const CSC& rels, int64_t level);

  void svAocFw(Vectors& Xo, const Vectors& Xc, const Matrices& A_oc, const CSC& rels, int64_t level);

  void svAocBk(Vectors& Xc, const Vectors& Xo, const Matrices& A_oc, const CSC& rels, int64_t level);

  void allocRightHandSides(RHSS& rhs, const Base base[], int64_t levels);

  void permuteAndMerge(char fwbk, RHS& prev, RHS& next, int64_t nlevel);

  void solveA(RHS X[], const Node A[], const Base B[], const CSC rels[], int64_t levels);

  void solveRelErr(double* err_out, const Vector X[], const Vectors& ref, int64_t level);


};
