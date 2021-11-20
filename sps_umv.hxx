
#pragma once
#include "domain.hxx"
#include "linalg.hxx"

namespace nbd {

  struct Node {
    Matrices A;
    Matrices A_cc;
    Matrices A_oc;
    Matrices A_oo;
    Matrices S;
  };

  typedef std::vector<Node> Nodes;

  void splitA(Matrices& A_out, const GlobalIndex& gi, const Matrices& A, const Matrices& U, const Matrices& V);

  void factorAcc(Matrices& A_cc, const GlobalIndex& gi);

  void factorAoc(Matrices& A_oc, const Matrices& A_cc, const GlobalIndex& gi);

  void schurCmplm(Matrices& S, const Matrices& A_oc, const GlobalIndex& gi);

  void axatLocal(Matrices& A, const GlobalIndex& gi);

  void A_cc_fw(Vectors& Xc, const Matrices& A_cc, const CSC& rels);

  void A_cc_bk(Vectors& Xc, const Matrices& A_cc, const CSC& rels);

  void A_oc_fw(Vectors& Xo, const Matrices& A_oc, const CSC& rels, const Vectors& Xc);

  void A_co_bk(Vectors& Xc, const Matrices& A_co, const CSC& rels, const Vectors& Xo);

  void axatDistribute(Matrices& A, const GlobalIndex& gi);

};