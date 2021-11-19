
#pragma once
#include "domain.hxx"
#include "linalg.hxx"

namespace nbd {

  void split_A(Matrices& A_out, const CSC& rels, const Matrices& A, const Matrices& U, const Matrices& V);

  void factor_Acc(Matrices& A_cc, const CSC& rels);

  void factor_Alow(Matrices& Alow, const CSC& rels_low, Matrices& A_cc, const CSC& rels_cc);

  void factor_Aup(Matrices& Aup, const CSC& rels_up, Matrices& A_cc, const CSC& rels_cc);

  void schur_cmplm_low(Matrices& S_oo, const Matrices& A_oc, const CSC& rels_oc, const Matrices& A_co, const CSC& rels_co);

  void schur_cmplm_up(Matrices& S_oo, const Matrices& A_oc, const CSC& rels_oc, const Matrices& A_co, const CSC& rels_co);

  void schur_cmplm_diag(Matrices& S_oo, const Matrices& A_oc, const Matrices& A_co, const CSC& rels);

  void A_cc_fw(Vectors& Xc, const Matrices& A_cc, const CSC& rels);

  void A_cc_bk(Vectors& Xc, const Matrices& A_cc, const CSC& rels);

  void A_oc_fw(Vectors& Xo, const Matrices& A_oc, const CSC& rels, const Vectors& Xc);

  void A_co_bk(Vectors& Xc, const Matrices& A_co, const CSC& rels, const Vectors& Xo);

  struct Node {
    Matrices A;
    Matrices A_cc;
    Matrices A_co;
    Matrices A_oc;
    Matrices A_oo;
  };

  typedef std::vector<Node> Nodes;

  

};