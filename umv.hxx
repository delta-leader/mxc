
#pragma once

#include "basis.hxx"

namespace nbd {

  struct Node {
    Matrices A;
    Matrices A_cc;
    Matrices A_oc;
    Matrices A_oo;
    Matrices S;
  };

  struct RHS {
    Vectors X;
    Vectors X_c;
    Vectors X_o;
  };

  typedef std::vector<Node> Nodes;
  typedef std::vector<RHS> RHSS;

  void splitA(Matrices& A_out, const GlobalIndex& gi, const Matrices& A, const Matrices& U, const Matrices& V);

  void factorAcc(Matrices& A_cc, const GlobalIndex& gi);

  void factorAoc(Matrices& A_oc, const Matrices& A_cc, const GlobalIndex& gi);

  void schurCmplm(Matrices& S, const Matrices& A_oc, const GlobalIndex& gi);

  void axatLocal(Matrices& A, const GlobalIndex& gi);

  Matrices* allocNodes(Nodes& nodes, const LocalDomain& domain);

  void allocA(Matrices& A, const GlobalIndex& gi, const int64_t* dims);

  void allocSubMatrices(Node& n, const GlobalIndex& gi, const int64_t* dims, const int64_t* dimo);

  void factorNode(Node& n, Base& basis, const GlobalIndex& gi, double repi, const double* R, int64_t lenR);

  void nextNode(Node& Anext, Base& bsnext, const GlobalIndex& Gnext, const Node& Aprev, const Base& bsprev, const GlobalIndex& Gprev);

  void factorA(Nodes& A, Basis& B, const LocalDomain& domain, double repi, const double* R, int64_t lenR);

  void svAcc(char fwbk, Vectors& Xc, const Matrices& A_cc, const GlobalIndex& gi);

  void svAocFw(Vectors& Xo, const Vectors& Xc, const Matrices& A_oc, const GlobalIndex& gi);

  void svAocBk(Vectors& Xc, const Vectors& Xo, const Matrices& A_oc, const GlobalIndex& gi);

};
