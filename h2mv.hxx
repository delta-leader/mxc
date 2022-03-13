
#pragma once
#include "build_tree.hxx"

namespace nbd {

  void upwardPassLeaf(const Cells& cells, const Matrices& base, const double* x, double* m);

  void downwardPassLeaf(const Cells& cells, const Matrices& base, const double* l, double* b);

  void horizontalPass(EvalFunc ef, const Cells& cells, int64_t dim, int64_t level, const double* m, double* l);

  void upwardPass(const Cells& cells, int64_t level, const Matrices& base, double* m);

  void downwardPass(const Cells& cells, int64_t level, const Matrices& base, double* l);

  void closeQuarter(EvalFunc ef, const Cells& cells, int64_t dim, const double* x, double* b);

  void zeroC(const Cell* cell, double* c, int64_t lmin, int64_t lmax);

  void multiplyC(EvalFunc ef, const Cells& cells, int64_t dim, int64_t level, const Matrices& base, double* m, double* l);

  void h2mv_complete(EvalFunc ef, const Cells& cells, int64_t dim, const Matrices& base, const double* x, double* b);

  int64_t C2X(const Cells& cells, int64_t level, const double* c, double* x);

  int64_t X2C(const Cells& cells, int64_t level, const double* x, double* c);

  void h2inv(EvalFunc ef, const Cells& cells, int64_t dim, const Matrices& base, const Matrices d[], const CSC rels[], double* x);

}
