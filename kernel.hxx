
#pragma once

#include "linalg.hxx"

namespace nbd {

  typedef void (*eval_func_t) (double&, double, double);

  struct Cell;
  struct Body;

  struct EvalFunc {
    eval_func_t r2f;
    double singularity;
    double alpha;
  };

  EvalFunc r2();

  EvalFunc l2d();

  EvalFunc l3d();

  void eval(EvalFunc ef, const Body* bi, const Body* bj, int64_t dim, double* out);

  void P2P(EvalFunc ef, const Cell* ci, const Cell* cj, int64_t dim, const double x[], double b[]);

  void P2Pmat(EvalFunc ef, const Cell* ci, const Cell* cj, int64_t dim, Matrix& a);

  void M2L(EvalFunc ef, const Cell* ci, const Cell* cj, int64_t dim, const double m[], double l[]);

  void M2Lmat(EvalFunc ef, const Cell* ci, const Cell* cj, int64_t dim, Matrix& a);

  void P2Mmat(EvalFunc ef, Cell* ci, const Body rm[], int64_t n, int64_t dim, Matrix& u, double epi, int64_t rank);

  void invBasis(const Matrix& u, Matrix& uinv);

  void D2C(const Matrix& d, const Matrix& u, const Matrix& v, Matrix& c, int64_t y, int64_t x);

  void L2C(EvalFunc ef, const Cell* ci, const Cell* cj, int64_t dim, Matrix& c, int64_t y, int64_t x);

  void P2M(const Cell* cell, const Matrix& ba, const double x[], double m[]);

  void M2M(const Cell* cell, const Matrix& ba, double m[]);

  void L2L(const Cell* cell, const Matrix& ba, double l[]);

  void L2P(const Cell* cell, const Matrix& ba, const double l[], double b[]);

  void M2X(const Cell* cell, const double m[], double x[]);

  void X2M(const Cell* cell, const double x[], double m[]);

  void factorD(Matrix& a);

  void invD(const Matrix& a, double* x);

}