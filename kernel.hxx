
#pragma once
#include <cstdint>

namespace nbd {

  typedef void (*eval_func_t) (double&, double, double);

  struct EvalFunc {
    eval_func_t r2f;
    double singularity;
    double alpha;
  };

  EvalFunc l2d();

  EvalFunc l3d();

  void eval(EvalFunc ef, const double* bi, const double* bj, int64_t dim, double* out);

  void mvec_kernel(EvalFunc ef, int64_t m, int64_t n, const double* bi, const double* bj, int64_t dim, const double* X, double* B);

  void matrix_kernel(EvalFunc ef, int64_t m, int64_t n, const double* bi, const double* bj, int64_t dim, double* A, int64_t lda);

}