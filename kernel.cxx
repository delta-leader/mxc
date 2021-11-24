
#include "kernel.hxx"

#include <cmath>

using namespace nbd;

EvalFunc nbd::l2d() {
  EvalFunc ef;
  ef.r2f = [](double& r2, double singularity, double alpha) -> void {
    r2 = r2 == 0. ? singularity : std::log(std::sqrt(r2));
  };
  ef.singularity = 1.e6;
  ef.alpha = 1.;
  return ef;
}

EvalFunc nbd::l3d() {
  EvalFunc ef;
  ef.r2f = [](double& r2, double singularity, double alpha) -> void {
    r2 = r2 == 0. ? singularity : 1. / std::sqrt(r2);
  };
  ef.singularity = 1.e6;
  ef.alpha = 1.;
  return ef;
}


void nbd::eval(EvalFunc ef, const double* bi, const double* bj, int64_t dim, double* out) {
  double& r2 = *out;
  r2 = 0.;
  for (int64_t i = 0; i < dim; i++) {
    double dX = bi[i] - bj[i];
    r2 += dX * dX;
  }
  ef.r2f(r2, ef.singularity, ef.alpha);
}


void nbd::mvec_kernel(EvalFunc ef, int64_t m, int64_t n, const double* bi, const double* bj, int64_t dim, const double* X, double* B) {
  for (int64_t y = 0; y < m; y++) {
    double sum = 0.;
    for (int64_t x = 0; x < n; x++) {
      double r2;
      eval(ef, bi + y * dim, bj + x * dim, dim, &r2);
      sum += r2 * X[x];
    }
    B[y] = sum;
  }
}


void nbd::matrix_kernel(EvalFunc ef, int64_t m, int64_t n, const double* bi, const double* bj, int64_t dim, double* A, int64_t lda) {
  for (int64_t i = 0; i < m * n; i++) {
    int64_t x = i / m, y = i - x * m;
    double r2;
    eval(ef, bi + y * dim, bj + x * dim, dim, &r2);
    A[y + x * lda] = r2;
  }
}

