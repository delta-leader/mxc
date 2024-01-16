
#include <kernel.hpp>

#include <cblas.h>
#include <algorithm>
#include <numeric>
#include <array>

void gen_matrix(const Eval& eval, int64_t m, int64_t n, const double* bi, const double* bj, std::complex<double> Aij[], int64_t lda) {
  const std::array<double, 3>* bi3 = reinterpret_cast<const std::array<double, 3>*>(bi);
  const std::array<double, 3>* bi3_end = reinterpret_cast<const std::array<double, 3>*>(&bi[3 * m]);
  const std::array<double, 3>* bj3 = reinterpret_cast<const std::array<double, 3>*>(bj);
  const std::array<double, 3>* bj3_end = reinterpret_cast<const std::array<double, 3>*>(&bj[3 * n]);

  std::for_each(bj3, bj3_end, [&](const std::array<double, 3>& j) -> void {
    int64_t ix = std::distance(bj3, &j);
    std::for_each(bi3, bi3_end, [&](const std::array<double, 3>& i) -> void {
      int64_t iy = std::distance(bi3, &i);
      double d = std::hypot(i[0] - j[0], i[1] - j[1], i[2] - j[2]);
      Aij[iy + ix * lda] = eval(d);
    });
  });
}


void mat_vec_reference(const Eval& eval, int64_t M, int64_t N, int64_t nrhs, std::complex<double> B[], int64_t ldB, const std::complex<double> X[], int64_t ldX, const double ibodies[], const double jbodies[]) {
  int64_t size = 1 << 8;
  std::vector<std::complex<double>> A(size * size);
  std::complex<double> one(1., 0.);
  
  for (int64_t i = 0; i < M; i += size) {
    int64_t m = std::min(M - i, size);
    const double* bi = &ibodies[i * 3];
    for (int64_t j = 0; j < N; j += size) {
      const double* bj = &jbodies[j * 3];
      int64_t n = std::min(N - j, size);
      gen_matrix(eval, m, n, bi, bj, &A[0], size);
      cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, nrhs, n, &one, &A[0], size, &X[j], ldX, &one, &B[i], ldB);
    }
  }
}


