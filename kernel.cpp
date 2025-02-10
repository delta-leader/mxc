
#include <kernel.hpp>

#include <algorithm>
#include <numeric>
#include <vector>
#include <array>
#include <random>

#include <Eigen/Dense>
#include <Eigen/SVD>

DenseDMat::DenseDMat(long long M, long long N) : Accessor(M, N), A(nullptr) {
  if (0 < M && 0 < N) {
    A = (double*)malloc(M * N * sizeof(double));
    std::fill(A, &A[M * N], 0.);
  }
}

DenseDMat::~DenseDMat() {
  if (A)
    free(A);
}

void DenseDMat::Aij(long long m, long long n, long long i, long long j, double* A_out, long long stride) const {
  Eigen::Stride<Eigen::Dynamic, 1> lda(stride, 1), ld;
  Eigen::Map<Eigen::MatrixXd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> mat_out(A_out, m, n, ld);
  mat_out = Eigen::Map<const Eigen::MatrixXd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>>(&A[i + j * M], m, n, lda);
}

void DenseDMat::Aij_mulB(long long mC, long long nC, long long k, long long iA, long long jA, const double* B_in, long long strideB, double* C_out, long long strideC) const {
  Eigen::Stride<Eigen::Dynamic, 1> lda(M, 1), ldb(strideB, 1), ldc(strideC, 1);
  Eigen::Map<Eigen::MatrixXd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> matC(C_out, mC, nC, ldc);
  Eigen::Map<const Eigen::MatrixXd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> matB(B_in, k, nC, ldb);
  matC.noalias() = Eigen::Map<const Eigen::MatrixXd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>>(&A[iA + jA * M], mC, k, lda) * matB;
}

DenseZMat::DenseZMat(long long M, long long N) : Accessor(M, N), A(nullptr) {
  if (0 < M && 0 < N) {
    A = (std::complex<double>*)malloc(M * N * sizeof(std::complex<double>));
    std::fill(A, &A[M * N], 0.);
  }
}

DenseZMat::~DenseZMat() {
  if (A)
    free(A);
}

void DenseZMat::Aij(long long m, long long n, long long i, long long j, std::complex<double>* A_out, long long stride) const {
  Eigen::Stride<Eigen::Dynamic, 1> lda(stride, 1), ld;
  Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> mat_out(A_out, m, n, ld);
  mat_out = Eigen::Map<const Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>>(&A[i + j * M], m, n, lda);
}

void DenseZMat::Aij_mulB(long long mC, long long nC, long long k, long long iA, long long jA, const std::complex<double>* B_in, long long strideB, std::complex<double>* C_out, long long strideC) const {
  Eigen::Stride<Eigen::Dynamic, 1> lda(M, 1), ldb(strideB, 1), ldc(strideC, 1);
  Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> matC(C_out, mC, nC, ldc);
  Eigen::Map<const Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> matB(B_in, k, nC, ldb);
  matC.noalias() = Eigen::Map<const Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>>(&A[iA + jA * M], mC, k, lda) * matB;
}

void Drsvd(long long m, long long n, long long k, long long p, long long niters, const double* A, long long lda, double* S, double* U, long long ldu, double* V, long long ldv) {
  k = std::min(k, std::min(m, n));
  p = std::min(k + p, std::min(m, n));
  Eigen::MatrixXd R(n, p);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> norm_dist(0., 1.);
  std::generate(R.reshaped().begin(), R.reshaped().end(), [&]() { return norm_dist(gen); });

  Eigen::Stride<Eigen::Dynamic, 1> ldA(lda, 1), ldU(ldu, 1), ldV(ldv, 1);
  Eigen::Map<const Eigen::MatrixXd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> matA(A, m, n, ldA);
  Eigen::MatrixXd Q = matA * R;
  while (0 < --niters)
    Q.noalias() = matA * (matA.adjoint() * Q);

  Eigen::Map<Eigen::MatrixXd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> matU(U, m, k, ldU);
  Eigen::Map<Eigen::MatrixXd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> matV(V, n, k, ldV);
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(Q);
  Q = qr.householderQ() * Eigen::MatrixXd::Identity(m, p);

  Eigen::Map<Eigen::VectorXd> vecS(S, k);
  Eigen::BDCSVD<Eigen::MatrixXd> svd(matV, Eigen::ComputeThinU | Eigen::ComputeThinV);
  vecS = svd.singularValues().topRows(k);
  matV = svd.matrixU().leftCols(k);
  matU.noalias() = Q * svd.matrixV().leftCols(k).adjoint();
}

void Zrsvd(long long m, long long n, long long k, long long p, long long niters, const std::complex<double>* A, long long lda, double* S, std::complex<double>* U, long long ldu, std::complex<double>* V, long long ldv) {
  k = std::min(k, std::min(m, n));
  p = std::min(k + p, std::min(m, n));
  Eigen::MatrixXcd R(n, p);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> norm_dist(0., 1.);
  std::generate(R.reshaped().begin(), R.reshaped().end(), [&]() { return std::complex<double>(norm_dist(gen), norm_dist(gen)); });

  Eigen::Stride<Eigen::Dynamic, 1> ldA(lda, 1), ldU(ldu, 1), ldV(ldv, 1);
  Eigen::Map<const Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> matA(A, m, n, ldA);
  Eigen::MatrixXcd Q = matA * R;
  while (0 < --niters)
    Q.noalias() = matA * (matA.adjoint() * Q);

  Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> matU(U, m, k, ldU);
  Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> matV(V, n, k, ldV);
  Eigen::HouseholderQR<Eigen::MatrixXcd> qr(Q);
  Q = qr.householderQ() * Eigen::MatrixXcd::Identity(m, p);

  Eigen::Map<Eigen::VectorXd> vecS(S, k);
  Eigen::BDCSVD<Eigen::MatrixXcd> svd(Q.adjoint() * matA, Eigen::ComputeThinU | Eigen::ComputeThinV);
  vecS = svd.singularValues().topRows(k);
  matV = svd.matrixV().leftCols(k);
  matU.noalias() = Q * svd.matrixU().leftCols(k);
}

void gen_matrix(const MatrixAccessor& eval, long long m, long long n, const double* bi, const double* bj, std::complex<double> Aij[]) {
  const std::array<double, 3>* bi3 = reinterpret_cast<const std::array<double, 3>*>(bi);
  const std::array<double, 3>* bi3_end = reinterpret_cast<const std::array<double, 3>*>(&bi[3 * m]);
  const std::array<double, 3>* bj3 = reinterpret_cast<const std::array<double, 3>*>(bj);
  const std::array<double, 3>* bj3_end = reinterpret_cast<const std::array<double, 3>*>(&bj[3 * n]);

  std::for_each(bj3, bj3_end, [&](const std::array<double, 3>& j) -> void {
    long long ix = std::distance(bj3, &j);
    std::for_each(bi3, bi3_end, [&](const std::array<double, 3>& i) -> void {
      long long iy = std::distance(bi3, &i);
      double d = std::hypot(i[0] - j[0], i[1] - j[1], i[2] - j[2]);
      Aij[iy + ix * m] = eval(d);
    });
  });
}
