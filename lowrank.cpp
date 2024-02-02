
#include <lowrank.hpp>

#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <random>
#include <iostream>

LowRank::LowRank(double epi, int64_t M, int64_t N, int64_t K, int64_t P, const std::complex<double> A[], int64_t lda) : M(M), N(N), Rank(0) {
  std::mt19937 gen(0);
  std::normal_distribution gauss(0., 1.);
  Eigen::MatrixXcd RND(N, P);
  for (std::complex<double>& i : RND.reshaped())
    i = std::complex<double>(gauss(gen), 0.);

  Eigen::Map<const Eigen::MatrixXcd, Eigen::Unaligned, Eigen::OuterStride<>> A_(A, M, N, Eigen::OuterStride(lda));
  Eigen::MatrixXcd Y = A_ * RND;

  Eigen::HouseholderQR<Eigen::MatrixXcd> qr(Y);
  Eigen::MatrixXcd Q = qr.householderQ() * Eigen::MatrixXcd::Identity(M, P);
  Eigen::MatrixXcd B = Q.adjoint() * A_;

  Eigen::ColPivHouseholderQR<Eigen::MatrixXcd> rrqr(B);
  rrqr.setThreshold(epi);

  Rank = std::min(K, rrqr.rank());
  U = std::vector<std::complex<double>>(M * Rank);
  V = std::vector<std::complex<double>>(Rank * N);

  Eigen::Map<Eigen::MatrixXcd> U_(&U[0], M, Rank);
  Eigen::Map<Eigen::MatrixXcd> V_(&V[0], Rank, N);
  U_ = Q * (rrqr.householderQ() * Eigen::MatrixXcd::Identity(P, Rank));
  V_ = rrqr.matrixR().topLeftCorner(Rank, N).template triangularView<Eigen::Upper>();
  rrqr.colsPermutation().transpose().applyThisOnTheRight(V_);
}
