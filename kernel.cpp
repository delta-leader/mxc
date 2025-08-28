
#include <kernel.hpp>

#include <algorithm>
#include <numeric>
#include <vector>
#include <array>
#include <random>

#include <iostream>
#include <iomanip>

#include <Eigen/Dense>
#include <Eigen/SVD>

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

void DenseZMat::op_Aij_mulB(char opA, long long mC, long long nC, long long k, long long iA, long long jA, const std::complex<double>* B_in, long long strideB, std::complex<double>* C_out, long long strideC) const {
  Eigen::Stride<Eigen::Dynamic, 1> lda(M, 1), ldb(strideB, 1), ldc(strideC, 1);
  Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> matC(C_out, mC, nC, ldc);
  Eigen::Map<const Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> matB(B_in, k, nC, ldb);
  if (opA == 'T' || opA == 't')
    matC.noalias() = Eigen::Map<const Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>>(&A[iA + jA * M], k, mC, lda).transpose() * matB;
  else if (opA == 'C' || opA == 'c')
    matC.noalias() = Eigen::Map<const Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>>(&A[iA + jA * M], k, mC, lda).adjoint() * matB;
  else
    matC.noalias() = Eigen::Map<const Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>>(&A[iA + jA * M], mC, k, lda) * matB;
}

void Zrsvd(double epi, long long m, long long n, long long* k, long long p, long long niters, const Accessor& A, long long iA, long long jA, double* S, std::complex<double>* U, long long ldu, std::complex<double>* V, long long ldv) {
  long long rank = std::min(*k, std::min(m, n));
  p = std::min(rank + p, std::min(m, n));
  Eigen::MatrixXcd R(n, p), Q(m, p);
  std::mt19937_64 gen;
  std::normal_distribution<double> norm_dist(0., 1.);
  std::generate(R.reshaped().begin(), R.reshaped().end(), [&]() { return std::complex<double>(norm_dist(gen), norm_dist(gen)); });

  A.op_Aij_mulB('N', m, p, n, iA, jA, R.data(), n, Q.data(), m);
  while (0 < --niters) {
    Eigen::HouseholderQR<Eigen::MatrixXcd> qr(Q);
    Q = qr.householderQ() * Eigen::MatrixXcd::Identity(m, p);
    A.op_Aij_mulB('C', n, p, m, iA, jA, Q.data(), m, R.data(), n);
    A.op_Aij_mulB('N', m, p, n, iA, jA, R.data(), n, Q.data(), m);
  }

  Eigen::HouseholderQR<Eigen::MatrixXcd> qr(Q);
  Q = qr.householderQ() * Eigen::MatrixXcd::Identity(m, p);
  A.op_Aij_mulB('C', n, p, m, iA, jA, Q.data(), m, R.data(), n);

  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(R, Eigen::ComputeThinU | Eigen::ComputeThinV);
  if (0. < epi && epi < 1.)
  { svd.setThreshold(epi); *k = rank = std::min(rank, (long long)svd.rank()); }

  Eigen::Stride<Eigen::Dynamic, 1> ldU(ldu, 1), ldV(ldv, 1);
  Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> matU(U, m, rank, ldU);
  Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> matV(V, n, rank, ldV);
  Eigen::Map<Eigen::VectorXd> vecS(S, rank);

  vecS = svd.singularValues().topRows(rank);
  matV = svd.matrixU().leftCols(rank);
  matU.noalias() = Q * svd.matrixV().leftCols(rank);
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

void gen_matrix(const Eigen::Ref<const Eigen::MatrixXcd> &mat, long long m, long long n, const long long* rows, const long long* cols, Eigen::Ref<Eigen::MatrixXcd> Aij) {
  for (long long i = 0; i < m; ++i) {
    for (long long j = 0; j < n; ++j) {
      Aij(i, j) = mat(rows[i], cols[j]);
    }
  }
}

void gen_matrix_hidr(const Eigen::Ref<const Eigen::MatrixXcd> &mat, long long m, long long n, const long long* rows, const long long* cols, Eigen::Ref<Eigen::MatrixXcd> Aij) {
  long long A_row_offset = 0, A_col_offset = 0;
  for (long long i = 0; i < m; ++i) {
    for (long long ii = 0; ii < 3; ++ii) {
      for (long long j = 0; j < n; ++j) {
        //std::cout<<i*3+ii<<", "<<j<<" = "<<rows[i]*3 + ii<<", "<<j<<std::endl;
        Aij(i*3+ii, j) = mat(rows[i]*3 + ii, cols[j]);
      }
    }
  }
}

// probably no longer needed
void gen_matrix(const Eigen::Ref<const Eigen::MatrixXcd> &mat, std::vector<long long>& rows, std::vector<long long>& cols, Eigen::MatrixXcd& Aij) {
  long long A_row_offset = 0, A_col_offset = 0;
  for (size_t i = 0; i < rows.size(); i += 2) {
    long long row_offset = rows[i] * 3;
    long long row_num = (rows[i + 1] - rows[i])* 3;
    for (size_t j = 0; j < cols.size(); j += 2) {
      long long col_offset = cols[i] * 3;
      long long col_num = (cols[i + 1] - cols[i])* 3;
      Aij.block(A_row_offset, A_col_offset, row_num, col_num) = mat.block(row_offset, col_offset, row_num, col_num);
    }
  }
}

MatrixGenerator::MatrixGenerator(double omega): omega(omega), alpha(elastWave3d::set_alpha(omega)), mu0(elastWave3d::get_mu(0, 1)), mu1(elastWave3d::get_mu(1, 1)) {};
void MatrixGenerator::gen_matrix(const elastWave3d::nodal_point* xnodes, long long num_xnodes, const elastWave3d::element* xelems, long long num_xelems, const elastWave3d::nodal_point* ynodes, long long num_ynodes, const elastWave3d::element* yelems, long long num_yelems, std::complex<double> cmat[]) const {
  long long nmat = (num_xnodes + num_xelems) * 3;
  // W0 + W1
  long long rowShift = num_xnodes;
  long long colShift = num_ynodes;
  std::vector<std::complex<double>> mat3x3(9, 0.0);
  std::vector<std::complex<double>> mat3x3_2nd(9, 0.0);
  #pragma omp parallel for firstprivate(mat3x3, mat3x3_2nd) collapse(2)
  for(int xindex = 0; xindex < num_xnodes; xindex++){
    for(int yindex = 0; yindex < num_ynodes; yindex++){
      //std::cout<<"row "<<xindex <<", col "<<yindex<<std::endl;
      
      // W0
      const int out_in = 0;
      const int slp_or_dlp = 4;
      const int linear_or_const = 0;
      std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
      //std::cout<<"X: "<<num_xnodes<<", "<<num_xelems<< "Y: "<<num_xnodes<<", "<<num_xelems<< std::endl;
      //std::cout<<xnodes[0].xc[0]<<", "<<xnodes[0].xc[1] <<", "<<xnodes[0].xc[2] <<" x "<<ynodes[0].xc[0]<<", "<<ynodes[0].xc[1] <<", "<<ynodes[0].xc[2] <<std::endl;
      //std::cout<<xelems[0].xc[0]<<", "<<xelems[0].xc[1] <<", "<<xelems[0].xc[2] <<" x "<<yelems[0].xc[0]<<", "<<yelems[0].xc[1] <<", "<<yelems[0].xc[2] <<std::endl;
      //std::cout<<"Omebe: "<<omega<<", "<<out_in<< ", "<<slp_or_dlp<<", "<<linear_or_const<< std::endl;
      //for (size_t i = 0; i<mat3x3.size(); ++i)
      //  std::cout<<mat3x3[i]<<", ";
      //std::cout<<std::endl;
      elastWave3d::mkmat_entrywise_3d_elast(xnodes, num_xnodes, xelems, num_xelems, xindex + 1, ynodes, num_ynodes, yelems, num_yelems, yindex + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
      // W1
      const int out_in_2nd = 1;
      std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(xnodes, num_xnodes, xelems, num_xelems, xindex + 1, ynodes, num_ynodes, yelems, num_yelems, yindex + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
      for(int j = 0; j < 3; j++){
        for(int i = 0; i < 3; i++){
          cmat[i + 3*xindex + (j + 3*yindex) * nmat] = mat3x3.at(i + 3*j) + (mu1/mu0)*mat3x3_2nd.at(i + 3*j);
          //cmat[xindex + rowShift*i + (yindex + colShift*j) * nmat] = mat3x3.at(i + 3*j) + (mu1/mu0)*mat3x3_2nd.at(i + 3*j);
        }
      }
    }
  }
  // -(aT0 + aT1)
  rowShift = num_xnodes;
  colShift = num_yelems;
  #pragma omp parallel for firstprivate(mat3x3, mat3x3_2nd) collapse(2)
  for(int xindex = 0; xindex < num_xnodes; xindex++){
    for(int yindex = 0; yindex < num_yelems; yindex++){
      // aT0
      const int out_in = 0;
      const int slp_or_dlp = 3;
      const int linear_or_const = 0;
      std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(xnodes, num_xnodes, xelems, num_xelems, xindex + 1, ynodes, num_ynodes, yelems, num_yelems, yindex + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
      // aT1
      const int out_in_2nd = 1;
      std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(xnodes, num_xnodes, xelems, num_xelems, xindex + 1, ynodes, num_ynodes, yelems, num_yelems, yindex + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
      for(int j = 0; j < 3; j++){
        for(int i = 0; i < 3; i++){
          cmat[i + 3*xindex + (j + 3*yindex + 3*num_xnodes) * nmat] = -mat3x3.at(i + 3*j) - mat3x3_2nd.at(i + 3*j);
          //cmat[xindex + rowShift*i + (yindex + colShift*j + 3*num_xnodes)*nmat] = -mat3x3.at(i + 3*j) - mat3x3_2nd.at(i + 3*j);
        }
      }
    }
  }
  // T0 + T1
  rowShift = num_xelems;
  colShift = num_ynodes;
  #pragma omp parallel for firstprivate(mat3x3, mat3x3_2nd) collapse(2)
  for(int xindex = 0; xindex < num_xelems; xindex++){
    for(int yindex = 0; yindex < num_ynodes; yindex++){
      // T0
      const int out_in = 0;
      const int slp_or_dlp = 2;
      const int linear_or_const = 1;
      std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(xnodes, num_xnodes, xelems, num_xelems, xindex + 1, ynodes, num_ynodes, yelems, num_yelems, yindex + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
      // T1
      const int out_in_2nd = 1;
      std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(xnodes, num_xnodes, xelems, num_xelems, xindex + 1, ynodes, num_ynodes, yelems, num_yelems, yindex + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
      for(int j = 0; j < 3; j++){
        for(int i = 0; i < 3; i++){
          cmat[i + 3*xindex + 3*num_xnodes + (j + 3*yindex) * nmat] = mat3x3.at(i + 3*j) + mat3x3_2nd.at(i + 3*j);
          //cmat[xindex + rowShift*i + 3*num_xnodes + (yindex + colShift*j) * nmat] = mat3x3.at(i + 3*j) + mat3x3_2nd.at(i + 3*j);
        }
      }
    }
  }
  // -(U0 + U1)
  rowShift = num_xelems;
  colShift = num_yelems;
  #pragma omp parallel for firstprivate(mat3x3, mat3x3_2nd) collapse(2)
  for(int xindex = 0; xindex < num_xelems; xindex++){
    for(int yindex = 0; yindex < num_yelems; yindex++){
      // U0
      const int out_in = 0;
      const int slp_or_dlp = 1;
      const int linear_or_const = 1;
      std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(xnodes, num_xnodes, xelems, num_xelems, xindex + 1, ynodes, num_ynodes, yelems, num_yelems, yindex + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
      // U1
      const int out_in_2nd = 1;
      std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(xnodes, num_xnodes, xelems, num_xelems, xindex + 1, ynodes, num_ynodes, yelems, num_yelems, yindex + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
      for(int j = 0; j < 3; j++){
        for(int i = 0; i < 3; i++){
          cmat[i + 3*xindex + 3*num_xnodes + (j + 3*yindex + 3*num_ynodes) * nmat] = -mat3x3.at(i + 3*j) - (mu0/mu1)*mat3x3_2nd.at(i + 3*j);
          //cmat[xindex + rowShift*i + 3*num_xnodes + (yindex + colShift*j + 3*num_ynodes) * nmat]= -mat3x3.at(i + 3*j) - (mu0/mu1)*mat3x3_2nd.at(i + 3*j);
        }
      }
    }
  }
}

void MatrixGenerator::gen_matrix_sorted(const elastWave3d::nodal_point* xnodes, long long num_xnodes, const elastWave3d::element* xelems, long long num_xelems, const elastWave3d::nodal_point* ynodes, long long num_ynodes, const elastWave3d::element* yelems, long long num_yelems, std::vector<long long>& nodes_indices, std::vector<long long>& elems_indices, const double scale, std::complex<double> cmat[]) const {
  long long nmat = (num_xnodes + num_xelems) * 3;
  // W0 + W1
  long long rowShift = num_xnodes;
  long long colShift = num_ynodes;
  std::vector<std::complex<double>> mat3x3(9, 0.0);
  std::vector<std::complex<double>> mat3x3_2nd(9, 0.0);
  #pragma omp parallel for firstprivate(mat3x3, mat3x3_2nd) collapse(2)
  for(int xindex = 0; xindex < num_xnodes; xindex++){
    for(int yindex = 0; yindex < num_ynodes; yindex++){
      //std::cout<<"row "<<xindex <<", col "<<yindex<<std::endl;
      
      // W0
      const int out_in = 0;
      const int slp_or_dlp = 4;
      const int linear_or_const = 0;
      std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
      //std::cout<<"X: "<<num_xnodes<<", "<<num_xelems<< "Y: "<<num_xnodes<<", "<<num_xelems<< std::endl;
      //std::cout<<xnodes[0].xc[0]<<", "<<xnodes[0].xc[1] <<", "<<xnodes[0].xc[2] <<" x "<<ynodes[0].xc[0]<<", "<<ynodes[0].xc[1] <<", "<<ynodes[0].xc[2] <<std::endl;
      //std::cout<<xelems[0].xc[0]<<", "<<xelems[0].xc[1] <<", "<<xelems[0].xc[2] <<" x "<<yelems[0].xc[0]<<", "<<yelems[0].xc[1] <<", "<<yelems[0].xc[2] <<std::endl;
      //std::cout<<"Omebe: "<<omega<<", "<<out_in<< ", "<<slp_or_dlp<<", "<<linear_or_const<< std::endl;
      //for (size_t i = 0; i<mat3x3.size(); ++i)
      //  std::cout<<mat3x3[i]<<", ";
      //std::cout<<std::endl;
      elastWave3d::mkmat_entrywise_3d_elast(xnodes, num_xnodes, xelems, num_xelems, nodes_indices[xindex] + 1, ynodes, num_ynodes, yelems, num_yelems, nodes_indices[yindex] + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
      // W1
      const int out_in_2nd = 1;
      std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(xnodes, num_xnodes, xelems, num_xelems, nodes_indices[xindex] + 1, ynodes, num_ynodes, yelems, num_yelems, nodes_indices[yindex] + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
      for(int j = 0; j < 3; j++){
        for(int i = 0; i < 3; i++){
          cmat[i + 3*xindex + (j + 3*yindex) * nmat] = mat3x3.at(i + 3*j) + (mu1/mu0)*mat3x3_2nd.at(i + 3*j);
          //cmat[xindex + rowShift*i + (yindex + colShift*j) * nmat] = mat3x3.at(i + 3*j) + (mu1/mu0)*mat3x3_2nd.at(i + 3*j);
        }
      }
    }
  }
  // -(aT0 + aT1)
  rowShift = num_xnodes;
  colShift = num_yelems;
  #pragma omp parallel for firstprivate(mat3x3, mat3x3_2nd) collapse(2)
  for(int xindex = 0; xindex < num_xnodes; xindex++){
    for(int yindex = 0; yindex < num_yelems; yindex++){
      // aT0
      const int out_in = 0;
      const int slp_or_dlp = 3;
      const int linear_or_const = 0;
      std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(xnodes, num_xnodes, xelems, num_xelems, nodes_indices[xindex] + 1, ynodes, num_ynodes, yelems, num_yelems, elems_indices[yindex] + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
      // aT1
      const int out_in_2nd = 1;
      std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(xnodes, num_xnodes, xelems, num_xelems, nodes_indices[xindex] + 1, ynodes, num_ynodes, yelems, num_yelems, elems_indices[yindex] + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
      for(int j = 0; j < 3; j++){
        for(int i = 0; i < 3; i++){
          cmat[i + 3*xindex + (j + 3*yindex + 3*num_xnodes) * nmat] = -mat3x3.at(i + 3*j) - mat3x3_2nd.at(i + 3*j);
          cmat[i + 3*xindex + (j + 3*yindex + 3*num_xnodes) * nmat] *= scale;
          //cmat[xindex + rowShift*i + (yindex + colShift*j + 3*num_xnodes)*nmat] = -mat3x3.at(i + 3*j) - mat3x3_2nd.at(i + 3*j);
        }
      }
    }
  }
  // T0 + T1
  rowShift = num_xelems;
  colShift = num_ynodes;
  #pragma omp parallel for firstprivate(mat3x3, mat3x3_2nd) collapse(2)
  for(int xindex = 0; xindex < num_xelems; xindex++){
    for(int yindex = 0; yindex < num_ynodes; yindex++){
      // T0
      const int out_in = 0;
      const int slp_or_dlp = 2;
      const int linear_or_const = 1;
      std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(xnodes, num_xnodes, xelems, num_xelems, elems_indices[xindex] + 1, ynodes, num_ynodes, yelems, num_yelems, nodes_indices[yindex] + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
      // T1
      const int out_in_2nd = 1;
      std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(xnodes, num_xnodes, xelems, num_xelems, elems_indices[xindex] + 1, ynodes, num_ynodes, yelems, num_yelems, nodes_indices[yindex] + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
      for(int j = 0; j < 3; j++){
        for(int i = 0; i < 3; i++){
          cmat[i + 3*xindex + 3*num_xnodes + (j + 3*yindex) * nmat] = mat3x3.at(i + 3*j) + mat3x3_2nd.at(i + 3*j);
          cmat[i + 3*xindex + 3*num_xnodes + (j + 3*yindex) * nmat] *= scale;
          //cmat[xindex + rowShift*i + 3*num_xnodes + (yindex + colShift*j) * nmat] = mat3x3.at(i + 3*j) + mat3x3_2nd.at(i + 3*j);
        }
      }
    }
  }
  // -(U0 + U1)
  rowShift = num_xelems;
  colShift = num_yelems;
  #pragma omp parallel for firstprivate(mat3x3, mat3x3_2nd) collapse(2)
  for(int xindex = 0; xindex < num_xelems; xindex++){
    for(int yindex = 0; yindex < num_yelems; yindex++){
      // U0
      const int out_in = 0;
      const int slp_or_dlp = 1;
      const int linear_or_const = 1;
      std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(xnodes, num_xnodes, xelems, num_xelems, elems_indices[xindex] + 1, ynodes, num_ynodes, yelems, num_yelems, elems_indices[yindex] + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
      // U1
      const int out_in_2nd = 1;
      std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(xnodes, num_xnodes, xelems, num_xelems, elems_indices[xindex] + 1, ynodes, num_ynodes, yelems, num_yelems, elems_indices[yindex] + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
      for(int j = 0; j < 3; j++){
        for(int i = 0; i < 3; i++){
          cmat[i + 3*xindex + 3*num_xnodes + (j + 3*yindex + 3*num_ynodes) * nmat] = -mat3x3.at(i + 3*j) - (mu0/mu1)*mat3x3_2nd.at(i + 3*j);
          cmat[i + 3*xindex + 3*num_xnodes + (j + 3*yindex + 3*num_ynodes) * nmat] *= scale * scale;
          //cmat[xindex + rowShift*i + 3*num_xnodes + (yindex + colShift*j + 3*num_ynodes) * nmat]= -mat3x3.at(i + 3*j) - (mu0/mu1)*mat3x3_2nd.at(i + 3*j);
        }
      }
    }
  }
}

void MatrixGenerator::gen_rhs(const elastWave3d::nodal_point* nodes, long long num_nodes, const elastWave3d::element* elems, long long num_elems, std::complex<double> rhs[], bool equation_type) const {
  if (equation_type){
    // PMCHWT
    // const size_t nodeShift = num_nodes;
    for(int i = 0; i < num_nodes; i++){
      std::complex<double> uout[3];
      elastWave3d::inc_trac(nodes, num_nodes, i + 1, elems, num_elems, omega, uout); // i + 1 is fortran index
      for(int j = 0; j < 3; j++){
        rhs[j + 3 * i] = uout[j];
        //rhs[i + nodeShift*j] = uout[j];
      }
    }
    //const size_t elemShift = num_elems;
    for(int i = 0; i < num_elems; i++){
      std::complex<double> uout[3];
      elastWave3d::inc_disp_const_x(nodes, num_nodes, elems[i], omega, uout);
      for(int j = 0; j < 3; j++){
        rhs[j + 3 * i + 3 * num_nodes] = uout[j];
        //rhs[i + elemShift*j + 3*num_nodes] = uout[j];
      }
    }
  }
  else{
    // Burton-Miller
    //const size_t nodeShift = num_nodes;
    for(int i = 0; i < num_nodes; i++){
      std::complex<double> uout[3];
      std::complex<double> tout[3];
      elastWave3d::inc_disp(nodes, num_nodes, i + 1, elems, num_elems, omega, uout); // i + 1 is fortran index
      elastWave3d::inc_trac(nodes, num_nodes, i + 1, elems, num_elems, omega, tout); // i + 1 is fortran index
      for(int j = 0; j < 3; j++){
        rhs[j + 3 * i] = uout[j] + alpha*tout[j];
        //rhs[j + nodeShift*i] = uout[j] + alpha*tout[j];
      }
    }
  }
}

void MatrixGenerator::gen_rhs_sorted(const elastWave3d::nodal_point* nodes, long long num_nodes, const elastWave3d::element* elems, long long num_elems, std::vector<long long>& nodes_indices, std::vector<long long>& elems_indices, const double scale, std::complex<double> rhs[], bool equation_type) const {
  if (equation_type){
    // PMCHWT
    // const size_t nodeShift = num_nodes;
    for(int i = 0; i < num_nodes; i++){
      std::complex<double> uout[3];
      elastWave3d::inc_trac(nodes, num_nodes, nodes_indices[i] + 1, elems, num_elems, omega, uout); // i + 1 is fortran index
      for(int j = 0; j < 3; j++){
        rhs[j + 3 * i] = uout[j];
        //rhs[i + nodeShift*j] = uout[j];
      }
    }
    //const size_t elemShift = num_elems;
    for(int i = 0; i < num_elems; i++){
      std::complex<double> uout[3];
      elastWave3d::inc_disp_const_x(nodes, num_nodes, elems[elems_indices[i]], omega, uout);
      for(int j = 0; j < 3; j++){
        rhs[j + 3 * i + 3 * num_nodes] = uout[j] * scale;
        //rhs[i + elemShift*j + 3*num_nodes] = uout[j];
      }
    }
  }
  else{
    // Burton-Miller
    //const size_t nodeShift = num_nodes;
    for(int i = 0; i < num_nodes; i++){
      std::complex<double> uout[3];
      std::complex<double> tout[3];
      elastWave3d::inc_disp(nodes, num_nodes, i + 1, elems, num_elems, omega, uout); // i + 1 is fortran index
      elastWave3d::inc_trac(nodes, num_nodes, i + 1, elems, num_elems, omega, tout); // i + 1 is fortran index
      for(int j = 0; j < 3; j++){
        rhs[j + 3 * i] = uout[j] + alpha*tout[j];
        //rhs[j + nodeShift*i] = uout[j] + alpha*tout[j];
      }
    }
  }
}

double MatrixGenerator::get_max_elems(const elastWave3d::nodal_point* xnodes, long long num_xnodes, const elastWave3d::element* xelems, long long num_xelems) const {
  std::vector<std::complex<double>> mat3x3(9, 0.0);
  std::vector<std::complex<double>> mat3x3_2nd(9, 0.0);
  double max = 0;
  #pragma omp parallel for firstprivate(mat3x3, mat3x3_2nd)
  for(int xindex = 0; xindex < num_xelems; xindex++){
      //std::cout<<"row "<<xindex <<", col "<<yindex<<std::endl;
      
      // U0
      const int out_in = 0;
      const int slp_or_dlp = 1;
      const int linear_or_const = 1;
      std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(xnodes, num_xnodes, xelems, num_xelems, xindex + 1, xnodes, num_xnodes, xelems, num_xelems, xindex + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
      // U1
      const int out_in_2nd = 1;
      std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(xnodes, num_xnodes, xelems, num_xelems, xindex + 1, xnodes, num_xnodes, xelems, num_xelems, xindex + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
      for(int j = 0; j < 3; j++){
        std::complex<double> value = -mat3x3.at(j + 3*j) - (mu0/mu1) * mat3x3_2nd.at(j + 3*j);
        if (std::abs(value.real()) > max)
          max = std::abs(value.real());
      }
  }
  return max;
}

double MatrixGenerator::get_max_nodes(const elastWave3d::nodal_point* xnodes, long long num_xnodes, const elastWave3d::element* xelems, long long num_xelems) const {
  //long long nmat = (num_xnodes + num_xelems) * 3;
  // W0 + W1
  //long long rowShift = num_xnodes;
  //long long colShift = num_ynodes;
  std::vector<std::complex<double>> mat3x3(9, 0.0);
  std::vector<std::complex<double>> mat3x3_2nd(9, 0.0);
  double max = 0;
  #pragma omp parallel for firstprivate(mat3x3, mat3x3_2nd)
  for(int xindex = 0; xindex < num_xnodes; xindex++){
      //std::cout<<"row "<<xindex <<", col "<<yindex<<std::endl;
      
      // W0
      const int out_in = 0;
      const int slp_or_dlp = 4;
      const int linear_or_const = 0;
      std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(xnodes, num_xnodes, xelems, num_xelems, xindex + 1, xnodes, num_xnodes, xelems, num_xelems, xindex + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
      // W1
      const int out_in_2nd = 1;
      std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(xnodes, num_xnodes, xelems, num_xelems, xindex + 1, xnodes, num_xnodes, xelems, num_xelems, xindex + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
      for(int j = 0; j < 3; j++){
        std::complex<double> value = mat3x3.at(j + 3*j) + (mu1/mu0) * mat3x3_2nd.at(j + 3*j);
          if (std::abs(value.real()) > max)
            max = std::abs(value.real());
      }
  }
  return max;
}