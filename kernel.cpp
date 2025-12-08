
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

#include <test_funcs.hpp>


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
  std::cout<<"Generating matrix "<<m<<" x "<<n<<std::endl;
  for (long long i = 0; i < m; ++i) {
    for (long long j = 0; j < n; ++j) {
      Aij(i, j) = mat(rows[i], cols[j]);
    }
  }
}

void gen_matrix(const Eigen::Ref<const Eigen::MatrixXcd> &mat, long long m, long long n, const long long* rows, const long long* cols, long long start, Eigen::Ref<Eigen::MatrixXcd> Aij) {
  std::cout<<"Generating matrix "<<m<<" x "<<n<<std::endl;
  std::cout<<"Start "<<start<<std::endl;
  for (long long i = 0; i < m; ++i) {
    for (long long j = 0; j < n; ++j) {
      Aij(i, j) = mat(rows[i] - start, cols[j]);
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

MatrixGenerator::MatrixGenerator(const int size, const int spheres) {
  //std::cout<<"GENERATING"<<std::endl;
  std::string filename;
  switch (spheres) {
    case 8:
      filename = "../input/new/eight_spheres_" + std::to_string(size) + ".inp";
      break;
    case 5:
      filename = "../input/torus/torus_" + std::to_string(size) + ".inp";
      break;
    case 4:
      filename = "../input/new/four_spheres_" + std::to_string(size) + ".inp";
      break;
    case 3:
      filename = "../input/salt/salt_" + std::to_string(size) + "k.inp";
      break;
    case 2:
      filename = "../input/new/two_spheres_" + std::to_string(size) + ".inp";
      break;
    case 1:
      filename = "../input/new/sphere_" + std::to_string(size) + ".inp";
      break;
    default:
      //filename = "../input/mesh_sphere_" + std::to_string(size) + "nodes.inp";
      filename = "../input/mesh_two_sphere_" + std::to_string(size) + "nodes.inp";
  }
  //std::cout<<"READ MESH SPECS"<<std::endl;
  read_mesh_specs(num_nodes, num_elems, filename);
  std::cout<<num_nodes<<" "<<num_elems<<std::endl;
  nodes.resize(num_nodes);
  elems.resize(num_elems);
  //std::cout<<num_nodes<<" "<<num_elems<<std::endl;
  //std::cout<<"READ Fortran"<<std::endl;
  read_mesh_fortran(num_nodes, nodes, num_elems, elems, size, spheres);
  std::cout<<num_nodes<<" "<<num_elems<<std::endl;
  nodes_idx.resize(num_nodes);
  std::iota(nodes_idx.begin(), nodes_idx.end(), 0);
  elems_idx.resize(num_elems);
  std::iota(elems_idx.begin(), elems_idx.end(), 0);
  // this needs to be done after reading the file
  mu0 = elastWave3d::get_mu(0, 1);
  mu1 = elastWave3d::get_mu(1, 1);
}

// generates the whole matrix
void MatrixGenerator::gen_matrix(std::complex<double> cmat[], const double omega, double scale) const {
  if (!scale)
    scale = this->scale;
  long long nmat = (num_nodes + num_elems) * 3;
  // W0 + W1
  //long long rowShift = num_xnodes;
  //long long colShift = num_ynodes;
  std::vector<std::complex<double>> mat3x3(9, 0.0);
  std::vector<std::complex<double>> mat3x3_2nd(9, 0.0);
  #pragma omp parallel for firstprivate(mat3x3, mat3x3_2nd) collapse(2)
  for(int xindex = 0; xindex < num_nodes; xindex++){
    for(int yindex = 0; yindex < num_nodes; yindex++){
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
      elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, xindex + 1, nodes.data(), num_nodes, elems.data(), num_elems, yindex + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
      // W1
      const int out_in_2nd = 1;
      std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, xindex + 1, nodes.data(), num_nodes, elems.data(), num_elems, yindex + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
      for(int j = 0; j < 3; j++){
        for(int i = 0; i < 3; i++){
          cmat[i + 3*xindex + (j + 3*yindex) * nmat] = mat3x3.at(i + 3*j) + (mu1/mu0)*mat3x3_2nd.at(i + 3*j);
          //cmat[xindex + rowShift*i + (yindex + colShift*j) * nmat] = mat3x3.at(i + 3*j) + (mu1/mu0)*mat3x3_2nd.at(i + 3*j);
        }
      }
    }
  }
  // -(aT0 + aT1)
  //rowShift = num_xnodes;
  //colShift = num_yelems;
  #pragma omp parallel for firstprivate(mat3x3, mat3x3_2nd) collapse(2)
  for(int xindex = 0; xindex < num_nodes; xindex++){
    for(int yindex = 0; yindex < num_elems; yindex++){
      // aT0
      const int out_in = 0;
      const int slp_or_dlp = 3;
      const int linear_or_const = 0;
      std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, xindex + 1, nodes.data(), num_nodes, elems.data(), num_elems, yindex + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
      // aT1
      const int out_in_2nd = 1;
      std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, xindex + 1, nodes.data(), num_nodes, elems.data(), num_elems, yindex + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
      for(int j = 0; j < 3; j++){
        for(int i = 0; i < 3; i++){
          cmat[i + 3*xindex + (j + 3*yindex + 3*num_nodes) * nmat] = -mat3x3.at(i + 3*j) - mat3x3_2nd.at(i + 3*j);
          //cmat[xindex + rowShift*i + (yindex + colShift*j + 3*num_xnodes)*nmat] = -mat3x3.at(i + 3*j) - mat3x3_2nd.at(i + 3*j);
        }
      }
    }
  }
  // T0 + T1
  //rowShift = num_xelems;
  //colShift = num_ynodes;
  #pragma omp parallel for firstprivate(mat3x3, mat3x3_2nd) collapse(2)
  for(int xindex = 0; xindex < num_elems; xindex++){
    for(int yindex = 0; yindex < num_nodes; yindex++){
      // T0
      const int out_in = 0;
      const int slp_or_dlp = 2;
      const int linear_or_const = 1;
      std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, xindex + 1, nodes.data(), num_nodes, elems.data(), num_elems, yindex + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
      // T1
      const int out_in_2nd = 1;
      std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, xindex + 1, nodes.data(), num_nodes, elems.data(), num_elems, yindex + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
      for(int j = 0; j < 3; j++){
        for(int i = 0; i < 3; i++){
          cmat[i + 3*xindex + 3*num_nodes + (j + 3*yindex) * nmat] = mat3x3.at(i + 3*j) + mat3x3_2nd.at(i + 3*j);
          //cmat[xindex + rowShift*i + 3*num_xnodes + (yindex + colShift*j) * nmat] = mat3x3.at(i + 3*j) + mat3x3_2nd.at(i + 3*j);
        }
      }
    }
  }
  // -(U0 + U1)
  //rowShift = num_xelems;
  //colShift = num_yelems;
  #pragma omp parallel for firstprivate(mat3x3, mat3x3_2nd) collapse(2)
  for(int xindex = 0; xindex < num_elems; xindex++){
    for(int yindex = 0; yindex < num_elems; yindex++){
      // U0
      const int out_in = 0;
      const int slp_or_dlp = 1;
      const int linear_or_const = 1;
      std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, xindex + 1, nodes.data(), num_nodes, elems.data(), num_elems, yindex + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
      // U1
      const int out_in_2nd = 1;
      std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, xindex + 1, nodes.data(), num_nodes, elems.data(), num_elems, yindex + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
      for(int j = 0; j < 3; j++){
        for(int i = 0; i < 3; i++){
          cmat[i + 3*xindex + 3*num_nodes + (j + 3*yindex + 3*num_nodes) * nmat] = -mat3x3.at(i + 3*j) - (mu0/mu1)*mat3x3_2nd.at(i + 3*j);
          //cmat[xindex + rowShift*i + 3*num_xnodes + (yindex + colShift*j + 3*num_ynodes) * nmat]= -mat3x3.at(i + 3*j) - (mu0/mu1)*mat3x3_2nd.at(i + 3*j);
        }
      }
    }
  }
}

// generates the whole matrix taking into account the reordering
// column major
void MatrixGenerator::gen_matrix_sorted(std::complex<double> cmat[], const double omega, double scale, bool cache) const {
  long long nmat = (num_nodes + num_elems) * 3;
  if (A.rows() == nmat || A.cols() == nmat) {
    // matrix has been cached
    for (long long i = 0; i < nmat; ++i)
      for (long long j = 0; j < nmat; ++j)
        cmat[i * nmat + j] = A(i, j);
  } else {
    if (!scale)
      scale = this->scale;
    long long nmat = (num_nodes + num_elems) * 3;
    // W0 + W1
    //long long rowShift = num_nodes;
    //long long colShift = num_nodes;
    std::vector<std::complex<double>> mat3x3(9, 0.0);
    std::vector<std::complex<double>> mat3x3_2nd(9, 0.0);
    #pragma omp parallel for firstprivate(mat3x3, mat3x3_2nd) collapse(2)
    for(int xindex = 0; xindex < num_nodes; xindex++){
      for(int yindex = 0; yindex < num_nodes; yindex++){
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
        elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[xindex] + 1, nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[yindex] + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
        // W1
        const int out_in_2nd = 1;
        std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
        elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[xindex] + 1, nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[yindex] + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
        for(int j = 0; j < 3; j++){
          for(int i = 0; i < 3; i++){
            cmat[i + 3*xindex + (j + 3*yindex) * nmat] = mat3x3.at(i + 3*j) + (mu1/mu0)*mat3x3_2nd.at(i + 3*j);
            //cmat[xindex + rowShift*i + (yindex + colShift*j) * nmat] = mat3x3.at(i + 3*j) + (mu1/mu0)*mat3x3_2nd.at(i + 3*j);
          }
        }
      }
    }
    // -(aT0 + aT1)
    //rowShift = num_xnodes;
    //colShift = num_yelems;
    #pragma omp parallel for firstprivate(mat3x3, mat3x3_2nd) collapse(2)
    for(int xindex = 0; xindex < num_nodes; xindex++){
      for(int yindex = 0; yindex < num_elems; yindex++){
        // aT0
        const int out_in = 0;
        const int slp_or_dlp = 3;
        const int linear_or_const = 0;
        std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
        elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[xindex] + 1, nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[yindex] + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
        // aT1
        const int out_in_2nd = 1;
        std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
        elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[xindex] + 1, nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[yindex] + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
        for(int j = 0; j < 3; j++){
          for(int i = 0; i < 3; i++){
            cmat[i + 3*xindex + (j + 3*yindex + 3*num_nodes) * nmat] = -mat3x3.at(i + 3*j) - mat3x3_2nd.at(i + 3*j);
            cmat[i + 3*xindex + (j + 3*yindex + 3*num_nodes) * nmat] *= scale;
            //cmat[xindex + rowShift*i + (yindex + colShift*j + 3*num_xnodes)*nmat] = -mat3x3.at(i + 3*j) - mat3x3_2nd.at(i + 3*j);
          }
        }
      }
    }
    // T0 + T1
    //rowShift = num_xelems;
    //colShift = num_ynodes;
    #pragma omp parallel for firstprivate(mat3x3, mat3x3_2nd) collapse(2)
    for(int xindex = 0; xindex < num_elems; xindex++){
      for(int yindex = 0; yindex < num_nodes; yindex++){
        // T0
        const int out_in = 0;
        const int slp_or_dlp = 2;
        const int linear_or_const = 1;
        std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
        elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[xindex] + 1, nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[yindex] + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
        // T1
        const int out_in_2nd = 1;
        std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
        elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[xindex] + 1, nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[yindex] + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
        for(int j = 0; j < 3; j++){
          for(int i = 0; i < 3; i++){
            cmat[i + 3*xindex + 3*num_nodes + (j + 3*yindex) * nmat] = mat3x3.at(i + 3*j) + mat3x3_2nd.at(i + 3*j);
            cmat[i + 3*xindex + 3*num_nodes + (j + 3*yindex) * nmat] *= scale;
            //cmat[xindex + rowShift*i + 3*num_xnodes + (yindex + colShift*j) * nmat] = mat3x3.at(i + 3*j) + mat3x3_2nd.at(i + 3*j);
          }
        }
      }
    }
    // -(U0 + U1)
    //rowShift = num_xelems;
    //colShift = num_yelems;
    #pragma omp parallel for firstprivate(mat3x3, mat3x3_2nd) collapse(2)
    for(int xindex = 0; xindex < num_elems; xindex++){
      for(int yindex = 0; yindex < num_elems; yindex++){
        // U0
        const int out_in = 0;
        const int slp_or_dlp = 1;
        const int linear_or_const = 1;
        std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
        elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[xindex] + 1, nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[yindex] + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
        // U1
        const int out_in_2nd = 1;
        std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
        elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[xindex] + 1, nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[yindex] + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
        for(int j = 0; j < 3; j++){
          for(int i = 0; i < 3; i++){
            cmat[i + 3*xindex + 3*num_nodes + (j + 3*yindex + 3*num_nodes) * nmat] = -mat3x3.at(i + 3*j) - (mu0/mu1)*mat3x3_2nd.at(i + 3*j);
            cmat[i + 3*xindex + 3*num_nodes + (j + 3*yindex + 3*num_nodes) * nmat] *= scale * scale;
            //cmat[xindex + rowShift*i + 3*num_xnodes + (yindex + colShift*j + 3*num_ynodes) * nmat]= -mat3x3.at(i + 3*j) - (mu0/mu1)*mat3x3_2nd.at(i + 3*j);
          }
        }
      }
    }
    if (cache) {
      // cache the whole matrix if desired
      if (A.rows() != nmat || A.cols() != nmat)
        const_cast<Eigen::MatrixXcd&>(A).resize(nmat, nmat);
        Eigen::Map<Eigen::MatrixXcd> A_created(cmat, nmat, nmat);
        const_cast<Eigen::MatrixXcd&>(A) = A_created;
    }
  }
}

// generates the whole matrix for caching purposes
void MatrixGenerator::generateA(const double omega, double scale) {
  if (!scale)
    scale = this->scale;
  long long nmat = (num_nodes + num_elems) * 3;
  A = Eigen::MatrixXcd(nmat, nmat);
  auto cmat = A.data();
  // W0 + W1
  //long long rowShift = num_nodes;
  //long long colShift = num_nodes;
  std::vector<std::complex<double>> mat3x3(9, 0.0);
  std::vector<std::complex<double>> mat3x3_2nd(9, 0.0);
  #pragma omp parallel for firstprivate(mat3x3, mat3x3_2nd) collapse(2)
  for(int xindex = 0; xindex < num_nodes; xindex++){
    for(int yindex = 0; yindex < num_nodes; yindex++){
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
      elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[xindex] + 1, nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[yindex] + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
      // W1
      const int out_in_2nd = 1;
      std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[xindex] + 1, nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[yindex] + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
      for(int j = 0; j < 3; j++){
        for(int i = 0; i < 3; i++){
          cmat[i + 3*xindex + (j + 3*yindex) * nmat] = mat3x3.at(i + 3*j) + (mu1/mu0)*mat3x3_2nd.at(i + 3*j);
          //cmat[xindex + rowShift*i + (yindex + colShift*j) * nmat] = mat3x3.at(i + 3*j) + (mu1/mu0)*mat3x3_2nd.at(i + 3*j);
        }
      }
    }
  }
  // -(aT0 + aT1)
  //rowShift = num_xnodes;
  //colShift = num_yelems;
  #pragma omp parallel for firstprivate(mat3x3, mat3x3_2nd) collapse(2)
  for(int xindex = 0; xindex < num_nodes; xindex++){
    for(int yindex = 0; yindex < num_elems; yindex++){
      // aT0
      const int out_in = 0;
      const int slp_or_dlp = 3;
      const int linear_or_const = 0;
      std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[xindex] + 1, nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[yindex] + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
      // aT1
      const int out_in_2nd = 1;
      std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[xindex] + 1, nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[yindex] + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
      for(int j = 0; j < 3; j++){
        for(int i = 0; i < 3; i++){
          cmat[i + 3*xindex + (j + 3*yindex + 3*num_nodes) * nmat] = -mat3x3.at(i + 3*j) - mat3x3_2nd.at(i + 3*j);
          cmat[i + 3*xindex + (j + 3*yindex + 3*num_nodes) * nmat] *= scale;
          //cmat[xindex + rowShift*i + (yindex + colShift*j + 3*num_xnodes)*nmat] = -mat3x3.at(i + 3*j) - mat3x3_2nd.at(i + 3*j);
        }
      }
    }
  }
  // T0 + T1
  //rowShift = num_xelems;
  //colShift = num_ynodes;
  #pragma omp parallel for firstprivate(mat3x3, mat3x3_2nd) collapse(2)
  for(int xindex = 0; xindex < num_elems; xindex++){
    for(int yindex = 0; yindex < num_nodes; yindex++){
      // T0
      const int out_in = 0;
      const int slp_or_dlp = 2;
      const int linear_or_const = 1;
      std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[xindex] + 1, nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[yindex] + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
      // T1
      const int out_in_2nd = 1;
      std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[xindex] + 1, nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[yindex] + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
      for(int j = 0; j < 3; j++){
        for(int i = 0; i < 3; i++){
          cmat[i + 3*xindex + 3*num_nodes + (j + 3*yindex) * nmat] = mat3x3.at(i + 3*j) + mat3x3_2nd.at(i + 3*j);
          cmat[i + 3*xindex + 3*num_nodes + (j + 3*yindex) * nmat] *= scale;
          //cmat[xindex + rowShift*i + 3*num_xnodes + (yindex + colShift*j) * nmat] = mat3x3.at(i + 3*j) + mat3x3_2nd.at(i + 3*j);
        }
      }
    }
  }
  // -(U0 + U1)
  //rowShift = num_xelems;
  //colShift = num_yelems;
  #pragma omp parallel for firstprivate(mat3x3, mat3x3_2nd) collapse(2)
  for(int xindex = 0; xindex < num_elems; xindex++){
    for(int yindex = 0; yindex < num_elems; yindex++){
      // U0
      const int out_in = 0;
      const int slp_or_dlp = 1;
      const int linear_or_const = 1;
      std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[xindex] + 1, nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[yindex] + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
      // U1
      const int out_in_2nd = 1;
      std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[xindex] + 1, nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[yindex] + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
      for(int j = 0; j < 3; j++){
        for(int i = 0; i < 3; i++){
          cmat[i + 3*xindex + 3*num_nodes + (j + 3*yindex + 3*num_nodes) * nmat] = -mat3x3.at(i + 3*j) - (mu0/mu1)*mat3x3_2nd.at(i + 3*j);
          cmat[i + 3*xindex + 3*num_nodes + (j + 3*yindex + 3*num_nodes) * nmat] *= scale * scale;
          //cmat[xindex + rowShift*i + 3*num_xnodes + (yindex + colShift*j + 3*num_ynodes) * nmat]= -mat3x3.at(i + 3*j) - (mu0/mu1)*mat3x3_2nd.at(i + 3*j);
        }
      }
    }
  }
  //Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(A);
  //std::cout<<"Eigenvalues"<<std::endl;
  //for (int i = 0; i<A.rows(); ++i){
  //  std::cout<<solver.eigenvalues().col(0)[i]<<std::endl;
  //}
  //A.triangularView<Eigen::StrictlyLower>() = A.triangularView<Eigen::StrictlyUpper>().transpose();
}

//generates a block of rows of the matrix, taking into account the reordering
// this creates the matrix in row major now
void MatrixGenerator::gen_matrix_sorted(std::complex<double> cmat[], long long start, const long long num_rows, const double omega, double scale, bool cache) const {
  long long nmat = (num_nodes + num_elems) * 3;
  //if (A.rows() == nmat || A.cols() == nmat) {
  //  // matrix has been cached
  //  for (long long i = 0; i < num_rows * 3; ++i)
  //    for (long long j = 0; j < nmat; ++j)
  //      cmat[i * nmat + j] = A(start * 3 + i, j);
  //} else {
    if (!scale)
      scale = this->scale;
    //long long nmat = num_rows * 3;
    // W0 + W1
    //long long rowShift = num_nodes;
    //long long colShift = num_nodes;
    std::vector<std::complex<double>> mat3x3(9, 0.0);
    std::vector<std::complex<double>> mat3x3_2nd(9, 0.0);
    //std::cout<<"Generate rows "<<start<<"-"<<start+num_rows<<std::endl;
    //std::cout<<mu0<<"-"<<mu1<<std::endl;
    if (start < num_nodes) {
      #pragma omp parallel for firstprivate(mat3x3, mat3x3_2nd) collapse(2)
      for(int xindex = 0; xindex < std::min(num_rows, num_nodes - start); xindex++){
        for(int yindex = 0; yindex < num_nodes; yindex++){
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
          elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[start + xindex] + 1, nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[yindex] + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
          // W1
          const int out_in_2nd = 1;
          std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
          elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[start + xindex] + 1, nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[yindex] + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
          for(int j = 0; j < 3; j++){
            for(int i = 0; i < 3; i++){
              cmat[(i + 3*xindex) * nmat + j + 3*yindex] = mat3x3.at(i + 3*j) + (mu1/mu0)*mat3x3_2nd.at(i + 3*j);
              //std::cout<<cmat[i + 3*xindex + (j + 3*yindex) * nmat]<<std::endl;
              //cmat[xindex + rowShift*i + (yindex + colShift*j) * nmat] = mat3x3.at(i + 3*j) + (mu1/mu0)*mat3x3_2nd.at(i + 3*j);
            }
          }
        }
      }
      // -(aT0 + aT1)
      //rowShift = num_xnodes;
      //colShift = num_yelems;
      #pragma omp parallel for firstprivate(mat3x3, mat3x3_2nd) collapse(2)
      for(int xindex = 0; xindex < std::min(num_rows, num_nodes - start); xindex++){
        for(int yindex = 0; yindex < num_elems; yindex++){
          // aT0
          const int out_in = 0;
          const int slp_or_dlp = 3;
          const int linear_or_const = 0;
          std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
          elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[start + xindex] + 1, nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[yindex] + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
          // aT1
          const int out_in_2nd = 1;
          std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
          elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[start + xindex] + 1, nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[yindex] + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
          for(int j = 0; j < 3; j++){
            for(int i = 0; i < 3; i++){
              cmat[(i + 3*xindex) * nmat + j + 3*yindex + 3*num_nodes] = -mat3x3.at(i + 3*j) - mat3x3_2nd.at(i + 3*j);
              cmat[(i + 3*xindex) * nmat + j + 3*yindex + 3*num_nodes] *= scale;
              //cmat[xindex + rowShift*i + (yindex + colShift*j + 3*num_xnodes)*nmat] = -mat3x3.at(i + 3*j) - mat3x3_2nd.at(i + 3*j);
            }
          }
        }
      }
    }
    long long rows_generated = num_nodes - start > 0 ? num_nodes - start : 0;
    if (num_rows - rows_generated) {
      start = start < num_nodes ? 0 : start - num_nodes;
      // T0 + T1
      //rowShift = num_xelems;
      //colShift = num_ynodes;
      #pragma omp parallel for firstprivate(mat3x3, mat3x3_2nd) collapse(2)
      for(int xindex = rows_generated; xindex < num_rows; xindex++){
        for(int yindex = 0; yindex < num_nodes; yindex++){
          // T0
          const int out_in = 0;
          const int slp_or_dlp = 2;
          const int linear_or_const = 1;
          std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
          elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[start + xindex - rows_generated] + 1, nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[yindex] + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
          // T1
          const int out_in_2nd = 1;
          std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
          elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[start + xindex - rows_generated] + 1, nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[yindex] + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
          for(int j = 0; j < 3; j++){
            for(int i = 0; i < 3; i++){
              cmat[(i + 3*xindex) * nmat + j + 3*yindex] = mat3x3.at(i + 3*j) + mat3x3_2nd.at(i + 3*j);
              cmat[(i + 3*xindex) * nmat + j + 3*yindex] *= scale;
              //cmat[xindex + rowShift*i + 3*num_xnodes + (yindex + colShift*j) * nmat] = mat3x3.at(i + 3*j) + mat3x3_2nd.at(i + 3*j);
            }
          }
        }
      }
      // -(U0 + U1)
      //rowShift = num_xelems;
      //colShift = num_yelems;
      #pragma omp parallel for firstprivate(mat3x3, mat3x3_2nd) collapse(2)
      for(int xindex = rows_generated; xindex < num_rows; xindex++){
        for(int yindex = 0; yindex < num_elems; yindex++){
          // U0
          const int out_in = 0;
          const int slp_or_dlp = 1;
          const int linear_or_const = 1;
          std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
          elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[start + xindex - rows_generated] + 1, nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[yindex] + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
          // U1
          const int out_in_2nd = 1;
          std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
          elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[start + xindex - rows_generated] + 1, nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[yindex] + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
          for(int j = 0; j < 3; j++){
            for(int i = 0; i < 3; i++){
              cmat[(i + 3*xindex) * nmat + j + 3*yindex + 3*num_nodes] = -mat3x3.at(i + 3*j) - (mu0/mu1)*mat3x3_2nd.at(i + 3*j);
              cmat[(i + 3*xindex) * nmat + j + 3*yindex + 3*num_nodes] *= scale * scale;
              //cmat[xindex + rowShift*i + 3*num_xnodes + (yindex + colShift*j + 3*num_ynodes) * nmat]= -mat3x3.at(i + 3*j) - (mu0/mu1)*mat3x3_2nd.at(i + 3*j);
            }
          }
        }
      }
    }
    if (cache) {
      // cache the whole matrix if desired
      if (A.rows() != nmat || A.cols() != nmat)
        const_cast<Eigen::MatrixXcd&>(A).resize(nmat, nmat);
        Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A_rows(cmat, num_rows * 3, nmat);
        const_cast<Eigen::MatrixXcd&>(A).middleRows(start * 3, num_rows) = A_rows;
    }
  //}
}

//generates a block of rows of the matrix, taking into account the reordering from a file
// this creates the matrix in row major now
void MatrixGenerator::gen_matrix_sorted_from_file(std::complex<double> cmat[], long long start, const long long num_rows) const {
  long long n_mat = (num_nodes + num_elems) * 3;
  MPI_Status status;
  MPI_Offset offset = 4 * sizeof(double) +  start * 3 * n_mat * sizeof(std::complex<double>);
  MPI_File_read_at(fh_matrix, offset, cmat, num_rows * 3 * n_mat, MPI_C_DOUBLE_COMPLEX, &status);
}

// generates the rhs (without reordering)
void MatrixGenerator::gen_rhs(std::complex<double> rhs[], const double omega, double scale, bool equation_type) const {
  std::complex<double> alpha = elastWave3d::set_alpha(omega);
  if (!scale)
    scale = this->scale;
  if (equation_type){
    // PMCHWT
    // const size_t nodeShift = num_nodes;
    for(int i = 0; i < num_nodes; i++){
      std::complex<double> uout[3];
      elastWave3d::inc_trac(nodes.data(), num_nodes, i + 1, elems.data(), num_elems, omega, uout); // i + 1 is fortran index
      for(int j = 0; j < 3; j++){
        rhs[j + 3 * i] = uout[j];
        //rhs[i + nodeShift*j] = uout[j];
      }
    }
    //const size_t elemShift = num_elems;
    for(int i = 0; i < num_elems; i++){
      std::complex<double> uout[3];
      elastWave3d::inc_disp_const_x(nodes.data(), num_nodes, elems[i], omega, uout);
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
      elastWave3d::inc_disp(nodes.data(), num_nodes, i + 1, elems.data(), num_elems, omega, uout); // i + 1 is fortran index
      elastWave3d::inc_trac(nodes.data(), num_nodes, i + 1, elems.data(), num_elems, omega, tout); // i + 1 is fortran index
      for(int j = 0; j < 3; j++){
        rhs[j + 3 * i] = uout[j] + alpha*tout[j];
        //rhs[j + nodeShift*i] = uout[j] + alpha*tout[j];
      }
    }
  }
}

// generates the rhs, taking into account the reordering
void MatrixGenerator::gen_rhs_sorted(std::complex<double> rhs[], const double omega, double scale, bool equation_type) const {
  std::complex<double> alpha = elastWave3d::set_alpha(omega);
  if (!scale)
    scale = this->scale;
  if (equation_type){
    // PMCHWT
    // const size_t nodeShift = num_nodes;
    for(int i = 0; i < num_nodes; i++){
      std::complex<double> uout[3];
      elastWave3d::inc_trac(nodes.data(), num_nodes, nodes_idx[i] + 1, elems.data(), num_elems, omega, uout); // i + 1 is fortran index
      for(int j = 0; j < 3; j++){
        rhs[j + 3 * i] = uout[j];
        //rhs[i + nodeShift*j] = uout[j];
      }
    }
    //const size_t elemShift = num_elems;
    for(int i = 0; i < num_elems; i++){
      std::complex<double> uout[3];
      elastWave3d::inc_disp_const_x(nodes.data(), num_nodes, elems[elems_idx[i]], omega, uout);
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
      elastWave3d::inc_disp(nodes.data(), num_nodes, i + 1, elems.data(), num_elems, omega, uout); // i + 1 is fortran index
      elastWave3d::inc_trac(nodes.data(), num_nodes, i + 1, elems.data(), num_elems, omega, tout); // i + 1 is fortran index
      for(int j = 0; j < 3; j++){
        rhs[j + 3 * i] = uout[j] + alpha*tout[j];
        //rhs[j + nodeShift*i] = uout[j] + alpha*tout[j];
      }
    }
  }
}

// generates a certain number of rows of the RHS, taking into account the reordering
void MatrixGenerator::gen_rhs_sorted(std::complex<double> rhs[], long long start, long long num_rows, const double omega, double scale, bool equation_type) const {
  std::complex<double> alpha = elastWave3d::set_alpha(omega);
  if (!scale)
    scale = this->scale;
  if (equation_type){
    // PMCHWT
    if (start < num_nodes) {
      for(int i = 0; i < std::min(num_rows, num_nodes - start); i++){
        std::complex<double> uout[3];
        elastWave3d::inc_trac(nodes.data(), num_nodes, nodes_idx[start + i] + 1, elems.data(), num_elems, omega, uout); // i + 1 is fortran index
        for(int j = 0; j < 3; j++){
          rhs[j + 3 * i] = uout[j];
          //rhs[i + nodeShift*j] = uout[j];
        }
      }
    }
    long long rows_generated = num_nodes - start > 0 ? num_nodes - start : 0;
    if (num_rows - rows_generated) {
      start = start < num_nodes ? 0 : start - num_nodes;
      //const size_t elemShift = num_elems;
      for(int i = rows_generated; i < num_rows; i++){
        std::complex<double> uout[3];
        elastWave3d::inc_disp_const_x(nodes.data(), num_nodes, elems[elems_idx[start + i - rows_generated]], omega, uout);
        for(int j = 0; j < 3; j++){
          rhs[j + 3 * i] = uout[j] * scale;
          //rhs[i + elemShift*j + 3*num_nodes] = uout[j];
      
        }
      }
    }
  }
  else{
    std::cout<<"Burton Miller evaluation, but expected PMCHWT"<<std::endl;
  }
}

// generates a certain number of rows of the RHS, taking into account the reordering from a file
// this creates the matrix in row major now
void MatrixGenerator::gen_rhs_sorted_from_file(std::complex<double> rhs[], long long start, const long long num_rows) const {
  MPI_Status status;
  MPI_Offset offset = start * sizeof(std::complex<double>);
  MPI_File_read_at(fh_rhs, offset, rhs, num_rows, MPI_C_DOUBLE_COMPLEX, &status);
}

void MatrixGenerator::read_mat_metadata(double& mat_size, double& scale, double& omega, double& leaf_size) const {
  MPI_Status status;
  MPI_Offset offset = 0;
  MPI_File_read_at(fh_matrix, offset, &mat_size, 1, MPI_DOUBLE, &status);
  offset += sizeof(double);
  MPI_File_read_at(fh_matrix, offset, &scale, 1, MPI_DOUBLE, &status);
  offset += sizeof(double);
  MPI_File_read_at(fh_matrix, offset, &omega, 1, MPI_DOUBLE, &status);
  offset += sizeof(double);
  MPI_File_read_at(fh_matrix, offset, &leaf_size, 1, MPI_DOUBLE, &status);
}

double MatrixGenerator::calc_scale(const double omega) {
  scale = std::sqrt(get_max_nodes(omega) / get_max_elems(omega));
  return scale;
}

double MatrixGenerator::get_max_elems(const double omega) const {
  std::vector<std::complex<double>> mat3x3(9, 0.0);
  std::vector<std::complex<double>> mat3x3_2nd(9, 0.0);
  double max = 0;
  #pragma omp parallel for firstprivate(mat3x3, mat3x3_2nd)
  for(int xindex = 0; xindex < num_elems; xindex++){
      //std::cout<<"row "<<xindex <<", col "<<yindex<<std::endl;
      
      // U0
      const int out_in = 0;
      const int slp_or_dlp = 1;
      const int linear_or_const = 1;
      std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, xindex + 1, nodes.data(), num_nodes, elems.data(), num_elems, xindex + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
      // U1
      const int out_in_2nd = 1;
      std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, xindex + 1, nodes.data(), num_nodes, elems.data(), num_elems, xindex + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
      for(int j = 0; j < 3; j++){
        std::complex<double> value = -mat3x3.at(j + 3*j) - (mu0/mu1) * mat3x3_2nd.at(j + 3*j);
        if (std::abs(value.real()) > max)
          max = std::abs(value.real());
      }
  }
  return max;
}

double MatrixGenerator::get_max_nodes(const double omega) const {
  //long long nmat = (num_xnodes + num_xelems) * 3;
  // W0 + W1
  //long long rowShift = num_xnodes;
  //long long colShift = num_ynodes;
  std::vector<std::complex<double>> mat3x3(9, 0.0);
  std::vector<std::complex<double>> mat3x3_2nd(9, 0.0);
  double max = 0;
  //#pragma omp parallel for firstprivate(mat3x3, mat3x3_2nd)
  //for(int xindex = 0; xindex < num_nodes; xindex++){
  for(int xindex = 0; xindex < num_nodes; xindex++){
      //std::cout<<"row "<<xindex <<", col "<<yindex<<std::endl;
      
      // W0
      const int out_in = 0;
      const int slp_or_dlp = 4;
      const int linear_or_const = 0;
      std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, xindex + 1, nodes.data(), num_nodes, elems.data(), num_elems, xindex + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
      // W1
      const int out_in_2nd = 1;
      std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, xindex + 1, nodes.data(), num_nodes, elems.data(), num_elems, xindex + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
      for(int j = 0; j < 3; j++){
        std::complex<double> value = mat3x3.at(j + 3*j) + (mu1/mu0) * mat3x3_2nd.at(j + 3*j);
          if (std::abs(value.real()) > max)
            max = std::abs(value.real());
      }
  }
  return max;
}

// generates a block of the matrix from row and colum indices, taking into account the reordering
// indices are actual matrix indices and not node/element indices
void MatrixGenerator::gen_matrix_element(std::complex<double> cmat[], const long long row_indices[], const long long num_rows, const long long col_indices[], const long long num_cols, const double omega, double scale) const {
  //long long nmat = (num_nodes + num_elems) * 3;
  //for (long long i = 0; i < num_rows; ++i)
  //  for (long long j = 0; j < num_cols; ++j)
  //    cmat[i + j * num_rows] = A(row_indices[i], col_indices[j]);
  if (!scale)
    scale = this->scale;
  long long nmat = (num_nodes + num_elems) * 3;
  std::vector<std::complex<double>> mat3x3(9, 0.0);
  std::vector<std::complex<double>> mat3x3_2nd(9, 0.0);
  for (long long i = 0; i < num_rows; ++i) {
    long long row_idx = row_indices[i] / 3;
    //std::cout<<row_indices[i]<<std::endl;
    if (row_idx < num_nodes) {
      for (long long j = 0; j < num_cols; ++j) {
        long long col_idx = col_indices[j] / 3;
        if (col_idx < num_nodes) {
          //std::cout<<row_idx<<" "<<col_idx<<std::endl;
          //std::cout<<nodes_idx_inv[row_idx]<< " "<<nodes_idx_inv[col_idx]<<std::endl;
          //std::cout<<nodes_idx[row_idx]<< " " <<nodes_idx[col_idx] <<std::endl;
          //std::cout<<"Write to mat at "<< i + j *num_rows<<" from "<<i%3 + 3* (j%3)<<std::endl;
          // W0 + W1
          // W0
          const int out_in = 0;
          const int slp_or_dlp = 4;
          const int linear_or_const = 0;
          std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
          elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[row_idx] + 1, nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[col_idx] + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
          // W1
          const int out_in_2nd = 1;
          std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
          elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[row_idx] + 1, nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[col_idx] + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
          cmat[i + j * num_rows] = mat3x3.at(row_indices[i]%3 + 3* (col_indices[j]%3)) + (mu1/mu0)*mat3x3_2nd.at(row_indices[i]%3 + 3* (col_indices[j]%3));
        } else {
          col_idx -= num_nodes;
          // -(aT0 + aT1)
          // aT0
          const int out_in = 0;
          const int slp_or_dlp = 3;
          const int linear_or_const = 0;
          std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
          elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[row_idx] + 1, nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[col_idx] + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
          // aT1
          const int out_in_2nd = 1;
          std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
          elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[row_idx] + 1, nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[col_idx] + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
          cmat[i + j * num_rows] = -mat3x3.at(row_indices[i]%3 + 3*(col_indices[j]%3)) - mat3x3_2nd.at(row_indices[i]%3 + 3*(col_indices[j]%3));
          cmat[i + j * num_rows] *= scale;
        }
      }
    } else {
      row_idx -= num_nodes;
      for (long long j = 0; j < num_cols; ++j) {
        long long col_idx = col_indices[j] / 3;
        if (col_idx < num_nodes) {
          // T0 + T1
          // T0
          const int out_in = 0;
          const int slp_or_dlp = 2;
          const int linear_or_const = 1;
          std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
          elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[row_idx] + 1, nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[col_idx] + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
          // T1
          const int out_in_2nd = 1;
          std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
          elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[row_idx] + 1, nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[col_idx] + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
          cmat[i + j * num_rows] = mat3x3.at(row_indices[i]%3 + 3*(col_indices[j]%3)) + mat3x3_2nd.at(row_indices[i]%3 + 3*(col_indices[j]%3));
          cmat[i + j * num_rows] *= scale;
        } else {
          col_idx -= num_nodes;
          // -(U0 + U1)
          // U0
          const int out_in = 0;
          const int slp_or_dlp = 1;
          const int linear_or_const = 1;
          std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
          elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[row_idx] + 1, nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[col_idx] + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
          // U1
          const int out_in_2nd = 1;
          std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
          elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[row_idx] + 1, nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[col_idx] + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
          cmat[i + j * num_rows] = -mat3x3.at(row_indices[i]%3 + 3*(col_indices[j]%3)) - (mu0/mu1)*mat3x3_2nd.at(row_indices[i]%3 + 3*(col_indices[j]%3));
          cmat[i + j * num_rows] *= scale * scale;
        }
      }
    }
  }
}

// generates a block of the matrix from row and colum indices, taking into account the reordering
// indices are actual matrix indices and not node/element indices
void MatrixGenerator::gen_matrix_element_from_file(std::complex<double> cmat[], const long long row_indices[], const long long num_rows, const long long col_indices[], const long long num_cols) const {
  long long n_mat = (num_nodes + num_elems) * 3;
  MPI_Status status;
  MPI_Offset offset;
  std::vector<std::complex<double>> tmp(n_mat);
  for (long long i = 0; i < num_rows; ++i) {
    offset = 4 * sizeof(double) + row_indices[i] * n_mat * sizeof(std::complex<double>);
    MPI_File_read_at(fh_matrix, offset, tmp.data(), n_mat, MPI_C_DOUBLE_COMPLEX, &status);
    for (long long j = 0; j < num_cols; ++j)
      cmat[i + j * num_rows] = tmp[col_indices[j]];
  }
}

// generates a block of the matrix, taking into account the reordering
// row indices are matrix indices, but column indices are node/element indices
// this function uses the actual 3x3 indices ofr the rows, but node+element indices for the column space
void MatrixGenerator::gen_matrix_idx_element(std::complex<double> cmat[], const long long row_indices[], const long long num_rows, const long long col_indices[], const long long num_cols, const double omega, double scale) const {
  //std::cout<<"Matrix with "<<num_rows<<" rows and "<<num_cols*3<<" columns"<<std::endl;
  /*long long nmat = (num_nodes + num_elems) * 3;
  for (long long i = 0; i < num_rows; ++i)
    for (long long j = 0; j < num_cols; ++j)
      for (long long d = 0; d < 3; ++d)
        cmat[i + (j * 3 + d) * num_rows] = A(row_indices[i], col_indices[j] * 3 + d);
  */
  if (!scale)
    scale = this->scale;
  long long nmat = (num_nodes + num_elems) * 3;
  std::vector<std::complex<double>> mat3x3(9, 0.0);
  std::vector<std::complex<double>> mat3x3_2nd(9, 0.0);
  for (long long i = 0; i < num_rows; ++i) {
    long long row_idx = row_indices[i] / 3;
    //std::cout<<row_indices[i]<<std::endl;
    if (row_idx < num_nodes) {
      for (long long j = 0; j < num_cols; ++j) {
        long long col_idx = col_indices[j];
        if (col_idx < num_nodes) {
          //std::cout<<row_idx<<" "<<col_idx<<std::endl;
          //std::cout<<nodes_idx_inv[row_idx]<< " "<<nodes_idx_inv[col_idx]<<std::endl;
          //std::cout<<nodes_idx[row_idx]<< " " <<nodes_idx[col_idx] <<std::endl;
          //std::cout<<"Write to mat at "<< i + j *num_rows<<" from "<<i%3 + 3* (j%3)<<std::endl;
          // W0 + W1
          // W0
          const int out_in = 0;
          const int slp_or_dlp = 4;
          const int linear_or_const = 0;
          std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
          elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[row_idx] + 1, nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[col_idx] + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
          // W1
          const int out_in_2nd = 1;
          std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
          elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[row_idx] + 1, nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[col_idx] + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
          for (long long d = 0; d < 3; ++d)
            cmat[i + (j * 3 + d) * num_rows] = mat3x3.at(row_indices[i]%3 + 3 * d) + (mu1/mu0)*mat3x3_2nd.at(row_indices[i]%3 + 3 * d);
        } else {
          col_idx -= num_nodes;
          // -(aT0 + aT1)
          // aT0
          const int out_in = 0;
          const int slp_or_dlp = 3;
          const int linear_or_const = 0;
          std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
          elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[row_idx] + 1, nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[col_idx] + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
          // aT1
          const int out_in_2nd = 1;
          std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
          elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[row_idx] + 1, nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[col_idx] + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
          for (long long d = 0; d < 3; ++d) {
            cmat[i + (j * 3 + d) * num_rows] = -mat3x3.at(row_indices[i]%3 + 3 * d) - mat3x3_2nd.at(row_indices[i]%3 + 3 * d);
            cmat[i + (j * 3 + d) * num_rows] *= scale;
          }
        }
      }
    } else {
      row_idx -= num_nodes;
      for (long long j = 0; j < num_cols; ++j) {
        long long col_idx = col_indices[j];
        if (col_idx < num_nodes) {
          // T0 + T1
          // T0
          const int out_in = 0;
          const int slp_or_dlp = 2;
          const int linear_or_const = 1;
          std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
          elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[row_idx] + 1, nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[col_idx] + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
          // T1
          const int out_in_2nd = 1;
          std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
          elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[row_idx] + 1, nodes.data(), num_nodes, elems.data(), num_elems, nodes_idx[col_idx] + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
          for (long long d = 0; d < 3; ++d) {
            cmat[i + (j * 3 + d) * num_rows] = mat3x3.at(row_indices[i]%3 + 3 * d) + mat3x3_2nd.at(row_indices[i]%3 + 3 * d);
            cmat[i + (j * 3 + d) * num_rows] *= scale;
          }
        } else {
          col_idx -= num_nodes;
          // -(U0 + U1)
          // U0
          const int out_in = 0;
          const int slp_or_dlp = 1;
          const int linear_or_const = 1;
          std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
          elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[row_idx] + 1, nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[col_idx] + 1, omega, out_in, slp_or_dlp, linear_or_const, mat3x3.data());
          // U1
          const int out_in_2nd = 1;
          std::fill(mat3x3_2nd.begin(), mat3x3_2nd.end(), 0.0);
          elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[row_idx] + 1, nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[col_idx] + 1, omega, out_in_2nd, slp_or_dlp, linear_or_const, mat3x3_2nd.data());
          for (long long d = 0; d < 3; ++d) {
            cmat[i + (j * 3 + d) * num_rows] = -mat3x3.at(row_indices[i]%3 + 3 * d) - (mu0/mu1)*mat3x3_2nd.at(row_indices[i]%3 + 3 * d);
            cmat[i + (j * 3 + d) * num_rows] *= scale * scale;
          }
        }
      }
    }
  }
}

// generates a block of the matrix, taking into account the reordering
// row indices are matrix indices, but column indices are node/element indices
// this function uses the actual 3x3 indices ofr the rows, but node+element indices for the column space
void MatrixGenerator::gen_matrix_idx_element_from_file(std::complex<double> cmat[], const long long row_indices[], const long long num_rows, const long long col_indices[], const long long num_cols) const {
  long long n_mat = (num_nodes + num_elems) * 3;
  MPI_Status status;
  MPI_Offset offset;
  long long num_elems = 1;
  for (long long i = 1; i < num_cols; ++i) {
    if (col_indices[i - 1] + 1 == col_indices[i])
      ++num_elems;
    else
      break;
  }
  for (long long i = 0; i < num_rows; ++i) {
    offset = 4 * sizeof(double) + row_indices[i] * n_mat * sizeof(std::complex<double>);
    MPI_File_read_at(fh_matrix, offset + col_indices[0] * 3 * sizeof(std::complex<double>), &cmat[i * num_cols * 3], num_elems * 3, MPI_C_DOUBLE_COMPLEX, &status);
    if (num_elems < num_cols) {
      MPI_File_read_at(fh_matrix, offset + col_indices[num_elems] * 3 * sizeof(std::complex<double>), &cmat[i * num_cols * 3 + num_elems * 3], (num_cols - num_elems) * 3, MPI_C_DOUBLE_COMPLEX, &status);
    }
  }
}

void MatrixGenerator::writeA(const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    //typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
    long long rows = A.rows();
    long long cols = A.cols();
    out.write((char*) (&rows), sizeof(long long));
    out.write((char*) (&cols), sizeof(long long));
    out.write((char*) (&scale), sizeof(double));
    out.write((char*) A.data(), rows*cols*sizeof(typename Eigen::MatrixXcd::Scalar));
    out.close();
}

void MatrixGenerator::readA(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    long long rows, cols;
    in.read((char*) (&rows),sizeof(long long));
    in.read((char*) (&cols),sizeof(long long));
    in.read((char*) (&scale),sizeof(double));
    A.resize(rows, cols);
    in.read( (char *) A.data(), rows*cols*sizeof(typename Eigen::MatrixXcd::Scalar));
    in.close();
}

void MatrixGenerator::open_matrix_file(const std::string& filename) {
  if (MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_matrix) != MPI_SUCCESS)
    std::cerr <<"could not open file '" << filename << "'" << std::endl;
}

void MatrixGenerator::open_rhs_file(const std::string& filename) {
  if (MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_rhs) != MPI_SUCCESS)
    std::cerr << "could not open file '" << filename << "'" << std::endl;
}