#include <kernel.hpp>

#include <iostream>
#include <numeric>

#include <Eigen/Dense>
#include <test_funcs.hpp>


// complex double
template void MatrixGenerator::gen_matrix_single_layer(std::complex<double>[], const double) const;
template void MatrixGenerator::gen_matrix_sorted_single_layer(std::complex<double>[], long long start, const long long num_rows, const double omega) const;
template void MatrixGenerator::gen_matrix_sorted_single_layer(std::complex<double>[], long long row_start, const long long num_rows, const long long col_start, const long long num_cols, const double omega) const;
template void MatrixGenerator::gen_matrix_element_single_layer(std::complex<double>[], const long long row_indices[], const long long num_rows, const long long col_indices[], const long long num_cols, const double omega) const;
template void MatrixGenerator::gen_matrix_idx_element_single_layer(std::complex<double>[], const long long row_indices[], const long long num_rows, const long long col_indices[], const long long num_cols, const double omega) const;
template void MatrixGenerator::gen_matrix_hidr_sorted_single_layer(std::complex<double>[], long long row_start, const long long num_rows, const long long col_indices[], const long long num_cols, const double omega) const;
// complex float
template void MatrixGenerator::gen_matrix_single_layer(std::complex<float> cmat[], const double omega) const;
template void MatrixGenerator::gen_matrix_sorted_single_layer(std::complex<float>[], long long start, const long long num_rows, const double omega) const;
template void MatrixGenerator::gen_matrix_sorted_single_layer(std::complex<float>[], long long row_start, const long long num_rows, const long long col_start, const long long num_cols, const double omega) const;
template void MatrixGenerator::gen_matrix_element_single_layer(std::complex<float>[], const long long row_indices[], const long long num_rows, const long long col_indices[], const long long num_cols, const double omega) const;
template void MatrixGenerator::gen_matrix_idx_element_single_layer(std::complex<float>[], const long long row_indices[], const long long num_rows, const long long col_indices[], const long long num_cols, const double omega) const;
template void MatrixGenerator::gen_matrix_hidr_sorted_single_layer(std::complex<float>[], long long row_start, const long long num_rows, const long long col_indices[], const long long num_cols, const double omega) const;


MatrixGenerator::MatrixGenerator(const int size, const int spheres) {
  std::string filename;
  switch (spheres) {
    case 64:
      filename = "../input/new/64_spheres_" + std::to_string(size) + ".inp";
      break;
    case 32:
      filename = "../input/new/32_spheres_" + std::to_string(size) + ".inp";
      break;
    case 16:
      filename = "../input/new/16_spheres_" + std::to_string(size) + ".inp";
      break;
    case 8:
      filename = "../input/new/eight_spheres_" + std::to_string(size) + ".inp";
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
      std::cerr << spheres << " is not a valid indicator for a geometry." << std::endl;
  }
  read_mesh_specs(num_nodes, num_elems, filename);
  nodes.resize(num_nodes);
  elems.resize(num_elems);
  read_mesh_data(num_nodes, nodes, num_elems, elems, size, spheres);
  // prepare index vectors
  nodes_idx.resize(num_nodes);
  std::iota(nodes_idx.begin(), nodes_idx.end(), 0);
  elems_idx.resize(num_elems);
  std::iota(elems_idx.begin(), elems_idx.end(), 0);
}

// generates the whole matrix for the single layer potential (no reordering)
template <typename DT>
void MatrixGenerator::gen_matrix_single_layer(DT cmat[], const double omega) const {
  long long nmat = num_elems * 3;
  std::vector<std::complex<double>> mat3x3(9, 0.0);
  #pragma omp parallel for firstprivate(mat3x3) collapse(2)
  for(int xindex = 0; xindex < num_elems; xindex++){
    for(int yindex = 0; yindex < num_elems; yindex++){
      std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, xindex + 1, nodes.data(), num_nodes, elems.data(), num_elems, yindex + 1, omega, mat3x3.data());
      for(int j = 0; j < 3; j++){
        for(int i = 0; i < 3; i++){
          cmat[i + 3*xindex + (j + 3*yindex) * nmat] = -mat3x3.at(i + 3*j);
        }
      }
    }
  }
}

// generates a block of rows of the matrix, taking into account the reordering
// this creates the matrix in row major layout (for the single layer potential)
template <typename DT>
void MatrixGenerator::gen_matrix_sorted_single_layer(DT cmat[], long long start, const long long num_rows, const double omega) const {
  long long nmat = num_elems * 3;
  std::vector<std::complex<double>> mat3x3(9, 0.0);
  #pragma omp parallel for firstprivate(mat3x3) collapse(2)
  for(int xindex = 0; xindex < num_rows; xindex++){
    for(int yindex = 0; yindex < num_elems; yindex++){
      std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[xindex + start] + 1, nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[yindex] + 1, omega, mat3x3.data());
      for(int j = 0; j < 3; j++){
        for(int i = 0; i < 3; i++){
          cmat[(i + 3*xindex) * nmat + j + 3*yindex] = -mat3x3.at(i + 3*j);
        }
      }
    }
  }
}

// generates a block of size num_rows x num_cols of the matrix, taking into account the reordering
// the block is taken at offset row_start x cols_start (for the single layer potential)
template <typename DT>
void MatrixGenerator::gen_matrix_sorted_single_layer(DT cmat[], long long row_start, const long long num_rows, const long long col_start, const long long num_cols, const double omega) const {
  long long nmat = num_rows * 3;
  std::vector<std::complex<double>> mat3x3(9, 0.0);
  #pragma omp parallel for firstprivate(mat3x3) collapse(2)
  for(int xindex = 0; xindex < num_rows; xindex++){
    for(int yindex = 0; yindex < num_cols; yindex++){
      std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[xindex + row_start] + 1, nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[yindex + col_start] + 1, omega, mat3x3.data());
      for(int j = 0; j < 3; j++){
        for(int i = 0; i < 3; i++){
          cmat[i + 3*xindex + (j + 3*yindex) * nmat] = -mat3x3.at(i + 3*j);
        }
      }
    }
  }
}

// generates a certain number of rows of the RHS, taking into account the reordering
// (for the single layer potential)
void MatrixGenerator::gen_rhs_sorted_single_layer(std::complex<double> rhs[], long long start, long long num_rows, const double omega, const double theta) const {
  std::complex<double> uout[3];
  // divide by 3 to get element indices
  start /= 3;
  #pragma omp parallel for firstprivate(uout)
  for(int i = 0; i < num_rows / 3; i++){
    elastWave3d::inc_disp_const_x(nodes.data(), num_nodes, elems[elems_idx[start + i]], omega, theta, uout);
    for(int j = 0; j < 3; j++){
      rhs[j + 3 * i] = uout[j];   
    }
  }
}

// generates a block of the matrix from row and colum indices, taking into account the reordering
// indices are actual matrix indices and not node/element indices (for the single layer potential)
template <typename DT>
void MatrixGenerator::gen_matrix_element_single_layer(DT cmat[], const long long row_indices[], const long long num_rows, const long long col_indices[], const long long num_cols, const double omega) const {
  std::vector<std::complex<double>> mat3x3(9, 0.0);
  #pragma omp parallel for firstprivate(mat3x3)  collapse(2)
  for (long long i = 0; i < num_rows; ++i) {
    for (long long j = 0; j < num_cols; ++j) {
      long long row_idx = row_indices[i] / 3;
      long long col_idx = col_indices[j] / 3;
      std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[row_idx] + 1, nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[col_idx] + 1, omega, mat3x3.data());
      cmat[i + j * num_rows] = -mat3x3.at(row_indices[i]%3 + 3*(col_indices[j]%3));
    }
  }
}

// generates a block of the matrix, taking into account the reordering
// row indices are matrix indices, but column indices are element indices
// this function uses the actual 3x3 indices for the rows, but element indices for the column space
// (for the single layer potential)
template <typename DT>
void MatrixGenerator::gen_matrix_idx_element_single_layer(DT cmat[], const long long row_indices[], const long long num_rows, const long long col_indices[], const long long num_cols, const double omega) const {
  std::vector<std::complex<double>> mat3x3(9, 0.0);
  #pragma omp parallel for firstprivate(mat3x3) collapse(2)
  for (long long i = 0; i < num_rows; ++i) {
    for (long long j = 0; j < num_cols; ++j) {
      long long row_idx = row_indices[i] / 3;
      long long col_idx = col_indices[j];
      std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[row_idx] + 1, nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[col_idx] + 1, omega, mat3x3.data());
      for (long long d = 0; d < 3; ++d) {  
        cmat[i * num_cols * 3 + j * 3 + d] = -mat3x3.at(row_indices[i]%3 + 3 * d);
      }
    }
  }
}

// generates a block of size num_rows x (num_cols * 3) of the matrix, taking into account the reordering
// the block is taken at offset row_start x 0
// column indices are element indices
// only creates the single layer potential
// used if far field indices are available via HiDR
template <typename DT>
void MatrixGenerator::gen_matrix_hidr_sorted_single_layer(DT cmat[], long long row_start, const long long num_rows, const long long col_indices[], const long long num_cols, const double omega) const {
  long long nmat = num_rows * 3;
  std::vector<std::complex<double>> mat3x3(9, 0.0);
  #pragma omp parallel for firstprivate(mat3x3) collapse(2)
  for(int xindex = 0; xindex < num_rows; xindex++){
    for (long long y = 0; y < num_cols; ++y) {
      long long col_idx = col_indices[y];
      std::fill(mat3x3.begin(), mat3x3.end(), 0.0);
      elastWave3d::mkmat_entrywise_3d_elast(nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[xindex + row_start] + 1, nodes.data(), num_nodes, elems.data(), num_elems, elems_idx[col_idx] + 1, omega, mat3x3.data());
      for(int j = 0; j < 3; j++){
        for(int i = 0; i < 3; i++){
          cmat[i + 3*xindex + (j + 3*y) * nmat] = -mat3x3.at(i + 3*j);
        }
      }
    }
  }
}

void mat_vec_reference(const MatrixGenerator& matgen, long long M, long long N, std::complex<double> B[], const std::complex<double> X[], const long long row_offset, const double omega) {
  constexpr long long size = 128;
  Eigen::Map<const Eigen::VectorXcd> x(X, N * 3);
  Eigen::Map<Eigen::VectorXcd> b(B, M * 3);
  
  for (long long i = 0; i < M; i += size) {
    long long m = std::min(M - i, size);
    Eigen::MatrixXcd A(m * 3, size * 3);

    for (long long j = 0; j < N; j += size) {
      long long n = std::min(N - j, size);
      matgen.gen_matrix_sorted_single_layer(A.data(), row_offset + i, m, j, n, omega);
      b.segment(i * 3, m * 3) += A.leftCols(n * 3) * x.segment(j * 3, n * 3);
    }
  }
}