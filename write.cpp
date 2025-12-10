
#include <solver.hpp>
#include <test_funcs.hpp>
#include <include/elast3d.hpp>
#include <kernel.hpp>
#include <string>

#include <Eigen/Dense>

#include <fstream>

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  long long M = argc > 1 ? std::atoll(argv[1]) : 160;
  long long num_spheres = argc > 2 ? std::atoll(argv[2]) : 1;
  double omega = argc > 3 ? std::atof(argv[3]) : 1;
  long long leaf_size = argc > 4 ? std::atoll(argv[4]) : 32;

  const std::string MAT = std::to_string(M);
  MatrixGenerator matgen(M, num_spheres);
  long long Nbody = matgen.get_num_elems();
  std::cout<<"Elements: "<<matgen.get_num_elems()<<std::endl;

  leaf_size = Nbody < leaf_size ? Nbody : leaf_size;
  
  std::vector<Cell> cell;
  long long levels = (long long) std::ceil(std::log2((double)matgen.get_num_elems() / leaf_size));
  long long Nleaf = (long long)1 << levels;
  long long ncells = Nleaf + Nleaf - 1;
  cell.resize(ncells);
  buildBinaryTreeElemsOnly(&cell[0], matgen.get_elems().data(), matgen.get_elems_idx().data(), matgen.get_num_elems(), levels);
  std::cout<<"N = "<<Nbody<<", Leaf = "<<leaf_size<<", Levels = "<<levels<<", #Leafs = "<<Nleaf<<", #Cells = "<<ncells<<std::endl;

  long long n_mat = Nbody * 3;
  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  long long num_rows = Nbody / mpi_size;
  long long own_rows = num_rows;
  if (mpi_rank == mpi_size - 1) {
  // last slice needs to be adapted if we don't divide evenly
    own_rows = Nbody - num_rows * (mpi_size - 1);
  }

  std::cout<<"Allocating memory "<<std::endl;
  std::vector<std::complex<double>> A_gen(own_rows * 3 * n_mat);
  // todo : store the rhs
  Eigen::VectorXcd rhs_gen(own_rows * 3);
  //std::cout<<"Generate scale"<<std::endl;
  //double scale = matgen.calc_scale(omega);
  std::cout<<"Generating matrix "<<own_rows*3<<" "<<n_mat<<std::endl;
  std::cout<< num_rows *mpi_rank<<" "<<own_rows<<std::endl;
  matgen.gen_matrix_sorted_single_layer(A_gen.data(), num_rows * mpi_rank, own_rows, omega);
  std::cout<<"Generated matrix"<<std::endl;
  MPI_File fh;
  std::string filename = "../input/cache/" + std::to_string(num_spheres) + "_" + MAT + "_" + std::to_string((int)omega) + "_" + std::to_string(leaf_size) + ".dat";
  std::cout<<"Open File "<<filename<<std::endl;
  MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  MPI_Offset offset = 0;
  MPI_Status status;
  std::cout<<"Write Preamble"<<std::endl;
  if (mpi_rank == 0) {
    double nmat = n_mat;
    MPI_File_write_at(fh, offset, &nmat, 1, MPI_DOUBLE, &status);
    offset += sizeof(double);
    MPI_File_write_at(fh, offset, &omega, 1, MPI_DOUBLE, &status);
    offset += sizeof(double);
    double leafS = leaf_size;
    MPI_File_write_at(fh, offset, &leafS, 1, MPI_DOUBLE, &status);
  }
  //std::cout<<"Write Data"<<std::endl;
  offset = 3 * sizeof(double) + mpi_rank * num_rows * 3 * n_mat * sizeof(std::complex<double>);
  // write into the actual file
  std::cout<<"Writing matrix "<<mpi_rank<<std::endl;
  MPI_File_write_at(fh, offset, A_gen.data(), own_rows * 3 * n_mat, MPI_C_DOUBLE_COMPLEX, &status);
  //std::cout<<"Process "<<mpi_rank<<" finished writing"<<std::endl;
  // I still need an offset into this
  //matgen.gen_rhs_sorted(rhs_gen.data(), omega, scale);
  // Serialize into a single file here
  MPI_File_close(&fh);

  std::string filename_rhs = "../input/cache/rhs_" + std::to_string(num_spheres) + "_" + MAT + "_" + std::to_string((int)omega) + "_" + std::to_string(leaf_size) + ".dat";
  std::cout<<"Open File "<<filename_rhs<<std::endl;
  MPI_File_open(MPI_COMM_WORLD, filename_rhs.c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  offset = mpi_rank * num_rows * 3 * sizeof(std::complex<double>);
  std::cout<<"Writing RHS "<<mpi_rank<<std::endl;
  matgen.gen_rhs_sorted_single_layer(rhs_gen.data(), num_rows * mpi_rank, own_rows, omega);
  // write into the actual file
  MPI_File_write_at(fh, offset, rhs_gen.data(), own_rows * 3, MPI_C_DOUBLE_COMPLEX, &status);
  std::cout<<"Process "<<mpi_rank<<" finished writing"<<std::endl;
  // Serialize into a single file here
  MPI_File_close(&fh);

  // testing
  
  /*MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
  if (mpi_rank == 0) {
  std::vector<std::complex<double>> A2(n_mat * n_mat);
  double read;
  offset = 0;
  MPI_File_read_at(fh, offset, &read, 1, MPI_DOUBLE, &status);
  std::cout<<"Nmat diff "<<n_mat - read<<std::endl;
  offset += sizeof(double);
  MPI_File_read_at(fh, offset, &read, 1, MPI_DOUBLE, &status);
  std::cout<<"Scale diff "<<scale - read<<std::endl;
  offset += sizeof(double);
  MPI_File_read_at(fh, offset, &read, 1, MPI_DOUBLE, &status);
  std::cout<<"Omega diff "<<omega - read<<std::endl;
  offset += sizeof(double);
  MPI_File_read_at(fh, offset, &read, 1, MPI_DOUBLE, &status);
  std::cout<<"leaf diff "<<leaf_size - read<<std::endl;
  offset += sizeof(double);
  MPI_File_read_at(fh, offset, A2.data(), n_mat * n_mat, MPI_C_DOUBLE_COMPLEX, &status);
  double diff = 0;
  for (size_t i = 0; i<5; ++i)
    std::cout<<A_gen[i]<<" vs "<<A2[i]<<std::endl;
  for (size_t i = 0; i<A_gen.size(); ++i)
    diff += A_gen[i].real() - A2[i].real();
  std::cout<<"Matrix diff "<<diff<<std::endl;

  Eigen::MatrixXcd A_check(n_mat, n_mat);
  matgen.gen_matrix_sorted(A_check.data(), omega, scale);
  Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A2_check(A2.data(), n_mat, n_mat);
  std::cout<<"Matrix norm: "<<(A_check - A2_check).norm()<<std::endl;
  }
  MPI_File_close(&fh);

  MPI_File_open(MPI_COMM_WORLD, filename_rhs.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
  if (mpi_rank == 0) {
  std::vector<std::complex<double>> rhs2(n_mat);
  double read;
  offset = 0;
  MPI_File_read_at(fh, offset, rhs2.data(), n_mat, MPI_C_DOUBLE_COMPLEX, &status);
  double diff = 0;
  for (size_t i = 0; i<rhs_gen.size(); ++i)
    diff += rhs_gen[i].real() - rhs2[i].real();
  std::cout<<"RHS diff "<<diff<<std::endl;

  Eigen::VectorXcd rhs_check(n_mat);
  matgen.gen_rhs_sorted(rhs_check.data(), omega, scale);
  Eigen::Map<Eigen::VectorXcd> rhs2_check(rhs2.data(), n_mat);
  std::cout<<"RHS norm: "<<(rhs_check - rhs2_check).norm()<<std::endl;
  }*/
  MPI_Finalize();
  return 0;
}

