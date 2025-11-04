
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
  long long num_spheres = argc > 2 ? std::atof(argv[2]) : 1;
  double omega = argc > 3 ? std::atof(argv[3]) : 1;
  long long leaf_size = argc > 4 ? std::atoll(argv[4]) : 32;

  const std::string MAT = std::to_string(M);

  MatrixGenerator matgen(M, num_spheres);
  long long Nbody = matgen.get_num_nodes() + matgen.get_num_elems();
  std::cout<<"Nodes/Elements: "<<matgen.get_num_nodes()<<" "<<matgen.get_num_elems()<<std::endl;

  leaf_size = Nbody < leaf_size ? Nbody : leaf_size;
  
  long long levels, Nleaf, ncells;
  std::vector<Cell> cell;
  long long levels_elems = (long long) std::ceil(std::log2((double)matgen.get_num_elems() / leaf_size));
  long long Nleaf_elems = (long long)1 << levels_elems;
  long long ncells_elems = Nleaf_elems + Nleaf_elems - 1;
  long long levels_nodes = (long long) std::ceil(std::log2((double)matgen.get_num_nodes() / leaf_size));
  long long Nleaf_nodes = (long long)1 << levels_nodes;
  long long ncells_nodes = Nleaf_nodes + Nleaf_nodes - 1;
  levels = levels_elems;
  Nleaf = Nleaf_nodes + Nleaf_elems;
  ncells = ncells_elems + ncells_nodes + 1;
  cell.resize(ncells);
  buildBinaryTreeNodes(&cell[0], matgen.get_nodes().data(), matgen.get_nodes_idx().data(), matgen.get_num_nodes(), levels_nodes, 1);
  buildBinaryTreeElems(&cell[0], matgen.get_elems().data(), matgen.get_elems_idx().data(), matgen.get_num_elems(), levels_elems, 0, matgen.get_num_nodes());
  /* root has three children */
  cell[0].Child[0] = 1;
  cell[0].Child[1] = 4;
  cell[0].Body[0] = 0;
  cell[0].Body[1] = matgen.get_num_nodes() + matgen.get_num_elems();
  for (int d = 0; d < 3; ++d) {
    cell[0].R[d] = cell[1].R[d];
    cell[0].C[d] = (cell[0].C[d] + cell[1].C[d]) / 2;
  }

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

  std::vector<std::complex<double>> A_gen(own_rows * 3 * n_mat);
  Eigen::VectorXcd rhs_gen(own_rows * 3);
  double scale = matgen.calc_scale(omega);
  std::cout<<"Generating matrix "<<own_rows*3<<" "<<n_mat<<std::endl;
  std::cout<< num_rows *mpi_rank<<" "<<own_rows<<std::endl;
  if (mpi_rank == 0){
    matgen.gen_matrix_sorted(A_gen.data(), num_rows * mpi_rank, own_rows, omega, scale);
    std::cout<<"Process "<<mpi_rank<<" finished"<<std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (mpi_rank == 1){
    matgen.gen_matrix_sorted(A_gen.data(), num_rows * mpi_rank, own_rows, omega, scale);
    std::cout<<"Process "<<mpi_rank<<" finished"<<std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (mpi_rank == 2){
    matgen.gen_matrix_sorted(A_gen.data(), num_rows * mpi_rank, own_rows, omega, scale);
    std::cout<<"Process "<<mpi_rank<<" finished"<<std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout<<"Generated matrix"<<std::endl;
  MPI_File fh;
  std::string filename = "../input/cache/sphere_" + std::to_string(num_spheres) + "_" + MAT + "_" + std::to_string((int)omega) + "_" + std::to_string(leaf_size) + ".dat";
  std::cout<<"Open File "<<filename<<std::endl;
  MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  MPI_Offset offset = 0;
  MPI_Status status;
  std::cout<<"Write Preamble"<<std::endl;
  if (mpi_rank == 0) {
    double nmat = n_mat;
    MPI_File_write_at(fh, offset, &nmat, 1, MPI_DOUBLE, &status);
    //std::cout<<"Status "<<(int)status<<std::endl;
    offset += sizeof(double);
    MPI_File_write_at(fh, offset, &scale, 1, MPI_DOUBLE, &status);
    offset += sizeof(double);
    MPI_File_write_at(fh, offset, &omega, 1, MPI_DOUBLE, &status);
    offset += sizeof(double);
    double leafS = leaf_size;
    MPI_File_write_at(fh, offset, &leafS, 1, MPI_DOUBLE, &status);
  }
  std::cout<<"Write Data"<<std::endl;
  offset = 4 * sizeof(double) + mpi_rank * num_rows * 3 * n_mat * sizeof(std::complex<double>);
  // write into the actual file
  MPI_File_write_at(fh, offset, A_gen.data(), own_rows * 3 * n_mat, MPI_C_DOUBLE_COMPLEX, &status);
  std::cout<<"Process "<<mpi_rank<<" finished writing"<<std::endl;
  // I still need an offset into this
  //matgen.gen_rhs_sorted(rhs_gen.data(), omega, scale);
  // Serialize into a single file here
  MPI_File_close(&fh);

  // testing
  
  MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
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
  MPI_Finalize();
  return 0;
}

