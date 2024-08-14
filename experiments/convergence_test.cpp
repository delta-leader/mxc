#include <Eigen/Dense>

#include <solver.hpp>
#include <test_funcs.hpp>
#include <string>

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  typedef std::complex<double> DT; typedef std::complex<float> DT_low;
  //typedef double DT; typedef float DT_low;

  // N
  long long Nbody = argc > 1 ? std::atoll(argv[1]) : 2048;
  // admis
  double theta = argc > 2 ? std::atof(argv[2]) : 1e0;
  // size of dense blocks
  long long leaf_size = argc > 3 ? std::atoll(argv[3]) : 256;
  long long rank = argc > 4 ? std::atoll(argv[4]) : 100;
  // epsilon
  double epi = argc > 5 ? std::atof(argv[5]) : 1e-10;
  // hmatrix mode
  std::string mode = argc > 6 ? std::string(argv[6]) : "h2";

  // if N <= leaf_size, we basically have a dense matrix
  leaf_size = Nbody < leaf_size ? Nbody : leaf_size;
  // number of levels, works only for multiples of 2
  long long levels = (long long)std::log2((double)Nbody / leaf_size);
  // the max number of leaf level nodes (i.e. if we completely split the matrix)
  long long Nleaf = (long long)1 << levels;
  // the number of cells (i.e. nodes) in the cluster tree
  long long ncells = Nleaf + Nleaf - 1;
  
  // kernel functions, here we select the appropriate function
  // by setting the corresponding parameters
  // In this case the template argument deduction fails for the matvec (double)
  // I might need to specifiy them explicitly
  //Laplace3D<DT> eval(0.1);
  //Yukawa3D<DT> eval(0.1, 1);
  //Gaussian<DT> eval(5.);
  //IMQ<DT> eval(1.);
  //Matern3<DT> eval(1);
  Helmholtz3D<DT> eval(1, 0.1);
  //bool fixed_rank = true;
  double admis = 0;
  
  // body contains the points
  // 3 corresponds to the dimension
  std::vector<double> body(Nbody * 3);
  // contains the charges for each point?
  //std::vector<std::complex<double>> Xbody(Nbody);
  Vector_dt<DT> Xbody(Nbody);
  // array containing the nodes in the cluster tree
  std::vector<Cell> cell(ncells);

  // create the points (i.e. bodies)
  mesh_unit_sphere(&body[0], Nbody);
  //uniform_unit_cube_rnd(&body[0], Nbody, std::pow(Nbody, 1./3.), 3, 999);
  //uniform_unit_cube(&body[0], Nbody, std::pow(Nbody, 1./3.), 3);
  //build the tree (i.e. set the values in the cell array)
  buildBinaryTree(levels, Nbody, &body[0], &cell[0]);

  // generate a random vector Xbody (used in Matvec)
  Xbody.generate_random();
  
  // H2 construction
  MPI_Barrier(MPI_COMM_WORLD);
  double h2_construct_time = MPI_Wtime(), h2_construct_comm_time;
  // create the H2 matrix
  H2MatrixSolver h2(eval, epi, rank, cell, theta, &body[0], levels, false, false);
  MPI_Barrier(MPI_COMM_WORLD);
  h2_construct_time = MPI_Wtime() - h2_construct_time;
  h2_construct_comm_time = ColCommMPI::get_comm_time();

  // creates two vectors of zeroes with the same length as the number of local bodies
  long long lenX = h2.local_bodies.second - h2.local_bodies.first;
  Vector_dt<DT> x(lenX);
  Vector_dt<DT> vh2(lenX);
  Vector_dt<DT> vh2_fixed(lenX);
  Vector_dt<DT> ref(lenX);
  // copy the random vector
  std::copy(&Xbody[h2.local_bodies.first], &Xbody[h2.local_bodies.second], &vh2[0]);
  std::copy(&Xbody[h2.local_bodies.first], &Xbody[h2.local_bodies.second], &vh2_fixed[0]);

  MPI_Barrier(MPI_COMM_WORLD);
  double h2_matvec_time = MPI_Wtime(), h2_matvec_comm_time;
  // Sample matrix vector multiplication
  h2.matVecMul(&vh2[0]);
  MPI_Barrier(MPI_COMM_WORLD);
  h2_matvec_time = MPI_Wtime() - h2_matvec_time;
  h2_matvec_comm_time = ColCommMPI::get_comm_time();

  double refmatvec_time = MPI_Wtime();
  // Reference matrix vector multiplication
  mat_vec_reference(eval, lenX, Nbody, &ref[0], &Xbody[0], &body[h2.local_bodies.first * 3], &body[0]);
  refmatvec_time = MPI_Wtime() - refmatvec_time;
  // calculate relative error between H-matvec and dense matvec
  double h2_err = computeRelErr(lenX, &vh2[0], &ref[0]);

  // single process only
  std::vector<DT> A(Nbody * Nbody);
  gen_matrix(eval, Nbody, Nbody, &body[0], &body[0], A.data());
  Eigen::Map<const Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic>> Amap(&A[0], Nbody, Nbody);
  Eigen::JacobiSVD<Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic>> svd(Amap);
  double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);

  int mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0) {
    std::cout << "Condition Number: " << cond << std::endl;
    std::cout << "Construct Err H2: " << h2_err << std::endl;
    std::cout << "Construct Time H2: " << h2_construct_time << ", " << h2_construct_comm_time << std::endl;
    std::cout << "Matvec Time H2: " << h2_matvec_time << ", " << h2_matvec_comm_time << std::endl<<std::endl;
    std::cout << "Dense Matvec Time: " << refmatvec_time << std::endl<<std::endl;
  }
  
  // copy X3 (aka B) into X1, X2
  std::copy(ref.begin(), ref.end(), vh2.begin());

  H2MatrixSolver h2_fixed(eval, epi, rank, cell, admis, &body[0], levels, true, true);
  h2_fixed.matVecMul(&vh2_fixed[0]);
  double h2_err_fixed = computeRelErr(lenX, &vh2_fixed[0], &ref[0]);

  // H2
  MPI_Barrier(MPI_COMM_WORLD);
  double h2_factor_time = MPI_Wtime(), h2_factor_comm_time;
  h2_fixed.factorizeM();
  MPI_Barrier(MPI_COMM_WORLD);
  h2_factor_time = MPI_Wtime() - h2_factor_time;
  h2_factor_comm_time = ColCommMPI::get_comm_time();

  MPI_Barrier(MPI_COMM_WORLD);
  double ir_time = MPI_Wtime(), ir_comm_time;
  long long iters = h2.solveIR(epi, h2_fixed, &x[0], &vh2[0], 50);
  MPI_Barrier(MPI_COMM_WORLD);
  ir_time = MPI_Wtime() - ir_time;
  ir_comm_time = ColCommMPI::get_comm_time();

  if (mpi_rank == 0) {
    std::cout << "Construct Err H2: " << h2_err_fixed << std::endl;
    std::cout << "Factorization Time H2: " << h2_factor_time << ", " << h2_factor_comm_time << std::endl;
    std::cout << "IR Residual: " << h2.resid[iters] << ", Iters: " << iters << std::endl;
    std::cout << "IR Time: " << ir_time << ", Comm: " << ir_comm_time << std::endl;
  }
  h2.free_all_comms();
  h2_fixed.free_all_comms();
  /*
  MPI_Barrier(MPI_COMM_WORLD);
  double gmres_time = MPI_Wtime(), gmres_comm_time;
  matA.solveGMRES(epi, matM, &X1[0], &X2[0], 10, 50);

  MPI_Barrier(MPI_COMM_WORLD);
  gmres_time = MPI_Wtime() - gmres_time;
  gmres_comm_time = ColCommMPI::get_comm_time();

  if (mpi_rank == 0) {
    std::cout << "GMRES Residual: " << matA.resid[matA.iters] << ", Iters: " << matA.iters << std::endl;
    std::cout << "GMRES Time: " << gmres_time << ", Comm: " << gmres_comm_time << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double ir_time = MPI_Wtime(), ir_comm_time;
  long long iters = matA.solveIR(epi, matM, &X1[0], &X2[0], 50);

  MPI_Barrier(MPI_COMM_WORLD);
  ir_time = MPI_Wtime() - ir_time;
  ir_comm_time = ColCommMPI::get_comm_time();

  if (mpi_rank == 0) {
    std::cout << "IR Residual: " << matA.resid[iters] << ", Iters: " << iters << std::endl;
    std::cout << "IR Time: " << ir_time << ", Comm: " << ir_comm_time << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double ir_time_low = MPI_Wtime(), ir_comm_time_low;
  long long iters_low = matA.solveIR(epi, matM_low, &X1[0], &X2[0], 50);

  MPI_Barrier(MPI_COMM_WORLD);
  ir_time_low = MPI_Wtime() - ir_time_low;
  ir_comm_time_low = ColCommMPI::get_comm_time();

  if (mpi_rank == 0) {
    std::cout << "IR Residual(low): " << matA.resid[iters_low] << ", Iters: " << iters_low << std::endl;
    std::cout << "IR Time(low): " << ir_time_low << ", Comm: " << ir_comm_time_low << std::endl;
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  double gmres_ir_time = MPI_Wtime(), gmres_ir_comm_time;
  long long gmres_iters = matA.solveGMRESIR(epi, matM, &X1[0], &X2[0], 10, 50, 1);

  MPI_Barrier(MPI_COMM_WORLD);
  gmres_ir_time = MPI_Wtime() - gmres_ir_time;
  gmres_ir_comm_time = ColCommMPI::get_comm_time();

  if (mpi_rank == 0) {
    std::cout << "GMRES-IR Residual: " << matA.resid[gmres_iters] << ", Iters: " << gmres_iters << std::endl;
    std::cout << "GMRES-IR Time: " << gmres_ir_time << ", Comm: " << gmres_ir_comm_time << std::endl;
  }

  H2MatrixSolver<DT> matM_low_high(matM_low);

  MPI_Barrier(MPI_COMM_WORLD);
  double gmres_ir_time_low = MPI_Wtime(), gmres_ir_comm_time_low;
  long long gmres_iters_low = matA.solveGMRESIR(epi, matM_low_high, &X1[0], &X2[0], 10, 50, 1);

  MPI_Barrier(MPI_COMM_WORLD);
  gmres_ir_time_low = MPI_Wtime() - gmres_ir_time_low;
  gmres_ir_comm_time_low = ColCommMPI::get_comm_time();

  if (mpi_rank == 0) {
    std::cout << "GMRES-IR Residual: " << matA.resid[gmres_iters_low] << ", Iters: " << gmres_iters_low << std::endl;
    std::cout << "GMRES-IR Time: " << gmres_ir_time_low << ", Comm: " << gmres_ir_comm_time_low << std::endl;
  }
  */

  
  MPI_Finalize();
  return 0;
}

