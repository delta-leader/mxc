
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
  //Laplace3D<DT> eval(1.);
  //Yukawa3D<DT> eval(1, 1.);
  //Gaussian<DT> eval(8);
  Helmholtz3D<DT> eval(1., 1.);
  
  // body contains the points
  // 3 corresponds to the dimension
  std::vector<double> body(Nbody * 3);
  // contains the charges for each point?
  //std::vector<std::complex<double>> Xbody(Nbody);
  Vector_dt<DT> Xbody(Nbody);
  // array containing the nodes in the cluster tree
  std::vector<Cell> cell(ncells);

  // create the points (i.e. bodies)
  //mesh_sphere(&body[0], Nbody, std::pow(Nbody, 1./2.));
  uniform_unit_cube_rnd(&body[0], Nbody, std::pow(Nbody, 1./3.), 3, 999);
  //uniform_unit_cube(&body[0], Nbody, std::pow(Nbody, 1./3.), 3);
  //build the tree (i.e. set the values in the cell array)
  buildBinaryTree(levels, Nbody, &body[0], &cell[0]);

  // generate a random vector Xbody (used in Matvec)
  Xbody.generate_random();
  
  // HSS construction
  MPI_Barrier(MPI_COMM_WORLD);
  double hss_construct_time = MPI_Wtime(), hss_construct_comm_time;
  H2MatrixSolver hss(eval, epi, rank, cell, 0, &body[0], levels);
  MPI_Barrier(MPI_COMM_WORLD);
  hss_construct_time = MPI_Wtime() - hss_construct_time;
  hss_construct_comm_time = ColCommMPI::get_comm_time();

  // H2 construction
  MPI_Barrier(MPI_COMM_WORLD);
  double h2_construct_time = MPI_Wtime(), h2_construct_comm_time;
  // create the H2 matrix
  H2MatrixSolver h2(eval, epi, rank, cell, theta, &body[0], levels);
  MPI_Barrier(MPI_COMM_WORLD);
  h2_construct_time = MPI_Wtime() - h2_construct_time;
  h2_construct_comm_time = ColCommMPI::get_comm_time();

  // creates two vectors of zeroes with the same length as the number of local bodies
  long long lenX = hss.local_bodies.second - hss.local_bodies.first;
  Vector_dt<DT> X1(lenX);
  Vector_dt<DT> X2(lenX);
  Vector_dt<DT> X3(lenX);
  // copy the random vector
  std::copy(&Xbody[hss.local_bodies.first], &Xbody[hss.local_bodies.second], &X1[0]);
  std::copy(&Xbody[h2.local_bodies.first], &Xbody[h2.local_bodies.second], &X2[0]);

  MPI_Barrier(MPI_COMM_WORLD);
  double hss_matvec_time = MPI_Wtime(), hss_matvec_comm_time;
  // Sample matrix vector multiplication
  hss.matVecMul(&X1[0]);
  MPI_Barrier(MPI_COMM_WORLD);
  hss_matvec_time = MPI_Wtime() - hss_matvec_time;
  hss_matvec_comm_time = ColCommMPI::get_comm_time();

  MPI_Barrier(MPI_COMM_WORLD);
  double h2_matvec_time = MPI_Wtime(), h2_matvec_comm_time;
  // Sample matrix vector multiplication
  h2.matVecMul(&X2[0]);
  MPI_Barrier(MPI_COMM_WORLD);
  h2_matvec_time = MPI_Wtime() - h2_matvec_time;
  h2_matvec_comm_time = ColCommMPI::get_comm_time();

  double refmatvec_time = MPI_Wtime();
  // Reference matrix vector multiplication
  mat_vec_reference(eval, lenX, Nbody, &X3[0], &Xbody[0], &body[hss.local_bodies.first * 3], &body[0]);
  refmatvec_time = MPI_Wtime() - refmatvec_time;
  // calculate relative error between H-matvec and dense matvec
  double hss_err = computeRelErr(lenX, &X1[0], &X3[0]);
  double h2_err = computeRelErr(lenX, &X2[0], &X3[0]);

  int mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0) {
    std::cout << "Construct Err HSS: " << hss_err << std::endl;
    std::cout << "Construct Time HSS: " << hss_construct_time << ", " << hss_construct_comm_time << std::endl;
    std::cout << "Matvec Time HSS: " << hss_matvec_time << ", " << hss_matvec_comm_time << std::endl<<std::endl;
    std::cout << "Construct Err H2: " << h2_err << std::endl;
    std::cout << "Construct Time H2: " << h2_construct_time << ", " << h2_construct_comm_time << std::endl;
    std::cout << "Matvec Time H2: " << h2_matvec_time << ", " << h2_matvec_comm_time << std::endl<<std::endl;
    std::cout << "Dense Matvec Time: " << refmatvec_time << std::endl<<std::endl;
  }
  
  // copy X3 (aka B) into X1, X2
  std::copy(X3.begin(), X3.end(), X1.begin());
  std::copy(X3.begin(), X3.end(), X2.begin());
  std::cout<<"Factorize"<<std::endl;

  // H2
  MPI_Barrier(MPI_COMM_WORLD);
  double h2_factor_time = MPI_Wtime(), h2_factor_comm_time;
  h2.factorizeM();
  MPI_Barrier(MPI_COMM_WORLD);
  h2_factor_time = MPI_Wtime() - h2_factor_time;
  h2_factor_comm_time = ColCommMPI::get_comm_time();

  std::cout<<"Substitute"<<std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  double h2_sub_time = MPI_Wtime(), h2_sub_comm_time;
  h2.solvePrecondition(&X1[0]);
  MPI_Barrier(MPI_COMM_WORLD);
  h2_sub_time = MPI_Wtime() - h2_sub_time;
  h2_sub_comm_time = ColCommMPI::get_comm_time();
  std::cout<<"Error"<<std::endl;
  h2_err = computeRelErr(lenX, &X1[0], &Xbody[hss.local_bodies.first]);
  Vector_dt<DT> R(lenX);
  std::cout<<"REference"<<std::endl;
  mat_vec_reference(eval, lenX, Nbody, &R[0], &X1[0], &body[hss.local_bodies.first * 3], &body[0]);
  std::cout<<"Error"<<std::endl;
  double h2_res = computeRelErr(lenX, &R[0], &X3[0]);

  // HSS
  /*MPI_Barrier(MPI_COMM_WORLD);
  double hss_factor_time = MPI_Wtime(), hss_factor_comm_time;
  hss.factorizeM();
  MPI_Barrier(MPI_COMM_WORLD);
  hss_factor_time = MPI_Wtime() - hss_factor_time;
  hss_factor_comm_time = ColCommMPI::get_comm_time();

  MPI_Barrier(MPI_COMM_WORLD);
  double hss_sub_time = MPI_Wtime(), hss_sub_comm_time;
  hss.solvePrecondition(&X2[0]);
  MPI_Barrier(MPI_COMM_WORLD);
  hss_sub_time = MPI_Wtime() - hss_sub_time;
  hss_sub_comm_time = ColCommMPI::get_comm_time();
  hss_err = computeRelErr(lenX, &X2[0], &Xbody[hss.local_bodies.first]);
  mat_vec_reference(eval, lenX, Nbody, &R[0], &X2[0], &body[hss.local_bodies.first * 3], &body[0]);
  double hss_res = computeRelErr(lenX, &R[0], &X3[0]);

  if (mpi_rank == 0) {
    std::cout << "Factorization Time HSS: " << hss_factor_time << ", " << hss_factor_comm_time << std::endl;
    std::cout << "Substitution Time HSS: " << hss_sub_time << ", " << hss_sub_comm_time << std::endl;
    std::cout << "Substitution Err HSS: " << hss_err << std::endl;
    std::cout << "Residual HSS: " << hss_res << std::endl<<std::endl;
    std::cout << "Factorization Time H2: " << h2_factor_time << ", " << h2_factor_comm_time << std::endl;
    std::cout << "Substitution Time H2: " << h2_sub_time << ", " << h2_sub_comm_time << std::endl;
    std::cout << "Substitution Err H2: " << h2_err << std::endl;
    std::cout << "Residual H2: " << h2_res << std::endl;
  }*/
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

  h2.free_all_comms();
  hss.free_all_comms();
  MPI_Finalize();
  return 0;
}

