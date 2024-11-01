
#include <solver.hpp>
#include <test_funcs.hpp>
#include <string>

#include <Eigen/Dense>

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  typedef std::complex<double> DT;

  deviceHandle_t handle;
  ncclComms nccl_comms = nullptr;
  cudaSetDevice();
  initGpuEnvs(&handle);

  long long Nbody = argc > 1 ? std::atoll(argv[1]) : 2048;
  double theta = argc > 2 ? std::atof(argv[2]) : 1e0;
  long long leaf_size = argc > 3 ? std::atoll(argv[3]) : 256;
  long long rank = argc > 4 ? std::atoll(argv[4]) : 50;
  long long leveled_rank =  argc > 5 ? std::atoll(argv[5]) : 0;
  double epi = argc > 6 ? std::atof(argv[6]) : 1e-10;
  std::string mode = argc > 7 ? std::string(argv[7]) : "h2";
  const char* csv = argc > 8 ? argv[8] : nullptr;

  leaf_size = Nbody < leaf_size ? Nbody : leaf_size;
  long long levels = (long long)std::log2((double)Nbody / leaf_size);
  long long Nleaf = (long long)1 << levels;
  long long ncells = Nleaf + Nleaf - 1;
  
  //Laplace3D eval(1.);
  //Yukawa3D eval(1, 1.);
  //Gaussian eval(0.005);
  //Helmholtz3D eval(4., 1e-1);
  Helmholtz3D eval(4., 1.);
  
  std::vector<double> body(Nbody * 3);
  //std::vector<std::complex<double>> Xbody(Nbody);
  MyVector<DT> Xbody(Nbody);
  std::vector<Cell> cell(ncells);

  //mesh_sphere(&body[0], Nbody, std::sqrt(Nbody / (4 * M_PI)));
  //uniform_unit_cube_rnd(&body[0], Nbody, 1, 3, 999);
  //uniform_unit_cube(&body[0], Nbody, std::pow(Nbody, 1./3.), 3);
  mesh_sphere(&body[0], Nbody);
  buildBinaryTree(&cell[0], &body[0], Nbody, levels);
  Xbody.generate_random(999, -0.5, 0.5);

  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  H2MatrixSolver matM;
  if (mode.compare("h2") == 0)
    matM = H2MatrixSolver(eval, 0., rank, leveled_rank, cell, theta, &body[0], levels);
  else if (mode.compare("hss") == 0)
    matM = H2MatrixSolver(eval, 0., rank, leveled_rank, cell, 0., &body[0], levels);

  long long lenX = matM.local_bodies.second - matM.local_bodies.first;
  MyVector<DT> X1(lenX);
  MyVector<DT> X2(lenX);

  std::copy(&Xbody[matM.local_bodies.first], &Xbody[matM.local_bodies.second], &X1[0]);

  matM.matVecMul(&X1[0]);
  mat_vec_reference(eval, lenX, Nbody, &X2[0], &Xbody[0], &body[matM.local_bodies.first * 3], &body[0]);
  double approx_err = solveRelErr(lenX, &X1[0], &X2[0]);

  initNcclComms(&nccl_comms, matM.allocedComm);
  matM.init_gpu_handles(nccl_comms);

  //matM.factorizeM();
  matM.factorizeDeviceM(handle);
  std::copy(X2.begin(), X2.end(), X1.begin());

  //matM.solvePrecondition(&X1[0]);
  matM.solvePreconditionDevice(handle, &X1[0]);

  double ferr = solveRelErr(lenX, &X1[0], &Xbody[matM.local_bodies.first]);
  //double berr = rel_backward_error(eval, lenX, Nbody, &X2[0], &X1[0], &body[matM.local_bodies.first * 3], &body[0]);

  if (mpi_rank == 0) {
    std::cout << "Approximation Error: " << approx_err << std::endl;
    std::cout << "Forward Error: " << ferr << std::endl;
    //std::cout << "Backward Error: " << berr << std::endl;
  }

  matM.free_all_comms();
  MPI_Finalize();

  matM.free_gpu_handles();
  finalizeGpuEnvs(handle);
  finalizeNcclComms(nccl_comms);
  return 0;
}

