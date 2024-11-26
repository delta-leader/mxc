
#include <solver.hpp>
#include <test_funcs.hpp>
#include <string>
#include <omp.h>
#include <mkl.h>

#include <Eigen/Dense>

int main(int argc, char* argv[]) {
  typedef std::complex<double> DT;
  typedef std::complex<float> DT_low;
  MPI_Init(&argc, &argv);

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
  std::string geom = argc > 8 ? std::string(argv[8]) : "cube";
  const char* csv = argc > 9 ? argv[9] : nullptr;

  leaf_size = Nbody < leaf_size ? Nbody : leaf_size;
  long long levels = (long long)std::log2((double)Nbody / leaf_size);
  long long Nleaf = (long long)1 << levels;
  long long ncells = Nleaf + Nleaf - 1;
  
  //Laplace3D eval(1.);
  //Yukawa3D eval(1, 1.);
  //Gaussian eval(0.005);
  Helmholtz3D<DT> eval(4., 1e-1);
  
  std::vector<double> body(Nbody * 3);
  MyVector<DT> Xbody(Nbody);
  std::vector<Cell> cell(ncells);
  if (geom == "cube") {
    uniform_unit_cube_rnd(&body[0], Nbody, 1, 3, 999);
  } else {
    if (geom == "sphere") {
      mesh_sphere(&body[0], Nbody, std::sqrt(Nbody / (4 * M_PI)));
    } else {
      if (geom == "ball") {
        mesh_ball(&body[0], Nbody, 999);
      } else {
        std::cout<<geom<<" is not a valid geometry!"<<std::endl;
        return 1;
      }
    }
  }
  //mesh_sphere(&body[0], Nbody, std::sqrt(Nbody / (4 * M_PI)));
  //uniform_unit_cube_rnd(&body[0], Nbody, 1, 3, 999);
  //uniform_unit_cube(&body[0], Nbody, std::pow(Nbody, 1./3.), 3);
  buildBinaryTree(&cell[0], &body[0], Nbody, levels);

  Xbody.generate_random(999, 0, 1);

  /*cell.erase(cell.begin() + 1, cell.begin() + Nleaf - 1);
  cell[0].Child[0] = 1; cell[0].Child[1] = Nleaf + 1;
  ncells = Nleaf + 1;
  levels = 1;*/
  //omp_set_num_threads(4);
  //mkl_set_num_threads(2);
  std::cout<<"OpenMP: "<<omp_get_max_threads()<<std::endl;
  //std::cout<<"OpenMP: "<<omp_get_num_threads()<<std::endl;
  std::cout<<"MKL: "<<mkl_get_max_threads()<<std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  double h2_construct_time = MPI_Wtime(), h2_construct_comm_time;
  H2MatrixSolver<DT> matA(eval, epi, rank, leveled_rank, cell, theta, &body[0], levels);

  MPI_Barrier(MPI_COMM_WORLD);
  h2_construct_time = MPI_Wtime() - h2_construct_time;
  h2_construct_comm_time = ColCommMPI::get_comm_time();

  initNcclComms(&nccl_comms, matA.allocedComm);
  matA.init_gpu_handles(nccl_comms);
  matA.allocSparseMV(handle, nccl_comms);

  long long lenX = matA.local_bodies.second - matA.local_bodies.first;
  MyVector<DT> X1(lenX, DT(0., 0.));
  MyVector<DT> X2(lenX, DT(0., 0.));

  std::copy(&Xbody[matA.local_bodies.first], &Xbody[matA.local_bodies.second], &X1[0]);

  MPI_Barrier(MPI_COMM_WORLD);
  double matvec_time = MPI_Wtime(), matvec_comm_time;
  //matA.matVecMul(&X1[0]);
  matA.matVecMulSp(handle, &X1[0]);

  MPI_Barrier(MPI_COMM_WORLD);
  matvec_time = MPI_Wtime() - matvec_time;
  matvec_comm_time = ColCommMPI::get_comm_time();

  double refmatvec_time = MPI_Wtime();

  mat_vec_reference(eval, lenX, Nbody, &X2[0], &Xbody[0], &body[matA.local_bodies.first * 3], &body[0]);
  refmatvec_time = MPI_Wtime() - refmatvec_time;
  double cerr = solveRelErr(lenX, &X1[0], &X2[0]);

  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  if (mpi_rank == 0) {
    std::cout << "Construct Err: " << cerr << std::endl;
    std::cout << "H^2-Matrix Construct Time: " << h2_construct_time << ", " << h2_construct_comm_time << std::endl;
    std::cout << "H^2-Matvec Time: " << matvec_time << ", " << matvec_comm_time << std::endl;
    std::cout << "Dense Matvec Time: " << refmatvec_time << std::endl;
    /*Eigen::MatrixXcd A(Nbody, Nbody);
    gen_matrix(eval, Nbody, Nbody, body.data(), body.data(), A.data());
    double cond = 1. / A.lu().rcond();
    std::cout << "Condition #: " << cond << std::endl;*/
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double m_construct_time = MPI_Wtime(), m_construct_comm_time;
  H2MatrixSolver<DT> matM;
  std::cout<<"init"<<std::endl;
  if (mode.compare("h2") == 0)
    matM.construct_factorize(eval, 0., rank, leveled_rank, cell, theta, &body[0], levels, nccl_comms, handle);
  else if (mode.compare("hss") == 0)
    matM.construct_factorize(eval, 0., rank, leveled_rank, cell, 0., &body[0], levels, nccl_comms, handle);
  std::cout<<"init finished"<<std::endl;

  //H2MatrixSolver<DT_low> matM_low(matM);
  MPI_Barrier(MPI_COMM_WORLD);
  m_construct_time = MPI_Wtime() - m_construct_time;
  m_construct_comm_time = ColCommMPI::get_comm_time();

  std::copy(&Xbody[matM.local_bodies.first], &Xbody[matM.local_bodies.second], &X1[0]);
  matM.matVecMul(&X1[0]);
  double cerr_m = solveRelErr(lenX, &X1[0], &X2[0]);

  //initNcclComms(&nccl_comms, matM.allocedComm);
  //matM.init_gpu_handles(nccl_comms);

  MPI_Barrier(MPI_COMM_WORLD);
  double h2_factor_time = MPI_Wtime(), h2_factor_comm_time;
  //matM.factorizeM();
  //matM.factorizeDeviceM(handle);
  //matM_low.factorizeDeviceM(handle, CUBLAS_COMPUTE_32F_FAST_16F);

  MPI_Barrier(MPI_COMM_WORLD);
  h2_factor_time = MPI_Wtime() - h2_factor_time;
  h2_factor_comm_time = ColCommMPI::get_comm_time();
  std::copy(X2.begin(), X2.end(), X1.begin());
  //MyVector<DT_low> X1_low(X1);

  MPI_Barrier(MPI_COMM_WORLD);
  double h2_sub_time = MPI_Wtime(), h2_sub_comm_time;

  //matM.solvePrecondition(&X1[0]);
  matM.solvePreconditionDevice(handle, &X1[0]);
  //matM_low.solvePreconditionDevice(handle, &X1_low[0]);
  //X1 = MyVector<DT>(X1_low);

  MPI_Barrier(MPI_COMM_WORLD);
  h2_sub_time = MPI_Wtime() - h2_sub_time;
  h2_sub_comm_time = ColCommMPI::get_comm_time();
  double serr = solveRelErr(lenX, &X1[0], &Xbody[matM.local_bodies.first]);
  std::fill(X1.begin(), X1.end(), DT(0., 0.));

  if (mpi_rank == 0) {
    std::cout << "H^2-Preconditioner Construct Time: " << m_construct_time << ", " << m_construct_comm_time << std::endl;
    std::cout << "H^2-Preconditioner Construct Err: " << cerr_m << std::endl;
    std::cout << "H^2-Matrix Factorization Time: " << h2_factor_time << ", " << h2_factor_comm_time << std::endl;
    std::cout << "H^2-Matrix Substitution Time: " << h2_sub_time << ", " << h2_sub_comm_time << std::endl;
    std::cout << "H^2-Matrix Substitution Err: " << serr << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double gmres_time = MPI_Wtime(), gmres_comm_time;
  //matA.solveGMRES(epi, matM, &X1[0], &X2[0], 10, 50);
  matA.solveGMRESDevice(handle, epi, matM, &X1[0], &X2[0], 10, 50);

  MPI_Barrier(MPI_COMM_WORLD);
  gmres_time = MPI_Wtime() - gmres_time;
  gmres_comm_time = ColCommMPI::get_comm_time();

  if (mpi_rank == 0) {
    std::cout << "GMRES Residual: " << matA.resid[matA.iters] << ", Iters: " << matA.iters << std::endl;
    std::cout << "GMRES Time: " << gmres_time << ", Comm: " << gmres_comm_time << std::endl;
    for (long long i = 0; i <= matA.iters; i++)
      std::cout << "iter "<< i << ": " << matA.resid[i] << std::endl;

    if (csv != nullptr)
      write_to_csv(csv, mpi_size, Nbody, theta, leaf_size, rank, epi, mode.data(), cerr, 
        h2_construct_time, h2_construct_comm_time, matvec_time, matvec_comm_time, refmatvec_time, 
        m_construct_time, m_construct_comm_time, cerr_m, h2_factor_time, h2_factor_comm_time, h2_sub_time, h2_sub_comm_time, serr, 
        matA.resid[matA.iters], matA.iters, gmres_time, gmres_comm_time, matA.resid.data());
  }

  matA.free_all_comms();
  matM.free_all_comms();
  MPI_Finalize();

  matA.freeSparseMV();
  matA.free_gpu_handles();
  matM.free_gpu_handles();
  finalizeGpuEnvs(handle);
  finalizeNcclComms(nccl_comms);
  return 0;
}
