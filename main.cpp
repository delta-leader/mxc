
#include <solver.hpp>
#include <test_funcs.hpp>
#include <string>
#include <float.h>

#include <mkl.h>
#include <Eigen/Dense>
#include <cublas_v2.h>

int main(int argc, char* argv[]) {
  enum Kernel {LAPLACE, GAUSSIAN, IMQ, MATERN, YUKAWA, HELMHOLTZ};
  std::vector<std::string> kernel_names = {"Laplace", "Gaussian", "IMQ", "Matern", "Yukawa", "Helmholtz"};
  enum Geometry {SPHERE, BALL};
  std::vector<std::string> geometry_names = {"Sphere", "Ball"};
  std::vector<cublasComputeType_t> comp = {CUBLAS_COMPUTE_32F, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_COMPUTE_32F_FAST_16BF};
  std::vector<std::string> ctype_names = {"Single", "TF32", "Half", "Bfloat"};

  MPI_Init(&argc, &argv);
  //typedef std::complex<double> DT; typedef std::complex<float> DT_low;
  typedef double DT; typedef float DT_low;
  //std::vector<cublasComputeType_t> comp = {CUBLAS_COMPUTE_32F, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_COMPUTE_32F_FAST_16BF};
  //COMP = CUBLAS_COMPUTE_32F_FAST_TF32;
  //const auto COMP = CUBLAS_COMPUTE_32F_FAST_16BF;
  //const auto COMP = CUBLAS_COMPUTE_32F_FAST_16F;
  

  // N
  long long Nbody = argc > 1 ? std::atoll(argv[1]) : 2048;
  // admis
  double theta = argc > 2 ? std::atof(argv[2]) : 1e0;
  // size of dense blocks
  long long leaf_size = argc > 3 ? std::atoll(argv[3]) : 256;
  long long rank = argc > 4 ? std::atoll(argv[4]) : 100;
   double epi = argc > 5 ? std::atof(argv[5]) : 1e-10;
  bool fact_basis = argc > 6 ? (bool) std::atoi(argv[6]) : true;
  Kernel kfunc = argc > 7 ? (Kernel) std::atoi(argv[7]) : LAPLACE;
  Geometry geom = argc > 8 ? (Geometry) std::atoi(argv[8]) : SPHERE;
  double alpha = argc > 9 ? std::atof(argv[9]) : 1;
  int ctype = argc > 10 ? std::atof(argv[10]) : 0;
  

  // if N <= leaf_size, we basically have a dense matrix
  leaf_size = Nbody < leaf_size ? Nbody : leaf_size;
  // number of levels, works only for multiples of 2
  long long levels = (long long)std::log2((double)Nbody / leaf_size);
  // the max number of leaf level nodes (i.e. if we completely split the matrix)
  long long Nleaf = (long long)1 << levels;
  // the number of cells (i.e. nodes) in the cluster tree
  long long ncells = Nleaf + Nleaf - 1;

  std::vector<double> params_laplace = {0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100};
  std::vector<double> params_matern = {0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1};
  std::vector<double> params_gaussian = {0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1};
  std::vector<double> params_imq = {0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1};
  std::vector<double> params_yukawa = {0, 0.25, 0.5, 1, 2, 3, 4, 5, 6, 8, 10};
  std::vector<double> params_helmholtz = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  std::vector<double> params;
  switch (kfunc) {
    case LAPLACE:
      params = params_laplace;
      break;
    case GAUSSIAN:
      params = params_gaussian;
      break;
    case IMQ:
      params = params_imq;
      break;
    case MATERN:
      params = params_matern;
      break;
    case YUKAWA:
      params = params_yukawa;
      //params = params_laplace;
      break;
    case HELMHOLTZ:
      params = params_helmholtz;
      //params = params_laplace;
      break;
    default:
      printf("Unknown kernel function\n");
  }
  
  // body contains the points
  // 3 corresponds to the dimension
  std::vector<double> body(Nbody * 3);
  Vector_dt<DT> Xbody(Nbody);
  Xbody.generate_random(999, -0.5, 0.5);
  // array containing the nodes in the cluster tree
  std::vector<Cell> cell(ncells);

  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  if (mpi_rank == 0) {
    std::cout<<kernel_names[kfunc] << ", " << geometry_names[geom] << ", N = "<< Nbody << ", L = "<< leaf_size << ", Admis = " << theta << ", ";
    std::cout << "rank = " << rank << ", epsilon = " << epi << ", fact_basis = " << fact_basis << ", alpha = " << alpha << ", ";
    std::cout << "precision = " << ctype_names[ctype]<<std::endl;
  }
  if (geom == SPHERE)
    //uniform_unit_cube(&body[0], Nbody, std::pow(Nbody, 1./3.), 3);
    mesh_sphere(&body[0], Nbody);
  else
    if (geom == BALL)
      //uniform_unit_cube_rnd(&body[0], Nbody, std::pow(Nbody, 1./3.), 3, 999);
      mesh_ball(&body[0], Nbody, 999);
    else 
      if (mpi_rank == 0) {
        std::cout<<"Unknown Geometry"<<std::endl;
      }
  buildBinaryTree(levels, Nbody, &body[0], &cell[0]);

  MatrixAccessor<DT>* eval;
  switch (kfunc) {
    case LAPLACE:
      eval = new Laplace3D<DT>(alpha);
      break;
    case GAUSSIAN:
      //eval = new Gaussian<DT>(params[i]);
      eval = new Gaussian<DT>(alpha, 1e-2);
      break;
    case IMQ:
      //eval = new Imq<DT>(params[i]);
      eval = new Imq<DT>(alpha, 1e-2);
      break;
    case MATERN:
      //eval = new Matern3<DT>(params[i]);
      eval = new Matern3<DT>(alpha, 1e-2);
      break;
    case YUKAWA:
      eval = new Yukawa3D<DT>(1, alpha);
      //eval = new Yukawa3D<DT>(params[i], 0.5);
      break;
    case HELMHOLTZ:
      eval = new Helmholtz3D<DT>(1, alpha);
      //eval = new Helmholtz3D<DT>(params[i], 1);
  }

  H2MatrixSolver h2_epi(*eval, epi, rank, 0, cell, theta, &body[0], levels, false, false);
  long long lenX = h2_epi.local_bodies.second - h2_epi.local_bodies.first;
  Vector_dt<DT> x(lenX);
  Vector_dt<DT> b(lenX);
  mat_vec_reference(*eval, lenX, Nbody, &b[0], &Xbody[0], &body[h2_epi.local_bodies.first * 3], &body[0]);
  std::copy(b.begin(), b.end(), x.begin()); 
  
  // single process only
  //std::vector<DT> A(Nbody * Nbody);
  //gen_matrix(*eval, Nbody, Nbody, &body[0], &body[0], A.data());
  //Eigen::Map<const Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic>> Amap(&A[0], Nbody, Nbody);
  //Eigen::JacobiSVD<Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic>> svd(Amap);
  //double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
    
  H2MatrixSolver<DT_low> h2_rank_low = H2MatrixSolver(*eval, 1e-12, rank, 0, cell, theta, &body[0], levels, true, fact_basis);
  //h2_rank.factorizeM();
  h2_rank_low.factorizeDeviceM(mpi_rank % mpi_size, comp[ctype]);

  MPI_Barrier(MPI_COMM_WORLD);
  double ir_time = MPI_Wtime(), ir_comm_time;
  long long iters = h2_epi.solveIR(epi, h2_rank_low, &x[0], &b[0], 200);
  MPI_Barrier(MPI_COMM_WORLD);
  ir_time = MPI_Wtime() - ir_time;
  ir_comm_time = ColCommMPI::get_comm_time();

  std::cout<<"IR: ";
  for(size_t i = 0; i < iters + 1; ++i) {
    std::cout<<h2_epi.resid[i]<<", ";
  }
  std::cout<<std::endl;
  
  H2MatrixSolver<DT> h2_rank(h2_rank_low);
  MPI_Barrier(MPI_COMM_WORLD);
  double gmres_ir_time = MPI_Wtime(), gmres_ir_comm_time;
  long long gmres_iters = h2_epi.solveGMRESIR(epi, h2_rank, &x[0], &b[0], 10, 500, 1);

  MPI_Barrier(MPI_COMM_WORLD);
  gmres_ir_time = MPI_Wtime() - gmres_ir_time;
  gmres_ir_comm_time = ColCommMPI::get_comm_time();

  std::cout<<"GMRES-IR: ";
  for(size_t i = 0; i < gmres_iters + 1; ++i) {
    std::cout<<h2_epi.resid[i]<<", ";
  }
  std::cout<<std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  double gmres_time = MPI_Wtime(), gmres_comm_time;
  h2_epi.GMRES_no_restart_direct(epi, h2_rank, &x[0], &b[0], 200);
  MPI_Barrier(MPI_COMM_WORLD);
  gmres_time = MPI_Wtime() - gmres_time;
  gmres_comm_time = ColCommMPI::get_comm_time();

  std::cout<<"GMRES: ";
  for(size_t i = 0; i < h2_epi.iters+1; ++i) {
    std::cout<<h2_epi.resid[i]<<", ";
  }
  std::cout<<std::endl;
  
  x.reset();
  MPI_Barrier(MPI_COMM_WORLD);
  gmres_time = MPI_Wtime();
  h2_epi.solveGMRES(epi, h2_rank, &x[0], &b[0], 50, 10);
  MPI_Barrier(MPI_COMM_WORLD);
  gmres_time = MPI_Wtime() - gmres_time;
  gmres_comm_time = ColCommMPI::get_comm_time();

  std::cout<<"restarted GMRES: ";
  for(size_t i = 0; i < h2_epi.iters+1; ++i) {
    std::cout<<h2_epi.resid[i]<<", ";
  }
  std::cout<<std::endl;
  

  h2_rank.free_all_comms();
  h2_epi.free_all_comms();

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}

