
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

  MPI_Init(&argc, &argv);
  typedef std::complex<double> DT; typedef std::complex<float> DT_low;
  //typedef double DT; typedef float DT_low;
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
  Kernel kfunc = argc > 4 ? (Kernel) std::atoi(argv[4]) : LAPLACE;
  bool fact_basis = argc > 5 ? (bool) std::atoi(argv[5]) : true;

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

  std::vector<double> ranks = {16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256};
  int mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0) {
    std::cout<<kernel_names[kfunc] << " N = "<< Nbody << ", L = "<< leaf_size << ", Admis = " << theta << std::endl;
  }
  for (size_t g=0; g<geometry_names.size(); ++g) {
    if (mpi_rank == 0) {
      std::cout<<geometry_names[g]<<std::endl;
    }
    if (g == SPHERE)
      //uniform_unit_cube(&body[0], Nbody, std::pow(Nbody, 1./3.), 3);
      mesh_sphere(&body[0], Nbody);
    else
      if (g == BALL)
        //uniform_unit_cube_rnd(&body[0], Nbody, std::pow(Nbody, 1./3.), 3, 999);
        mesh_ball(&body[0], Nbody, 999);
      else 
        if (mpi_rank == 0) {
          std::cout<<"Unknown Geometry"<<std::endl;
        }
    buildBinaryTree(levels, Nbody, &body[0], &cell[0]);

    for (size_t i=0; i<params.size(); ++i) {
      MatrixAccessor<DT>* eval;
      switch (kfunc) {
        case LAPLACE:
          eval = new Laplace3D<DT>(params[i]);
          break;
        case GAUSSIAN:
          //eval = new Gaussian<DT>(params[i]);
          eval = new Gaussian<DT>(params[i], 1e-2);
          break;
        case IMQ:
          //eval = new Imq<DT>(params[i]);
          eval = new Imq<DT>(params[i], 1e-2);
          break;
        case MATERN:
          //eval = new Matern3<DT>(params[i]);
          eval = new Matern3<DT>(params[i], 1e-2);
          break;
        case YUKAWA:
          eval = new Yukawa3D<DT>(1, params[i]);
          //eval = new Yukawa3D<DT>(params[i], 0.5);
          break;
        case HELMHOLTZ:
          eval = new Helmholtz3D<DT>(1, params[i]);
          //eval = new Helmholtz3D<DT>(params[i], 1);
      }
      
  
      std::vector<double> a_accs(ranks.size());
      for (size_t j=0; j<ranks.size(); ++j) {
        H2MatrixSolver<DT> matM = H2MatrixSolver(*eval, 1e-12, ranks[j], cell, theta, &body[0], levels, true, fact_basis);
        long long lenX = matM.local_bodies.second - matM.local_bodies.first;
        Vector_dt<DT> X1(lenX);
        std::copy(&Xbody[matM.local_bodies.first], &Xbody[matM.local_bodies.second], &X1[0]);
        matM.matVecMul(&X1[0]);
        Vector_dt<DT> X2(lenX);
        mat_vec_reference(*eval, lenX, Nbody, &X2[0], &Xbody[0], &body[matM.local_bodies.first*3], &body[0]);
        a_accs[j] = computeRelErr(lenX, &X1[0], &X2[0]);
        matM.free_all_comms();
        if (a_accs[j] < 1e-12)
          break;
      }

      MPI_Barrier(MPI_COMM_WORLD);
      if (mpi_rank == 0) {
        std::cout<<params[i]<<", ";
        //for (size_t j=0; j<ranks.size(); ++j)
          //std::cout<<ranks[j]<<", ";
        //std::cout<<std::endl;
        for (size_t j=0; j<a_accs.size(); ++j)
          std::cout<<a_accs[j]<<", ";
        std::cout<<std::endl;
      }
    }
    if (mpi_rank == 0) {
      std::cout<<std::endl;
    }
  }
  MPI_Finalize();
  return 0;
}

