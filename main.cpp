
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
  //ncclComms nccl_comms = nullptr;
  cudaSetDevice();
  initGpuEnvs(&handle);

  long long Nbody = argc > 1 ? std::atoll(argv[1]) : 2048;
  double theta = argc > 2 ? std::atof(argv[2]) : 1e0;
  long long leaf_size = argc > 3 ? std::atoll(argv[3]) : 256;
  long long rank = argc > 4 ? std::atoll(argv[4]) : 50;
  long long leveled_rank =  argc > 5 ? std::atoll(argv[5]) : 0;
  //double epi = argc > 6 ? std::atof(argv[6]) : 1e-10;
  //std::string mode = argc > 7 ? std::string(argv[7]) : "h2";
  std::string geom = argc > 6 ? std::string(argv[6]) : "cube";
  //const char* csv = argc > 9 ? argv[9] : nullptr;

  leaf_size = Nbody < leaf_size ? Nbody : leaf_size;
  long long levels = (long long)std::log2((double)Nbody / leaf_size);
  long long Nleaf = (long long)1 << levels;
  long long ncells = Nleaf + Nleaf - 1;

  std::cout<<"N, admis, leaf-size, rank, leveled-rank, geometry"<<std::endl;
  std::cout<<Nbody<<", "<<theta<<", "<<leaf_size<<", "<<rank<<", "<<leveled_rank<<", "<<geom<<std::endl;
  std::cout<<"Construction, Factorization, Substitution"<<std::endl;
  
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
  buildBinaryTree(&cell[0], &body[0], Nbody, levels);
  Xbody.generate_random(999, 0, 1);

  //omp_set_num_threads(4);
  //mkl_set_num_threads(2);
  std::cout<<"OpenMP: "<<omp_get_max_threads()<<std::endl;
  //std::cout<<"OpenMP: "<<omp_get_num_threads()<<std::endl;
  std::cout<<"MKL: "<<mkl_get_max_threads()<<std::endl;

  MyVector<DT> X1(Nbody);
  X1.generate_random(999, 0, 1);

  const int RUNS = 6;
  for (int i = 0; i<RUNS; ++i) {
    double construct_time = MPI_Wtime();
    H2MatrixSolver<DT> matM(eval, 0., rank, leveled_rank, cell, theta, &body[0], levels);
    construct_time = MPI_Wtime() - construct_time;
    std::cout<<construct_time<<", ";

    double factor_time = MPI_Wtime();
    matM.init_gpu_handles();
    matM.factorizeDeviceM(handle);
    //matM_low.factorizeDeviceM(handle, CUBLAS_COMPUTE_32F_FAST_16F);
    factor_time = MPI_Wtime() - factor_time;
    std::cout<<factor_time<<", ";

    double sub_time = MPI_Wtime();
    //matM.solvePrecondition(&X1[0]);
    matM.solvePreconditionDevice(handle, &X1[0]);
    //matM_low.solvePreconditionDevice(handle, &X1_low[0]);
    //X1 = MyVector<DT>(X1_low);
    sub_time = MPI_Wtime() - sub_time;
    std::cout<<sub_time<<std::endl;
    matM.free_gpu_handles();
  }
  MPI_Finalize();
  finalizeGpuEnvs(handle);
  return 0;
}
