
#include <solver.hpp>
#include <test_funcs.hpp>
#include <string>

#include <Eigen/Dense>

/*class Helmholtz3D : public MatrixAccessor {
public:
  double k;
  double singularity;
  Helmholtz3D(double wave_number, double s) : k(wave_number), singularity(1. / s) {}
  std::complex<double> operator()(double d) const override {
    if (d == 0.)
      return std::complex<double>(singularity, 0.);
    else
      return std::exp(std::complex(0., -k * d)) / d;
  }
};

void gen_matrix(const MatrixAccessor& eval, long long m, long long n, const double* bi, const double* bj, std::complex<double> Aij[]) {
  const std::array<double, 3>* bi3 = reinterpret_cast<const std::array<double, 3>*>(bi);
  const std::array<double, 3>* bi3_end = reinterpret_cast<const std::array<double, 3>*>(&bi[3 * m]);
  const std::array<double, 3>* bj3 = reinterpret_cast<const std::array<double, 3>*>(bj);
  const std::array<double, 3>* bj3_end = reinterpret_cast<const std::array<double, 3>*>(&bj[3 * n]);

  std::for_each(bj3, bj3_end, [&](const std::array<double, 3>& j) -> void {
    long long ix = std::distance(bj3, &j);
    std::for_each(bi3, bi3_end, [&](const std::array<double, 3>& i) -> void {
      long long iy = std::distance(bi3, &i);
      double d = std::hypot(i[0] - j[0], i[1] - j[1], i[2] - j[2]);
      Aij[iy + ix * m] = eval(d);
    });
  });
}*/

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  /*deviceHandle_t handle;
  ncclComms nccl_comms = nullptr;
  cudaSetDevice();
  initGpuEnvs(&handle);*/

  long long Nbody = argc > 1 ? std::atoll(argv[1]) : 2048;
  double theta = argc > 2 ? std::atof(argv[2]) : 1e0;
  long long leaf_size = argc > 3 ? std::atoll(argv[3]) : 256;
  long long rank = argc > 4 ? std::atoll(argv[4]) : 50;
  long long leveled_rank =  argc > 5 ? std::atoll(argv[5]) : 0;
  double epi = argc > 6 ? std::atof(argv[6]) : 1e-10;
  std::string mode = argc > 7 ? std::string(argv[7]) : "h2";
  const char* csv = argc > 8 ? argv[8] : nullptr;

  leaf_size = Nbody < leaf_size ? Nbody : leaf_size;
  long long levels = (long long) std::ceil(std::log2((double)Nbody / leaf_size));
  long long Nleaf = (long long)1 << levels;
  long long ncells = Nleaf + Nleaf - 1;

  long long n_nodes, n_elems;
  std::vector<double> nodes;
  std::vector<double> elems;
  std::vector<double> elems_polar;
  // Reading the mes data (i.e. nodes and elems)
  // For the elements we calculate the centroid and store it in elems
  // indices is ignored for now
  std::vector<long long> indices = read_mesh_data(n_nodes, nodes, n_elems, elems, elems_polar, "../input/mesh_sphere_160nodes.inp");
  std::cout<<nodes.size()/3<<" " <<elems.size()/3<<std::endl;

  std::cout<<Nbody<<" "<<leaf_size<<" "<<levels<<" "<<Nleaf<<" "<<ncells<<std::endl;
  std::vector<Cell> cell(ncells);
  // create index array for the elements
  std::vector<long long> idx(n_elems);
  std::iota(idx.begin(), idx.end(), 0);
  // build the tree for U (element-element interactions and reorder the corresponding indices)
  buildBinaryTree(&cell[0], &elems[0], idx.data(), Nbody, levels);
  //buildBinaryTree2(&cell[0], &elems_polar[0], idx.data(), Nbody, levels);

  /* kmeans */
  /*std::vector<int> counts = {13, 8, 9, 7, 9, 7, 17, 10, 10, 12, 9, 11, 9, 10, 11, 11, 9, 8, 10, 10, 10, 7, 9, 11, 9, 9, 10, 11, 8, 13, 11, 8};
  long long cnt = 0;
  for (long long i = 0; i < Nleaf; ++i) {
     cell[Nleaf - 1 + i].Body[0] = cnt;
     cnt += counts[i];
     cell[Nleaf - 1 + i].Body[1] = cnt;
  }
  std::vector<int> offsets(counts.size());
  offsets[0] =0;
  for (long long i = 1; i < Nleaf; ++i) {
     offsets[i] = offsets[i - 1] + counts[i - 1];
  }*/
  
  // read the rhs, reference solution and matrix from the file
  long long n_mat = (n_nodes + n_elems) * 3;
  std::vector<std::complex<double>> b(n_mat);
  read_data(b.data(), "../input/rhs_sphere_160.dat", n_mat);
  std::vector<std::complex<double>> x(n_mat);
  read_data(x.data(), "../input/x_sphere_160.dat", n_mat);
  std::vector<std::complex<double>> mat(n_mat * n_mat);
  read_data(mat.data(), "../input/mat_sphere_160.dat", n_mat * n_mat);

  // Get the U matrix (element/element interactions and sort it according to the tree)
  Eigen::Map<Eigen::MatrixXcd> A(mat.data(), n_mat,  n_mat);
  Eigen::MatrixXcd U = A.bottomRightCorner(n_elems * 3, n_elems * 3);
  Eigen::MatrixXcd U_sorted(Nbody * 3, Nbody * 3);
  std::vector<long long> idx2(Nbody);
  for (long long i = 0; i < Nbody; ++i) {
    idx2[idx[i]] = i;
  }

  /*std::vector<int> kmeans = {19,  3, 26,  4, 26,  2,  3, 24, 17,  7,  6, 19, 28, 26, 30, 20, 11,  9, 28,  0, 28, 30, 23,  3,
    23, 30,  9,  1, 11,  6, 12, 19, 29, 11, 19, 28,  1, 19, 20, 11, 24, 10,  0, 26, 12,  2, 18, 14,
    30, 16,  4, 28, 20, 19,  9, 14, 16, 12, 12, 19, 26, 29, 22, 12, 31,  6,  0,  8, 6, 31,  7, 11,
    11, 11, 11,  4,  9,  4,  4,  9, 16, 16, 27,  4, 27, 16, 16, 16, 20, 20, 29,  7, 31, 31, 20, 20,
     6,  8, 26, 10,  8, 10,  8, 10, 26,  2, 10, 14,  2, 14, 10,  2,  2, 10, 25,  2, 30, 25,  2, 25,
    1, 25,  1, 23, 13, 21, 25, 17, 13, 13, 25, 17, 17, 13,  5,  5, 17, 15, 15,  5, 24, 22,  9, 15,
    15,  5, 22, 19, 27, 15,  5, 22, 22,  7,  6, 15,  9, 24, 22, 18, 18, 11, 20, 12, 18, 17,  7, 30,
     6,  0, 19, 12, 15,  0,  1, 26, 13, 21, 21,  4, 15, 18, 29, 26, 30,  0, 23,  0,  7,  8,  5, 24,
    29, 20, 18, 22,  7, 22,  7,  6, 31, 28, 22,  9, 27,  1, 15, 23, 20,  9, 11, 27,  0, 24, 15,  0,
    13, 30, 12, 25,  2, 23, 26,  1, 30, 10, 24, 27, 18,  1,  9, 19, 14, 13,  6,  7, 25, 15, 31, 13,
    21,  4,  0, 30, 14,  9,  6, 27, 24, 10, 27, 29, 27, 12, 29, 21, 18, 14,  5, 17, 17, 16,  6, 29,
     7, 24,  3, 23, 18, 14,  3,  3, 31, 13, 23,  0,  1, 29,  0,  9,  6, 28, 8,  8, 29, 18,  0, 27,
    30,  6, 11, 16, 29,  8,27, 14, 28, 23,  6,  3, 31, 29, 29, 25, 21,  6, 23, 23, 14,  8,  8, 21,
     6,  6, 14,  4};
     for (long long i = 0; i < n_elems; ++i) {
      idx2[i] = offsets[kmeans[i]]++;
    }*/
  for (int i = 0; i < Nbody; ++i) {
    for (int j = 0; j < Nbody; ++j) {
      for (int ii = 0; ii < 3; ++ii) {
        for (int jj = 0; jj < 3; ++jj) {
           //U_sorted(idx2[i] * 3 + ii, idx2[j] * 3 + jj) = U(i * 3 + ii, j * 3 + jj);
           U_sorted(i * 3 + ii, j * 3 + jj) = U(idx[i] * 3 + ii , idx[j] * 3 + jj);
        }
      }
    }
  }

    /*for (int i = 0; i < 30; ++i) {
      //for (int ii = 0; ii < 3; ++ii) {
        for (int j = 0; j < 30; ++j) {
          //for (int jj = 0; jj < 3; ++jj) {
            std::cout<<U_sorted(88*3+j, 88*3+i)<<", ";
          //}
        }
      //}
      std::cout<<std::endl;
    }*/
  // generate the H2 matrix
  //H2MatrixSolver matA(U, epi, rank, leveled_rank, cell, theta, levels);
  std::cout<<"Solver"<<std::endl;
  H2MatrixSolver matA(U_sorted, epi, rank, leveled_rank, cell, theta, levels);
  
  //Laplace3D eval(1.);
  //Yukawa3D eval(1, 1.);
  //Gaussian eval(0.005);
//   Helmholtz3D eval(1., 1e-1);
  
//   std::vector<double> body(Nbody * 3);
   std::vector<std::complex<double>> Xbody(Nbody * 3);
//   std::vector<Cell> cell(ncells);

//   mesh_sphere(&body[0], Nbody, std::sqrt(Nbody / (4 * M_PI)));
//   //uniform_unit_cube_rnd(&body[0], Nbody, 1, 3, 999);
//   //uniform_unit_cube(&body[0], Nbody, std::pow(Nbody, 1./3.), 3);
//   buildBinaryTree(&cell[0], &body[0], Nbody, levels);

   std::mt19937_64 gen;
   std::uniform_real_distribution uniform_dist(0., 1.);
   std::generate(Xbody.begin(), Xbody.end(), 
     [&]() { return std::complex<double>(uniform_dist(gen), 0.); });

//   /*cell.erase(cell.begin() + 1, cell.begin() + Nleaf - 1);
//   cell[0].Child[0] = 1; cell[0].Child[1] = Nleaf + 1;
//   ncells = Nleaf + 1;
//   levels = 1;*/

//   DenseZMat denseA(Nbody, Nbody);
//   gen_matrix(eval, Nbody, Nbody, &body[0], &body[0], denseA.A);
  
//   MPI_Barrier(MPI_COMM_WORLD);
//   double h2_construct_time = MPI_Wtime(), h2_construct_comm_time;
//   H2MatrixSolver matA(denseA, eval, epi, rank, leveled_rank, cell, theta, &body[0], levels);

//   MPI_Barrier(MPI_COMM_WORLD);
//   h2_construct_time = MPI_Wtime() - h2_construct_time;
//   h2_construct_comm_time = ColCommMPI::get_comm_time();

//   /*initNcclComms(&nccl_comms, matA.allocedComm);
//   matA.init_gpu_handles(nccl_comms);
//   matA.allocSparseMV(handle, nccl_comms);*/

  long long lenX = Nbody * 3;
  std::vector<std::complex<double>> X1(lenX, std::complex<double>(0., 0.));
  std::vector<std::complex<double>> X2(lenX, std::complex<double>(0., 0.));

  std::copy(&Xbody[0], &Xbody[lenX], &X1[0]);
  std::copy(&Xbody[0], &Xbody[lenX], &X2[0]);

  MPI_Barrier(MPI_COMM_WORLD);
  double matvec_time = MPI_Wtime(), matvec_comm_time;
  matA.matVecMul(&X1[0]);
  //matA.matVecMulSp(handle, &X1[0]);
  //Eigen::MatrixXcd RX = U_sorted.triangularView<Eigen::Lower>();
  //Eigen::MatrixXcd RX2 = U_sorted.triangularView<Eigen::StrictlyLower>().transpose();
  //Eigen::MatrixXcd RX3 = RX + RX2;
  //Eigen::Map<Eigen::VectorXcd> t2(&X2[0], lenX);
  //Eigen::VectorXcd r = RX3 * t2;

  MPI_Barrier(MPI_COMM_WORLD);
  matvec_time = MPI_Wtime() - matvec_time;
  matvec_comm_time = ColCommMPI::get_comm_time();

  double refmatvec_time = MPI_Wtime();
  Eigen::Map<Eigen::VectorXcd> t(&X2[0], lenX);
  Eigen::VectorXcd result = U_sorted * t;

  refmatvec_time = MPI_Wtime() - refmatvec_time;
  double cerr = H2MatrixSolver::solveRelErr(lenX, &X1[0], result.data());
  //double cerr = H2MatrixSolver::solveRelErr(lenX, r.data(), result.data());

  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  if (mpi_rank == 0) {
    std::cout << "Construct Err: " << cerr << std::endl;
    //std::cout << "H^2-Matrix Construct Time: " << h2_construct_time << ", " << h2_construct_comm_time << std::endl;
    //std::cout << "H^2-Matvec Time: " << matvec_time << ", " << matvec_comm_time << std::endl;
    //std::cout << "Dense Matvec Time: " << refmatvec_time << std::endl;
    /*Eigen::MatrixXcd A(Nbody, Nbody);
    gen_matrix(eval, Nbody, Nbody, body.data(), body.data(), A.data());
    double cond = 1. / A.lu().rcond();
    std::cout << "Condition #: " << cond << std::endl;*/
  }

//   MPI_Barrier(MPI_COMM_WORLD);
//   double m_construct_time = MPI_Wtime(), m_construct_comm_time;
//   H2MatrixSolver matM;
//   if (mode.compare("h2") == 0)
//     matM = H2MatrixSolver(denseA, eval, 0., rank, leveled_rank, cell, theta, &body[0], levels);
//   else if (mode.compare("hss") == 0)
//     matM = H2MatrixSolver(denseA, eval, 0., rank, leveled_rank, cell, 0., &body[0], levels);

//   MPI_Barrier(MPI_COMM_WORLD);
//   m_construct_time = MPI_Wtime() - m_construct_time;
//   m_construct_comm_time = ColCommMPI::get_comm_time();

//   std::copy(&Xbody[matM.local_bodies.first], &Xbody[matM.local_bodies.second], &X1[0]);
//   matM.matVecMul(&X1[0]);
//   double cerr_m = H2MatrixSolver::solveRelErr(lenX, &X1[0], &X2[0]);

//   //initNcclComms(&nccl_comms, matM.allocedComm);
//   //matM.init_gpu_handles(nccl_comms);

//   MPI_Barrier(MPI_COMM_WORLD);
//   double h2_factor_time = MPI_Wtime(), h2_factor_comm_time;

//   matM.factorizeM();
//   //matM.factorizeDeviceM(handle);

//   MPI_Barrier(MPI_COMM_WORLD);
//   h2_factor_time = MPI_Wtime() - h2_factor_time;
//   h2_factor_comm_time = ColCommMPI::get_comm_time();
//   std::copy(X2.begin(), X2.end(), X1.begin());

//   MPI_Barrier(MPI_COMM_WORLD);
//   double h2_sub_time = MPI_Wtime(), h2_sub_comm_time;

//   matM.solvePrecondition(&X1[0]);
//   //matM.solvePreconditionDevice(handle, &X1[0]);

//   MPI_Barrier(MPI_COMM_WORLD);
//   h2_sub_time = MPI_Wtime() - h2_sub_time;
//   h2_sub_comm_time = ColCommMPI::get_comm_time();
//   double serr = H2MatrixSolver::solveRelErr(lenX, &X1[0], &Xbody[matM.local_bodies.first]);
//   std::fill(X1.begin(), X1.end(), std::complex<double>(0., 0.));

//   if (mpi_rank == 0) {
//     std::cout << "H^2-Preconditioner Construct Time: " << m_construct_time << ", " << m_construct_comm_time << std::endl;
//     std::cout << "H^2-Preconditioner Construct Err: " << cerr_m << std::endl;
//     std::cout << "H^2-Matrix Factorization Time: " << h2_factor_time << ", " << h2_factor_comm_time << std::endl;
//     std::cout << "H^2-Matrix Substitution Time: " << h2_sub_time << ", " << h2_sub_comm_time << std::endl;
//     std::cout << "H^2-Matrix Substitution Err: " << serr << std::endl;
//   }

//   MPI_Barrier(MPI_COMM_WORLD);
//   double gmres_time = MPI_Wtime(), gmres_comm_time;
//   matA.solveGMRES(epi, matM, &X1[0], &X2[0], 10, 50);
//   //matA.solveGMRESDevice(handle, epi, matM, &X1[0], &X2[0], 10, 50, nccl_comms);

//   MPI_Barrier(MPI_COMM_WORLD);
//   gmres_time = MPI_Wtime() - gmres_time;
//   gmres_comm_time = ColCommMPI::get_comm_time();

//   if (mpi_rank == 0) {
//     std::cout << "GMRES Residual: " << matA.resid[matA.iters] << ", Iters: " << matA.iters << std::endl;
//     std::cout << "GMRES Time: " << gmres_time << ", Comm: " << gmres_comm_time << std::endl;
//     for (long long i = 0; i <= matA.iters; i++)
//       std::cout << "iter "<< i << ": " << matA.resid[i] << std::endl;

//     if (csv != nullptr)
//       write_to_csv(csv, mpi_size, Nbody, theta, leaf_size, rank, epi, mode.data(), cerr, 
//         h2_construct_time, h2_construct_comm_time, matvec_time, matvec_comm_time, refmatvec_time, 
//         m_construct_time, m_construct_comm_time, cerr_m, h2_factor_time, h2_factor_comm_time, h2_sub_time, h2_sub_comm_time, serr, 
//         matA.resid[matA.iters], matA.iters, gmres_time, gmres_comm_time, matA.resid.data());
//   }

//   matA.free_all_comms();
//   matM.free_all_comms();
  MPI_Finalize();

//   /*matA.freeSparseMV();
//   matA.free_gpu_handles();
//   matM.free_gpu_handles();
//   finalizeGpuEnvs(handle);
//   finalizeNcclComms(nccl_comms);*/
  return 0;
}

