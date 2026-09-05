#include <complex>
#include <fstream>
#include <random>
#include <string>

#include <Eigen/Dense>

#include <include/elast3d.hpp>
#include <kernel.hpp>
#include <solver.hpp>


int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  /* identifies which source file to read */
  // M corresponds to the number of Nodes (not elements)
  long long M = argc > 1 ? std::atoll(argv[1]) : 5697;
  // the geometry
  long long geom = argc > 2 ? std::atoll(argv[2]) : 1;
  // the angular frequency
  double omega = argc > 3 ? std::atof(argv[3]) : 1;
  // The maximum number of elements contained in each leaf-level node
  long long leaf_size = argc > 4 ? std::atoll(argv[4]) : 32;
  // admisibility of the preconditioner
  double admis_precon = argc > 5 ? std::atof(argv[5]) : 2e0;
  // fixed rank
  long long rank = argc > 6 ? std::atoll(argv[6]) : 32;
  // leveled rank
  long long leveled_rank =  argc > 7 ? std::atoll(argv[7]) : 0;
  // accuracy
  double epsilon = argc > 8 ? std::atof(argv[8]) : 1e-8;
  // admisibility of the H^2 in the matrix-vector products
  double admis = argc > 9 ? std::atof(argv[9]) : 2e0;
  // GMRES parameters
  long long inner_iter = argc > 10 ? std::atoll(argv[10]) : 10;
  long long max_iter = argc > 11 ? std::atoll(argv[11]) : 50;
  // HiDR parameters
  long long r1 = argc > 12 ? std::atoll(argv[12]) : 0;
  long long r2 = argc > 13 ? std::atoll(argv[13]) : 0;
  // write solution
  bool write_result = argc > 14 ? std::atoi(argv[14]) : 0;
  // load H^2 matrix from file
  std::string read_folder = argc > 15 ? argv[15] : "";
  // number of runs
  long long runs = argc > 16 ? std::atoll(argv[16]) : 1;

  const std::string MAT = std::to_string(M);
  MatrixGenerator matgen(M, geom);
  long long Nbody = matgen.get_num_elems();
  leaf_size = Nbody < leaf_size ? Nbody : leaf_size;

  std::vector<Cell> cell;
  long long levels = (long long) std::ceil(std::log2((double)matgen.get_num_elems() / leaf_size));
  long long Nleaf = (long long)1 << levels;
  long long ncells = Nleaf + Nleaf - 1;
  cell.resize(ncells);
  buildBinaryTreeElemsOnly(&cell[0], matgen.get_elems().data(), matgen.get_elems_idx().data(), matgen.get_num_elems(), levels);
  
  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  auto elems = matgen.get_elems();
  auto idx = matgen.get_elems_idx();

  if (write_result && mpi_rank == 0) {
    std::string filename = "results/solution_indices.dat";
    std::ofstream mf(filename);
    
    for (long long i = 0; i < matgen.get_num_elems(); ++i) {
      mf <<idx[i]<<std::endl;
    }
  }

  
  // generate random x
  std::vector<std::complex<double>> Xbody(Nbody * 3);
  std::mt19937_64 gen;
  std::uniform_real_distribution uniform_dist(0., 1.);
  std::generate(Xbody.begin(), Xbody.end(), 
     [&]() { return std::complex<double>(uniform_dist(gen), 0.); });

  if (mpi_rank == 0) {
    std::cout<<"M = "<<M<<", geom = "<<geom<<std::endl;
    std::cout<<"Omega = "<<omega<<", Leaf-size = "<<leaf_size<<", admis_precon = "<<admis_precon<<", rank = "<<rank<<", leveled rank = "<<leveled_rank;
    std::cout<<", epsilon = "<<epsilon<<", admis = "<<admis<<", inner iter = "<<inner_iter<<", max iter = "<<max_iter;
    std::cout<<", r1 = "<<r1<<", r2 = "<<r2<<std::endl;
     std::cout<<"N = "<<Nbody<<", Leaf = "<<leaf_size<<", Levels = "<<levels<<", #Leafs = "<<Nleaf<<", #Cells = "<<ncells<<std::endl;
    std::cout<<"Elements per leaf: "<<(matgen.get_num_elems() >> levels)<<std::endl;
  }
 
  MPI_Barrier(MPI_COMM_WORLD);
  double h2_construct_time = MPI_Wtime(), h2_construct_comm_time;
  H2MatrixSolver<std::complex<double>> matA(matgen, epsilon, rank, leveled_rank, cell, admis, levels, omega, read_folder, true);
  MPI_Barrier(MPI_COMM_WORLD);
  h2_construct_time = MPI_Wtime() - h2_construct_time;
  h2_construct_comm_time = ColCommMPI::get_comm_time();

  // multiply by 3 to get the actual length
  long long lenX = (matA.local_bodies.second - matA.local_bodies.first) * 3;
  long long offset = matA.local_bodies.first * 3;
  std::vector<std::complex<double>> X1(lenX, std::complex<double>(0., 0.));
  std::vector<std::complex<double>> X2(lenX, std::complex<double>(0., 0.));

  // copy random x into X1
  std::copy(&Xbody[offset], &Xbody[offset + lenX], &X1[0]);
  // calculate H-matvec into X1
  MPI_Barrier(MPI_COMM_WORLD);
  double matvec_time = MPI_Wtime(), matvec_comm_time;
  matA.matVecMul(&X1[0]);
  MPI_Barrier(MPI_COMM_WORLD);
  matvec_time = MPI_Wtime() - matvec_time;
  matvec_comm_time = ColCommMPI::get_comm_time();

  // calculate reference into X2
  MPI_Barrier(MPI_COMM_WORLD);
  double refmatvec_time = MPI_Wtime(), refmatvec_comm_time;
  mat_vec_reference(matgen, lenX/3, Nbody, &X2[0], &Xbody[0], matA.local_bodies.first, omega);
  MPI_Barrier(MPI_COMM_WORLD);
  refmatvec_time = MPI_Wtime() - refmatvec_time;
  refmatvec_comm_time = ColCommMPI::get_comm_time();

  double cerr = solveRelErr(lenX, &X1[0], &X2[0]);
  if (mpi_rank == 0) {
    std::cout << "H^2-Matrix Construct Err: " << cerr << std::endl;
    std::cout << "H^2-Matrix Construct Time: " << h2_construct_time << ", " << h2_construct_comm_time << std::endl;
    std::cout << "H^2-Matvec Time: " << matvec_time << ", " << matvec_comm_time << std::endl;
    std::cout << "Dense Matvec Time: " << refmatvec_time << ", " << refmatvec_comm_time << std::endl;
  }
  
  // build preconditioner
  MPI_Barrier(MPI_COMM_WORLD);
  double precon_construct_time = MPI_Wtime(), precon_construct_comm_time;
  H2MatrixSolver<std::complex<double>> precon(matgen, 0, rank, leveled_rank, cell, admis_precon, levels, omega, matgen.get_elems(), r1, r2, true);
  MPI_Barrier(MPI_COMM_WORLD);
  precon_construct_time = MPI_Wtime() - precon_construct_time;
  precon_construct_comm_time = ColCommMPI::get_comm_time();

  // copy random x into X1
  std::copy(&Xbody[offset], &Xbody[offset + lenX], &X1[0]);
  // precon matvec
  MPI_Barrier(MPI_COMM_WORLD);
  double precon_matvec_time = MPI_Wtime(), precon_matvec_comm_time;
  precon.matVecMul(&X1[0]);
  MPI_Barrier(MPI_COMM_WORLD);
  precon_matvec_time = MPI_Wtime() - precon_matvec_time;
  precon_matvec_comm_time = ColCommMPI::get_comm_time();

  cerr = solveRelErr(lenX, &X1[0], &X2[0]);
  MPI_Barrier(MPI_COMM_WORLD);
  if (mpi_rank == 0) {
    std::cout << "H^2-Preconditioner Construct Err: " << cerr << std::endl;
    std::cout << "H^2-Preconditioner Construct Time: " << precon_construct_time << ", " << precon_construct_comm_time << std::endl;
    std::cout << "H^2-Matvec Time: " << precon_matvec_time << ", " << precon_matvec_comm_time << std::endl;
  }

  std::vector<double> fact_time;
  std::vector<double> subst_time;
  std::vector<double> gm_time;

  for (int i = 0; i < runs; ++i) {
    // we would need to initialize X1 and X1_low here
    if (mpi_rank == 0) {
      std::cout<<"Run "<<i<<std::endl;
    }
    H2MatrixSolver<std::complex<double>> precon_tmp(precon);
    std::vector<std::complex<double>> X1_tmp(X1);
 
    // factorize preconditioner
    MPI_Barrier(MPI_COMM_WORLD);
    double precon_factor_time = MPI_Wtime(), precon_factor_comm_time;
    precon_tmp.factorizeM();
    MPI_Barrier(MPI_COMM_WORLD);
    precon_factor_time = MPI_Wtime() - precon_factor_time;
    precon_factor_comm_time = ColCommMPI::get_comm_time();
    fact_time.push_back(precon_factor_time);
  
    MPI_Barrier(MPI_COMM_WORLD);
    double precon_sub_time = MPI_Wtime(), precon_sub_comm_time;
    precon_tmp.solvePrecondition(&X1_tmp[0]);
    MPI_Barrier(MPI_COMM_WORLD);
    precon_sub_time = MPI_Wtime() - precon_sub_time;
    precon_sub_comm_time = ColCommMPI::get_comm_time();
    double serr = solveRelErr(lenX, &X1_tmp[0], &X2[0]);
    subst_time.push_back(precon_sub_time);


    if (!i && mpi_rank == 0) {
      std::cout << "H^2-Matrix Factorization Time: " << precon_factor_time << ", " << precon_factor_comm_time << std::endl;
      std::cout << "H^2-Matrix Substitution Time: " << precon_sub_time << ", " << precon_sub_comm_time << std::endl;
      std::cout << "H^2-Matrix Substitution Err: " << serr << std::endl;
    }

    std::vector<std::complex<double>> rhs(lenX);
    std::vector<double> thetas = {0, 10, 20, 30, 40};
    double gmres_time, gmres_comm_time;
    for (auto& theta: thetas) {
       matgen.gen_rhs_sorted_single_layer(rhs.data(), offset, lenX, omega, theta);
      
      // iterative solver
      std::fill(X1_tmp.begin(), X1_tmp.end(), std::complex<double>(0., 0.));
      MPI_Barrier(MPI_COMM_WORLD);
      gmres_time = MPI_Wtime();
      matA.solveGMRES(epsilon, precon_tmp, &X1_tmp[0], &rhs[0], inner_iter, max_iter);
      MPI_Barrier(MPI_COMM_WORLD);
      gmres_time = MPI_Wtime() - gmres_time;
      gmres_comm_time = ColCommMPI::get_comm_time();
      gm_time.push_back(gmres_time);

      if (mpi_rank == 0) {
        std::cout << "  GMRES Residual: " << matA.resid[matA.iters] << ", Iters: " << matA.iters << std::endl;
        std::cout << "  GMRES Time: " << gmres_time << ", Comm: " << gmres_comm_time << std::endl;
        for (long long i = 0; i <= matA.iters; i++)
          std::cout << "    iter "<< i << ": " << matA.resid[i] << std::endl;
      }
      
      if (write_result) {
        MPI_File fh;
        std::string filename = "results/solution_" + std::to_string(mpi_rank) + ".bin";
        MPI_File_open(MPI_COMM_SELF, filename.c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
        MPI_Offset offset = 0;
        MPI_Status status;
        MPI_File_write_at(fh, offset, X1_tmp.data(), lenX, MPI_C_DOUBLE_COMPLEX, &status);
        MPI_File_close(&fh);
      }
    }
    precon_tmp.free_all_comms();
  }
  if (mpi_rank == 0) {
    std::cout << std::endl << "Factorization time: ";
    for (size_t i = 0; i < fact_time.size(); ++i)
      std::cout << fact_time[i] <<", ";
    std::cout << std::endl;
    std::cout << "Substitution time: ";
    for (size_t i = 0; i < subst_time.size(); ++i)
      std::cout << subst_time[i] <<", ";
    std::cout << std::endl;
    std::cout << "GMRES time: ";
    for (size_t i = 0; i < gm_time.size(); ++i)
      std::cout << gm_time[i]<<", ";
    std::cout << std::endl;
  }

  matA.free_all_comms();
  precon.free_all_comms();
  MPI_Finalize();

  return 0;
}

