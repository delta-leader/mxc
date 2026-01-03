
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
  long long geom = argc > 2 ? std::atoll(argv[2]) : 1;
  double omega = argc > 3 ? std::atof(argv[3]) : 1;
  long long leaf_size = argc > 4 ? std::atoll(argv[4]) : 32;
  double theta = argc > 5 ? std::atof(argv[5]) : 1e0;
  long long rank = argc > 6 ? std::atoll(argv[6]) : 32;
  long long leveled_rank =  argc > 7 ? std::atoll(argv[7]) : 0;
  double epi = argc > 8 ? std::atof(argv[8]) : 1e-10;
  long long inner_iter = argc > 9 ? std::atoll(argv[9]) : 10;
  long long max_iter = argc > 10 ? std::atoll(argv[10]) : 50;
  //std::string mode = argc > 7 ? std::string(argv[7]) : "h2";
  //const char* csv = argc > 8 ? argv[8] : nullptr;

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
  
  // generate random x
  std::vector<std::complex<double>> Xbody(Nbody * 3);
  std::mt19937_64 gen;
  std::uniform_real_distribution uniform_dist(0., 1.);
  std::generate(Xbody.begin(), Xbody.end(), 
     [&]() { return std::complex<double>(uniform_dist(gen), 0.); });

  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  std::string prefix = "../input/cache/";
  std::string filename = std::to_string(geom) + "_" + MAT + "_" + std::to_string((int)omega) + "_" + std::to_string(leaf_size) + ".dat";
  if (mpi_rank == 0) {
    std::cout<<"Omega = "<<omega<<", Leaf-size = "<<leaf_size<<", admis = "<<theta<<", rank = "<<rank<<", leveled rank = "<<leveled_rank;
    std::cout<<", epsilon = "<<epi<<", inner iter = "<<inner_iter<<", max iter = "<<max_iter<<std::endl;
    std::cout<<"Reading file "<<filename<<std::endl;
    std::cout<<"N = "<<Nbody<<", Leaf = "<<leaf_size<<", Levels = "<<levels<<", #Leafs = "<<Nleaf<<", #Cells = "<<ncells<<std::endl;
    std::cout<<"Elements per leaf: "<<(matgen.get_num_elems() >> levels)<<std::endl;
  }
  //matgen.open_matrix_file(prefix + filename);
  //matgen.open_rhs_file(prefix + "rhs_" + filename);
  //double nmat, omega2, lsize;
  //matgen.read_mat_metadata_single_layer(nmat, omega2, lsize);
  //std::cout<<nmat<<", "<<omega2<<", "<<lsize<<std::endl;

  //Eigen::MatrixXcd A_gen(Nbody * 3, Nbody * 3);
  //matgen.gen_matrix_sorted_from_file_single_layer(A_gen.data(), 0, Nbody);
  //Eigen::MatrixXcd U = A_gen.triangularView<Eigen::StrictlyUpper>();
  //Eigen::MatrixXcd L = A_gen.triangularView<Eigen::StrictlyLower>();
  //double error = (U - L.transpose()).norm() / U.norm();
  //std::cout<<"Symmetry error: "<<error<<std::endl;

  //Eigen::JacobiSVD<Eigen::MatrixXcd> svd(A_gen);
  //double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
  //std::cout<<"Condition number: "<<cond<<std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  double m_construct_time = MPI_Wtime(), m_construct_comm_time;
  H2MatrixSolver matM(matgen, 0, rank, leveled_rank, cell, theta, levels, omega);
  MPI_Barrier(MPI_COMM_WORLD);
  m_construct_time = MPI_Wtime() - m_construct_time;
  m_construct_comm_time = ColCommMPI::get_comm_time();

  // multiply by 3 to get the actual length
  long long lenX = (matM.local_bodies.second - matM.local_bodies.first) * 3;
  long long offset = matM.local_bodies.first * 3;
  std::vector<std::complex<double>> X1(lenX, std::complex<double>(0., 0.));
  std::vector<std::complex<double>> X2(lenX, std::complex<double>(0., 0.));

  // copy random x into X1, X2
  std::copy(&Xbody[offset], &Xbody[offset + lenX], &X1[0]);
  std::copy(&Xbody[offset], &Xbody[offset + lenX], &X2[0]);

   // calculate reference into X2
  double refmatvec_time = MPI_Wtime();
  matM.matVecMulDense(&Xbody[0], &X2[0]);
  refmatvec_time = MPI_Wtime() - refmatvec_time;

  std::copy(&Xbody[offset], &Xbody[offset + lenX], &X1[0]);
  matM.matVecMul(&X1[0]);
  MPI_Barrier(MPI_COMM_WORLD);
  double cerr_m = H2MatrixSolver::solveRelErr(lenX, &X1[0], &X2[0]);
  MPI_Barrier(MPI_COMM_WORLD);
  if (mpi_rank == 0) {
    std::cout << "H^2-Preconditioner Construct Err: " << cerr_m << std::endl;
    std::cout << "H^2-Preconditioner Construct Time: " << m_construct_time << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double h2_factor_time = MPI_Wtime(), h2_factor_comm_time;

  matM.factorizeM();
 
  MPI_Barrier(MPI_COMM_WORLD);
  h2_factor_time = MPI_Wtime() - h2_factor_time;
  h2_factor_comm_time = ColCommMPI::get_comm_time();
  std::copy(X2.begin(), X2.end(), X1.begin());

  MPI_Barrier(MPI_COMM_WORLD);
  double h2_sub_time = MPI_Wtime(), h2_sub_comm_time;

  matM.solvePrecondition(&X1[0]);

  MPI_Barrier(MPI_COMM_WORLD);
  h2_sub_time = MPI_Wtime() - h2_sub_time;
  h2_sub_comm_time = ColCommMPI::get_comm_time();
  double serr = H2MatrixSolver::solveRelErr(lenX, &X1[0], &Xbody[offset]);
  std::fill(X1.begin(), X1.end(), std::complex<double>(0., 0.));

  if (mpi_rank == 0) {
    std::cout << "H^2-Matrix Factorization Time: " << h2_factor_time << ", " << h2_factor_comm_time << std::endl;
    std::cout << "H^2-Matrix Substitution Time: " << h2_sub_time << ", " << h2_sub_comm_time << std::endl;
    std::cout << "H^2-Matrix Substitution Err: " << serr << std::endl;
  }

  std::vector<std::complex<double>> rhs(lenX);
  matgen.gen_rhs_sorted_single_layer(rhs.data(), offset, lenX, omega);
  MPI_Barrier(MPI_COMM_WORLD);
  double gmres_time = MPI_Wtime(), gmres_comm_time;
  matM.solveGMRESDense(epi, &X1[0], &rhs[0], inner_iter, max_iter);
  //matM.solveGMRESDensePrecon(epi, lu, A_gen, &X1[0], &rhs[0], 10, 50);
  //matM.solveGMRESDensePrecon(epi, lu, A_gen, &X1[0], &rhs_gen[offset], 10, 50);
  //matM.solveGMRESDense(epi, A_gen, &X1[0], &rhs_gen[0], 10, 50);
  //matA.solveGMRES(epi, matM, &X1[0], &X2[0], 10, 50);
  //matA.solveGMRESDevice(handle, epi, matM, &X1[0], &X2[0], 10, 50, nccl_comms);

  MPI_Barrier(MPI_COMM_WORLD);
  gmres_time = MPI_Wtime() - gmres_time;
  gmres_comm_time = ColCommMPI::get_comm_time();

  if (mpi_rank == 0) {
    std::cout << "GMRES Residual: " << matM.resid[matM.iters] << ", Iters: " << matM.iters << std::endl;
    std::cout << "GMRES Time: " << gmres_time << ", Comm: " << gmres_comm_time << std::endl;
    for (long long i = 0; i <= matM.iters; i++)
      std::cout << "iter "<< i << ": " << matM.resid[i] << std::endl;
  }
  
  matM.free_all_comms();
  MPI_Finalize();

  return 0;
}

