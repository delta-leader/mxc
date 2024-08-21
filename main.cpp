
#include <solver.hpp>
#include <test_funcs.hpp>
#include <string>
#include <float.h>

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  //typedef std::complex<double> DT; typedef std::complex<float> DT_low;
  //typedef std::complex<double> DT; typedef std::complex<Eigen::half> DT_low;
  //typedef double DT; typedef float DT_low;
  typedef double DT; typedef Eigen::half DT_low;

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
  double alpha = 0.05;
  std::vector<double> params = {0.1, 0.5, 1, 2, 5, 10, 50, 100};
  for (size_t i=0; i < params.size(); ++i) {
    for (size_t j=0; j < params.size(); ++j) {
  //Laplace3D<DT> eval(params[i]);
  Yukawa3D<DT> eval(params[i], params[j]);
  //Gaussian<DT> eval(params[i]);
  //IMQ<DT> eval(params[i]);
  //Matern3<DT> eval(params[i]);
  //Helmholtz3D<DT> eval(1., 1.);
  
  // body contains the points
  // 3 corresponds to the dimension
  std::vector<double> body(Nbody * 3);
  // contains the charges for each point?
  //std::vector<std::complex<double>> Xbody(Nbody);
  Vector_dt<DT> Xbody(Nbody);
  // array containing the nodes in the cluster tree
  std::vector<Cell> cell(ncells);

  // create the points (i.e. bodies)
  mesh_sphere(&body[0], Nbody, std::pow(Nbody, 1./2.));
  //uniform_unit_cube_rnd(&body[0], Nbody, std::pow(Nbody, 1./3.), 3, 999);
  //uniform_unit_cube(&body[0], Nbody, std::pow(Nbody, 1./3.), 3);
  //build the tree (i.e. set the values in the cell array)
  buildBinaryTree(levels, Nbody, &body[0], &cell[0]);

  // generate a random vector Xbody (used in Matvec)
  Xbody.generate_random();
  //std::mt19937 gen(999);
  //std::uniform_real_distribution uniform_dist(0., 1.);
  //std::generate(Xbody.begin(), Xbody.end(), 
  //  [&]() { return std::complex<double>(uniform_dist(gen), 0.); });

  /*cell.erase(cell.begin() + 1, cell.begin() + Nleaf - 1);
  cell[0].Child[0] = 1; cell[0].Child[1] = Nleaf + 1;
  ncells = Nleaf + 1;
  levels = 1;*/

  // single process only
  std::vector<DT> A(Nbody * Nbody);
  gen_matrix(eval, Nbody, Nbody, &body[0], &body[0], A.data());
  Eigen::Map<const Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic>> Amap(&A[0], Nbody, Nbody);
  Eigen::JacobiSVD<Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic>> svd(Amap);
  double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
  
  MPI_Barrier(MPI_COMM_WORLD);
  double h2_construct_time = MPI_Wtime(), h2_construct_comm_time;
  // create the H2 matrix
  H2MatrixSolver matA(eval, epi, rank, cell, theta, &body[0], levels);

  // timing of construction
  MPI_Barrier(MPI_COMM_WORLD);
  h2_construct_time = MPI_Wtime() - h2_construct_time;
  h2_construct_comm_time = ColCommMPI::get_comm_time();

  // creates two vectors of zeroes with the same length as the number of local bodies
  long long lenX = matA.local_bodies.second - matA.local_bodies.first;
  //std::vector<std::complex<double>> X1(lenX, std::complex<double>(0., 0.));
  //std::vector<std::complex<double>> X2(lenX, std::complex<double>(0., 0.));
  Vector_dt<DT> X1(lenX);
  Vector_dt<DT> X2(lenX);

  // copy the random vector
  std::copy(&Xbody[matA.local_bodies.first], &Xbody[matA.local_bodies.second], &X1[0]);

  Vector_dt<DT_low> X1_low(X1);
  Vector_dt<DT_low> X1_low2(X1);

  MPI_Barrier(MPI_COMM_WORLD);
  double matvec_time = MPI_Wtime(), matvec_comm_time;
  // Sample matrix vector multiplication
  matA.matVecMul(&X1[0]);

  // MatVec timing
  MPI_Barrier(MPI_COMM_WORLD);
  matvec_time = MPI_Wtime() - matvec_time;
  matvec_comm_time = ColCommMPI::get_comm_time();

  double refmatvec_time = MPI_Wtime();

  // Reference matrix vector multiplication
  mat_vec_reference(eval, lenX, Nbody, &X2[0], &Xbody[0], &body[matA.local_bodies.first * 3], &body[0]);
  refmatvec_time = MPI_Wtime() - refmatvec_time;
  // calculate relative error between H-matvec and dense matvec
  double cerr = computeRelErr(lenX, &X1[0], &X2[0]);

  int mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0) {
    std::cout << params[i] << std::endl;
    std::cout << params[j] << std::endl;
    std::cout << cond << std::endl;
    std::cout << cerr << std::endl;
    //std::cout << "Construct Err (low): " << cerr_low << std::endl;
    //std::cout << "H^2-Matrix Construct Time: " << h2_construct_time << ", " << h2_construct_comm_time << std::endl;
    //std::cout << "H^2-Matvec Time: " << matvec_time << ", " << matvec_comm_time << std::endl;
    //std::cout << "Dense Matvec Time: " << refmatvec_time << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double m_construct_time = MPI_Wtime(), m_construct_comm_time;
  
  // new H2 matrix using a fixed rank
  H2MatrixSolver<DT> h2(eval, epi, rank, cell, theta, &body[0], levels, true, true);
  H2MatrixSolver<DT> hss(eval, epi, rank, cell, 0., &body[0], levels, true);

  MPI_Barrier(MPI_COMM_WORLD);
  m_construct_time = MPI_Wtime() - m_construct_time;
  m_construct_comm_time = ColCommMPI::get_comm_time();

  H2MatrixSolver<DT_low> hss_low(hss);
  H2MatrixSolver<DT_low> h2_low(h2);

  hss_low.matVecMul(&X1_low[0]);
  Vector_dt<DT> result(X1_low);
  double hss_cerr = computeRelErr(lenX, &result[0], &X2[0]);
  
  h2_low.matVecMul(&X1_low2[0]);
  result = Vector_dt<DT>(X1_low2);
  double h2_cerr = computeRelErr(lenX, &result[0], &X2[0]);

  // copy X2 into X1
  Vector_dt<DT_low> X2_low(X2);
  std::copy(X2_low.begin(), X2_low.end(), X1_low.begin());
  std::copy(X2_low.begin(), X2_low.end(), X1_low2.begin());

  h2_low.factorizeM();
  hss_low.factorizeM();
  hss_low.solvePrecondition(&X1_low[0]);
  result = Vector_dt<DT>(X1_low);
  double hss_serr = computeRelErr(lenX, &result[0], &Xbody[matA.local_bodies.first]);
  h2_low.solvePrecondition(&X1_low2[0]);
  result = Vector_dt<DT>(X1_low2);
  double h2_serr = computeRelErr(lenX, &result[0], &Xbody[matA.local_bodies.first]);
  
  if (mpi_rank == 0) {
    std::cout << hss_cerr << std::endl;
    std::cout << hss_serr << std::endl;
    std::cout << h2_cerr << std::endl;
    std::cout << h2_serr << std::endl;
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  double ir_time = MPI_Wtime(), ir_comm_time;
  long long iters = matA.solveIR(epi, hss_low, &X1[0], &X2[0], 100);

  MPI_Barrier(MPI_COMM_WORLD);
  ir_time = MPI_Wtime() - ir_time;
  ir_comm_time = ColCommMPI::get_comm_time();

  if (mpi_rank == 0) {
    std::cout << matA.resid[iters] << std::endl;
    std::cout << iters << std::endl;
    //std::cout << "IR Time: " << ir_time << ", Comm: " << ir_comm_time << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double ir_time_low = MPI_Wtime(), ir_comm_time_low;
  iters = matA.solveIR(epi, h2_low, &X1[0], &X2[0], 100);

  MPI_Barrier(MPI_COMM_WORLD);
  ir_time_low = MPI_Wtime() - ir_time_low;
  ir_comm_time_low = ColCommMPI::get_comm_time();

  if (mpi_rank == 0) {
    std::cout << matA.resid[iters] << std::endl;
    std::cout << iters << std::endl;
    //std::cout << "IR Time(low): " << ir_time_low << ", Comm: " << ir_comm_time_low << std::endl;
  }

  H2MatrixSolver<DT> hss_fact(hss_low);
  MPI_Barrier(MPI_COMM_WORLD);
  double gmres_ir_time = MPI_Wtime(), gmres_ir_comm_time;
  iters = matA.solveGMRESIR(epi, hss_fact, &X1[0], &X2[0], 5, 50, 1);

  MPI_Barrier(MPI_COMM_WORLD);
  gmres_ir_time = MPI_Wtime() - gmres_ir_time;
  gmres_ir_comm_time = ColCommMPI::get_comm_time();

  if (mpi_rank == 0) {
    std::cout << matA.resid[iters] << std::endl;
    std::cout << iters << std::endl;
    //std::cout << "GMRES-IR Time: " << gmres_ir_time << ", Comm: " << gmres_ir_comm_time << std::endl;
  }

  H2MatrixSolver<DT> h2_fact(h2_low);
  MPI_Barrier(MPI_COMM_WORLD);
  double gmres_ir_time_low = MPI_Wtime(), gmres_ir_comm_time_low;
  iters = matA.solveGMRESIR(epi, h2_fact, &X1[0], &X2[0], 5, 50, 1);

  MPI_Barrier(MPI_COMM_WORLD);
  gmres_ir_time_low = MPI_Wtime() - gmres_ir_time_low;
  gmres_ir_comm_time_low = ColCommMPI::get_comm_time();

  if (mpi_rank == 0) {
    std::cout << matA.resid[iters] << std::endl;
    std::cout << iters << std::endl << std::endl;
    //std::cout << "GMRES-IR Time: " << gmres_ir_time_low << ", Comm: " << gmres_ir_comm_time_low << std::endl;
  }
  
  matA.free_all_comms();
  hss.free_all_comms();
  hss_low.free_all_comms();
  hss_fact.free_all_comms();
  h2.free_all_comms();
  h2_low.free_all_comms();
  h2_fact.free_all_comms();
  }}
  MPI_Finalize();
  return 0;
}

