
#include <solver.hpp>
#include <test_funcs.hpp>
#include <string>

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

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
  //Laplace3D eval(1.);
  //Yukawa3D eval(1, 1.);
  //Gaussian eval(8);
  Helmholtz3D eval(1., 1.);
  
  // body contains the points
  // 3 corresponds to the dimension
  std::vector<double> body(Nbody * 3);
  // contains the charges for each point?
  std::vector<std::complex<double>> Xbody(Nbody);
  // array containing the nodes in the cluster tree
  std::vector<Cell> cell(ncells);

  // create the points (i.e. bodies)
  //mesh_sphere(&body[0], Nbody, std::pow(Nbody, 1./2.));
  uniform_unit_cube_rnd(&body[0], Nbody, std::pow(Nbody, 1./3.), 3, 999);
  //uniform_unit_cube(&body[0], Nbody, std::pow(Nbody, 1./3.), 3);
  //build the tree (i.e. set the values in the cell array)
  buildBinaryTree(levels, Nbody, &body[0], &cell[0]);

  // generate the charges
  std::mt19937 gen(999);
  std::uniform_real_distribution uniform_dist(0., 1.);
  std::generate(Xbody.begin(), Xbody.end(), 
    [&]() { return std::complex<double>(uniform_dist(gen), 0.); });

  /*cell.erase(cell.begin() + 1, cell.begin() + Nleaf - 1);
  cell[0].Child[0] = 1; cell[0].Child[1] = Nleaf + 1;
  ncells = Nleaf + 1;
  levels = 1;*/

  MPI_Barrier(MPI_COMM_WORLD);
  double h2_construct_time = MPI_Wtime(), h2_construct_comm_time;
  // create the H2 matrix
  H2MatrixSolver matA(eval, epi, rank, cell, theta, &body[0], levels);

  // timing of construction
  MPI_Barrier(MPI_COMM_WORLD);
  h2_construct_time = MPI_Wtime() - h2_construct_time;
  h2_construct_comm_time = ColCommMPI::get_comm_time();

  // creates two vectors that of length as the number of local bodies
  // initialized with (0, 0) pairs
  long long lenX = matA.local_bodies.second - matA.local_bodies.first;
  std::vector<std::complex<double>> X1(lenX, std::complex<double>(0., 0.));
  std::vector<std::complex<double>> X2(lenX, std::complex<double>(0., 0.));

  // copy the local charges?
  std::copy(&Xbody[matA.local_bodies.first], &Xbody[matA.local_bodies.second], &X1[0]);

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
  // solve
  double cerr = H2MatrixSolver::solveRelErr(lenX, &X1[0], &X2[0]);

  int mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0) {
    std::cout << "Construct Err: " << cerr << std::endl;
    std::cout << "H^2-Matrix Construct Time: " << h2_construct_time << ", " << h2_construct_comm_time << std::endl;
    std::cout << "H^2-Matvec Time: " << matvec_time << ", " << matvec_comm_time << std::endl;
    std::cout << "Dense Matvec Time: " << refmatvec_time << std::endl;
  }

  std::copy(X2.begin(), X2.end(), X1.begin());
  MPI_Barrier(MPI_COMM_WORLD);
  double m_construct_time = MPI_Wtime(), m_construct_comm_time;
  H2MatrixSolver matM;
  if (mode.compare("h2") == 0)
    matM = H2MatrixSolver(eval, epi, rank, cell, theta, &body[0], levels, true);
  else if (mode.compare("hss") == 0)
    matM = H2MatrixSolver(eval, epi, rank, cell, 0., &body[0], levels, true);

  MPI_Barrier(MPI_COMM_WORLD);
  m_construct_time = MPI_Wtime() - m_construct_time;
  m_construct_comm_time = ColCommMPI::get_comm_time();

  MPI_Barrier(MPI_COMM_WORLD);
  double h2_factor_time = MPI_Wtime(), h2_factor_comm_time;

  matM.factorizeM();

  MPI_Barrier(MPI_COMM_WORLD);
  h2_factor_time = MPI_Wtime() - h2_factor_time;
  h2_factor_comm_time = ColCommMPI::get_comm_time();

  MPI_Barrier(MPI_COMM_WORLD);
  double h2_sub_time = MPI_Wtime(), h2_sub_comm_time;

  matM.solvePrecondition(&X1[0]);

  MPI_Barrier(MPI_COMM_WORLD);
  h2_sub_time = MPI_Wtime() - h2_sub_time;
  h2_sub_comm_time = ColCommMPI::get_comm_time();
  double serr = H2MatrixSolver::solveRelErr(lenX, &X1[0], &Xbody[matA.local_bodies.first]);
  std::fill(X1.begin(), X1.end(), std::complex<double>(0., 0.));

  if (mpi_rank == 0) {
    std::cout << "H^2-Preconditioner Construct Time: " << m_construct_time << ", " << m_construct_comm_time << std::endl;
    std::cout << "H^2-Matrix Factorization Time: " << h2_factor_time << ", " << h2_factor_comm_time << std::endl;
    std::cout << "H^2-Matrix Substitution Time: " << h2_sub_time << ", " << h2_sub_comm_time << std::endl;
    std::cout << "H^2-Matrix Substitution Err: " << serr << std::endl;
  }

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

  matA.free_all_comms();
  matM.free_all_comms();
  MPI_Finalize();
  return 0;
}

