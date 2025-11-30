
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
  double theta = argc > 2 ? std::atof(argv[2]) : 1e0;
  long long leaf_size = argc > 3 ? std::atoll(argv[3]) : 32;
  long long rank = argc > 4 ? std::atoll(argv[4]) : 32;
  long long leveled_rank =  argc > 5 ? std::atoll(argv[5]) : 0;
  double epi = argc > 6 ? std::atof(argv[6]) : 1e-10;
  std::string tree_mode = argc > 7 ? std::string(argv[7]) : "default";
  double omega = argc > 8 ? std::atof(argv[8]) : 1;
  //std::string mode = argc > 7 ? std::string(argv[7]) : "h2";
  //const char* csv = argc > 8 ? argv[8] : nullptr;

  const std::string MAT = std::to_string(M);
 
  MatrixGenerator matgen(M, 1);
  long long Nbody = matgen.get_num_nodes() + matgen.get_num_elems();
  std::cout<<"Nodes/Elements: "<<matgen.get_num_nodes()<<" "<<matgen.get_num_elems()<<std::endl;
  leaf_size = Nbody < leaf_size ? Nbody : leaf_size;

  long long levels, Nleaf, ncells;
  std::vector<Cell> cell;
  if (tree_mode == "standard") {
    std::cout<<"Tree mode '" + tree_mode +"' no longer supported"<<std::endl;
  } else {
    long long levels_elems = (long long) std::ceil(std::log2((double)matgen.get_num_elems() / leaf_size));
    long long Nleaf_elems = (long long)1 << levels_elems;
    long long ncells_elems = Nleaf_elems + Nleaf_elems - 1;
    long long levels_nodes = (long long) std::ceil(std::log2((double)matgen.get_num_nodes() / leaf_size));
    long long Nleaf_nodes = (long long)1 << levels_nodes;
    long long ncells_nodes = Nleaf_nodes + Nleaf_nodes - 1;
    if (tree_mode == "fused1") {
      std::cout<<"Tree mode '" + tree_mode +"' no longer supported"<<std::endl;
    } else {
      if (tree_mode == "fused2") {
        levels = levels_elems;
        Nleaf = Nleaf_nodes + Nleaf_elems;
        ncells = ncells_elems + ncells_nodes + 1;
        cell.resize(ncells);
        buildBinaryTreeNodes(&cell[0], matgen.get_nodes().data(), matgen.get_nodes_idx().data(), matgen.get_num_nodes(), levels_nodes, 1);
        buildBinaryTreeElems(&cell[0], matgen.get_elems().data(), matgen.get_elems_idx().data(), matgen.get_num_elems(), levels_elems, 0, matgen.get_num_nodes());
        /* root has three children */
        cell[0].Child[0] = 1;
        cell[0].Child[1] = 4;
        cell[0].Body[0] = 0;
        cell[0].Body[1] = matgen.get_num_nodes() + matgen.get_num_elems();
        for (int d = 0; d < 3; ++d) {
          cell[0].R[d] = cell[1].R[d];
          cell[0].C[d] = (cell[0].C[d] + cell[1].C[d]) / 2;
        }
      } else {
        std::cout<<"Invalid tree mode '" + tree_mode +"'"<<std::endl;
        return -1;
      }
    }
  }

  
  std::cout<<"N = "<<Nbody<<", Leaf = "<<leaf_size<<", Levels = "<<levels<<", #Leafs = "<<Nleaf<<", #Cells = "<<ncells<<std::endl;

  // read the rhs, reference solution and matrix from the file
  long long n_mat = Nbody * 3;

  Eigen::MatrixXcd A_gen(n_mat, n_mat);
  Eigen::VectorXcd rhs_gen(n_mat);
  double get_scale_time = MPI_Wtime();
  double scale = matgen.calc_scale(omega);
  get_scale_time = MPI_Wtime() - get_scale_time;
  double gen_matrix_time = MPI_Wtime();
  matgen.gen_matrix_sorted(A_gen.data(), omega, scale, true);
  gen_matrix_time = MPI_Wtime() - gen_matrix_time;
  double gen_rhs_time = MPI_Wtime();
  matgen.gen_rhs_sorted(rhs_gen.data(), omega, scale);
  gen_rhs_time = MPI_Wtime() - gen_rhs_time;

  std::cout<<"MATRIX ASSEMBLY"<<std::endl;
  std::cout<<"Calc scale time "<<get_scale_time<<std::endl;
  std::cout<<"Gen matrix time "<<gen_matrix_time<<std::endl;
  std::cout<<"Gen rhs time "<<gen_rhs_time<<std::endl;

 
  // generate random x
  std::vector<std::complex<double>> Xbody(Nbody * 3);
  std::mt19937_64 gen;
  std::uniform_real_distribution uniform_dist(0., 1.);
  std::generate(Xbody.begin(), Xbody.end(), 
     [&]() { return std::complex<double>(uniform_dist(gen), 0.); });



  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);


  matgen.open_matrix_file("../input/cache/1_160_1_32.dat");
  matgen.open_rhs_file("../input/cache/rhs_1_160_1_32.dat");
  double nmat, scale2, omega2, lsize;
  matgen.read_mat_metadata(nmat, scale2, omega2, lsize);
  std::cout<<nmat<<", "<<scale<<", "<<omega2<<", "<<lsize<<std::endl;
  MPI_Barrier(MPI_COMM_WORLD);
  double m_construct_time = MPI_Wtime(), m_construct_comm_time;
  H2MatrixSolver matM(matgen, 0, rank, leveled_rank, cell, theta, levels, omega, scale);
  MPI_Barrier(MPI_COMM_WORLD);
  m_construct_time = MPI_Wtime() - m_construct_time;
  m_construct_comm_time = ColCommMPI::get_comm_time();
  std::cout<<"Construction Finished"<<std::endl;

  // multiply by 3 to get the actual length
  // this way, we can reduce the number of elems if necessary
  //long long lenX = Nbody * 3;
  long long lenX = (matM.local_bodies.second - matM.local_bodies.first) * 3;
  long long offset = matM.local_bodies.first * 3;
  // make the vectors full size and pass them on in a strided fashion
  std::vector<std::complex<double>> X1(lenX, std::complex<double>(0., 0.));
  std::vector<std::complex<double>> X2(lenX, std::complex<double>(0., 0.));
  //std::vector<std::complex<double>> X3(lenX, std::complex<double>(0., 0.));
  std::cout<<"offset: "<<offset<<" "<<lenX<<std::endl;

  // copy random x into X1, X2
  std::copy(&Xbody[offset], &Xbody[offset + lenX], &X1[0]);
  std::copy(&Xbody[offset], &Xbody[offset + lenX], &X2[0]);
  //std::copy(&Xbody[matM.local_bodies.first * 3], &Xbody[matM.local_bodies.second * 3], &X1[0]);
  //std::copy(&Xbody[matM.local_bodies.first * 3], &Xbody[matM.local_bodies.second * 3], &X2[0]);

   // calculate reference into X2
  double refmatvec_time = MPI_Wtime();
  matM.matVecMulDense(&Xbody[0], &X2[0]);
  refmatvec_time = MPI_Wtime() - refmatvec_time;
  std::cout<<"Ref Matvec finished"<<std::endl;

  std::copy(&Xbody[offset], &Xbody[offset + lenX], &X1[0]);
  matM.matVecMul(&X1[0]);
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout<<"Matvec finished"<<std::endl;
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
    //std::cout << "H^2-Preconditioner Construct Time: " << m_construct_time << ", " << m_construct_comm_time << std::endl;
    //std::cout << "H^2-Preconditioner Construct Err: " << cerr_m << std::endl;
    std::cout << "H^2-Matrix Factorization Time: " << h2_factor_time << ", " << h2_factor_comm_time << std::endl;
    std::cout << "H^2-Matrix Substitution Time: " << h2_sub_time << ", " << h2_sub_comm_time << std::endl;
    std::cout << "H^2-Matrix Substitution Err: " << serr << std::endl;
  }

  std::vector<std::complex<double>> rhs(lenX);
  matgen.gen_rhs_sorted_from_file(rhs.data(), offset, lenX);
  MPI_Barrier(MPI_COMM_WORLD);
  double gmres_time = MPI_Wtime(), gmres_comm_time;
  matM.solveGMRESDense(epi, &X1[0], &rhs[0], 10, 50);
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

