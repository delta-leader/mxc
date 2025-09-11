
#include <solver.hpp>
#include <test_funcs.hpp>
#include <include/elast3d.hpp>
#include <kernel.hpp>
#include <string>

#include <Eigen/Dense>

#include <fstream>

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  /*deviceHandle_t handle;
  ncclComms nccl_comms = nullptr;
  cudaSetDevice();
  initGpuEnvs(&handle);*/

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
  //long long n_nodes, n_elems;
  //std::vector<double> nodes;
  //std::vector<double> elems;
  //std::vector<double> elems_polar;
  // Reading the mes data (i.e. nodes and elems)
  // For the elements we calculate the centroid and store it in elems
  //read_mesh_data(n_nodes, nodes, n_elems, elems, "../input/mesh_sphere_" + MAT + "nodes.inp");
  //long long num_nodes, num_elems;
  //read_mesh_specs(num_nodes, num_elems, "../input/mesh_sphere_" + MAT + "nodes.inp");
  //long long Nbody = num_nodes + num_elems;
  // leaf size is expressed in terms of #elems, since we don't want to split an elment
  
  
  //std::vector<struct elastWave3d::nodal_point> nodes(num_nodes);
  //std::vector<struct elastWave3d::element> elems(num_elems);
  //read_mesh_fortran(num_nodes, nodes, num_elems, elems, stoi(MAT));
  MatrixGenerator matgen(M);
  long long Nbody = matgen.get_num_nodes() + matgen.get_num_elems();
  std::cout<<"Nodes/Elements: "<<matgen.get_num_nodes()<<" "<<matgen.get_num_elems()<<std::endl;
  leaf_size = Nbody < leaf_size ? Nbody : leaf_size;
  // Mesh check
  //for (int i = 0; i < 5; ++i)
  //  std::cout<<nodes2[i].xc[0]<<", "<<nodes2[i].xc[1]<<", "<<nodes2[i].xc[2]<<std::endl;
  //std::cout<<std::endl;
  //for (int i = 0; i < 5; ++i){
  //  for (int d=0 ; d < 3; ++d)
  //    std::cout<<nodes[i*3+d]<<", ";
  //  std::cout<<std::endl;
  //}

  // check that the sizes match
  //std::cout<<nodes.size()/3<<" " <<elems.size()/3<<std::endl;

  // create index array nodes + elements
  //std::vector<long long> idx(n_nodes + n_elems);
  //std::iota(idx.begin(), idx.end(), 0);
  //std::vector<double> all(nodes);
  //all.insert(all.end(), elems.begin(), elems.end());
  //std::vector<long long> nodes_indices(num_nodes);
  //std::iota(nodes_indices.begin(), nodes_indices.end(), 0);
  //std::vector<long long> elems_indices(num_elems);
  //std::iota(elems_indices.begin(), elems_indices.end(), 0);

  long long levels, Nleaf, ncells;
  std::vector<Cell> cell;
  if (tree_mode == "standard") {
    std::cout<<"Tree mode '" + tree_mode +"' no longer supported"<<std::endl;
    /*levels = (long long) std::ceil(std::log2((double)Nbody / leaf_size));
    Nleaf = (long long)1 << levels;
    ncells = Nleaf + Nleaf - 1;
    cell.resize(ncells);
    // build the tree for the whole matrix
    buildBinaryTree(&cell[0], &all[0], idx.data(), Nbody, levels);*/
  } else {
    long long levels_elems = (long long) std::ceil(std::log2((double)matgen.get_num_elems() / leaf_size));
    long long Nleaf_elems = (long long)1 << levels_elems;
    long long ncells_elems = Nleaf_elems + Nleaf_elems - 1;
    long long levels_nodes = (long long) std::ceil(std::log2((double)matgen.get_num_nodes() / leaf_size));
    long long Nleaf_nodes = (long long)1 << levels_nodes;
    long long ncells_nodes = Nleaf_nodes + Nleaf_nodes - 1;
    if (tree_mode == "fused1") {
      std::cout<<"Tree mode '" + tree_mode +"' no longer supported"<<std::endl;
      /*levels = levels_elems + 1;
      Nleaf = Nleaf_nodes + Nleaf_elems;
      ncells = ncells_elems + ncells_nodes + 2;
      cell.resize(ncells);
      // build the tree for the nodes
      buildBinaryTree(&cell[0], &nodes[0], idx.data(), n_nodes, levels_nodes, 3, 0);
      // build the tree for the elements
      buildBinaryTree(&cell[0], &elems[0], &idx[n_nodes], n_elems, levels_elems, 2, n_nodes);
     //set the root
     cell[0].Child[0] = 1;
     cell[0].Child[1] = 3;
     cell[0].Body[0] = 0;
     cell[0].Body[1] = n_nodes + n_elems;
     // duplicate the root node from the nodes tree
     cell[1] = cell[3];
     cell[1].Child[0] = 3;
     cell[1].Child[1] = 4;*/
    } else {
      if (tree_mode == "fused2") {
        levels = levels_elems;
        Nleaf = Nleaf_nodes + Nleaf_elems;
        ncells = ncells_elems + ncells_nodes + 1;
        cell.resize(ncells);
        //buildBinaryTree3(&cell[0], &nodes[0], idx.data(), n_nodes, levels_nodes, 1, 0);
        //buildBinaryTree3(&cell[0], &elems[0], &idx[n_nodes], n_elems, levels_elems, 0, n_nodes);
        buildBinaryTreeNodes(&cell[0], matgen.get_nodes().data(), matgen.get_nodes_idx().data(), matgen.get_num_nodes(), levels_nodes, 1);
        buildBinaryTreeElems(&cell[0], matgen.get_elems().data(), matgen.get_elems_idx().data(), matgen.get_num_elems(), levels_elems, 0, matgen.get_num_nodes());
        //buildBinaryTree3(&cell[0], &nodes[0], nodes_indices.data(), n_nodes, levels_nodes, 1, 0);
        //buildBinaryTree3(&cell[0], &elems[0], elems_indices.data(), n_elems, levels_elems, 0, n_nodes);
        /* root has three children */
        cell[0].Child[0] = 1;
        cell[0].Child[1] = 4;
        cell[0].Body[0] = 0;
        cell[0].Body[1] = matgen.get_num_nodes() + matgen.get_num_elems();
        /*std::cout<<"Root"<<std::endl;
        std::cout<<"R: "<<cell[0].R[0]<<", "<<cell[0].R[1]<<", "<<cell[0].R[2]<<std::endl;
        std::cout<<"C: "<<cell[0].C[0]<<", "<<cell[0].C[1]<<", "<<cell[0].C[2]<<std::endl;
        std::cout<<"Node 1"<<std::endl;
        std::cout<<"R: "<<cell[1].R[0]<<", "<<cell[1].R[1]<<", "<<cell[1].R[2]<<std::endl;
        std::cout<<"C: "<<cell[1].C[0]<<", "<<cell[1].C[1]<<", "<<cell[1].C[2]<<std::endl;
        std::cout<<"Node 2"<<std::endl;
        std::cout<<"R: "<<cell[2].R[0]<<", "<<cell[2].R[1]<<", "<<cell[2].R[2]<<std::endl;
        std::cout<<"C: "<<cell[2].C[0]<<", "<<cell[2].C[1]<<", "<<cell[2].C[2]<<std::endl;
        std::cout<<"Node 3"<<std::endl;
        std::cout<<"R: "<<cell[3].R[0]<<", "<<cell[3].R[1]<<", "<<cell[3].R[2]<<std::endl;
        std::cout<<"C: "<<cell[3].C[0]<<", "<<cell[3].C[1]<<", "<<cell[3].C[2]<<std::endl;*/
        for (int d = 0; d < 3; ++d) {
          cell[0].R[d] = cell[1].R[d];
          cell[0].C[d] = (cell[0].C[d] + cell[1].C[d]) / 2;
        }
        //std::cout<<"Root"<<std::endl;
        //std::cout<<"R: "<<cell[0].R[0]<<", "<<cell[0].R[1]<<", "<<cell[0].R[2]<<std::endl;
        //std::cout<<"C: "<<cell[0].C[0]<<", "<<cell[0].C[1]<<", "<<cell[0].C[2]<<std::endl;
      } else {
        std::cout<<"Invalid tree mode '" + tree_mode +"'"<<std::endl;
        return -1;
      }
    }
  }

  
  std::cout<<"N = "<<Nbody<<", Leaf = "<<leaf_size<<", Levels = "<<levels<<", #Leafs = "<<Nleaf<<", #Cells = "<<ncells<<std::endl;

  // read the rhs, reference solution and matrix from the file
  long long n_mat = Nbody * 3;
  //std::vector<std::complex<double>> mat(n_mat * n_mat);
  //read_data2(mat.data(), "../input/checkMatrix.dat", n_mat * n_mat);
  //Eigen::Map<Eigen::MatrixXcd> A(mat.data(), n_mat,  n_mat);
  //std::vector<std::complex<double>> b(n_mat);
  //read_data2(b.data(), "../input/checkRHS.dat", n_mat);
  //Eigen::Map<Eigen::VectorXcd> rhs(b.data(), n_mat);
 
  //MatrixGenerator matgen(omega);
  Eigen::MatrixXcd A_gen(n_mat, n_mat);
  Eigen::VectorXcd rhs_gen(n_mat);
  double get_scale_time = MPI_Wtime();
  double scale = matgen.calc_scale(omega);
  get_scale_time = MPI_Wtime() - get_scale_time;
  double gen_matrix_time = MPI_Wtime();
  matgen.gen_matrix_sorted(A_gen.data(), omega, scale);
  //matgen.gen_matrix(A_gen.data(), omega, 1);
  //for (int i = 0; i < n_mat; ++i)
  //  std::cout<<rhs_gen(i)<<std::endl;
  gen_matrix_time = MPI_Wtime() - gen_matrix_time;
  double gen_rhs_time = MPI_Wtime();
  //matgen.gen_rhs_sorted(rhs_gen.data(), omega, scale);
  //matgen.gen_rhs(rhs_gen.data(), omega, scale);
  //for (int i = 0; i < n_mat; ++i)
  //  std::cout<<rhs_gen(i)<<std::endl;
  gen_rhs_time = MPI_Wtime() - gen_rhs_time;
  //std::cout<<"MAX "<<matgen.get_max_nodes(nodes2.data(), num_nodes, elems2.data(), num_elems)<<std::endl;
  //std::cout<<"MAX Elems "<<matgen.get_max_elems(nodes2.data(), num_nodes, elems2.data(), num_elems)<<std::endl;
  //std::cout<<"Scale "<<scale<<std::endl;
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

  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // if (mpi_rank == 0) {
  //   std::cout << "Construct Err (without factorization basis): " << cerr << std::endl;
  //   //std::cout << "H^2-Matrix Construct Time: " << h2_construct_time << ", " << h2_construct_comm_time << std::endl;
  //   //std::cout << "H^2-Matvec Time: " << matvec_time << ", " << matvec_comm_time << std::endl;
  //   //std::cout << "Dense Matvec Time: " << refmatvec_time << std::endl;
  //   /*Eigen::MatrixXcd A(Nbody, Nbody);
  //   gen_matrix(eval, Nbody, Nbody, body.data(), body.data(), A.data());
  //   double cond = 1. / A.lu().rcond();
  //   std::cout << "Condition #: " << cond << std::endl;*/
  // }

  //Eigen::MatrixXcd RX = U_sorted.topLeftCorner(237, 237);
  //Eigen::MatrixXcd RX = U_sorted.topLeftCorner(10, 10);
  //Eigen::PartialPivLU<Eigen::MatrixXcd> fac(U);
  //Eigen::FullPivLU<Eigen::MatrixXcd> fac(RX);
  //fac.setThreshold(1e-6);
  //std::cout<<"is invertible "<<fac.isInvertible()<<" "<<fac.rank()<<std::endl;
  //std::cout<<"SINGULAR "<<(std::abs(fac.determinant()) <= std::numeric_limits<double>::min()) <<" "<<std::abs(fac.determinant())<<" "<<std::numeric_limits<double>::min()<<std::endl;
  
  //std::vector<double> all_sorted(Nbody * 3);
  // We construct the matrix from a dense, so we need the sorted nodes elements
  //for (int i = 0; i < Nbody; ++i) {
  //  for (int ii = 0; ii < 3; ++ii) {
      // shuffle the all vector so that the order of points matches the sorted matrix
   //   all_sorted[i * 3 + ii] = all[all_indices[i] * 3 + ii];
   // }
  //}

  MPI_Barrier(MPI_COMM_WORLD);
  double m_construct_time = MPI_Wtime(), m_construct_comm_time;
  H2MatrixSolver matM(matgen, 0, rank, leveled_rank, cell, theta, levels, omega, scale);
  //H2MatrixSolver matM(A_gen, 0, rank, leveled_rank, cell, theta, levels);
  //H2MatrixSolver matM(A_sorted, 0, rank, leveled_rank, cell, theta, levels);
  //H2MatrixSolver matM(A_sorted, epi, rank, leveled_rank, cell, theta, levels, all_sorted);
  //std::cout<<"Construction finished"<<std::endl;
  MPI_Barrier(MPI_COMM_WORLD);
  m_construct_time = MPI_Wtime() - m_construct_time;
  m_construct_comm_time = ColCommMPI::get_comm_time();

  // multiply by 3 to get the actual length
  // this way, we can reduce the number of elems if necessary
  long long lenX = Nbody * 3;
  long long lenX_local = (matM.local_bodies.second - matM.local_bodies.first) * 3;
  long long offset_local = matM.local_bodies.first * 3;
  // make the vectors full size and pass them on in a strided fashion
  std::vector<std::complex<double>> X1(lenX, std::complex<double>(0., 0.));
  std::vector<std::complex<double>> X2(lenX, std::complex<double>(0., 0.));
  std::vector<std::complex<double>> X3(lenX, std::complex<double>(0., 0.));

  // copy random x into X1, X2
  std::copy(&Xbody[0], &Xbody[lenX], &X1[0]);
  std::copy(&Xbody[0], &Xbody[lenX], &X2[0]);
  //std::copy(&Xbody[matM.local_bodies.first * 3], &Xbody[matM.local_bodies.second * 3], &X1[0]);
  //std::copy(&Xbody[matM.local_bodies.first * 3], &Xbody[matM.local_bodies.second * 3], &X2[0]);

   // calculate reference into X2
  double refmatvec_time = MPI_Wtime();
  Eigen::Map<Eigen::VectorXcd> ref(&X2[0], lenX);
  ref = A_gen * ref;
  //for (int i = 0; i < lenX; ++i)
  //  std::cout<<t(i)<<std::endl;
  //std::cout<<"Ref finished"<<std::endl;

  refmatvec_time = MPI_Wtime() - refmatvec_time;
  // double cerr = H2MatrixSolver::solveRelErr(lenX, &X1[0], &X2[0]);
  // //double cerr = H2MatrixSolver::solveRelErr(lenX, r.data(), result.data());

  //std::copy(&Xbody[matM.local_bodies.first * 3], &Xbody[matM.local_bodies.second * 3], &X1[0]);
  std::copy(&Xbody[0], &Xbody[lenX], &X1[0]);
  matM.matVecMulDense(&X1[0], &X3[offset_local]);
  //matM.matVecMul(&X1[0]);
  MPI_Barrier(MPI_COMM_WORLD);
  double cerr_m = H2MatrixSolver::solveRelErr(lenX_local, &X3[offset_local], &X2[offset_local]);
  MPI_Barrier(MPI_COMM_WORLD);
  if (mpi_rank == 0) {
    std::cout << "H^2-Preconditioner Construct Err: " << cerr_m << std::endl;
    std::cout << "H^2-Preconditioner Construct Time: " << m_construct_time << std::endl;
  }

  //initNcclComms(&nccl_comms, matM.allocedComm);
  //matM.init_gpu_handles(nccl_comms);

  /*MPI_Barrier(MPI_COMM_WORLD);
  double h2_factor_time = MPI_Wtime(), h2_factor_comm_time;

  matM.factorizeM();
  //matM.factorizeDeviceM(handle);

  MPI_Barrier(MPI_COMM_WORLD);
  h2_factor_time = MPI_Wtime() - h2_factor_time;
  h2_factor_comm_time = ColCommMPI::get_comm_time();
  std::copy(X2.begin(), X2.end(), X1.begin());

  MPI_Barrier(MPI_COMM_WORLD);
  double h2_sub_time = MPI_Wtime(), h2_sub_comm_time;

  matM.solvePrecondition(&X1[0]);
  //matM.solvePreconditionDevice(handle, &X1[0]);

  MPI_Barrier(MPI_COMM_WORLD);
  h2_sub_time = MPI_Wtime() - h2_sub_time;
  h2_sub_comm_time = ColCommMPI::get_comm_time();
  double serr = H2MatrixSolver::solveRelErr(lenX, &X1[0], &Xbody[matM.local_bodies.first]);
  std::fill(X1.begin(), X1.end(), std::complex<double>(0., 0.));

  if (mpi_rank == 0) {
    //std::cout << "H^2-Preconditioner Construct Time: " << m_construct_time << ", " << m_construct_comm_time << std::endl;
    //std::cout << "H^2-Preconditioner Construct Err: " << cerr_m << std::endl;
    std::cout << "H^2-Matrix Factorization Time: " << h2_factor_time << ", " << h2_factor_comm_time << std::endl;
    std::cout << "H^2-Matrix Substitution Time: " << h2_sub_time << ", " << h2_sub_comm_time << std::endl;
    std::cout << "H^2-Matrix Substitution Err: " << serr << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double gmres_time = MPI_Wtime(), gmres_comm_time;
  matM.solveGMRESDense(epi, A_gen, &X1[0], &rhs_gen[0], 10, 50);
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
  */
  //   /*if (csv != nullptr)
  //     write_to_csv(csv, mpi_size, Nbody, theta, leaf_size, rank, epi, mode.data(), cerr, 
  //       h2_construct_time, h2_construct_comm_time, matvec_time, matvec_comm_time, refmatvec_time, 
  //       m_construct_time, m_construct_comm_time, cerr_m, h2_factor_time, h2_factor_comm_time, h2_sub_time, h2_sub_comm_time, serr, 
  //       matA.resid[matA.iters], matA.iters, gmres_time, gmres_comm_time, matA.resid.data());*/
  //}

  // //GMRES without preconditioning
  // std::fill(X1.begin(), X1.end(), std::complex<double>(0., 0.));
  // matM.solveGMRESDenseNoPrecon(1e-13, A_sorted, &X1[0], &X2[0], 10, 50);
  // if (mpi_rank == 0) {
  //   std::cout << "GMRES (no preconditioner) Residual: " << matM.resid[matM.iters] << ", Iters: " << matM.iters << std::endl;
  //   for (long long i = 0; i <= matM.iters; i++)
  //     std::cout << "iter "<< i << ": " << matM.resid[i] << std::endl;
  //}

  // //GMRES with dense preconditioning
  // Eigen::PartialPivLU<Eigen::MatrixXcd> precon(A_sorted);
  // std::fill(X1.begin(), X1.end(), std::complex<double>(0., 0.));
  // matM.solveGMRESDensePrecon(1e-13, precon, A_sorted, &X1[0], &X2[0], 10, 50);
  // if (mpi_rank == 0) {
  //   std::cout << "GMRES (dense preconditioner) Residual: " << matM.resid[matM.iters] << ", Iters: " << matM.iters << std::endl;
  //   for (long long i = 0; i <= matM.iters; i++)
  //     std::cout << "iter "<< i << ": " << matM.resid[i] << std::endl;
  // }

  // Eigen::MatrixXcd RX = A_sorted.triangularView<Eigen::Lower>();
  // Eigen::MatrixXcd RX2 = A_sorted.triangularView<Eigen::StrictlyLower>().transpose();
  // Eigen::MatrixXcd RX3 = RX + RX2;
  // Eigen::Map<Eigen::VectorXcd> t2(&X1[0], lenX);

  // /*for (long long j = 0; j < Nleaf; ++j) {
  //   long long offset = cell[Nleaf - 1 + j].Body[0];
  //   long long num = cell[Nleaf - 1 + j].Body[1] - offset;
  //   RX3.block(offset, offset, num, num) = A_sorted.block(offset, offset, num, num);
  // }*/
  // t2 = RX3 * t2;
  // std::cout<<"Symm Error:" << H2MatrixSolver::solveRelErr(lenX, &X1[0], &X2[0])<<std::endl;
  // Eigen::PartialPivLU<Eigen::MatrixXcd> precon2(RX3);
  // //GMRES with dense preconditioning and symmetric mtrix
  // std::fill(X1.begin(), X1.end(), std::complex<double>(0., 0.));
  // matM.solveGMRESDensePrecon(1e-13, precon2, A_sorted, &X1[0], &X2[0], 10, 50);
  // if (mpi_rank == 0) {
  //   std::cout << "GMRES (dense preconditioner) Residual: " << matM.resid[matM.iters] << ", Iters: " << matM.iters << std::endl;
  //   for (long long i = 0; i <= matM.iters; i++)
  //     std::cout << "iter "<< i << ": " << matM.resid[i] << std::endl;
  // }

  //matA.free_all_comms();
  matM.free_all_comms();
  MPI_Finalize();

//   /*matA.freeSparseMV();
//   matA.free_gpu_handles();
//   matM.free_gpu_handles();
//   finalizeGpuEnvs(handle);
//   finalizeNcclComms(nccl_comms);*/
  return 0;
}

