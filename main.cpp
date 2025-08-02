
#include <solver.hpp>
#include <test_funcs.hpp>
#include <include/elast3d.hpp>
#include <string>

#include <Eigen/Dense>

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
  std::string tree_mode = argc > 7 ? std::string(argv[7]) : "default";
  //std::string mode = argc > 7 ? std::string(argv[7]) : "h2";
  //const char* csv = argc > 8 ? argv[8] : nullptr;

  // leaf size is expressed in terms of #elems, since we don't want to split an elment
  leaf_size = Nbody < leaf_size ? Nbody : leaf_size;

  const std::string MAT = "160";
  long long n_nodes, n_elems;
  std::vector<double> nodes;
  std::vector<double> elems;
  //std::vector<double> elems_polar;
  // Reading the mes data (i.e. nodes and elems)
  // For the elements we calculate the centroid and store it in elems
  read_mesh_data(n_nodes, nodes, n_elems, elems, "../input/mesh_sphere_" + MAT + "nodes.inp");
  long long num_nodes, num_elems;
  read_mesh_specs(num_nodes, num_elems, "../input/mesh_sphere_" + MAT + "nodes.inp");
  std::cout<<"New "<<num_nodes<<" "<<num_elems<<std::endl;
  std::vector<struct elastWave3d::nodal_point> nodes2(num_nodes);
  std::vector<struct elastWave3d::element> elems2(num_elems);
  read_mesh_fortran(num_nodes, nodes2, num_elems, elems2);
  // Mesh check
  for (int i = 0; i < 5; ++i)
    std::cout<<nodes2[i].xc[0]<<", "<<nodes2[i].xc[1]<<", "<<nodes2[i].xc[2]<<std::endl;
  std::cout<<std::endl;
  for (int i = 0; i < 5; ++i){
    for (int d=0 ; d < 3; ++d)
      std::cout<<nodes[i*3+d]<<", ";
    std::cout<<std::endl;
  }


  // check that the sizes match
  std::cout<<nodes.size()/3<<" " <<elems.size()/3<<std::endl;

  // create index array nodes + elements
  std::vector<long long> idx(n_nodes + n_elems);
  std::iota(idx.begin(), idx.end(), 0);
  std::vector<double> all(nodes);
  all.insert(all.end(), elems.begin(), elems.end());

  long long levels, Nleaf, ncells;
  std::vector<Cell> cell;
  if (tree_mode == "standard") {
    levels = (long long) std::ceil(std::log2((double)Nbody / leaf_size));
    Nleaf = (long long)1 << levels;
    ncells = Nleaf + Nleaf - 1;
    cell.resize(ncells);
    // build the tree for the whole matrix
    buildBinaryTree(&cell[0], &all[0], idx.data(), Nbody, levels);
  } else {
    long long levels_elems = (long long) std::ceil(std::log2((double)n_elems / leaf_size));
    long long Nleaf_elems = (long long)1 << levels_elems;
    long long ncells_elems = Nleaf_elems + Nleaf_elems - 1;
    long long levels_nodes = (long long) std::ceil(std::log2((double)n_nodes / leaf_size));
    long long Nleaf_nodes = (long long)1 << levels_nodes;
    long long ncells_nodes = Nleaf_nodes + Nleaf_nodes - 1;
    if (tree_mode == "fused1") {
      levels = levels_elems + 1;
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
     cell[1].Child[1] = 4;
    } else {
      if (tree_mode == "fused2") {
        levels = levels_elems;
        Nleaf = Nleaf_nodes + Nleaf_elems;
        ncells = ncells_elems + ncells_nodes + 1;
        cell.resize(ncells);
        buildBinaryTree3(&cell[0], &nodes[0], idx.data(), n_nodes, levels_nodes, 1, 0);
        buildBinaryTree3(&cell[0], &elems[0], &idx[n_nodes], n_elems, levels_elems, 0, n_nodes);
        /* root has three children */
        cell[0].Child[0] = 1;
        cell[0].Child[1] = 4;
        cell[0].Body[0] = 0;
        cell[0].Body[1] = n_nodes + n_elems;
      } else {
        std::cout<<"Invalid tree mode '" + tree_mode +"'"<<std::endl;
        return -1;
      }
    }
  }
  
  std::cout<<"Elements = "<<Nbody<<", Leaf = "<<leaf_size<<", Levels = "<<levels<<", #Leafs = "<<Nleaf<<", #Cells = "<<ncells<<std::endl;
  //buildBinaryTree2(&cell[0], &elems_polar[0], idx.data(), Nbody, levels); 
  //for (long long i = 0; i < ncells; ++i) {
  //  std::cout<<"Cell "<<i<<": "<<cell[i].Body[1] - cell[i].Body[0]<<", "<<cell[i].Body[0]<<" - "<<cell[i].Body[1]<<std::endl;
  //}

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
  read_data(b.data(), "../input/rhs_sphere_" + MAT + ".dat", n_mat);
  std::vector<std::complex<double>> x(n_mat);
  read_data(x.data(), "../input/x_sphere_" + MAT + ".dat", n_mat);
  std::vector<std::complex<double>> mat(n_mat * n_mat);
  read_data(mat.data(), "../input/mat_sphere_" + MAT + ".dat", n_mat * n_mat);

  // Get the U matrix (element/element interactions and sort it according to the tree)
  Eigen::Map<Eigen::MatrixXcd> A(mat.data(), n_mat,  n_mat);
  // scale the matrix
  Eigen::MatrixXcd RN = A.topLeftCorner(n_nodes * 3, n_nodes * 3);
  Eigen::MatrixXcd RE = A.bottomRightCorner(n_elems * 3, n_elems * 3);
  std::cout<<"Nodes "<<RN.diagonal().real().minCoeff()<< " " << RN.diagonal().real().maxCoeff()<<std::endl;
  std::cout<<"Elements "<<RE.diagonal().real().minCoeff()<< " " << RE.diagonal().real().maxCoeff()<<std::endl;
  //std::cout<<"Nodes "<<RN.diagonal().imag().minCoeff()<< " " << RN.diagonal().imag().maxCoeff()<<std::endl;
  //std::cout<<"Elements "<<RE.diagonal().imag().minCoeff()<< " " << RE.diagonal().imag().maxCoeff()<<std::endl;
  Eigen::MatrixXcd S = Eigen::MatrixXcd::Identity(n_mat, n_mat);
  for (long long i = n_nodes * 3; i < n_mat; ++i)
    S(i, i) = std::sqrt(std::max(std::abs(RN.diagonal().real().minCoeff()), std::abs(RN.diagonal().real().maxCoeff())) / std::max(std::abs(RE.diagonal().real().minCoeff()), std::abs(RE.diagonal().real().maxCoeff()))); //17;//32;
  A = S * A * S;
  Eigen::MatrixXcd RN2 = A.topLeftCorner(n_nodes * 3, n_nodes * 3);
  Eigen::MatrixXcd RE2 = A.bottomRightCorner(n_elems * 3, n_elems * 3);
  std::cout<<"Nodes "<<RN2.diagonal().real().minCoeff()<< " " << RN2.diagonal().real().maxCoeff()<<std::endl;
  std::cout<<"Elements "<<RE2.diagonal().real().minCoeff()<< " " << RE2.diagonal().real().maxCoeff()<<std::endl;
  //std::cout<<"Nodes "<<RN2.diagonal().imag().minCoeff()<< " " << RN2.diagonal().imag().maxCoeff()<<std::endl;
  //std::cout<<"Elements "<<RE2.diagonal().imag().minCoeff()<< " " << RE2.diagonal().imag().maxCoeff()<<std::endl;

  //Eigen::MatrixXcd U = A.bottomRightCorner(n_elems * 3, n_elems * 3);
  Eigen::MatrixXcd A_sorted(Nbody * 3, Nbody * 3);
  Eigen::Map<Eigen::VectorXcd> B(b.data(), n_mat);
  Eigen::VectorXcd B_sorted(Nbody * 3);
  Eigen::Map<Eigen::VectorXcd> X(x.data(), n_mat);
  Eigen::VectorXcd X_sorted(Nbody * 3);
  std::vector<double> all_sorted(Nbody * 3);

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
        // shuffle the all vector so that the order of points matches the sorted matrix
        all_sorted[i * 3 + ii] = all[idx[i] * 3 + ii];
        for (int jj = 0; jj < 3; ++jj) {
           A_sorted(i * 3 + ii, j * 3 + jj) = A(idx[i] * 3 + ii , idx[j] * 3 + jj);
        }
      }
    }
  }
  // for (int i = 0; i < Nbody; ++i) {
  //     for (int ii = 0; ii < 3; ++ii) {
  //          B_sorted(i * 3 + ii) = B(idx[i] * 3 + ii);
  //          X_sorted(i * 3 + ii) = X(idx[i] * 3 + ii);
  //     }
  // }

  // Eigen::PartialPivLU<Eigen::MatrixXcd> fac(A);
  // Eigen::VectorXcd test1 = fac.solve(B);
  // Eigen::PartialPivLU<Eigen::MatrixXcd> fac2(A_sorted);
  // Eigen::VectorXcd test2 = fac2.solve(B_sorted);
  // double err1 = H2MatrixSolver::solveRelErr(n_mat, x.data(), test1.data());
  // double err2 = H2MatrixSolver::solveRelErr(n_mat, X_sorted.data(), test2.data());
  // std::cout<<"Error (unsorted): "<<err1<<std::endl;
  // std::cout<<"Error (sorted): "<<err2<<std::endl;
  //Eigen::FullPivLU<Eigen::MatrixXcd> fac(RX);
  //fac.setThreshold(1e-6);

  //double threshold = 1e-4 * std::max(A.real().maxCoeff(), std::abs(A.real().minCoeff()));
  //double norm = A.lpNorm<Eigen::Infinity>();
  //double norm = A.norm();
  /*double threshold = 1e-2;
  // Try taking the norm per row
  //std::cout<<"Norm: "<<norm<<std::endl;
  std::cout<<"Threshold: "<< threshold <<std::endl;
  for (int i = 0; i < 48; ++i) {
    //double norm = A.block(i, 48, 1, n_mat-48).lpNorm<Eigen::Infinity>();
    double norm = A.block(i, 48, 1, n_mat-48).norm();
    //double norm = A.row(i).lpNorm<Eigen::Infinity>();
    std::cout<<"Norm: "<<norm<<std::endl;
    long long count = 0;
    for (int j = 48; j < n_mat; ++j)
      if (std::abs(A(i,j)) >= threshold * norm)
        count++;
    std::cout<<"Row "<< i <<": " << count<<", Density: "<< ((double)count)/(n_mat - 48)<<std::endl;
  }*/


  // std::cout<<"Solver"<<std::endl;
  // // generate the H2 matrix (with normal basis)
  // H2MatrixSolver matA(A_sorted, epi, rank, leveled_rank, cell, theta, levels);
  // std::cout<<"Construction"<<std::endl;

  // Lets assume we have the first leaf level node with 48 elements
  // near field dim
  /*long long M = 48;
  // far field dim
  long long N = A_sorted.rows() - M;
  Eigen::MatrixXcd far = A_sorted.bottomLeftCorner(N, M);
  std::cout<<"Rows: "<<N/3<<std::endl;
  long long K = std::min(M, N);
  Eigen::MatrixXcd RX = Eigen::MatrixXcd::Zero(K, M);
  if (K < N) {
    Eigen::HouseholderQR<Eigen::MatrixXcd> qr(far);
    RX = qr.matrixQR().topRows(K).triangularView<Eigen::Upper>();
  }
  else
    std::cout<<"N < M"<<std::endl;

  Eigen::ColPivHouseholderQR<Eigen::MatrixXcd> rrqr(RX);
  //rrqr.setThreshold(1e-4);
  rank = 16;//rrqr.rank();
  std::cout<<"Rank: "<<rank<<std::endl;
  Eigen::MatrixXcd T(rank, M), Q(M, rank);
  Eigen::MatrixXcd TQ = rrqr.householderQ(); 
  Eigen::MatrixXcd test = TQ.leftCols(rank) * (rrqr.matrixR().topRows(rank).template triangularView<Eigen::Upper>());
  double terror = (RX * rrqr.colsPermutation() - test).norm() / RX.norm();
  std::cout<<"QR Error: "<<terror<<std::endl;
  T.topRows(rank) = rrqr.matrixR().topRows(rank);
  T.topLeftCorner(rank, rank).triangularView<Eigen::Upper>().solveInPlace(T.topRightCorner(rank, M - rank));
  T.topLeftCorner(rank, rank) = Eigen::MatrixXcd::Identity(rank, rank);
  bool orth = false;
  if (orth) {
    RX = (rrqr.colsPermutation() * T.topRows(rank).transpose());
    Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixXcd>> qr(RX);
    Q = qr.householderQ();
    T.setZero();
    T.topLeftCorner(rank, rank) = qr.matrixQR().topRows(rank).triangularView<Eigen::Upper>();
  } else { 
    Q.setZero();
    Q.leftCols(rank) = rrqr.colsPermutation() * T.topRows(rank).transpose();
    T.setZero();
    T.topLeftCorner(rank, rank) = Eigen::MatrixXcd::Identity(rank, rank);
  }
  std::cout<<"Q "<<Q.rows()<<" "<<Q.cols()<<std::endl;
  std::cout<<"T "<<T.rows()<<" "<<T.cols()<<std::endl;

  // create a random vector to estimate the accuracy via matvec
  std::vector<std::complex<double>> y(rank);
  std::mt19937_64 rng(42);
  std::uniform_real_distribution distx(0., 1.);
  std::generate(y.begin(), y.end(), 
     [&]() { return std::complex<double>(distx(rng), 0.); });
  Eigen::Map<Eigen::VectorXcd> Y(y.data(), rank);
  Eigen::VectorXcd approx_ref = Q * Y;

  long long num_pts =  N / 3;
  long long node_samples = 144 / 4;
  long long elem_samples = 316 / 4;
  long long num_samples = node_samples  + elem_samples;
  long long N2 = num_samples * 3;
  std::cout<<"Rows: "<<N2/3<<std::endl;

  std::uniform_int_distribution<long long> dist(0, 143);
  Eigen::MatrixXcd Sample(N2, M);
  std::vector<int> selected(num_samples);
  long long i;
  for (i = 0; i< node_samples; ++i) {
    long long idx = dist(rng);
    while (std::find(selected.begin(), selected.end(), idx) != selected.end())
      idx = dist(rng);
    selected.emplace_back(idx);
    for (long long j = 0; j<3; ++j) {
      Sample.row(i * 3 + j) = far.row(idx * 3 + j);
    }
  }
  std::uniform_int_distribution<long long> distE(144, num_pts - 1);
  for (; i< num_samples; ++i) {
    long long idx = distE(rng);
    while (std::find(selected.begin(), selected.end(), idx) != selected.end())
      idx = distE(rng);
    selected.emplace_back(idx);
    for (long long j = 0; j<3; ++j) {
      Sample.row(i * 3 + j) = far.row(idx * 3 + j);
    }
  }
  std::cout<<"Sample Points: "<<std::endl;
  K = std::min(M, N2);
  RX = Eigen::MatrixXcd::Zero(K, M);
  if (K < N2) {
    Eigen::HouseholderQR<Eigen::MatrixXcd> qr(Sample);
    RX = qr.matrixQR().topRows(K).triangularView<Eigen::Upper>();
  }
  else
    std::cout<<"N2 < M"<<std::endl;

  Eigen::ColPivHouseholderQR<Eigen::MatrixXcd> rrqr2(RX);
  Eigen::MatrixXcd T2(rank, M), Q2(M, rank);
  T2.topRows(rank) = rrqr2.matrixR().topRows(rank);
  T2.topLeftCorner(rank, rank).triangularView<Eigen::Upper>().solveInPlace(T2.topRightCorner(rank, M - rank));
  T2.topLeftCorner(rank, rank) = Eigen::MatrixXcd::Identity(rank, rank);
  if (orth) {
    RX = (rrqr2.colsPermutation() * T2.topRows(rank).transpose());
    Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixXcd>> qr2(RX);
    Q2 = qr2.householderQ();
    T2.setZero();
    T2.topLeftCorner(rank, rank) = qr2.matrixQR().topRows(rank).triangularView<Eigen::Upper>();
  } else { 
    Q2.setZero();
    Q2.leftCols(rank) = rrqr2.colsPermutation() * T2.topRows(rank).transpose();
    T2.setZero();
    T2.topLeftCorner(rank, rank) = Eigen::MatrixXcd::Identity(rank, rank);
  }
  std::cout<<"Q2 "<<Q2.rows()<<" "<<Q2.cols()<<std::endl;
  std::cout<<"T2 "<<T2.rows()<<" "<<T2.cols()<<std::endl;
  Eigen::VectorXcd approx = Q2 * Y;
  double testerr = std::sqrt((approx_ref - approx).squaredNorm() / approx_ref.squaredNorm());
  std::cout<<"Error: "<<testerr<<std::endl;*/



  
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

  // multiply by 3 to get the actual length
  // this way, we can reduce the number of elems if necessary
  long long lenX = Nbody * 3;
  std::vector<std::complex<double>> X1(lenX, std::complex<double>(0., 0.));
  std::vector<std::complex<double>> X2(lenX, std::complex<double>(0., 0.));

  // copy random x into X1, X2
  std::copy(&Xbody[0], &Xbody[lenX], &X1[0]);
  std::copy(&Xbody[0], &Xbody[lenX], &X2[0]);

  // MPI_Barrier(MPI_COMM_WORLD);
  // double matvec_time = MPI_Wtime(), matvec_comm_time;
  // std::cout<<"Matvec"<<std::endl;
  // matA.matVecMul(&X1[0]);
  // std::cout<<"Matvec finished"<<std::endl;
  // todo remove this testing code of the transpose
  //matA.matVecMulSp(handle, &X1[0]);
  //Eigen::MatrixXcd RX = U_sorted.triangularView<Eigen::Lower>();
  //Eigen::MatrixXcd RX2 = U_sorted.triangularView<Eigen::StrictlyLower>().transpose();
  //Eigen::MatrixXcd RX3 = RX + RX2;
  //Eigen::Map<Eigen::VectorXcd> t2(&X2[0], lenX);
  //Eigen::VectorXcd r = RX3 * t2;

  //MPI_Barrier(MPI_COMM_WORLD);
  //matvec_time = MPI_Wtime() - matvec_time;
  //matvec_comm_time = ColCommMPI::get_comm_time();

  // calculate reference into X2
  double refmatvec_time = MPI_Wtime();
  Eigen::Map<Eigen::VectorXcd> t(&X2[0], lenX);
  t = A_sorted * t;

  refmatvec_time = MPI_Wtime() - refmatvec_time;
  // double cerr = H2MatrixSolver::solveRelErr(lenX, &X1[0], &X2[0]);
  // //double cerr = H2MatrixSolver::solveRelErr(lenX, r.data(), result.data());

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

  MPI_Barrier(MPI_COMM_WORLD);
  double m_construct_time = MPI_Wtime(), m_construct_comm_time;
  H2MatrixSolver matM(A_sorted, 0, rank, leveled_rank, cell, theta, levels, all_sorted);
  //H2MatrixSolver matM(A_sorted, 0, rank, leveled_rank, cell, theta, levels);
  //H2MatrixSolver matM(A_sorted, epi, rank, leveled_rank, cell, theta, levels, all_sorted);

  MPI_Barrier(MPI_COMM_WORLD);
  m_construct_time = MPI_Wtime() - m_construct_time;
  m_construct_comm_time = ColCommMPI::get_comm_time();

  std::copy(&Xbody[0], &Xbody[lenX], &X1[0]);
  matM.matVecMul(&X1[0]);
  double cerr_m = H2MatrixSolver::solveRelErr(lenX, &X1[0], &X2[0]);

  if (mpi_rank == 0) {
    std::cout << "H^2-Preconditioner Construct Err: " << cerr_m << std::endl;
  }

  //initNcclComms(&nccl_comms, matM.allocedComm);
  //matM.init_gpu_handles(nccl_comms);

  MPI_Barrier(MPI_COMM_WORLD);
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
    std::cout << "H^2-Preconditioner Construct Err: " << cerr_m << std::endl;
    //std::cout << "H^2-Matrix Factorization Time: " << h2_factor_time << ", " << h2_factor_comm_time << std::endl;
    //std::cout << "H^2-Matrix Substitution Time: " << h2_sub_time << ", " << h2_sub_comm_time << std::endl;
    std::cout << "H^2-Matrix Substitution Err: " << serr << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double gmres_time = MPI_Wtime(), gmres_comm_time;
  matM.solveGMRESDense(1e-12, A_sorted, &X1[0], &X2[0], 10, 50);
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

  //   /*if (csv != nullptr)
  //     write_to_csv(csv, mpi_size, Nbody, theta, leaf_size, rank, epi, mode.data(), cerr, 
  //       h2_construct_time, h2_construct_comm_time, matvec_time, matvec_comm_time, refmatvec_time, 
  //       m_construct_time, m_construct_comm_time, cerr_m, h2_factor_time, h2_factor_comm_time, h2_sub_time, h2_sub_comm_time, serr, 
  //       matA.resid[matA.iters], matA.iters, gmres_time, gmres_comm_time, matA.resid.data());*/
  }

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

