
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
  double theta_precon = argc > 5 ? std::atof(argv[5]) : 1e0;
  long long rank = argc > 6 ? std::atoll(argv[6]) : 32;
  long long leveled_rank =  argc > 7 ? std::atoll(argv[7]) : 0;
  double epi = argc > 8 ? std::atof(argv[8]) : 1e-10;
  double theta = argc > 9 ? std::atof(argv[9]) : 1e0;
  long long inner_iter = argc > 10 ? std::atoll(argv[10]) : 10;
  long long max_iter = argc > 11 ? std::atoll(argv[11]) : 50;
  long long r1 = argc > 12 ? std::atoll(argv[12]) : 0;
  //long long leveled_r1 = argc > 13 ? std::atoll(argv[13]) : 0;
  long long r2 = argc > 13 ? std::atoll(argv[13]) : 0;
  //long long leveled_r2 = argc > 15 ? std::atoll(argv[15]) : 0;
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

  //std::string prefix = "../input/cache/";
  //std::string filename = std::to_string(geom) + "_" + MAT + "_" + std::to_string((int)omega) + "_" + std::to_string(leaf_size) + ".dat";
  if (mpi_rank == 0) {
    std::cout<<"M = "<<M<<", geom = "<<geom<<std::endl;
    std::cout<<"Omega = "<<omega<<", Leaf-size = "<<leaf_size<<", admis_precon = "<<theta_precon<<", rank = "<<rank<<", leveled rank = "<<leveled_rank;
    std::cout<<", epsilon = "<<epi<<", theta = "<<theta<<", inner iter = "<<inner_iter<<", max iter = "<<max_iter;
    std::cout<<", r1 = "<<r1<<", r2 = "<<r2<<std::endl;
    //std::cout<<", r1 = "<<r1<<", leveled_r1 = "<<leveled_r1<<", r2 = "<<r2<<", leveled_r2 = "<<leveled_r2<<std::endl; 
    //std::cout<<"Reading file "<<filename<<std::endl;
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
  double h2_construct_time = MPI_Wtime(), h2_construct_comm_time;
  //H2MatrixSolver matA;
  //if (r1)
  //  matA = H2MatrixSolver(matgen, epi, rank, leveled_rank, cell, theta, levels, omega, matgen.get_elems(), r1, leveled_r1, r2, leveled_r2);
  //else
  bool io = false;
  std::cout<<"Reading: "<<io<<std::endl;
  H2MatrixSolver<std::complex<double>> matA(matgen, epi, rank, leveled_rank, cell, theta, levels, omega, io);
  MPI_Barrier(MPI_COMM_WORLD);
  h2_construct_time = MPI_Wtime() - h2_construct_time;
  h2_construct_comm_time = ColCommMPI::get_comm_time();
  //std::cout<<"Construction finished"<<std::endl;

  // multiply by 3 to get the actual length
  long long lenX = (matA.local_bodies.second - matA.local_bodies.first) * 3;
  long long offset = matA.local_bodies.first * 3;
  std::vector<std::complex<double>> X1(lenX, std::complex<double>(0., 0.));
  //std::vector<std::complex<float>> X1_low(lenX, std::complex<float>(0., 0.));
  std::vector<std::complex<double>> X2(lenX, std::complex<double>(0., 0.));

  // copy random x into X1
  std::copy(&Xbody[offset], &Xbody[offset + lenX], &X1[0]);
  //std::copy(&Xbody[offset], &Xbody[offset + lenX], &X2[0]);
  //std::cout<<"Matvec start"<<std::endl;
  // calculate H-matvec into X1
  MPI_Barrier(MPI_COMM_WORLD);
  double matvec_time = MPI_Wtime(), matvec_comm_time;
  matA.matVecMul(&X1[0]);
  MPI_Barrier(MPI_COMM_WORLD);
  matvec_time = MPI_Wtime() - matvec_time;
  matvec_comm_time = ColCommMPI::get_comm_time();
  //std::cout<<"Matvec finished"<<std::endl;

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
  // testing hidr for the salt model? might be better to do it for the sphere first
  H2MatrixSolver<std::complex<double>> precon(matgen, 0, rank, leveled_rank, cell, theta_precon, levels, omega, matgen.get_elems(), r1, 1, r2, 0);
  //H2MatrixSolver<std::complex<double>> precon(matgen, 0, rank, leveled_rank, cell, theta_precon, levels, omega);
  MPI_Barrier(MPI_COMM_WORLD);
  precon_construct_time = MPI_Wtime() - precon_construct_time;
  precon_construct_comm_time = ColCommMPI::get_comm_time();

  // copy random x into X1
  std::copy(&Xbody[offset], &Xbody[offset + lenX], &X1[0]);
  //for (long long i = 0; i<lenX; ++i)
  //  X1_low[i] = X1[i];
  // precon matvec
  MPI_Barrier(MPI_COMM_WORLD);
  double precon_matvec_time = MPI_Wtime(), precon_matvec_comm_time;
  precon.matVecMul(&X1[0]);
  MPI_Barrier(MPI_COMM_WORLD);
  precon_matvec_time = MPI_Wtime() - precon_matvec_time;
  precon_matvec_comm_time = ColCommMPI::get_comm_time();

  //for (long long i = 0; i<lenX; ++i)
  //  X1[i] = X1_low[i];
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
  const int RUNS = 1;

  for (int i = 0; i < RUNS; ++i) {
    // we would need to initialize X1 and X1_low here
    if (mpi_rank == 0) {
      std::cout<<"Run "<<i<<std::endl;
    }
    H2MatrixSolver<std::complex<double>> precon_tmp(precon);
 
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
    precon_tmp.solvePrecondition(&X1[0]);
    MPI_Barrier(MPI_COMM_WORLD);
    precon_sub_time = MPI_Wtime() - precon_sub_time;
    precon_sub_comm_time = ColCommMPI::get_comm_time();
    //for (long long i = 0; i<lenX; ++i)
    // X1[i] = X1_low[i];
    double serr = solveRelErr(lenX, &X1[0], &X2[0]);
    subst_time.push_back(precon_sub_time);


    if (!i &&mpi_rank == 0) {
      std::cout << "H^2-Matrix Factorization Time: " << precon_factor_time << ", " << precon_factor_comm_time << std::endl;
      std::cout << "H^2-Matrix Substitution Time: " << precon_sub_time << ", " << precon_sub_comm_time << std::endl;
      std::cout << "H^2-Matrix Substitution Err: " << serr << std::endl;
    }

    std::vector<std::complex<double>> rhs(lenX);
    //std::vector<double> incident = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90};
    std::vector<double> incident = {0};//, 10, 20};//, 30, 40, 50, 60, 70, 80, 90};
    double gmres_time, gmres_comm_time;
    for (size_t w = 0; w < incident.size(); w++) {
      //std::cout<<"Incident wave: "<<incident[w]<<std::endl;
      matgen.gen_rhs_sorted_single_layer(rhs.data(), offset, lenX, omega, incident[w]);
      
      // iterative solver
      std::fill(X1.begin(), X1.end(), std::complex<double>(0., 0.));
      MPI_Barrier(MPI_COMM_WORLD);
      gmres_time = MPI_Wtime();
      matA.solveGMRES(epi, precon_tmp, &X1[0], &rhs[0], inner_iter, max_iter);
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

      // calculate residual with respect to the dense matrix
      /*long long offset2 = matA.local_bodies.second * 3;
      std::vector<long long> offsets(mpi_size+1);
      MPI_Allgather(&offset2, 1, MPI_LONG_LONG_INT, &offsets[1], 1, MPI_LONG_LONG_INT, MPI_COMM_WORLD);
      std::vector<int> recvcounts(offsets.size() - 1), displs(offsets.size() - 1);
      std::transform(offsets.begin() + 1, offsets.end(), offsets.begin(), recvcounts.begin(), [](long long end, long long begin) { return (int)(end - begin); });
      std::transform(offsets.begin(), std::prev(offsets.end()), displs.begin(), [](long long begin) { return (int)begin; });
      Eigen::VectorXcd global_x(Nbody * 3);
      MPI_Allgatherv(&X1[0], lenX, MPI_C_DOUBLE_COMPLEX, global_x.data(), &recvcounts[0], &displs[0], MPI_C_DOUBLE_COMPLEX, MPI_COMM_WORLD);

      std::fill(X1.begin(), X1.end(), std::complex<double>(0., 0.));
      mat_vec_reference(matgen, lenX/3, Nbody, &X1[0], global_x.data(), matA.local_bodies.first, omega);
      serr = solveRelErr(lenX, &X1[0], &rhs[0]);
      if (mpi_rank == 0) {
        std::cout << "  Actual Residual: " << serr << std::endl;
      }*/
      MPI_File fh;
      std::string filename = "test2/data8_" + std::to_string(mpi_rank) + ".bin";
      MPI_File_open(MPI_COMM_SELF, filename.c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
      MPI_Offset offset = 0;
      MPI_Status status;
      MPI_File_write_at(fh, offset, X1.data(), lenX, MPI_C_DOUBLE_COMPLEX, &status);
      MPI_File_close(&fh);
    }
  }
  if (mpi_rank == 0) {
    std::cout<<std::endl<<"Factorization time: ";
    for (size_t i = 0; i < fact_time.size(); ++i)
      std::cout<<fact_time[i]<<", ";
    std::cout<<std::endl;
    std::cout<<"Substitution time: ";
    for (size_t i = 0; i < subst_time.size(); ++i)
      std::cout<<subst_time[i]<<", ";
    std::cout<<std::endl;
    std::cout<<"GMRES time: ";
    for (size_t i = 0; i < gm_time.size(); ++i)
      std::cout<<gm_time[i]<<", ";
    std::cout<<std::endl;
  }

  matA.free_all_comms();
  precon.free_all_comms();
  MPI_Finalize();

  return 0;
}

