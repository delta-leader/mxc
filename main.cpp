
#include <solver.hpp>
#include <test_funcs.hpp>
#include <string>
#include <float.h>

#include <mkl.h>
#include <Eigen/Dense>
#include <cublas_v2.h>

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  typedef std::complex<double> DT; typedef std::complex<float> DT_low;
  //typedef double DT; typedef float DT_low;
  std::vector<cublasComputeType_t> comp = {CUBLAS_COMPUTE_32F, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_COMPUTE_32F_FAST_16BF};
  //COMP = CUBLAS_COMPUTE_32F_FAST_TF32;
  //const auto COMP = CUBLAS_COMPUTE_32F_FAST_16BF;
  //const auto COMP = CUBLAS_COMPUTE_32F_FAST_16F;
  

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
  for (size_t t=0; t<comp.size(); ++t) {
  std::vector<double> params = {1};//{1, 2, 5, 10, 20, 50};
  std::vector<double> conds(params.size());
  std::vector<double> approx(params.size());
  std::vector<double> consterr(params.size());
  std::vector<double> solverr(params.size());
  std::vector<double> ir_b(params.size());
  std::vector<double> ir_f(params.size());
  std::vector<double> gmres_f(params.size());
  std::vector<double> gmres_b(params.size());
  for (size_t i=0; i<params.size(); ++i) {
  // kernel functions, here we select the appropriate function
  // by setting the corresponding parameters
  // In this case the template argument deduction fails for the matvec (double)
  // I might need to specifiy them explicitly
  //Laplace3D<DT> eval(params[i]);
  //Yukawa3D<DT> eval(params[i], params[j]);
  //Gaussian<DT> eval(params[i]);
  //IMQ<DT> eval(params[i]);
  //Matern3<DT> eval(params[i]);
  Helmholtz3D<DT> eval(params[i], 1.);
  
  // body contains the points
  // 3 corresponds to the dimension
  std::vector<double> body(Nbody * 3);
  Vector_dt<DT> Xbody(Nbody);
  Vector_dt<DT> Ones(Nbody);
  Ones.ones();
  // array containing the nodes in the cluster tree
  std::vector<Cell> cell(ncells);

  // create the points (i.e. bodies)
  //mesh_sphere(&body[0], Nbody, std::pow(Nbody, 1./2.));
  uniform_unit_cube_rnd(&body[0], Nbody, std::pow(Nbody, 1./3.), 3, 999);
  //uniform_unit_cube_rnd(&body[0], Nbody, 1, 3, 999);
  //uniform_unit_cube(&body[0], Nbody, std::pow(Nbody, 1./3.), 3);
  //build the tree (i.e. set the values in the cell array)
  buildBinaryTree(levels, Nbody, &body[0], &cell[0]);

  // generate a random vector Xbody (used in Matvec)
  Xbody.generate_random();

  // get condition number (single process only)
  std::vector<DT> A(Nbody * Nbody);
  gen_matrix(eval, Nbody, Nbody, &body[0], &body[0], A.data());
  Eigen::Map<const Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic>> Amap(&A[0], Nbody, Nbody);
  Eigen::JacobiSVD<Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic>> svd(Amap);
  double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
  conds[i] = cond;
  
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
  Vector_dt<DT> X1(lenX);
  Vector_dt<DT> X2(lenX);
  Vector_dt<DT> B(Ones);
  Vector_dt<DT> B_ref(lenX);

  // copy the random vector
  std::copy(&Xbody[matA.local_bodies.first], &Xbody[matA.local_bodies.second], &X1[0]);

  Vector_dt<DT_low> X1_low(X1);

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

  matA.matVecMul(&B[0]);
  mat_vec_reference(eval, lenX, Nbody, &B_ref[0], &Ones[0], &body[matA.local_bodies.first * 3], &body[0]);
  double approx_err = computeRelErr(lenX, &B[0], &B_ref[0]);
  approx[i] = approx_err;

  int mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0) {
    //std::cout<< params[i] << std::endl;
    //std::cout << cond <<std::endl;
    //std::cout << cerr << std::endl;
    //std::cout << approx_err << std::endl;
    
    std::cout << "Condition #: " << cond <<std::endl;
    std::cout << "Construct Err: " << cerr << std::endl;
    std::cout << "Approximation Err: " << approx_err << std::endl;
    //std::cout << "H^2-Matrix Construct Time: " << h2_construct_time << ", " << h2_construct_comm_time << std::endl;
    //std::cout << "H^2-Matvec Time: " << matvec_time << ", " << matvec_comm_time << std::endl;
    //std::cout << "Dense Matvec Time: " << refmatvec_time << std::endl;
    /*Eigen::MatrixXcd A(Nbody, Nbody);
    gen_matrix(eval, Nbody, Nbody, body.data(), body.data(), A.data());
    double cond = 1. / A.lu().rcond();
    std::cout << "Condition #: " << cond << std::endl;*/
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  double m_construct_time = MPI_Wtime(), m_construct_comm_time;
  
  // new H2 matrix using a fixed rank
  H2MatrixSolver<DT_low> matM;
  if (mode.compare("h2") == 0)
    matM = H2MatrixSolver(eval, epi, rank, cell, theta, &body[0], levels, true, true);
  else if (mode.compare("hss") == 0)
    matM = H2MatrixSolver(eval, epi, rank, cell, 0., &body[0], levels, true);

  MPI_Barrier(MPI_COMM_WORLD);
  m_construct_time = MPI_Wtime() - m_construct_time;
  m_construct_comm_time = ColCommMPI::get_comm_time();

  std::copy(&Xbody[matA.local_bodies.first], &Xbody[matA.local_bodies.second], &X1[0]);
  matM.matVecMul(&X1_low[0]);
  double cerr_m = computeRelErr(lenX, &Vector_dt<DT>(X1_low)[0], &X2[0]);
  Vector_dt<DT_low> B_low(Ones);
  matM.matVecMul(&B_low[0]);
  approx_err = computeRelErr(lenX, &Vector_dt<DT>(B_low)[0], &B_ref[0]);
  consterr[i] = approx_err;

  MPI_Barrier(MPI_COMM_WORLD);
  double h2_factor_time = MPI_Wtime(), h2_factor_comm_time;

  matM.factorizeM();
  //matM.factorizeM(comp[t]);
  
  //Vector_dt<DT_low> test(Ones);
  //matM.matVecMul(&test[0]);
  //approx_err = computeRelErr(lenX, &Vector_dt<DT>(test)[0], &B_ref[0]);
  //std::cout<<"After fact "<<approx_err<<std::endl;
  
  MPI_Barrier(MPI_COMM_WORLD);
  h2_factor_time = MPI_Wtime() - h2_factor_time;
  h2_factor_comm_time = ColCommMPI::get_comm_time();
  std::copy(B.begin(), B.end(), B_ref.begin());
  B_low = Vector_dt<DT_low>(B);

  MPI_Barrier(MPI_COMM_WORLD);
  double h2_sub_time = MPI_Wtime(), h2_sub_comm_time;

  matM.solvePrecondition(&B_low[0]);

  MPI_Barrier(MPI_COMM_WORLD);
  h2_sub_time = MPI_Wtime() - h2_sub_time;
  h2_sub_comm_time = ColCommMPI::get_comm_time();
  double serr = computeRelErr(lenX, &Vector_dt<DT>(B_low)[0], &Ones[0]);
  //std::fill(X1.begin(), X1.end(), std::complex<double>(0., 0.));

  // testing application of the preconditioner in high precision
  H2MatrixSolver<DT> matM_test(matM);
  matM_test.solvePrecondition(&B[0]);
  double serr_test = computeRelErr(lenX, &B[0], &Ones[0]);
  solverr[i] = serr_test;

  if (mpi_rank == 0) {
    //std::cout << "H^2-Preconditioner Construct Time: " << m_construct_time << ", " << m_construct_comm_time << std::endl;
    std::cout << "H^2-Preconditioner Construct Err: " << cerr_m << std::endl;
    std::cout << "H^2-Preconditi Approximation Err: " << approx_err << std::endl;
    //std::cout << "H^2-Matrix Factorization Time: " << h2_factor_time << ", " << h2_factor_comm_time << std::endl;
    //std::cout << "H^2-Matrix Substitution Time: " << h2_sub_time << ", " << h2_sub_comm_time << std::endl;
    std::cout << "H^2-Matrix Substitution Err: " << serr << std::endl;
    std::cout << "H^2-Matrix Substitution Err: " << serr_test << std::endl;

    //std::cout << cerr_m << std::endl;
    //std::cout << approx_err << std::endl;
    //std::cout << serr << std::endl;
    //std::cout << serr_test << std::endl;
  }
  
  B.reset();
  MPI_Barrier(MPI_COMM_WORLD);
  double ir_time = MPI_Wtime(), ir_comm_time;
 // long long iters = matA.solveIR(epi, matM, &X1[0], &X2[0], 200);
 long long iters = matA.solveIR(epi, matM, &B[0], &B_ref[0], 200);

  MPI_Barrier(MPI_COMM_WORLD);
  ir_time = MPI_Wtime() - ir_time;
  ir_comm_time = ColCommMPI::get_comm_time();
  ir_b[i] = matA.resid[iters];
  ir_f[i] = computeRelErr(lenX, &B[0], &Ones[0]);

  if (mpi_rank == 0) {
    std::cout << matA.resid[iters] << std::endl;
    std::cout << computeRelErr(lenX, &B[0], &Ones[0]) << std::endl;
    std::cout << iters << std::endl;
    std::cout << "IR Residual: " << matA.resid[iters] << ", Iters: " << iters << std::endl;
    //std::cout << "Forward Error: " << computeRelErr(lenX, &B[0], &Ones[0]) << std::endl;
    //std::cout << "IR Time: " << ir_time << ", Comm: " << ir_comm_time << std::endl;
    //for (long long i = 0; i <= matA.iters; i++)
      //std::cout << "iter "<< i << ": " << matA.resid[i] << std::endl;
  }

  B.reset();
  H2MatrixSolver<DT> matM_high(matM);
  MPI_Barrier(MPI_COMM_WORLD);
  double gmres_ir_time = MPI_Wtime(), gmres_ir_comm_time;
  //iters = matA.solveGMRESIR(epi, matM_high, &X1[0], &X2[0], 5, 50, 1);
  iters = matA.solveGMRESIR(epi, matM_high, &B[0], &B_ref[0], 5, 50, 1);

  MPI_Barrier(MPI_COMM_WORLD);
  gmres_ir_time = MPI_Wtime() - gmres_ir_time;
  gmres_ir_comm_time = ColCommMPI::get_comm_time();

  gmres_b[i] = matA.resid[iters];
  gmres_f[i] = computeRelErr(lenX, &B[0], &Ones[0]);

  if (mpi_rank == 0) {
    //std::cout << matA.resid[iters] << std::endl;
    //std::cout << computeRelErr(lenX, &B[0], &Ones[0]) << std::endl;
    //std::cout << iters << std::endl;
    //std::cout << "GMRES-IR Residual: " << matA.resid[iters] << ", Iters: " << iters << std::endl;
    //std::cout << "Forward Error: " << computeRelErr(lenX, &B[0], &Ones[0]) << std::endl;
    //std::cout << "GMRES-IR Time: " << gmres_ir_time << ", Comm: " << gmres_ir_comm_time << std::endl;
  }
  
  matA.free_all_comms();
  matM.free_all_comms();
  matM_high.free_all_comms();
  matM_test.free_all_comms();
  //std::cout<<std::endl;
  }
  std::vector<std::vector<double>> results = {params, conds, approx, consterr, solverr, ir_b, ir_f, gmres_b, gmres_f};
  std::cout<<Nbody<<std::endl;
  for (size_t i=0; i<results.size(); ++i){
    for (size_t j=0; j<results[i].size(); ++j)
      std::cout<<results[i][j]<<", ";
    std::cout<<std::endl;
  }
  std::cout<<std::endl;
  }
  MPI_Finalize();
  return 0;
}

