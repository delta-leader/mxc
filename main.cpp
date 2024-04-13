
#include <geometry.hpp>
#include <kernel.hpp>
#include <build_tree.hpp>
#include <basis.hpp>
#include <comm.hpp>

#include <random>
#include <algorithm>

void solveRelErr(double* err_out, const std::complex<double>* X, const std::complex<double>* ref, long long lenX) {
  double err[2] = { 0., 0. };
  for (long long i = 0; i < lenX; i++) {
    std::complex<double> diff = X[i] - ref[i];
    err[0] = err[0] + (diff.real() * diff.real());
    err[1] = err[1] + (ref[i].real() * ref[i].real());
  }
  MPI_Allreduce(MPI_IN_PLACE, err, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  *err_out = std::sqrt(err[0] / err[1]);
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  long long Nbody = argc > 1 ? atoll(argv[1]) : 8192;
  double theta = argc > 2 ? atof(argv[2]) : 1e0;
  long long leaf_size = argc > 3 ? atoll(argv[3]) : 256;
  double epi = argc > 4 ? atof(argv[4]) : 1e-10;
  long long rank = argc > 5 ? atoll(argv[5]) : 100;
  long long nrhs = argc > 6 ? atoll(argv[6]) : 2;

  leaf_size = Nbody < leaf_size ? Nbody : leaf_size;
  long long levels = (long long)log2((double)Nbody / leaf_size);
  long long Nleaf = (long long)1 << levels;
  long long ncells = Nleaf + Nleaf - 1;
  MPI_Comm world;
  MPI_Comm_dup(MPI_COMM_WORLD, &world);

  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(world, &mpi_rank);
  MPI_Comm_size(world, &mpi_size);
  
  //Laplace3D eval(1);
  //Yukawa3D eval(1, 1.);
  //Gaussian eval(8);
  Helmholtz3D eval(1.e-1, 1.);
  
  std::vector<double> body(Nbody * 3);
  std::vector<std::complex<double>> Xbody(Nbody * nrhs);
  Cells cell(ncells);

  std::vector<CellComm> cell_comm(levels + 1);
  std::vector<ClusterBasis> basis(levels + 1);

  //mesh_unit_sphere(&body[0], Nbody, std::pow(Nbody, 1./2.));
  //mesh_unit_cube(&body[0], Nbody);
  uniform_unit_cube_rnd(&body[0], Nbody, std::pow(Nbody, 1./3.), 3, 999);
  //uniform_unit_cube(&body[0], Nbody, std::pow(Nbody, 1./3.), 3);
  buildTree(&cell[0], &body[0], Nbody, levels);

  std::mt19937 gen(999);
  std::uniform_real_distribution uniform_dist(0., 1.);
  std::generate(Xbody.begin(), Xbody.end(), 
    [&]() { return std::complex<double>(uniform_dist(gen), 0.); });

  /*cell.erase(cell.begin() + 1, cell.begin() + Nleaf - 1);
  cell[0].Child[0] = 1; cell[0].Child[1] = Nleaf + 1;
  ncells = Nleaf + 1;
  levels = 1;*/

  CSR cellNear('N', ncells, &cell[0], theta);
  CSR cellFar('F', ncells, &cell[0], theta);

  std::pair<double, double> timer(0, 0);
  std::vector<MPI_Comm> mpi_comms;
  std::vector<std::pair<long long, long long>> mapping(mpi_size, std::make_pair(0, 1));
  
  for (long long i = 0; i <= levels; i++) {
    cell_comm[i] = CellComm(&cell[0], &mapping[0], cellNear, cellFar, mpi_comms, world);
    cell_comm[i].timer = &timer;
  }

  std::vector<WellSeparatedApproximation> wsa(levels + 1);
  double h_construct_time = MPI_Wtime();

  for (long long l = 1; l <= levels; l++)
    wsa[l] = WellSeparatedApproximation(eval, epi, rank, cell_comm[l].oGlobal(), cell_comm[l].lenLocal(), &cell[0], cellFar, &body[0], wsa[l - 1]);
  
  h_construct_time = MPI_Wtime() - h_construct_time;
  MPI_Barrier(MPI_COMM_WORLD);
  double h2_construct_time = MPI_Wtime(), h2_construct_comm_time;

  basis[levels] = ClusterBasis(eval, epi, &cell[0], cellFar, &body[0], wsa[levels], cell_comm[levels], basis[levels], cell_comm[levels]);
  for (long long l = levels - 1; l >= 0; l--)
    basis[l] = ClusterBasis(eval, epi, &cell[0], cellFar, &body[0], wsa[l], cell_comm[l], basis[l + 1], cell_comm[l + 1]);

  MPI_Barrier(MPI_COMM_WORLD);
  h2_construct_time = MPI_Wtime() - h2_construct_time;
  h2_construct_comm_time = timer.first;
  timer.first = 0;

  long long llen = cell_comm[levels].lenLocal();
  long long gbegin = cell_comm[levels].oGlobal();
  long long body_local[2] = { cell[gbegin].Body[0], cell[gbegin + llen - 1].Body[1] };
  long long lenX = body_local[1] - body_local[0];
  std::vector<std::complex<double>> X1(lenX * nrhs, std::complex<double>(0., 0.));
  std::vector<std::complex<double>> X2(lenX * nrhs, std::complex<double>(0., 0.));

  MatVec mv(eval, &basis[0], &body[0], &cell[0], cellNear, &cell_comm[0], levels);
  for (long long i = 0; i < nrhs; i++)
    std::copy(&Xbody[i * Nbody] + body_local[0], &Xbody[i * Nbody] + body_local[1], &X1[i * lenX]);

  MPI_Barrier(MPI_COMM_WORLD);
  double matvec_time = MPI_Wtime(), matvec_comm_time;
  mv(nrhs, &X1[0]);

  MPI_Barrier(MPI_COMM_WORLD);
  matvec_time = MPI_Wtime() - matvec_time;
  matvec_comm_time = timer.first;
  timer.first = 0;

  double cerr = 0.;
  double refmatvec_time = MPI_Wtime();

  mat_vec_reference(eval, lenX, Nbody, nrhs, &X2[0], &Xbody[0], &body[body_local[0] * 3], &body[0]);
  refmatvec_time = MPI_Wtime() - refmatvec_time;

  solveRelErr(&cerr, &X1[0], &X2[0], lenX * nrhs);

  if (mpi_rank == 0) {
    std::cout << "Construct Err: " << cerr << std::endl;
    std::cout << "H-Matrix: " << h_construct_time << std::endl;
    std::cout << "H^2-Matrix: " << h2_construct_time << ", " << h2_construct_comm_time << std::endl;
    std::cout << "Matvec: " << matvec_time << ", " << matvec_comm_time << std::endl;
    std::cout << "Dense Matvec: " << refmatvec_time << std::endl;
  }

  for (MPI_Comm& c : mpi_comms)
    MPI_Comm_free(&c);
  MPI_Comm_free(&world);
  MPI_Finalize();
  return 0;
}
