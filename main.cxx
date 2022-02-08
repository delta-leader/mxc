

#include "bodies.hxx"
#include "solver.hxx"
#include "dist.hxx"
#include "minblas.h"

#include "mpi.h"
#include <random>
#include <cstdio>
#include <cmath>

using namespace nbd;

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);
  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  double ptime, ftime;
  if (mpi_rank == 0) startTimer(&ptime);

  int64_t Nbody = 40000;
  int64_t Ncrit = 100;
  int64_t theta = 1;
  int64_t dim = 2;
  EvalFunc ef = dim == 2 ? l2d() : l3d();

  std::srand(100);
  std::vector<double> R(1 << 16);
  for (int64_t i = 0; i < R.size(); i++)
    R[i] = -1. + 2. * ((double)std::rand() / RAND_MAX);

  GlobalDomain domain;
  LocalDomain local;
  LocalBodies bodies;

  Global_Partition(domain, mpi_rank, mpi_size, Nbody, Ncrit, dim, 0., std::pow(Nbody, 1. / dim));
  GlobalIndex* leaf = Local_Partition(local, domain, theta);

  Random_bodies(bodies, domain, *leaf, std::pow(987, mpi_rank));

  Nodes nodes;
  Matrices* A = allocNodes(nodes, local);
  BlockCSC(*A, ef, *leaf, bodies);

  Basis basis;
  allocBasis(basis, local, bodies.LENS.data());

  MPI_Barrier(MPI_COMM_WORLD);
  if (mpi_rank == 0) startTimer(&ftime);
  factorA(nodes, basis, local, 1.e-6, R.data(), R.size());

  MPI_Barrier(MPI_COMM_WORLD);
  if (mpi_rank == 0) stopTimer(ftime, "factor");

  Vectors X;
  Vector* Xlocal = randomVectors(X, *leaf, bodies, -1., 1., std::pow(654, mpi_rank));

  RHSS rhs;
  Vector* B = allocRightHandSides(rhs, basis, local);
  blockAxEb(B, ef, X, *leaf, bodies);

  MPI_Barrier(MPI_COMM_WORLD);
  if (mpi_rank == 0) startTimer(&ftime);
  solveA(rhs, nodes, basis, local);

  MPI_Barrier(MPI_COMM_WORLD);
  if (mpi_rank == 0) stopTimer(ftime, "solve");
  double err;
  solveRelErr(&err, rhs.back(), X, *leaf);
  printf("%d ERR: %e\n", mpi_rank, err);

  if (mpi_rank == 0) stopTimer(ptime, "program");

  int64_t* flops = getFLOPS();
  double gf = flops[0] * 1.e-9;
  printf("%d GFLOPS: %f\n", mpi_rank, gf);
  MPI_Finalize();
  return 0;
}
