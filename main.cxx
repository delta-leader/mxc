

#include "bodies.hxx"
#include "solver.hxx"

#include "timer.h"
#include "mpi.h"
#include <random>
#include <cstdio>
#include <cmath>

using namespace nbd;


int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);
  int dim = 2;
  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  if (mpi_rank == 0) start("program");

  int64_t Nbody = 40000;
  int64_t Ncrit = 100;
  int64_t theta = 1;

  std::srand(100);
  std::vector<double> R(1 << 18);
  for (int64_t i = 0; i < R.size(); i++)
    R[i] = -1. + 2. * ((double)std::rand() / RAND_MAX);

  GlobalDomain domain;
  LocalDomain local;
  LocalBodies bodies;

  Global_Partition(domain, mpi_rank, mpi_size, Nbody, Ncrit, dim, 0., std::pow(Nbody, 1. / dim));
  GlobalIndex* leaf = Local_Partition(local, domain, theta);

  //if(mpi_rank == 0) for(auto& gi : local) printGlobalI(gi);

  Random_bodies(bodies, domain, *leaf, std::pow(999, mpi_rank));

  checkBodies(mpi_rank, domain, *leaf, bodies);

  EvalFunc ef = l2d();

  Nodes nodes;
  Matrices* A = allocNodes(nodes, local);
  BlockCSC(*A, ef, *leaf, bodies);

  Basis basis;
  allocBasis(basis, local, bodies.LENS.data());

  factorA(nodes, basis, local, 1.e-8, R.data(), R.size());

  Vectors X;
  Vector* Xlocal = randomVectors(X, *leaf, bodies, -1., 1., std::pow(999, mpi_rank));

  RHSS rhs;
  Vector* B = allocRightHandSides(rhs, basis, local);
  blockAxEb(B, ef, X, *leaf, bodies);

  solveA(rhs, nodes, basis, local);
  double err;
  solveRelErr(&err, rhs.back(), X, *leaf);
  printf("ERR: %e\n", err);

  MPI_Finalize();
  if (mpi_rank == 0) stop("program");
  return 0;
}
