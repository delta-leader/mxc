

#include "bodies.hxx"
#include "basis.hxx"
#include "umv.hxx"

#include "timer.h"
#include "mpi.h"
#include <random>
#include <cstdio>

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
    R[i] = -1 + 2 * ((double)std::rand() / RAND_MAX);

  GlobalDomain domain;
  LocalDomain local;
  LocalBodies bodies;

  Global_Partition(domain, mpi_rank, mpi_size, Nbody, Ncrit, dim, 0., 1.);
  GlobalIndex* leaf = Local_Partition(local, domain, theta);

  //if(mpi_rank == 0) for(auto& gi : local) printGlobalI(gi);

  Random_bodies(bodies, domain, *leaf, 100 ^ mpi_rank);

  checkBodies(mpi_rank, domain, *leaf, bodies);

  Nodes nodes;
  Matrices* A = allocNodes(nodes, local);
  BlockCSC(*A, l2d(), *leaf, bodies);

  Basis basis;
  allocBasis(basis, local, bodies.LENS.data());

  Vectors X, B;
  Vector* Xlocal = randomVectors(X, *leaf, bodies, -1., 1., 100 ^ mpi_rank);
  blockAxEb(B, l2d(), X, *leaf, bodies);

  MPI_Finalize();
  if (mpi_rank == 0) stop("program");
  return 0;
}
