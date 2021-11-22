

#include "bodies.hxx"
#include "sps_basis.hxx"
#include "sps_umv.hxx"

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

  Global_Partition(domain, Nbody, Ncrit, dim, 0., 1.);
  Local_Partition(local, domain, mpi_rank, mpi_size, theta);

  //if(mpi_rank == 0) for(auto& gi : local.MY_IDS) printGlobalI(gi);

  Random_bodies(bodies, domain, local, 100 ^ mpi_rank);
  DistributeBodies(bodies, local.MY_IDS.back());

  //checkBodies(domain, local, bodies);

  Nodes nodes;
  Matrices* A = allocNodes(nodes, local);
  BlockCSC(*A, l2d(), local, bodies);

  Basis basis;
  int64_t* LeafD = allocBasis(basis, local);
  std::copy(bodies.LENS.begin(), bodies.LENS.end(), LeafD);

  Base& base = basis.back();
  sampleA(base, 1.e-8, local.MY_IDS.back(), *A, R.data(), R.size());

  //checkBasis(base);

  MPI_Finalize();
  return 0;
}
