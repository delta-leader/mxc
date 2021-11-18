

#include "build_tree.hxx"
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

  int64_t Nbody = 50000;
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

  Random_bodies(bodies, domain, local, 100 ^ mpi_rank);
  DistributeBodies(bodies);

  CSC rels;
  Matrices A;
  BlockCSC(A, rels, l2d(), local, bodies);

  printf("%d: %ld %ld %ld\n", mpi_rank, rels.M, rels.N, A.size());

  //Base base;
  //local_row_base(base, 1.e-8, A, R.data(), rels, R.size());

  //for (auto& u : base.Uo)
  //  printf("%ld %ld\n", u.M, u.N);

  MPI_Finalize();
  return 0;
}
