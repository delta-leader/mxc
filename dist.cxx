
#include "bodies.hxx"

#include "mpi.h"
#include <cstdlib>
#include <algorithm>

using namespace nbd;

void nbd::DistributeBodies(LocalBodies& bodies, const GlobalIndex& gi) {
  int64_t my_ind = gi.SELF_I;
  int64_t my_rank = gi.NGB_RNKS[my_ind];
  int64_t nboxes = gi.BOXES;
  int64_t dim = bodies.DIM;
  const std::vector<int64_t>& neighbors = gi.NGB_RNKS;

  int64_t my_nbody = bodies.NBODIES[my_ind] * dim;
  int64_t my_offset = bodies.OFFSETS[my_ind * nboxes];
  const double* my_bodies = &bodies.BODIES[my_offset * dim];
  const int64_t* my_lens = &bodies.LENS[my_ind * nboxes];

  std::vector<MPI_Request> requests1(neighbors.size());
  std::vector<MPI_Request> requests2(neighbors.size());

  for (int64_t i = 0; i < neighbors.size(); i++) {
    int64_t rm_rank = neighbors[i];
    if (rm_rank != my_rank) {
      MPI_Isend(my_bodies, my_nbody, MPI_DOUBLE, rm_rank, 0, MPI_COMM_WORLD, &requests1[i]);
      MPI_Isend(my_lens, nboxes, MPI_INT64_T, rm_rank, 0, MPI_COMM_WORLD, &requests2[i]);
    }
  }

  for (int64_t i = 0; i < neighbors.size(); i++) {
    int64_t rm_rank = neighbors[i];
    if (rm_rank != my_rank) {
      int64_t rm_nbody = bodies.NBODIES[i] * dim;
      int64_t rm_offset = bodies.OFFSETS[i];
      double* rm_bodies = &bodies.BODIES[rm_offset * dim];
      int64_t* rm_lens = &bodies.LENS[i * nboxes];
      
      MPI_Recv(rm_bodies, rm_nbody, MPI_DOUBLE, rm_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(rm_lens, nboxes, MPI_INT64_T, rm_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
  
  for (int64_t i = 1; i < bodies.OFFSETS.size(); i++)
    bodies.OFFSETS[i] = bodies.OFFSETS[i - 1] + bodies.LENS[i - 1];

  for (int64_t i = 0; i < neighbors.size(); i++) {
    int64_t rm_rank = neighbors[i];
    if (rm_rank != my_rank) {
      MPI_Wait(&requests1[i], MPI_STATUS_IGNORE);
      MPI_Wait(&requests2[i], MPI_STATUS_IGNORE);
    }
  }
}
