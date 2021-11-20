
#include "bodies.hxx"
#include "sps_basis.hxx"
#include "sps_umv.hxx"

#include "mpi.h"
#include <cstdlib>
#include <algorithm>

using namespace nbd;

void nbd::DistributeBodies(LocalBodies& bodies, const GlobalIndex& gi) {
  int64_t my_ind = gi.SELF_I;
  int64_t my_rank = gi.COMM_RNKS[my_ind];
  int64_t nboxes = gi.BOXES;
  int64_t dim = bodies.DIM;
  const std::vector<int64_t>& neighbors = gi.COMM_RNKS;

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


void nbd::DistributeMatricesList(Matrices& lis, const GlobalIndex& gi) {
  int64_t my_ind = gi.SELF_I;
  int64_t my_rank = gi.COMM_RNKS[my_ind];
  int64_t nboxes = gi.BOXES;
  const std::vector<int64_t>& neighbors = gi.COMM_RNKS;
  std::vector<MPI_Request> requests(neighbors.size() * nboxes);

  for (int64_t i = 0; i < neighbors.size(); i++) {
    int64_t rm_rank = neighbors[i];
    if (rm_rank != my_rank)
      for (int64_t n = 0; n < nboxes; n++) {
        int64_t rm_i = i * nboxes + n;
        int64_t my_i = my_ind * nboxes + n;
        const double* data = lis[my_i].A.data();
        int64_t len = lis[my_i].M * lis[my_i].N;
        MPI_Isend(data, len, MPI_DOUBLE, rm_rank, 0, MPI_COMM_WORLD, &requests[rm_i]);
      }
  }

  for (int64_t i = 0; i < neighbors.size(); i++) {
    int64_t rm_rank = neighbors[i];
    if (rm_rank != my_rank)
      for (int64_t n = 0; n < nboxes; n++) {
        int64_t rm_i = i * nboxes + n;
        double* data = lis[rm_i].A.data();
        int64_t len = lis[rm_i].M * lis[rm_i].N;
        MPI_Recv(data, len, MPI_DOUBLE, rm_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
  }

  for (int64_t i = 0; i < neighbors.size(); i++) {
    int64_t rm_rank = neighbors[i];
    if (rm_rank != my_rank)
      for (int64_t n = 0; n < nboxes; n++) {
        int64_t rm_i = i * nboxes + n;
        MPI_Wait(&requests[rm_i], MPI_STATUS_IGNORE);
      }
  }
}


void nbd::DistributeDims(std::vector<int64_t>& dims, const GlobalIndex& gi) {
  int64_t my_ind = gi.SELF_I;
  int64_t my_rank = gi.COMM_RNKS[my_ind];
  int64_t nboxes = gi.BOXES;
  const std::vector<int64_t>& neighbors = gi.COMM_RNKS;
  std::vector<MPI_Request> requests(neighbors.size());

  const int64_t* my_data = &dims[my_ind * nboxes];

  for (int64_t i = 0; i < neighbors.size(); i++) {
    int64_t rm_rank = neighbors[i];
    if (rm_rank != my_rank)
      MPI_Isend(my_data, nboxes, MPI_INT64_T, rm_rank, 0, MPI_COMM_WORLD, &requests[i]);
  }

  for (int64_t i = 0; i < neighbors.size(); i++) {
    int64_t rm_rank = neighbors[i];
    if (rm_rank != my_rank) {
      int64_t* data = &dims[i * nboxes];
      MPI_Recv(data, nboxes, MPI_INT64_T, rm_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  for (int64_t i = 0; i < neighbors.size(); i++) {
    int64_t rm_rank = neighbors[i];
    if (rm_rank != my_rank)
      MPI_Wait(&requests[i], MPI_STATUS_IGNORE);
  }
}

void constructCOMM_AXAT(double** DATA, int64_t* LEN, int64_t RM_BOX, const Matrices& A, const GlobalIndex& gi) {
  int64_t nboxes = gi.BOXES;
  const CSC& rels = gi.RELS;
  std::vector<int64_t> lens(nboxes);
  std::fill(lens.begin(), lens.end(), 0);

  for (int64_t i = 0; i < rels.N; i++)
    for (int64_t ji = rels.CSC_COLS[i]; ji < rels.CSC_COLS[i + 1]; ji++) {
      int64_t j = rels.CSC_ROWS[ji];
      int64_t rm_box = j / nboxes;
      if (rm_box == RM_BOX) {
        const Matrix& A_ji = A[ji];
        int64_t len = A_ji.M * A_ji.N;
        int64_t box_i = j - rm_box * nboxes;
        lens[box_i] = lens[box_i] + len;
      }
    }

  std::vector<int64_t> offsets(nboxes + 1);
  offsets[0] = 0;
  for (int64_t i = 1; i <= nboxes; i++)
    offsets[i] = offsets[i - 1] + lens[i - 1];

  int64_t tot_len = offsets[nboxes];
  double* data = (double*)malloc(sizeof(double) * tot_len);
  
  for (int64_t i = 0; i < rels.N; i++)
    for (int64_t ji = rels.CSC_COLS[i]; ji < rels.CSC_COLS[i + 1]; ji++) {
      int64_t j = rels.CSC_ROWS[ji];
      int64_t rm_box = j / nboxes;
      if (rm_box == RM_BOX) {
        const Matrix& A_ji = A[ji];
        int64_t len = A_ji.M * A_ji.N;
        int64_t box_i = j - rm_box * nboxes;
        double* tar = data + offsets[box_i];
        cpyFromMatrix('T', A_ji, tar);
        offsets[box_i] = offsets[box_i] + len;
      }
    }

  *LEN = tot_len;
  *DATA = data;
}

void axRemoteV(int64_t RM_BOX, Matrices& A, const GlobalIndex& gi, const double* rmv) {
  int64_t nboxes = gi.BOXES;
  const CSC& rels = gi.RELS;

  int64_t offset = 0;
  for (int64_t i = 0; i < rels.N; i++)
    for (int64_t ji = rels.CSC_COLS[i]; ji < rels.CSC_COLS[i + 1]; ji++) {
      int64_t j = rels.CSC_ROWS[ji];
      int64_t rm_box = j / nboxes;
      if (rm_box == RM_BOX) {
        Matrix& A_ji = A[ji];
        int64_t len = A_ji.M * A_ji.N;
        const double* rmv_i = rmv + offset;
        maxpy(A_ji, rmv_i, 1.);
        offset = offset + len;
      }
    }
}

void nbd::axatDistribute(Matrices& A, const GlobalIndex& gi) {
  int64_t my_ind = gi.SELF_I;
  int64_t my_rank = gi.COMM_RNKS[my_ind];
  const std::vector<int64_t>& neighbors = gi.COMM_RNKS;
  std::vector<MPI_Request> requests(neighbors.size());

  std::vector<int64_t> LENS;
  std::vector<double*> SRC_DATA;
  std::vector<double*> RM_DATA;

  LENS.resize(neighbors.size());
  SRC_DATA.resize(neighbors.size());
  RM_DATA.resize(neighbors.size());

  for (int64_t i = 0; i < neighbors.size(); i++) {
    int64_t rm_rank = neighbors[i];
    if (rm_rank != my_rank) {
      constructCOMM_AXAT(&SRC_DATA[i], &LENS[i], gi.NGB_RNKS[i], A, gi);
      RM_DATA[i] = (double*)malloc(sizeof(double) * LENS[i]);
    }
  }

  for (int64_t i = 0; i < neighbors.size(); i++) {
    int64_t rm_rank = neighbors[i];
    if (rm_rank != my_rank) {
      const double* my_data = SRC_DATA[i];
      int64_t my_len = LENS[i];
      MPI_Isend(my_data, my_len, MPI_DOUBLE, rm_rank, 0, MPI_COMM_WORLD, &requests[i]);
    }
  }

  for (int64_t i = 0; i < neighbors.size(); i++) {
    int64_t rm_rank = neighbors[i];
    if (rm_rank != my_rank) {
      double* data = RM_DATA[i];
      int64_t len = LENS[i];
      MPI_Recv(data, len, MPI_DOUBLE, rm_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  for (int64_t i = 0; i < neighbors.size(); i++) {
    int64_t rm_rank = neighbors[i];
    if (rm_rank != my_rank)
      MPI_Wait(&requests[i], MPI_STATUS_IGNORE);
  }

  for (int64_t i = 0; i < neighbors.size(); i++) {
    int64_t rm_rank = neighbors[i];
    if (rm_rank != my_rank)
      axRemoteV(gi.NGB_RNKS[i], A, gi, RM_DATA[i]);
  }

  for (int64_t i = 0; i < neighbors.size(); i++) {
    int64_t rm_rank = neighbors[i];
    if (rm_rank != my_rank) {
      free(SRC_DATA[i]);
      free(RM_DATA[i]);
    }
  }
}
