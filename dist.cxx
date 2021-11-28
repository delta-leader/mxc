
#include "dist.hxx"

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
  int64_t my_offset = bodies.OFFSETS[my_ind * nboxes] * dim;
  const double* my_bodies = &bodies.BODIES[my_offset];
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
      int64_t rm_offset = bodies.OFFSETS[i * nboxes] * dim;
      double* rm_bodies = &bodies.BODIES[rm_offset];
      int64_t* rm_lens = &bodies.LENS[i * nboxes];
      
      MPI_Recv(rm_bodies, rm_nbody, MPI_DOUBLE, rm_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(rm_lens, nboxes, MPI_INT64_T, rm_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  for (int64_t i = 0; i < neighbors.size(); i++) {
    int64_t rm_rank = neighbors[i];
    if (rm_rank != my_rank) {
      MPI_Wait(&requests1[i], MPI_STATUS_IGNORE);
      MPI_Wait(&requests2[i], MPI_STATUS_IGNORE);
    }
  }

  bodies.OFFSETS[0] = 0;
  for (int64_t i = 1; i < bodies.OFFSETS.size(); i++)
    bodies.OFFSETS[i] = bodies.OFFSETS[i - 1] + bodies.LENS[i - 1];
}

void nbd::DistributeVectorsList(Vectors& lis, const GlobalIndex& gi) {
  int64_t my_ind = gi.SELF_I;
  int64_t my_rank = gi.COMM_RNKS[my_ind];
  int64_t nboxes = gi.BOXES;
  const std::vector<int64_t>& neighbors = gi.COMM_RNKS;
  std::vector<MPI_Request> requests(neighbors.size());

  std::vector<int64_t> LENS(neighbors.size());
  std::vector<double*> DATA(neighbors.size());

  for (int64_t i = 0; i < neighbors.size(); i++) {
    int64_t rm_rank = neighbors[i];
    int64_t tot_len = 0;
    for (int64_t n = 0; n < nboxes; n++) {
      int64_t rm_i = i * nboxes + n;
      const Vector& B_i = lis[rm_i];
      int64_t len = B_i.N;
      tot_len = tot_len + len;
    }
    LENS[i] = tot_len;
    DATA[i] = (double*)malloc(sizeof(double) * tot_len);
  }

  int64_t offset = 0;
  double* my_data = DATA[my_ind];
  int64_t my_len = LENS[my_ind];
  for (int64_t n = 0; n < nboxes; n++) {
    int64_t my_i = my_ind * nboxes + n;
    const Vector& B_i = lis[my_i];
    int64_t len = B_i.N;
    cpyFromVector(B_i, my_data + offset);
    offset = offset + len;
  }

  for (int64_t i = 0; i < neighbors.size(); i++) {
    int64_t rm_rank = neighbors[i];
    if (rm_rank != my_rank)
      MPI_Isend(my_data, my_len, MPI_DOUBLE, rm_rank, 0, MPI_COMM_WORLD, &requests[i]);
  }

  for (int64_t i = 0; i < neighbors.size(); i++) {
    int64_t rm_rank = neighbors[i];
    if (rm_rank != my_rank) {
      double* data = DATA[i];
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
    if (rm_rank != my_rank) {
      offset = 0;
      const double* rm_v = DATA[i];
      for (int64_t n = 0; n < nboxes; n++) {
        int64_t rm_i = i * nboxes + n;
        Vector& B_i = lis[rm_i];
        int64_t len = B_i.N;
        vaxpby(B_i, rm_v + offset, 1., 0.);
        offset = offset + len;
      }
    }
  }

  for (int64_t i = 0; i < neighbors.size(); i++)
    free(DATA[i]);
}

void nbd::DistributeMatricesList(Matrices& lis, const GlobalIndex& gi) {
  int64_t my_ind = gi.SELF_I;
  int64_t my_rank = gi.COMM_RNKS[my_ind];
  int64_t nboxes = gi.BOXES;
  const std::vector<int64_t>& neighbors = gi.COMM_RNKS;
  std::vector<MPI_Request> requests(neighbors.size());

  std::vector<int64_t> LENS(neighbors.size());
  std::vector<double*> DATA(neighbors.size());

  for (int64_t i = 0; i < neighbors.size(); i++) {
    int64_t rm_rank = neighbors[i];
    int64_t tot_len = 0;
    for (int64_t n = 0; n < nboxes; n++) {
      int64_t rm_i = i * nboxes + n;
      const Matrix& A_i = lis[rm_i];
      int64_t len = A_i.M * A_i.N;
      tot_len = tot_len + len;
    }
    LENS[i] = tot_len;
    DATA[i] = (double*)malloc(sizeof(double) * tot_len);
  }

  int64_t offset = 0;
  double* my_data = DATA[my_ind];
  int64_t my_len = LENS[my_ind];
  for (int64_t n = 0; n < nboxes; n++) {
    int64_t my_i = my_ind * nboxes + n;
    const Matrix& A_i = lis[my_i];
    int64_t len = A_i.M * A_i.N;
    cpyFromMatrix('N', A_i, my_data + offset);
    offset = offset + len;
  }

  for (int64_t i = 0; i < neighbors.size(); i++) {
    int64_t rm_rank = neighbors[i];
    if (rm_rank != my_rank)
      MPI_Isend(my_data, my_len, MPI_DOUBLE, rm_rank, 0, MPI_COMM_WORLD, &requests[i]);
  }

  for (int64_t i = 0; i < neighbors.size(); i++) {
    int64_t rm_rank = neighbors[i];
    if (rm_rank != my_rank) {
      double* data = DATA[i];
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
    if (rm_rank != my_rank) {
      offset = 0;
      const double* rm_v = DATA[i];
      for (int64_t n = 0; n < nboxes; n++) {
        int64_t rm_i = i * nboxes + n;
        Matrix& A_i = lis[rm_i];
        int64_t len = A_i.M * A_i.N;
        maxpby(A_i, rm_v + offset, 1., 0.);
        offset = offset + len;
      }
    }
  }

  for (int64_t i = 0; i < neighbors.size(); i++)
    free(DATA[i]);
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
        maxpby(A_ji, rmv_i, 1., 1.);
        offset = offset + len;
      }
    }
}

void nbd::axatDistribute(Matrices& A, const GlobalIndex& gi) {
  int64_t my_ind = gi.SELF_I;
  int64_t my_rank = gi.COMM_RNKS[my_ind];
  const std::vector<int64_t>& neighbors = gi.COMM_RNKS;
  std::vector<MPI_Request> requests(neighbors.size());

  std::vector<int64_t> LENS(neighbors.size());
  std::vector<double*> SRC_DATA(neighbors.size());
  std::vector<double*> RM_DATA(neighbors.size());

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


void nbd::butterflySumA(Matrices& A, const GlobalIndex& gi) {
  int64_t my_ind = gi.SELF_I;
  int64_t my_twi = gi.TWIN_I;
  int64_t my_rank = gi.COMM_RNKS[my_ind];
  int64_t rm_rank = gi.COMM_RNKS[my_twi];

  MPI_Request request;
  int64_t LEN = 0;
  double* SRC_DATA, *RM_DATA;

  for (int64_t i = 0; i < A.size(); i++) {
    const Matrix& A_i = A[i];
    int64_t len = A_i.M * A_i.N;
    LEN = LEN + len;
  }

  SRC_DATA = (double*)malloc(sizeof(double) * LEN);
  RM_DATA = (double*)malloc(sizeof(double) * LEN);

  int64_t offset = 0;
  for (int64_t i = 0; i < A.size(); i++) {
    const Matrix& A_i = A[i];
    int64_t len = A_i.M * A_i.N;
    cpyFromMatrix('N', A_i, SRC_DATA + offset);
    offset = offset + len;
  }

  MPI_Isend(SRC_DATA, LEN, MPI_DOUBLE, rm_rank, 0, MPI_COMM_WORLD, &request);
  MPI_Recv(RM_DATA, LEN, MPI_DOUBLE, rm_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Wait(&request, MPI_STATUS_IGNORE);

  offset = 0;
  for (int64_t i = 0; i < A.size(); i++) {
    Matrix& A_i = A[i];
    int64_t len = A_i.M * A_i.N;
    maxpby(A_i, RM_DATA + offset, 1., 1.);
    offset = offset + len;
  }

  free(SRC_DATA);
  free(RM_DATA);
}


void nbd::sendSubstituted(char fwbk, const Vectors& X, const GlobalIndex& gi) {
  int64_t my_ind = gi.SELF_I;
  int64_t my_rank = gi.COMM_RNKS[my_ind];
  int64_t LEN = 0;
  std::vector<int64_t> neighbors;

  int64_t lbegin = my_ind * gi.BOXES;
  int64_t lend = lbegin + gi.BOXES;
  for (int64_t i = lbegin; i < lend; i++)
    LEN = LEN + X[i].N;
  double* DATA = (double*)malloc(sizeof(double) * LEN);

  LEN = 0;
  for (int64_t i = lbegin; i < lend; i++) {
    cpyFromVector(X[i], DATA + LEN);
    LEN = LEN + X[i].N;
  }

  neighbors.reserve(gi.COMM_RNKS.size());
  if (fwbk == 'F' || fwbk == 'f')
    for (int64_t i = 0; i < gi.COMM_RNKS.size(); i++)
      if (gi.COMM_RNKS[i] > my_rank)
        neighbors.emplace_back(gi.COMM_RNKS[i]);
  if (fwbk == 'B' || fwbk == 'b')
    for (int64_t i = 0; i < gi.COMM_RNKS.size(); i++)
      if (gi.COMM_RNKS[i] < my_rank)
        neighbors.emplace_back(gi.COMM_RNKS[i]);
  
  for (int64_t i = 0; i < neighbors.size(); i++) {
    int64_t rm_rank = neighbors[i];
    MPI_Send(DATA, LEN, MPI_DOUBLE, rm_rank, 0, MPI_COMM_WORLD);
  }

  free(DATA);
}

void constructCOMM_SUBST(double** DATA, int64_t* LEN, int64_t RM_BOX, const Vectors& X, const GlobalIndex& gi) {
  int64_t nboxes = gi.BOXES;
  int64_t lbegin = RM_BOX * nboxes;
  int64_t lend = lbegin + nboxes;

  int64_t len = 0;
  for (int64_t i = lbegin; i < lend; i++)
    len = len + X[i].N;

  double* data = (double*)malloc(sizeof(double) * len);
  len = 0;
  for (int64_t i = lbegin; i < lend; i++) {
    cpyFromVector(X[i], data + len);
    len = len + X[i].N;
  }
  
  *LEN = len;
  *DATA = data;
}

void substRemoteV(int64_t RM_BOX, Vectors& X, const GlobalIndex& gi, const double* rmv) {
  int64_t nboxes = gi.BOXES;
  int64_t lbegin = RM_BOX * nboxes;
  int64_t lend = lbegin + nboxes;

  int64_t len = 0;
  for (int64_t i = lbegin; i < lend; i++) {
    vaxpby(X[i], rmv + len, 1., 0.);
    len = len + X[i].N;
  }
}

void nbd::recvSubstituted(char fwbk, Vectors& X, const GlobalIndex& gi) {
  int64_t my_ind = gi.SELF_I;
  int64_t my_rank = gi.COMM_RNKS[my_ind];
  const std::vector<int64_t>& neighbors = gi.COMM_RNKS;

  std::vector<MPI_Request> requests(neighbors.size());
  std::vector<double*> DATA(neighbors.size());
  std::vector<int64_t> LENS(neighbors.size());

  if (fwbk == 'F' || fwbk == 'f') {
    for (int64_t i = 0; i < neighbors.size(); i++) {
      int64_t rm_rank = neighbors[i];
      if (rm_rank > my_rank) {
        constructCOMM_SUBST(&DATA[i], &LENS[i], gi.NGB_RNKS[i], X, gi);
        MPI_Irecv(DATA[i], LENS[i], MPI_DOUBLE, rm_rank, 0, MPI_COMM_WORLD, &requests[i]);
      }
    }

    for (int64_t i = 0; i < neighbors.size(); i++) {
      int64_t rm_rank = neighbors[i];
      if (rm_rank > my_rank)
        MPI_Wait(&requests[i], MPI_STATUS_IGNORE);
    }

    for (int64_t i = 0; i < neighbors.size(); i++) {
      int64_t rm_rank = neighbors[i];
      if (rm_rank > my_rank) {
        substRemoteV(gi.NGB_RNKS[i], X, gi, DATA[i]);
        free(DATA[i]);
      }
    }
  }
  else if (fwbk == 'B' || fwbk == 'b') {
    for (int64_t i = 0; i < neighbors.size(); i++) {
      int64_t rm_rank = neighbors[i];
      if (rm_rank < my_rank) {
        constructCOMM_SUBST(&DATA[i], &LENS[i], gi.NGB_RNKS[i], X, gi);
        MPI_Irecv(DATA[i], LENS[i], MPI_DOUBLE, rm_rank, 0, MPI_COMM_WORLD, &requests[i]);
      }
    }

    for (int64_t i = 0; i < neighbors.size(); i++) {
      int64_t rm_rank = neighbors[i];
      if (rm_rank < my_rank)
        MPI_Wait(&requests[i], MPI_STATUS_IGNORE);
    }

    for (int64_t i = 0; i < neighbors.size(); i++) {
      int64_t rm_rank = neighbors[i];
      if (rm_rank < my_rank) {
        substRemoteV(gi.NGB_RNKS[i], X, gi, DATA[i]);
        free(DATA[i]);
      }
    }
  }
}

void nbd::distributeSubstituted(Vectors& X, const GlobalIndex& gi) {
  int64_t my_ind = gi.SELF_I;
  int64_t my_rank = gi.COMM_RNKS[my_ind];
  const std::vector<int64_t>& neighbors = gi.COMM_RNKS;
  std::vector<MPI_Request> requests(neighbors.size());

  std::vector<int64_t> LENS(neighbors.size());
  std::vector<double*> SRC_DATA(neighbors.size());
  std::vector<double*> RM_DATA(neighbors.size());

  int64_t LEN = 0;
  int64_t lbegin = my_ind * gi.BOXES;
  int64_t lend = lbegin + gi.BOXES;
  for (int64_t i = lbegin; i < lend; i++)
    LEN = LEN + X[i].N;

  for (int64_t i = 0; i < neighbors.size(); i++) {
    int64_t rm_rank = neighbors[i];
    if (rm_rank != my_rank) {
      constructCOMM_SUBST(&SRC_DATA[i], &LENS[i], gi.NGB_RNKS[i], X, gi);
      RM_DATA[i] = (double*)malloc(sizeof(double) * LEN);
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
      MPI_Recv(data, LEN, MPI_DOUBLE, rm_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
      substRemoteV(gi.NGB_RNKS[i], X, gi, RM_DATA[i]);
  }

  for (int64_t i = 0; i < neighbors.size(); i++) {
    int64_t rm_rank = neighbors[i];
    if (rm_rank != my_rank) {
      free(SRC_DATA[i]);
      free(RM_DATA[i]);
    }
  }
}

void nbd::butterflySumX(Vectors& X, const GlobalIndex& gi) {
  int64_t my_ind = gi.SELF_I;
  int64_t my_twi = gi.TWIN_I;
  int64_t my_rank = gi.COMM_RNKS[my_ind];
  int64_t rm_rank = gi.COMM_RNKS[my_twi];

  MPI_Request request;
  int64_t LEN = 0;
  double* SRC_DATA, *RM_DATA;

  for (int64_t i = 0; i < X.size(); i++)
    LEN = LEN + X[i].N;

  SRC_DATA = (double*)malloc(sizeof(double) * LEN);
  RM_DATA = (double*)malloc(sizeof(double) * LEN);

  int64_t offset = 0;
  for (int64_t i = 0; i < X.size(); i++) {
    cpyFromVector(X[i], SRC_DATA + offset);
    offset = offset + X[i].N;
  }

  MPI_Isend(SRC_DATA, LEN, MPI_DOUBLE, rm_rank, 0, MPI_COMM_WORLD, &request);
  MPI_Recv(RM_DATA, LEN, MPI_DOUBLE, rm_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Wait(&request, MPI_STATUS_IGNORE);

  offset = 0;
  for (int64_t i = 0; i < X.size(); i++) {
    vaxpby(X[i], RM_DATA + offset, 1., 1.);
    offset = offset + X[i].N;
  }

  free(SRC_DATA);
  free(RM_DATA);
}
