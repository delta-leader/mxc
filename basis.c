
#include "nbd.h"
#include "profile.h"

#include "stdlib.h"
#include "math.h"
#include "string.h"

void dist_int_64_xlen(int64_t arr_xlen[], const struct CellComm* comm) {
  int64_t plen = comm->Proc[0] == comm->worldRank ? comm->lenTargets : 0;
  const int64_t* row = comm->ProcTargets;
  int64_t lbegin = 0;
#ifdef _PROF
  double stime = MPI_Wtime();
#endif
  for (int64_t i = 0; i < plen; i++) {
    int64_t p = row[i];
    int64_t len = comm->ProcBoxesEnd[p] - comm->ProcBoxes[p];
    MPI_Bcast(&arr_xlen[lbegin], len, MPI_INT64_T, comm->ProcRootI[p], comm->Comm_box[p]);
    lbegin = lbegin + len;
  }

  int64_t xlen = 0;
  content_length(&xlen, comm);
  if (comm->Proc[1] - comm->Proc[0] > 1)
    MPI_Bcast(arr_xlen, xlen, MPI_INT64_T, 0, comm->Comm_share);
#ifdef _PROF
  double etime = MPI_Wtime() - stime;
  recordCommTime(etime);
#endif
}

void dist_int_64(int64_t arr[], const int64_t offsets[], const struct CellComm* comm) {
  int64_t plen = comm->Proc[0] == comm->worldRank ? comm->lenTargets : 0;
  const int64_t* row = comm->ProcTargets;
  int64_t lbegin = 0;
#ifdef _PROF
  double stime = MPI_Wtime();
#endif
  for (int64_t i = 0; i < plen; i++) {
    int64_t p = row[i];
    int64_t llen = comm->ProcBoxesEnd[p] - comm->ProcBoxes[p];
    int64_t offset = offsets[lbegin];
    int64_t len = offsets[lbegin + llen] - offset;
    MPI_Bcast(&arr[offset], len, MPI_INT64_T, comm->ProcRootI[p], comm->Comm_box[p]);
    lbegin = lbegin + llen;
  }

  int64_t xlen = 0;
  content_length(&xlen, comm);
  int64_t alen = offsets[xlen];
  if (comm->Proc[1] - comm->Proc[0] > 1)
    MPI_Bcast(arr, alen, MPI_INT64_T, 0, comm->Comm_share);
#ifdef _PROF
  double etime = MPI_Wtime() - stime;
  recordCommTime(etime);
#endif
}

void dist_double(double* arr[], const struct CellComm* comm) {
  int64_t plen = comm->Proc[0] == comm->worldRank ? comm->lenTargets : 0;
  const int64_t* row = comm->ProcTargets;
  double* data = arr[0];
  int64_t lbegin = 0;
#ifdef _PROF
  double stime = MPI_Wtime();
#endif
  for (int64_t i = 0; i < plen; i++) {
    int64_t p = row[i];
    int64_t llen = comm->ProcBoxesEnd[p] - comm->ProcBoxes[p];
    int64_t offset = arr[lbegin] - data;
    int64_t len = arr[lbegin + llen] - arr[lbegin];
    MPI_Bcast(&data[offset], len, MPI_DOUBLE, comm->ProcRootI[p], comm->Comm_box[p]);
    lbegin = lbegin + llen;
  }

  int64_t xlen = 0;
  content_length(&xlen, comm);
  int64_t alen = arr[xlen] - data;
  if (comm->Proc[1] - comm->Proc[0] > 1)
    MPI_Bcast(data, alen, MPI_DOUBLE, 0, comm->Comm_share);
#ifdef _PROF
  double etime = MPI_Wtime() - stime;
  recordCommTime(etime);
#endif
}

void buildBasis(void(*ef)(double*), struct Base basis[], int64_t ncells, const struct Cell* cells, int64_t levels, const struct CellComm* comm, const double* bodies, double epi, int64_t mrank, int64_t sp_pts) {
  int64_t nbodies = cells[0].Body[1];
  for (int64_t l = levels; l >= 0; l--) {
    int64_t xlen = 0;
    content_length(&xlen, &comm[l]);
    basis[l].Ulen = xlen;
    int64_t* arr_i = (int64_t*)malloc(sizeof(int64_t) * (xlen * 4 + 1));
    basis[l].Lchild = arr_i;
    basis[l].Dims = &arr_i[xlen];
    basis[l].DimsLr = &arr_i[xlen * 2];
    basis[l].Offsets = &arr_i[xlen * 3];

    struct Matrix* arr_m = (struct Matrix*)calloc(xlen * 3, sizeof(struct Matrix));
    basis[l].Uo = arr_m;
    basis[l].Uc = &arr_m[xlen];
    basis[l].R = &arr_m[xlen * 2];

    int64_t jbegin = 0, jend = ncells;
    get_level(&jbegin, &jend, cells, l, -1);
    for (int64_t i = jbegin; i < jend; i++) {
      int64_t cc = cells[i].Child;
      int64_t li = cells[i].LID;
      if (li >= 0) {
        int64_t lc = -1;
        int64_t ske_len = cells[i].Body[1] - cells[i].Body[0];
        if (cc >= 0) {
          lc = cells[cc].LID;
          ske_len = 0;
          for (int64_t j = 0; j < 2; j++)
            ske_len = ske_len + basis[l + 1].DimsLr[lc + j];
        }
        basis[l].Lchild[li] = lc;
        basis[l].Dims[li] = ske_len;
        basis[l].DimsLr[li] = i;
      }
    }
    dist_int_64_xlen(basis[l].Dims, &comm[l]);

    int64_t count = 0;
    int64_t count_m = 0;
    for (int64_t i = 0; i < xlen; i++) {
      int64_t ske_len = basis[l].Dims[i];
      basis[l].Offsets[i] = count;
      count = count + ske_len;
      count_m = count_m + ske_len * ske_len * 2;
    }
    basis[l].Offsets[xlen] = count;
    basis[l].Multipoles = (int64_t*)malloc(sizeof(int64_t) * count);
    double* matrix_data = (double*)malloc(sizeof(double) * count_m);
    double** matrix_ptrs = (double**)malloc(sizeof(double*) * (xlen + 1));

    for (int64_t i = 0; i < xlen; i++) {
      int64_t lc = basis[l].Lchild[i];
      int64_t ske_len = basis[l].Dims[i];
      int64_t offset = basis[l].Offsets[i];
      if (lc >= 0)
        for (int64_t j = 0; j < 2; j++) {
          int64_t len = basis[l + 1].DimsLr[lc + j];
          int64_t offset_lo = basis[l + 1].Offsets[lc + j];
          memcpy(&basis[l].Multipoles[offset], &basis[l + 1].Multipoles[offset_lo], sizeof(int64_t) * len);
          offset = offset + len;
        }
      else {
        int64_t gi = basis[l].DimsLr[i];
        int64_t body = cells[gi].Body[0];
        for (int64_t j = 0; j < ske_len; j++)
          basis[l].Multipoles[offset + j] = body + j;
      }
      matrix_ptrs[i] = matrix_data;
      matrix_data = matrix_data + ske_len * ske_len * 2;
    }
    matrix_ptrs[xlen] = matrix_data;

    int64_t ibegin = 0, iend = xlen;
    self_local_range(&ibegin, &iend, &comm[l]);
    int64_t nodes = iend - ibegin;

    for (int64_t i = 0; i < nodes; i++) {
      int64_t ske_len = basis[l].Dims[i + ibegin];
      int64_t gi = basis[l].DimsLr[i + ibegin];
      double* mat = matrix_ptrs[i + ibegin];
      struct Matrix S = (struct Matrix){ mat, ske_len, ske_len };
      double* work_data = (double*)malloc(sizeof(double) * (ske_len * sp_pts + (ske_len + sp_pts) * 3));

      int64_t bbegin = cells[gi].Body[0];
      int64_t blen = cells[gi].Body[1] - bbegin;
      int64_t close_avail = nbodies - blen;
      int64_t close_len = sp_pts < close_avail ? sp_pts : close_avail;
      struct Matrix S_work = (struct Matrix){ work_data, ske_len, close_len };
      double* ske_bodies = &work_data[ske_len * sp_pts];
      double* smp_bodies = &work_data[ske_len * (sp_pts + 3)];

      int64_t* mul = basis[l].Multipoles + basis[l].Offsets[i + ibegin];
      for (int64_t j = 0; j < ske_len; j++) {
        int64_t mj = mul[j];
        for (int64_t b = 0; b < 3; b++)
          ske_bodies[j * 3 + b] = bodies[mj * 3 + b];
      }

      double step = close_len == 0 ? 1. : ((double)close_avail / (double)close_len);
      for (int64_t j = 0; j < close_len; j++) {
        int64_t loc = (int64_t)(step * j);
        loc = loc + (loc >= bbegin ? blen : 0);
        for (int64_t b = 0; b < 3; b++)
          smp_bodies[j * 3 + b] = bodies[loc * 3 + b];
      }
      
      gen_matrix(ef, ske_len, close_len, ske_bodies, smp_bodies, S_work.A, NULL, NULL);
      mmult('N', 'T', &S_work, &S_work, &S, 1., 0.);

      int64_t rank = (mrank > 0) && (mrank < ske_len) ? mrank : ske_len;
      double* Svec = &mat[ske_len * ske_len];
      svd_U(&S, Svec);

      if (epi > 0.) {
        int64_t r = 0;
        double sepi = Svec[0];
        sepi = sepi < 1. ? epi : (sepi * epi);
        while(r < rank && Svec[r] > sepi)
          r += 1;
        rank = r;
      }
      basis[l].DimsLr[i + ibegin] = rank;

      int32_t* pa = (int32_t*)malloc(sizeof(int32_t) * ske_len);
      basis[l].Uo[i + ibegin] = (struct Matrix){ mat, ske_len, rank };
      basis[l].Uc[i + ibegin] = (struct Matrix){ &mat[ske_len * rank], ske_len, ske_len - rank };
      basis[l].R[i + ibegin] = (struct Matrix){ &mat[ske_len * ske_len], rank, rank };

      id_row(&basis[l].Uo[i + ibegin], pa, &mat[ske_len * rank]);
      int64_t lc = basis[l].Lchild[i + ibegin];
      basis_reflec(lc >= 0 ? 2 : 0, lc >= 0 ? &basis[l + 1].R[lc] : NULL, &basis[l].Uo[i + ibegin]);
      if (ske_len > 0 && rank > 0)
        qr_full(&basis[l].Uo[i + ibegin], &basis[l].Uc[i + ibegin], &basis[l].R[i + ibegin]);

      for (int64_t j = 0; j < rank; j++) {
        int64_t piv = (int64_t)pa[j] - 1;
        if (piv != j) {
          int64_t c = mul[piv];
          mul[piv] = mul[j];
          mul[j] = c;
        }
      }

      free(work_data);
      free(pa);
    }
    dist_int_64_xlen(basis[l].DimsLr, &comm[l]);
    dist_int_64(basis[l].Multipoles, basis[l].Offsets, &comm[l]);
    dist_double(matrix_ptrs, &comm[l]);

    for (int64_t i = 0; i < xlen; i++) {
      int64_t ske_len = basis[l].Dims[i];
      int64_t rank = basis[l].DimsLr[i];
      double* mat = matrix_ptrs[i];
      basis[l].Uo[i] = (struct Matrix){ mat, ske_len, rank };
      basis[l].Uc[i] = (struct Matrix){ &mat[ske_len * rank], ske_len, ske_len - rank };
      basis[l].R[i] = (struct Matrix){ &mat[ske_len * ske_len], rank, rank };
    }
    free(matrix_ptrs);
  }
}

void basis_free(struct Base* basis) {
  double* data = basis->Uo[0].A;
  if (data)
    free(data);
  if (basis->Multipoles)
    free(basis->Multipoles);
  free(basis->Lchild);
  free(basis->Uo);
}

void evalS(void(*ef)(double*), struct Matrix* S, const struct Base* basis, const double* bodies, const struct CSC* rels, const struct CellComm* comm) {
  int64_t ibegin = 0, iend = 0;
  self_local_range(&ibegin, &iend, comm);
  int64_t lbegin = ibegin;
  i_global(&lbegin, comm);

  for (int64_t x = 0; x < rels->N; x++) {
    for (int64_t yx = rels->ColIndex[x]; yx < rels->ColIndex[x + 1]; yx++) {
      int64_t y = rels->RowIndex[yx];
      int64_t box_y = y;
      i_local(&box_y, comm);
      int64_t m = basis->DimsLr[box_y];
      int64_t n = basis->DimsLr[x + ibegin];
      int64_t* multipoles = basis->Multipoles;
      int64_t off_y = basis->Offsets[box_y];
      int64_t off_x = basis->Offsets[x + ibegin];
      gen_matrix(ef, m, n, bodies, bodies, S[yx].A, &multipoles[off_y], &multipoles[off_x]);
      upper_tri_reflec_mult('L', &basis->R[box_y], &S[yx]);
      upper_tri_reflec_mult('R', &basis->R[x + ibegin], &S[yx]);
    }
  }
}

