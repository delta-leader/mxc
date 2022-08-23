
#include "nbd.h"
#include "profile.h"

#include "stdlib.h"
#include "math.h"
#include "string.h"

struct SampleBodies 
{ int64_t LTlen, *CloseLens, *CloseAvails, **CloseBodies, *SkeLens, **Skeletons; };

void buildSampleBodies(struct SampleBodies* sample, int64_t sp_max, int64_t ncells, const struct Cell* cells, 
const struct CSC* rels, const int64_t* lt_child, const struct Base* basis_lo, int64_t level) {
  int __mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &__mpi_rank);
  int64_t mpi_rank = __mpi_rank;
  const int64_t LEN_CHILD = 2;
  
  int64_t jbegin = 0, jend = ncells;
  get_level(&jbegin, &jend, cells, level, -1);
  int64_t ibegin = jbegin, iend = jend;
  get_level(&ibegin, &iend, cells, level, mpi_rank);
  int64_t nodes = iend - ibegin;
  int64_t* arr_ctrl = (int64_t*)malloc(sizeof(int64_t) * nodes * 3);
  int64_t** arr_list = (int64_t**)malloc(sizeof(int64_t*) * nodes * 2);
  sample->LTlen = nodes;
  sample->CloseLens = arr_ctrl;
  sample->SkeLens = &arr_ctrl[nodes];
  sample->CloseAvails = &arr_ctrl[nodes * 2];
  sample->CloseBodies = arr_list;
  sample->Skeletons = &arr_list[nodes];
  
  int64_t count_c = 0, count_s = 0;
  for (int64_t i = 0; i < nodes; i++) {
    int64_t li = ibegin + i;
    int64_t nbegin = rels->ColIndex[i];
    int64_t nlen = rels->ColIndex[i + 1] - nbegin;
    const int64_t* ngbs = &rels->RowIndex[nbegin];

    int64_t close_avail = 0;
    for (int64_t j = 0; j < nlen; j++) {
      int64_t lj = ngbs[j] + jbegin;
      const struct Cell* cj = &cells[lj];
      int64_t len = cj->Body[1] - cj->Body[0];
      if (lj != li)
        close_avail = close_avail + len;
    }

    int64_t lc = lt_child[i];
    int64_t ske_len = 0;
    if (basis_lo != NULL && lc >= 0)
      for (int64_t j = 0; j < LEN_CHILD; j++)
        ske_len = ske_len + basis_lo->DimsLr[lc + j];
    else
      ske_len = cells[li].Body[1] - cells[li].Body[0];

    int64_t close_len = sp_max < close_avail ? sp_max : close_avail;
    arr_ctrl[i] = close_len;
    arr_ctrl[i + nodes] = ske_len;
    arr_ctrl[i + nodes * 2] = close_avail;
    count_c = count_c + close_len;
    count_s = count_s + ske_len;
  }

  int64_t* arr_bodies = NULL;
  if ((count_c + count_s) > 0)
    arr_bodies = (int64_t*)malloc(sizeof(int64_t) * (count_c + count_s));
  count_s = count_c;
  count_c = 0;
  for (int64_t i = 0; i < nodes; i++) {
    int64_t* close = &arr_bodies[count_c];
    int64_t* skeleton = &arr_bodies[count_s];
    int64_t close_len = arr_ctrl[i];
    int64_t ske_len = arr_ctrl[i + nodes];
    arr_list[i] = close;
    arr_list[i + nodes] = skeleton;
    count_c = count_c + close_len;
    count_s = count_s + ske_len;
  }

  for (int64_t i = 0; i < nodes; i++) {
    int64_t nbegin = rels->ColIndex[i];
    int64_t nlen = rels->ColIndex[i + 1] - nbegin;
    const int64_t* ngbs = &rels->RowIndex[nbegin];

    int64_t* close = arr_list[i];
    int64_t* skeleton = arr_list[i + nodes];
    int64_t close_len = arr_ctrl[i];
    int64_t ske_len = arr_ctrl[i + nodes];
    int64_t close_avail = arr_ctrl[i + nodes * 2];

    int64_t li = i + ibegin - jbegin;
    int64_t cpos = 0;
    while (cpos < nlen && ngbs[cpos] != li)
      cpos = cpos + 1;
    
    int64_t box_i = (int64_t)(cpos == 0);
    int64_t s_lens = 0, ic = 0, offset_i = 0, len_i = 0;
    if (box_i < nlen) {
      ic = jbegin + ngbs[box_i];
      offset_i = cells[ic].Body[0];
      len_i = cells[ic].Body[1] - offset_i;
    }

    for (int64_t j = 0; j < close_len; j++) {
      int64_t loc = (int64_t)((double)(close_avail * j) / close_len);
      while (loc - s_lens >= len_i) {
        s_lens = s_lens + len_i;
        box_i = box_i + 1;
        box_i = box_i + (int64_t)(box_i == cpos);
        ic = jbegin + ngbs[box_i];
        offset_i = cells[ic].Body[0];
        len_i = cells[ic].Body[1] - offset_i;
      }
      close[j] = loc + offset_i - s_lens;
    }

    int64_t lc = lt_child[i];
    int64_t sbegin = cells[i + ibegin].Body[0];
    if (basis_lo != NULL && lc >= 0)
      memcpy(skeleton, basis_lo->Multipoles + basis_lo->Offsets[lc], sizeof(int64_t) * ske_len);
    else
      for (int64_t j = 0; j < ske_len; j++)
        skeleton[j] = j + sbegin;
  }
}

void sampleBodies_free(struct SampleBodies* sample) {
  int64_t* data = sample->CloseBodies[0];
  if (data)
    free(data);
  free(sample->CloseLens);
  free(sample->CloseBodies);
}

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

void buildBasis(void(*ef)(double*), struct Base basis[], int64_t ncells, const struct Cell* cells, const struct CSC* rel_near, int64_t levels, 
const struct CellComm* comm, const double* bodies, double epi, int64_t mrank, int64_t sp_pts) {

  for (int64_t l = levels; l >= 0; l--) {
    int64_t xlen = 0;
    content_length(&xlen, &comm[l]);
    basis[l].Ulen = xlen;
    int64_t* arr_i = (int64_t*)malloc(sizeof(int64_t) * (xlen * 4 + 1));
    basis[l].Lchild = arr_i;
    basis[l].Dims = &arr_i[xlen];
    basis[l].DimsLr = &arr_i[xlen * 2];
    basis[l].Offsets = &arr_i[xlen * 3];
    basis[l].Multipoles = NULL;

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
      }
    }
    dist_int_64_xlen(basis[l].Dims, &comm[l]);

    struct Matrix* arr_m = (struct Matrix*)calloc(xlen * 3, sizeof(struct Matrix));
    basis[l].Uo = arr_m;
    basis[l].Uc = &arr_m[xlen];
    basis[l].R = &arr_m[xlen * 2];

    int64_t ibegin = 0, iend = xlen;
    self_local_range(&ibegin, &iend, &comm[l]);
    int64_t nodes = iend - ibegin;

    struct SampleBodies samples;
    buildSampleBodies(&samples, sp_pts, ncells, cells, &rel_near[l], &basis[l].Lchild[ibegin], l == levels ? NULL : &basis[l + 1], l);

    int64_t count = 0;
    int64_t count_m = 0;
    for (int64_t i = 0; i < nodes; i++) {
      int64_t ske_len = samples.SkeLens[i];
      count = count + ske_len;
      count_m = count_m + ske_len * (ske_len + 2 + sp_pts);
    }

    int32_t* ipiv_data = (int32_t*)malloc(sizeof(int32_t) * count);
    int32_t** ipiv_ptrs = (int32_t**)malloc(sizeof(int32_t*) * xlen);
    double* matrix_data = (double*)malloc(sizeof(double) * count_m);
    double** matrix_ptrs = (double**)malloc(sizeof(double*) * (xlen + 1));

    count = 0;
    count_m = 0;
    for (int64_t i = 0; i < nodes; i++) {
      int64_t ske_len = samples.SkeLens[i];
      ipiv_ptrs[i] = &ipiv_data[count];
      matrix_ptrs[i + ibegin] = &matrix_data[count_m];
      count = count + ske_len;
      count_m = count_m + ske_len * (ske_len + 2 + sp_pts);
    }

    for (int64_t i = 0; i < nodes; i++) {
      int64_t ske_len = samples.SkeLens[i];
      double* mat = matrix_ptrs[i + ibegin];
      struct Matrix S = (struct Matrix){ mat, ske_len, ske_len };
      struct Matrix S_work = (struct Matrix){ &mat[ske_len * ske_len], ske_len, samples.CloseLens[i] };
      
      gen_matrix(ef, ske_len, samples.CloseLens[i], bodies, bodies, S_work.A, samples.Skeletons[i], samples.CloseBodies[i]);
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

      int32_t* pa = ipiv_ptrs[i];
      struct Matrix Qo = (struct Matrix){ mat, ske_len, rank };
      id_row(&Qo, pa, &mat[ske_len * rank]);
      int64_t lc = basis[l].Lchild[i + ibegin];
      basis_reflec(lc >= 0 ? 2 : 0, lc >= 0 ? &basis[l + 1].R[lc] : NULL, &Qo);

      for (int64_t j = 0; j < rank; j++) {
        int64_t piv = (int64_t)pa[j] - 1;
        if (piv != j) {
          int64_t c = samples.Skeletons[i][piv];
          samples.Skeletons[i][piv] = samples.Skeletons[i][j];
          samples.Skeletons[i][j] = c;
        }
      }
    }

    dist_int_64_xlen(basis[l].DimsLr, &comm[l]);

    count = 0;
    count_m = 0;
    for (int64_t i = 0; i < xlen; i++) {
      int64_t m = basis[l].Dims[i];
      int64_t n = basis[l].DimsLr[i];
      basis[l].Offsets[i] = count;
      count = count + n;
      count_m = count_m + m * m + n * n;
    }
    basis[l].Offsets[xlen] = count;

    if (count > 0)
      basis[l].Multipoles = (int64_t*)malloc(sizeof(int64_t) * count);
    for (int64_t i = 0; i < nodes; i++) {
      int64_t offset = basis[l].Offsets[i + ibegin];
      int64_t n = basis[l].DimsLr[i + ibegin];
      if (n > 0)
        memcpy(&basis[l].Multipoles[offset], samples.Skeletons[i], sizeof(int64_t) * n);
    }
    dist_int_64(basis[l].Multipoles, basis[l].Offsets, &comm[l]);

    double* data_basis = NULL;
    if (count_m > 0)
      data_basis = (double*)malloc(sizeof(double) * count_m);
    for (int64_t i = 0; i < xlen; i++) {
      int64_t m = basis[l].Dims[i];
      int64_t n = basis[l].DimsLr[i];
      int64_t size = m * n;
      if (ibegin <= i && i < iend && size > 0)
        memcpy(data_basis, matrix_ptrs[i], sizeof(double) * size);
      basis[l].Uo[i] = (struct Matrix){ data_basis, m, n };
      matrix_ptrs[i] = data_basis;
      data_basis = &data_basis[size];
    }
    matrix_ptrs[xlen] = data_basis;
    dist_double(matrix_ptrs, &comm[l]);

    for (int64_t i = 0; i < xlen; i++) {
      int64_t m = basis[l].Dims[i];
      int64_t n = basis[l].DimsLr[i];
      int64_t size = m * (m - n) + n * n;
      basis[l].Uc[i] = (struct Matrix){ data_basis, m, m - n };
      basis[l].R[i] = (struct Matrix){ &data_basis[m * (m - n)], n, n };
      data_basis = &data_basis[size];
      if (m > 0 && n > 0)
        qr_full(&basis[l].Uo[i], &basis[l].Uc[i], &basis[l].R[i]);
    }

    free(ipiv_data);
    free(ipiv_ptrs);
    free(matrix_data);
    free(matrix_ptrs);
    sampleBodies_free(&samples);
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

