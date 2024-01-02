
#include <basis.hpp>
#include <build_tree.hpp>
#include <comm.hpp>
#include <linalg.hpp>

#include <algorithm>
#include <numeric>
#include <cstring>
#include <cmath>
#include <tuple>

template <typename T>
void memcpy2d(T* dst, const T* src, int64_t rows, int64_t cols, int64_t ld_dst, int64_t ld_src) {
  if (rows == ld_dst && rows == ld_src)
    std::copy(src, src + rows * cols, dst);
  else 
    for (int64_t i = 0; i < cols; i++)
      std::copy(&src[i * ld_src], &src[i * ld_src + rows], &dst[i * ld_dst]);
}

void buildBasis(const EvalDouble& eval, Base basis[], Cell* cells, const CSR* rel_near, int64_t levels,
  const CellComm* comm, const double* bodies, int64_t nbodies, double epi, int64_t alignment) {

  for (int64_t l = levels; l >= 0; l--) {
    int64_t xlen = 0, ibegin = 0, nodes = 0;
    content_length(&nodes, &xlen, &ibegin, &comm[l]);
    int64_t iend = ibegin + nodes;
    basis[l].Dims = std::vector<int64_t>(xlen, 0);
    basis[l].DimsLr = std::vector<int64_t>(xlen, 0);

    Matrix* arr_m = (Matrix*)calloc(xlen * 2, sizeof(Matrix));
    basis[l].Uo = arr_m;
    basis[l].R = &arr_m[xlen];
    std::vector<std::tuple<int64_t, int64_t, int64_t>> celli(xlen);

    for (int64_t i = 0; i < xlen; i++) {
      int64_t gi = comm[l].iGlobal(i);
      int64_t childi = l < levels ? comm[l + 1].iLocal(cells[gi].Child[0]) : -1;
      int64_t clen = cells[gi].Child[1] - cells[gi].Child[0];
      celli[i] = std::make_tuple(gi, childi, clen);

      if (childi >= 0 && l < levels)
        for (int64_t j = 0; j < clen; j++)
          basis[l].Dims[i] = basis[l].Dims[i] + basis[l + 1].DimsLr[childi + j];
      else
        basis[l].Dims[i] = cells[gi].Body[1] - cells[gi].Body[0];
    }

    std::vector<std::pair<int64_t, int64_t>> LocalDims(nodes), SumLocalDims(nodes + 1);
    std::transform(&basis[l].Dims[ibegin], &basis[l].Dims[iend], LocalDims.begin(), 
      [](const int64_t& d) { return std::make_pair(3 * d, 2 * d * d); });
    std::inclusive_scan(LocalDims.begin(), LocalDims.end(), SumLocalDims.begin() + 1, 
      [](const std::pair<int64_t, int64_t>& i, const std::pair<int64_t, int64_t>& j) { return std::make_pair(i.first + j.first, i.second + j.second); });
    SumLocalDims[0] = std::make_pair(0, 0);

    std::vector<double> Skeletons(SumLocalDims[nodes].first, 0.);
    std::vector<double> MatrixData(SumLocalDims[nodes].second, 0.);
    
    if (l < levels) {
      int64_t seg = basis[l + 1].dimS;
      for (int64_t i = 0; i < nodes; i++) {
        int64_t dim = basis[l].Dims[i + ibegin];
        int64_t childi = std::get<1>(celli[i + ibegin]);
        int64_t clen = std::get<2>(celli[i + ibegin]);

        int64_t y = 0;
        for (int64_t j = 0; j < clen; j++) {
          int64_t len = basis[l + 1].DimsLr[childi + j];
          memcpy(&Skeletons[SumLocalDims[i].first + y * 3], &basis[l + 1].M_cpu[(childi + j) * seg * 3], len * 3 * sizeof(double));
          memcpy2d(&MatrixData[SumLocalDims[i].second + y * (dim + 1)], 
            &basis[l + 1].R_cpu[(childi + j) * seg * seg], len, len, dim, seg);
          y = y + len;
        }
      }
    }
    else 
      for (int64_t i = 0; i < nodes; i++) {
        int64_t dim = basis[l].Dims[i + ibegin];
        int64_t ci = std::get<0>(celli[i + ibegin]);
        int64_t len = cells[ci].Body[1] - cells[ci].Body[0];
        int64_t offset_body = 3 * cells[ci].Body[0];
        
        memcpy(&Skeletons[SumLocalDims[i].first], &bodies[offset_body], len * 3 * sizeof(double));
        for (int64_t j = 0; j < len; j++)
          MatrixData[SumLocalDims[i].second + j * (dim + 1)] = 1.;
      }

    for (int64_t i = 0; i < nodes; i++) {
      int64_t ske_len = basis[l].Dims[i + ibegin];
      double* mat = &MatrixData[SumLocalDims[i].second];
      double* Xbodies = &Skeletons[SumLocalDims[i].first];

      int64_t ci = std::get<0>(celli[i + ibegin]);
      int64_t nbegin = rel_near->RowIndex[ci];
      int64_t nlen = rel_near->RowIndex[ci + 1] - nbegin;
      const int64_t* ngbs = &rel_near->ColIndex[nbegin];
      std::vector<const double*> remote;
      std::vector<int64_t> lens;

      int64_t loc = 0, len_f = 0;
      for (int64_t j = 0; j < nlen; j++) {
        int64_t cj = ngbs[j];
        int64_t len = cells[cj].Body[0] - loc;
        if (len > 0) {
          remote.emplace_back(&bodies[loc * 3]);
          lens.emplace_back(len);
          len_f = len_f + len;
        }
        loc = cells[cj].Body[1];
      }
      if (loc < nbodies) {
        remote.emplace_back(&bodies[loc * 3]);
        lens.emplace_back(nbodies - loc);
        len_f = len_f + nbodies - loc;
      }
      
      int64_t rank = compute_basis(eval, epi, ske_len, mat, ske_len, &Xbodies[0], remote.size(), &lens[0], &remote[0]);
      basis[l].DimsLr[i + ibegin] = rank;
    }
    comm[l].neighbor_bcast_sizes(basis[l].Dims.data());
    comm[l].neighbor_bcast_sizes(basis[l].DimsLr.data());

    int64_t max[3] = { 0, 0, 0 };
    for (int64_t i = 0; i < xlen; i++) {
      int64_t i1 = basis[l].DimsLr[i];
      int64_t i2 = basis[l].Dims[i] - basis[l].DimsLr[i];
      int64_t rem1 = i1 & (alignment - 1);
      int64_t rem2 = i2 & (alignment - 1);

      i1 = std::max(alignment, i1 - rem1 + (rem1 ? alignment : 0));
      i2 = std::max(alignment, i2 - rem2 + (rem2 ? alignment : 0));
      max[0] = std::max(max[0], i1);
      max[1] = std::max(max[1], i2);
      max[2] = std::max(max[2], std::get<2>(celli[i]));
    }
    MPI_Allreduce(MPI_IN_PLACE, max, 3, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);

    basis[l].dimN = std::max(max[0] + max[1], basis[l + 1].dimS * max[2]);
    basis[l].dimS = max[0];
    basis[l].dimR = basis[l].dimN - basis[l].dimS;
    int64_t stride = basis[l].dimN * basis[l].dimN;
    int64_t stride_r = basis[l].dimS * basis[l].dimS;
    int64_t LD = basis[l].dimN;

    basis[l].M_cpu = (double*)calloc(basis[l].dimS * xlen * 3, sizeof(double));
    basis[l].U_cpu = (double*)calloc(stride * xlen + nodes * basis[l].dimR, sizeof(double));
    basis[l].R_cpu = (double*)calloc(stride_r * xlen, sizeof(double));

    for (int64_t i = 0; i < xlen; i++) {
      double* M_ptr = basis[l].M_cpu + i * basis[l].dimS * 3;
      double* Uc_ptr = basis[l].U_cpu + i * stride;
      double* Uo_ptr = Uc_ptr + basis[l].dimR * basis[l].dimN;
      double* R_ptr = basis[l].R_cpu + i * stride_r;

      int64_t Nc = basis[l].Dims[i] - basis[l].DimsLr[i];
      int64_t No = basis[l].DimsLr[i];
      int64_t M = basis[l].Dims[i];

      if (i >= ibegin && i < iend) {
        memcpy2d(Uc_ptr, &MatrixData[SumLocalDims[i - ibegin].second + No * M], M, Nc, LD, M);
        memcpy2d(Uo_ptr, &MatrixData[SumLocalDims[i - ibegin].second], M, No, LD, M);
        memcpy2d(R_ptr, &MatrixData[SumLocalDims[i - ibegin].second + M * M], No, No, basis[l].dimS, M);
        memcpy(M_ptr, &Skeletons[SumLocalDims[i - ibegin].first], 3 * No * sizeof(double));
      }

      basis[l].Uo[i] = (Matrix) { Uo_ptr, M, No, basis[l].dimN };
      basis[l].R[i] = (Matrix) { R_ptr, No, No, basis[l].dimS };
    }
    neighbor_bcast_cpu(basis[l].M_cpu, 3 * basis[l].dimS, &comm[l]);
    comm[l].dup_bcast(basis[l].M_cpu, 3 * basis[l].dimS * xlen);
    neighbor_bcast_cpu(basis[l].U_cpu, stride, &comm[l]);
    comm[l].dup_bcast(basis[l].U_cpu, stride * xlen);
    neighbor_bcast_cpu(basis[l].R_cpu, stride_r, &comm[l]);
    comm[l].dup_bcast(basis[l].R_cpu, stride_r * xlen);
  }
}


void basis_free(Base* basis) {
  free(basis->Uo);
  if (basis->M_cpu)
    free(basis->M_cpu);
  if (basis->U_cpu)
    free(basis->U_cpu);
  if (basis->R_cpu)
    free(basis->R_cpu);
}


void matVecA(const EvalDouble& eval, const Base basis[], const double bodies[], const Cell cells[], const CSR* rels_near, const CSR* rels_far, double X[], const CellComm comm[], int64_t levels) {
  int64_t lbegin = 0, llen = 0;
  content_length(&llen, NULL, &lbegin, &comm[levels]);

  std::vector<std::vector<double>> rhsX(levels + 1), rhsB(levels + 1);
  std::vector<std::vector<double*>> rhsXptr(levels + 1), rhsXoptr(levels + 1), rhsBptr(levels + 1), rhsBoptr(levels + 1);

  for (int64_t l = levels; l >= 0; l--) {
    int64_t xlen = 0;
    content_length(NULL, &xlen, NULL, &comm[l]);
    std::vector<int64_t> offsets(xlen + 1);
    std::inclusive_scan(basis[l].Dims.begin(), basis[l].Dims.end(), offsets.begin() + 1);

    rhsX[l] = std::vector<double>(offsets[xlen], 0);
    rhsB[l] = std::vector<double>(offsets[xlen], 0);
    rhsXptr[l] = std::vector<double*>(xlen, NULL);
    rhsBptr[l] = std::vector<double*>(xlen, NULL);
    rhsXoptr[l] = std::vector<double*>(xlen, NULL);
    rhsBoptr[l] = std::vector<double*>(xlen, NULL);

    std::transform(offsets.begin(), offsets.begin() + xlen, rhsXptr[l].begin(), [&](const int64_t d) { return &rhsX[l][d]; });
    std::transform(offsets.begin(), offsets.begin() + xlen, rhsBptr[l].begin(), [&](const int64_t d) { return &rhsB[l][d]; });

    if (l < levels)
      for (int64_t i = 0; i < xlen; i++) {
        int64_t ci = comm[l].iGlobal(i);
        int64_t child = comm[l + 1].iLocal(cells[ci].Child[0]);
        int64_t clen = cells[ci].Child[1] - cells[ci].Child[0];

        if (child >= 0 && clen > 0) {
          std::vector<int64_t> offsets_child(clen + 1, 0);
          std::inclusive_scan(&basis[l + 1].DimsLr[child], &basis[l + 1].DimsLr[child + clen], offsets_child.begin() + 1);
          std::transform(offsets_child.begin(), offsets_child.begin() + clen, &rhsXoptr[l + 1][child], [&](const int64_t d) { return rhsXptr[l][i] + d; });
          std::transform(offsets_child.begin(), offsets_child.begin() + clen, &rhsBoptr[l + 1][child], [&](const int64_t d) { return rhsBptr[l][i] + d; });
        }
      }
  }

  int64_t lenX = std::accumulate(&basis[levels].Dims[lbegin], &basis[levels].Dims[lbegin + llen], 0);
  std::copy(X, &X[lenX], rhsXptr[levels][lbegin]);

  for (int64_t i = levels; i > 0; i--) {
    int64_t ibegin = 0, iboxes = 0, xlen = 0;
    content_length(&iboxes, &xlen, &ibegin, &comm[i]);

    int64_t lenI = std::accumulate(&basis[i].Dims[0], &basis[i].Dims[xlen], 0);
    comm[i].level_merge(rhsX[i].data(), lenI);
    comm[i].neighbor_bcast(rhsX[i].data(), basis[i].Dims.data());
    comm[i].dup_bcast(rhsX[i].data(), lenI);

    for (int64_t j = 0; j < iboxes; j++) {
      Matrix Xj = (Matrix) { rhsXptr[i][j + ibegin], basis[i].Dims[j + ibegin], 1, basis[i].Dims[j + ibegin] };
      Matrix Xo = (Matrix) { rhsXoptr[i][j + ibegin], basis[i].DimsLr[j + ibegin], 1, basis[i].DimsLr[j + ibegin] };
      mmult('T', 'N', &basis[i].Uo[j + ibegin], &Xj, &Xo, 1., 0.);
    }
  }

  if (basis[0].Dims[0] > 0) {
    comm[0].level_merge(rhsX[0].data(), basis[0].Dims[0]);
    comm[0].dup_bcast(rhsX[0].data(), basis[0].Dims[0]);
  }

  for (int64_t i = 1; i <= levels; i++) {
    int64_t ibegin = 0, iboxes = 0;
    content_length(&iboxes, NULL, &ibegin, &comm[i]);
    int64_t gbegin = comm[i].iGlobal(ibegin);

    for (int64_t y = 0; y < iboxes; y++)
      for (int64_t yx = rels_far->RowIndex[y + gbegin]; yx < rels_far->RowIndex[y + gbegin + 1]; yx++) {
        int64_t x = comm[i].iLocal(rels_far->ColIndex[yx]);
        int64_t M = basis[i].DimsLr[y + ibegin];
        int64_t N = basis[i].DimsLr[x];

        Matrix Xo = (Matrix) { rhsXoptr[i][x], N, 1, N };
        Matrix Bo = (Matrix) { rhsBoptr[i][y + ibegin], M, 1, M };
        std::vector<double> TMPX(N, 0.), TMPB(M, 0.);
        Matrix T1 = (Matrix) { &TMPX[0], N, 1, N };
        Matrix T2 = (Matrix) { &TMPB[0], M, 1, M };

        mmult('T', 'N', &basis[i].R[x], &Xo, &T1, 1., 0.);
        mat_vec_reference(eval, M, N, &TMPB[0], &TMPX[0], basis[i].M_cpu + 3 * basis[i].dimS * (y + ibegin), basis[i].M_cpu + 3 * basis[i].dimS * x);
        mmult('N', 'N', &basis[i].R[y + ibegin], &T2, &Bo, 1., 1.);
      }
  }
  
  for (int64_t i = 1; i <= levels; i++) {
    int64_t ibegin = 0, iboxes = 0;
    content_length(&iboxes, NULL, &ibegin, &comm[i]);
    for (int64_t j = 0; j < iboxes; j++) {
      Matrix Bj = (Matrix) { rhsBptr[i][j + ibegin], basis[i].Dims[j + ibegin], 1, basis[i].Dims[j + ibegin] };
      Matrix Bo = (Matrix) { rhsBoptr[i][j + ibegin], basis[i].DimsLr[j + ibegin], 1, basis[i].DimsLr[j + ibegin] };
      mmult('N', 'N', &basis[i].Uo[j + ibegin], &Bo, &Bj, 1., 1.);
    }
  }

  int64_t gbegin = comm[levels].iGlobal(lbegin);
  for (int64_t y = 0; y < llen; y++)
    for (int64_t yx = rels_near->RowIndex[y + gbegin]; yx < rels_near->RowIndex[y + gbegin + 1]; yx++) {
      int64_t x = rels_near->ColIndex[yx];
      int64_t x_loc = comm[levels].iLocal(x);
      int64_t M = cells[y + gbegin].Body[1] - cells[y + gbegin].Body[0];
      int64_t N = cells[x].Body[1] - cells[x].Body[0];
      mat_vec_reference(eval, M, N, rhsBptr[levels][y + lbegin], rhsXptr[levels][x_loc], &bodies[3 * cells[y + gbegin].Body[0]], &bodies[3 * cells[x].Body[0]]);
    }

  std::copy(rhsBptr[levels][lbegin], rhsBptr[levels][lbegin] + lenX, X);
}


void solveRelErr(double* err_out, const double* X, const double* ref, int64_t lenX) {
  double err[2] = { 0., 0. };
  for (int64_t i = 0; i < lenX; i++) {
    double diff = X[i] - ref[i];
    err[0] = err[0] + diff * diff;
    err[1] = err[1] + ref[i] * ref[i];
  }
  MPI_Allreduce(MPI_IN_PLACE, err, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  *err_out = std::sqrt(err[0] / err[1]);
}

