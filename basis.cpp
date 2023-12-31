
#include <basis.hpp>
#include <build_tree.hpp>
#include <comm.hpp>
#include <sparse_row.hpp>
#include <linalg.hpp>

#include <algorithm>
#include <numeric>
#include <cstring>

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
    std::vector<int64_t> celli(xlen, 0);

    for (int64_t i = 0; i < xlen; i++) {
      int64_t childi = comm[l].LocalChild[i].first;
      int64_t clen = comm[l].LocalChild[i].second;
      int64_t gi = comm[l].iGlobal(i);
      celli[i] = gi;

      if (childi >= 0 && l < levels)
        for (int64_t j = 0; j < clen; j++)
          basis[l].Dims[i] = basis[l].Dims[i] + basis[l + 1].DimsLr[childi + j];
      else
        basis[l].Dims[i] = cells[celli[i]].Body[1] - cells[celli[i]].Body[0];
    }

    std::vector<std::pair<int64_t, int64_t>> LocalDims(nodes), SumLocalDims(nodes + 1);
    std::transform(&basis[l].Dims[ibegin], &basis[l].Dims[iend], LocalDims.begin(), 
      [](const int64_t& d) { return std::make_pair(3 * d, 2 * d * d); });
    std::inclusive_scan(LocalDims.begin(), LocalDims.end(), SumLocalDims.begin() + 1, 
      [](const std::pair<int64_t, int64_t>& i, const std::pair<int64_t, int64_t>& j) { return std::make_pair(i.first + j.first, i.second + j.second); });
    SumLocalDims[0] = std::make_pair(0, 0);

    std::vector<double> Skeletons(SumLocalDims[nodes].first, 0.);
    std::vector<double> matrix_data(SumLocalDims[nodes].second, 0.);
    
    if (l < levels) {
      int64_t seg = basis[l + 1].dimS;
      for (int64_t i = 0; i < nodes; i++) {
        int64_t dim = basis[l].Dims[i + ibegin];
        int64_t childi = comm[l].LocalChild[i + ibegin].first;
        int64_t clen = comm[l].LocalChild[i + ibegin].second;

        int64_t y = 0;
        for (int64_t j = 0; j < clen; j++) {
          int64_t len = basis[l + 1].DimsLr[childi + j];
          memcpy(&Skeletons[SumLocalDims[i].first + y * 3], &basis[l + 1].M_cpu[(childi + j) * seg * 3], len * 3 * sizeof(double));
          memcpy2d(&matrix_data[SumLocalDims[i].second + y * (dim + 1)], 
            &basis[l + 1].R_cpu[(childi + j) * seg * seg], len, len, dim, seg);
          y = y + len;
        }
      }
    }
    else 
      for (int64_t i = 0; i < nodes; i++) {
        int64_t dim = basis[l].Dims[i + ibegin];
        int64_t ci = celli[i + ibegin];
        int64_t len = cells[ci].Body[1] - cells[ci].Body[0];
        int64_t offset_body = 3 * cells[ci].Body[0];
        
        memcpy(&Skeletons[SumLocalDims[i].first], &bodies[offset_body], len * 3 * sizeof(double));
        for (int64_t j = 0; j < len; j++)
          matrix_data[SumLocalDims[i].second + j * (dim + 1)] = 1.;
      }

    for (int64_t i = 0; i < nodes; i++) {
      int64_t ske_len = basis[l].Dims[i + ibegin];
      double* mat = &matrix_data[SumLocalDims[i].second];
      double* Xbodies = &Skeletons[SumLocalDims[i].first];

      int64_t ci = celli[i + ibegin];
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
    neighbor_bcast_sizes_cpu(basis[l].Dims.data(), &comm[l]);
    neighbor_bcast_sizes_cpu(basis[l].DimsLr.data(), &comm[l]);

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
      max[2] = std::max(max[2], comm[l].LocalChild[i].second);
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

      if (ibegin <= i && i < iend) {
        int64_t child = comm[l].LocalChild[i].first;
        int64_t clen = comm[l].LocalChild[i].second;
        if (child >= 0 && l < levels) {
          int64_t row = 0;
          for (int64_t j = 0; j < clen; j++) {
            int64_t N = basis[l + 1].DimsLr[child + j];
            int64_t Urow = j * basis[l + 1].dimS;
            memcpy2d(&Uc_ptr[Urow], &matrix_data[SumLocalDims[i - ibegin].second + No * M + row], N, Nc, LD, M);
            memcpy2d(&Uo_ptr[Urow], &matrix_data[SumLocalDims[i - ibegin].second + row], N, No, LD, M);
            row = row + N;
          }
        }
        else {
          memcpy2d(Uc_ptr, &matrix_data[SumLocalDims[i - ibegin].second + No * M], M, Nc, LD, M);
          memcpy2d(Uo_ptr, &matrix_data[SumLocalDims[i - ibegin].second], M, No, LD, M);
        }
        memcpy2d(R_ptr, &matrix_data[SumLocalDims[i - ibegin].second + M * M], No, No, basis[l].dimS, M);
        memcpy(M_ptr, &Skeletons[SumLocalDims[i - ibegin].first], 3 * No * sizeof(double));

        double* Ui_ptr = basis[l].U_cpu + xlen * stride + (i - ibegin) * basis[l].dimR; 
        for (int64_t j = Nc; j < basis[l].dimR; j++)
          Ui_ptr[j] = 1.;
      }

      basis[l].Uo[i] = (Matrix) { Uo_ptr, basis[l].dimN, basis[l].dimS, basis[l].dimN };
      basis[l].R[i] = (Matrix) { R_ptr, basis[l].dimS, basis[l].dimS, basis[l].dimS };
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
