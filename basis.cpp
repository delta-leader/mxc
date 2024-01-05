
#include <basis.hpp>
#include <build_tree.hpp>
#include <comm.hpp>
#include <linalg.hpp>

#include <algorithm>
#include <numeric>
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

const double* Base::ske_at_i(int64_t i) const {
  return Mdata.data() + 3 * std::accumulate(DimsLr.begin(), DimsLr.begin() + i, 0);
}

void buildBasis(const Eval& eval, double epi, Base basis[], const Cell* cells, const CSR& rel_near, int64_t levels, const CellComm* comm, const double* bodies, int64_t nbodies) {

  for (int64_t l = levels; l >= 0; l--) {
    int64_t xlen = comm[l].lenNeighbors();
    int64_t ibegin = comm[l].oLocal();
    int64_t nodes = comm[l].lenLocal();
    int64_t iend = ibegin + nodes;
    basis[l].Dims = std::vector<int64_t>(xlen, 0);
    basis[l].DimsLr = std::vector<int64_t>(xlen, 0);

    basis[l].Uo = std::vector<Matrix>(xlen);
    basis[l].R = std::vector<Matrix>(xlen);
    std::vector<std::tuple<int64_t, int64_t, int64_t>> celli(xlen);

    for (int64_t i = 0; i < xlen; i++) {
      int64_t gi = comm[l].iGlobal(i);
      int64_t child = l < levels ? comm[l + 1].iLocal(cells[gi].Child[0]) : -1;
      int64_t clen = cells[gi].Child[1] - cells[gi].Child[0];
      celli[i] = std::make_tuple(gi, child, clen);

      if (child >= 0 && l < levels)
        basis[l].Dims[i] = std::accumulate(&basis[l + 1].DimsLr[child], &basis[l + 1].DimsLr[child + clen], 0);
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
    std::vector<std::complex<double>> MatrixData(SumLocalDims[nodes].second, 0.);
    
    if (l < levels)
      for (int64_t i = 0; i < nodes; i++) {
        int64_t dim = basis[l].Dims[i + ibegin];
        int64_t childi = std::get<1>(celli[i + ibegin]);
        int64_t clen = std::get<2>(celli[i + ibegin]);

        int64_t y = 0;
        for (int64_t j = 0; j < clen; j++) {
          int64_t len = basis[l + 1].DimsLr[childi + j];
          memcpy2d(&MatrixData[SumLocalDims[i].second + y * (dim + 1)], basis[l + 1].R[childi + j].A, len, len, dim, len);
          y = y + len;
        }

        const double* mbegin = basis[l + 1].ske_at_i(childi);
        int64_t mlen = 3 * basis[l].Dims[i + ibegin];
        std::copy(mbegin, &mbegin[mlen], &Skeletons[SumLocalDims[i].first]);
      }
    else 
      for (int64_t i = 0; i < nodes; i++) {
        int64_t dim = basis[l].Dims[i + ibegin];
        int64_t ci = std::get<0>(celli[i + ibegin]);
        int64_t len = cells[ci].Body[1] - cells[ci].Body[0];
        int64_t offset_body = 3 * cells[ci].Body[0];
        
        std::copy(&bodies[offset_body], &bodies[offset_body + len * 3], &Skeletons[SumLocalDims[i].first]);
        for (int64_t j = 0; j < len; j++)
          MatrixData[SumLocalDims[i].second + j * (dim + 1)] = 1.;
      }

    for (int64_t i = 0; i < nodes; i++) {
      int64_t ske_len = basis[l].Dims[i + ibegin];
      std::complex<double>* mat = &MatrixData[SumLocalDims[i].second];
      double* Xbodies = &Skeletons[SumLocalDims[i].first];

      int64_t ci = std::get<0>(celli[i + ibegin]);
      int64_t nbegin = rel_near.RowIndex[ci];
      int64_t nlen = rel_near.RowIndex[ci + 1] - nbegin;
      const int64_t* ngbs = &rel_near.ColIndex[nbegin];
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

    std::vector<int64_t> Msizes(xlen), Usizes(xlen), Rsizes(xlen);
    std::transform(basis[l].DimsLr.begin(), basis[l].DimsLr.end(), Msizes.begin(), [](const int64_t& d) { return d * 3; });
    std::transform(basis[l].Dims.begin(), basis[l].Dims.end(), Usizes.begin(), [](const int64_t& d) { return d * d; });
    std::transform(basis[l].DimsLr.begin(), basis[l].DimsLr.end(), Rsizes.begin(), [](const int64_t& d) { return d * d; });

    std::vector<int64_t> Moffsets(xlen + 1), Uoffsets(xlen + 1), Roffsets(xlen + 1);
    std::inclusive_scan(Msizes.begin(), Msizes.end(), Moffsets.begin() + 1);
    std::inclusive_scan(Usizes.begin(), Usizes.end(), Uoffsets.begin() + 1);
    std::inclusive_scan(Rsizes.begin(), Rsizes.end(), Roffsets.begin() + 1);
    Moffsets[0] = 0;
    Uoffsets[0] = 0;
    Roffsets[0] = 0;

    basis[l].Mdata = std::vector<double>(Moffsets[xlen]);
    basis[l].Udata = std::vector<std::complex<double>>(Uoffsets[xlen]);
    basis[l].Rdata = std::vector<std::complex<double>>(Roffsets[xlen]);

    for (int64_t i = 0; i < xlen; i++) {
      int64_t Nc = basis[l].Dims[i] - basis[l].DimsLr[i];
      int64_t No = basis[l].DimsLr[i];
      int64_t M = basis[l].Dims[i];

      double* M_ptr = &basis[l].Mdata[Moffsets[i]];
      std::complex<double>* Uc_ptr = &basis[l].Udata[Uoffsets[i]];
      std::complex<double>* Uo_ptr = Uc_ptr + M * Nc;
      std::complex<double>* R_ptr = &basis[l].Rdata[Roffsets[i]];

      if (i >= ibegin && i < iend) {
        memcpy2d(Uc_ptr, &MatrixData[SumLocalDims[i - ibegin].second + No * M], M, Nc, M, M);
        memcpy2d(Uo_ptr, &MatrixData[SumLocalDims[i - ibegin].second], M, No, M, M);
        memcpy2d(R_ptr, &MatrixData[SumLocalDims[i - ibegin].second + M * M], No, No, No, M);
        std::copy(&Skeletons[SumLocalDims[i - ibegin].first], &Skeletons[SumLocalDims[i - ibegin].first + 3 * No], M_ptr);
      }

      basis[l].Uo[i] = (Matrix) { Uo_ptr, M, No, M };
      basis[l].R[i] = (Matrix) { R_ptr, No, No, No };
    }

    comm[l].neighbor_bcast(basis[l].Mdata.data(), Msizes.data());
    comm[l].dup_bcast(basis[l].Mdata.data(), Moffsets[xlen]);
    comm[l].neighbor_bcast(basis[l].Udata.data(), Usizes.data());
    comm[l].dup_bcast(basis[l].Udata.data(), Uoffsets[xlen]);
    comm[l].neighbor_bcast(basis[l].Rdata.data(), Rsizes.data());
    comm[l].dup_bcast(basis[l].Rdata.data(), Roffsets[xlen]);
  }
}


void matVecA(const Eval& eval, const Base basis[], const double bodies[], const Cell cells[], const CSR& rels_near, const CSR& rels_far, std::complex<double> X[], const CellComm comm[], int64_t levels) {
  int64_t lbegin = comm[levels].oLocal();
  int64_t llen = comm[levels].lenLocal();

  std::vector<std::vector<std::complex<double>>> rhsX(levels + 1), rhsB(levels + 1);
  std::vector<std::vector<std::complex<double>*>> rhsXptr(levels + 1), rhsXoptr(levels + 1), rhsBptr(levels + 1), rhsBoptr(levels + 1);

  for (int64_t l = levels; l >= 0; l--) {
    int64_t xlen = comm[l].lenNeighbors();
    std::vector<int64_t> offsets(xlen + 1);
    std::inclusive_scan(basis[l].Dims.begin(), basis[l].Dims.end(), offsets.begin() + 1);

    rhsX[l] = std::vector<std::complex<double>>(offsets[xlen], std::complex<double>(0., 0.));
    rhsB[l] = std::vector<std::complex<double>>(offsets[xlen], std::complex<double>(0., 0.));
    rhsXptr[l] = std::vector<std::complex<double>*>(xlen, nullptr);
    rhsBptr[l] = std::vector<std::complex<double>*>(xlen, nullptr);
    rhsXoptr[l] = std::vector<std::complex<double>*>(xlen, nullptr);
    rhsBoptr[l] = std::vector<std::complex<double>*>(xlen, nullptr);

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
    int64_t ibegin = comm[i].oLocal();
    int64_t iboxes = comm[i].lenLocal();
    int64_t xlen = comm[i].lenNeighbors();

    int64_t lenI = std::accumulate(&basis[i].Dims[0], &basis[i].Dims[xlen], 0);
    comm[i].level_merge(rhsX[i].data(), lenI);
    comm[i].neighbor_bcast(rhsX[i].data(), basis[i].Dims.data());
    comm[i].dup_bcast(rhsX[i].data(), lenI);

    for (int64_t j = 0; j < iboxes; j++) {
      Matrix Xj = (Matrix) { rhsXptr[i][j + ibegin], basis[i].Dims[j + ibegin], 1, basis[i].Dims[j + ibegin] };
      Matrix Xo = (Matrix) { rhsXoptr[i][j + ibegin], basis[i].DimsLr[j + ibegin], 1, basis[i].DimsLr[j + ibegin] };
      mmult('T', 'N', &basis[i].Uo[j + ibegin], &Xj, &Xo, std::complex<double>(1., 0.), std::complex<double>(0., 0.));
    }
  }

  if (basis[0].Dims[0] > 0) {
    comm[0].level_merge(rhsX[0].data(), basis[0].Dims[0]);
    comm[0].dup_bcast(rhsX[0].data(), basis[0].Dims[0]);
  }

  for (int64_t i = 1; i <= levels; i++) {
    int64_t ibegin = comm[i].oLocal();
    int64_t iboxes = comm[i].lenLocal();
    int64_t gbegin = comm[i].oGlobal();

    for (int64_t y = 0; y < iboxes; y++)
      for (int64_t yx = rels_far.RowIndex[y + gbegin]; yx < rels_far.RowIndex[y + gbegin + 1]; yx++) {
        int64_t x = comm[i].iLocal(rels_far.ColIndex[yx]);
        int64_t M = basis[i].DimsLr[y + ibegin];
        int64_t N = basis[i].DimsLr[x];

        Matrix Xo = (Matrix) { rhsXoptr[i][x], N, 1, N };
        Matrix Bo = (Matrix) { rhsBoptr[i][y + ibegin], M, 1, M };
        std::vector<std::complex<double>> TMPX(N, std::complex<double>(0., 0.)), TMPB(M, std::complex<double>(0., 0.));
        Matrix T1 = (Matrix) { &TMPX[0], N, 1, N };
        Matrix T2 = (Matrix) { &TMPB[0], M, 1, M };

        mmult('T', 'N', &basis[i].R[x], &Xo, &T1, std::complex<double>(1., 0.), std::complex<double>(0., 0.));
        mat_vec_reference(eval, M, N, &TMPB[0], &TMPX[0], basis[i].ske_at_i(y + ibegin), basis[i].ske_at_i(x));
        mmult('N', 'N', &basis[i].R[y + ibegin], &T2, &Bo, std::complex<double>(1., 0.), std::complex<double>(1., 0.));
      }
  }
  
  for (int64_t i = 1; i <= levels; i++) {
    int64_t ibegin = comm[i].oLocal();
    int64_t iboxes = comm[i].lenLocal();
    for (int64_t j = 0; j < iboxes; j++) {
      Matrix Bj = (Matrix) { rhsBptr[i][j + ibegin], basis[i].Dims[j + ibegin], 1, basis[i].Dims[j + ibegin] };
      Matrix Bo = (Matrix) { rhsBoptr[i][j + ibegin], basis[i].DimsLr[j + ibegin], 1, basis[i].DimsLr[j + ibegin] };
      mmult('N', 'N', &basis[i].Uo[j + ibegin], &Bo, &Bj, std::complex<double>(1., 0.), std::complex<double>(1., 0.));
    }
  }

  int64_t gbegin = comm[levels].oGlobal();
  for (int64_t y = 0; y < llen; y++)
    for (int64_t yx = rels_near.RowIndex[y + gbegin]; yx < rels_near.RowIndex[y + gbegin + 1]; yx++) {
      int64_t x = rels_near.ColIndex[yx];
      int64_t x_loc = comm[levels].iLocal(x);
      int64_t M = cells[y + gbegin].Body[1] - cells[y + gbegin].Body[0];
      int64_t N = cells[x].Body[1] - cells[x].Body[0];
      mat_vec_reference(eval, M, N, rhsBptr[levels][y + lbegin], rhsXptr[levels][x_loc], &bodies[3 * cells[y + gbegin].Body[0]], &bodies[3 * cells[x].Body[0]]);
    }

  std::copy(rhsBptr[levels][lbegin], rhsBptr[levels][lbegin] + lenX, X);
}


void solveRelErr(double* err_out, const std::complex<double>* X, const std::complex<double>* ref, int64_t lenX) {
  double err[2] = { 0., 0. };
  for (int64_t i = 0; i < lenX; i++) {
    std::complex<double> diff = X[i] - ref[i];
    err[0] = err[0] + (diff.real() * diff.real());
    err[1] = err[1] + (ref[i].real() * ref[i].real());
  }
  MPI_Allreduce(MPI_IN_PLACE, err, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  *err_out = std::sqrt(err[0] / err[1]);
}

