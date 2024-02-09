
#include <comm.hpp>
#include <build_tree.hpp>

#include <algorithm>
#include <set>
#include <numeric>
#include <cmath>

std::pair<int64_t, int64_t> local_to_pnx(int64_t ilocal, const std::vector<std::pair<int64_t, int64_t>>& ProcBoxes) {
  int64_t iter = 0;
  while (iter < (int64_t)ProcBoxes.size() && ProcBoxes[iter].second <= ilocal) {
    ilocal = ilocal - ProcBoxes[iter].second;
    iter = iter + 1;
  }
  if (0 <= ilocal && iter < (int64_t)ProcBoxes.size())
    return std::make_pair(iter, ilocal);
  else
    return std::make_pair(-1, -1);
}

std::pair<int64_t, int64_t> global_to_pnx(int64_t iglobal, const std::vector<std::pair<int64_t, int64_t>>& ProcBoxes) {
  int64_t iter = 0;
  while (iter < (int64_t)ProcBoxes.size() && (ProcBoxes[iter].first + ProcBoxes[iter].second) <= iglobal)
    iter = iter + 1;
  if (iter < (int64_t)ProcBoxes.size() && ProcBoxes[iter].first <= iglobal)
    return std::make_pair(iter, iglobal - ProcBoxes[iter].first);
  else
    return std::make_pair(-1, -1);
}

int64_t pnx_to_local(std::pair<int64_t, int64_t> pnx, const std::vector<std::pair<int64_t, int64_t>>& ProcBoxes) {
  if (pnx.first >= 0 && pnx.first < (int64_t)ProcBoxes.size() && pnx.second >= 0) {
    int64_t iter = 0, slen = 0;
    while (iter < pnx.first) {
      slen = slen + ProcBoxes[iter].second;
      iter = iter + 1;
    }
    return pnx.second + slen;
  }
  else
    return -1;
}

int64_t pnx_to_global(std::pair<int64_t, int64_t> pnx, const std::vector<std::pair<int64_t, int64_t>>& ProcBoxes) {
  if (pnx.first >= 0 && pnx.first < (int64_t)ProcBoxes.size() && pnx.second >= 0)
    return pnx.second + ProcBoxes[pnx.first].first;
  else
    return -1;
}

MPI_Comm MPI_Comm_split_unique(std::vector<MPI_Comm>& unique_comms, int color, int mpi_rank, MPI_Comm world) {
  MPI_Comm comm = MPI_COMM_NULL;
  MPI_Comm_split(world, color, mpi_rank, &comm);

  if (comm != MPI_COMM_NULL) {
    auto iter = std::find_if(unique_comms.begin(), unique_comms.end(), [comm](MPI_Comm c) -> bool { 
      int result; MPI_Comm_compare(comm, c, &result); return result == MPI_IDENT || result == MPI_CONGRUENT; });
    if (iter == unique_comms.end())
      unique_comms.emplace_back(comm);
    else {
      MPI_Comm_free(&comm);
      comm = *iter;
    }
  }
  return comm;
}

CellComm::CellComm(int64_t lbegin, int64_t lend, int64_t cbegin, int64_t cend, const std::vector<std::pair<int64_t, int64_t>>& ProcMapping, const CSR& Near, const CSR& Far, std::vector<MPI_Comm>& unique_comms, MPI_Comm world) {
  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(world, &mpi_rank);
  MPI_Comm_size(world, &mpi_size);

  int64_t p = ProcMapping[lbegin].first;
  int64_t lenp = ProcMapping[lbegin].second - p;

  std::vector<int64_t> ProcTargets;
  for (int64_t i = 0; i < (int64_t)mpi_size; i++) {
    std::set<int64_t> cols;
    cols.insert(Near.ColIndex.begin() + Near.RowIndex[lbegin], Near.ColIndex.begin() + Near.RowIndex[lend]);
    cols.insert(Far.ColIndex.begin() + Far.RowIndex[lbegin], Far.ColIndex.begin() + Far.RowIndex[lend]);
    
    if (cols.size() > 0 && (cols.end() != std::find_if(cols.begin(), cols.end(), [&](const int64_t col) { return ProcMapping[col].first == i; })))
      ProcTargets.emplace_back(i);
  }
  Proc = std::distance(ProcTargets.begin(), std::find(ProcTargets.begin(), ProcTargets.end(), p));
  ProcBoxes = std::vector<std::pair<int64_t, int64_t>>(ProcTargets.size());
  ProcBoxesNeighbors = std::vector<std::vector<std::pair<int64_t, int64_t>>>(ProcTargets.size());
  CommBox = std::vector<std::pair<int, MPI_Comm>>(p == mpi_rank ? ProcTargets.size() : 0);
  ProcBoxes[Proc] = std::make_pair(lbegin, lend - lbegin);

  int64_t k = 0;
  for (int i = 0; i < mpi_size; i++) {
    k = std::distance(ProcTargets.begin(), std::find_if(ProcTargets.begin() + k, ProcTargets.end(), [=](int64_t a) { return (int64_t)i <= a; }));
    MPI_Comm comm = MPI_Comm_split_unique(unique_comms, (p == mpi_rank && k != (int64_t)ProcTargets.size() && ProcTargets[k] == i) ? 1 : MPI_UNDEFINED, mpi_rank, world);
    if (comm != MPI_COMM_NULL) {
      int root = 0;
      if (i == mpi_rank)
        MPI_Comm_rank(comm, &root);
      MPI_Allreduce(MPI_IN_PLACE, &root, 1, MPI_INT, MPI_SUM, comm);
      CommBox[k] = std::make_pair(root, comm);
      MPI_Bcast(&ProcBoxes[k], sizeof(std::pair<int64_t, int64_t>), MPI_BYTE, root, comm);
    }
  }

  int color_merge = (lenp > 1 && cbegin >= 0 && cend >= 0) && (&ProcMapping[cend] != std::find_if(&ProcMapping[cbegin], &ProcMapping[cend], 
    [=](const std::pair<int64_t, int64_t>& a) { return a.first == (int64_t)mpi_rank; })) ? p : MPI_UNDEFINED;
  Comm_merge = MPI_Comm_split_unique(unique_comms, color_merge, mpi_rank, world);
  Comm_share = MPI_Comm_split_unique(unique_comms, lenp > 1 ? p : MPI_UNDEFINED, mpi_rank, world);

  if (Comm_share != MPI_COMM_NULL)
    MPI_Bcast(&ProcBoxes[0], sizeof(std::pair<int64_t, int64_t>) * ProcBoxes.size(), MPI_BYTE, 0, Comm_share);
}

int64_t CellComm::iLocal(int64_t iglobal) const {
  return pnx_to_local(global_to_pnx(iglobal, ProcBoxes), ProcBoxes);
}

int64_t CellComm::iGlobal(int64_t ilocal) const {
  return pnx_to_global(local_to_pnx(ilocal, ProcBoxes), ProcBoxes);
}

int64_t CellComm::oLocal() const {
  return (Proc >= 0 && Proc < (int64_t)ProcBoxes.size()) ? std::accumulate(ProcBoxes.begin(), ProcBoxes.begin() + Proc, 0,
    [](const int64_t& init, const std::pair<int64_t, int64_t>& p) { return init + p.second; }) : -1;
}

int64_t CellComm::oGlobal() const {
  return (Proc >= 0 && Proc < (int64_t)ProcBoxes.size()) ? ProcBoxes[Proc].first : -1;
}

int64_t CellComm::lenLocal() const {
  return (Proc >= 0 && Proc < (int64_t)ProcBoxes.size()) ? ProcBoxes[Proc].second : -1;
}

int64_t CellComm::lenNeighbors() const {
  return std::accumulate(ProcBoxes.begin(), ProcBoxes.end(), 0,
    [](const int64_t& init, const std::pair<int64_t, int64_t>& p) { return init + p.second; });
}

template<typename T> inline MPI_Datatype get_mpi_datatype() {
  if (typeid(T) == typeid(int64_t))
    return MPI_INT64_T;
  if (typeid(T) == typeid(double))
    return MPI_DOUBLE;
  if (typeid(T) == typeid(std::complex<double>))
    return MPI_DOUBLE_COMPLEX;
  return MPI_DATATYPE_NULL;
}

template<typename T> inline void CellComm::level_merge(T* data, int64_t len) const {
  if (Comm_merge != MPI_COMM_NULL) {
    record_mpi();
    MPI_Allreduce(MPI_IN_PLACE, data, len, get_mpi_datatype<T>(), MPI_SUM, Comm_merge);
    record_mpi();
  }
}

template<typename T> inline void CellComm::dup_bast(T* data, int64_t len) const {
  if (Comm_share != MPI_COMM_NULL) {
    record_mpi();
    MPI_Bcast(data, len, get_mpi_datatype<T>(), 0, Comm_share);
    record_mpi();
  }
}

template<typename T> inline void CellComm::neighbor_bcast(T* data, const int64_t box_dims[]) const {
  if (CommBox.size() > 0) {
    std::vector<int64_t> offsets(CommBox.size() + 1, 0);
    for (int64_t p = 0; p < (int64_t)CommBox.size(); p++) {
      int64_t end = ProcBoxes[p].second;
      offsets[p + 1] = std::accumulate(box_dims, &box_dims[end], offsets[p]);
      box_dims = &box_dims[end];
    }

    record_mpi();
    for (int64_t p = 0; p < (int64_t)CommBox.size(); p++) {
      int64_t llen = offsets[p + 1] - offsets[p];
      MPI_Bcast(&data[offsets[p]], llen, get_mpi_datatype<T>(), CommBox[p].first, CommBox[p].second);
    }
    record_mpi();
  }
}

template<typename T> inline void CellComm::neighbor_reduce(T* data, const int64_t box_dims[]) const {
  if (CommBox.size() > 0) {
    std::vector<int64_t> offsets(CommBox.size() + 1, 0);
    for (int64_t p = 0; p < (int64_t)CommBox.size(); p++) {
      int64_t end = ProcBoxes[p].second;
      offsets[p + 1] = std::accumulate(box_dims, &box_dims[end], offsets[p]);
      box_dims = &box_dims[end];
    }

    record_mpi();
    for (int64_t p = 0; p < (int64_t)CommBox.size(); p++) {
      int64_t llen = offsets[p + 1] - offsets[p];
      if (p == Proc)
        MPI_Reduce(MPI_IN_PLACE, &data[offsets[p]], llen, get_mpi_datatype<T>(), MPI_SUM, CommBox[p].first, CommBox[p].second);
      else
        MPI_Reduce(&data[offsets[p]], &data[offsets[p]], llen, get_mpi_datatype<T>(), MPI_SUM, CommBox[p].first, CommBox[p].second);
    }
    record_mpi();
  }
}

void CellComm::level_merge(std::complex<double>* data, int64_t len) const {
  level_merge<std::complex<double>>(data, len);
}

void CellComm::dup_bcast(int64_t* data, int64_t len) const {
  dup_bast<int64_t>(data, len);
}

void CellComm::dup_bcast(double* data, int64_t len) const {
  dup_bast<double>(data, len);
}

void CellComm::dup_bcast(std::complex<double>* data, int64_t len) const {
  dup_bast<std::complex<double>>(data, len);
}

void CellComm::neighbor_bcast(int64_t* data, const int64_t box_dims[]) const {
  neighbor_bcast<int64_t>(data, box_dims);
}

void CellComm::neighbor_bcast(double* data, const int64_t box_dims[]) const {
  neighbor_bcast<double>(data, box_dims);
}

void CellComm::neighbor_bcast(std::complex<double>* data, const int64_t box_dims[]) const {
  neighbor_bcast<std::complex<double>>(data, box_dims);
}

void CellComm::neighbor_reduce(std::complex<double>* data, const int64_t box_dims[]) const {
  neighbor_reduce<std::complex<double>>(data, box_dims);
}

void CellComm::record_mpi() const {
  if (timer && timer->second == 0.)
    timer->second = MPI_Wtime();
  else if (timer) {
    timer->first = MPI_Wtime() - timer->second;
    timer->second = 0.;
  }
}
