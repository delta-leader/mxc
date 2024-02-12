
#include <comm.hpp>
#include <build_tree.hpp>

#include <algorithm>
#include <set>
#include <numeric>
#include <cmath>

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

void getNextLevelMapping(std::pair<int64_t, int64_t> Mapping[], const Cell cells[], int64_t mpi_size) {
  int64_t p = 0;
  std::vector<std::pair<int64_t, int64_t>> MappingNext(mpi_size, std::make_pair(-1, -1));

  while (p < mpi_size) {
    int64_t lenP = std::distance(&Mapping[p], std::find_if_not(&Mapping[p], &Mapping[mpi_size], 
      [&](std::pair<int64_t, int64_t> a) { return a == Mapping[p]; }));
    int64_t pbegin = Mapping[p].first;
    int64_t pend = Mapping[p].second;
    int64_t child = cells[pbegin].Child[0];
    int64_t lenC = cells[pend - 1].Child[1] - child;

    if (child >= 0 && lenC > 0)
      for (int64_t j = 0; j < lenP; j++) {
        int64_t c0 = j * lenC / lenP;
        int64_t c1 = (j + 1) * lenC / lenP;
        c1 = std::max(c1, c0 + 1);
        MappingNext[p + j] = std::make_pair(child + c0, child + c1);
      }
    p += lenP;
  }
  std::copy(MappingNext.begin(), MappingNext.end(), Mapping);
}

CellComm::CellComm(const Cell cells[], std::pair<int64_t, int64_t> Mapping[], const CSR& Near, const CSR& Far, std::vector<MPI_Comm>& unique_comms, MPI_Comm world) {
  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(world, &mpi_rank);
  MPI_Comm_size(world, &mpi_size);

  int64_t pbegin = Mapping[mpi_rank].first;
  int64_t pend = Mapping[mpi_rank].second;
  int64_t p = std::distance(&Mapping[0], std::find(&Mapping[0], &Mapping[mpi_rank], Mapping[mpi_rank]));
  int64_t lenp = std::distance(&Mapping[p], 
    std::find_if_not(&Mapping[p], &Mapping[mpi_size], [&](std::pair<int64_t, int64_t> i) { return i == Mapping[mpi_rank]; }));

  auto col_to_mpi_rank = [&](int64_t col) { return std::distance(&Mapping[0], std::find_if(&Mapping[0], &Mapping[mpi_size], 
    [=](std::pair<int64_t, int64_t> i) { return i.first <= col && col < i.second; })); };
  std::set<int64_t> cols;
  std::for_each(Near.ColIndex.begin() + Near.RowIndex[pbegin], Near.ColIndex.begin() + Near.RowIndex[pend], [&](int64_t col) { cols.insert(col_to_mpi_rank(col)); });
  std::for_each(Far.ColIndex.begin() + Far.RowIndex[pbegin], Far.ColIndex.begin() + Far.RowIndex[pend], [&](int64_t col) { cols.insert(col_to_mpi_rank(col)); });

  std::vector<int64_t> NeighborRanks(cols.begin(), cols.end());
  Proc = std::distance(NeighborRanks.begin(), std::find(NeighborRanks.begin(), NeighborRanks.end(), p));
  NeighborComm = std::vector<std::pair<int, MPI_Comm>>(p == mpi_rank ? NeighborRanks.size() : 0);
  BoxOffsets = std::vector<int64_t>(NeighborRanks.size() + 1);
  BoxOffsets[0] = 0;

  std::vector<std::pair<int64_t, int64_t>> BoxesVector;
  for (int64_t i = 0; i < (int64_t)NeighborRanks.size(); i++) {
    int64_t ibegin = Mapping[NeighborRanks[i]].first;
    int64_t iend = Mapping[NeighborRanks[i]].second;
    std::set<int64_t> icols;
    std::for_each(Near.ColIndex.begin() + Near.RowIndex[ibegin], Near.ColIndex.begin() + Near.RowIndex[iend], [&](int64_t col) { icols.insert(col_to_mpi_rank(col)); });
    std::for_each(Far.ColIndex.begin() + Far.RowIndex[ibegin], Far.ColIndex.begin() + Far.RowIndex[iend], [&](int64_t col) { icols.insert(col_to_mpi_rank(col)); });

    BoxOffsets[i + 1] = BoxOffsets[i] + icols.size();
    BoxesVector.resize(BoxOffsets[i + 1]);
    std::transform(icols.begin(), icols.end(), BoxesVector.begin() + BoxOffsets[i], 
      [&](int64_t rank) { return std::make_pair(Mapping[rank].first, Mapping[rank].second - Mapping[rank].first); });

    if (p == mpi_rank)
      NeighborComm[i].first = std::distance(icols.begin(), std::find(icols.begin(), icols.end(), NeighborRanks[i]));
  }
  Boxes = std::vector<std::pair<int64_t, int64_t>>(BoxesVector.begin(), BoxesVector.end());

  int64_t k = 0;
  for (int i = 0; i < mpi_size; i++) {
    k = std::distance(NeighborRanks.begin(), std::find_if(NeighborRanks.begin() + k, NeighborRanks.end(), [=](int64_t a) { return (int64_t)i <= a; }));
    MPI_Comm comm = MPI_Comm_split_unique(unique_comms, (p == mpi_rank && k != (int64_t)NeighborRanks.size() && NeighborRanks[k] == i) ? 1 : MPI_UNDEFINED, mpi_rank, world);
    if (comm != MPI_COMM_NULL)
      NeighborComm[k].second = comm;
  }

  getNextLevelMapping(&Mapping[0], cells, mpi_size);
  int64_t p_next = std::distance(&Mapping[0], std::find(&Mapping[0], &Mapping[mpi_rank], Mapping[mpi_rank]));
  MergeComm = MPI_Comm_split_unique(unique_comms, (lenp > 1 && mpi_rank == p_next) ? p : MPI_UNDEFINED, mpi_rank, world);
  DupComm = MPI_Comm_split_unique(unique_comms, lenp > 1 ? p : MPI_UNDEFINED, mpi_rank, world);
}

std::pair<int64_t, int64_t> local_to_pnx(int64_t ilocal, const std::pair<int64_t, int64_t> ProcBoxes[], int64_t size) {
  int64_t iter = 0;
  while (iter < size && ProcBoxes[iter].second <= ilocal) {
    ilocal = ilocal - ProcBoxes[iter].second;
    iter = iter + 1;
  }
  if (0 <= ilocal && iter < size)
    return std::make_pair(iter, ilocal);
  else
    return std::make_pair(-1, -1);
}

std::pair<int64_t, int64_t> global_to_pnx(int64_t iglobal, const std::pair<int64_t, int64_t> ProcBoxes[], int64_t size) {
  int64_t iter = 0;
  while (iter < size && (ProcBoxes[iter].first + ProcBoxes[iter].second) <= iglobal)
    iter = iter + 1;
  if (iter < size && ProcBoxes[iter].first <= iglobal)
    return std::make_pair(iter, iglobal - ProcBoxes[iter].first);
  else
    return std::make_pair(-1, -1);
}

int64_t pnx_to_local(std::pair<int64_t, int64_t> pnx, const std::pair<int64_t, int64_t> ProcBoxes[], int64_t size) {
  if (pnx.first >= 0 && pnx.first < size && pnx.second >= 0) {
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

int64_t pnx_to_global(std::pair<int64_t, int64_t> pnx, const std::pair<int64_t, int64_t> ProcBoxes[], int64_t size) {
  if (pnx.first >= 0 && pnx.first < size && pnx.second >= 0)
    return pnx.second + ProcBoxes[pnx.first].first;
  else
    return -1;
}

int64_t CellComm::iLocal(int64_t iglobal) const {
  int64_t len = BoxOffsets[Proc + 1] - BoxOffsets[Proc];
  return 0 <= Proc ? pnx_to_local(global_to_pnx(iglobal, &Boxes[BoxOffsets[Proc]], len), &Boxes[BoxOffsets[Proc]], len) : -1;
}

int64_t CellComm::iNeighbors(int64_t iglobal) const {
  return 0 <= Proc ? pnx_to_local(global_to_pnx(iglobal, &Boxes[0], BoxOffsets.back()), &Boxes[0], BoxOffsets.back()) : -1;
}

int64_t CellComm::iGlobal(int64_t ilocal) const {
    int64_t len = BoxOffsets[Proc + 1] - BoxOffsets[Proc];
  return 0 <= Proc ? pnx_to_global(local_to_pnx(ilocal, &Boxes[BoxOffsets[Proc]], len), &Boxes[BoxOffsets[Proc]], len) : -1;
}

int64_t CellComm::oLocal() const {
  return 0 <= Proc ? std::accumulate(Boxes.begin() + BoxOffsets[Proc], Boxes.begin() + BoxOffsets[Proc] + Proc, 0,
    [](const int64_t& init, const std::pair<int64_t, int64_t>& p) { return init + p.second; }) : -1;
}

int64_t CellComm::oGlobal() const {
  return 0 <= Proc ? Boxes[BoxOffsets[Proc] + Proc].first : -1;
}

int64_t CellComm::lenLocal() const {
  return 0 <= Proc ? Boxes[BoxOffsets[Proc] + Proc].second : 0;
}

int64_t CellComm::lenNeighbors() const {
  return 0 <= Proc ? std::accumulate(Boxes.begin() + BoxOffsets[Proc], Boxes.begin() + BoxOffsets[Proc + 1], 0,
    [](const int64_t& init, const std::pair<int64_t, int64_t>& p) { return init + p.second; }) : 0; 
}

int64_t CellComm::lenNeighborOfNeighbors() const {
  return 0 <= Proc ? std::accumulate(Boxes.begin(), Boxes.end(), 0,
    [](const int64_t& init, const std::pair<int64_t, int64_t>& p) { return init + p.second; }) : 0;
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
  if (MergeComm != MPI_COMM_NULL) {
    record_mpi();
    MPI_Allreduce(MPI_IN_PLACE, data, len, get_mpi_datatype<T>(), MPI_SUM, MergeComm);
    record_mpi();
  }
}

template<typename T> inline void CellComm::dup_bast(T* data, int64_t len) const {
  if (DupComm != MPI_COMM_NULL) {
    record_mpi();
    MPI_Bcast(data, len, get_mpi_datatype<T>(), 0, DupComm);
    record_mpi();
  }
}

template<typename T> inline void CellComm::neighbor_bcast(T* data, const int64_t box_dims[]) const {
  if (NeighborComm.size() > 0) {
    std::vector<int64_t> offsets(NeighborComm.size() + 1, 0);
    for (int64_t p = 0; p < (int64_t)NeighborComm.size(); p++) {
      int64_t end = Boxes[BoxOffsets[Proc] + p].second;
      offsets[p + 1] = std::reduce(box_dims, &box_dims[end], offsets[p]);
      box_dims = &box_dims[end];
    }

    record_mpi();
    for (int64_t p = 0; p < (int64_t)NeighborComm.size(); p++) {
      int64_t llen = offsets[p + 1] - offsets[p];
      MPI_Bcast(&data[offsets[p]], llen, get_mpi_datatype<T>(), NeighborComm[p].first, NeighborComm[p].second);
    }
    record_mpi();
  }
}

template<typename T> inline void CellComm::neighbor_reduce(T* data, const int64_t box_dims[]) const {
  if (NeighborComm.size() > 0) {
    std::vector<int64_t> offsets(NeighborComm.size() + 1, 0);
    for (int64_t p = 0; p < (int64_t)NeighborComm.size(); p++) {
      int64_t end = Boxes[BoxOffsets[Proc] + p].second;
      offsets[p + 1] = std::reduce(box_dims, &box_dims[end], offsets[p]);
      box_dims = &box_dims[end];
    }

    record_mpi();
    for (int64_t p = 0; p < (int64_t)NeighborComm.size(); p++) {
      int64_t llen = offsets[p + 1] - offsets[p];
      if (p == Proc)
        MPI_Reduce(MPI_IN_PLACE, &data[offsets[p]], llen, get_mpi_datatype<T>(), MPI_SUM, NeighborComm[p].first, NeighborComm[p].second);
      else
        MPI_Reduce(&data[offsets[p]], &data[offsets[p]], llen, get_mpi_datatype<T>(), MPI_SUM, NeighborComm[p].first, NeighborComm[p].second);
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
