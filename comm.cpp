
#include <comm.hpp>
#include <build_tree.hpp>

#include <algorithm>
#include <set>
#include <numeric>

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

void getNextLevelMapping(std::pair<long long, long long> Mapping[], const Cell cells[], long long mpi_size) {
  long long p = 0;
  std::vector<std::pair<long long, long long>> MappingNext(mpi_size, std::make_pair(-1, -1));

  while (p < mpi_size) {
    long long lenP = std::distance(&Mapping[p], std::find_if_not(&Mapping[p], &Mapping[mpi_size], 
      [&](std::pair<long long, long long> a) { return a == Mapping[p]; }));
    long long pbegin = Mapping[p].first;
    long long pend = Mapping[p].second;
    long long child = cells[pbegin].Child[0];
    long long lenC = cells[pend - 1].Child[1] - child;

    if (child >= 0 && lenC > 0)
      for (long long j = 0; j < lenP; j++) {
        long long c0 = j * lenC / lenP;
        long long c1 = (j + 1) * lenC / lenP;
        c1 = std::max(c1, c0 + 1);
        MappingNext[p + j] = std::make_pair(child + c0, child + c1);
      }
    p += lenP;
  }
  std::copy(MappingNext.begin(), MappingNext.end(), Mapping);
}

CellComm::CellComm(const Cell cells[], std::pair<long long, long long> Mapping[], const CSR& Near, const CSR& Far, std::vector<MPI_Comm>& unique_comms, MPI_Comm world) {
  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(world, &mpi_rank);
  MPI_Comm_size(world, &mpi_size);

  long long pbegin = Mapping[mpi_rank].first;
  long long pend = Mapping[mpi_rank].second;
  long long p = std::distance(&Mapping[0], std::find(&Mapping[0], &Mapping[mpi_rank], Mapping[mpi_rank]));
  long long lenp = std::distance(&Mapping[p], 
    std::find_if_not(&Mapping[p], &Mapping[mpi_size], [&](std::pair<long long, long long> i) { return i == Mapping[mpi_rank]; }));

  auto col_to_mpi_rank = [&](long long col) { return std::distance(&Mapping[0], std::find_if(&Mapping[0], &Mapping[mpi_size], 
    [=](std::pair<long long, long long> i) { return i.first <= col && col < i.second; })); };
  std::set<long long> cols;
  std::for_each(Near.ColIndex.begin() + Near.RowIndex[pbegin], Near.ColIndex.begin() + Near.RowIndex[pend], [&](long long col) { cols.insert(col_to_mpi_rank(col)); });
  std::for_each(Far.ColIndex.begin() + Far.RowIndex[pbegin], Far.ColIndex.begin() + Far.RowIndex[pend], [&](long long col) { cols.insert(col_to_mpi_rank(col)); });

  std::vector<long long> NeighborRanks(cols.begin(), cols.end());
  Proc = std::distance(NeighborRanks.begin(), std::find(NeighborRanks.begin(), NeighborRanks.end(), p));
  Boxes = std::vector<std::pair<long long, long long>>(NeighborRanks.size());
  NeighborComm = std::vector<std::pair<int, MPI_Comm>>(p == mpi_rank ? NeighborRanks.size() : 0);

  std::vector<std::pair<long long, long long>> BoxesVector;
  for (long long i = 0; i < (long long)NeighborRanks.size(); i++) {
    long long ibegin = Mapping[NeighborRanks[i]].first;
    long long iend = Mapping[NeighborRanks[i]].second;
    std::set<long long> icols;
    std::for_each(Near.ColIndex.begin() + Near.RowIndex[ibegin], Near.ColIndex.begin() + Near.RowIndex[iend], [&](long long col) { icols.insert(col_to_mpi_rank(col)); });
    std::for_each(Far.ColIndex.begin() + Far.RowIndex[ibegin], Far.ColIndex.begin() + Far.RowIndex[iend], [&](long long col) { icols.insert(col_to_mpi_rank(col)); });

    Boxes[i] = std::make_pair(ibegin, iend - ibegin);
    if (p == mpi_rank)
      NeighborComm[i].first = std::distance(icols.begin(), std::find(icols.begin(), icols.end(), NeighborRanks[i]));
  }

  long long k = 0;
  for (int i = 0; i < mpi_size; i++) {
    k = std::distance(NeighborRanks.begin(), std::find_if(NeighborRanks.begin() + k, NeighborRanks.end(), [=](long long a) { return (long long)i <= a; }));
    MPI_Comm comm = MPI_Comm_split_unique(unique_comms, (p == mpi_rank && k != (long long)NeighborRanks.size() && NeighborRanks[k] == i) ? 1 : MPI_UNDEFINED, mpi_rank, world);
    if (comm != MPI_COMM_NULL)
      NeighborComm[k].second = comm;
  }

  getNextLevelMapping(&Mapping[0], cells, mpi_size);
  long long p_next = std::distance(&Mapping[0], std::find(&Mapping[0], &Mapping[mpi_rank], Mapping[mpi_rank]));
  MergeComm = MPI_Comm_split_unique(unique_comms, (lenp > 1 && mpi_rank == p_next) ? p : MPI_UNDEFINED, mpi_rank, world);
  DupComm = MPI_Comm_split_unique(unique_comms, lenp > 1 ? p : MPI_UNDEFINED, mpi_rank, world);
}

long long CellComm::iLocal(long long iglobal) const {
  std::vector<std::pair<long long, long long>>::const_iterator iter = std::find_if(Boxes.begin(), Boxes.end(), 
    [=](std::pair<long long, long long> i) { return i.first <= iglobal && iglobal < i.first + i.second; });
  return (0 <= iglobal && iter != Boxes.end()) ? (iglobal - (*iter).first + std::accumulate(Boxes.begin(), iter, 0, 
    [](const long long& init, std::pair<long long, long long> i) { return init + i.second; })) : -1;
}

long long CellComm::iGlobal(long long ilocal) const {
  long long iter = 0;
  while (iter < (long long)Boxes.size() && Boxes[iter].second <= ilocal) {
    ilocal = ilocal - Boxes[iter].second;
    iter = iter + 1;
  }
  return (0 <= ilocal && iter <= (long long)Boxes.size()) ? (Boxes[iter].first + ilocal) : -1;
}

long long CellComm::oLocal() const {
  return 0 <= Proc ? std::accumulate(Boxes.begin(), Boxes.begin() + Proc, 0,
    [](const long long& init, const std::pair<long long, long long>& p) { return init + p.second; }) : -1;
}

long long CellComm::oGlobal() const {
  return 0 <= Proc ? Boxes[Proc].first : -1;
}

long long CellComm::lenLocal() const {
  return 0 <= Proc ? Boxes[Proc].second : 0;
}

long long CellComm::lenNeighbors() const {
  return 0 <= Proc ? std::accumulate(Boxes.begin(), Boxes.end(), 0,
    [](const long long& init, const std::pair<long long, long long>& p) { return init + p.second; }) : 0; 
}

template<typename T> inline MPI_Datatype get_mpi_datatype() {
  if (typeid(T) == typeid(long long))
    return MPI_INT64_T;
  if (typeid(T) == typeid(double))
    return MPI_DOUBLE;
  if (typeid(T) == typeid(std::complex<double>))
    return MPI_DOUBLE_COMPLEX;
  return MPI_DATATYPE_NULL;
}

template<typename T> inline void CellComm::level_merge(T* data, long long len) const {
  if (MergeComm != MPI_COMM_NULL) {
    record_mpi();
    MPI_Allreduce(MPI_IN_PLACE, data, len, get_mpi_datatype<T>(), MPI_SUM, MergeComm);
    record_mpi();
  }
}

template<typename T> inline void CellComm::dup_bast(T* data, long long len) const {
  if (DupComm != MPI_COMM_NULL) {
    record_mpi();
    MPI_Bcast(data, len, get_mpi_datatype<T>(), 0, DupComm);
    record_mpi();
  }
}

template<typename T> inline void CellComm::neighbor_bcast(T* data, const long long box_dims[]) const {
  if (NeighborComm.size() > 0) {
    std::vector<long long> offsets(NeighborComm.size() + 1, 0);
    for (long long p = 0; p < (long long)NeighborComm.size(); p++) {
      long long end = Boxes[p].second;
      offsets[p + 1] = std::reduce(box_dims, &box_dims[end], offsets[p]);
      box_dims = &box_dims[end];
    }

    record_mpi();
    for (long long p = 0; p < (long long)NeighborComm.size(); p++) {
      long long llen = offsets[p + 1] - offsets[p];
      MPI_Bcast(&data[offsets[p]], llen, get_mpi_datatype<T>(), NeighborComm[p].first, NeighborComm[p].second);
    }
    record_mpi();
  }
}

template<typename T> inline void CellComm::neighbor_reduce(T* data, const long long box_dims[]) const {
  if (NeighborComm.size() > 0) {
    std::vector<long long> offsets(NeighborComm.size() + 1, 0);
    for (long long p = 0; p < (long long)NeighborComm.size(); p++) {
      long long end = Boxes[p].second;
      offsets[p + 1] = std::reduce(box_dims, &box_dims[end], offsets[p]);
      box_dims = &box_dims[end];
    }

    record_mpi();
    for (long long p = 0; p < (long long)NeighborComm.size(); p++) {
      long long llen = offsets[p + 1] - offsets[p];
      if (p == Proc)
        MPI_Reduce(MPI_IN_PLACE, &data[offsets[p]], llen, get_mpi_datatype<T>(), MPI_SUM, NeighborComm[p].first, NeighborComm[p].second);
      else
        MPI_Reduce(&data[offsets[p]], &data[offsets[p]], llen, get_mpi_datatype<T>(), MPI_SUM, NeighborComm[p].first, NeighborComm[p].second);
    }
    record_mpi();
  }
}

void CellComm::level_merge(std::complex<double>* data, long long len) const {
  level_merge<std::complex<double>>(data, len);
}

void CellComm::dup_bcast(long long* data, long long len) const {
  dup_bast<long long>(data, len);
}

void CellComm::dup_bcast(double* data, long long len) const {
  dup_bast<double>(data, len);
}

void CellComm::dup_bcast(std::complex<double>* data, long long len) const {
  dup_bast<std::complex<double>>(data, len);
}

void CellComm::neighbor_bcast(long long* data, const long long box_dims[]) const {
  neighbor_bcast<long long>(data, box_dims);
}

void CellComm::neighbor_bcast(double* data, const long long box_dims[]) const {
  neighbor_bcast<double>(data, box_dims);
}

void CellComm::neighbor_bcast(std::complex<double>* data, const long long box_dims[]) const {
  neighbor_bcast<std::complex<double>>(data, box_dims);
}

void CellComm::neighbor_reduce(long long* data, const long long box_dims[]) const {
  neighbor_reduce<long long>(data, box_dims);
}

void CellComm::neighbor_reduce(std::complex<double>* data, const long long box_dims[]) const {
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
