
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

CellComm::CellComm(int64_t P, const std::vector<int64_t>& Targets, int64_t cbegin, int64_t clen, int merge, int share, std::vector<MPI_Comm>& unique_comms, MPI_Comm world) : 
  Proc(P), ProcBoxes(Targets.size()), Comm_box(), Comm_share(MPI_COMM_NULL), Comm_merge(MPI_COMM_NULL), timer(nullptr) {
  int mpi_rank, mpi_size;
  MPI_Comm_rank(world, &mpi_rank);
  MPI_Comm_size(world, &mpi_size);

  std::vector<int64_t>::const_iterator iter = Targets.begin();
  for (int i = 0; i < mpi_size; i++) {
    iter = std::find_if(iter, Targets.end(), [=](const int64_t& a) { return i <= a; });
    int color = MPI_UNDEFINED;
    if (Targets[P] == mpi_rank && iter != Targets.end() && *iter == i)
      color = 1;
    MPI_Comm comm = MPI_Comm_split_unique(unique_comms, color, mpi_rank, world);

    if (comm != MPI_COMM_NULL) {
      int root = 0;
      if (i == mpi_rank)
        MPI_Comm_rank(comm, &root);
      MPI_Allreduce(MPI_IN_PLACE, &root, 1, MPI_INT, MPI_SUM, comm);
      Comm_box.emplace_back(root, comm);
    }
  }

  ProcBoxes[P] = std::make_pair(cbegin, clen);
  for (int64_t i = 0; i < (int64_t)Comm_box.size(); i++)
    MPI_Bcast(&ProcBoxes[i], sizeof(std::pair<int64_t, int64_t>), MPI_BYTE, Comm_box[i].first, Comm_box[i].second);

  Comm_merge = MPI_Comm_split_unique(unique_comms, merge, mpi_rank, world);
  Comm_share = MPI_Comm_split_unique(unique_comms, share, mpi_rank, world);

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
  return std::accumulate(ProcBoxes.begin(), ProcBoxes.begin() + Proc, 0,
    [](const int64_t& init, const std::pair<int64_t, int64_t>& p) { return init + p.second; });
}

int64_t CellComm::oGlobal() const {
  return ProcBoxes[Proc].first;
}

int64_t CellComm::lenLocal() const {
  return ProcBoxes[Proc].second;
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
  if (Comm_box.size() > 0) {
    std::vector<int64_t> offsets(Comm_box.size() + 1, 0);
    for (int64_t p = 0; p < (int64_t)Comm_box.size(); p++) {
      int64_t end = ProcBoxes[p].second;
      offsets[p + 1] = std::accumulate(box_dims, &box_dims[end], offsets[p]);
      box_dims = &box_dims[end];
    }

    record_mpi();
    for (int64_t p = 0; p < (int64_t)Comm_box.size(); p++) {
      int64_t llen = offsets[p + 1] - offsets[p];
      MPI_Bcast(&data[offsets[p]], llen, get_mpi_datatype<T>(), Comm_box[p].first, Comm_box[p].second);
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

void CellComm::record_mpi() const {
  if (timer && timer->second == 0.)
    timer->second = MPI_Wtime();
  else if (timer) {
    timer->first = MPI_Wtime() - timer->second;
    timer->second = 0.;
  }
}

void get_level_procs(std::vector<std::pair<int64_t, int64_t>>& Procs, std::vector<std::pair<int64_t, int64_t>>& Levels, 
  int64_t mpi_rank, int64_t mpi_size, const std::vector<std::pair<int64_t, int64_t>>& Child, int64_t levels) {
  int64_t ncells = (int64_t)Child.size();
  std::vector<int64_t> levels_cell(ncells);
  Procs[0] = std::make_pair(0, mpi_size);
  levels_cell[0] = 0;

  for (int64_t i = 0; i < ncells; i++) {
    int64_t child = Child[i].first;
    int64_t lenC = Child[i].second;
    int64_t lenP = Procs[i].second - Procs[i].first;
    int64_t p = Procs[i].first;
    
    if (child >= 0 && lenC > 0) {
      double divP = (double)lenP / (double)lenC;
      for (int64_t j = 0; j < lenC; j++) {
        int64_t p0 = j == 0 ? 0 : (int64_t)std::floor(j * divP);
        int64_t p1 = j == (lenC - 1) ? lenP : (int64_t)std::floor((j + 1) * divP);
        p1 = std::max(p1, p0 + 1);
        Procs[child + j] = std::make_pair(p + p0, p + p1);
        levels_cell[child + j] = levels_cell[i] + 1;
      }
    }
  }
  
  int64_t begin = 0;
  for (int64_t i = 0; i <= levels; i++) {
    int64_t ibegin = std::distance(levels_cell.begin(), 
      std::find(levels_cell.begin() + begin, levels_cell.end(), i));
    int64_t iend = std::distance(levels_cell.begin(), 
      std::find(levels_cell.begin() + begin, levels_cell.end(), i + 1));
    int64_t pbegin = std::distance(Procs.begin(), 
      std::find_if(Procs.begin() + ibegin, Procs.begin() + iend, [=](std::pair<int64_t, int64_t>& p) -> bool {
        return p.first <= mpi_rank && mpi_rank < p.second;
      }));
    int64_t pend = std::distance(Procs.begin(), 
      std::find_if_not(Procs.begin() + pbegin, Procs.begin() + iend, [=](std::pair<int64_t, int64_t>& p) -> bool {
        return p.first <= mpi_rank && mpi_rank < p.second;
      }));
    Levels[i] = std::make_pair(pbegin, pend);
    begin = iend;
  }
}

std::vector<MPI_Comm> buildComm(CellComm* comms, int64_t ncells, const Cell* cells, const CSR& cellFar, const CSR& cellNear, int64_t levels, MPI_Comm world) {
  int __mpi_rank = 0, __mpi_size = 1;
  MPI_Comm_rank(world, &__mpi_rank);
  MPI_Comm_size(world, &__mpi_size);
  int64_t mpi_rank = __mpi_rank;
  int64_t mpi_size = __mpi_size;

  std::vector<MPI_Comm> unique_comms;
  std::vector<std::pair<int64_t, int64_t>> Child(ncells), Procs(ncells), Levels(levels + 1);
  std::transform(cells, &cells[ncells], Child.begin(), [](const Cell& c) {
    return std::make_pair(c.Child[0], c.Child[1] - c.Child[0]);
  });
  get_level_procs(Procs, Levels, mpi_rank, mpi_size, Child, levels);

  for (int64_t l = levels; l >= 0; l--) {
    int64_t mbegin = Levels[l].first;
    int64_t mend = Levels[l].second;
    int64_t p = Procs[mbegin].first;
    int64_t lenp = Procs[mbegin].second - p;

    std::vector<int64_t> ProcTargets;
    for (int64_t i = 0; i < mpi_size; i++) {
      std::set<int64_t> unique_cols;
      unique_cols.insert(cellNear.ColIndex.begin() + cellNear.RowIndex[mbegin], cellNear.ColIndex.begin() + cellNear.RowIndex[mend]);
      unique_cols.insert(cellFar.ColIndex.begin() + cellFar.RowIndex[mbegin], cellFar.ColIndex.begin() + cellFar.RowIndex[mend]);
      bool is_ngb = unique_cols.size() > 0 ? (unique_cols.end() != std::find_if(unique_cols.begin(), unique_cols.end(), 
        [&](const int64_t col) { return Procs[col].first == i; })) : false;
      
      if (is_ngb)
        ProcTargets.emplace_back(i);
    }
    int64_t Proc_d = std::distance(ProcTargets.begin(), std::find(ProcTargets.begin(), ProcTargets.end(), p));

    int color = MPI_UNDEFINED;
    int64_t cc = Child[mbegin].first;
    int64_t clen = Child[mbegin].second;
    if (lenp > 1 && cc >= 0)
      color = std::find_if(&Procs[cc], &Procs[cc + clen], 
        [=](const std::pair<int64_t, int64_t>& a) { return a.first == mpi_rank; }) == &Procs[cc + clen] ? MPI_UNDEFINED : p;

    comms[l] = CellComm(Proc_d, ProcTargets, mbegin, mend - mbegin, color, lenp > 1 ? p : MPI_UNDEFINED, unique_comms, world);
  }

  return unique_comms;
}

/*void neighbor_reduce_cpu(double* data, int64_t seg, const CellComm* comm) {
  if (comm->Comm_box.size() > 0) {
    comm->record_mpi();
    int64_t y = 0;
    for (int64_t p = 0; p < (int64_t)comm->Comm_box.size(); p++) {
      int64_t llen = comm->ProcBoxes[p].second * seg;
      double* loc = &data[y];
      if (p == comm->Proc)
        MPI_Reduce(MPI_IN_PLACE, loc, llen, MPI_DOUBLE, MPI_SUM, comm->Comm_box[p].first, comm->Comm_box[p].second);
      else
        MPI_Reduce(loc, loc, llen, MPI_DOUBLE, MPI_SUM, comm->Comm_box[p].first, comm->Comm_box[p].second);
      y = y + llen;
    }
    comm->record_mpi();
  }
}*/

