
#pragma once

#include "bodies.hxx"
#include "solver.hxx"

namespace nbd {

  void DistributeBodies(LocalBodies& bodies, const GlobalIndex& gi);

  void DistributeVectorsList(Vectors& B, const GlobalIndex& gi);

  void DistributeMatricesList(Matrices& lis, const GlobalIndex& gi);

  void DistributeDims(std::vector<int64_t>& dims, const GlobalIndex& gi);

  void axatDistribute(Matrices& A, const GlobalIndex& gi);

  void butterflySumA(Matrices& A, const GlobalIndex& gi);

  void recvSubstituted(char fwbk, Vectors& X, const GlobalIndex& gi);

  void sendSubstituted(char fwbk, const Vectors& X, const GlobalIndex& gi);

  void distributeSubstituted(Vectors& X, const GlobalIndex& gi);

  void butterflySumX(Vectors& X, const GlobalIndex& gi);

};