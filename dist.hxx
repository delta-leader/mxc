
#pragma once

#include "bodies.hxx"
#include "basis.hxx"
#include "umv.hxx"

namespace nbd {

  void DistributeBodies(LocalBodies& bodies, const GlobalIndex& gi);

  void DistributeVectorsList(Vectors& B, const GlobalIndex& gi);

  void DistributeMatricesList(Matrices& lis, const GlobalIndex& gi);

  void DistributeDims(std::vector<int64_t>& dims, const GlobalIndex& gi);

  void axatDistribute(Matrices& A, const GlobalIndex& gi);

  void butterflySum(Matrices& A, const GlobalIndex& gi);

};