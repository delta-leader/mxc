#pragma once

#include <vector>
#include <cstdint>
#include <complex>

class CSR;
class MatVecBasis;
class CellComm;

class ULV {
public:
  std::vector<int64_t> Dims;
  std::vector<int64_t> DimsLr;
  std::vector<int64_t> Ranks;

  std::vector<std::complex<double>*> Q;
  std::vector<std::complex<double>*> R;
  std::vector<std::complex<double>> Qdata;
  std::vector<std::complex<double>> Rdata;

  std::vector<std::complex<double>*> A;
  std::vector<int64_t> X;
  std::vector<int64_t> Y;
  std::vector<std::complex<double>> Adata;

  std::vector<std::complex<double>*> F;
  std::vector<int64_t> FX;
  std::vector<int64_t> FY;
  std::vector<std::complex<double>> Fdata;

  ULV(const int64_t dims[], const int64_t dims_lr[], const CSR& Near, const CellComm& comm);
  
};

