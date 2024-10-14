#pragma once

#include <vector>
#include <complex>

class ColCommMPI;

class CsrMatVecDesc_t {
public:
  long long lenX = 0;
  long long lenZ = 0;
  long long xbegin = 0;
  long long zbegin = 0;
  long long lowerZ = 0;
  
  std::complex<double>* X = nullptr;
  std::complex<double>* Y = nullptr;
  std::complex<double>* Z = nullptr;
  std::complex<double>* W = nullptr;
  long long* NeighborX = nullptr;
  long long* NeighborZ = nullptr;

  int* RowOffsetsU = nullptr;
  int* ColIndU = nullptr;
  std::complex<double>* ValuesU = nullptr;

  int* RowOffsetsC = nullptr;
  int* ColIndC = nullptr;
  std::complex<double>* ValuesC = nullptr;

  int* RowOffsetsA = nullptr;
  int* ColIndA = nullptr;
  std::complex<double>* ValuesA = nullptr;

  void* descV = nullptr;
  void* descU = nullptr;
  void* descC = nullptr;
  void* descA = nullptr;
};

constexpr int hint_number = 512;

void convertCsrEntries(int RowOffsets[], int Columns[], std::complex<double> Values[], long long Mb, long long Nb, const long long RowDims[], const long long ColDims[], const long long ARows[], const long long ACols[], const std::complex<double>* DataPtrs[], const long long LDs[] = nullptr);

void createSpMatrixDesc(CsrMatVecDesc_t* desc, bool is_leaf, long long lowerZ, const long long Dims[], const long long Ranks[], const std::complex<double>* U[], const std::complex<double>* C[], const std::complex<double>* A[], const ColCommMPI& comm);
void destroySpMatrixDesc(CsrMatVecDesc_t desc);

void matVecUpwardPass(CsrMatVecDesc_t desc, const std::complex<double>* X_in, const ColCommMPI& comm);
void matVecHorizontalandDownwardPass(CsrMatVecDesc_t desc, std::complex<double>* Y_out);
void matVecLeafHorizontalPass(CsrMatVecDesc_t desc, std::complex<double>* X_io, const ColCommMPI& comm);

/*class CsrMatVec {
public:
  long long levels;
  

};*/
