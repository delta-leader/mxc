#pragma once

#include <gpu_handles.cuh>
#include <complex>

class ColCommMPI;

struct CsrContainer {
  long long M = 0;
  long long N = 0;
  long long NNZ = 0;
  int* RowOffsets = nullptr;
  int* ColInd = nullptr;
  std::complex<double>* Vals = nullptr;
};

struct CsrMatVecDesc_t {
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

typedef struct CsrContainer* CsrContainer_t;

long long computeCooNNZ(long long Mb, const long long RowDims[], const long long ColDims[], const long long ARows[], const long long ACols[]);
void genCsrEntries(long long csrM, long long devRowIndx[], long long devColIndx[], std::complex<double> devVals[], long long Mb, long long Nb, const long long RowDims[], const long long ColDims[], const long long ARows[], const long long ACols[]);
void createDeviceCsr(CsrContainer_t* A, long long Mb, long long Nb, const long long RowDims[], const long long ColDims[], const long long ARows[], const long long ACols[], const std::complex<double> data[]);

void createSpMatrixDesc(CsrMatVecDesc_t* desc, bool is_leaf, long long lowerZ, const long long Dims[], const long long Ranks[], const std::complex<double> U[], const std::complex<double> C[], const std::complex<double> A[], const ColCommMPI& comm);
void destroySpMatrixDesc(CsrMatVecDesc_t desc);

void matVecUpwardPass(CsrMatVecDesc_t desc, const std::complex<double>* X_in, const ColCommMPI& comm);
void matVecHorizontalandDownwardPass(CsrMatVecDesc_t desc, std::complex<double>* Y_out);
void matVecLeafHorizontalPass(CsrMatVecDesc_t desc, std::complex<double>* X_io, const ColCommMPI& comm);

constexpr int hint_number = 512;
