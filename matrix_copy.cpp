
#include <complex>
#include <cstdint>

#include <cblas-openblas.h>

void zomatcopy(char trans, int64_t rows, int64_t cols, const std::complex<double>* src, int64_t ld_src, std::complex<double>* dst, int64_t ld_dst) {
  CBLAS_TRANSPOSE ctrans = (CBLAS_TRANSPOSE)-1;
  if (trans == 'N' || trans == 'n')
    ctrans = CblasNoTrans;
  else if (trans == 'T' || trans == 't')
    ctrans = CblasTrans;
  else if (trans == 'C' || trans == 'c')
    ctrans = CblasConjNoTrans;
  else if (trans == 'H' || trans == 'h')
    ctrans = CblasConjTrans;
  
  std::complex<double> one(1., 0.);
  cblas_zomatcopy(CblasColMajor, ctrans, rows, cols, reinterpret_cast<const double*>(&one), 
    reinterpret_cast<const double*>(src), ld_src, reinterpret_cast<double*>(dst), ld_dst);
}
