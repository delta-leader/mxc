
#include <complex>
#include <mkl.h>
#include <Eigen/Dense>


inline double get_real(double value) {
  return value;
}

inline float get_real(float value) {
  return value;
}

inline double get_real(std::complex<double> value) {
  return std::real(value);
}

inline float get_real(std::complex<float> value) {
  return std::real(value);
}

inline Eigen::half get_real(Eigen::half value) {
  return value;
}

inline Eigen::half get_real(std::complex<Eigen::half> value) {
  return std::real(value);
}
