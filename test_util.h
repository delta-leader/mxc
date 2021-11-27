
#pragma once

#include "linalg.hxx"
#include <iostream>

void printA(const nbd::Matrix& A) {
  int m = A.M;
  int n = A.N;
  const double* d = A.A.data();
  for (int y = 0; y < m; y++) {
    for (int x = 0; x < n; x++)
      std::cout << d[y + x * m] << " ";
    std::cout << std::endl;
  }
}

void printX(const nbd::Vector& A) {
  int n = A.N;
  const double* d = A.X.data();
  for (int x = 0; x < n; x++)
    std::cout << d[x] << " ";
  std::cout << std::endl;
}
