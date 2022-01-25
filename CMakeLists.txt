
cmake_minimum_required (VERSION 3.9)

project ("shimo" LANGUAGES C CXX)

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(./)

add_library(shimo
  minblas.c minblas.h
  domain.cxx domain.hxx
  kernel.cxx kernel.hxx
  linalg.cxx linalg.hxx
  bodies.cxx bodies.hxx
  basis.cxx basis.hxx
  umv.cxx umv.hxx
  solver.cxx solver.hxx
  dist.cxx dist.hxx
)

add_executable(main
  main.cxx
)
target_link_libraries(main shimo ${MPI_C_LIBRARIES})
