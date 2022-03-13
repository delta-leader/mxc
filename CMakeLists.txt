
cmake_minimum_required (VERSION 3.9)

project ("shimo" LANGUAGES C CXX)

find_package(MPI REQUIRED)

message(STATUS ${MPI_INCLUDE_PATH})
message(STATUS ${MPI_C_LIBRARIES})
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(./)

add_executable(main
  minblas.c minblas.h
  build_tree.cxx build_tree.hxx
  kernel.cxx kernel.hxx
  linalg.cxx linalg.hxx
  bodies.cxx bodies.hxx
  basis.cxx basis.hxx
  umv.cxx umv.hxx
  h2mv.cxx h2mv.hxx
  solver.cxx solver.hxx
  dist.cxx dist.hxx
  main.cxx
)
target_compile_options(main PRIVATE ${MPI_C_COMPILE_FLAGS})
target_link_libraries(main ${MPI_C_LIBRARIES} ${MPI_C_LINK_FLAGS})

