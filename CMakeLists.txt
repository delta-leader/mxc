
cmake_minimum_required (VERSION 3.9)

project ("shimo" LANGUAGES C CXX)

find_package(MPI REQUIRED)

message(STATUS ${MPI_INCLUDE_PATH})
message(STATUS ${MPI_C_LIBRARIES})
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_library(nbd STATIC 
  minblas.c minblas.h
  build_tree.cxx build_tree.hxx
  kernel.cxx kernel.hxx
  linalg.cxx linalg.hxx
  basis.cxx basis.hxx
  umv.cxx umv.hxx
  h2mv.cxx h2mv.hxx
  solver.cxx solver.hxx
  dist.cxx dist.hxx
)
target_include_directories(nbd PRIVATE "./" )
target_compile_options(nbd PRIVATE ${MPI_C_COMPILE_FLAGS})
target_link_libraries(nbd ${MPI_C_LIBRARIES} ${MPI_C_LINK_FLAGS})

add_executable(h2_example
  h2_example.cpp
)
target_link_libraries(h2_example nbd)
target_include_directories(h2_example PRIVATE "./" )

add_executable(lra_example
  lra_example.cpp
)
target_link_libraries(lra_example nbd)
target_include_directories(lra_example PRIVATE "./" )

add_executable(main
  main.cxx
)
target_compile_options(main PRIVATE ${MPI_C_COMPILE_FLAGS})
target_link_libraries(main nbd ${MPI_C_LIBRARIES} ${MPI_C_LINK_FLAGS})

