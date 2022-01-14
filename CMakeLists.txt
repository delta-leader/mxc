
cmake_minimum_required (VERSION 3.9)

project ("shimo" LANGUAGES CXX)

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_executable(main
  main.cxx
)
target_link_libraries(main ${MPI_C_LIBRARIES})