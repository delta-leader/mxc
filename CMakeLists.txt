
cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)

project ("H2-ULV-MA" LANGUAGES CXX CUDA)

set(MKL_ARCH intel64)
set(MKL_LINK dynamic)
set(MKL_INTERFACE_FULL intel_lp64)
set(MKL_THREADING sequential)
set(MKL_MPI intelmpi)

find_package(MKL REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

file(GLOB sources "*.cpp")
file(GLOB headers "*.hpp")

add_executable(main main.cpp ${sources} ${headers})
target_include_directories(main PRIVATE ./)
target_link_libraries(main MPI::MPI_CXX OpenMP::OpenMP_CXX MKL::MKL -L$ENV{CUDADIR}/lib64 -lcublas -lcudart -lcusolver -lnccl)
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "60;70;72;80")


