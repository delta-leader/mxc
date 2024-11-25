cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project ("shimo" LANGUAGES CXX CUDA)

set(BLA_VENDOR Intel10_64lp)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(CUDAToolkit REQUIRED)

file(GLOB sources "*.cpp" "*.cu")
file(GLOB headers "*.hpp" "*.cuh")

add_executable(main.app ${sources} ${headers})
target_link_libraries(main.app BLAS::BLAS LAPACK::LAPACK lapacke OpenMP::OpenMP_CXX MPI::MPI_CXX Eigen3::Eigen CUDA::cudart CUDA::cublas CUDA::cusparse nccl)

target_compile_definitions(main.app PRIVATE EIGEN_USE_BLAS EIGEN_USE_LAPACKE)
target_include_directories(main.app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(main.app PRIVATE "-Wall" "-Wextra" "-fexceptions" "-O3" "$<$<CONFIG:DEBUG>:-O0;-g3;-ggdb>")
endif()
