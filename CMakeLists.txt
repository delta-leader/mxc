cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project ("shimo" LANGUAGES CXX CUDA)

set(BLA_VENDOR OpenBLAS)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(MPI REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

file(GLOB sources "*.cpp" "*.cu")
file(GLOB headers "*.hpp" "*.cuh")

add_executable(main.app ${sources} ${headers})
target_link_libraries(main.app BLAS::BLAS LAPACK::LAPACK lapacke Eigen3::Eigen MPI::MPI_CXX cudart cublas cusparse cusolver nccl)

target_compile_definitions(main.app PRIVATE EIGEN_USE_BLAS EIGEN_USE_LAPACKE)
target_include_directories(main.app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(main.app PRIVATE "-Wall" "-Wextra" "-fexceptions" "$<$<CONFIG:DEBUG>:-O0;-g3;-ggdb>")
endif()
