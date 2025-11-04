cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#enable_language(Fortran)

project ("shimo" LANGUAGES CXX CUDA Fortran)

set(BLA_VENDOR OpenBLAS)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(CUDAToolkit REQUIRED)

file(GLOB sources "*.cpp" "*.cu")
file(GLOB headers "*.hpp" "*.cuh")
list(REMOVE_AT sources -1)

add_subdirectory(
  BEM3D_transmission_each_component
)

add_executable(main.app ${sources} ${headers})
target_link_libraries(main.app BLAS::BLAS LAPACK::LAPACK lapacke OpenMP::OpenMP_CXX MPI::MPI_CXX Eigen3::Eigen CUDA::cudart CUDA::cublas CUDA::cusparse nccl elastWave3d)

target_compile_definitions(main.app PRIVATE EIGEN_USE_BLAS EIGEN_USE_LAPACKE)
target_include_directories(main.app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(main.app PRIVATE "-Wall" "-Wextra" "-fexceptions" "$<$<CONFIG:DEBUG>:-O0;-g3;-ggdb>" "-fPIE")
endif()

file(GLOB wsources "*.cpp" "*.cu")
list(REMOVE_AT wsources -4)
add_executable(write.app ${wsources} ${headers})
target_link_libraries(write.app BLAS::BLAS LAPACK::LAPACK lapacke OpenMP::OpenMP_CXX MPI::MPI_CXX Eigen3::Eigen CUDA::cudart CUDA::cublas CUDA::cusparse nccl elastWave3d)

target_compile_definitions(write.app PRIVATE EIGEN_USE_BLAS EIGEN_USE_LAPACKE)
target_include_directories(write.app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(write.app PRIVATE "-Wall" "-Wextra" "-fexceptions" "$<$<CONFIG:DEBUG>:-O0;-g3;-ggdb>" "-fPIE")
endif()
