cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_STANDARD 11)
project(p3)

include_directories(   
    src
)

set(THREAD_FILES
    src/thread_dft.cc
    src/complex.h
    src/complex.cc
    src/input_image.h
    src/input_image.cc
)

set(MPI_FILES
    src/mpi_dft.cc
    src/complex.h
    src/complex.cc
    src/input_image.h
    src/input_image.cc
)

set(CUDA_FILES
    src/cuda_dft.cu
    src/complex.cuh
    src/input_image.cuh
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -pthread")
add_executable(p31 ${THREAD_FILES})

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
add_executable(p32 ${MPI_FILES})
target_link_libraries(p32 ${MPI_LIBRARIES})

find_package(CUDA QUIET REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -g -std=c++11)
cuda_add_executable(p33 ${CUDA_FILES})
link_directories()
