cmake_minimum_required(VERSION 3.2)
project(p2)

find_package(CUDA QUIET REQUIRED)

set(
    CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
	-g -std=c++11
    )

include_directories(
	src
	)

cuda_add_executable(
	heat2D3D
	src/heat2D3D.cu
	)

link_directories()
