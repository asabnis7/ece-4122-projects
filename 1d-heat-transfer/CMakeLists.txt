cmake_minimum_required(VERSION 3.9)
project(se2)

find_package(MPI REQUIRED)

include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

add_executable(heat1d src/heat1d.cc)

target_link_libraries(heat1d ${MPI_LIBRARIES})
