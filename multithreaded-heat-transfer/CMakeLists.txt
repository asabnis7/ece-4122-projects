cmake_minimum_required (VERSION 3.8)
project (se3)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -pthread")
find_package(Threads)

include_directories(
        src
)

add_executable(
	heat2D3D 
	src/heat2D3D.cc
    )

target_link_libraries(
    heat2D3D 
    ${CMAKE_THREAD_LIBS_INIT}
    )
