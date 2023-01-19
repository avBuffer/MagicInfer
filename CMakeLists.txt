cmake_minimum_required(VERSION 3.16)

project(magic_infer)
set(CMAKE_CXX_STANDARD 17)
include_directories(./include)

option(USE_CUDA   "Support Cuda Backend")
option(BUILD_DEMO "Build The Demo Project" )

set(USE_CUDA   ON)
set(BUILD_DEMO ON)

if (BUILD_DEMO)
    MESSAGE(STATUS "Build Demo Project")
    add_subdirectory(demo)
endif ()

if (USE_CUDA)
    MESSAGE(STATUS "Support Cuda Backend")
    add_definitions(-DUSE_CUDA)
endif ()

find_package(benchmark REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Armadillo REQUIRED)
find_package(glog REQUIRED)

aux_source_directory(./src/data DIR_DATA)
aux_source_directory(./src/runtime DIR_PARSER)
aux_source_directory(./src/layer/abstract DIR_ABSTRACT_LAYER)
aux_source_directory(./src/layer/details DIR_DETAILS_LAYER)
aux_source_directory(./src/parser DIR_PARSER)
aux_source_directory(./src/nets DIR_NETS)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set(link_lib glog::glog pthread)
set(link_math_lib ${ARMADILLO_LIBRARIES} blas lapack)

add_library(magic SHARED ${DIR_DATA} ${DIR_PARSER} ${DIR_ABSTRACT_LAYER} ${DIR_DETAILS_LAYER} ${DIR_PARSER} ${DIR_NETS}
    include/utils/platform.hpp)
target_link_libraries(magic ${link_lib} ${link_math_lib} OpenMP::OpenMP_CXX)

target_include_directories(magic PUBLIC ${benchmark_INCLUDE_DIRS})
target_include_directories(magic PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(magic PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(magic PUBLIC ${Armadillo_INCLUDE_DIR})

enable_testing()
add_subdirectory(bench)
add_subdirectory(test)
