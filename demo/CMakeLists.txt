cmake_minimum_required(VERSION 3.16)

project(yolo_infer)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

aux_source_directory(../demo DIR_DEMOS)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -march=native")

add_executable(yolo_infer ${DIR_DEMOS} image_util.hpp image_util.cpp)
target_include_directories(yolo_infer PUBLIC ../include)
target_link_directories(yolo_infer PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(yolo_infer ${OpenCV_LIBS} magic)
