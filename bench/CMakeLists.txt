find_package(benchmark REQUIRED) 
aux_source_directory(../bench DIR_BENCH)
set(link_lib benchmark::benchmark benchmark::benchmark_main)

add_executable(bench_magic ${DIR_BENCH})
target_link_directories(bench_magic PUBLIC ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -march=native")

target_link_directories(bench_magic PUBLIC ${PROJECT_SOURCE_DIR}/lib /usr/lib/x86_64-linux-gnu)
target_link_libraries(bench_magic ${link_lib} OpenMP::OpenMP_CXX)
target_link_libraries(bench_magic magic)

target_include_directories(bench_magic PUBLIC ${benchmark_INCLUDE_DIRS})
target_include_directories(bench_magic PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(bench_magic PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(bench_magic PUBLIC ${Armadillo_INCLUDE_DIR})
