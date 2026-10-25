cmake_minimum_required(VERSION 3.18)
project(CUDADehazing CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc highgui)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -lineinfo")

set(UTIL_SOURCES
    src/utils/image_io.cpp
    src/utils/metrics.cpp
)
set(CPU_SOURCES
    src/cpu/dcp_cpu.cpp
    src/cpu/dcp_openmp.cpp
)
set(CUDA_PIPELINE_SOURCES
    src/pipeline/dark_channel.cu
    src/pipeline/atm_light.cu
    src/pipeline/transmission.cu
    src/pipeline/guided_filter.cu
    src/pipeline/recovery.cu
)

add_executable(dehazer
    src/main.cu
    ${UTIL_SOURCES}
    ${CPU_SOURCES}
    ${CUDA_PIPELINE_SOURCES}
)
set_target_properties(dehazer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(dehazer ${OpenCV_LIBS} OpenMP::OpenMP_CXX CUDA::cudart)

add_executable(benchmark
    src/benchmark/batch_bench.cpp
    ${UTIL_SOURCES}
    ${CPU_SOURCES}
    ${CUDA_PIPELINE_SOURCES}
)
set_target_properties(benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(benchmark ${OpenCV_LIBS} OpenMP::OpenMP_CXX CUDA::cudart)
