cmake_minimum_required(VERSION 3.16)
project(InfoEfficiency LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find packages
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# CUDA configuration
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -lineinfo --use_fast_math)

# CPU optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -mavx512f -mavx512dq -mavx512vl")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -pthread")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDA_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/external/rapidjson/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/arrow/include
)

# External libraries
set(ARROW_LIBS arrow parquet)
set(ZSTD_LIBS zstd)
set(CURL_LIBS curl)

# CUDA sources
set(CUDA_SOURCES
    src/cuda/variance_ratio.cu
    src/cuda/autocorrelation.cu
    src/cuda/multi_gpu_manager.cu
)

# SIMD sources
set(SIMD_SOURCES
    src/simd/simd_operations.cpp
    src/simd/avx_kernels.cpp
)

# Core sources
set(CORE_SOURCES
    src/core/polygon_reader.cpp
)

# Create CUDA library
cuda_add_library(info_efficiency_cuda SHARED ${CUDA_SOURCES})
target_link_libraries(info_efficiency_cuda ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cufft_LIBRARIES})

# Create SIMD library
add_library(info_efficiency_simd SHARED ${SIMD_SOURCES})
target_link_libraries(info_efficiency_simd OpenMP::OpenMP_CXX)

# Create core library
add_library(info_efficiency_core SHARED ${CORE_SOURCES})
target_link_libraries(info_efficiency_core ${ARROW_LIBS} ${ZSTD_LIBS} ${CURL_LIBS})

# Python bindings
pybind11_add_module(info_efficiency_cpp src/bindings/python_bindings.cpp)
target_link_libraries(info_efficiency_cpp PRIVATE 
    info_efficiency_cuda 
    info_efficiency_simd 
    info_efficiency_core
)

# Installation
install(TARGETS info_efficiency_cuda info_efficiency_simd info_efficiency_core
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(TARGETS info_efficiency_cpp
        LIBRARY DESTINATION ${Python3_SITELIB})
