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
find_package(Threads REQUIRED)

# CUDA Architecture configuration
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

# CUDA flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
    -O3 
    -lineinfo 
    --use_fast_math
    -Xcompiler -fopenmp
)

# CPU optimization flags - AVX2 ONLY (no AVX-512)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native")

# Check for AVX2 support ONLY
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-mavx2" COMPILER_SUPPORTS_AVX2)

if(COMPILER_SUPPORTS_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
    add_definitions(-DUSE_AVX2)
    message(STATUS "AVX2 support enabled")
else()
    message(WARNING "AVX2 not supported, using scalar operations")
endif()

# OpenMP flags
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/src/simd
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CUDA_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

# CUDA library
cuda_add_library(info_efficiency_cuda SHARED 
    src/cuda/variance_ratio.cu
)

target_link_libraries(info_efficiency_cuda
    ${CUDA_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)

# SIMD library - WITH AVX2 ONLY
add_library(info_efficiency_simd SHARED 
    src/simd/simd_operations.cpp
)

target_link_libraries(info_efficiency_simd
    OpenMP::OpenMP_CXX
    ${CMAKE_THREAD_LIBS_INIT}
    m
)

# Explicitly set AVX2 only for SIMD library
target_compile_definitions(info_efficiency_simd PRIVATE USE_AVX2)
target_compile_options(info_efficiency_simd PRIVATE -mavx2 -mfma)

# Core library
add_library(info_efficiency_core SHARED 
    src/core/polygon_reader.cpp
)

target_link_libraries(info_efficiency_core
    ${CMAKE_THREAD_LIBS_INIT}
)

# Installation
install(TARGETS info_efficiency_cuda info_efficiency_simd info_efficiency_core
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

# Print configuration summary
message(STATUS "========================================")
message(STATUS "Configuration Summary:")
message(STATUS "  CUDA Version: ${CUDA_VERSION}")
message(STATUS "  Python3 Version: ${Python3_VERSION}")
message(STATUS "  OpenMP Found: ${OpenMP_CXX_FOUND}")
message(STATUS "  AVX2 Support: ${COMPILER_SUPPORTS_AVX2}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "========================================")
