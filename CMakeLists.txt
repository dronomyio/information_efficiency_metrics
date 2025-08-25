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

# Find pybind11 (optional)
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_include())"
        OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()

# CUDA Architecture configuration
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Detect GPU architecture
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    string(REPLACE "+" "" CUDA_ARCH_LIST "${CUDA_ARCH_LIST}")
    
    if(CUDA_ARCH_LIST)
        set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
    else()
        # Default architectures
        set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
    endif()
endif()

message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# CUDA flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
    -O3 
    -lineinfo 
    --use_fast_math
    -Xcompiler -fopenmp
    --expt-extended-lambda
    --expt-relaxed-constexpr
)

# Add gencode flags for each architecture
foreach(arch ${CMAKE_CUDA_ARCHITECTURES})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_${arch},code=sm_${arch})
endforeach()

# CPU optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native")

# Check for AVX support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-mavx512f" COMPILER_SUPPORTS_AVX512)
CHECK_CXX_COMPILER_FLAG("-mavx2" COMPILER_SUPPORTS_AVX2)

if(COMPILER_SUPPORTS_AVX512)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl")
    add_definitions(-D__AVX512F__)
    message(STATUS "AVX-512 support enabled")
elseif(COMPILER_SUPPORTS_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
    add_definitions(-D__AVX2__)
    message(STATUS "AVX2 support enabled")
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

if(PYBIND11_INCLUDE_DIR)
    include_directories(${PYBIND11_INCLUDE_DIR})
endif()

# CUDA library
cuda_add_library(info_efficiency_cuda SHARED 
    src/cuda/variance_ratio.cu
)

target_link_libraries(info_efficiency_cuda
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    ${CMAKE_THREAD_LIBS_INIT}
)

# Enable CUDA separable compilation for complex kernels
set_property(TARGET info_efficiency_cuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET info_efficiency_cuda PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# SIMD library
add_library(info_efficiency_simd SHARED 
    src/simd/simd_operations.cpp
)

target_link_libraries(info_efficiency_simd
    OpenMP::OpenMP_CXX
    ${CMAKE_THREAD_LIBS_INIT}
    m  # Math library
)

# Set SIMD-specific compile flags
if(COMPILER_SUPPORTS_AVX512)
    target_compile_definitions(info_efficiency_simd PRIVATE USE_AVX512)
elseif(COMPILER_SUPPORTS_AVX2)
    target_compile_definitions(info_efficiency_simd PRIVATE USE_AVX2)
endif()

# Core library
add_library(info_efficiency_core SHARED 
    src/core/polygon_reader.cpp
)

target_link_libraries(info_efficiency_core
    ${CMAKE_THREAD_LIBS_INIT}
)

# Python bindings
add_library(info_efficiency_cpp MODULE 
    src/bindings/python_bindings.cpp
)

target_link_libraries(info_efficiency_cpp PRIVATE 
    info_efficiency_cuda 
    info_efficiency_simd 
    info_efficiency_core
    ${Python3_LIBRARIES}
)

set_target_properties(info_efficiency_cpp PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

# Test executables (optional)
option(BUILD_TESTS "Build test executables" OFF)
if(BUILD_TESTS)
    # CUDA test
    cuda_add_executable(test_variance_ratio tests/test_variance_ratio.cpp)
    target_link_libraries(test_variance_ratio info_efficiency_cuda)
    
    # SIMD test
    add_executable(test_autocorrelation tests/test_autocorrelation.cpp)
    target_link_libraries(test_autocorrelation info_efficiency_simd)
endif()

# Installation
install(TARGETS info_efficiency_cuda info_efficiency_simd info_efficiency_core
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(TARGETS info_efficiency_cpp
        LIBRARY DESTINATION ${Python3_SITELIB})

# Print configuration summary
message(STATUS "========================================")
message(STATUS "Configuration Summary:")
message(STATUS "  CUDA Version: ${CUDA_VERSION}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Python3 Version: ${Python3_VERSION}")
message(STATUS "  OpenMP Found: ${OpenMP_CXX_FOUND}")
message(STATUS "  AVX-512 Support: ${COMPILER_SUPPORTS_AVX512}")
message(STATUS "  AVX2 Support: ${COMPILER_SUPPORTS_AVX2}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "========================================")
