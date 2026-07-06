# TVM Inference Client - uses TVM runtime with existing Python artifacts
# Replicates functionality of inference.py using C++ and TVM runtime

cmake_minimum_required(VERSION 3.10)
project(rpmsg_inference_example)

# Enable C++ support
enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)

# Find TVM installation (neo-TVM)
# User must specify via: cmake -DTVM_ROOT=/path/to/neo-tvm ..
if(NOT DEFINED TVM_ROOT)
    # Try environment variable
    set(TVM_ROOT $ENV{NEO_TVM_PATH})
    if(NOT TVM_ROOT)
        message(FATAL_ERROR "TVM_ROOT not specified. Please set via:\n"
                           "  cmake -DTVM_ROOT=/path/to/neo-tvm ..\n"
                           "Or set environment variable:\n"
                           "  export NEO_TVM_PATH=/path/to/neo-tvm")
    endif()
endif()

message(STATUS "Using TVM_ROOT: ${TVM_ROOT}")

# Source files (Interactive Pipeline Application)
set(EDGE_AI_SOURCES
    src/main.cpp
    src/tvm_inference_client.cpp
    src/generic_task_client.cpp
    src/pipeline_manager.cpp
)

# Create the executable
add_executable(rpmsg_inference_example ${EDGE_AI_SOURCES})

# Include directories
target_include_directories(rpmsg_inference_example PRIVATE
    include
    ${TVM_ROOT}/include
    ${TVM_ROOT}/3rdparty/dlpack/include
    ${TVM_ROOT}/3rdparty/dmlc-core/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../library/include
)

# Preprocessor definitions
target_compile_definitions(rpmsg_inference_example PRIVATE
    DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>
)

# Find TVM runtime library
if(NOT TVM_RUNTIME_LIB)
    # Use the ARM64 build (for local builds)
    set(TVM_ARM64_BUILD_PATH "${TVM_ROOT}/build_aarch64_py314")
    set(TVM_RUNTIME_LIB "${TVM_ARM64_BUILD_PATH}/libtvm_runtime.so")
endif()

# Verify the library exists
if(NOT EXISTS ${TVM_RUNTIME_LIB})
    message(FATAL_ERROR "TVM runtime not found: ${TVM_RUNTIME_LIB}")
endif()

message(STATUS "Using TVM runtime: ${TVM_RUNTIME_LIB}")

# Find json-c library for JSON pipeline configuration
find_library(JSON_C_LIB json-c REQUIRED)
find_path(JSON_C_INCLUDE_DIR json-c/json.h
    PATHS /usr/include /usr/local/include
    REQUIRED)

# Add json-c include directory
target_include_directories(rpmsg_inference_example PRIVATE ${JSON_C_INCLUDE_DIR})

# Add cnpy for NPZ loading
set(CNPY_DIR "${TVM_ROOT}/3rdparty/cnpy")
include_directories(${CNPY_DIR})
add_library(cnpy STATIC ${CNPY_DIR}/cnpy.cpp)
target_include_directories(cnpy PRIVATE ${CNPY_DIR})
target_include_directories(rpmsg_inference_example PRIVATE ${CNPY_DIR})

# Find readline library for interactive mode with command line editing
find_library(READLINE_LIB readline REQUIRED)

# Find audio libraries for WAV file processing and playback
find_library(SNDFILE_LIB sndfile REQUIRED)
find_library(ALSA_LIB asound REQUIRED)

# Link with required libraries
target_link_libraries(rpmsg_inference_example
    ${TVM_RUNTIME_LIB}
    ti_rpmsg_dma
    ${JSON_C_LIB}
    ${READLINE_LIB}
    ${SNDFILE_LIB}
    ${ALSA_LIB}
    cnpy
    z
    pthread
    dl
)

# Install target
install(TARGETS rpmsg_inference_example DESTINATION ${CMAKE_INSTALL_BINDIR})