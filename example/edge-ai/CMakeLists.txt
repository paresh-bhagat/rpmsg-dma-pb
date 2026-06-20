# TVM Inference Client - uses TVM runtime with existing Python artifacts
# Replicates functionality of inference.py using C++ and TVM runtime

cmake_minimum_required(VERSION 3.10)
project(tvm_inference_client)

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

# Source files (TVM runtime approach)
set(EDGE_AI_SOURCES
    src/main.cpp
    src/tvm_inference_client.cpp
)

# Create the executable
add_executable(tvm_inference_client ${EDGE_AI_SOURCES})

# Include directories
target_include_directories(tvm_inference_client PRIVATE
    src
    ${TVM_ROOT}/include
    ${TVM_ROOT}/3rdparty/dlpack/include
    ${TVM_ROOT}/3rdparty/dmlc-core/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../library/include
)

# Preprocessor definitions
target_compile_definitions(tvm_inference_client PRIVATE
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

# Link with required libraries
target_link_libraries(tvm_inference_client
    ${TVM_RUNTIME_LIB}
    ti_rpmsg_dma
    pthread
    dl
)

# Install target
install(TARGETS tvm_inference_client DESTINATION ${CMAKE_INSTALL_BINDIR})