# Edge-AI Dual Task Client

A simple C++ application that communicates with both TISP (endpoint 13) and TVM (endpoint 20) tasks running on the C7x DSP.

## Overview

This example demonstrates basic RPMsg communication with two independent tasks:
- **TISP Client**: Communicates with generic/TISP processing task (endpoint 13)  
- **TVM Client**: Communicates with TVM inference task (endpoint 20)

## Features

-  **Auto-detects C7x DSP** by device tree address (`7e000000.dsp`)
-  **Uses TVM runtime** directly with RPMsg communication
-  **Simple ping/status testing** for both endpoints
-  **Configurable** via config.txt
-  **Extensible** for future DMA buffers and full TVM integration

## Prerequisites

1. **C7x DSP firmware** loaded with dual-task application
2. **Neo-TVM source** available
3. **Both endpoints (13 and 20)** responding on C7x

## Build

```bash
# Method 1: Set NEO_TVM_PATH environment variable
export NEO_TVM_PATH=/path/to/neo-tvm
mkdir build
cd build
cmake ..
make

# Method 2: Specify directly to cmake
mkdir build
cd build
cmake -DTVM_ROOT=/path/to/neo-tvm ..
make
```

## Run

```bash
cd build
./edge_ai_client

# Or with custom config:
./edge_ai_client ../custom_config.txt
```

## Configuration

Edit `config.txt` to customize:

```
TISP_ENDPOINT=13        # TISP task endpoint
TVM_ENDPOINT=20         # TVM task endpoint  
ENABLE_TISP=1           # Enable TISP testing
ENABLE_TVM=1            # Enable TVM testing
TEST_ITERATIONS=3       # Number of ping tests
```

## Expected Output

```
 Edge-AI Dual Task Client
============================
Found C7x DSP: 7e000000.dsp -> remoteproc1
=== Edge-AI Configuration ===
C7x Proc ID:    1
TISP Endpoint:  13
TVM Endpoint:   20

 === Testing TISP Client (Endpoint 13) ===
[TISP]  Connected successfully (fd=3)
[TISP]  Sending ping: "ping from tisp client"
[TISP]  Received response: "[C7x Task 1 #1]: ping from tisp client"
[TISP]  PING successful - Task 1 responded!

 === Testing TVM Client (Endpoint 20) ===  
[TVM]  Connected successfully (fd=4)
[TVM]  Sending ping: "ping from tvm client"  
[TVM]  Received response: "[C7x Task 2 #1]: ping from tvm client"
[TVM]  PING successful - Task 2 responded!
```

## Troubleshooting

1. **"C7x remoteproc not found"**:
   ```bash
   # Check if DSP is loaded:
   cat /sys/class/remoteproc/remoteproc*/name
   # Should show: 7e000000.dsp
   ```

2. **"Failed to connect to endpoint X"**:
   ```bash
   # Check DSP traces:
   cat /sys/kernel/debug/remoteproc/remoteproc1/trace0
   # Should show both tasks started
   ```

3. **Build errors**:
   ```bash
   # Check NEO_TVM_PATH:
   echo $NEO_TVM_PATH
   ls -la $NEO_TVM_PATH/src/runtime/ti_dsp/firmware/c7x/
   ```

## File Structure

```
edge-ai/
├── CMakeLists.txt                   # Build configuration
├── README.md                        # This file
└── src/
    ├── main.cpp                     # Main application and RPMsg handling
    ├── tvm_inference_client.cpp     # TVM runtime integration
    └── tvm_inference_client.h       # TVM client header
```
