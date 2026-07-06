#include "generic_task_client.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <cstring>
#include <iostream>
#include <errno.h>

// Include rpmsg-dma-pb library
extern "C" {
#include "rpmsg.h"
#include "dmabuf.h"
}

// Constants for Generic Task communication
#define C7_PROC_ID      8
#define RMT_EP          13      // Generic Service endpoint

// TVM shared memory regions (zero-copy approach - no offsets)
#define TVM_STAGING_PHYS        0xa3000000ULL   // Pre-process writes here, TVM reads from same location
#define TVM_RESULT_PHYS         0xabc00000ULL   // TVM writes here, post-process reads from same location

// Linux-side protocol definitions (matching firmware exactly)
struct c7x_msg_hdr {
    uint32_t type;              // Must match firmware: uint32_t (not uint16_t!)
    uint32_t seq;               // Must match firmware: uint32_t (not uint16_t!)
    uint32_t len;
    int32_t  status;
} __attribute__((packed));

// Message structures (matching firmware protocol)
struct c7x_msg_pre_process {
    struct c7x_msg_hdr hdr;
    uint32_t data_type;
    uint64_t input_addr;
    uint64_t output_addr;
    uint32_t input_size;
    uint32_t output_size;
    uint32_t num_samples;
    uint32_t num_channels;
} __attribute__((packed));

struct c7x_msg_pre_process_resp {
    struct c7x_msg_hdr hdr;
    int32_t return_value;
    uint64_t cycles;
    uint32_t samples_processed;
    uint32_t dsp_load_percent;
    uint32_t reserved;
} __attribute__((packed));

struct c7x_msg_post_process {
    struct c7x_msg_hdr hdr;
    uint32_t data_type;
    uint64_t input_addr;
    uint64_t output_addr;
    uint32_t input_size;
    uint32_t output_size;
    uint32_t num_samples;
    uint32_t num_channels;
} __attribute__((packed));

struct c7x_msg_post_process_resp {
    struct c7x_msg_hdr hdr;
    int32_t return_value;
    uint64_t cycles;
    uint32_t samples_processed;
    uint32_t dsp_load_percent;
    uint32_t reserved;
} __attribute__((packed));

// Message types (matching firmware protocol)
#define C7X_MSG_PRE_PROCESS          0x0031
#define C7X_MSG_PRE_PROCESS_RESP     0x1031
#define C7X_MSG_POST_PROCESS         0x0032
#define C7X_MSG_POST_PROCESS_RESP    0x1032
#define C7X_STATUS_SUCCESS           0

GenericTaskClient::GenericTaskClient()
    : rpmsg_fd_(-1)
    , initialized_(false)
    , sequence_number_(1)
    , max_input_buffer_size_(1024 * 1024)   // Default 1MB
    , max_output_buffer_size_(1024 * 1024)  // Default 1MB
    , shared_input_buffer_(nullptr)
    , shared_output_buffer_(nullptr)
    , shared_input_addr_(0)
    , shared_output_addr_(0)
{
}

GenericTaskClient::~GenericTaskClient()
{
    shutdown();
}

bool GenericTaskClient::initialize(uint32_t max_input_size, uint32_t max_output_size)
{
    if (initialized_) {
        return true;
    }

    max_input_buffer_size_ = max_input_size;
    max_output_buffer_size_ = max_output_size;


    // Initialize shared buffer addresses (zero-copy)
    if (!allocate_shared_buffers()) {
        std::cerr << "Failed to allocate shared buffers" << std::endl;
        return false;
    }

    // Initialize RPMsg communication
    if (!open_rpmsg_device()) {
        std::cerr << "Failed to open RPMsg device" << std::endl;
        free_shared_buffers();
        return false;
    }

    // Test ping
    if (!ping_service()) {
        std::cerr << "Generic Task ping test failed" << std::endl;
        close_rpmsg_device();
        free_shared_buffers();
        return false;
    }

    initialized_ = true;
    return true;
}

GenericTaskClient::ProcessingResult GenericTaskClient::pre_process(DataType data_type,
                                                                 void* input_data,
                                                                 uint32_t input_size,
                                                                 void* output_data,
                                                                 uint32_t output_size)
{
    ProcessingResult result = {};
    result.success = false;

    if (!initialized_) {
        result.error_message = "Client not initialized";
        return result;
    }

    // Zero-copy approach: DSP will read directly from TVM staging region
    // Input data copying is handled by pipeline manager to TVM staging buffer

    // Prepare Pre-Process message
    struct c7x_msg_pre_process req = {};
    req.hdr.type = C7X_MSG_PRE_PROCESS;
    req.hdr.seq = sequence_number_++;
    req.hdr.len = sizeof(struct c7x_msg_pre_process);
    req.hdr.status = 0;

    req.data_type = static_cast<uint32_t>(data_type);
    req.input_addr = shared_input_addr_;   // TVM staging physical address (0xa3000000)
    req.output_addr = shared_input_addr_;  // Pre-process writes back to TVM staging (in-place)
    req.input_size = input_size;
    req.output_size = output_size;

    // Send message
    if (send_msg(rpmsg_fd_, (char*)&req, sizeof(req)) < 0) {
        result.error_message = "Failed to send pre-process message";
        return result;
    }

    // Receive response
    struct c7x_msg_pre_process_resp resp = {};
    int resp_len = sizeof(resp);
    if (recv_msg(rpmsg_fd_, sizeof(resp), (char*)&resp, &resp_len) < 0) {
        result.error_message = "Failed to receive pre-process response";
        return result;
    }

    // Check response validation
    if (resp.hdr.type != C7X_MSG_PRE_PROCESS_RESP) {
        result.error_message = "Invalid response type: got 0x" + std::to_string(resp.hdr.type) +
                              ", expected 0x" + std::to_string(C7X_MSG_PRE_PROCESS_RESP);
        return result;
    }

    if (resp.hdr.status != C7X_STATUS_SUCCESS) {
        result.error_message = "DSP processing failed: status=" + std::to_string(resp.hdr.status) +
                              ", return_value=" + std::to_string(resp.return_value);
        return result;
    }


    // Zero-copy: Data is already processed in TVM staging region, no copying needed

    // Fill result
    result.success = true;
    result.return_value = resp.return_value;
    result.cycles = static_cast<uint32_t>(resp.cycles);
    result.samples_processed = resp.samples_processed;
    result.dsp_load_percent = resp.dsp_load_percent;

    return result;
}

GenericTaskClient::ProcessingResult GenericTaskClient::post_process(DataType data_type,
                                                                  void* input_data,
                                                                  uint32_t input_size,
                                                                  void* output_data,
                                                                  uint32_t output_size)
{
    ProcessingResult result = {};
    result.success = false;

    if (!initialized_) {
        result.error_message = "Client not initialized";
        return result;
    }

    // Zero-copy approach: DSP will read directly from TVM result region
    // TVM output is already in the result buffer

    // Prepare Post-Process message
    struct c7x_msg_post_process req = {};
    req.hdr.type = C7X_MSG_POST_PROCESS;
    req.hdr.seq = sequence_number_++;
    req.hdr.len = sizeof(struct c7x_msg_post_process);
    req.hdr.status = 0;

    req.data_type = static_cast<uint32_t>(data_type);
    req.input_addr = shared_output_addr_;  // TVM result physical address (0xabc00000)
    req.output_addr = shared_output_addr_; // Post-process writes back to TVM result (in-place)
    req.input_size = input_size;
    req.output_size = output_size;

    // Send message
    if (send_msg(rpmsg_fd_, (char*)&req, sizeof(req)) < 0) {
        result.error_message = "Failed to send post-process message";
        return result;
    }

    // Receive response
    struct c7x_msg_post_process_resp resp = {};
    int resp_len = sizeof(resp);
    if (recv_msg(rpmsg_fd_, sizeof(resp), (char*)&resp, &resp_len) < 0) {
        result.error_message = "Failed to receive post-process response";
        return result;
    }

    // Check response
    if (resp.hdr.type != C7X_MSG_POST_PROCESS_RESP || resp.hdr.status != C7X_STATUS_SUCCESS) {
        result.error_message = "DSP post-processing failed: status=" + std::to_string(resp.hdr.status);
        return result;
    }

    // Zero-copy: Data is already processed in TVM result region, no copying needed

    // Fill result
    result.success = true;
    result.return_value = resp.return_value;
    result.cycles = static_cast<uint32_t>(resp.cycles);
    result.samples_processed = resp.samples_processed;
    result.dsp_load_percent = resp.dsp_load_percent;

    return result;
}

GenericTaskClient::ProcessingResult GenericTaskClient::get_service_status()
{
    ProcessingResult result = {};
    result.success = false;
    result.error_message = "Status not implemented with rpmsg-dma-pb";
    return result;
}

bool GenericTaskClient::ping_service(const std::string& message)
{
    if (!initialized_ && rpmsg_fd_ < 0) {
        return false;
    }

    // Simple ping - just test if we can communicate
    // For now, return true if RPMsg is connected
    return (rpmsg_fd_ >= 0);
}

void GenericTaskClient::shutdown()
{
    if (initialized_) {
        close_rpmsg_device();
        free_shared_buffers();
        initialized_ = false;
    }
}

bool GenericTaskClient::open_rpmsg_device()
{

    rpmsg_fd_ = init_rpmsg(C7_PROC_ID, RMT_EP);
    if (rpmsg_fd_ < 0) {
        std::cerr << "Failed to initialize RPMessage" << std::endl;
        return false;
    }

    return true;
}

void GenericTaskClient::close_rpmsg_device()
{
    if (rpmsg_fd_ >= 0) {
        cleanup_rpmsg(rpmsg_fd_);
        rpmsg_fd_ = -1;
    }
}

bool GenericTaskClient::allocate_shared_buffers()
{
    // Zero-copy approach: Use TVM's existing shared memory regions directly
    // No separate DMA allocation needed - just use TVM physical addresses

    // Pre-process uses TVM staging region
    shared_input_addr_ = TVM_STAGING_PHYS;

    // Post-process uses TVM result region
    shared_output_addr_ = TVM_RESULT_PHYS;

    // No virtual memory buffers needed - we only pass physical addresses to DSP
    shared_input_buffer_ = nullptr;  // Not used in zero-copy approach
    shared_output_buffer_ = nullptr; // Not used in zero-copy approach


    return true;
}

void GenericTaskClient::free_shared_buffers()
{
    // Zero-copy approach: No DMA buffers to free, just reset addresses
    shared_input_buffer_ = nullptr;
    shared_output_buffer_ = nullptr;
    shared_input_addr_ = 0;
    shared_output_addr_ = 0;
}

std::string GenericTaskClient::get_error_string(int32_t error_code)
{
    switch (error_code) {
        case 0: return "Success";
        case -1: return "Invalid parameters";
        case -2: return "Unsupported data type";
        case -3: return "Unsupported process type";
        default: return "Unknown error (" + std::to_string(error_code) + ")";
    }
}