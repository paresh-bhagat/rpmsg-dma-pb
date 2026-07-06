#include "generic_task_client.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <cstring>
#include <iostream>
#include <errno.h>

extern "C" {
#include "rpmsg.h"
#include "dmabuf.h"
}

#define C7_PROC_ID      8
#define RMT_EP          13

#define TVM_STAGING_PHYS        0xa3000000UL
#define TVM_RESULT_PHYS         0xabc00000UL

struct c7x_msg_hdr {
    uint32_t type;
    uint32_t seq;
    uint32_t len;
    int32_t  status;
} __attribute__((packed));

struct stft_process_msg {
    struct c7x_msg_hdr hdr;
    uint32_t input_buffer;
    uint32_t output_buffer;
    uint32_t input_size;
    uint32_t output_size;
    uint32_t graph_id;
} __attribute__((packed));

enum c7x_msg_type {
    C7X_MSG_STFT_ANALYZE = 0x1020,
    C7X_MSG_STFT_ANALYZE_RESP = 0x2020,
    C7X_MSG_ISTFT_SYNTHESIZE = 0x1030,
    C7X_MSG_ISTFT_SYNTHESIZE_RESP = 0x2030
};

enum c7x_status {
    C7X_STATUS_SUCCESS = 0,
    C7X_STATUS_ERROR = -1
};

GenericTaskClient::GenericTaskClient()
    : rpmsg_fd_(-1), initialized_(false), sequence_number_(1)
{
    shared_input_addr_ = TVM_STAGING_PHYS;
    shared_output_addr_ = TVM_RESULT_PHYS;
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

    if (!open_rpmsg_device()) {
        return false;
    }

    initialized_ = true;
    return true;
}

bool GenericTaskClient::open_rpmsg_device()
{
    rpmsg_fd_ = init_rpmsg(C7_PROC_ID, RMT_EP);
    if (rpmsg_fd_ < 0) {
        return false;
    }
    return true;
}

void GenericTaskClient::close_rpmsg_device()
{
    if (rpmsg_fd_ >= 0) {
        close(rpmsg_fd_);
        rpmsg_fd_ = -1;
    }
}

GenericTaskClient::ProcessingResult GenericTaskClient::process(const std::string& message_type,
                                                             void* input_data,
                                                             uint32_t input_size,
                                                             void* output_data,
                                                             uint32_t output_size,
                                                             const std::map<std::string, std::string>& parameters)
{
    ProcessingResult result = {};
    result.success = false;

    if (!initialized_) {
        result.error_message = "Client not initialized";
        return result;
    }

    // Determine message type and send appropriate struct
    if (message_type == "C7X_MSG_STFT_ANALYZE") {
        struct stft_process_msg req = {};
        req.hdr.type = C7X_MSG_STFT_ANALYZE;
        req.hdr.seq = sequence_number_++;
        req.hdr.len = sizeof(struct stft_process_msg);
        req.hdr.status = 0;

        // Read all values from parameters
        auto input_buffer_it = parameters.find("input_buffer");
        auto output_buffer_it = parameters.find("output_buffer");
        auto input_size_it = parameters.find("input_size");
        auto output_size_it = parameters.find("output_size");
        auto graph_id_it = parameters.find("graph_id");

        req.input_buffer = (input_buffer_it != parameters.end()) ?
                          std::stoul(input_buffer_it->second, nullptr, 16) : shared_input_addr_;
        req.output_buffer = (output_buffer_it != parameters.end()) ?
                           std::stoul(output_buffer_it->second, nullptr, 16) : shared_output_addr_;
        req.input_size = (input_size_it != parameters.end()) ?
                        std::stoul(input_size_it->second) : input_size;
        req.output_size = (output_size_it != parameters.end()) ?
                         std::stoul(output_size_it->second) : output_size;
        req.graph_id = (graph_id_it != parameters.end()) ?
                      std::stoul(graph_id_it->second) : 0;

        std::cout << "[GenericClient] STFT_ANALYZE - Sending to firmware:" << std::endl;
        std::cout << "[GenericClient]   input_buffer=0x" << std::hex << req.input_buffer << std::endl;
        std::cout << "[GenericClient]   output_buffer=0x" << std::hex << req.output_buffer << std::endl;
        std::cout << "[GenericClient]   input_size=" << std::dec << req.input_size << " bytes" << std::endl;
        std::cout << "[GenericClient]   output_size=" << std::dec << req.output_size << " bytes" << std::endl;
        std::cout << "[GenericClient]   graph_id=" << req.graph_id << std::endl;

        if (send_msg(rpmsg_fd_, (char*)&req, sizeof(req)) < 0) {
            result.error_message = "Failed to send STFT analyze message";
            return result;
        }

        struct stft_process_msg resp = {};
        int resp_len = sizeof(resp);
        if (recv_msg(rpmsg_fd_, sizeof(resp), (char*)&resp, &resp_len) < 0) {
            result.error_message = "Failed to receive STFT analyze response";
            return result;
        }

        if (resp.hdr.type != C7X_MSG_STFT_ANALYZE_RESP) {
            result.error_message = "Invalid STFT analyze response type";
            return result;
        }

        std::cout << "[GenericClient] STFT_ANALYZE - Firmware responded:" << std::endl;
        std::cout << "[GenericClient]   status=" << resp.hdr.status << std::endl;
        std::cout << "[GenericClient]   resp.input_size=" << resp.input_size << " bytes" << std::endl;
        std::cout << "[GenericClient]   resp.output_size=" << resp.output_size << " bytes" << std::endl;

        if (resp.hdr.status != C7X_STATUS_SUCCESS) {
            result.error_message = "DSP STFT analyze failed";
            return result;
        }

        result.success = true;
        result.input_size = resp.input_size;
        result.output_size = resp.output_size;

    } else if (message_type == "C7X_MSG_ISTFT_SYNTHESIZE") {
        struct stft_process_msg req = {};
        req.hdr.type = C7X_MSG_ISTFT_SYNTHESIZE;
        req.hdr.seq = sequence_number_++;
        req.hdr.len = sizeof(struct stft_process_msg);
        req.hdr.status = 0;

        // Read all values from parameters
        auto input_buffer_it = parameters.find("input_buffer");
        auto output_buffer_it = parameters.find("output_buffer");
        auto input_size_it = parameters.find("input_size");
        auto output_size_it = parameters.find("output_size");
        auto graph_id_it = parameters.find("graph_id");

        req.input_buffer = (input_buffer_it != parameters.end()) ?
                          std::stoul(input_buffer_it->second, nullptr, 16) : shared_input_addr_;
        req.output_buffer = (output_buffer_it != parameters.end()) ?
                           std::stoul(output_buffer_it->second, nullptr, 16) : shared_output_addr_;
        req.input_size = (input_size_it != parameters.end()) ?
                        std::stoul(input_size_it->second) : input_size;
        req.output_size = (output_size_it != parameters.end()) ?
                         std::stoul(output_size_it->second) : output_size;
        req.graph_id = (graph_id_it != parameters.end()) ?
                      std::stoul(graph_id_it->second) : 0;

        std::cout << "[GenericClient] ISTFT_SYNTHESIZE - Sending to firmware:" << std::endl;
        std::cout << "[GenericClient]   input_buffer=0x" << std::hex << req.input_buffer << std::endl;
        std::cout << "[GenericClient]   output_buffer=0x" << std::hex << req.output_buffer << std::endl;
        std::cout << "[GenericClient]   input_size=" << std::dec << req.input_size << " bytes" << std::endl;
        std::cout << "[GenericClient]   output_size=" << std::dec << req.output_size << " bytes" << std::endl;
        std::cout << "[GenericClient]   graph_id=" << req.graph_id << std::endl;

        if (send_msg(rpmsg_fd_, (char*)&req, sizeof(req)) < 0) {
            result.error_message = "Failed to send ISTFT synthesize message";
            return result;
        }

        struct stft_process_msg resp = {};
        int resp_len = sizeof(resp);
        if (recv_msg(rpmsg_fd_, sizeof(resp), (char*)&resp, &resp_len) < 0) {
            result.error_message = "Failed to receive ISTFT synthesize response";
            return result;
        }

        if (resp.hdr.type != C7X_MSG_ISTFT_SYNTHESIZE_RESP) {
            result.error_message = "Invalid ISTFT synthesize response type";
            return result;
        }

        std::cout << "[GenericClient] ISTFT_SYNTHESIZE - Firmware responded:" << std::endl;
        std::cout << "[GenericClient]   status=" << resp.hdr.status << std::endl;
        std::cout << "[GenericClient]   resp.input_size=" << resp.input_size << " bytes" << std::endl;
        std::cout << "[GenericClient]   resp.output_size=" << resp.output_size << " bytes" << std::endl;

        if (resp.hdr.status != C7X_STATUS_SUCCESS) {
            result.error_message = "DSP ISTFT synthesize failed";
            return result;
        }

        result.success = true;
        result.input_size = resp.input_size;
        result.output_size = resp.output_size;

    } else {
        result.error_message = "Unknown message type: " + message_type;
        return result;
    }

    return result;
}

GenericTaskClient::ProcessingResult GenericTaskClient::get_service_status()
{
    ProcessingResult result = {};
    result.success = initialized_;
    if (!initialized_) {
        result.error_message = "Client not initialized";
    }
    return result;
}

bool GenericTaskClient::ping_service()
{
    // STFT service doesn't have separate ping - just return initialized status
    return initialized_;
}

void GenericTaskClient::shutdown()
{
    if (initialized_) {
        close_rpmsg_device();
        initialized_ = false;
    }
}

std::string GenericTaskClient::get_error_string(int32_t error_code)
{
    switch (error_code) {
        case C7X_STATUS_SUCCESS: return "Success";
        case C7X_STATUS_ERROR: return "Error";
        default: return "Unknown error";
    }
}