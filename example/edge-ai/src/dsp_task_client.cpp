#include "dsp_task_client.h"
#include <unistd.h>
#include <iostream>
#include <limits>
#include <stdexcept>

extern "C" {
#include "rpmsg.h"
#include "dmabuf.h"
}

namespace {


uint32_t parameter_value(const std::map<std::string, std::string>& parameters,
                         const std::string& name, uint32_t default_value,
                         int base = 10)
{
    const auto parameter = parameters.find(name);
    if (parameter == parameters.end())
        return default_value;

    size_t parsed = 0;
    const auto value = std::stoull(parameter->second, &parsed, base);
    if (parsed != parameter->second.size() ||
        value > std::numeric_limits<uint32_t>::max())
        throw std::out_of_range{name + " is not a uint32 value"};
    return static_cast<uint32_t>(value);
}

struct c7x_msg_hdr {
    uint32_t type;
    uint32_t seq;
    uint32_t len;
    int32_t  status;
};

struct stft_process_msg {
    struct c7x_msg_hdr hdr;
    uint32_t input_buffer;
    uint32_t output_buffer;
    uint32_t input_frame;
    uint32_t output_frame;
    uint32_t graph_id;
};

enum c7x_msg_type {
    C7X_MSG_STFT_ANALYZE = 0x1020,
    C7X_MSG_STFT_ANALYZE_RESP = 0x2020,
    C7X_MSG_ISTFT_SYNTHESIZE = 0x1030,
    C7X_MSG_ISTFT_SYNTHESIZE_RESP = 0x2030,
    C7X_DEINTERLEAVE_MSG_ANALYZE = 0x1040,
    C7X_DEINTERLEAVE_MSG_ANALYZE_RESP = 0x2040
};

struct deinterleave_interleave_msg {
    struct c7x_msg_hdr hdr;
    uint32_t input_buffer;
    uint32_t output_buffer;
    uint32_t input_frame;
    uint32_t fft_size;
    uint32_t flag;              // 0: deinterleave, 1: interleave
};

static_assert(sizeof(c7x_msg_hdr) == 16);
static_assert(sizeof(stft_process_msg) == 36);
static_assert(sizeof(deinterleave_interleave_msg) == 36);

enum c7x_status {
    C7X_STATUS_SUCCESS = 0,
    C7X_STATUS_ERROR = -1
};

template <typename Message>
bool exchange_message(int descriptor, Message& request, Message& response)
{
    if (send_msg(descriptor, reinterpret_cast<char*>(&request), sizeof(request)) < 0)
        return false;

    int response_length = sizeof(response);
    if (recv_msg(descriptor, sizeof(response), reinterpret_cast<char*>(&response),
                 &response_length) < 0)
        return false;
    return response_length >= static_cast<int>(sizeof(c7x_msg_hdr));
}

} // namespace

DspTaskClient::DspTaskClient()
    : rpmsg_fd_(-1), proc_id_(0), endpoint_(0), initialized_(false), sequence_number_(1)
{
}

DspTaskClient::~DspTaskClient()
{
    shutdown();
}

bool DspTaskClient::initialize(int proc_id, int endpoint,
                               uint32_t max_input_size, uint32_t max_output_size)
{
    (void)max_input_size;
    (void)max_output_size;
    if (initialized_) {
        return true;
    }

    proc_id_  = proc_id;
    endpoint_ = endpoint;

    if (!open_rpmsg_device()) {
        return false;
    }

    initialized_ = true;
    return true;
}

bool DspTaskClient::open_rpmsg_device()
{
    rpmsg_fd_ = init_rpmsg(proc_id_, endpoint_);
    if (rpmsg_fd_ < 0) {
        return false;
    }
    return true;
}

void DspTaskClient::close_rpmsg_device()
{
    if (rpmsg_fd_ >= 0) {
        ::close(rpmsg_fd_);
        rpmsg_fd_ = -1;
    }
}

DspTaskClient::ProcessingResult DspTaskClient::process(const std::string& message_type,
                                                             void* input_data,
                                                             uint32_t input_size,
                                                             void* output_data,
                                                             uint32_t output_size,
                                                             const std::map<std::string, std::string>& parameters)
{
    (void)input_data;
    (void)input_size;
    (void)output_data;
    (void)output_size;

    ProcessingResult result = {};
    result.success = false;

    if (!initialized_) {
        result.error_message = "Client not initialized";
        return result;
    }

    try {
    // Determine message type and send appropriate struct
    if (message_type == "C7X_MSG_STFT_ANALYZE") {
        struct stft_process_msg req = {};
        req.hdr.type = C7X_MSG_STFT_ANALYZE;
        req.hdr.seq = sequence_number_++;
        req.hdr.len = sizeof(struct stft_process_msg);
        req.hdr.status = 0;

        req.input_buffer = parameter_value(parameters, "input_buffer", 0, 16);
        req.output_buffer = parameter_value(parameters, "output_buffer", 0, 16);
        req.input_frame = parameter_value(parameters, "input_frame", 0);
        req.output_frame = parameter_value(parameters, "output_frame", 0);
        req.graph_id = parameter_value(parameters, "graph_id", 0);
#ifdef DEBUG
        std::cout << "[GenericClient] STFT_ANALYZE - Sending to firmware:" << std::endl;
        std::cout << "[GenericClient]   input_buffer=0x" << std::hex << req.input_buffer << std::endl;
        std::cout << "[GenericClient]   output_buffer=0x" << std::hex << req.output_buffer << std::endl;
        std::cout << "[GenericClient]   input_frame=" << std::dec << req.input_frame << " frames" << std::endl;
        std::cout << "[GenericClient]   output_frame=" << std::dec << req.output_frame << " frames" << std::endl;
        std::cout << "[GenericClient]   graph_id=" << req.graph_id << std::endl;
#endif
        struct stft_process_msg resp = {};
        if (!exchange_message(rpmsg_fd_, req, resp)) {
            result.error_message = "STFT analyze message exchange failed";
            return result;
        }

        if (resp.hdr.type != C7X_MSG_STFT_ANALYZE_RESP || resp.hdr.seq != req.hdr.seq) {
            result.error_message = "Invalid STFT analyze response";
            return result;
        }
#ifdef DEBUG
        std::cout << "[GenericClient] STFT_ANALYZE - Firmware responded:" << std::endl;
        std::cout << "[GenericClient]   status=" << resp.hdr.status << std::endl;
        std::cout << "[GenericClient]   resp.input_frame=" << resp.input_frame << " frames" << std::endl;
        std::cout << "[GenericClient]   resp.output_frame=" << resp.output_frame << " frames" << std::endl;
#endif
        if (resp.hdr.status != C7X_STATUS_SUCCESS) {
            result.error_message = "DSP STFT analyze failed";
            return result;
        }

        result.success = true;
        result.input_size = resp.input_frame;
        result.output_size = resp.output_frame;

    } else if (message_type == "C7X_MSG_ISTFT_SYNTHESIZE") {
        struct stft_process_msg req = {};
        req.hdr.type = C7X_MSG_ISTFT_SYNTHESIZE;
        req.hdr.seq = sequence_number_++;
        req.hdr.len = sizeof(struct stft_process_msg);
        req.hdr.status = 0;

        req.input_buffer = parameter_value(parameters, "input_buffer", 0, 16);
        req.output_buffer = parameter_value(parameters, "output_buffer", 0, 16);
        req.input_frame = parameter_value(parameters, "input_frame", 0);
        req.output_frame = parameter_value(parameters, "output_frame", 0);
        req.graph_id = parameter_value(parameters, "graph_id", 0);
#ifdef DEBUG
        std::cout << "[GenericClient] ISTFT_SYNTHESIZE - Sending to firmware:" << std::endl;
        std::cout << "[GenericClient]   input_buffer=0x" << std::hex << req.input_buffer << std::endl;
        std::cout << "[GenericClient]   output_buffer=0x" << std::hex << req.output_buffer << std::endl;
        std::cout << "[GenericClient]   input_frame=" << std::dec << req.input_frame << " frames" << std::endl;
        std::cout << "[GenericClient]   output_frame=" << std::dec << req.output_frame << " frames" << std::endl;
        std::cout << "[GenericClient]   graph_id=" << req.graph_id << std::endl;
#endif
        struct stft_process_msg resp = {};
        if (!exchange_message(rpmsg_fd_, req, resp)) {
            result.error_message = "ISTFT synthesize message exchange failed";
            return result;
        }

        if (resp.hdr.type != C7X_MSG_ISTFT_SYNTHESIZE_RESP || resp.hdr.seq != req.hdr.seq) {
            result.error_message = "Invalid ISTFT synthesize response";
            return result;
        }
#ifdef DEBUG
        std::cout << "[GenericClient] ISTFT_SYNTHESIZE - Firmware responded:" << std::endl;
        std::cout << "[GenericClient]   status=" << resp.hdr.status << std::endl;
        std::cout << "[GenericClient]   resp.input_frame=" << resp.input_frame << " frames" << std::endl;
        std::cout << "[GenericClient]   resp.output_frame=" << resp.output_frame << " frames" << std::endl;
#endif
        if (resp.hdr.status != C7X_STATUS_SUCCESS) {
            result.error_message = "DSP ISTFT synthesize failed";
            return result;
        }

        result.success = true;
        result.input_size = resp.input_frame;
        result.output_size = resp.output_frame;

    } else if (message_type == "C7X_DEINTERLEAVE_MSG_ANALYZE") {
        struct deinterleave_interleave_msg req = {};
        req.hdr.type   = C7X_DEINTERLEAVE_MSG_ANALYZE;
        req.hdr.seq    = sequence_number_++;
        req.hdr.len    = sizeof(struct deinterleave_interleave_msg);
        req.hdr.status = 0;

        req.input_buffer = parameter_value(parameters, "input_buffer", 0, 16);
        req.output_buffer = parameter_value(parameters, "output_buffer", 0, 16);
        req.input_frame = parameter_value(parameters, "input_frame", 0);
        req.fft_size = parameter_value(parameters, "fft_size", 0);
        req.flag = parameter_value(parameters, "flag", 0);

        struct deinterleave_interleave_msg resp = {};
        if (!exchange_message(rpmsg_fd_, req, resp)) {
            result.error_message = "Layout conversion message exchange failed";
            return result;
        }

        if (resp.hdr.type != C7X_DEINTERLEAVE_MSG_ANALYZE_RESP || resp.hdr.seq != req.hdr.seq) {
            result.error_message = "Invalid layout conversion response";
            return result;
        }

        if (resp.hdr.status != C7X_STATUS_SUCCESS) {
            result.error_message = "DSP deinterleave/interleave failed";
            return result;
        }

        result.success = true;
        result.input_size  = resp.input_frame;
        result.output_size = resp.input_frame;

    } else {
        result.error_message = "Unknown message type: " + message_type;
        return result;
    }
    } catch (const std::exception& error) {
        result.error_message = "Invalid DSP stage parameter: " + std::string{error.what()};
        return result;
    }

    return result;
}

DspTaskClient::ProcessingResult DspTaskClient::process(
    const std::string& message_type,
    const std::map<std::string, std::string>& parameters)
{
    return process(message_type, nullptr, 0, nullptr, 0, parameters);
}

DspTaskClient::ProcessingResult DspTaskClient::get_service_status()
{
    ProcessingResult result = {};
    result.success = initialized_;
    if (!initialized_) {
        result.error_message = "Client not initialized";
    }
    return result;
}

bool DspTaskClient::ping_service()
{
    // STFT service doesn't have separate ping - just return initialized status
    return initialized_;
}

void DspTaskClient::shutdown()
{
    if (initialized_) {
        close_rpmsg_device();
        initialized_ = false;
    }
}

std::string DspTaskClient::get_error_string(int32_t error_code)
{
    switch (error_code) {
        case C7X_STATUS_SUCCESS: return "Success";
        case C7X_STATUS_ERROR: return "Error";
        default: return "Unknown error";
    }
}
