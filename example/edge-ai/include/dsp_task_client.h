#ifndef DSP_TASK_CLIENT_H
#define DSP_TASK_CLIENT_H

#include <stdint.h>
#include <string>
#include <map>

// Include rpmsg-dma-pb library headers
extern "C" {
#include "dmabuf.h"
}

/**
 * @brief Generic Task client for communicating with DSP Generic Service
 *
 * Handles communication with the Generic Service running on DSP endpoint 13.
 * Message type determines which struct and processing logic to use.
 * Uses zero-copy approach with TVM shared memory regions.
 */
class DspTaskClient {
public:
    struct ProcessingResult {
        bool success;
        uint32_t input_size;
        uint32_t output_size;
        std::string error_message;
    };

    DspTaskClient();
    ~DspTaskClient();

    /**
     * @brief Initialize the Generic Task client
     * @param max_input_size Maximum input buffer size in bytes
     * @param max_output_size Maximum output buffer size in bytes
     * @return true on success, false on failure
     */
    bool initialize(int proc_id, int endpoint,
                    uint32_t max_input_size = 1024*1024,
                    uint32_t max_output_size = 1024*1024);

    /**
     * @brief Generic processing function
     * @param message_type Message type string (determines struct and processing)
     * @param input_data Input data buffer (not used in zero-copy mode)
     * @param input_size Size of input data in bytes
     * @param output_data Output data buffer (not used in zero-copy mode)
     * @param output_size Size of output buffer in bytes
     * @return ProcessingResult with status and information
     */
    ProcessingResult process(const std::string& message_type,
                           void* input_data,
                           uint32_t input_size,
                           void* output_data,
                           uint32_t output_size,
                           const std::map<std::string, std::string>& parameters = {});

    ProcessingResult process(
        const std::string& message_type,
        const std::map<std::string, std::string>& parameters = {});

    /**
     * @brief Get status from the Generic Service
     * @return ProcessingResult with service statistics
     */
    ProcessingResult get_service_status();

    /**
     * @brief Ping the Generic Service
     * @return true if service responds, false otherwise
     */
    bool ping_service();

    /**
     * @brief Shutdown and cleanup
     */
    void shutdown();

    /**
     * @brief Check if client is initialized and ready
     * @return true if ready, false otherwise
     */
    bool is_ready() const { return initialized_; }

private:
    // RPMsg communication
    int rpmsg_fd_;
    int proc_id_;
    int endpoint_;
    bool initialized_;
    uint32_t sequence_number_;

    // Internal methods
    bool open_rpmsg_device();
    void close_rpmsg_device();
    bool allocate_shared_buffers();
    void free_shared_buffers();
    std::string get_error_string(int32_t error_code);
};

#endif // DSP_TASK_CLIENT_H
