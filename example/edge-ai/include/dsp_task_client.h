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
 * @brief Client for communicating with the DSP generic service over RPMsg.
 * DSP endpoint and proc_id are configured at initialize() time via JSON pipeline config.
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
    bool initialize(int proc_id, int endpoint);

    ProcessingResult process(
        const std::string& message_type,
        const std::map<std::string, std::string>& parameters = {});

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
};

#endif // DSP_TASK_CLIENT_H
