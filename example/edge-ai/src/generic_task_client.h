#ifndef GENERIC_TASK_CLIENT_H
#define GENERIC_TASK_CLIENT_H

#include <stdint.h>
#include <string>

// Include rpmsg-dma-pb library headers
extern "C" {
#include "dmabuf.h"
}

/**
 * @brief Generic Task client for communicating with DSP Generic Service
 *
 * Handles communication with the Generic Service running on DSP endpoint 13.
 * Supports dual data types (float32 and int16) and various processing operations.
 * Uses zero-copy approach with TVM shared memory regions.
 */
class GenericTaskClient {
public:
    enum class DataType {
        FLOAT32 = 0,
        INT16 = 1
    };

    enum class ProcessType {
        PASSTHROUGH = 0,    // Simple copy operation
        FORMAT_CONVERT = 1  // Format conversion/normalization
    };

    struct ProcessingResult {
        bool success;
        int32_t return_value;
        uint32_t cycles;
        uint32_t samples_processed;
        uint32_t dsp_load_percent;
        std::string error_message;
    };

    GenericTaskClient();
    ~GenericTaskClient();

    /**
     * @brief Initialize the Generic Task client
     * @param max_input_size Maximum input buffer size in bytes
     * @param max_output_size Maximum output buffer size in bytes
     * @return true on success, false on failure
     */
    bool initialize(uint32_t max_input_size = 1024*1024, uint32_t max_output_size = 1024*1024);

    /**
     * @brief Pre-processing function (framework testing)
     * @param data_type Data type (float32 or int16)
     * @param input_data Input data buffer
     * @param input_size Size of input data in bytes
     * @param output_data Output data buffer (must be pre-allocated)
     * @param output_size Size of output buffer in bytes
     * @return ProcessingResult with status and performance metrics
     */
    ProcessingResult pre_process(DataType data_type,
                               void* input_data,
                               uint32_t input_size,
                               void* output_data,
                               uint32_t output_size);

    /**
     * @brief Post-processing function (framework testing)
     * @param data_type Data type (float32 or int16)
     * @param input_data Input data buffer (TVM output)
     * @param input_size Size of input data in bytes
     * @param output_data Output data buffer (must be pre-allocated)
     * @param output_size Size of output buffer in bytes
     * @return ProcessingResult with status and performance metrics
     */
    ProcessingResult post_process(DataType data_type,
                                void* input_data,
                                uint32_t input_size,
                                void* output_data,
                                uint32_t output_size);

    /**
     * @brief Get status from the Generic Service
     * @return ProcessingResult with service statistics
     */
    ProcessingResult get_service_status();

    /**
     * @brief Ping the Generic Service
     * @param message Test message to send
     * @return true if service responds, false otherwise
     */
    bool ping_service(const std::string& message = "ping");

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
    bool initialized_;
    uint32_t sequence_number_;

    // Buffer management (zero-copy approach)
    uint32_t max_input_buffer_size_;   // Configured maximum input buffer size
    uint32_t max_output_buffer_size_;  // Configured maximum output buffer size
    void* shared_input_buffer_;        // Not used in zero-copy (set to nullptr)
    void* shared_output_buffer_;       // Not used in zero-copy (set to nullptr)
    uint64_t shared_input_addr_;       // TVM staging physical address (0xa3000000)
    uint64_t shared_output_addr_;      // TVM result physical address (0xabc00000)

    // Internal methods
    bool open_rpmsg_device();
    void close_rpmsg_device();
    bool allocate_shared_buffers();
    void free_shared_buffers();
    std::string get_error_string(int32_t error_code);
};

#endif // GENERIC_TASK_CLIENT_H