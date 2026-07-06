#ifndef PIPELINE_MANAGER_H
#define PIPELINE_MANAGER_H

#include <string>
#include <vector>
#include <memory>
#include "tvm_inference_client.h"
#include "generic_task_client.h"

/**
 * @brief Pipeline Manager for orchestrating Generic→TVM→Post-processing flow
 *
 * Manages the complete EdgeAI pipeline with configurable stages:
 * 1. Generic Task Pre-processing (DSP endpoint 13)
 * 2. TVM Inference (DSP endpoint 20)
 * 3. Generic Task Post-processing (DSP endpoint 13)
 */
class PipelineManager {
public:
    enum class DataType {
        FLOAT32 = 0,
        INT16 = 1
    };

    struct StageConfig {
        bool enabled;
        uint32_t priority;
        uint32_t input_buffer_size;
        uint32_t output_buffer_size;
    };

    struct PipelineConfig {
        StageConfig pre_process;
        StageConfig tvm_inference;
        StageConfig post_process;
    };

    struct PipelineResult {
        bool success;
        std::string error_message;
        uint32_t total_cycles;
        uint32_t iterations_completed;
        double total_duration_ms;
        std::vector<float> classification_results;
    };

    PipelineManager();
    ~PipelineManager();

    /**
     * @brief Initialize pipeline with TVM client
     * @param tvm_client Shared pointer to initialized TVM client
     * @return true on success, false on failure
     */
    bool initialize(std::shared_ptr<TvmInferenceClient> tvm_client);

    /**
     * @brief Configure pipeline stages with buffer sizes
     * @param config Pipeline configuration with enabled stages and buffer sizes
     * @return true on success, false on failure
     */
    bool configure(const PipelineConfig& config);

    /**
     * @brief Execute complete pipeline with input data
     * @param input_data Input tensor data
     * @param input_size Size of input data in bytes
     * @param data_type Data type (FLOAT32 or INT16)
     * @param iterations Number of iterations to run
     * @return PipelineResult with success status and performance metrics
     */
    PipelineResult execute_pipeline(void* input_data, size_t input_size,
                                   DataType data_type, int iterations = 1);

    /**
     * @brief Get current pipeline configuration
     */
    const PipelineConfig& get_config() const { return config_; }

    /**
     * @brief Check if pipeline is initialized and ready
     */
    bool is_ready() const { return initialized_; }

private:
    // Core components
    std::shared_ptr<TvmInferenceClient> tvm_client_;
    std::unique_ptr<GenericTaskClient> generic_client_;

    // Configuration and state
    PipelineConfig config_;
    bool initialized_;

    // Buffer management
    std::vector<uint8_t> input_buffer_;
    std::vector<uint8_t> output_buffer_;

    // Internal pipeline execution methods
    bool execute_pre_process_stage(void* input_data, size_t input_size, DataType data_type);
    bool execute_tvm_stage(void* input_data, size_t input_size, DataType data_type);
    bool execute_post_process_stage(void* input_data, size_t input_size, DataType data_type);

    // Helper methods
    void* create_input_buffer(DataType data_type, size_t input_size);
    bool validate_configuration();
    GenericTaskClient::DataType convert_data_type(DataType type);
    std::vector<float> extract_classification_results(void* output_data, size_t output_size, DataType data_type);
};

#endif // PIPELINE_MANAGER_H