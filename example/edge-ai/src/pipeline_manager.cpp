#include "pipeline_manager.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <thread>
#include <iomanip>

PipelineManager::PipelineManager()
    : initialized_(false)
{
    // Default configuration - all stages enabled
    config_.pre_process = {true, 1, 602112, 602112};      // MobileNet float32 tensor size
    config_.tvm_inference = {true, 2, 602112, 4000};      // Input tensor → classification output
    config_.post_process = {true, 3, 4000, 4000};         // Classification results processing
}

PipelineManager::~PipelineManager()
{
    // Cleanup handled by smart pointers
}

bool PipelineManager::initialize(std::shared_ptr<TvmInferenceClient> tvm_client)
{
    if (!tvm_client) {
        std::cerr << "[Pipeline] Error: TVM client is null" << std::endl;
        return false;
    }

    tvm_client_ = tvm_client;


    // Initialize Generic Task client
    generic_client_ = std::make_unique<GenericTaskClient>();
    if (!generic_client_->initialize(config_.pre_process.input_buffer_size,
                                    config_.post_process.output_buffer_size)) {
        std::cerr << "[Pipeline] Failed to initialize Generic Task client" << std::endl;
        return false;
    }

    std::cout << "[Pipeline] Pipeline initialized successfully" << std::endl;
    initialized_ = true;
    return true;
}

bool PipelineManager::configure(const PipelineConfig& config)
{
    config_ = config;

    if (!validate_configuration()) {
        std::cerr << "[Pipeline] Invalid pipeline configuration" << std::endl;
        return false;
    }

    // Reinitialize Generic Task client with new buffer sizes if needed
    if (initialized_ && generic_client_) {
        generic_client_.reset();
        generic_client_ = std::make_unique<GenericTaskClient>();
        if (!generic_client_->initialize(config_.pre_process.input_buffer_size,
                                        config_.post_process.output_buffer_size)) {
            std::cerr << "[Pipeline] Failed to reinitialize Generic Task client" << std::endl;
            return false;
        }
    }

    std::cout << "[Pipeline] Pipeline reconfigured successfully" << std::endl;
    return true;
}

PipelineManager::PipelineResult PipelineManager::execute_pipeline(void* input_data, size_t input_size,
                                                                  DataType data_type, int iterations)
{
    PipelineResult result = {};
    result.success = false;

    if (!initialized_) {
        result.error_message = "Pipeline not initialized";
        return result;
    }


    std::cout << "[Pipeline] Executing " << iterations << " iterations with "
              << (data_type == DataType::FLOAT32 ? "float32" : "int16") << " data" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    uint32_t total_cycles = 0;
    int successful_iterations = 0;

    for (int i = 0; i < iterations; i++) {

        // Create input buffer for this iteration
        void* iteration_input = create_input_buffer(data_type, input_size);
        if (!iteration_input) {
            result.error_message = "Failed to create input buffer for iteration " + std::to_string(i);
            break;
        }


        // Copy input data
        std::memcpy(iteration_input, input_data, input_size);

        bool iteration_success = true;

        // Execute Pre-processing stage
        if (config_.pre_process.enabled) {
            std::cout << "[PRE] Processing input data..." << std::endl;
            if (!execute_pre_process_stage(iteration_input, input_size, data_type)) {
                result.error_message = "Pre-processing failed in iteration " + std::to_string(i);
                iteration_success = false;
            } else {
                std::cout << "[PRE] Completed successfully" << std::endl;
            }
        }

        // Execute TVM inference stage
        if (iteration_success && config_.tvm_inference.enabled) {
            std::cout << "[TVM] Running inference..." << std::endl;
            if (!execute_tvm_stage(iteration_input, input_size, data_type)) {
                result.error_message = "TVM inference failed in iteration " + std::to_string(i);
                iteration_success = false;
            } else {
                std::cout << "[TVM] Completed successfully" << std::endl;
            }
        }

        // Execute Post-processing stage
        if (iteration_success && config_.post_process.enabled) {
            std::cout << "[POST] Processing TVM output..." << std::endl;
            if (!execute_post_process_stage(nullptr, config_.tvm_inference.output_buffer_size, data_type)) {
                result.error_message = "Post-processing failed in iteration " + std::to_string(i);
                iteration_success = false;
            } else {
                std::cout << "[POST] Completed successfully" << std::endl;
            }
        }

        if (iteration_success) {
            successful_iterations++;
        } else {
            std::cout << "[Pipeline] Iteration " << (i + 1) << " failed: " << result.error_message << std::endl;
            break;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    // Fill result
    result.success = (successful_iterations == iterations);
    result.iterations_completed = successful_iterations;
    result.total_cycles = total_cycles;
    result.total_duration_ms = duration.count() / 1000.0;

    if (result.success) {
        std::cout << "[App] Pipeline completed successfully" << std::endl;
        std::cout << "[App] Total duration: " << std::fixed << std::setprecision(2)
                  << result.total_duration_ms << " ms" << std::endl;
    } else {
        std::cout << "[App] Pipeline failed: " << result.error_message << std::endl;
    }

    return result;
}

bool PipelineManager::execute_pre_process_stage(void* input_data, size_t input_size, DataType data_type)
{
    if (!generic_client_) {
        std::cerr << "[Pipeline] Generic Task client not initialized" << std::endl;
        return false;
    }

    // Verify Generic Task client is ready (should be initialized in pipeline init)
    if (!generic_client_->is_ready()) {
        std::cerr << "[PRE] Generic Task client not ready - pipeline initialization failed" << std::endl;
        return false;
    }
    // Execute pre-processing
    auto result = generic_client_->pre_process(convert_data_type(data_type),
                                              input_data, input_size,
                                              input_data, input_size);  // In-place processing

    if (result.success) {
        std::cout << "[PRE] Generic Task completed: DSP cycles=" << result.cycles
                  << ", load=" << result.dsp_load_percent << "%" << std::endl;
        return true;
    } else {
        std::cout << "[PRE] Generic Task failed: " << result.error_message << std::endl;
        return false;
    }
}

bool PipelineManager::execute_tvm_stage(void* input_data, size_t input_size, DataType data_type)
{
    if (!tvm_client_) {
        std::cerr << "[Pipeline] TVM client not initialized" << std::endl;
        return false;
    }

    // For simplicity, assume TVM client handles the inference internally
    // In a full implementation, this would prepare tensors, run inference, etc.

    // Prepare input tensor (simplified)

    // Calculate number of elements based on data type
    size_t element_size = (data_type == DataType::FLOAT32) ? sizeof(float) : sizeof(int16_t);
    size_t num_elements = input_size / element_size;

    std::cout << "[TVM] Input data prepared: " << num_elements << " elements" << std::endl;

    // Run actual TVM inference
    if (!tvm_client_->run_inference_benchmark(1)) {
        std::cerr << "[TVM] Inference failed" << std::endl;
        return false;
    }

    return true;
}

bool PipelineManager::execute_post_process_stage(void* input_data, size_t input_size, DataType data_type)
{
    if (!generic_client_) {
        std::cerr << "[Pipeline] Generic Task client not initialized" << std::endl;
        return false;
    }

    // Post-processing: Get TVM output from result buffer and process it
    // For verification, we'll process the TVM output and save it to a file

    // Get TVM output data - this should be the classification results (1000 floats)
    if (!tvm_client_ || !tvm_client_->is_initialized()) {
        std::cerr << "[POST] TVM client not available for output" << std::endl;
        return false;
    }

    const auto& tvm_output = tvm_client_->get_output();
    if (tvm_output.empty()) {
        std::cerr << "[POST] No TVM output data available" << std::endl;
        return false;
    }

    // Process the TVM output through Generic Task (passthrough for now)
    auto result = generic_client_->post_process(convert_data_type(data_type),
                                               (void*)tvm_output.data(), tvm_output.size() * sizeof(float),
                                               (void*)tvm_output.data(), tvm_output.size() * sizeof(float));

    if (!result.success) {
        std::cerr << "[POST] Generic Task post-processing failed: " << result.error_message << std::endl;
        return false;
    }

    // Save post-processed output to file for verification
    std::string output_file = "/tmp/pipeline_output.bin";
    std::ofstream file(output_file, std::ios::binary);
    if (file.is_open()) {
        file.write((const char*)tvm_output.data(), tvm_output.size() * sizeof(float));
        file.close();
        std::cout << "[POST] Output saved to " << output_file << " (" << (tvm_output.size() * sizeof(float)) << " bytes)" << std::endl;
    }

    return true;
}

void* PipelineManager::create_input_buffer(DataType data_type, size_t input_size)
{

    // Ensure input buffer is large enough
    if (input_buffer_.size() < input_size) {
        input_buffer_.resize(input_size);
    }


    return input_buffer_.data();
}

bool PipelineManager::validate_configuration()
{
    // At least one stage must be enabled
    if (!config_.pre_process.enabled && !config_.tvm_inference.enabled && !config_.post_process.enabled) {
        return false;
    }

    // Buffer sizes must be reasonable
    if (config_.pre_process.enabled &&
        (config_.pre_process.input_buffer_size == 0 || config_.pre_process.output_buffer_size == 0)) {
        return false;
    }

    return true;
}

GenericTaskClient::DataType PipelineManager::convert_data_type(DataType type)
{
    switch (type) {
        case DataType::FLOAT32:
            return GenericTaskClient::DataType::FLOAT32;
        case DataType::INT16:
            return GenericTaskClient::DataType::INT16;
        default:
            return GenericTaskClient::DataType::FLOAT32;
    }
}

std::vector<float> PipelineManager::extract_classification_results(void* output_data, size_t output_size, DataType data_type)
{
    std::vector<float> results;

    if (data_type == DataType::FLOAT32) {
        float* float_data = static_cast<float*>(output_data);
        size_t num_classes = output_size / sizeof(float);

        for (size_t i = 0; i < std::min(num_classes, (size_t)10); i++) {  // Top 10 results
            results.push_back(float_data[i]);
        }
    }

    return results;
}