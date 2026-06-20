#include "tvm_inference_client.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <algorithm>
#include <memory>

// TVM runtime includes
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/container/map.h>

using namespace tvm::runtime;

TvmInferenceClient::TvmInferenceClient() : initialized_(false) {
}

TvmInferenceClient::~TvmInferenceClient() {
    cleanup();
}

bool TvmInferenceClient::initialize(const std::string& artifacts_path) {
    artifacts_path_ = artifacts_path;

    printf("[TVM] Initializing with artifacts from: %s\n", artifacts_path.c_str());

    try {
        if (!load_artifacts()) {
            printf("[TVM]  Failed to load artifacts\n");
            return false;
        }

        initialized_ = true;
        return true;

    } catch (const std::exception& e) {
        printf("[TVM]  Exception during initialization: %s\n", e.what());
        return false;
    }
}

void TvmInferenceClient::cleanup() {
    if (initialized_) {
        graph_executor_.reset();
        set_input_.reset();
        run_.reset();
        get_output_.reset();
        initialized_ = false;
    }
}

std::string TvmInferenceClient::load_json_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    std::string content;
    std::string line;
    while (std::getline(file, line)) {
        content += line + "\n";
    }

    return content;
}

bool TvmInferenceClient::load_artifacts() {
    // 1. Load compiled library (ARM shared library with TIDL integration)
    std::string lib_path = artifacts_path_ + "/deploy_lib.so";

    Module lib = Module::LoadFromFile(lib_path);
    // 2. Load graph JSON
    std::string graph_path = artifacts_path_ + "/deploy_graph.json";
    std::string graph_json = load_json_file(graph_path);

    // 3. Create graph executor
    tvm::Device cpu_dev{kDLCPU, 0};

    auto graph_executor_create = Registry::Get("tvm.graph_executor.create");
    if (!graph_executor_create) {
        printf("[TVM]  Failed to find graph_executor.create function\n");
        return false;
    }


    Module executor;

    // The API expects device as integers, not DLDevice struct
    try {
        executor = (*graph_executor_create)(String(graph_json), lib, int(kDLCPU), int(0));
    } catch (const std::exception& e) {
        // Fallback: 5-parameter approach (some TVM versions)
        try {
            Map<String, ObjectRef> empty_map;
            executor = (*graph_executor_create)(String(graph_json), lib, int(kDLCPU), int(0), empty_map);
        } catch (const std::exception& e2) {
            printf("[TVM] Failed to create graph executor: %s\n", e2.what());
            return false;
        }
    }
    graph_executor_ = std::make_unique<Module>(executor);

    // Get runtime functions
    set_input_ = std::make_unique<PackedFunc>(graph_executor_->GetFunction("set_input"));
    run_ = std::make_unique<PackedFunc>(graph_executor_->GetFunction("run"));
    get_output_ = std::make_unique<PackedFunc>(graph_executor_->GetFunction("get_output"));


    // 4. Load parameters
    std::string param_path = artifacts_path_ + "/deploy_param.params";
    printf("[TVM] Loading parameters: %s\n", param_path.c_str());

    std::ifstream param_file(param_path, std::ios::binary);
    if (!param_file.is_open()) {
        printf("[TVM]  Failed to open param file: %s\n", param_path.c_str());
        return false;
    }

    param_file.seekg(0, std::ios::end);
    size_t param_size = param_file.tellg();
    param_file.seekg(0, std::ios::beg);

    std::vector<uint8_t> param_data(param_size);
    param_file.read(reinterpret_cast<char*>(param_data.data()), param_size);
    param_file.close();

    printf("[TVM] Parameters loaded (%zu bytes)\n", param_size);

    // Load parameters into executor
    auto load_params = graph_executor_->GetFunction("load_params");
    TVMByteArray param_array;
    param_array.data = reinterpret_cast<const char*>(param_data.data());
    param_array.size = param_size;
    load_params(param_array);


    // 5. Set up input configuration (MobileNet v2 default)
    input_name_ = "input.1";
    input_shape_ = {1, 3, 224, 224};  // NCHW format

    printf("[TVM] Input configuration: %s, shape=[%d,%d,%d,%d]\n",
           input_name_.c_str(), input_shape_[0], input_shape_[1],
           input_shape_[2], input_shape_[3]);

    return true;
}

bool TvmInferenceClient::prepare_input_data() {
    // Calculate input size
    int input_size = 1;
    for (int dim : input_shape_) {
        input_size *= dim;
    }

    input_data_.resize(input_size);

    // Generate random input data (like Python script)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < input_size; i++) {
        input_data_[i] = dis(gen);
    }

    printf("[TVM] Input data prepared: %d elements\n", input_size);
    return true;
}

void TvmInferenceClient::process_output_data() {
    // MobileNet v2 output: (1, 1000) ImageNet classes
    const int output_size = 1000;
    output_data_.resize(output_size);

    // Get output array from TVM
    NDArray output_array = (*get_output_)(0);

    // Copy data from NDArray
    output_array.CopyToBytes(output_data_.data(), output_size * sizeof(float));
}

bool TvmInferenceClient::run_inference_benchmark(int num_iterations) {
    if (!initialized_) {
        printf("[TVM]  Client not initialized\n");
        return false;
    }

    printf("[TVM] Preparing input data...\n");
    if (!prepare_input_data()) {
        return false;
    }

    printf("[TVM] Running inference benchmark (%d iterations)...\n", num_iterations);
    printf("------------------------------\n");

    std::vector<double> times;

    for (int i = 0; i < num_iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        try {
            // Create input tensor
            std::vector<int64_t> shape_vec(input_shape_.begin(), input_shape_.end());
            NDArray input_array = NDArray::Empty(shape_vec, DLDataType{kDLFloat, 32, 1}, {kDLCPU, 0});

            // Copy input data
            input_array.CopyFromBytes(input_data_.data(), input_data_.size() * sizeof(float));

            // Set input
            (*set_input_)(input_name_, input_array);

            // Run inference (this automatically uses TIDL acceleration)
            (*run_)();

            // Process output
            process_output_data();

        } catch (const std::exception& e) {
            printf("[TVM]  Inference %d failed: %s\n", i + 1, e.what());
            return false;
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double time_ms = duration.count() / 1000.0;

        times.push_back(time_ms);

        if (i == 0) {
            printf("First run (includes init): %.2f ms\n", time_ms);
            printf("Output shape: (1, 1000)\n");
        } else if (i < 3) {
            printf("Run %d: %.2f ms\n", i + 1, time_ms);
        }
    }

    if (times.size() > 1) {
        // Calculate statistics (excluding first run like Python script)
        std::vector<double> steady_times(times.begin() + 1, times.end());

        double avg_time = 0;
        double min_time = steady_times[0];
        double max_time = steady_times[0];

        for (double t : steady_times) {
            avg_time += t;
            min_time = std::min(min_time, t);
            max_time = std::max(max_time, t);
        }
        avg_time /= steady_times.size();

        double fps = 1000.0 / avg_time;

        printf("\nPerformance Results:\n");
        printf("  Average: %.2f ms\n", avg_time);
        printf("  Min:     %.2f ms\n", min_time);
        printf("  Max:     %.2f ms\n", max_time);
        printf("  FPS:     %.1f\n", fps);

        printf("\nInference completed successfully!\n");
        printf("   Average inference time: %.2f ms\n", avg_time);
        printf("   Using TVM+TIDL on AM62D C7x DSP\n");
    }

    return true;
}

void TvmInferenceClient::print_top5_results() {
    if (output_data_.size() != 1000) {
        printf("[TVM]  Expected 1000 output classes, got %zu\n", output_data_.size());
        return;
    }

    printf("\nModel Output:\n");
    printf("  Raw output shape: (1, 1000)\n");

    // Find top 5 indices
    std::vector<std::pair<float, int>> scores;
    for (int i = 0; i < 1000; i++) {
        scores.push_back({output_data_[i], i});
    }

    std::sort(scores.begin(), scores.end(), std::greater<std::pair<float, int>>());

    printf("  Top 5 ImageNet predictions:\n");
    for (int i = 0; i < 5; i++) {
        printf("    %d. Class %4d: %.4f (%.2f%%)\n",
               i + 1, scores[i].second, scores[i].first, scores[i].first * 100.0f);
    }
}