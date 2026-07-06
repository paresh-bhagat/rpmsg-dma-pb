#include "tvm_inference_client.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <algorithm>
#include <memory>
#include <iomanip>
#include <cnpy.h>

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
    if (initialized_) {
        cleanup();
    }

    artifacts_path_ = artifacts_path;

    try {
        if (!load_artifacts()) {
            return false;
        }

        initialized_ = true;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "TVM initialization failed: " << e.what() << std::endl;
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
        std::cerr << "Failed to find graph_executor.create function" << std::endl;
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
            std::cerr << "Failed to create graph executor: " << e2.what() << std::endl;
            return false;
        }
    }
    graph_executor_ = std::make_unique<Module>(executor);

    // Get runtime functions
    set_input_ = std::make_unique<PackedFunc>(graph_executor_->GetFunction("set_input"));
    run_ = std::make_unique<PackedFunc>(graph_executor_->GetFunction("run"));
    get_output_ = std::make_unique<PackedFunc>(graph_executor_->GetFunction("get_output"));


    // Load model parameters
    std::string param_path = artifacts_path_ + "/deploy_param.params";

    std::ifstream param_file(param_path, std::ios::binary);
    if (!param_file.is_open()) {
        return false;
    }

    param_file.seekg(0, std::ios::end);
    size_t param_size = param_file.tellg();
    param_file.seekg(0, std::ios::beg);

    std::vector<uint8_t> param_data(param_size);
    param_file.read(reinterpret_cast<char*>(param_data.data()), param_size);
    param_file.close();

    // Load parameters into executor
    auto load_params = graph_executor_->GetFunction("load_params");
    TVMByteArray param_array;
    param_array.data = reinterpret_cast<const char*>(param_data.data());
    param_array.size = param_size;
    load_params(param_array);


    // 5. Set up input configuration (MobileNet v2 default)
    input_name_ = "input.1";
    input_shape_ = {1, 3, 224, 224};

    return true;
}

bool TvmInferenceClient::prepare_input_data() {
    int input_size = 1;
    for (int dim : input_shape_) {
        input_size *= dim;
    }

    input_data_.resize(input_size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < input_size; i++) {
        input_data_[i] = dis(gen);
    }

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
        return false;
    }

    if (!prepare_input_data()) {
        return false;
    }

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
            std::cerr << "Inference " << (i + 1) << " failed: " << e.what() << std::endl;
            return false;
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double time_ms = duration.count() / 1000.0;

        times.push_back(time_ms);

        if (i == 0) {
            std::cout << "First run (includes init): " << std::fixed << std::setprecision(2) << time_ms << " ms" << std::endl;
            std::cout << "Output shape: (1, 1000)" << std::endl;
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
    }

    return true;
}

bool TvmInferenceClient::run_inference_with_data(const void* input_data, size_t input_size) {
    if (!initialized_) {
        std::cerr << "TVM client not initialized" << std::endl;
        return false;
    }

    // Verify input size matches expected input
    size_t expected_size = 1; // Calculate expected size
    for (int dim : input_shape_) {
        expected_size *= dim;
    }
    expected_size *= sizeof(float); // Assuming float32 input

    if (input_size != expected_size) {
        std::cerr << "Input size mismatch: expected " << expected_size
                  << " bytes, got " << input_size << " bytes" << std::endl;
        return false;
    }

    try {
        auto start = std::chrono::high_resolution_clock::now();

        // Create input tensor
        std::vector<int64_t> shape_vec(input_shape_.begin(), input_shape_.end());
        NDArray input_array = NDArray::Empty(shape_vec, DLDataType{kDLFloat, 32, 1}, {kDLCPU, 0});

        // Copy pipeline input data (instead of test data)
        input_array.CopyFromBytes(input_data, input_size);

        // Set input
        (*set_input_)(input_name_, input_array);

        // Run inference
        (*run_)();

        // Get output
        NDArray output_array = (*get_output_)(0);

        // Process output data
        process_output_data();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double time_ms = duration.count() / 1000.0;

        std::cout << "Pipeline inference: " << std::fixed << std::setprecision(2) << time_ms << " ms" << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Pipeline inference failed: " << e.what() << std::endl;
        return false;
    }
}

bool TvmInferenceClient::run_inference_with_npz(const std::string& npz_path) {
    if (!initialized_) {
        std::cerr << "TVM client not initialized" << std::endl;
        return false;
    }

    try {
        std::cout << "[TVM] Loading NPZ file: " << npz_path << std::endl;

        // Load NPZ file
        cnpy::npz_t npz_data = cnpy::npz_load(npz_path);

        if (npz_data.empty()) {
            std::cerr << "[TVM] Error: NPZ file is empty or invalid" << std::endl;
            return false;
        }

        // Get first array from NPZ (or use hardcoded key like "input_0")
        cnpy::NpyArray npz_array;
        if (npz_data.find("input_0") != npz_data.end()) {
            npz_array = npz_data["input_0"];
        } else {
            npz_array = npz_data.begin()->second;
        }

        std::cout << "[TVM] NPZ array shape: [";
        for (size_t i = 0; i < npz_array.shape.size(); i++) {
            std::cout << npz_array.shape[i];
            if (i < npz_array.shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // Update input shape from NPZ
        input_shape_.clear();
        for (size_t dim : npz_array.shape) {
            input_shape_.push_back(static_cast<int>(dim));
        }

        auto start = std::chrono::high_resolution_clock::now();

        // Create input tensor
        std::vector<int64_t> shape_vec(input_shape_.begin(), input_shape_.end());
        NDArray input_array = NDArray::Empty(shape_vec, DLDataType{kDLFloat, 32, 1}, {kDLCPU, 0});

        // Copy NPZ data
        input_array.CopyFromBytes(npz_array.data<float>(), npz_array.num_vals * sizeof(float));

        // Set input
        (*set_input_)(0, input_array);  // Use index 0 instead of name

        // Run inference
        (*run_)();

        // Get output
        process_output_data();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double time_ms = duration.count() / 1000.0;

        std::cout << "[TVM] Inference completed: " << std::fixed << std::setprecision(2) << time_ms << " ms" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[TVM] NPZ inference failed: " << e.what() << std::endl;
        return false;
    }
}

void TvmInferenceClient::print_top5_results() {
    if (output_data_.size() != 1000) {
        return;
    }

    // Find top 5 indices
    std::vector<std::pair<float, int>> scores;
    for (int i = 0; i < 1000; i++) {
        scores.push_back({output_data_[i], i});
    }

    std::sort(scores.begin(), scores.end(), std::greater<std::pair<float, int>>());
}