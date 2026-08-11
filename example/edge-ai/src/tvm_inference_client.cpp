#include "tvm_inference_client.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <memory>
#include <iomanip>


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

    return true;
}

void TvmInferenceClient::process_output_data() {
    NDArray output_array = (*get_output_)(0);

    // Compute output size from tensor shape
    size_t output_size = 1;
    for (int i = 0; i < output_array->ndim; i++)
        output_size *= output_array->shape[i];

    output_data_.resize(output_size);
    output_array.CopyToBytes(output_data_.data(), output_size * sizeof(float));
}

bool TvmInferenceClient::run_inference(std::vector<float>& dint_data, std::vector<float>& inter_data, size_t data_size) {
    if (!initialized_) {
        std::cerr << "[TVM] Not initialized" << std::endl;
        return false;
    }
    try {
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<int64_t> shape(input_shape_.begin(), input_shape_.end());

        NDArray input_array = NDArray::Empty(shape, DLDataType{kDLFloat, 32, 1}, {kDLCPU, 0});
        input_array.CopyFromBytes(dint_data.data(), data_size);
        (*set_input_)(0, input_array);
        (*run_)();

        NDArray out = (*get_output_)(0);
        size_t out_bytes = 1;
        for (int i = 0; i < out->ndim; i++) out_bytes *= out->shape[i];
        inter_data.resize(out_bytes);
        out.CopyToBytes(inter_data.data(), data_size);

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

        std::cout << "[TVM] Inference done in " << ms << " ms, output: " << out_bytes << " floats" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[TVM] Inference failed: " << e.what() << std::endl;
        return false;
    }
}

bool TvmInferenceClient::run_inference_from_bin(const std::string& bin_path) {
    if (!initialized_) {
        std::cerr << "[TVM] Not initialized" << std::endl;
        return false;
    }

    std::ifstream f(bin_path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "[TVM] Cannot open: " << bin_path << std::endl;
        return false;
    }
    f.seekg(0, std::ios::end);
    size_t file_bytes = f.tellg();
    f.seekg(0, std::ios::beg);

    size_t num_floats = file_bytes / sizeof(float);
    input_data_.resize(num_floats);
    f.read(reinterpret_cast<char*>(input_data_.data()), file_bytes);
    f.close();

    std::cout << "[TVM] Loaded " << num_floats << " floats from " << bin_path << std::endl;

    try {
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<int64_t> shape = {static_cast<int64_t>(num_floats)};
        NDArray input_array = NDArray::Empty(shape, DLDataType{kDLFloat, 32, 1}, {kDLCPU, 0});
        input_array.CopyFromBytes(input_data_.data(), input_data_.size() * sizeof(float));
        (*set_input_)(0, input_array);
        (*run_)();

        NDArray out = (*get_output_)(0);
        size_t out_bytes = 1;
        for (int i = 0; i < out->ndim; i++) out_bytes *= out->shape[i];
        output_data_.resize(out_bytes);
        out.CopyToBytes(output_data_.data(), out_bytes * sizeof(float));

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

        std::cout << "[TVM] Inference done in " << ms << " ms, output: " << out_bytes << " floats" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[TVM] Inference failed: " << e.what() << std::endl;
        return false;
    }
}

