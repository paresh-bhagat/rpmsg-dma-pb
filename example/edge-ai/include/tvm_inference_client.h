#ifndef TVM_INFERENCE_CLIENT_H
#define TVM_INFERENCE_CLIENT_H

#include <string>
#include <vector>
#include <memory>

// Forward declarations to avoid including TVM headers here
namespace tvm {
namespace runtime {
class Module;
class PackedFunc;
}
}

class TvmInferenceClient {
private:
    std::string artifacts_path_;
    std::unique_ptr<tvm::runtime::Module> graph_executor_;
    std::unique_ptr<tvm::runtime::PackedFunc> set_input_;
    std::unique_ptr<tvm::runtime::PackedFunc> run_;
    std::unique_ptr<tvm::runtime::PackedFunc> get_output_;

    bool initialized_;

    // Input/output data
    std::vector<float> input_data_;
    std::vector<float> output_data_;

    // Model info
    std::string input_name_;
    std::vector<int> input_shape_;

public:
    TvmInferenceClient();
    ~TvmInferenceClient();

    // Initialization
    bool initialize(const std::string& artifacts_path);
    void cleanup();

    // Inference
    bool run_inference_benchmark(int num_iterations = 10);
    bool run_inference_with_data(const void* input_data, size_t input_size);
    bool run_inference_from_bin(const std::string& bin_path);
    void print_top5_results();

    // Status
    bool is_initialized() const { return initialized_; }
    const std::vector<float>& get_output() const { return output_data_; }
    const std::vector<int>& get_input_shape() const { return input_shape_; }

private:
    // Helper methods
    bool load_artifacts();
    bool prepare_input_data();
    void process_output_data();
    std::string load_json_file(const std::string& path);
};

#endif // TVM_INFERENCE_CLIENT_H