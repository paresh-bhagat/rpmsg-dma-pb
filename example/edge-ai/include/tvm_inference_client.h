#ifndef TVM_INFERENCE_CLIENT_H
#define TVM_INFERENCE_CLIENT_H

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

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
    int dma_buffer_fd_{-1};

public:
    TvmInferenceClient();
    ~TvmInferenceClient();

    // Initialization
    bool initialize(const std::string& artifacts_path);
    void cleanup();

    // Inference
    bool run_inference(const std::vector<float>& input_data,
                       std::vector<float>& output_data);
    bool run_inference(const std::vector<float>& input_data,
                       std::vector<float>& output_data,
                       const std::vector<int64_t>& input_shape);
    bool run_inference(std::vector<float>& dint_data, std::vector<float>& inter_data, size_t data_size);
    bool run_inference(const std::string& bin_path);

    // Status
    bool is_initialized() const { return initialized_; }
    const std::vector<float>& get_output() const { return output_data_; }
    const std::vector<int>& get_input_shape() const { return input_shape_; }
    void set_input_shape(const std::vector<int>& shape) { input_shape_ = shape; }
    void set_input_name(const std::string& name) { input_name_ = name; }
    void set_dma_buffer_fd(int descriptor) noexcept { dma_buffer_fd_ = descriptor; }

private:
    // Helper methods
    bool load_artifacts();
    void process_output_data();
    bool synchronize_dma_buffer(int operation) const noexcept;
    std::string load_json_file(const std::string& path);
};

#endif // TVM_INFERENCE_CLIENT_H
