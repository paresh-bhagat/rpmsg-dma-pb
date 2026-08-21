#include "tvm_inference_client.h"
#include "tvm_daemon_proto.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

// TVM runtime includes
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/container/map.h>

extern "C" {
#include "dmabuf.h"
}

using namespace tvm::runtime;

namespace {

static bool sock_write_all(int fd, const void* buf, size_t len) {
    const auto* p = static_cast<const uint8_t*>(buf);
    while (len > 0) {
        ssize_t n = ::write(fd, p, len);
        if (n <= 0) return false;
        p += static_cast<size_t>(n); len -= static_cast<size_t>(n);
    }
    return true;
}

static bool sock_read_all(int fd, void* buf, size_t len) {
    auto* p = static_cast<uint8_t*>(buf);
    while (len > 0) {
        ssize_t n = ::read(fd, p, len);
        if (n <= 0) return false;
        p += static_cast<size_t>(n); len -= static_cast<size_t>(n);
    }
    return true;
}

template <typename Tensor>
size_t tensor_element_count(const Tensor* tensor)
{
    if (!tensor || tensor->ndim <= 0)
        throw std::runtime_error{"TVM returned an invalid tensor"};

    size_t elements = 1;
    for (int dimension = 0; dimension < tensor->ndim; ++dimension) {
        if (tensor->shape[dimension] <= 0 ||
            elements > std::numeric_limits<size_t>::max() /
                           static_cast<size_t>(tensor->shape[dimension]))
            throw std::runtime_error{"TVM returned an invalid tensor shape"};
        elements *= static_cast<size_t>(tensor->shape[dimension]);
    }
    return elements;
}

} // namespace

bool TvmInferenceClient::synchronize_dma_buffer(int fd, int operation) const noexcept {
    return fd < 0 || dmabuf_sync(fd, operation) == 0;
}

TvmInferenceClient::TvmInferenceClient() : initialized_(false) {
}

TvmInferenceClient::~TvmInferenceClient() {
    cleanup();
}

bool TvmInferenceClient::initialize(const std::string& artifacts_path) {
    cleanup();
    if (artifacts_path.empty()) {
        std::cerr << "[TVM] Artifacts path is empty" << std::endl;
        return false;
    }

    artifacts_path_ = artifacts_path;

    /* Fast path: delegate to persistent daemon — skips Module::LoadFromFile */
    if (!daemon_skip_ && try_daemon_connect()) {
        initialized_ = true;
        std::cout << "[TVM] Connected to model daemon at "
                  << TvmDaemon::SOCKET_PATH << std::endl;
        return true;
    }

    /* Fall back to loading artifacts locally */
    try {
        if (!load_artifacts())
            return false;

        initialized_ = true;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "TVM initialization failed: " << e.what() << std::endl;
        return false;
    }
}

bool TvmInferenceClient::try_daemon_connect() {
    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return false;

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, TvmDaemon::SOCKET_PATH, sizeof(addr.sun_path) - 1);

    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(fd);
        return false;
    }

    /* Ping/pong: confirm daemon is alive and ready */
    TvmDaemon::Header ping{TvmDaemon::MAGIC,
                           static_cast<uint32_t>(TvmDaemon::MsgType::PING), 0};
    TvmDaemon::Header pong{};
    if (!sock_write_all(fd, &ping, sizeof(ping)) ||
        !sock_read_all(fd, &pong, sizeof(pong)) ||
        pong.magic != TvmDaemon::MAGIC ||
        pong.type  != static_cast<uint32_t>(TvmDaemon::MsgType::PONG)) {
        ::close(fd);
        return false;
    }

    daemon_fd_ = fd;
    return true;
}

bool TvmInferenceClient::run_via_daemon(const float* input, float* output, size_t count) {
    const uint32_t in_bytes = static_cast<uint32_t>(count * sizeof(float));
    TvmDaemon::Header req{TvmDaemon::MAGIC,
                          static_cast<uint32_t>(TvmDaemon::MsgType::INFER_REQ),
                          in_bytes};

    if (!sock_write_all(daemon_fd_, &req, sizeof(req)) ||
        !sock_write_all(daemon_fd_, input, in_bytes)) {
        std::cerr << "[TVM] Daemon: failed to send request\n";
        return false;
    }

    TvmDaemon::Header resp{};
    if (!sock_read_all(daemon_fd_, &resp, sizeof(resp)) ||
        resp.magic != TvmDaemon::MAGIC) {
        std::cerr << "[TVM] Daemon: invalid response header\n";
        return false;
    }

    if (resp.type == static_cast<uint32_t>(TvmDaemon::MsgType::ERROR_RESP)) {
        std::vector<char> msg(resp.len + 1, '\0');
        sock_read_all(daemon_fd_, msg.data(), resp.len);
        std::cerr << "[TVM] Daemon error: " << msg.data() << '\n';
        return false;
    }

    if (resp.type != static_cast<uint32_t>(TvmDaemon::MsgType::INFER_RESP)) {
        std::cerr << "[TVM] Daemon: unexpected response type " << resp.type << '\n';
        return false;
    }

    const size_t out_floats = resp.len / sizeof(float);
    if (out_floats > count) {
        std::cerr << "[TVM] Daemon: output too large (" << out_floats
                  << " > " << count << ")\n";
        return false;
    }

    if (!sock_read_all(daemon_fd_, output, resp.len)) {
        std::cerr << "[TVM] Daemon: failed to read output\n";
        return false;
    }

    return true;
}

void TvmInferenceClient::cleanup() {
    if (daemon_fd_ >= 0) { ::close(daemon_fd_); daemon_fd_ = -1; }
    get_output_.reset();
    run_.reset();
    set_input_.reset();
    graph_executor_.reset();
    input_data_.clear();
    output_data_.clear();
    initialized_ = false;
}

std::string TvmInferenceClient::load_json_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    return {std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
}

bool TvmInferenceClient::load_artifacts() {
    // 1. Load compiled library (ARM shared library with TIDL integration)
    const std::filesystem::path artifacts{artifacts_path_};
    const auto lib_path = artifacts / "deploy_lib.so";

    Module lib = Module::LoadFromFile(lib_path.string());
    // 2. Load graph JSON
    const auto graph_path = artifacts / "deploy_graph.json";
    std::string graph_json = load_json_file(graph_path.string());

    // 3. Create graph executor
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
    auto set_input = graph_executor_->GetFunction("set_input");
    auto run = graph_executor_->GetFunction("run");
    auto get_output = graph_executor_->GetFunction("get_output");
    if (set_input == nullptr || run == nullptr || get_output == nullptr) {
        std::cerr << "[TVM] Graph executor is missing a required function" << std::endl;
        return false;
    }
    set_input_ = std::make_unique<PackedFunc>(std::move(set_input));
    run_ = std::make_unique<PackedFunc>(std::move(run));
    get_output_ = std::make_unique<PackedFunc>(std::move(get_output));


    // Load model parameters
    const auto param_path = artifacts / "deploy_param.params";

    std::ifstream param_file(param_path, std::ios::binary | std::ios::ate);
    if (!param_file.is_open()) {
        return false;
    }

    const auto end_position = param_file.tellg();
    if (end_position <= 0)
        return false;
    const auto param_size = static_cast<size_t>(end_position);
    param_file.seekg(0, std::ios::beg);

    std::vector<uint8_t> param_data(param_size);
    if (!param_file.read(reinterpret_cast<char*>(param_data.data()),
                         static_cast<std::streamsize>(param_size)))
        return false;

    // Load parameters into executor
    auto load_params = graph_executor_->GetFunction("load_params");
    if (load_params == nullptr)
        return false;
    TVMByteArray param_array;
    param_array.data = reinterpret_cast<const char*>(param_data.data());
    param_array.size = param_size;
    load_params(param_array);

    return true;
}

void TvmInferenceClient::process_output_data() {
    NDArray output_array = (*get_output_)(0);
    const size_t output_size = tensor_element_count(output_array.operator->());
    output_data_.resize(output_size);
    output_array.CopyToBytes(output_data_.data(), output_size * sizeof(float));
}

bool TvmInferenceClient::run_inference(const std::vector<float>& input_data,
                                       std::vector<float>& output_data) {
    std::vector<int64_t> shape;
    if (input_shape_.empty())
        shape = {static_cast<int64_t>(input_data.size())};
    else
        shape.assign(input_shape_.begin(), input_shape_.end());
    return run_inference(input_data, output_data, shape);
}

bool TvmInferenceClient::run_inference(const std::vector<float>& input_data,
                                       std::vector<float>& output_data,
                                       const std::vector<int64_t>& input_shape) {
    if (!initialized_) {
        std::cerr << "[TVM] Not initialized" << std::endl;
        return false;
    }
    if (input_data.empty() || input_shape.empty()) {
        std::cerr << "[TVM] Input data size or shape is invalid" << std::endl;
        return false;
    }

    if (daemon_fd_ >= 0) {
        output_data.resize(input_data.size());
        return run_via_daemon(input_data.data(), output_data.data(), input_data.size());
    }

    try {
        const size_t shape_elements = std::accumulate(
            input_shape.begin(), input_shape.end(), size_t{1},
            [](size_t total, int64_t extent) {
                if (extent <= 0 || total > std::numeric_limits<size_t>::max() /
                                             static_cast<size_t>(extent))
                    throw std::overflow_error{"Invalid TVM input shape"};
                return total * static_cast<size_t>(extent);
            });
        if (shape_elements != input_data.size())
            throw std::runtime_error{"TVM input shape does not match the input data"};

        const auto start = std::chrono::steady_clock::now();

        if (!synchronize_dma_buffer(dma_input_fd_, DMA_BUF_SYNC_START))
            throw std::runtime_error{"Failed to sync input DMA buffer for CPU access"};

        NDArray input_array = NDArray::Empty(input_shape, DLDataType{kDLFloat, 32, 1}, {kDLCPU, 0});
        input_array.CopyFromBytes(input_data.data(), input_data.size() * sizeof(float));
        if (input_name_.empty())
            (*set_input_)(0, input_array);
        else
            (*set_input_)(String{input_name_}, input_array);
        (*run_)();

        NDArray out = (*get_output_)(0);
        const size_t output_elements = tensor_element_count(out.operator->());
        output_data.resize(output_elements);
        out.CopyToBytes(output_data.data(), output_elements * sizeof(float));

        if (!synchronize_dma_buffer(dma_output_fd_, DMA_BUF_SYNC_END))
            throw std::runtime_error{"Failed to complete DMA buffer cache synchronization"};

        const auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

        std::cout << "[TVM] Inference done in " << ms << " ms, output: "
                  << output_elements << " floats" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[TVM] Inference failed: " << e.what() << std::endl;
        return false;
    }
}

bool TvmInferenceClient::run_inference(const float* input, float* output, size_t count) {
    if (!initialized_) {
        std::cerr << "[TVM] Not initialized" << std::endl;
        return false;
    }
    if (!input || count == 0 || !output) {
        std::cerr << "[TVM] Invalid input/output pointer or count" << std::endl;
        return false;
    }

    /* Daemon mode — send over socket instead of running TVM locally */
    if (daemon_fd_ >= 0) {
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = run_via_daemon(input, output, count);
        const double ms = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count() / 1000.0;
        if (ok) std::cout << "[TVM] Daemon inference done in " << ms
                          << " ms, " << count << " floats\n";
        return ok;
    }

    std::vector<int64_t> shape;
    if (input_shape_.empty())
        shape = {static_cast<int64_t>(count)};
    else
        shape.assign(input_shape_.begin(), input_shape_.end());

    try {
        const size_t shape_elements = std::accumulate(
            shape.begin(), shape.end(), size_t{1},
            [](size_t total, int64_t extent) {
                if (extent <= 0 || total > std::numeric_limits<size_t>::max() /
                                             static_cast<size_t>(extent))
                    throw std::overflow_error{"Invalid TVM input shape"};
                return total * static_cast<size_t>(extent);
            });
        if (shape_elements != count)
            throw std::runtime_error{"TVM input shape does not match the input count"};

        const auto start = std::chrono::steady_clock::now();

        if (!synchronize_dma_buffer(dma_input_fd_, DMA_BUF_SYNC_START))
            throw std::runtime_error{"Failed to sync input DMA buffer for CPU access"};

        NDArray input_array = NDArray::Empty(shape, DLDataType{kDLFloat, 32, 1}, {kDLCPU, 0});
        input_array.CopyFromBytes(input, count * sizeof(float));
        if (input_name_.empty())
            (*set_input_)(0, input_array);
        else
            (*set_input_)(String{input_name_}, input_array);

        if (!synchronize_dma_buffer(dma_input_fd_, DMA_BUF_SYNC_END))
            throw std::runtime_error{"Failed to end input DMA buffer sync"};

        (*run_)();

        if (!synchronize_dma_buffer(dma_output_fd_, DMA_BUF_SYNC_START))
            throw std::runtime_error{"Failed to sync output DMA buffer for CPU access"};

        NDArray out = (*get_output_)(0);
        const size_t output_elements = tensor_element_count(out.operator->());
        if (output_elements > count)
            throw std::runtime_error{"Output buffer too small: need " +
                                     std::to_string(output_elements) + " floats, got " +
                                     std::to_string(count)};
        out.CopyToBytes(output, output_elements * sizeof(float));

        if (!synchronize_dma_buffer(dma_output_fd_, DMA_BUF_SYNC_END))
            throw std::runtime_error{"Failed to complete output DMA buffer sync"};

        const auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

        std::cout << "[TVM] Inference done in " << ms << " ms, output: "
                  << output_elements << " floats" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[TVM] Inference failed: " << e.what() << std::endl;
        return false;
    }
}

bool TvmInferenceClient::run_inference(std::vector<float>& input_data,
                                       std::vector<float>& output_data,
                                       size_t input_bytes) {
    if (input_bytes != input_data.size() * sizeof(float)) {
        std::cerr << "[TVM] Legacy input byte count does not match input data" << std::endl;
        return false;
    }
    return run_inference(static_cast<const std::vector<float>&>(input_data), output_data);
}

bool TvmInferenceClient::run_inference(const std::string& bin_path) {
    if (!initialized_) {
        std::cerr << "[TVM] Not initialized" << std::endl;
        return false;
    }

    std::ifstream f(bin_path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        std::cerr << "[TVM] Cannot open: " << bin_path << std::endl;
        return false;
    }
    const auto end_position = f.tellg();
    if (end_position <= 0 ||
        end_position % static_cast<std::streamoff>(sizeof(float)) != 0) {
        std::cerr << "[TVM] Input must be a non-empty float32 tensor" << std::endl;
        return false;
    }
    const auto file_bytes = static_cast<size_t>(end_position);
    f.seekg(0, std::ios::beg);

    const size_t num_floats = file_bytes / sizeof(float);
    input_data_.resize(num_floats);
    if (!f.read(reinterpret_cast<char*>(input_data_.data()),
                static_cast<std::streamsize>(file_bytes))) {
        std::cerr << "[TVM] Failed to read complete input tensor" << std::endl;
        return false;
    }

    std::cout << "[TVM] Loaded " << num_floats << " floats from " << bin_path << std::endl;

    return run_inference(input_data_, output_data_,
                         std::vector<int64_t>{static_cast<int64_t>(num_floats)});
}
