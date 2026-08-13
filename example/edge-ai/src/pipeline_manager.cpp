#include "pipeline_manager.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <iomanip>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <sndfile.h>
#include <alsa/asoundlib.h>

extern "C" {
#include "fw_loader.h"
#include <readline/readline.h>
#include <readline/history.h>
}

namespace {

// Pipeline JSON file paths for --mode command-line invocation
static const char* PIPELINE_FILE_STFT_ISTFT = "json_files/pipeline_stft_istft.json";
static const char* PIPELINE_FILE_TVM_ONLY = "json_files/pipeline_tvm_inference.json";
static const char* PIPELINE_FILE_FULL = "json_files/pipeline_audio_enhancement.json";

class PipelineError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

std::string hex_address(uint64_t address)
{
    std::ostringstream value;
    value << "0x" << std::hex << address;
    return value.str();
}

class DmaBuffer {
public:
    DmaBuffer(size_t bytes, std::string_view purpose)
    {
        if (bytes > std::numeric_limits<uint32_t>::max())
            throw PipelineError{"DMA allocation is larger than the API limit"};
        char heap[] = "linux,cma";
        char remoteproc[] = "/dev/remoteproc0";
        if (dmabuf_heap_init(heap, static_cast<uint32_t>(bytes), remoteproc, &params_) != 0)
            throw PipelineError{"Failed to allocate DMA buffer for " + std::string{purpose}};
        allocated_ = true;
    }

    ~DmaBuffer() { if (allocated_) dmabuf_heap_destroy(&params_); }
    DmaBuffer(const DmaBuffer&) = delete;
    DmaBuffer& operator=(const DmaBuffer&) = delete;

    dma_buf_params* operator->() noexcept { return &params_; }
    const dma_buf_params* operator->() const noexcept { return &params_; }

    template <typename T>
    T* data() noexcept { return reinterpret_cast<T*>(params_.kern_addr); }

    void begin_cpu_access() const { sync(DMA_BUF_SYNC_START); }
    void end_cpu_access() const { sync(DMA_BUF_SYNC_END); }

private:
    void sync(int operation) const
    {
        if (dmabuf_sync(params_.dma_buf_fd, operation) != 0)
            throw PipelineError{"DMA buffer synchronization failed"};
    }

    dma_buf_params params_{};
    bool allocated_{false};
};

class AudioStream {
public:
    AudioStream() noexcept { open(); }
    ~AudioStream() { close_fd(client_); close_fd(server_); ::unlink(socket_path_.data()); }
    AudioStream(const AudioStream&) = delete;
    AudioStream& operator=(const AudioStream&) = delete;

    void send_frame(uint8_t direction, const void* pcm, size_t bytes) noexcept
    {
        if (server_ < 0 || !pcm || bytes > std::numeric_limits<uint32_t>::max())
            return;
        if (client_ < 0)
            client_ = ::accept(server_, nullptr, nullptr);
        if (client_ < 0)
            return;
        const auto header = make_header(direction, static_cast<uint32_t>(bytes));
        if (!send_all(header.data(), header.size()) || !send_all(pcm, bytes))
            close_fd(client_);
    }

private:
    static std::array<std::byte, 13> make_header(uint8_t direction,
                                                 uint32_t pcm_bytes) noexcept
    {
        std::array<std::byte, 13> header{};
        header[0] = std::byte{'E'};
        header[1] = std::byte{'A'};
        header[2] = std::byte{'S'};
        header[3] = std::byte{'P'};
        header[4] = static_cast<std::byte>(direction);
        write_u32_le(header, 5, 16000);
        write_u32_le(header, 9, pcm_bytes);
        return header;
    }

    static void write_u32_le(std::array<std::byte, 13>& destination,
                             size_t offset, uint32_t value) noexcept
    {
        for (size_t index = 0; index < sizeof(value); ++index)
            destination[offset + index] =
                static_cast<std::byte>((value >> (index * 8)) & 0xffU);
    }

    void open() noexcept
    {
        ::unlink(socket_path_.data());
        server_ = ::socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK, 0);
        if (server_ < 0)
            return;
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        std::copy(socket_path_.begin(), socket_path_.end(), address.sun_path);
        if (::bind(server_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
            ::listen(server_, 1) != 0)
            close_fd(server_);
    }

    bool send_all(const void* data, size_t bytes) noexcept
    {
        const auto* cursor = static_cast<const std::byte*>(data);
        size_t sent = 0;
        while (sent < bytes) {
            const auto count = ::send(client_, cursor + sent, bytes - sent, MSG_NOSIGNAL);
            if (count > 0) {
                sent += static_cast<size_t>(count);
            } else if (count < 0 && errno == EINTR) {
                continue;
            } else {
                return false;
            }
        }
        return true;
    }

    static void close_fd(int& descriptor) noexcept
    {
        if (descriptor >= 0)
            ::close(descriptor);
        descriptor = -1;
    }

    static constexpr std::string_view socket_path_{"/tmp/edge-ai-speech.sock"};
    static_assert(socket_path_.size() < sizeof(sockaddr_un{}.sun_path),
                  "Audio stream socket path is too long");
    int server_{-1};
    int client_{-1};
};

} // namespace

void validate_pipeline_files() {
    const std::vector<const char*> pipeline_files = {
        PIPELINE_FILE_STFT_ISTFT,
        PIPELINE_FILE_TVM_ONLY,
        PIPELINE_FILE_FULL
    };

    for (const auto& file_path : pipeline_files) {
        // Try to open the file for reading
        if (!std::filesystem::is_regular_file(file_path))
            throw PipelineError{"Required pipeline file is missing: " + std::string{file_path}};
    }
}

// Helper function to load JSON from file
static std::string load_json_file(const std::string& file_path);


PipelineManager::PipelineManager()
    : initialized_(false), app_name_("edge-ai")
{
    state_.artifacts_loaded = false;
    state_.tvm_artifacts_configured = false;
    state_.input_configured = false;
    state_.current_pipeline_file = "";
    state_.current_input_file = "";
    state_.tvm_staging_buffer = nullptr;
    state_.tvm_result_buffer = nullptr;
    state_.staging_buffer_size = 0;
    state_.result_buffer_size = 0;
}

PipelineManager::~PipelineManager()
{
}

bool PipelineManager::initialize()
{
    tvm_client_ = std::make_shared<TvmInferenceClient>();
    generic_client_ = std::make_unique<GenericTaskClient>();

    // TVM client will be initialized later when pipeline is loaded (needs artifacts path from config)

    initializeCommands();
    initialized_ = true;
    return true;
}

int PipelineManager::run()
{
    if (!initialize()) {
        std::cout << "[App] Failed to initialize application" << std::endl;
        return -1;
    }

    printWelcome();

    while (true) {
        std::string prompt = getCurrentPrompt();
        std::unique_ptr<char, decltype(&std::free)> input_line{
            readline(prompt.c_str()), &std::free};

        if (!input_line) {
            std::cout << std::endl << "[App] EOF received, exiting..." << std::endl;
            break;
        }

        std::string line(input_line.get());

        if (!line.empty() && line.find_first_not_of(" \t") != std::string::npos) {
            add_history(input_line.get());
        }

        if (line.empty() || line.find_first_not_of(" \t") == std::string::npos) {
            continue;
        }

        auto tokens = parseCommand(line);
        if (tokens.empty()) {
            continue;
        }

        CommandResult result = executeCommand(tokens);
        if (result == CommandResult::QUIT) {
            break;
        }
    }

    return 0;
}

// Helper function implementation
static std::string load_json_file(const std::string& file_path)
{
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "[App] Error: Cannot open pipeline file: " << file_path << std::endl;
        return "";
    }

    std::string json_content((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
    return json_content;
}

int PipelineManager::run_direct(PipelineMode mode, const std::string& input_file,
                                const std::string& artifacts_path)
{
    if (!initialize()) {
        std::cout << "[App] Failed to initialize application" << std::endl;
        return -1;
    }

    std::string json_file;
    switch (mode) {
        case PipelineMode::STFT_ISTFT: json_file = PIPELINE_FILE_STFT_ISTFT; break;
        case PipelineMode::TVM_ONLY:   json_file = PIPELINE_FILE_TVM_ONLY;   break;
        case PipelineMode::FULL:       json_file = PIPELINE_FILE_FULL;        break;
    }

    std::string json_content = load_json_file(json_file);
    if (json_content.empty()) {
        std::cout << "[App] Error: Failed to read pipeline file: " << json_file << std::endl;
        std::cout << "[App] Hint: Make sure json_files directory exists in the working directory" << std::endl;
        return -1;
    }

    if (!loadPipelineFromJson(json_content)) {
        std::cout << "[App] Error: Failed to load pipeline configuration" << std::endl;
        return -1;
    }

    std::cout << "[App] Pipeline: " << state_.pipeline_config.pipeline_id << std::endl;
    std::cout << "[App] Description: " << state_.pipeline_config.description << std::endl;

    if (!artifacts_path.empty()) {
        if (handleTvmArtifacts({artifacts_path}) != CommandResult::SUCCESS) {
            return -1;
        }
    }

    if (handleInput({input_file}) != CommandResult::SUCCESS) {
        return -1;
    }

    CommandResult result = handleRun({});
    return (result == CommandResult::SUCCESS) ? 0 : 1;
}

int PipelineManager::run_from_json_file(const std::string& json_file_path)
{
    if (!initialize()) {
        std::cout << "[App] Failed to initialize application" << std::endl;
        return -1;
    }

    std::ifstream file(json_file_path);
    if (!file.is_open()) {
        std::cout << "[App] Error: JSON file not found: " << json_file_path << std::endl;
        return -1;
    }

    std::string json_content((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
    if (!loadPipelineFromJson(json_content)) {
        std::cout << "[App] Error: Failed to parse pipeline configuration" << std::endl;
        return -1;
    }

    state_.current_pipeline_file = json_file_path;

    std::cout << "[App] Pipeline: " << state_.pipeline_config.pipeline_id << std::endl;
    std::cout << "[App] Description: " << state_.pipeline_config.description << std::endl;

    if (!state_.pipeline_config.artifacts_path.empty()) {
        if (handleTvmArtifacts({state_.pipeline_config.artifacts_path}) != CommandResult::SUCCESS) {
            return -1;
        }
    }

    if (state_.pipeline_config.input_file.empty()) {
        std::cout << "[App] Error: No input_file specified in pipeline JSON" << std::endl;
        return -1;
    }

    if (handleInput({state_.pipeline_config.input_file}) != CommandResult::SUCCESS) {
        return -1;
    }

    CommandResult result = handleRun({});
    return (result == CommandResult::SUCCESS) ? 0 : 1;
}

void PipelineManager::initializeCommands()
{
    registerCommand("help", "Show available commands",
                   {"help"},
                   [this](const auto& args) { return handleHelp(args); });

    registerCommand("pipeline", "Load pipeline configuration from JSON file",
                   {"pipeline <file.json>", "pipeline sample.json"},
                   [this](const auto& args) { return handlePipeline(args); });

    registerCommand("tvm_artifacts", "Configure TVM model artifacts",
                   {"tvm_artifacts <file1.so> [file2.so] ...", "tvm_artifacts model.so"},
                   [this](const auto& args) { return handleTvmArtifacts(args); });

    registerCommand("input", "Set input data for pipeline execution",
                   {"input <file>", "input sample.wav"},
                   [this](const auto& args) { return handleInput(args); });

    registerCommand("show_pipeline", "Display current pipeline structure",
                   {"show_pipeline"},
                   [this](const auto& args) { return handleShowPipeline(args); });

    registerCommand("run", "Execute pipeline (loads input and artifacts as needed)",
                   {"run"},
                   [this](const auto& args) { return handleRun(args); });

    registerCommand("status", "Show current application state",
                   {"status"},
                   [this](const auto& args) { return handleStatus(args); });

    registerCommand("quit", "Exit the application",
                   {"quit", "exit"},
                   [this](const auto& args) { return handleQuit(args); });
}

void PipelineManager::printWelcome()
{
    std::cout << "\n===========================================\n";
    std::cout << "       RPMsg Inference Example\n";
    std::cout << "===========================================\n";
    std::cout << "Type 'help' for available commands\n" << std::endl;
}

void PipelineManager::printPrompt()
{
    std::cout << getCurrentPrompt();
}

std::vector<std::string> PipelineManager::parseCommand(const std::string& input)
{
    std::vector<std::string> tokens;
    std::istringstream iss(input);
    std::string token;

    while (iss >> token) {
        tokens.push_back(token);
    }

    return tokens;
}

PipelineManager::CommandResult PipelineManager::executeCommand(const std::vector<std::string>& tokens)
{
    if (tokens.empty()) {
        return CommandResult::SUCCESS;
    }

    std::string command = tokens[0];
    std::vector<std::string> args(tokens.begin() + 1, tokens.end());

    if (command == "exit") {
        command = "quit";
    }

    auto it = commands_.find(command);
    if (it != commands_.end()) {
        return it->second.handler(args);
    } else {
        std::cout << "[App] Unknown command: " << command << std::endl;
        std::cout << "[App] Type 'help' for available commands" << std::endl;
        return CommandResult::ERROR;
    }
}

PipelineManager::CommandResult PipelineManager::handleHelp(const std::vector<std::string>&)
{
    std::cout << "\nAvailable commands:\n";
    std::cout << "===================\n";

    for (const auto& [name, info] : commands_) {
        std::cout << name << " - " << info.description << "\n";
        std::cout << "Examples: ";
        for (size_t i = 0; i < info.examples.size(); ++i) {
            std::cout << info.examples[i];
            if (i < info.examples.size() - 1) std::cout << ", ";
        }
        std::cout << "\n\n";
    }

    return CommandResult::SUCCESS;
}

PipelineManager::CommandResult PipelineManager::handlePipeline(const std::vector<std::string>& args)
{
    if (args.empty()) {
        std::cout << "[App] Error: Pipeline file required" << std::endl;
        std::cout << "Usage: pipeline <file.json>" << std::endl;
        return CommandResult::ERROR;
    }

    std::string pipeline_file = args[0];

    std::ifstream file(pipeline_file);
    if (!file.is_open()) {
        std::cout << "[App] Error: Pipeline file not found: " << pipeline_file << std::endl;
        return CommandResult::ERROR;
    }

    std::string json_content((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
    if (!loadPipelineFromJson(json_content)) {
        return CommandResult::ERROR;
    }

    state_.current_pipeline_file = pipeline_file;

    std::cout << "[App] Pipeline configuration loaded: " << state_.pipeline_config.pipeline_id << std::endl;
    std::cout << "[App] Description: " << state_.pipeline_config.description << std::endl;
    std::cout << "[App] Stages: " << state_.pipeline_config.stages.size() << std::endl;

    return CommandResult::SUCCESS;
}

bool PipelineManager::loadPipelineFromJson(const std::string& json_content)
{
    std::unique_ptr<json_object, decltype(&json_object_put)> root{
        json_tokener_parse(json_content.c_str()), &json_object_put};
    if (!root || !json_object_is_type(root.get(), json_type_object)) {
        std::cout << "[App] Error: Invalid JSON format" << std::endl;
        return false;
    }

    const auto read_string = [](json_object* object, const char* key,
                                std::string& destination, bool required) {
        json_object* value = nullptr;
        if (!json_object_object_get_ex(object, key, &value))
            return !required;
        if (!json_object_is_type(value, json_type_string))
            return false;
        destination = json_object_get_string(value);
        return !required || !destination.empty();
    };

    PipelineConfig config;
    if (!read_string(root.get(), "pipeline_id", config.pipeline_id, true) ||
        !read_string(root.get(), "description", config.description, false) ||
        !read_string(root.get(), "input_file", config.input_file, true) ||
        !read_string(root.get(), "artifacts_path", config.artifacts_path, false)) {
        std::cout << "[App] Error: Missing or invalid pipeline string field" << std::endl;
        return false;
    }

    json_object* stages = nullptr;
    if (!json_object_object_get_ex(root.get(), "stages", &stages) ||
        !json_object_is_type(stages, json_type_array) ||
        json_object_array_length(stages) == 0) {
        std::cout << "[App] Error: Pipeline requires a non-empty stages array" << std::endl;
        return false;
    }

    const size_t stage_count = json_object_array_length(stages);
    config.stages.reserve(stage_count);
    for (size_t index = 0; index < stage_count; ++index) {
        json_object* object = json_object_array_get_idx(stages, index);
        PipelineStage stage;
        if (!object || !json_object_is_type(object, json_type_object) ||
            !read_string(object, "stage_id", stage.stage_id, true) ||
            !read_string(object, "service", stage.service, true) ||
            !read_string(object, "message_type", stage.message_type, true) ||
            (stage.service != "generic" && stage.service != "tvm")) {
            std::cout << "[App] Error: Invalid pipeline stage at index " << index << std::endl;
            return false;
        }

        json_object* parameters = nullptr;
        if (json_object_object_get_ex(object, "parameters", &parameters)) {
            if (!json_object_is_type(parameters, json_type_object)) {
                std::cout << "[App] Error: Stage parameters must be an object" << std::endl;
                return false;
            }
            json_object_object_foreach(parameters, key, value) {
                if (!json_object_is_type(value, json_type_string) &&
                    !json_object_is_type(value, json_type_int)) {
                    std::cout << "[App] Error: Stage parameter '" << key
                              << "' must be a string or integer" << std::endl;
                    return false;
                }
                stage.parameters.emplace(key, json_object_get_string(value));
            }
        }
        config.stages.push_back(std::move(stage));
    }

    config.loaded = true;
    state_.pipeline_config = std::move(config);
    return true;
}

PipelineManager::CommandResult PipelineManager::handleTvmArtifacts(const std::vector<std::string>& args)
{
    if (args.empty()) {
        std::cout << "[App] Error: At least one artifact file/directory required" << std::endl;
        std::cout << "Usage: tvm_artifacts <file1.so> [file2.so] ..." << std::endl;
        return CommandResult::ERROR;
    }

    state_.tvm_artifacts_paths.clear();

    for (const std::string& artifact : args) {
        if (!std::filesystem::exists(artifact)) {
            std::cout << "[App] Error: Artifact not found: " << artifact << std::endl;
            return CommandResult::ERROR;
        }
        state_.tvm_artifacts_paths.push_back(artifact);
    }

    state_.tvm_artifacts_configured = true;

    std::cout << "[App] TVM artifacts configured (" << args.size() << " files):" << std::endl;
    for (const std::string& artifact : state_.tvm_artifacts_paths) {
        std::cout << "[App]   " << artifact << std::endl;
    }

    std::cout << "[App] TVM artifacts path set (will be used when TVM stage is enabled)" << std::endl;
    return CommandResult::SUCCESS;
}

PipelineManager::CommandResult PipelineManager::handleInput(const std::vector<std::string>& args)
{
    if (args.empty()) {
        std::cout << "[App] Error: Input file required" << std::endl;
        std::cout << "Usage: input <file>" << std::endl;
        return CommandResult::ERROR;
    }

    std::string input_file = args[0];

    std::ifstream file(input_file);
    if (!file.is_open()) {
        std::cout << "[App] Error: Input file not found: " << input_file << std::endl;
        return CommandResult::ERROR;
    }
    const auto extension = std::filesystem::path{input_file}.extension();
    if (extension == ".wav") {
        state_.input_type = InputType::AUDIO_WAV;
    } else if (extension == ".bin") {
        state_.input_type = InputType::TENSOR_BIN;
    } else {
        std::cout << "[App] Warning: Unknown input type for file: " << input_file << std::endl;
        state_.input_type = InputType::UNKNOWN;
    }

    state_.current_input_file = input_file;
    state_.input_configured = true;

    std::string type_str;
    switch (state_.input_type) {
        case InputType::AUDIO_WAV: type_str = "WAV audio"; break;
        case InputType::TENSOR_BIN: type_str = "BIN tensor"; break;
        default: type_str = "unknown"; break;
    }

    std::cout << "[App] Input configured: " << input_file << " (type: " << type_str << ")" << std::endl;

    return CommandResult::SUCCESS;
}

PipelineManager::CommandResult PipelineManager::handleShowPipeline(const std::vector<std::string>&)
{
    if (!state_.pipeline_config.loaded) {
        std::cout << "[App] No pipeline configuration loaded" << std::endl;
        std::cout << "Use 'pipeline <file.json>' to load a pipeline configuration" << std::endl;
        return CommandResult::SUCCESS;
    }

    std::cout << std::endl << "Pipeline Configuration:" << std::endl;
    std::cout << "======================" << std::endl;
    std::cout << "Pipeline ID: " << state_.pipeline_config.pipeline_id << std::endl;
    std::cout << "Description: " << state_.pipeline_config.description << std::endl;
    std::cout << "Source File: " << state_.current_pipeline_file << std::endl;
    std::cout << std::endl << "Stages (" << state_.pipeline_config.stages.size() << "):" << std::endl;

    for (size_t i = 0; i < state_.pipeline_config.stages.size(); i++) {
        const auto& stage = state_.pipeline_config.stages[i];
        std::cout << std::endl << "  [" << (i + 1) << "] " << stage.stage_id << " (" << stage.service << " service)" << std::endl;
        std::cout << "      Message Type: " << stage.message_type << std::endl;

        if (!stage.parameters.empty()) {
            std::cout << "      Parameters:" << std::endl;
            for (const auto& [key, value] : stage.parameters) {
                std::cout << "        " << key << ": " << value << std::endl;
            }
        }
    }

    return CommandResult::SUCCESS;
}

PipelineManager::CommandResult PipelineManager::handleRun(const std::vector<std::string>&)
{
    if (!validateConfiguration()) {
        return CommandResult::ERROR;
    }

    std::cout << "[App] Executing sequential pipeline: " << state_.pipeline_config.pipeline_id << std::endl;
    std::cout << "[App] Stages: " << state_.pipeline_config.stages.size() << std::endl;

    return executeSequentialPipeline();
}

PipelineManager::CommandResult PipelineManager::handleStatus(const std::vector<std::string>&)
{
    std::cout << std::endl << "System Status:" << std::endl;
    std::cout << "==============" << std::endl;
    std::cout << "Application: " << (initialized_ ? "Initialized" : "Not Initialized") << std::endl;

    if (state_.pipeline_config.loaded) {
        std::cout << std::endl << "Pipeline Configuration:" << std::endl;
        std::cout << "======================" << std::endl;
        std::cout << "Pipeline ID: " << state_.pipeline_config.pipeline_id << std::endl;
        std::cout << "Description: " << state_.pipeline_config.description << std::endl;
        std::cout << "Source File: " << state_.current_pipeline_file << std::endl;
        for (size_t i = 0; i < state_.pipeline_config.stages.size(); i++) {
            std::cout << "  [" << (i + 1) << "] " << state_.pipeline_config.stages[i].stage_id
                      << " (" << state_.pipeline_config.stages[i].service << ")" << std::endl;
        }
    } else {
        std::cout << "Pipeline Configuration: Not loaded" << std::endl;
    }

    std::cout << std::endl << "Configuration Status:" << std::endl;
    std::cout << "=====================" << std::endl;
    std::cout << "TVM Artifacts: " << (state_.tvm_artifacts_configured ? "Configured" : "Not configured") << std::endl;
    if (state_.tvm_artifacts_configured) {
        std::cout << "  Files: " << state_.tvm_artifacts_paths.size() << std::endl;
    }
    std::cout << "Input Data: " << (state_.input_configured ? "Configured" : "Not configured") << std::endl;
    if (state_.input_configured) {
        std::cout << "  File: " << state_.current_input_file << std::endl;
    }
    std::cout << "Artifacts Loaded: " << (state_.artifacts_loaded ? "Yes" : "No") << std::endl;

    std::cout << std::endl << "Readiness Check:" << std::endl;
    std::cout << "================" << std::endl;
    bool ready = state_.pipeline_config.loaded && state_.tvm_artifacts_configured && state_.input_configured;
    std::cout << "Ready to run: " << (ready ? "Yes" : "No") << std::endl;

    return CommandResult::SUCCESS;
}

PipelineManager::CommandResult PipelineManager::handleQuit(const std::vector<std::string>&)
{
    std::cout << "[App] Exiting..." << std::endl;
    return CommandResult::QUIT;
}

bool PipelineManager::validateConfiguration()
{
    if (!state_.pipeline_config.loaded) {
        std::cout << "[App] Error: No pipeline configuration loaded" << std::endl;
        std::cout << "Use 'pipeline <file.json>' to load a pipeline configuration" << std::endl;
        return false;
    }

    bool has_tvm_stages = false;
    for (const auto& stage : state_.pipeline_config.stages) {
        if (stage.service == "tvm") {
            has_tvm_stages = true;
            break;
        }
    }

    if (has_tvm_stages && !state_.tvm_artifacts_configured) {
        std::cout << "[App] Error: Pipeline has TVM stages but no TVM artifacts configured" << std::endl;
        std::cout << "Use 'tvm_artifacts <file1.so> ...' to configure TVM artifacts" << std::endl;
        return false;
    }

    if (!state_.input_configured) {
        std::cout << "[App] Error: No input data configured" << std::endl;
        std::cout << "Use 'input <file>' to configure input data" << std::endl;
        return false;
    }

    return true;
}

std::string PipelineManager::getCurrentPrompt()
{
    return app_name_ + "> ";
}

void PipelineManager::registerCommand(const std::string& name, const std::string& description,
                    const std::vector<std::string>& examples,
                    std::function<CommandResult(const std::vector<std::string>&)> handler)
{
    commands_[name] = {description, examples, handler};
}

PipelineManager::CommandResult PipelineManager::executeTensorPipeline() {
    std::cout << "\n[App] === Executing Tensor-Only Pipeline ===" << std::endl;

    // Verify we have exactly 1 TVM stage
    if (state_.pipeline_config.stages.size() != 1 ||
        state_.pipeline_config.stages[0].service != "tvm") {
        std::cout << "[App] Error: Tensor pipeline must have exactly 1 TVM stage" << std::endl;
        return CommandResult::ERROR;
    }

    std::cout << "[App] Running TVM inference" << std::endl;
    std::cout << "[App] Artifacts: " << state_.tvm_artifacts_paths[0] << std::endl;
    std::cout << "[App] Input:     " << state_.current_input_file << std::endl;

    if (!tvm_client_->initialize(state_.tvm_artifacts_paths[0])) {
        std::cout << "[App] Error: Failed to initialize TVM client" << std::endl;
        return CommandResult::ERROR;
    }

    if (!tvm_client_->run_inference(state_.current_input_file)) {
        std::cout << "[App] Error: TVM inference failed" << std::endl;
        return CommandResult::ERROR;
    }

    // Save output to .bin file
    const std::filesystem::path input_path{state_.current_input_file};
    const std::string output_file =
        (input_path.parent_path() / (input_path.stem().string() + "_output.bin")).string();

    const std::vector<float>& output = tvm_client_->get_output();
    if (!saveTensorFile(output_file, output)) {
        std::cout << "[App] Error: Failed to save output tensor" << std::endl;
        return CommandResult::ERROR;
    }

    std::cout << "[App] Output saved to: " << output_file << std::endl;
    std::cout << "[App] Pipeline completed successfully" << std::endl;

    return CommandResult::SUCCESS;
}

PipelineManager::CommandResult PipelineManager::executeSequentialPipeline()
{
    if (state_.input_type == InputType::TENSOR_BIN)
        return executeTensorPipeline();

    try {
        if (state_.input_type != InputType::AUDIO_WAV)
            throw PipelineError{"Unknown input type"};

        // Load audio file
        std::vector<int16_t> audio_data;
        if (!loadAudioFile(state_.current_input_file, audio_data))
            throw PipelineError{"Failed to load input audio"};
        const size_t original_sample_count = audio_data.size();

        // GCRN signal processing parameters (from model_config.h)
        const size_t GCRN_HOP_SIZE    = 160;  // STFT_INPUT_SAMPLES
        const size_t GCRN_MODEL_ELEMS = 322;  // STFT_NUM_BINS*2 = 161*2
        const size_t GCRN_FFT_SIZE    = 320;  // actual FFT size (fft_size/2+1 = 161 bins)
        const size_t GCRN_BATCH_N     = 64;   // max frames per STFT/ISTFT call
        const size_t GCRN_PAD_FRAMES  = 17;   // padding frames to reach 401 per TVM window
        // TVM window = 6×64 + 17 = 401 frames, fixed input shape [1,2,401,161]
        const size_t GCRN_TOTAL_FRAMES   = 6 * GCRN_BATCH_N + GCRN_PAD_FRAMES; // 401
        const size_t GCRN_NUM_BATCHES    = 7;

        // Buffer sizes
        size_t audio_batch_bytes    = GCRN_BATCH_N      * GCRN_HOP_SIZE    * sizeof(int16_t); // 20480 bytes
        size_t spectral_total_bytes = GCRN_TOTAL_FRAMES  * GCRN_MODEL_ELEMS * sizeof(float);   // 516488 bytes

        // Detect if pipeline has a TVM stage
        const bool has_tvm_stage = std::any_of(
            state_.pipeline_config.stages.begin(), state_.pipeline_config.stages.end(),
            [](const PipelineStage& stage) { return stage.service == "tvm"; });

        std::cout << "[App] GCRN configuration:" << std::endl;
        std::cout << "[App]   HOP=" << GCRN_HOP_SIZE << " MODEL_ELEMS=" << GCRN_MODEL_ELEMS
                  << " BATCH_N=" << GCRN_BATCH_N << " TOTAL_FRAMES=" << GCRN_TOTAL_FRAMES << std::endl;
        std::cout << "[App]   Spectral buffer: " << spectral_total_bytes << " bytes (401 frames)" << std::endl;
        std::cout << "[App]   TVM stage: " << (has_tvm_stage ? "enabled" : "bypassed") << std::endl;

        // Initialize generic client
        if (!generic_client_->initialize())
            throw PipelineError{"Failed to initialize Generic Task client"};

        DmaBuffer dma_buf1{audio_batch_bytes, "STFT input"};
        DmaBuffer dma_buf2{spectral_total_bytes, "STFT output"};
        DmaBuffer dma_buf3{spectral_total_bytes, "interleave output"};
        DmaBuffer dma_buf4{audio_batch_bytes, "ISTFT output"};
        DmaBuffer dma_buf5{spectral_total_bytes, "deinterleave output"};
        DmaBuffer dma_buf6{spectral_total_bytes, "interleave input"};

        std::cout << "[App] DMA buffers:" << std::endl;
        std::cout << "[App]   buf1 (STFT audio in):      phys=0x" << std::hex << dma_buf1->phys_addr
                  << std::dec << " size=" << dma_buf1->size << std::endl;
        std::cout << "[App]   buf2 (STFT out/deint in):  phys=0x" << std::hex << dma_buf2->phys_addr
                  << std::dec << " size=" << dma_buf2->size << std::endl;
        std::cout << "[App]   buf3 (int out/ISTFT in):   phys=0x" << std::hex << dma_buf3->phys_addr
                  << std::dec << " size=" << dma_buf3->size << std::endl;
        std::cout << "[App]   buf4 (ISTFT audio out):    phys=0x" << std::hex << dma_buf4->phys_addr
                  << std::dec << " size=" << dma_buf4->size << std::endl;
        std::cout << "[App]   buf5 (deint out):          phys=0x" << std::hex << dma_buf5->phys_addr
                  << std::dec << " size=" << dma_buf5->size << std::endl;
        std::cout << "[App]   buf6 (interleave in):      phys=0x" << std::hex << dma_buf6->phys_addr
                  << std::dec << " size=" << dma_buf6->size << std::endl;

        // Find STFT and ISTFT stages once
        const PipelineStage* stft_stage_ptr  = nullptr;
        const PipelineStage* istft_stage_ptr = nullptr;
        for (const auto& stage : state_.pipeline_config.stages) {
            if (stage.message_type == "C7X_MSG_STFT_ANALYZE")         stft_stage_ptr  = &stage;
            else if (stage.message_type == "C7X_MSG_ISTFT_SYNTHESIZE") istft_stage_ptr = &stage;
        }
        if (!stft_stage_ptr || !istft_stage_ptr)
            throw PipelineError{"Audio pipeline requires STFT and ISTFT stages"};

        // Each full chunk consumes exactly 401 frames.
        // Remaining frames < 401 are zero-padded to 401 and processed as a partial chunk;
        // only the real frames are kept from its ISTFT output.
        const size_t total_frames =
            (audio_data.size() + GCRN_HOP_SIZE - 1) / GCRN_HOP_SIZE;
        audio_data.resize(total_frames * GCRN_HOP_SIZE, int16_t{0});
        size_t num_full_chunks  = total_frames / GCRN_TOTAL_FRAMES;
        size_t partial_frames   = total_frames % GCRN_TOTAL_FRAMES;
        size_t num_chunks       = num_full_chunks + (partial_frames > 0 ? 1 : 0);

        std::cout << "[App] Full file processing:" << std::endl;
        std::cout << "[App]   Total frames: " << total_frames
                  << " | Full chunks: " << num_full_chunks
                  << " | Partial chunk: " << partial_frames << " real frames"
                  << " (zero-padded to 401)" << std::endl;

        std::vector<int16_t> processed_audio_data;
        processed_audio_data.reserve(total_frames * GCRN_HOP_SIZE);

        AudioStream audio_stream;

        uint64_t stft_out_base  = dma_buf2->phys_addr;
        uint64_t istft_src_base = dma_buf3->phys_addr;

        // Initialize TVM once before the chunk loop
        if (has_tvm_stage && state_.tvm_artifacts_configured && !tvm_client_->is_initialized()) {
            if (!tvm_client_->initialize(state_.tvm_artifacts_paths[0]))
                throw PipelineError{"Failed to initialize TVM client"};
            tvm_client_->set_input_shape({1, 2, 401, 161});
            std::cout << "[App] TVM initialized, input shape [1,2,401,161]" << std::endl;
        }

        auto t_total_start = std::chrono::steady_clock::now();

        for (size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
            size_t chunk_frame_offset = chunk_idx * GCRN_TOTAL_FRAMES;
            const size_t num_batches_this_chunk = GCRN_NUM_BATCHES;
            // For the partial chunk, how many real frames it contains (0 means full chunk)
            size_t real_frames_this_chunk = (chunk_idx == num_full_chunks && partial_frames > 0)
                                            ? partial_frames : GCRN_TOTAL_FRAMES;

            auto t_chunk_start = std::chrono::steady_clock::now();

            // =====================================================
            // Phase 1: STFT
            // =====================================================
            auto t_stft_start = std::chrono::steady_clock::now();
            for (size_t batch_idx = 0; batch_idx < num_batches_this_chunk; batch_idx++) {
                size_t frames_this_batch  = (batch_idx < 6) ? GCRN_BATCH_N : GCRN_PAD_FRAMES;
                size_t samples_this_batch = frames_this_batch * GCRN_HOP_SIZE;
                size_t audio_offset       = (chunk_frame_offset + batch_idx * GCRN_BATCH_N) * GCRN_HOP_SIZE;
                size_t audio_bytes        = samples_this_batch * sizeof(int16_t);
                uint64_t spectral_offset  = batch_idx * GCRN_BATCH_N * GCRN_MODEL_ELEMS * sizeof(float);

                dma_buf1.begin_cpu_access();
                std::fill_n(dma_buf1.data<std::byte>(), audio_batch_bytes, std::byte{});
                if (audio_offset < audio_data.size()) {
                    const size_t available = std::min(
                        samples_this_batch, audio_data.size() - audio_offset);
                    std::copy_n(audio_data.begin() + static_cast<std::ptrdiff_t>(audio_offset),
                                available, dma_buf1.data<int16_t>());
                }
                dma_buf1.end_cpu_access();
                audio_stream.send_frame(0, dma_buf1.data<std::byte>(), audio_bytes);

                auto params = stft_stage_ptr->parameters;
                params["input_buffer"]  = hex_address(dma_buf1->phys_addr);
                params["output_buffer"] = hex_address(stft_out_base + spectral_offset);
                params["input_frame"]   = std::to_string(frames_this_batch);
                params["output_frame"]  = std::to_string(frames_this_batch);

                if (debug_)
                    std::cout << "[App]   STFT batch " << (batch_idx+1) << "/" << num_batches_this_chunk
                              << ": " << frames_this_batch << " frames"
                              << " in=0x" << std::hex << dma_buf1->phys_addr
                              << " out=0x" << (stft_out_base + spectral_offset) << std::dec << std::endl;

                auto r = generic_client_->process("C7X_MSG_STFT_ANALYZE", params);
                if (!r.success)
                    throw PipelineError{"STFT batch " + std::to_string(batch_idx + 1) +
                                        " failed: " + r.error_message};
            }
            double t_stft_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t_stft_start).count() / 1000.0;

            // =====================================================
            // Phase 2: TVM
            // =====================================================
            double t_tvm_ms = 0.0;
            auto t_tvm_start = std::chrono::steady_clock::now();

            if (debug_)
                std::cout << "[App]   Deinterleave: 0x" << std::hex << dma_buf2->phys_addr
                          << " -> 0x" << dma_buf5->phys_addr << std::dec << std::endl;
            {
                std::map<std::string, std::string> params;
                params["input_buffer"]  = hex_address(dma_buf2->phys_addr);
                params["output_buffer"] = hex_address(dma_buf5->phys_addr);
                params["input_frame"]   = std::to_string(GCRN_TOTAL_FRAMES);
                params["fft_size"]      = std::to_string(GCRN_FFT_SIZE);
                params["flag"]          = "0";  // deinterleave
                auto r = generic_client_->process("C7X_DEINTERLEAVE_MSG_ANALYZE", params);
                if (!r.success)
                    throw PipelineError{"Deinterleave failed: " + r.error_message};
            }
            deint_output_data_.resize(spectral_total_bytes / sizeof(float));
            inter_input_data_.resize(spectral_total_bytes / sizeof(float));

            dma_buf5.begin_cpu_access();
            std::copy_n(dma_buf5.data<float>(), deint_output_data_.size(),
                        deint_output_data_.begin());
            dma_buf5.end_cpu_access();
            // TVM only runs when pipeline has TVM stage
            if (has_tvm_stage && tvm_client_->is_initialized()) {
                if (!tvm_client_->run_inference(deint_output_data_, inter_input_data_))
                    throw PipelineError{"TVM inference failed"};
                t_tvm_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t_tvm_start).count() / 1000.0;
            } else {
                inter_input_data_ = deint_output_data_;
            }
            if (inter_input_data_.size() != spectral_total_bytes / sizeof(float))
                throw PipelineError{"TVM output size does not match the speech pipeline"};

            dma_buf6.begin_cpu_access();
            std::copy(inter_input_data_.begin(), inter_input_data_.end(),
                      dma_buf6.data<float>());
            dma_buf6.end_cpu_access();
            if (debug_)
                std::cout << "[App]   Interleave: 0x" << std::hex << dma_buf6->phys_addr
                          << " -> 0x" << dma_buf3->phys_addr << std::dec << std::endl;
            {
                std::map<std::string, std::string> params;
                params["input_buffer"]  = hex_address(dma_buf6->phys_addr);
                params["output_buffer"] = hex_address(dma_buf3->phys_addr);
                params["input_frame"]   = std::to_string(GCRN_TOTAL_FRAMES);
                params["fft_size"]      = std::to_string(GCRN_FFT_SIZE);
                params["flag"]          = "1";  // interleave
                auto r = generic_client_->process("C7X_DEINTERLEAVE_MSG_ANALYZE", params);
                if (!r.success)
                    throw PipelineError{"Interleave failed: " + r.error_message};
            }

            // =====================================================
            // Phase 3: ISTFT
            // =====================================================
            auto t_istft_start = std::chrono::steady_clock::now();
            size_t real_samples_remaining = real_frames_this_chunk * GCRN_HOP_SIZE;
            for (size_t batch_idx = 0; batch_idx < num_batches_this_chunk; batch_idx++) {
                size_t frames_this_batch  = (batch_idx < 6) ? GCRN_BATCH_N : GCRN_PAD_FRAMES;
                size_t samples_this_batch = frames_this_batch * GCRN_HOP_SIZE;
                size_t audio_offset       = (chunk_frame_offset + batch_idx * GCRN_BATCH_N) * GCRN_HOP_SIZE;
                uint64_t spectral_offset  = batch_idx * GCRN_BATCH_N * GCRN_MODEL_ELEMS * sizeof(float);
                auto params = istft_stage_ptr->parameters;
                params["input_buffer"]  = hex_address(istft_src_base + spectral_offset);
                params["output_buffer"] = hex_address(dma_buf4->phys_addr);
                params["input_frame"]   = std::to_string(frames_this_batch);
                params["output_frame"]  = std::to_string(frames_this_batch);

                if (debug_)
                    std::cout << "[App]   ISTFT batch " << (batch_idx+1) << "/" << num_batches_this_chunk
                              << ": " << frames_this_batch << " frames"
                              << " in=0x" << std::hex << (istft_src_base + spectral_offset)
                              << " out=0x" << dma_buf4->phys_addr << std::dec << std::endl;

                auto r = generic_client_->process("C7X_MSG_ISTFT_SYNTHESIZE", params);
                if (!r.success)
                    throw PipelineError{"ISTFT batch " + std::to_string(batch_idx + 1) +
                                        " failed: " + r.error_message};

                dma_buf4.begin_cpu_access();
                const auto* out_ptr = dma_buf4.data<int16_t>();

                if (debug_) {
                    std::cout << "[App]   Frame | InRMS  OutRMS | In[0..4]              | Out[0..4]" << std::endl;
                    const size_t available_frames = std::min(
                        frames_this_batch, real_samples_remaining / GCRN_HOP_SIZE);
                    for (size_t f = 0; f < std::min(size_t{4}, available_frames); ++f) {
                        size_t in_off  = audio_offset + f * GCRN_HOP_SIZE;
                        size_t out_off = f * GCRN_HOP_SIZE;
                        float in_sum = 0.0f, out_sum = 0.0f;
                        for (size_t i = 0; i < GCRN_HOP_SIZE; i++) {
                            const float s = static_cast<float>(audio_data[in_off + i]) / 32768.0f;
                            const float o = static_cast<float>(out_ptr[out_off + i]) / 32768.0f;
                            in_sum += s * s;
                            out_sum += o * o;
                        }
                        std::cout << "[App]   " << std::setw(5) << (chunk_frame_offset + batch_idx*GCRN_BATCH_N + f + 1)
                                  << " | " << std::fixed << std::setprecision(4)
                                  << std::sqrt(in_sum / GCRN_HOP_SIZE) << "  "
                                  << std::sqrt(out_sum / GCRN_HOP_SIZE)
                                  << " | In:";
                        for (size_t i = 0; i < 5; i++)
                            std::cout << std::setw(6) << audio_data[in_off + i] << (i<4?",":"");
                        std::cout << " | Out:";
                        for (size_t i = 0; i < 5; i++)
                            std::cout << std::setw(6) << out_ptr[out_off + i] << (i<4?",":"");
                        std::cout << std::endl;
                    }
                }

                size_t samples_to_collect = std::min(samples_this_batch, real_samples_remaining);
                std::copy_n(out_ptr, samples_to_collect,
                            std::back_inserter(processed_audio_data));
                real_samples_remaining -= samples_to_collect;

                dma_buf4.end_cpu_access();
                audio_stream.send_frame(1, out_ptr, samples_to_collect * sizeof(int16_t));
            }
            double t_istft_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t_istft_start).count() / 1000.0;

            double t_chunk_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t_chunk_start).count() / 1000.0;

            std::cout << "[App] Chunk " << (chunk_idx+1) << "/" << num_chunks
                      << " [" << real_frames_this_chunk << " real frames"
                      << (real_frames_this_chunk < GCRN_TOTAL_FRAMES ? " + zero-pad" : "") << "]"
                      << " | STFT=" << std::fixed << std::setprecision(1) << t_stft_ms << "ms"
                      << " TVM=" << t_tvm_ms << "ms"
                      << " ISTFT=" << t_istft_ms << "ms"
                      << " total=" << t_chunk_ms << "ms" << std::endl;
        }

        double t_total_ms = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t_total_start).count() / 1000.0;

        std::cout << "[App] All chunks done | total=" << std::fixed << std::setprecision(1)
                  << t_total_ms << "ms | " << (processed_audio_data.size() / 160) << " output GCRN frames ("
                  << processed_audio_data.size() << " samples)" << std::endl;

        if (processed_audio_data.size() < original_sample_count)
            throw PipelineError{"Speech pipeline produced fewer samples than expected"};
        processed_audio_data.resize(original_sample_count);

        std::string output_filename = "processed_output.wav";
        if (!saveAudioFile(output_filename, processed_audio_data))
            throw PipelineError{"Failed to save output file"};
        std::cout << "[App] Saved to " << output_filename << std::endl;
        return CommandResult::SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "[App] Pipeline failed: " << error.what() << std::endl;
        return CommandResult::ERROR;
    }
}



bool PipelineManager::loadAudioFile(const std::string& filename, std::vector<int16_t>& audio_data)
{
    SF_INFO sfinfo{};

    std::unique_ptr<SNDFILE, decltype(&sf_close)> infile{
        sf_open(filename.c_str(), SFM_READ, &sfinfo), &sf_close};
    if (!infile) {
        std::cout << "[App] Error: Failed to open audio file: " << filename << std::endl;
        return false;
    }

    // Validate audio format
    if (sfinfo.channels != 1) {
        std::cout << "[App] Error: Audio must be mono (1 channel), got " << sfinfo.channels << " channels" << std::endl;
        return false;
    }

    if (sfinfo.frames <= 0) {
        std::cout << "[App] Error: Audio file contains no samples" << std::endl;
        return false;
    }

    if (sfinfo.samplerate != 16000) {
        std::cout << "[App] Error: Audio sample rate is " << sfinfo.samplerate
                  << "Hz; this pipeline requires 16kHz" << std::endl;
        return false;
    }

    std::cout << "[App] Audio file info: " << (sfinfo.frames / 160) << " GCRN frames ("
              << sfinfo.frames << " samples), "
              << sfinfo.samplerate << "Hz, " << sfinfo.channels << " channel(s)" << std::endl;

    // Read all audio data
    audio_data.resize(sfinfo.frames);
    const sf_count_t frames_read = sf_readf_short(
        infile.get(), audio_data.data(), sfinfo.frames);

    if (frames_read < 0) {
        std::cout << "[App] Error: Failed while reading audio data" << std::endl;
        return false;
    }
    if (frames_read != sfinfo.frames) {
        std::cout << "[App] Warning: Read " << frames_read << " frames, expected " << sfinfo.frames << std::endl;
        audio_data.resize(frames_read);
    }

    std::cout << "[App] Loaded " << audio_data.size() << " audio samples ("
              << (static_cast<double>(audio_data.size()) / sfinfo.samplerate)
              << " seconds)" << std::endl;

    return true;
}

bool PipelineManager::saveAudioFile(const std::string& filename, const std::vector<int16_t>& audio_data)
{
    if (audio_data.empty()) {
        std::cout << "[App] Error: Refusing to write an empty audio file" << std::endl;
        return false;
    }
    SF_INFO sfinfo{};

    // Set output file parameters to match our audio format
    sfinfo.samplerate = 16000;    // 16kHz
    sfinfo.channels = 1;          // mono
    sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;  // WAV file with 16-bit PCM

    std::unique_ptr<SNDFILE, decltype(&sf_close)> outfile{
        sf_open(filename.c_str(), SFM_WRITE, &sfinfo), &sf_close};
    if (!outfile) {
        std::cout << "[App] Error: Failed to create output audio file: " << filename << std::endl;
        std::cout << "[App] Error details: " << sf_strerror(nullptr) << std::endl;
        return false;
    }

    std::cout << "[App] Saving audio to: " << filename << std::endl;
    std::cout << "[App] Output file info: " << (audio_data.size() / 160) << " GCRN frames ("
              << audio_data.size() << " samples), "
              << sfinfo.samplerate << "Hz, " << sfinfo.channels << " channel(s)" << std::endl;

    // Write audio data to file
    const sf_count_t frames_written = sf_writef_short(
        outfile.get(), audio_data.data(), static_cast<sf_count_t>(audio_data.size()));

    if (frames_written != static_cast<sf_count_t>(audio_data.size())) {
        std::cout << "[App] Warning: Wrote " << frames_written << " frames, expected " << audio_data.size() << std::endl;
        return false;
    }

    std::cout << "[App] Successfully saved " << (frames_written / 160) << " GCRN frames ("
              << frames_written << " samples) to " << filename << std::endl;
    std::cout << "[App] Duration: "
              << (static_cast<double>(frames_written) / sfinfo.samplerate)
              << " seconds" << std::endl;

    return true;
}

bool PipelineManager::saveTensorFile(const std::string& filename, const std::vector<float>& tensor_data) {
    if (tensor_data.empty()) {
        std::cout << "[App] Error: Refusing to write an empty tensor" << std::endl;
        return false;
    }
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "[App] Error: Cannot open file for writing: " << filename << std::endl;
        return false;
    }

    file.write(reinterpret_cast<const char*>(tensor_data.data()),
               static_cast<std::streamsize>(tensor_data.size() * sizeof(float)));
    if (!file) {
        std::cout << "[App] Error: Failed while writing tensor data" << std::endl;
        return false;
    }

    std::cout << "[App] Successfully saved " << tensor_data.size() << " float values ("
              << (tensor_data.size() * sizeof(float)) << " bytes) to " << filename << std::endl;

    return true;
}

bool PipelineManager::loadBinTensor(const std::string& filename, std::vector<float>& tensor_data) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cout << "[App] Error: Cannot open BIN file: " << filename << std::endl;
        return false;
    }

    // Get file size
    const std::streamsize size = file.tellg();
    if (size <= 0 || size % static_cast<std::streamsize>(sizeof(float)) != 0) {
        std::cout << "[App] Error: BIN file is not a non-empty float32 tensor" << std::endl;
        return false;
    }
    file.seekg(0, std::ios::beg);

    // Calculate number of float values
    const size_t num_floats = static_cast<size_t>(size) / sizeof(float);
    tensor_data.resize(num_floats);

    // Read binary data
    if (!file.read(reinterpret_cast<char*>(tensor_data.data()), size)) {
        std::cout << "[App] Error: Failed to read BIN file" << std::endl;
        return false;
    }

    return true;
}
