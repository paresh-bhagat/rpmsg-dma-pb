#include "pipeline_manager.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <set>
#include <cstring>
#include <cstdint>
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

char rproc_path[] = "/dev/remoteproc0";
char dma_pool_name[] = "linux,cma";

// TVM Fixed Memory Addresses (hardcoded - no JSON)
#define TVM_INPUT_ADDR     0xa3000000UL   // STFT output → TVM input
#define TVM_OUTPUT_ADDR    0xabc00000UL   // TVM output → ISTFT input

// Pipeline JSON file paths for --mode command-line invocation
static const char* PIPELINE_FILE_STFT_ISTFT = "json_files/pipeline_stft_istft.json";
static const char* PIPELINE_FILE_TVM_ONLY = "json_files/pipeline_tvm_inference.json";
static const char* PIPELINE_FILE_FULL = "json_files/pipeline_audio_enhancement.json";

// Header: "EASP", direction (0=input, 1=output), sample-rate, PCM bytes.
struct __attribute__((packed)) AudioStreamHeader { char magic[4]; uint8_t direction; uint32_t rate; uint32_t bytes; };
static int stream_server = -1, stream_client = -1;
static const char* stream_socket = "/tmp/edge-ai-speech.sock";

static void stream_open() {
    unlink(stream_socket);
    stream_server = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK, 0);
    if (stream_server < 0)
	return;
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    strncpy(address.sun_path, stream_socket, sizeof(address.sun_path) - 1);
    if (bind(stream_server, reinterpret_cast<sockaddr*>(&address), sizeof(address)) || listen(stream_server, 1)) {
	close(stream_server);
	stream_server = -1;
    }
}

static void stream_frame(uint8_t direction, const void* pcm, size_t bytes) {
    if (stream_server < 0)
	return;
    if (stream_client < 0)
	stream_client = accept(stream_server, nullptr, nullptr);
    if (stream_client < 0)
	return;
    AudioStreamHeader h{{'E','A','S','P'}, direction, 16000, static_cast<uint32_t>(bytes)};
    if (send(stream_client, &h, sizeof(h), MSG_NOSIGNAL) != sizeof(h) ||
        send(stream_client, pcm, bytes, MSG_NOSIGNAL) != static_cast<ssize_t>(bytes)) {
	close(stream_client);
	stream_client = -1;
    }
}

static void stream_close()
{
    if (stream_client >= 0)
        close(stream_client);
    if (stream_server >= 0)
        close(stream_server);
    stream_client = stream_server = -1;
    unlink(stream_socket);
}

void validate_pipeline_files() {
    const std::vector<const char*> pipeline_files = {
        PIPELINE_FILE_STFT_ISTFT,
        PIPELINE_FILE_TVM_ONLY,
        PIPELINE_FILE_FULL
    };

    for (const auto& file_path : pipeline_files) {
        // Try to open the file for reading
        FILE* file = std::fopen(file_path, "r");
        if (!file) {
            std::cerr << "Critical Error: Required pipeline file missing: " << file_path << std::endl;
            std::exit(EXIT_FAILURE); // Instantly terminates the program
        }
        std::fclose(file); // Clean up the file handle if it exists
    }
}

// Helper function to load JSON from file
static std::string load_json_file(const std::string& file_path);


PipelineManager::PipelineManager()
    : initialized_(false)
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


    extern char* __progname;
    app_name_ = std::string(__progname);
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

    char* input_line = nullptr;
    while (true) {
        std::string prompt = getCurrentPrompt();
        input_line = readline(prompt.c_str());

        if (!input_line) {
            std::cout << std::endl << "[App] EOF received, exiting..." << std::endl;
            break;
        }

        std::string line(input_line);

        if (!line.empty() && line.find_first_not_of(" \t") != std::string::npos) {
            add_history(input_line);
        }

        free(input_line);

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
    file.close();

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
    file.close();

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
                   std::bind(&PipelineManager::handleHelp, this, std::placeholders::_1));

    registerCommand("pipeline", "Load pipeline configuration from JSON file",
                   {"pipeline <file.json>", "pipeline sample.json"},
                   std::bind(&PipelineManager::handlePipeline, this, std::placeholders::_1));

    registerCommand("tvm_artifacts", "Configure TVM model artifacts",
                   {"tvm_artifacts <file1.so> [file2.so] ...", "tvm_artifacts model.so"},
                   std::bind(&PipelineManager::handleTvmArtifacts, this, std::placeholders::_1));

    registerCommand("input", "Set input data for pipeline execution",
                   {"input <file>", "input sample.wav"},
                   std::bind(&PipelineManager::handleInput, this, std::placeholders::_1));

    registerCommand("show_pipeline", "Display current pipeline structure",
                   {"show_pipeline"},
                   std::bind(&PipelineManager::handleShowPipeline, this, std::placeholders::_1));

    registerCommand("run", "Execute pipeline (loads input and artifacts as needed)",
                   {"run"},
                   std::bind(&PipelineManager::handleRun, this, std::placeholders::_1));

    registerCommand("status", "Show current application state",
                   {"status"},
                   std::bind(&PipelineManager::handleStatus, this, std::placeholders::_1));

    registerCommand("quit", "Exit the application",
                   {"quit", "exit"},
                   std::bind(&PipelineManager::handleQuit, this, std::placeholders::_1));
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

PipelineManager::CommandResult PipelineManager::handleHelp(const std::vector<std::string>& args)
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
    file.close();

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
    json_object* root = json_tokener_parse(json_content.c_str());
    if (!root) {
        std::cout << "[App] Error: Invalid JSON format" << std::endl;
        return false;
    }

    state_.pipeline_config = PipelineConfig();

    json_object* pipeline_id_obj;
    if (json_object_object_get_ex(root, "pipeline_id", &pipeline_id_obj)) {
        state_.pipeline_config.pipeline_id = json_object_get_string(pipeline_id_obj);
    }

    json_object* description_obj;
    if (json_object_object_get_ex(root, "description", &description_obj)) {
        state_.pipeline_config.description = json_object_get_string(description_obj);
    }

    json_object* input_file_obj;
    if (json_object_object_get_ex(root, "input_file", &input_file_obj)) {
        state_.pipeline_config.input_file = json_object_get_string(input_file_obj);
    }

    json_object* artifacts_path_obj;
    if (json_object_object_get_ex(root, "artifacts_path", &artifacts_path_obj)) {
        state_.pipeline_config.artifacts_path = json_object_get_string(artifacts_path_obj);
    }

    json_object* stages_obj;
    if (json_object_object_get_ex(root, "stages", &stages_obj)) {
        int stages_len = json_object_array_length(stages_obj);
        state_.pipeline_config.stages.clear();

        for (int i = 0; i < stages_len; i++) {
            json_object* stage_obj = json_object_array_get_idx(stages_obj, i);
            PipelineStage stage;

            json_object* stage_id_obj;
            if (json_object_object_get_ex(stage_obj, "stage_id", &stage_id_obj)) {
                stage.stage_id = json_object_get_string(stage_id_obj);
            }

            json_object* service_obj;
            if (json_object_object_get_ex(stage_obj, "service", &service_obj)) {
                stage.service = json_object_get_string(service_obj);
            }

            json_object* message_type_obj;
            if (json_object_object_get_ex(stage_obj, "message_type", &message_type_obj)) {
                stage.message_type = json_object_get_string(message_type_obj);
            }

            json_object* parameters_obj;
            if (json_object_object_get_ex(stage_obj, "parameters", &parameters_obj)) {
                json_object_object_foreach(parameters_obj, key, val) {
                    stage.parameters[key] = json_object_get_string(val);
                }
            }

            state_.pipeline_config.stages.push_back(stage);
        }
    }

    json_object_put(root);
    state_.pipeline_config.loaded = true;
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
    file.close();

    // Detect input type by extension
    if (input_file.size() >= 4 && input_file.substr(input_file.size() - 4) == ".wav") {
        state_.input_type = InputType::AUDIO_WAV;
    } else if (input_file.size() >= 4 && input_file.substr(input_file.size() - 4) == ".bin") {
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

PipelineManager::CommandResult PipelineManager::handleShowPipeline(const std::vector<std::string>& args)
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

PipelineManager::CommandResult PipelineManager::handleRun(const std::vector<std::string>& args)
{
    if (!validateConfiguration()) {
        return CommandResult::ERROR;
    }

    std::cout << "[App] Executing sequential pipeline: " << state_.pipeline_config.pipeline_id << std::endl;
    std::cout << "[App] Stages: " << state_.pipeline_config.stages.size() << std::endl;

    return executeSequentialPipeline();
}

PipelineManager::CommandResult PipelineManager::handleStatus(const std::vector<std::string>& args)
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

PipelineManager::CommandResult PipelineManager::handleQuit(const std::vector<std::string>& args)
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

    // Initialize TVM client (sets up buffers at 0xa3000000 and 0xabc00000)
    if (!tvm_client_->initialize(state_.tvm_artifacts_paths[0])) {
        std::cout << "[App] Error: Failed to initialize TVM client" << std::endl;
        return CommandResult::ERROR;
    }

    // Load BIN file and run inference (C7x DSP triggers automatically)
    if (!tvm_client_->run_inference_from_bin(state_.current_input_file)) {
        std::cout << "[App] Error: TVM inference failed" << std::endl;
        return CommandResult::ERROR;
    }

    // Save output to .bin file
    std::string output_file = state_.current_input_file.substr(0, state_.current_input_file.find_last_of('.')) + "_output.bin";

    const std::vector<float>& output = tvm_client_->get_output();
    if (!saveTensorFile(output_file, output)) {
        std::cout << "[App] Error: Failed to save output tensor" << std::endl;
        return CommandResult::ERROR;
    }

    std::cout << "[App] Output saved to: " << output_file << std::endl;
    std::cout << "[App] Pipeline completed successfully" << std::endl;

    return CommandResult::SUCCESS;
}

PipelineManager::CommandResult PipelineManager::executeAudioPipeline() {
    std::cout << "\n[App] === Executing Audio Pipeline ===" << std::endl;

    // Determine pipeline type
    bool has_stft = false;
    bool has_tvm = false;
    bool has_istft = false;

    for (const auto& stage : state_.pipeline_config.stages) {
        if (stage.message_type == "C7X_MSG_STFT_ANALYZE") has_stft = true;
        if (stage.service == "tvm") has_tvm = true;
        if (stage.message_type == "C7X_MSG_ISTFT_SYNTHESIZE") has_istft = true;
    }

    std::cout << "[App] Pipeline stages: "
              << (has_stft ? "STFT " : "")
              << (has_tvm ? "TVM " : "")
              << (has_istft ? "ISTFT" : "") << std::endl;

    // For audio pipelines, use existing implementation
    return executeSequentialPipeline();
}

PipelineManager::CommandResult PipelineManager::executeSequentialPipeline()
{
    // Router logic: Dispatch to appropriate pipeline executor
    if (state_.input_type == InputType::TENSOR_BIN) {
        return executeTensorPipeline();
    } else if (state_.input_type == InputType::AUDIO_WAV) {
        // Continue with audio pipeline execution below
    } else {
        std::cout << "[App] Error: Unknown input type" << std::endl;
        return CommandResult::ERROR;
    }

    // Load audio file
    std::vector<int16_t> audio_data;
    if (!loadAudioFile(state_.current_input_file, audio_data)) {
        return CommandResult::ERROR;
    }

    // STFT/ISTFT configuration from TI model_config.h (DCCRN model)
    const size_t BATCH_N = 30;                 // Number of windows/chunks to process
    const size_t STFT_INPUT_SAMPLES = 256;     // Audio samples per window (STFT_HOP_SIZE)
    const size_t STFT_NUM_BINS = 257;          // FFT bins
    const size_t STFT_MODEL_ELEMS = 514;       // 2 × STFT_NUM_BINS (real + imaginary)

    size_t total_audio_samples = BATCH_N * STFT_INPUT_SAMPLES;  // 30 × 100 = 3000 samples
    size_t audio_chunk_bytes = total_audio_samples * sizeof(int16_t);  // 6000 bytes
    size_t spectral_data_bytes = BATCH_N * STFT_MODEL_ELEMS * sizeof(float);  // 30 × 514 × 4 = 61680 bytes

#ifdef DEBUG
    std::cout << "[App] STFT configuration:" << std::endl;
    std::cout << "[App] Batch size (BATCH_N): " << BATCH_N << " windows" << std::endl;
    std::cout << "[App] Samples per window: " << STFT_INPUT_SAMPLES << " samples" << std::endl;
    std::cout << "[App] Total audio input: " << total_audio_samples << " samples (" << audio_chunk_bytes << " bytes)" << std::endl;
    std::cout << "[App] Spectral data: " << STFT_MODEL_ELEMS << " elements: " << BATCH_N << " = " << spectral_data_bytes << " bytes" << std::endl;
#endif

    // Initialize generic client
    if (!generic_client_->initialize()) {
        std::cout << "[App] Error: Failed to initialize Generic Task client" << std::endl;
        return CommandResult::ERROR;
    }

    // Check if pipeline has TVM stage
    bool has_tvm_stage = false;
    for (const auto& stage : state_.pipeline_config.stages) {
        if (stage.service == "tvm") {
            has_tvm_stage = true;
            break;
        }
    }

    // Initialize TVM client with artifacts only if pipeline has TVM stage
    if (has_tvm_stage && state_.tvm_artifacts_configured && !tvm_client_->is_initialized()) {
        if (!tvm_client_->initialize(state_.tvm_artifacts_paths[0])) {
            std::cout << "[App] Error: Failed to initialize TVM client" << std::endl;
            std::cout << "[App] WARNING: Continuing without TVM" << std::endl;
        }
    }

    // Allocate DMA buffers for the pipeline stages (STFT/ISTFT)
    struct dma_buf_params dma_buf1, dma_buf2, dma_buf3;
#ifdef DEBUG
    std::cout << "[App] Allocating DMA buffers from linux,cma heap..." << std::endl;
    std::cout << "[App] Using fixed TVM memory addresses:" << std::endl;
    std::cout << "[App]   TVM_INPUT_ADDR:  0x" << std::hex << TVM_INPUT_ADDR << std::dec << std::endl;
    std::cout << "[App]   TVM_OUTPUT_ADDR: 0x" << std::hex << TVM_OUTPUT_ADDR << std::dec << std::endl;
#endif
    // DMA Buffer 1: STFT input (audio samples)
    int ret1 = dmabuf_heap_init((char*)"linux,cma",
                                audio_chunk_bytes, (char*)"/dev/remoteproc0", &dma_buf1);
    if (ret1 != 0) {
        std::cout << "[App] Error: Failed to allocate DMA buffer 1 (STFT input)" << std::endl;
        return CommandResult::ERROR;
    }

    // DMA Buffer 2: STFT output / ISTFT input (spectral data - larger!)
    int ret2 = dmabuf_heap_init((char*)"linux,cma",
                                spectral_data_bytes, (char*)"/dev/remoteproc0", &dma_buf2);
    if (ret2 != 0) {
        std::cout << "[App] Error: Failed to allocate DMA buffer 2 (STFT output / ISTFT input)" << std::endl;
        dmabuf_heap_destroy(&dma_buf1);
        return CommandResult::ERROR;
    }

    // DMA Buffer 3: ISTFT output (audio samples)
    int ret3 = dmabuf_heap_init((char*)"linux,cma",
                                audio_chunk_bytes, (char*)"/dev/remoteproc0", &dma_buf3);
    if (ret3 != 0) {
        std::cout << "[App] Error: Failed to allocate DMA buffer 3 (ISTFT output)" << std::endl;
        dmabuf_heap_destroy(&dma_buf1);
        dmabuf_heap_destroy(&dma_buf2);
        return CommandResult::ERROR;
    }
#ifdef DEBUG
    std::cout << "[App] DMA buffer allocation successful:" << std::endl;
    std::cout << "[App]   DMA Buffer 1 (STFT input):  phys=0x" << std::hex << dma_buf1.phys_addr
              << ", virt=" << dma_buf1.kern_addr << ", size=" << std::dec << dma_buf1.size << std::endl;
    std::cout << "[App]   DMA Buffer 2 (STFT→ISTFT):  phys=0x" << std::hex << dma_buf2.phys_addr
              << ", virt=" << dma_buf2.kern_addr << ", size=" << std::dec << dma_buf2.size << std::endl;
    std::cout << "[App]   DMA Buffer 3 (ISTFT output): phys=0x" << std::hex << dma_buf3.phys_addr
              << ", virt=" << dma_buf3.kern_addr << ", size=" << std::dec << dma_buf3.size << std::endl;
#endif

    // Calculate how many full batches we can process
    size_t total_samples = audio_data.size();
    size_t num_full_batches = total_samples / total_audio_samples;
    size_t remainder_samples = total_samples % total_audio_samples;
    size_t samples_to_process = num_full_batches * total_audio_samples;

#ifdef DEBUG
    std::cout << "[App] Sequential processing of " << total_samples << " samples" << std::endl;
    std::cout << "[App] Batch configuration: " << BATCH_N << " windows per batch, "
              << total_audio_samples << " samples per batch" << std::endl;
    std::cout << "[App] Total batches to process: " << num_full_batches << std::endl;
    if (remainder_samples > 0) {
        std::cout << "[App] Ignoring " << remainder_samples << " remainder samples (not a full batch)" << std::endl;
    }
#endif

    // Collect all processed audio chunks for final output file
    std::vector<int16_t> processed_audio_data;
    processed_audio_data.reserve(samples_to_process);

    stream_open();

    // Process audio in batches of 30 windows
    for (size_t batch_idx = 0; batch_idx < num_full_batches; batch_idx++) {
        size_t batch_start = batch_idx * total_audio_samples;
        size_t batch_end = batch_start + total_audio_samples;
        size_t actual_samples = total_audio_samples;
        size_t actual_bytes = actual_samples * sizeof(int16_t);

	std::cout << "\n[App] Processing batch " << (batch_idx + 1) << "/" << num_full_batches
                  << " (" << BATCH_N << " windows"
                  << ", samples " << batch_start << "-" << batch_end
                  << ", " << actual_bytes << " bytes)" << std::endl;

        // === THREE-STAGE SEQUENTIAL PIPELINE ===

#ifdef DEBUG
        // Step 1: Copy entire batch TO DMA Buffer 1 (STFT input)
        std::cout << "[App] Step 1: Copying entire batch to DMA Buffer 1 (STFT input)..." << std::endl;
#endif
        // Calculate input RMS for verification
        float input_sum = 0.0f;
        for (size_t i = batch_start; i < batch_start + actual_samples; i++) {
            float sample = (float)audio_data[i] / 32768.0f;
            input_sum += sample * sample;
        }
        float input_rms = sqrtf(input_sum / actual_samples);
        //std::cout << "[App] Input batch RMS: " << input_rms << std::endl;

        dmabuf_sync(dma_buf1.dma_buf_fd, DMA_BUF_SYNC_START);
        memcpy(dma_buf1.kern_addr, &audio_data[batch_start], actual_bytes);
	stream_frame(0, dma_buf1.kern_addr, actual_bytes);
#ifdef DEBUG
        // Debug: Show STFT input samples from DMA Buffer 1
        int16_t* input_samples = (int16_t*)dma_buf1.kern_addr;
        std::cout << "[App] DEBUG: STFT INPUT (DMA Buf 1) first 8 samples: ";
        for (int i = 0; i < 8; i++) {
            std::cout << input_samples[i] << " ";
        }
        std::cout << std::endl;
#endif
        dmabuf_sync(dma_buf1.dma_buf_fd, DMA_BUF_SYNC_END);

        // Find stage configurations by message type
        const PipelineStage* stft_stage_ptr = nullptr;
        const PipelineStage* istft_stage_ptr = nullptr;
        for (const auto& stage : state_.pipeline_config.stages) {
            if (stage.message_type == "C7X_MSG_STFT_ANALYZE") stft_stage_ptr = &stage;
            else if (stage.message_type == "C7X_MSG_ISTFT_SYNTHESIZE") istft_stage_ptr = &stage;
        }
        if (!stft_stage_ptr || !istft_stage_ptr) {
            std::cout << "[App] Error: Audio pipeline requires STFT and ISTFT stages" << std::endl;
            dmabuf_heap_destroy(&dma_buf1);
            dmabuf_heap_destroy(&dma_buf2);
            dmabuf_heap_destroy(&dma_buf3);
            return CommandResult::ERROR;
        }
        const auto& stft_stage = *stft_stage_ptr;
        const auto& istft_stage = *istft_stage_ptr;

        // === THREE-STAGE PIPELINE: STFT → TVM → ISTFT ===

        // Step 2: STFT analyze (DMA Buffer 1 → DMA Buffer 2)
        //std::cout << "[App] Step 2: STFT analyze (DMA Buffer 1 → DMA Buffer 2)..." << std::endl;

        // Update STFT parameters to output to DMA Buffer 2
        auto stft_params = stft_stage.parameters;
        char input_addr_str[32], output_addr_str[32];
        snprintf(input_addr_str, sizeof(input_addr_str), "0x%lx", dma_buf1.phys_addr);
        snprintf(output_addr_str, sizeof(output_addr_str), "0x%lx", dma_buf2.phys_addr);
        stft_params["input_buffer"] = input_addr_str;
        stft_params["output_buffer"] = output_addr_str;

        // Process entire batch: 30 windows × 514 elements × 4 bytes = 61680 bytes
        auto stft_result = generic_client_->process(
            stft_stage.message_type,
            nullptr, actual_bytes, nullptr, spectral_data_bytes,
            stft_params
        );

        if (!stft_result.success) {
            std::cout << "[App] Error: STFT analyze failed: " << stft_result.error_message << std::endl;
            return CommandResult::ERROR;
        }
#ifdef DEBUG
        std::cout << "[App] STFT analyze success: input=" << stft_result.input_size
                  << " bytes, output=" << stft_result.output_size << " bytes" << std::endl;

        // Step 3: Verify STFT output in DMA Buffer 2
        std::cout << "[App] Step 3: Verifying STFT output in DMA Buffer 2..." << std::endl;
#endif
        dmabuf_sync(dma_buf2.dma_buf_fd, DMA_BUF_SYNC_START);

#ifdef DEBUG
        // Debug: Read spectral data from STFT output (float values)
        float* spectral_data = (float*)dma_buf2.kern_addr;
        std::cout << "[App] DEBUG: STFT OUTPUT (DMA Buf 2) first 8 spectral values: ";
        for (int i = 0; i < 8; i++) {
            printf("%.6f ", spectral_data[i]);
        }
        std::cout << std::endl;
#endif
        dmabuf_sync(dma_buf2.dma_buf_fd, DMA_BUF_SYNC_END);
#ifdef DEBUG
        std::cout << "[App]   STFT processing complete for batch" << std::endl;

        // Step 4: ISTFT synthesize (no TVM in STFT-only pipeline)
        std::cout << "[App] Step 4: ISTFT synthesize..." << std::endl;
#endif
        // Update ISTFT parameters
        auto istft_params = istft_stage.parameters;
        snprintf(input_addr_str, sizeof(input_addr_str), "0x%lx", dma_buf2.phys_addr);
        snprintf(output_addr_str, sizeof(output_addr_str), "0x%lx", dma_buf3.phys_addr);
        istft_params["input_buffer"] = input_addr_str;
        istft_params["output_buffer"] = output_addr_str;

        // Process entire batch: input 61680 bytes, output 3000 samples (6000 bytes)
        auto istft_result = generic_client_->process(
            istft_stage.message_type,
            nullptr, spectral_data_bytes, nullptr, actual_bytes,
            istft_params
        );

        if (!istft_result.success) {
            std::cout << "[App] Error: ISTFT synthesize failed: " << istft_result.error_message << std::endl;
            return CommandResult::ERROR;
        }
#ifdef DEBUG
        std::cout << "[App] ISTFT synthesize success: input=" << istft_result.input_size
                  << " bytes, output=" << istft_result.output_size << " bytes" << std::endl;

        // Step 5: Verify final output in DMA Buffer 3
        std::cout << "[App] Step 5: Verifying batch output in DMA Buffer 3..." << std::endl;
#endif
        // Invalidate cache to ensure we read fresh data from firmware
        dmabuf_sync(dma_buf3.dma_buf_fd, DMA_BUF_SYNC_START);

        // Copy entire batch output
        std::vector<int16_t> final_output(total_audio_samples);
        memcpy(final_output.data(), dma_buf3.kern_addr, actual_bytes);
        dmabuf_sync(dma_buf3.dma_buf_fd, DMA_BUF_SYNC_END);
	stream_frame(1,  dma_buf3.kern_addr, actual_bytes);

#ifdef DEBUG
        // Per-chunk verification (30 chunks × 100 samples)
        std::cout << "[App] Per-chunk verification:" << std::endl;
        std::cout << "[App]   Chunk | Input RMS | Output RMS | Error RMS | Input Samples (first 5)      | Output Samples (first 5)" << std::endl;
        std::cout << "[App]   ------+-----------+------------+-----------+-------------------------------+-------------------------------" << std::endl;
#endif
        // Latency compensation: DCCRN has 7-chunk delay (700 samples)
        const size_t LATENCY_CHUNKS = 1;
        const size_t LATENCY_SAMPLES = LATENCY_CHUNKS * STFT_INPUT_SAMPLES;  // 700 samples

        float total_error_sum = 0.0f;
        for (size_t chunk = 0; chunk < BATCH_N; chunk++) {
            size_t chunk_offset = chunk * STFT_INPUT_SAMPLES;

            // Calculate input RMS for this chunk
            float chunk_input_sum = 0.0f;
            for (size_t i = 0; i < STFT_INPUT_SAMPLES; i++) {
                float sample = (float)audio_data[batch_start + chunk_offset + i] / 32768.0f;
                chunk_input_sum += sample * sample;
            }
            float chunk_input_rms = sqrtf(chunk_input_sum / STFT_INPUT_SAMPLES);

            // Calculate output RMS for this chunk
            float chunk_output_sum = 0.0f;
            for (size_t i = 0; i < STFT_INPUT_SAMPLES; i++) {
                float sample = (float)final_output[chunk_offset + i] / 32768.0f;
                chunk_output_sum += sample * sample;
            }
            float chunk_output_rms = sqrtf(chunk_output_sum / STFT_INPUT_SAMPLES);

            // Calculate error with latency compensation
            float chunk_error_sum = 0.0f;
            if (chunk + LATENCY_CHUNKS < BATCH_N) {
                for (size_t i = 0; i < STFT_INPUT_SAMPLES; i++) {
                    size_t input_idx = batch_start + chunk_offset + i;
                    size_t output_idx = chunk_offset + i + LATENCY_SAMPLES;
                    int16_t error = abs(audio_data[input_idx] - final_output[output_idx]);
                    chunk_error_sum += (float)(error * error);
                }
            }
            float chunk_error_rms = sqrtf(chunk_error_sum / STFT_INPUT_SAMPLES);

            total_error_sum += chunk_error_sum;
#ifdef DEBUG
            // Print all 30 chunks with RMS
            std::cout << "[App]   " << std::setw(5) << (chunk + 1)
                      << " | " << std::fixed << std::setprecision(4) << std::setw(9) << chunk_input_rms
                      << " | " << std::setw(10) << chunk_output_rms
                      << " | " << std::setw(9) << chunk_error_rms;

            // Print first 5 samples for this chunk (input and output) on the same line
            std::cout << " | In[0-4]: ";
            for (size_t i = 0; i < 5 && i < STFT_INPUT_SAMPLES; i++) {
                std::cout << std::setw(6) << audio_data[batch_start + chunk_offset + i];
                if (i < 4) std::cout << ",";
            }
            std::cout << " | Out[0-4]: ";
            for (size_t i = 0; i < 5 && i < STFT_INPUT_SAMPLES; i++) {
                std::cout << std::setw(6) << final_output[chunk_offset + i];
                if (i < 4) std::cout << ",";
            }
            std::cout << std::endl;
#endif
        }

        float total_error_rms = sqrtf(total_error_sum / (STFT_INPUT_SAMPLES * (BATCH_N - LATENCY_CHUNKS)));
        std::cout << "[App] Overall error RMS (with 700-sample latency compensation): " << total_error_rms << std::endl;

        // Collect processed batch for final output file
        for (size_t i = 0; i < total_audio_samples; i++) {
            processed_audio_data.push_back(final_output[i]);
        }
    }

    std::cout << "\n[App] Batch processing complete:" << std::endl;
    std::cout << "[App]   Processed " << num_full_batches << " batches" << std::endl;
    std::cout << "[App]   Total samples processed: " << processed_audio_data.size() << std::endl;
    std::cout << "[App]   Samples ignored: " << remainder_samples << std::endl;

    // Save processed audio to file for verification
    std::string output_filename = "processed_output.wav";
    //std::cout << "[App] Saving processed audio to: " << output_filename << std::endl;

    if (saveAudioFile(output_filename, processed_audio_data)) {
        std::cout << "[App]  Successfully saved " << processed_audio_data.size() << " samples to " << output_filename << std::endl;
    } else {
        std::cout << "[App]  ERROR: Failed to save processed audio file" << std::endl;
    }

    // Cleanup DMA buffers
    std::cout << "[App] Cleaning up DMA buffers..." << std::endl;
    dmabuf_heap_destroy(&dma_buf1);
    dmabuf_heap_destroy(&dma_buf2);
    dmabuf_heap_destroy(&dma_buf3);
    stream_close();

    return CommandResult::SUCCESS;
}



bool PipelineManager::loadAudioFile(const std::string& filename, std::vector<int16_t>& audio_data)
{
    SF_INFO sfinfo;
    memset(&sfinfo, 0, sizeof(sfinfo));

    SNDFILE* infile = sf_open(filename.c_str(), SFM_READ, &sfinfo);
    if (!infile) {
        std::cout << "[App] Error: Failed to open audio file: " << filename << std::endl;
        return false;
    }

    // Validate audio format
    if (sfinfo.channels != 1) {
        std::cout << "[App] Error: Audio must be mono (1 channel), got " << sfinfo.channels << " channels" << std::endl;
        sf_close(infile);
        return false;
    }

    if (sfinfo.samplerate != 16000) {
        std::cout << "[App] Warning: Audio sample rate is " << sfinfo.samplerate << "Hz, expected 16kHz" << std::endl;
    }

    std::cout << "[App] Audio file info: " << sfinfo.frames << " frames, "
              << sfinfo.samplerate << "Hz, " << sfinfo.channels << " channel(s)" << std::endl;

    // Read all audio data
    audio_data.resize(sfinfo.frames);
    sf_count_t frames_read = sf_readf_short(infile, audio_data.data(), sfinfo.frames);
    sf_close(infile);

    if (frames_read != sfinfo.frames) {
        std::cout << "[App] Warning: Read " << frames_read << " frames, expected " << sfinfo.frames << std::endl;
        audio_data.resize(frames_read);
    }

    std::cout << "[App] Loaded " << audio_data.size() << " audio samples ("
              << (audio_data.size() / (float)sfinfo.samplerate) << " seconds)" << std::endl;

    return true;
}

bool PipelineManager::playAudioData(const std::vector<int16_t>& audio_data)
{
    snd_pcm_t* pcm_handle;
    int rc = snd_pcm_open(&pcm_handle, "default", SND_PCM_STREAM_PLAYBACK, 0);
    if (rc < 0) {
        std::cout << "[App] Error: Cannot open audio device: " << snd_strerror(rc) << std::endl;
        return false;
    }

    // Set audio parameters: 16kHz, 16-bit, mono
    rc = snd_pcm_set_params(pcm_handle,
                           SND_PCM_FORMAT_S16_LE,
                           SND_PCM_ACCESS_RW_INTERLEAVED,
                           1,      // channels (mono)
                           16000,  // sample rate
                           1,      // allow resampling
                           500000); // latency in microseconds

    if (rc < 0) {
        std::cout << "[App] Error: Cannot set audio parameters: " << snd_strerror(rc) << std::endl;
        snd_pcm_close(pcm_handle);
        return false;
    }

    // Play audio data
    std::cout << "[App] Playing processed audio..." << std::endl;
    snd_pcm_sframes_t frames_written = snd_pcm_writei(pcm_handle, audio_data.data(), audio_data.size());

    if (frames_written < 0) {
        std::cout << "[App] Error: Audio playback failed: " << snd_strerror(frames_written) << std::endl;
        snd_pcm_close(pcm_handle);
        return false;
    }

    // Wait for playback to complete
    snd_pcm_drain(pcm_handle);
    snd_pcm_close(pcm_handle);

    std::cout << "[App] Audio playback completed (" << frames_written << " frames)" << std::endl;
    return true;
}

bool PipelineManager::saveAudioFile(const std::string& filename, const std::vector<int16_t>& audio_data)
{
    SF_INFO sfinfo;
    memset(&sfinfo, 0, sizeof(sfinfo));

    // Set output file parameters to match our audio format
    sfinfo.samplerate = 16000;    // 16kHz
    sfinfo.channels = 1;          // mono
    sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;  // WAV file with 16-bit PCM

    SNDFILE* outfile = sf_open(filename.c_str(), SFM_WRITE, &sfinfo);
    if (!outfile) {
        std::cout << "[App] Error: Failed to create output audio file: " << filename << std::endl;
        std::cout << "[App] Error details: " << sf_strerror(nullptr) << std::endl;
        return false;
    }

    std::cout << "[App] Saving audio to: " << filename << std::endl;
    std::cout << "[App] Output file info: " << audio_data.size() << " frames, "
              << sfinfo.samplerate << "Hz, " << sfinfo.channels << " channel(s)" << std::endl;

    // Write audio data to file
    sf_count_t frames_written = sf_writef_short(outfile, audio_data.data(), audio_data.size());
    sf_close(outfile);

    if (frames_written != static_cast<sf_count_t>(audio_data.size())) {
        std::cout << "[App] Warning: Wrote " << frames_written << " frames, expected " << audio_data.size() << std::endl;
        return false;
    }

    std::cout << "[App] Successfully saved " << frames_written << " audio samples to " << filename << std::endl;
    std::cout << "[App] Duration: " << (frames_written / (float)sfinfo.samplerate) << " seconds" << std::endl;

    return true;
}

bool PipelineManager::saveTensorFile(const std::string& filename, const std::vector<float>& tensor_data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "[App] Error: Cannot open file for writing: " << filename << std::endl;
        return false;
    }

    file.write(reinterpret_cast<const char*>(tensor_data.data()), tensor_data.size() * sizeof(float));
    file.close();

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
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Calculate number of float values
    size_t num_floats = size / sizeof(float);
    tensor_data.resize(num_floats);

    // Read binary data
    if (!file.read(reinterpret_cast<char*>(tensor_data.data()), size)) {
        std::cout << "[App] Error: Failed to read BIN file" << std::endl;
        return false;
    }

    return true;
}

