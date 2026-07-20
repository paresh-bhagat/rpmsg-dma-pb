#ifndef PIPELINE_MANAGER_H
#define PIPELINE_MANAGER_H

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <memory>
#include "tvm_inference_client.h"
#include "generic_task_client.h"

extern "C" {
#include <json-c/json.h>
#include "dmabuf.h"
}

class PipelineManager {
public:
    enum class CommandResult {
        SUCCESS,
        ERROR,
        QUIT
    };

    // JSON Pipeline Stage
    struct PipelineStage {
        std::string stage_id;
        std::string service;         // "generic" or "tvm"
        std::string message_type;    // "C7X_MSG_*" or "TVM_INFERENCE"
        std::map<std::string, std::string> parameters; // Stage-specific parameters
    };


    struct PipelineConfig {
        std::string pipeline_id;
        std::string description;
        std::string input_file;
        std::string artifacts_path;
        std::vector<PipelineStage> stages;
        bool loaded;

        PipelineConfig() : loaded(false) {}
    };

    enum class InputType {
        UNKNOWN,
        AUDIO_WAV,
        TENSOR_BIN
    };

    enum class PipelineMode {
        STFT_ISTFT,   // --ISTFT : STFT -> ISTFT only
        TVM_ONLY,     // --TVM   : TVM inference only (BIN input)
        FULL          // --full  : STFT -> TVM -> ISTFT
    };

    struct State {
        bool artifacts_loaded;
        PipelineConfig pipeline_config;
        std::string current_pipeline_file;
        std::vector<std::string> tvm_artifacts_paths;
        bool tvm_artifacts_configured;
        std::string current_input_file;
        InputType input_type;
        bool input_configured;

        void* tvm_staging_buffer;
        void* tvm_result_buffer;
        size_t staging_buffer_size;
        size_t result_buffer_size;

        State() : input_type(InputType::UNKNOWN) {}
    };

    PipelineManager();
    ~PipelineManager();

    bool initialize();
    int run();
    int run_direct(PipelineMode mode, const std::string& input_file, const std::string& artifacts_path = "");
    int run_from_json_file(const std::string& json_file_path);

private:
    std::shared_ptr<TvmInferenceClient> tvm_client_;
    std::unique_ptr<GenericTaskClient> generic_client_;
    State state_;
    bool initialized_;
    std::string app_name_;

    struct CommandInfo {
        std::string description;
        std::vector<std::string> examples;
        std::function<CommandResult(const std::vector<std::string>&)> handler;
    };

    std::map<std::string, CommandInfo> commands_;

    void initializeCommands();
    void printWelcome();
    void printPrompt();
    std::vector<std::string> parseCommand(const std::string& input);
    CommandResult executeCommand(const std::vector<std::string>& tokens);

    CommandResult handleHelp(const std::vector<std::string>& args);
    CommandResult handlePipeline(const std::vector<std::string>& args);
    CommandResult handleTvmArtifacts(const std::vector<std::string>& args);
    CommandResult handleInput(const std::vector<std::string>& args);
    CommandResult handleShowPipeline(const std::vector<std::string>& args);
    CommandResult handleRun(const std::vector<std::string>& args);
    CommandResult handleStatus(const std::vector<std::string>& args);
    CommandResult handleQuit(const std::vector<std::string>& args);

    bool validateConfiguration();
    bool loadPipelineFromJson(const std::string& json_content);

    CommandResult executeAudioPipeline();
    CommandResult executeTensorPipeline();
    CommandResult executeSequentialPipeline();

    bool loadAudioFile(const std::string& filename, std::vector<int16_t>& audio_data);
    bool loadBinTensor(const std::string& filename, std::vector<float>& tensor_data);
    bool saveTensorFile(const std::string& filename, const std::vector<float>& tensor_data);
    bool playAudioData(const std::vector<int16_t>& audio_data);
    bool saveAudioFile(const std::string& filename, const std::vector<int16_t>& audio_data);
    std::string getCurrentPrompt();
    void registerCommand(const std::string& name, const std::string& description,
                        const std::vector<std::string>& examples,
                        std::function<CommandResult(const std::vector<std::string>&)> handler);
};

#endif // PIPELINE_MANAGER_H