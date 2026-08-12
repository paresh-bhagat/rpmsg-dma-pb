#include "pipeline_manager.h"

#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

namespace {

void signal_handler(int signal_number)
{
    std::_Exit(128 + signal_number);
}

void setup_signal_handlers()
{
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
}

void print_usage(std::string_view program)
{
    std::cout
        << "Usage:\n"
        << "  " << program << "                          Interactive mode\n"
        << "  " << program << " <pipeline.json>          Run a JSON pipeline\n"
        << "  " << program << " <pipeline.json> --debug  Enable per-batch logs\n"
        << "  " << program << " --help                   Show this help\n\n"
        << "Examples:\n"
        << "  " << program << " pipeline_tvm_inference.json\n"
        << "  " << program << " pipeline_audio_enhancement.json --debug\n";
}

} // namespace

int main(int argc, char* argv[])
{
    try {
        std::string json_file;
        bool debug = false;

        for (int index = 1; index < argc; ++index) {
            const std::string_view argument{argv[index]};
            if (argument == "--help" || argument == "-h") {
                print_usage(argv[0]);
                return EXIT_SUCCESS;
            }
            if (argument == "--debug" || argument == "-d") {
                debug = true;
                continue;
            }
            if (argument.rfind("--", 0) == 0) {
                std::cerr << "[App] Unknown argument: " << argument << '\n';
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
            if (!json_file.empty()) {
                std::cerr << "[App] Only one pipeline file may be specified\n";
                return EXIT_FAILURE;
            }
            json_file = argument;
        }

        setup_signal_handlers();
        std::cout << "===========================================\n"
                     "       RPMsg Inference Example\n"
                     "===========================================\n\n";

        PipelineManager application;
        application.set_debug(debug);
        const int exit_code = json_file.empty()
                                  ? application.run()
                                  : application.run_from_json_file(json_file);
        std::cout << "[App] Application exited with code " << exit_code << '\n';
        return exit_code;
    } catch (const std::exception& error) {
        std::cerr << "[App] Fatal error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
