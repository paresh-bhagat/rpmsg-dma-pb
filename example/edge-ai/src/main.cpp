#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <cstring>

#include "pipeline_manager.h"

// Firmware switching support
extern "C" {
#include "fw_loader.h"
}

/** @brief Path to the base firmware for the C7x processor. */
#define C7_BASE_FW      "/lib/firmware/ti-ipc/am62dxx/ipc_echo_test_c7x_1_release_strip.xe71"

/** @brief Path to the TVM firmware for the C7x processor. */
#define C7_TVM_FW       "/lib/firmware/c7x_compute.release.out"

/** @brief Symbolic link to the firmware for the C7x processor. */
#define C7_FW_LINK      "/lib/firmware/am62d-c71_0-fw"

/** @brief Path to the state file for the remote processor. */
#define C7_FW_STATE     "/sys/class/remoteproc/remoteproc0/state"

// Global application instance for signal handling
static PipelineManager* g_app = nullptr;

/**
 * @brief Signal handler for graceful shutdown
 */
void signal_handler(int signum)
{
    printf("\n[App] Received signal %d, shutting down...\n", signum);

    exit(signum);
}

/**
 * @brief Setup signal handlers for graceful shutdown
 */
void setup_signal_handlers()
{
    signal(SIGINT, signal_handler);   // Ctrl+C
    signal(SIGTERM, signal_handler);  // Termination request
}

/**
 * @brief Print command-line usage
 */
void print_usage(const char* prog_name)
{
    printf("Usage:\n");
    printf("  %s                          Interactive mode (readline shell)\n", prog_name);
    printf("  %s <pipeline.json>          Run pipeline from JSON configuration file\n", prog_name);
    printf("  %s <pipeline.json> --debug  Run with verbose per-batch debug logs\n", prog_name);
    printf("  %s --help                   Show this help\n", prog_name);
    printf("\nJSON configuration file fields:\n");
    printf("  pipeline_id     Pipeline identifier string\n");
    printf("  description     Human-readable description\n");
    printf("  input_file      Path to input BIN file\n");
    printf("  output_file     Output data size\n");
    printf("  artifacts_path  Path to TVM artifacts directory (optional, for TVM stages)\n");
    printf("  stages          Array of pipeline stage objects\n");
    printf("\nExample:\n");
    printf("  %s pipeline_stft_istft.json\n", prog_name);
    printf("  %s pipeline_tvm_inference.json\n", prog_name);
    printf("  %s pipeline_audio_enhancement.json\n", prog_name);
}

/**
 * @brief Main application entry point
 */
int main(int argc, char *argv[])
{
    printf("===========================================\n");
    printf("       RPMsg Inference Example\n");
    printf("===========================================\n\n");

    std::string json_file;
    bool debug = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--debug" || arg == "-d") {
            debug = true;
        } else if (arg.rfind("--", 0) == 0) {
            printf("[App] Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return -1;
        } else {
            json_file = arg;
        }
    }

    // Setup signal handlers for graceful shutdown
    setup_signal_handlers();

    // Create and run pipeline manager
    PipelineManager app;
    g_app = &app;
    app.set_debug(debug);

    int exit_code;
    if (!json_file.empty()) {
        exit_code = app.run_from_json_file(json_file);
    } else {
        exit_code = app.run();
    }

    printf("[App] Application exited with code %d\n", exit_code);
    return exit_code;
}