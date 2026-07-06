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

    // Switch back to base firmware on exit
    printf("[App] Switching back to base firmware...\n");
    if (switch_firmware((char*)C7_BASE_FW, (char*)C7_FW_LINK, (char*)C7_FW_STATE) != 0) {
        printf("[App] Warning: Failed to switch back to base firmware\n");
    } else {
        printf("[App] Base firmware restored\n");
    }

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
 * @brief Initialize firmware for TVM operation
 */
bool initialize_firmware()
{
    printf("[App] Initializing TVM firmware...\n");

    // Switch to TVM firmware
    if (switch_firmware((char*)C7_TVM_FW, (char*)C7_FW_LINK, (char*)C7_FW_STATE) != 0) {
        printf("[App] Error: Failed to switch to TVM firmware\n");
        return false;
    }

    printf("[App] TVM firmware loaded successfully\n");
    return true;
}

/**
 * @brief Main application entry point
 */
int main(int argc, char *argv[])
{
    printf("===========================================\n");
    printf("       RPMsg Inference Example\n");
    printf("===========================================\n\n");

    // Setup signal handlers for graceful shutdown
    setup_signal_handlers();

    // Initialize firmware
    if (!initialize_firmware()) {
        printf("[App] Failed to initialize firmware\n");
        return 1;
    }

    // Create and run pipeline manager
    PipelineManager app;
    g_app = &app;

    int exit_code = app.run();

    // Cleanup: switch back to base firmware
    printf("[App] Switching back to base firmware...\n");
    if (switch_firmware((char*)C7_BASE_FW, (char*)C7_FW_LINK, (char*)C7_FW_STATE) != 0) {
        printf("[App] Warning: Failed to switch back to base firmware\n");
    } else {
        printf("[App] Base firmware restored\n");
    }

    printf("[App] Application exited with code %d\n", exit_code);
    return exit_code;
}