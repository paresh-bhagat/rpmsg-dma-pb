#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <cstring>
#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/time.h>

#include "tvm_inference_client.h"

// Firmware switching support
extern "C" {
#include "fw_loader.h"
}

/** @brief Path to the base firmware for the C7x processor. */
#define C7_BASE_FW		"/lib/firmware/ti-ipc/am62dxx/ipc_echo_test_c7x_1_release_strip.xe71"

/** @brief Path to the TVM firmware for the C7x processor. */
#define C7_TVM_FW		"/lib/firmware/c7x_compute.release.out"

/** @brief Symbolic link to the firmware for the C7x processor. */
#define C7_FW_LINK		"/lib/firmware/am62d-c71_0-fw"

/** @brief Path to the state file for the remote processor. */
#define C7_FW_STATE		"/sys/class/remoteproc/remoteproc0/state"

// RPMsg and Generic Service Support
#define RPMSG_CREATE_EPT_IOCTL  _IOW(0xb5, 0x1, struct rpmsg_endpoint_info)

struct rpmsg_endpoint_info {
    char name[32];
    unsigned int src;
    unsigned int dst;
};

// Generic service protocol (correct message types)
#define C7X_MSG_GENERIC_PING      0x1001
#define C7X_MSG_GENERIC_PING_RESP 0x2001
#define C7X_MSG_GENERIC_ECHO      0x1002
#define C7X_MSG_GENERIC_ECHO_RESP 0x2002
#define C7X_MSG_GENERIC_STATUS    0x1003
#define C7X_MSG_GENERIC_STATUS_RESP 0x2003

struct c7x_msg_hdr {
    unsigned int type;              /* Message type (C7X_MSG_*) */
    unsigned int seq;               /* Sequence number for correlation */
    unsigned int len;               /* Total message length including header */
    int status;                     /* Response status (0 = success) */
} __attribute__((packed));

struct generic_ping_msg {
    struct c7x_msg_hdr hdr;
    unsigned int timestamp;
    char message[64];
} __attribute__((packed));

struct generic_status_msg {
    struct c7x_msg_hdr hdr;
    unsigned int uptime_ticks;
    unsigned int messages_processed;
    unsigned int current_time;
    unsigned int service_status;
} __attribute__((packed));

volatile bool g_running = true;

/**
 * @brief Cleans up resources and switches firmware back to the base version.
 *
 * This function is called during program termination to reset the firmware
 * to its base state and perform any necessary cleanup.
 */
static void cleanup()
{
    printf("Switching firmware back to base firmware...\n");
    switch_firmware((char*)C7_BASE_FW, (char*)C7_FW_LINK, (char*)C7_FW_STATE);
}

static void signal_handler(int sig) {
    (void)sig;
    printf("\nReceived signal, shutting down...\n");
    g_running = false;

    // Perform cleanup
    cleanup();

    // Force exit after second signal
    static int signal_count = 0;
    signal_count++;
    if (signal_count >= 2) {
        printf("Force exit on second signal\n");
        exit(1);
    }

    exit(0);
}

void run_tvm_inference(const char* artifacts_path) {
    printf("\nTVM Inference Client\n");

    TvmInferenceClient client;

    // Initialize with artifacts
    printf("\nInitializing TVM Client...\n");
    if (!client.initialize(artifacts_path)) {
        printf("Failed to initialize TVM client\n");
        return;
    }

    // Run inference benchmark (same as Python script)
    printf("\nMobileNet v2 Inference Benchmark\n");
    if (client.run_inference_benchmark(10)) {
        printf("TVM inference benchmark completed\n");

        // Print top-5 results like Python script
        client.print_top5_results();
    } else {
        printf("TVM inference benchmark failed\n");
    }

    printf("TVM inference testing completed\n");
}

int test_generic_service(const char* ping_message) {
    printf("\nGeneric Service Test\n");
    printf("Testing Generic Service (endpoint 13)...\n");

    // Open rpmsg device for endpoint 13 (we know it's rpmsg3)
    int rpmsg_fd = open("/dev/rpmsg3", O_RDWR);
    if (rpmsg_fd < 0) {
        perror("Failed to open /dev/rpmsg3");
        printf("Make sure DSP firmware is loaded with dual-task support\n");
        return -1;
    }

    printf("[GENERIC] Connected to Generic Service\n");

    // Prepare ping message
    struct generic_ping_msg ping_req = {0};
    ping_req.hdr.type = C7X_MSG_GENERIC_PING;
    ping_req.hdr.seq = 1;
    ping_req.hdr.len = sizeof(ping_req);
    ping_req.timestamp = 0;  // Will be set by DSP
    strncpy(ping_req.message, ping_message, sizeof(ping_req.message) - 1);

    printf("[GENERIC] Sending PING: \"%s\"\n", ping_req.message);

    // Send ping request
    ssize_t sent = write(rpmsg_fd, &ping_req, sizeof(ping_req));
    if (sent < 0) {
        perror("Failed to send ping");
        close(rpmsg_fd);
        return -1;
    }


    // Wait for response with timeout
    fd_set readfds;
    struct timeval timeout;
    FD_ZERO(&readfds);
    FD_SET(rpmsg_fd, &readfds);
    timeout.tv_sec = 5;  // 5 second timeout
    timeout.tv_usec = 0;

    int ready = select(rpmsg_fd + 1, &readfds, NULL, NULL, &timeout);
    if (ready <= 0) {
        printf("[GENERIC] Timeout waiting for response\n");
        printf("Check DSP trace: cat /sys/kernel/debug/remoteproc/remoteproc0/trace0\n");
        close(rpmsg_fd);
        return -1;
    }

    // Read response
    struct generic_ping_msg ping_resp;
    ssize_t received = read(rpmsg_fd, &ping_resp, sizeof(ping_resp));
    if (received < 0) {
        perror("Failed to read ping response");
        close(rpmsg_fd);
        return -1;
    }

    printf("[GENERIC] Response received (%zd bytes)\n", received);
    printf("          Echo Message: \"%s\"\n", ping_resp.message);

    close(rpmsg_fd);
    return 0;
}

int get_generic_status() {
    printf("\nGeneric Service Status\n");

    int rpmsg_fd = open("/dev/rpmsg3", O_RDWR);
    if (rpmsg_fd < 0) {
        perror("Failed to open /dev/rpmsg3");
        return -1;
    }

    // Prepare status request
    struct c7x_msg_hdr status_req = {0};
    status_req.type = C7X_MSG_GENERIC_STATUS;
    status_req.seq = 2;
    status_req.len = sizeof(status_req);

    printf("[GENERIC] Requesting service status...\n");

    ssize_t sent = write(rpmsg_fd, &status_req, sizeof(status_req));
    if (sent < 0) {
        perror("Failed to send status request");
        close(rpmsg_fd);
        return -1;
    }

    // Wait for response
    fd_set readfds;
    struct timeval timeout;
    FD_ZERO(&readfds);
    FD_SET(rpmsg_fd, &readfds);
    timeout.tv_sec = 3;
    timeout.tv_usec = 0;

    int ready = select(rpmsg_fd + 1, &readfds, NULL, NULL, &timeout);
    if (ready <= 0) {
        printf("[GENERIC] Timeout waiting for status response\n");
        close(rpmsg_fd);
        return -1;
    }

    // Read response
    struct generic_status_msg status_resp;
    ssize_t received = read(rpmsg_fd, &status_resp, sizeof(status_resp));
    if (received < 0) {
        perror("Failed to read status response");
        close(rpmsg_fd);
        return -1;
    }

    printf("[GENERIC] Service Status:\n");
    printf("          Uptime: %u ticks\n", status_resp.uptime_ticks);
    printf("          Messages Processed: %u\n", status_resp.messages_processed);
    printf("          Current Time: %u ticks\n", status_resp.current_time);
    printf("          Service Status: 0x%04x\n", status_resp.service_status);

    close(rpmsg_fd);
    return 0;
}

void print_usage(const char* program_name) {
    printf("Usage: %s [OPTIONS]\n", program_name);
    printf("\nOptions:\n");
    printf("  -h, --help            Show this help message\n");
    printf("  -a, --artifacts PATH  Path to TVM artifacts (default: /root/artifacts)\n");
    printf("  -g, --generic         Test Generic Service instead of TVM\n");
    printf("  -p, --ping MESSAGE    Send PING to Generic Service\n");
    printf("  -s, --status          Get status from Generic Service\n");
    printf("\nExamples:\n");
    printf("  %s                           # Run TVM inference (default)\n", program_name);
    printf("  %s -a artifacts/             # TVM inference with custom path\n", program_name);
    printf("  %s -g -p \"Hello DSP!\"        # Ping Generic Service\n", program_name);
    printf("  %s -g -s                     # Get Generic Service status\n", program_name);
}

int main(int argc, char *argv[]) {
    printf("Dual-Task Client\n");
    printf("========================\n");
    printf("TVM Compute Service (endpoint 20) + Generic Service (endpoint 13)\n\n");

    // Setup signal handler
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Switch to TVM firmware at startup
    printf("Loading TVM firmware for C7x processor...\n");
    if (switch_firmware((char*)C7_TVM_FW, (char*)C7_FW_LINK, (char*)C7_FW_STATE) < 0) {
        fprintf(stderr, "Failed to load TVM firmware\n");
        return -1;
    }
    printf("TVM firmware loaded successfully\n");
    sleep(1);  // Give firmware time to initialize

    // Parse command line arguments
    const char *artifacts_path = "/root/artifacts";
    bool test_generic = false;
    const char *ping_message = nullptr;
    bool get_status = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--artifacts") == 0) {
            if (i + 1 < argc) {
                artifacts_path = argv[++i];
            } else {
                printf("Error: -a/--artifacts requires a path\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-g") == 0 || strcmp(argv[i], "--generic") == 0) {
            test_generic = true;
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--ping") == 0) {
            if (i + 1 < argc) {
                ping_message = argv[++i];
                test_generic = true;  // Auto-enable generic service
            } else {
                printf("Error: -p/--ping requires a message\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--status") == 0) {
            get_status = true;
            test_generic = true;  // Auto-enable generic service
        } else {
            printf("Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (test_generic) {
        // Test Generic Service
        printf("Configuration:\n");
        printf("- Service: Generic Service (endpoint 13)\n");
        if (ping_message) printf("- PING message: %s\n", ping_message);
        if (get_status) printf("- Get status: Yes\n");
        printf("\n");

        bool success = true;

        if (ping_message) {
            if (test_generic_service(ping_message) != 0) {
                success = false;
            }
        }

        if (get_status) {
            if (get_generic_status() != 0) {
                success = false;
            }
        }

        if (!ping_message && !get_status) {
            // Default generic test
            if (test_generic_service("Hello from Linux!") != 0) {
                success = false;
            }
        }

        if (success) {
            printf("\nGeneric Service test completed successfully!\n");
        } else {
            printf("\nGeneric Service test failed!\n");
            return 1;
        }

    } else {
        // Test TVM Compute Service (default)
        printf("Configuration:\n");
        printf("- Service: TVM Compute Service (endpoint 20)\n");
        printf("- Artifacts path: %s\n", artifacts_path);
        printf("\n");

        // Run TVM inference (replicate inference.py)
        run_tvm_inference(artifacts_path);
    }

    // Cleanup before exit
    cleanup();

    return 0;
}
