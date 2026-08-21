/*
 * tvm_model_daemon.cpp — Persistent TVM model loader daemon
 *
 * Calls Module::LoadFromFile once at startup, then serves inference requests
 * from rpmsg_inference_example instances over a Unix domain socket.  Each demo
 * run connects, issues one INFER_REQ per chunk, then disconnects.  The model
 * stays resident for the lifetime of this process.
 *
 * Usage:
 *   tvm_model_daemon [--artifacts /path/to/artifacts/dir]
 *
 * Managed by tvm-model-daemon.service (started at boot, before the webserver).
 */

#include "tvm_inference_client.h"
#include "tvm_daemon_proto.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <unistd.h>
#include <csignal>
#include <cstring>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

static constexpr const char* DEFAULT_ARTIFACTS = "/usr/share/tvm_inference/artifacts/";

namespace {

volatile sig_atomic_t g_running  = 1;
int                   g_server_fd = -1;

void signal_handler(int) {
    g_running = 0;
    if (g_server_fd >= 0) { ::close(g_server_fd); g_server_fd = -1; }
}

bool write_all(int fd, const void* buf, size_t len) {
    const auto* p = static_cast<const uint8_t*>(buf);
    while (len > 0) {
        ssize_t n = ::write(fd, p, len);
        if (n <= 0) return false;
        p += static_cast<size_t>(n); len -= static_cast<size_t>(n);
    }
    return true;
}

bool read_all(int fd, void* buf, size_t len) {
    auto* p = static_cast<uint8_t*>(buf);
    while (len > 0) {
        ssize_t n = ::read(fd, p, len);
        if (n <= 0) return false;
        p += static_cast<size_t>(n); len -= static_cast<size_t>(n);
    }
    return true;
}

bool send_hdr(int fd, TvmDaemon::MsgType type, uint32_t payload_len) {
    TvmDaemon::Header h{TvmDaemon::MAGIC, static_cast<uint32_t>(type), payload_len};
    return write_all(fd, &h, sizeof(h));
}

/* Serve a connected client until it disconnects or an error occurs.
 * Handles PING and repeated INFER_REQ messages on a single connection. */
void handle_client(int cfd, TvmInferenceClient& tvm) {
    TvmDaemon::Header hdr{};
    while (true) {
        if (!read_all(cfd, &hdr, sizeof(hdr))) break;
        if (hdr.magic != TvmDaemon::MAGIC) {
            std::cerr << "[daemon] Bad magic 0x" << std::hex << hdr.magic
                      << std::dec << " — closing\n";
            break;
        }

        if (hdr.type == static_cast<uint32_t>(TvmDaemon::MsgType::PING)) {
            send_hdr(cfd, TvmDaemon::MsgType::PONG, 0);
            continue;
        }

        if (hdr.type != static_cast<uint32_t>(TvmDaemon::MsgType::INFER_REQ) ||
            hdr.len == 0 || hdr.len % sizeof(float) != 0) {
            const std::string err = "unexpected message type or bad payload size";
            send_hdr(cfd, TvmDaemon::MsgType::ERROR_RESP,
                     static_cast<uint32_t>(err.size()));
            write_all(cfd, err.data(), err.size());
            break;
        }

        const size_t n_floats = hdr.len / sizeof(float);
        std::vector<float> input(n_floats), output;

        if (!read_all(cfd, input.data(), hdr.len)) {
            std::cerr << "[daemon] Failed to read input payload\n";
            break;
        }

        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = tvm.run_inference(input, output);
        const double ms = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count() / 1000.0;

        if (!ok) {
            const std::string err = "inference failed";
            send_hdr(cfd, TvmDaemon::MsgType::ERROR_RESP,
                     static_cast<uint32_t>(err.size()));
            write_all(cfd, err.data(), err.size());
            continue;
        }

        std::cout << "[daemon] Inference " << ms << " ms ("
                  << output.size() << " floats out)\n";

        const uint32_t out_bytes = static_cast<uint32_t>(output.size() * sizeof(float));
        if (!send_hdr(cfd, TvmDaemon::MsgType::INFER_RESP, out_bytes) ||
            !write_all(cfd, output.data(), out_bytes)) {
            std::cerr << "[daemon] Failed to send response\n";
            break;
        }
    }
    ::close(cfd);
}

} // namespace

int main(int argc, char* argv[]) {
    std::string artifacts = DEFAULT_ARTIFACTS;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--artifacts" && i + 1 < argc)
            artifacts = argv[++i];
    }

    std::cout << "[daemon] Loading TVM artifacts from: " << artifacts << '\n';

    TvmInferenceClient tvm;
    tvm.disable_daemon();               /* must not try to connect to itself */
    tvm.set_input_shape({1, 2, 401, 161});

    if (!tvm.initialize(artifacts)) {
        std::cerr << "[daemon] Failed to load model — aborting\n";
        return 1;
    }
    std::cout << "[daemon] Model ready. Listening on "
              << TvmDaemon::SOCKET_PATH << '\n';

    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);
    std::signal(SIGPIPE, SIG_IGN);   /* suppress broken-pipe crashes */

    ::unlink(TvmDaemon::SOCKET_PATH);
    g_server_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (g_server_fd < 0) { perror("socket"); return 1; }

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, TvmDaemon::SOCKET_PATH, sizeof(addr.sun_path) - 1);

    if (::bind(g_server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        perror("bind"); return 1;
    }
    ::chmod(TvmDaemon::SOCKET_PATH, 0660);

    if (::listen(g_server_fd, 4) < 0) { perror("listen"); return 1; }
    std::cout << "[daemon] Ready\n";

    while (g_running) {
        int cfd = ::accept(g_server_fd, nullptr, nullptr);
        if (cfd < 0) {
            if (g_running) perror("accept");
            break;
        }
        std::cout << "[daemon] Client connected\n";
        handle_client(cfd, tvm);
        std::cout << "[daemon] Client done\n";
    }

    ::unlink(TvmDaemon::SOCKET_PATH);
    std::cout << "[daemon] Shut down\n";
    return 0;
}
