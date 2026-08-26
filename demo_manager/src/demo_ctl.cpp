// demo-ctl - CLI client for the demo-manager daemon
//
// Usage:
//   demo-ctl list
//   demo-ctl status
//   demo-ctl run <demo-name> [extra args...]
//   demo-ctl stop
//   demo-ctl preload
//   demo-ctl quit

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <json-c/json.h>

static constexpr const char *SOCKET_PATH = "/var/run/demo-manager.sock";

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s <command> [args]\n\n"
        "Commands:\n"
        "  list                      List available demos\n"
        "  status                    Show currently running demo\n"
        "  run <demo> [args...]      Start a demo (args forwarded to executable)\n"
        "  stop                      Stop the running demo\n"
        "  preload                   Preload the TVM model for edge-ai\n"
        "  quit                      Shut down the demo-manager daemon\n\n"
        "Examples:\n"
        "  %s run edge-ai\n"
        "  %s run edge-ai pipeline_audio_enhancement.json\n"
        "  %s run edge-ai pipeline_tvm_inference.json --debug\n"
        "  %s run 2dfft\n"
        "  %s stop\n",
        prog, prog, prog, prog, prog, prog);
}

static int send_command(const std::string &json_str) {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); return 1; }

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);

    if (connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
        fprintf(stderr, "Cannot connect to demo-manager (%s): %s\n"
                        "Is the daemon running? Try: systemctl start demo-manager\n",
                SOCKET_PATH, strerror(errno));
        close(fd);
        return 1;
    }

    write(fd, json_str.c_str(), json_str.size());

    // Accumulate response (daemon sends one JSON line)
    std::string buf;
    char chunk[1024];
    ssize_t n;
    while ((n = read(fd, chunk, sizeof(chunk))) > 0)
        buf.append(chunk, static_cast<size_t>(n));
    close(fd);

    if (buf.empty()) { fprintf(stderr, "no response from daemon\n"); return 1; }

    // Pretty-print and check status
    json_object *resp = json_tokener_parse(buf.c_str());
    if (!resp) { printf("%s\n", buf.c_str()); return 0; }

    printf("%s\n", json_object_to_json_string_ext(resp,
           JSON_C_TO_STRING_PRETTY | JSON_C_TO_STRING_SPACED));

    int rc = 0;
    json_object *status_obj = nullptr;
    if (json_object_object_get_ex(resp, "status", &status_obj) &&
        strcmp(json_object_get_string(status_obj), "error") == 0)
        rc = 1;

    json_object_put(resp);
    return rc;
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(argv[0]); return 1; }

    std::string cmd = argv[1];
    json_object *req = json_object_new_object();

    if (cmd == "list" || cmd == "status" || cmd == "stop" ||
        cmd == "preload" || cmd == "quit") {
        json_object_object_add(req, "cmd", json_object_new_string(cmd.c_str()));

    } else if (cmd == "run") {
        if (argc < 3) {
            fprintf(stderr, "'run' requires a demo name\nAvailable: edge-ai, 2dfft, audio-offload, sigchain-biquad\n");
            json_object_put(req);
            return 1;
        }
        json_object_object_add(req, "cmd",  json_object_new_string("run"));
        json_object_object_add(req, "demo", json_object_new_string(argv[2]));
        if (argc > 3) {
            json_object *arr = json_object_new_array();
            for (int i = 3; i < argc; i++)
                json_object_array_add(arr, json_object_new_string(argv[i]));
            json_object_object_add(req, "args", arr);
        }
    } else {
        fprintf(stderr, "Unknown command: %s\n\n", cmd.c_str());
        usage(argv[0]);
        json_object_put(req);
        return 1;
    }

    std::string json_str = json_object_to_json_string(req);
    json_object_put(req);
    return send_command(json_str);
}
