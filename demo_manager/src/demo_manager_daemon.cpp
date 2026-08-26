// Demo Manager Daemon
// Listens on a Unix domain socket for JSON commands to run/stop demos.
// Automatically preloads the TVM model before edge-ai demos and invalidates
// the TVM model cache after any DSP compute demo completes or is stopped
// (since DSP compute demos load their own firmware, overwriting the TVM state).
//
// Socket: /var/run/demo-manager.sock
// Protocol: newline-terminated JSON request/response
//
// Commands:
//   {"cmd":"list"}                             - list available demos
//   {"cmd":"status"}                           - show running demo
//   {"cmd":"run","demo":"<name>"}              - start a demo
//   {"cmd":"run","demo":"<name>","args":[...]} - start with extra args
//   {"cmd":"stop"}                             - stop running demo
//   {"cmd":"preload"}                          - preload TVM model
//   {"cmd":"quit"}                             - shutdown daemon

#include <cerrno>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <syslog.h>
#include <unistd.h>

#include <json-c/json.h>

static constexpr const char *SOCKET_PATH = "/var/run/demo-manager.sock";
static constexpr const char *PID_FILE    = "/var/run/demo-manager.pid";
static constexpr const char *TVM_CACHE   = "/var/lib/tvm_inference/loaded_model";
static constexpr const char *PRELOAD_EXE = "/usr/bin/rpmsg_inference_example";

struct DemoEntry {
    const char *name;
    const char *description;
    const char *executable;
    bool        is_edge_ai;     // auto-preload TVM before run
    bool        is_dsp_compute; // invalidate TVM cache after run
};

static constexpr DemoEntry DEMOS[] = {
    {
        "edge-ai",
        "TVM ML inference (GCRN speech enhancement) with C7x DSP offload",
        "/usr/bin/rpmsg_inference_example",
        true, false
    },
    {
        "2dfft",
        "2D FFT computation offloaded to C7x DSP",
        "/usr/bin/rpmsg_2dfft_example",
        false, true
    },
    {
        "audio-offload",
        "FFT-based audio processing with C7x DSP offload",
        "/usr/bin/rpmsg_audio_offload_example",
        false, true
    },
    {
        "sigchain-biquad",
        "3-stage parametric equalizer biquad cascade on C7x DSP",
        "/usr/bin/rpmsg_sigchain_biquad_example",
        false, true
    },
};
static constexpr int N_DEMOS = static_cast<int>(sizeof(DEMOS) / sizeof(DEMOS[0]));

// Global state - accessed from signal handlers so volatile sig_atomic_t
static volatile sig_atomic_t g_quit       = 0;
static volatile sig_atomic_t g_child_done = 0;
static pid_t  g_child_pid          = -1;
static bool   g_invalidate_on_exit = false;
static char   g_active_demo[64]    = {};

static void on_signal(int sig) {
    if (sig == SIGTERM || sig == SIGINT) g_quit = 1;
}
static void on_sigchld(int) { g_child_done = 1; }

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool file_exists(const char *path) { return access(path, F_OK) == 0; }

static void invalidate_tvm_cache() {
    if (!file_exists(TVM_CACHE)) return;
    if (unlink(TVM_CACHE) == 0)
        syslog(LOG_INFO, "TVM model cache invalidated (%s)", TVM_CACHE);
    else
        syslog(LOG_WARNING, "Failed to remove TVM cache %s: %s", TVM_CACHE, strerror(errno));
}

// Build a null-terminated argv array from a vector and exec, returning exit code.
static int run_sync(const std::vector<std::string> &args) {
    std::vector<const char *> av;
    av.reserve(args.size() + 1);
    for (auto &s : args) av.push_back(s.c_str());
    av.push_back(nullptr);

    pid_t pid = fork();
    if (pid < 0) {
        syslog(LOG_ERR, "fork: %s", strerror(errno));
        return -1;
    }
    if (pid == 0) {
        execv(av[0], const_cast<char *const *>(av.data()));
        _exit(127);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

// Fork and exec without waiting - returns child pid, or -1 on error.
static pid_t launch_async(const std::vector<std::string> &args) {
    std::vector<const char *> av;
    av.reserve(args.size() + 1);
    for (auto &s : args) av.push_back(s.c_str());
    av.push_back(nullptr);

    pid_t pid = fork();
    if (pid < 0) { syslog(LOG_ERR, "fork: %s", strerror(errno)); return -1; }
    if (pid == 0) {
        execv(av[0], const_cast<char *const *>(av.data()));
        _exit(127);
    }
    return pid;
}

// Called from main loop whenever g_child_done is set.
static void reap_child() {
    g_child_done = 0;
    if (g_child_pid < 0) return;

    int status = 0;
    pid_t r = waitpid(g_child_pid, &status, WNOHANG);
    if (r != g_child_pid) return;

    int code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    syslog(LOG_INFO, "Demo '%s' pid=%d exited with code %d",
           g_active_demo, (int)r, code);

    if (g_invalidate_on_exit) invalidate_tvm_cache();

    g_child_pid = -1;
    g_active_demo[0]    = '\0';
    g_invalidate_on_exit = false;
}

// Stop the active child cleanly (SIGTERM, then SIGKILL after 5s).
static void stop_child() {
    if (g_child_pid < 0) return;

    syslog(LOG_INFO, "Stopping demo '%s' pid=%d", g_active_demo, (int)g_child_pid);
    kill(g_child_pid, SIGTERM);

    for (int i = 0; i < 50; i++) {
        usleep(100000); // 100 ms per tick
        int status = 0;
        if (waitpid(g_child_pid, &status, WNOHANG) == g_child_pid) goto done;
    }
    kill(g_child_pid, SIGKILL);
    waitpid(g_child_pid, nullptr, 0);

done:
    if (g_invalidate_on_exit) invalidate_tvm_cache();
    g_child_pid = -1;
    g_active_demo[0]    = '\0';
    g_invalidate_on_exit = false;
}

// ---------------------------------------------------------------------------
// JSON response builders
// ---------------------------------------------------------------------------

static std::string resp_ok(json_object *data = nullptr) {
    json_object *o = json_object_new_object();
    json_object_object_add(o, "status", json_object_new_string("ok"));
    if (data) json_object_object_add(o, "data", data);
    std::string s = json_object_to_json_string(o);
    json_object_put(o);
    return s + "\n";
}

static std::string resp_err(const std::string &msg) {
    json_object *o = json_object_new_object();
    json_object_object_add(o, "status",  json_object_new_string("error"));
    json_object_object_add(o, "message", json_object_new_string(msg.c_str()));
    std::string s = json_object_to_json_string(o);
    json_object_put(o);
    return s + "\n";
}

// ---------------------------------------------------------------------------
// Command handlers
// ---------------------------------------------------------------------------

static std::string handle_list() {
    json_object *arr = json_object_new_array();
    for (int i = 0; i < N_DEMOS; i++) {
        json_object *d = json_object_new_object();
        json_object_object_add(d, "name",        json_object_new_string(DEMOS[i].name));
        json_object_object_add(d, "description", json_object_new_string(DEMOS[i].description));
        json_object_object_add(d, "type",
            json_object_new_string(DEMOS[i].is_edge_ai ? "edge-ai" : "dsp-compute"));
        json_object_array_add(arr, d);
    }
    return resp_ok(arr);
}

static std::string handle_status() {
    json_object *d = json_object_new_object();
    if (g_child_pid > 0) {
        json_object_object_add(d, "running", json_object_new_string(g_active_demo));
        json_object_object_add(d, "pid",     json_object_new_int(static_cast<int32_t>(g_child_pid)));
    } else {
        json_object_object_add(d, "running", nullptr);
        json_object_object_add(d, "pid",     json_object_new_int(-1));
    }
    return resp_ok(d);
}

static std::string handle_stop() {
    if (g_child_pid < 0) return resp_err("no demo is running");
    stop_child();
    return resp_ok();
}

static std::string handle_preload() {
    if (g_child_pid > 0)
        return resp_err(std::string("stop '") + g_active_demo + "' before preloading");

    syslog(LOG_INFO, "Running TVM model preload");
    int rc = run_sync({PRELOAD_EXE, "--preload"});
    if (rc != 0)
        return resp_err("preload failed with exit code " + std::to_string(rc));

    syslog(LOG_INFO, "TVM model preload completed");
    json_object *d = json_object_new_object();
    json_object_object_add(d, "message", json_object_new_string("TVM model preload completed"));
    return resp_ok(d);
}

static std::string handle_run(const std::string &name, const std::vector<std::string> &extra) {
    if (g_child_pid > 0)
        return resp_err(std::string("demo '") + g_active_demo + "' is already running - stop it first");

    const DemoEntry *demo = nullptr;
    for (int i = 0; i < N_DEMOS; i++) {
        if (name == DEMOS[i].name) { demo = &DEMOS[i]; break; }
    }
    if (!demo) return resp_err("unknown demo: " + name);

    // For edge-ai: silently preload if the TVM model cache doesn't exist
    if (demo->is_edge_ai && !file_exists(TVM_CACHE)) {
        syslog(LOG_INFO, "TVM model cache not found, running preload before '%s'", name.c_str());
        int rc = run_sync({PRELOAD_EXE, "--preload"});
        if (rc != 0)
            syslog(LOG_WARNING, "Auto-preload failed (rc=%d), proceeding anyway", rc);
    }

    std::vector<std::string> argv = {demo->executable};
    argv.insert(argv.end(), extra.begin(), extra.end());

    pid_t pid = launch_async(argv);
    if (pid < 0) return resp_err("failed to fork process");

    g_child_pid = pid;
    strncpy(g_active_demo, name.c_str(), sizeof(g_active_demo) - 1);
    g_active_demo[sizeof(g_active_demo) - 1] = '\0';
    g_invalidate_on_exit = demo->is_dsp_compute;

    syslog(LOG_INFO, "Started demo '%s' pid=%d (invalidate_tvm_on_exit=%s)",
           name.c_str(), (int)pid, demo->is_dsp_compute ? "yes" : "no");

    json_object *d = json_object_new_object();
    json_object_object_add(d, "demo", json_object_new_string(name.c_str()));
    json_object_object_add(d, "pid",  json_object_new_int(static_cast<int32_t>(pid)));
    return resp_ok(d);
}

// ---------------------------------------------------------------------------
// Command dispatch
// ---------------------------------------------------------------------------

static std::string dispatch(const char *line) {
    json_object *root = json_tokener_parse(line);
    if (!root) return resp_err("invalid JSON");

    json_object *cmd_obj = nullptr;
    if (!json_object_object_get_ex(root, "cmd", &cmd_obj)) {
        json_object_put(root);
        return resp_err("missing 'cmd' field");
    }
    std::string cmd = json_object_get_string(cmd_obj);

    std::string result;
    if      (cmd == "list")    { result = handle_list(); }
    else if (cmd == "status")  { result = handle_status(); }
    else if (cmd == "stop")    { result = handle_stop(); }
    else if (cmd == "preload") { result = handle_preload(); }
    else if (cmd == "quit")    { g_quit = 1; result = resp_ok(); }
    else if (cmd == "run") {
        json_object *name_obj = nullptr;
        if (!json_object_object_get_ex(root, "demo", &name_obj)) {
            result = resp_err("'run' requires a 'demo' field");
        } else {
            std::vector<std::string> args;
            json_object *arr = nullptr;
            if (json_object_object_get_ex(root, "args", &arr)) {
                int n = json_object_array_length(arr);
                args.reserve(static_cast<size_t>(n));
                for (int i = 0; i < n; i++) {
                    auto *el = json_object_array_get_idx(arr, i);
                    args.emplace_back(json_object_get_string(el));
                }
            }
            result = handle_run(json_object_get_string(name_obj), args);
        }
    } else {
        result = resp_err("unknown command: " + cmd);
    }

    json_object_put(root);
    return result;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    // Signal setup
    struct sigaction sa{};
    sigemptyset(&sa.sa_mask);
    sa.sa_handler = on_signal;
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGINT,  &sa, nullptr);
    sa.sa_handler = on_sigchld;
    sa.sa_flags   = SA_NOCLDSTOP;
    sigaction(SIGCHLD, &sa, nullptr);

    openlog("demo-manager", LOG_PID | LOG_CONS, LOG_DAEMON);
    syslog(LOG_INFO, "demo-manager daemon starting");

    // PID file
    if (FILE *f = fopen(PID_FILE, "w")) {
        fprintf(f, "%d\n", getpid());
        fclose(f);
    }

    // Create Unix domain socket
    int srv = socket(AF_UNIX, SOCK_STREAM, 0);
    if (srv < 0) { syslog(LOG_ERR, "socket: %s", strerror(errno)); return 1; }

    unlink(SOCKET_PATH);

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);

    if (bind(srv, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
        syslog(LOG_ERR, "bind %s: %s", SOCKET_PATH, strerror(errno));
        return 1;
    }
    chmod(SOCKET_PATH, 0666);
    listen(srv, 8);

    syslog(LOG_INFO, "listening on %s", SOCKET_PATH);

    // Main event loop
    while (!g_quit) {
        if (g_child_done) reap_child();

        pollfd pfd{srv, POLLIN, 0};
        int r = poll(&pfd, 1, 500); // 500 ms so signals are checked regularly
        if (r < 0) {
            if (errno == EINTR) continue;
            syslog(LOG_ERR, "poll: %s", strerror(errno));
            break;
        }
        if (!r || !(pfd.revents & POLLIN)) continue;

        int cli = accept(srv, nullptr, nullptr);
        if (cli < 0) { if (errno != EINTR) syslog(LOG_WARNING, "accept: %s", strerror(errno)); continue; }

        char buf[4096] = {};
        ssize_t n = read(cli, buf, sizeof(buf) - 1);
        if (n > 0) {
            // Strip trailing whitespace / newlines
            while (n > 0 && static_cast<unsigned char>(buf[n - 1]) <= ' ') buf[--n] = '\0';
            syslog(LOG_DEBUG, "cmd: %s", buf);
            std::string resp = dispatch(buf);
            write(cli, resp.c_str(), resp.size());
        }
        close(cli);
    }

    syslog(LOG_INFO, "shutting down");
    stop_child(); // stop any running demo and invalidate cache if needed

    close(srv);
    unlink(SOCKET_PATH);
    unlink(PID_FILE);
    closelog();
    return 0;
}
