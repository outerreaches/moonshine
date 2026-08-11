#include "k3_chat.h"
#include "k3_json.h"
#include "k3_openai.h"
#include "moonshine_version.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

enum {
    K3_SERVER_DEFAULT_PORT = 8080,
    K3_SERVER_DEFAULT_CONTEXT = 8192,
    K3_SERVER_DEFAULT_SEQUENTIAL_LIMIT =
        K3_CHAT_MEASURED_SEQUENTIAL_LIMIT,
    K3_SERVER_DEFAULT_EXPERTS = 32,
    K3_SERVER_DEFAULT_STAGING = 16,
    K3_SERVER_DEFAULT_MAX_OUTPUT_TOKENS = 8192,
    K3_SERVER_MAX_OUTPUT_TOKENS = 65536,
    K3_SERVER_MAX_HEADERS = 64 * 1024,
    K3_SERVER_DEFAULT_MAX_BODY = 8 * 1024 * 1024,
    K3_SERVER_KEEPALIVE_SECONDS = 10,
    K3_SERVER_LOG_PROGRESS_SECONDS = 60,
};

typedef enum {
    SERVER_LOG_INFO = 0,
    SERVER_LOG_WARN = 1,
    SERVER_LOG_ERROR = 2,
} server_log_level;

typedef struct {
    char   method[16];
    char   path[2048];
    char  *body;
    size_t body_size;
    char  *authorization;
} http_request;

typedef struct {
    const char *model_root;
    const char *host;
    const char *api_key;
    uint16_t    port;
    uint32_t    context;
    uint32_t    sequential_limit;
    uint16_t    experts;
    uint16_t    staging;
    uint32_t    max_output_tokens;
    size_t      max_body;
    k3_prefill_projection_backend range_backend;
    bool        clear_expert_cache_per_request;
    const char *decode_diagnostics_prefix;
    bool        capture_state_digest;
} server_config;

typedef struct {
    int         fd;
    const char *completion_id;
    time_t      created;
    char       *pending;
    size_t      pending_size;
    size_t      pending_capacity;
    bool        pending_reasoning;
    bool        thinking;
    bool        defer_content;
    struct timespec last_progress;
    bool        failed;
} stream_state;

typedef struct {
    const char     *completion_id;
    stream_state   *stream;
    struct timespec request_start;
    struct timespec prefill_start;
    struct timespec decode_start;
    struct timespec last_prefill_log;
    const char     *decode_phase;
    uint32_t        max_output_tokens;
    bool            thinking;
    bool            clear_expert_cache;
} request_observer;

static volatile sig_atomic_t stop_requested = 0;
static volatile sig_atomic_t active_listener = -1;
static unsigned long long completion_counter = 0u;
static bool interactive_log = false;

static void stop_handler(int signum) {
    (void)signum;
    stop_requested = 1;
    if (active_listener >= 0) {
        close((int)active_listener);
        active_listener = -1;
    }
}

static void set_error(char *error, size_t error_size, const char *fmt, ...) {
    if (error == NULL || error_size == 0u) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(error, error_size, fmt, ap);
    va_end(ap);
}

static double elapsed_seconds(struct timespec start,
                              struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) +
           (double)(end.tv_nsec - start.tv_nsec) / 1e9;
}

static const char *log_level_name(server_log_level level) {
    return level == SERVER_LOG_ERROR ? "ERROR" :
        (level == SERVER_LOG_WARN ? "WARN" : "INFO");
}

static const char *log_level_color(server_log_level level) {
    return level == SERVER_LOG_ERROR ? "\033[31m" :
        (level == SERVER_LOG_WARN ? "\033[33m" : "\033[36m");
}

static void server_log(server_log_level level,
                       const char *event,
                       const char *completion_id,
                       const char *fmt, ...) {
    struct timespec now;
    struct tm utc;
    char timestamp[40];
    clock_gettime(CLOCK_REALTIME, &now);
    gmtime_r(&now.tv_sec, &utc);
    if (strftime(timestamp, sizeof(timestamp),
                 "%Y-%m-%dT%H:%M:%S.000Z", &utc) == 0u) {
        memcpy(timestamp, "1970-01-01T00:00:00.000Z", 25u);
    }
    const unsigned milliseconds =
        (unsigned)((now.tv_nsec / 1000000L) % 1000L);
    timestamp[20] = (char)('0' + milliseconds / 100u);
    timestamp[21] = (char)('0' + (milliseconds / 10u) % 10u);
    timestamp[22] = (char)('0' + milliseconds % 10u);

    char message[3072] = { 0 };
    bool truncated = false;
    if (fmt != NULL && fmt[0] != '\0') {
        va_list ap;
        va_start(ap, fmt);
        const int required = vsnprintf(
            message, sizeof(message), fmt, ap);
        va_end(ap);
        if (required >= 0) {
            truncated = (size_t)required >= sizeof(message);
            for (size_t i = 0u; message[i] != '\0'; i++) {
                if ((unsigned char)message[i] < 0x20u) {
                    message[i] = ' ';
                }
            }
        }
    }

    char line[4096];
    const char *id_prefix = completion_id == NULL ? "" : " id=";
    const char *id_value = completion_id == NULL ? "" : completion_id;
    const char *message_prefix = message[0] == '\0' ? "" : " ";
    const char *truncation = truncated ? " log_truncated=yes" : "";
    const int required = interactive_log ?
        snprintf(
            line, sizeof(line),
            "%s %s%-5s\033[0m %-25s%s%s%s%s%s\n",
            timestamp, log_level_color(level),
            log_level_name(level), event,
            id_prefix, id_value, message_prefix, message, truncation) :
        snprintf(
            line, sizeof(line),
            "%s %-5s %-25s%s%s%s%s%s\n",
            timestamp, log_level_name(level), event,
            id_prefix, id_value, message_prefix, message, truncation);
    if (required <= 0) {
        return;
    }
    const size_t line_size = (size_t)required < sizeof(line) ?
        (size_t)required : sizeof(line) - 1u;
    ssize_t written;
    do {
        written = write(STDERR_FILENO, line, line_size);
    } while (written < 0 && errno == EINTR);
}

static const char *prefill_strategy_name(
        k3_chat_prefill_strategy strategy) {
    return strategy == K3_CHAT_PREFILL_LAYER_MAJOR ?
        "layer-major" : "token-major";
}

static const char *tool_choice_name(k3_tool_choice choice) {
    return choice == K3_TOOL_CHOICE_REQUIRED ? "required" :
        (choice == K3_TOOL_CHOICE_NONE ? "none" : "auto");
}

static const char *response_format_name(k3_response_format format) {
    return format == K3_RESPONSE_FORMAT_JSON_SCHEMA ? "json_schema" :
        (format == K3_RESPONSE_FORMAT_JSON_OBJECT ?
            "json_object" : "text");
}

static bool reject_config(const char *fmt, ...) {
    char error[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(error, sizeof(error), fmt, ap);
    va_end(ap);
    server_log(
        SERVER_LOG_ERROR, "server.config.reject", NULL,
        "error=\"%s\"", error);
    return false;
}

static bool parse_u32(const char *text, uint32_t min_value,
                      uint32_t max_value, uint32_t *value) {
    if (text == NULL || text[0] == '\0') {
        return false;
    }
    errno = 0;
    char *end = NULL;
    const unsigned long parsed = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed < min_value || parsed > max_value) {
        return false;
    }
    *value = (uint32_t)parsed;
    return true;
}

static void usage(FILE *stream, const char *program) {
    fprintf(
        stream,
        "Usage: %s [options] MODEL\n"
        "\n"
        "Serve Moonshine through a one-slot OpenAI-compatible API.\n"
        "\n"
        "Options:\n"
        "  --host ADDRESS        Bind address (default 127.0.0.1)\n"
        "  --port PORT           TCP port (default 8080)\n"
        "  --api-key KEY         Require Authorization: Bearer KEY\n"
        "  --context TOKENS      Context capacity (default 8192)\n"
        "  --sequential-limit N  Token-major prompt limit (default 7)\n"
        "  --experts N           Resident expert slots per layer (default 32)\n"
        "  --staging N           Expert staging slots (default 16)\n"
        "  --max-output-tokens N Per-request output ceiling\n"
        "                        (default 8192; maximum 65536)\n"
        "  --range-backend NAME  Diagnostic range backend: default|kda-blas\n"
        "  --max-body BYTES      Maximum JSON request body (default 8388608)\n"
        "  --clear-expert-cache-per-request\n"
        "                        Force cold-cache request benchmarks\n"
        "  --decode-diagnostics PREFIX\n"
        "                        Write sensitive decode cache/ledger/route CSVs\n"
        "  --decode-state-digest\n"
        "                        Log state-comparison fingerprints (expensive)\n"
        "  -h, --help            Show this help\n"
        "  --version             Show the Moonshine version\n"
        "\n"
        "MODEL may instead be supplied in MOONSHINE_MODEL. "
        "MOONSHINE_API_KEY is used when\n"
        "--api-key is omitted. A non-loopback bind requires an API key.\n",
        program);
}

static bool parse_args(int argc, char **argv, server_config *config) {
    const char *environment_model = getenv("MOONSHINE_MODEL");
    *config = (server_config) {
        .model_root = environment_model,
        .host = "127.0.0.1",
        .api_key = getenv("MOONSHINE_API_KEY"),
        .port = K3_SERVER_DEFAULT_PORT,
        .context = K3_SERVER_DEFAULT_CONTEXT,
        .sequential_limit = K3_SERVER_DEFAULT_SEQUENTIAL_LIMIT,
        .experts = K3_SERVER_DEFAULT_EXPERTS,
        .staging = K3_SERVER_DEFAULT_STAGING,
        .max_output_tokens = K3_SERVER_DEFAULT_MAX_OUTPUT_TOKENS,
        .max_body = K3_SERVER_DEFAULT_MAX_BODY,
    };
    bool positional_model = false;
    for (int i = 1; i < argc; i++) {
        const char *argument = argv[i];
        if (strcmp(argument, "-h") == 0 ||
            strcmp(argument, "--help") == 0) {
            usage(stdout, argv[0]);
            exit(0);
        }
        if (strcmp(argument, "--version") == 0) {
            printf("%s %s\n", MOONSHINE_NAME, MOONSHINE_VERSION);
            exit(0);
        }
        if (strcmp(
                argument,
                "--clear-expert-cache-per-request") == 0) {
            config->clear_expert_cache_per_request = true;
            continue;
        }
        if (strcmp(argument, "--decode-state-digest") == 0) {
            config->capture_state_digest = true;
            continue;
        }
        if (strcmp(argument, "--host") == 0 ||
            strcmp(argument, "--port") == 0 ||
            strcmp(argument, "--api-key") == 0 ||
            strcmp(argument, "--context") == 0 ||
            strcmp(argument, "--sequential-limit") == 0 ||
            strcmp(argument, "--experts") == 0 ||
            strcmp(argument, "--staging") == 0 ||
            strcmp(argument, "--max-output-tokens") == 0 ||
            strcmp(argument, "--range-backend") == 0 ||
            strcmp(argument, "--decode-diagnostics") == 0 ||
            strcmp(argument, "--max-body") == 0) {
            if (++i >= argc) {
                return reject_config("%s needs a value", argument);
            }
            const char *value = argv[i];
            uint32_t parsed = 0u;
            if (strcmp(argument, "--host") == 0) {
                config->host = value;
            } else if (strcmp(argument, "--api-key") == 0) {
                config->api_key = value;
            } else if (strcmp(argument, "--port") == 0) {
                if (!parse_u32(value, 1u, 65535u, &parsed)) {
                    return reject_config("invalid port %s", value);
                }
                config->port = (uint16_t)parsed;
            } else if (strcmp(argument, "--context") == 0) {
                if (!parse_u32(value, 64u, 1048576u, &parsed)) {
                    return reject_config("invalid context %s", value);
                }
                config->context = parsed;
            } else if (strcmp(argument, "--sequential-limit") == 0) {
                if (!parse_u32(value, 1u, 8192u, &parsed)) {
                    return reject_config(
                        "invalid sequential limit %s", value);
                }
                config->sequential_limit = parsed;
            } else if (strcmp(argument, "--experts") == 0) {
                if (!parse_u32(value, 1u, 256u, &parsed)) {
                    return reject_config(
                        "invalid expert count %s", value);
                }
                config->experts = (uint16_t)parsed;
            } else if (strcmp(argument, "--staging") == 0) {
                if (!parse_u32(value, 1u, 256u, &parsed)) {
                    return reject_config(
                        "invalid staging count %s", value);
                }
                config->staging = (uint16_t)parsed;
            } else if (strcmp(argument, "--max-output-tokens") == 0) {
                if (!parse_u32(
                        value, 1u, K3_SERVER_MAX_OUTPUT_TOKENS,
                        &parsed)) {
                    return reject_config(
                        "invalid maximum output tokens %s", value);
                }
                config->max_output_tokens = parsed;
            } else if (strcmp(argument, "--range-backend") == 0) {
                if (strcmp(value, "default") == 0) {
                    config->range_backend =
                        K3_PREFILL_PROJECTION_DEFAULT;
                } else if (strcmp(value, "kda-blas") == 0) {
                    config->range_backend =
                        K3_PREFILL_PROJECTION_KDA_DEQUANT_BLAS_EXPERIMENT;
                } else {
                    return reject_config(
                        "invalid range backend %s", value);
                }
            } else if (strcmp(
                           argument,
                           "--decode-diagnostics") == 0) {
                if (value[0] == '\0') {
                    return reject_config(
                        "decode diagnostics prefix is empty");
                }
                config->decode_diagnostics_prefix = value;
            } else {
                if (!parse_u32(value, 1024u, UINT32_MAX, &parsed)) {
                    return reject_config(
                        "invalid maximum body %s", value);
                }
                config->max_body = parsed;
            }
            continue;
        }
        if (argument[0] == '-') {
            return reject_config("unknown option %s", argument);
        }
        if (positional_model) {
            return reject_config("more than one model path supplied");
        }
        config->model_root = argument;
        positional_model = true;
    }
    if (config->model_root == NULL || config->model_root[0] == '\0') {
        return reject_config("MODEL or MOONSHINE_MODEL is required");
    }
    if (config->api_key != NULL && config->api_key[0] == '\0') {
        config->api_key = NULL;
    }
    return true;
}

static bool loopback_host(const char *host) {
    if (strcmp(host, "localhost") == 0 ||
        strcmp(host, "::1") == 0) {
        return true;
    }
    struct in_addr address;
    return inet_pton(AF_INET, host, &address) == 1 &&
           (ntohl(address.s_addr) >> 24) == 127u;
}

static uint32_t effective_max_output_tokens(
        const server_config *config) {
    return config->max_output_tokens < config->context ?
        config->max_output_tokens : config->context;
}

static int create_listener(const server_config *config,
                           char *error, size_t error_size) {
    char service[16];
    snprintf(service, sizeof(service), "%u", config->port);
    const struct addrinfo hints = {
        .ai_family = AF_UNSPEC,
        .ai_socktype = SOCK_STREAM,
        .ai_flags = AI_PASSIVE,
    };
    struct addrinfo *addresses = NULL;
    const int status = getaddrinfo(
        config->host, service, &hints, &addresses);
    if (status != 0) {
        set_error(error, error_size, "resolving %s:%s failed: %s",
                  config->host, service, gai_strerror(status));
        return -1;
    }
    int listener = -1;
    int saved_errno = 0;
    for (const struct addrinfo *address = addresses;
         address != NULL; address = address->ai_next) {
        listener = socket(
            address->ai_family, address->ai_socktype,
            address->ai_protocol);
        if (listener < 0) {
            saved_errno = errno;
            continue;
        }
        const int enabled = 1;
        (void)setsockopt(
            listener, SOL_SOCKET, SO_REUSEADDR,
            &enabled, sizeof(enabled));
        if (bind(listener, address->ai_addr,
                 address->ai_addrlen) == 0 &&
            listen(listener, 16) == 0) {
            break;
        }
        saved_errno = errno;
        close(listener);
        listener = -1;
    }
    freeaddrinfo(addresses);
    if (listener < 0) {
        set_error(error, error_size, "binding %s:%s failed: %s",
                  config->host, service,
                  strerror(saved_errno == 0 ? errno : saved_errno));
    }
    return listener;
}

static bool send_all(int fd, const void *data, size_t size) {
    const char *bytes = (const char *)data;
    while (size != 0u) {
#ifdef MSG_NOSIGNAL
        const ssize_t sent = send(fd, bytes, size, MSG_NOSIGNAL);
#else
        const ssize_t sent = send(fd, bytes, size, 0);
#endif
        if (sent > 0) {
            bytes += (size_t)sent;
            size -= (size_t)sent;
            continue;
        }
        if (sent < 0 && errno == EINTR) {
            continue;
        }
        return false;
    }
    return true;
}

static const char *status_text(int status) {
    switch (status) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 411: return "Length Required";
    case 413: return "Content Too Large";
    case 500: return "Internal Server Error";
    case 501: return "Not Implemented";
    default: return "Error";
    }
}

static bool send_response(int fd, int status, const char *content_type,
                          const char *body, size_t body_size,
                          const char *extra_headers) {
    char headers[2048];
    const int length = snprintf(
        headers, sizeof(headers),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "Cache-Control: no-store\r\n"
        "%s"
        "\r\n",
        status, status_text(status), content_type, body_size,
        extra_headers == NULL ? "" : extra_headers);
    return length > 0 && (size_t)length < sizeof(headers) &&
           send_all(fd, headers, (size_t)length) &&
           send_all(fd, body, body_size);
}

static bool send_json_error(int fd, int status, const char *message) {
    char *escaped = NULL;
    char error[256];
    if (!k3_json_escape(message, strlen(message), &escaped, NULL,
                        error, sizeof(error))) {
        static const char fallback[] =
            "{\"error\":{\"message\":\"internal error\","
            "\"type\":\"server_error\",\"param\":null,\"code\":null}}";
        return send_response(
            fd, 500, "application/json", fallback,
            sizeof(fallback) - 1u, NULL);
    }
    const char *type = status >= 500 ?
        "server_error" : "invalid_request_error";
    const int required = snprintf(
        NULL, 0,
        "{\"error\":{\"message\":%s,\"type\":\"%s\","
        "\"param\":null,\"code\":null}}",
        escaped, type);
    char *body = required < 0 ? NULL :
        (char *)malloc((size_t)required + 1u);
    if (body != NULL) {
        snprintf(
            body, (size_t)required + 1u,
            "{\"error\":{\"message\":%s,\"type\":\"%s\","
            "\"param\":null,\"code\":null}}",
            escaped, type);
    }
    free(escaped);
    if (body == NULL) {
        static const char fallback[] =
            "{\"error\":{\"message\":\"internal error\","
            "\"type\":\"server_error\",\"param\":null,\"code\":null}}";
        return send_response(
            fd, 500, "application/json", fallback,
            sizeof(fallback) - 1u, NULL);
    }
    const bool ok = send_response(
        fd, status, "application/json", body,
        (size_t)required, status == 401 ?
            "WWW-Authenticate: Bearer\r\n" : NULL);
    free(body);
    return ok;
}

static void http_request_free(http_request *request) {
    if (request == NULL) {
        return;
    }
    free(request->body);
    free(request->authorization);
    memset(request, 0, sizeof(*request));
}

static char *trim_header_value(char *value) {
    while (*value == ' ' || *value == '\t') {
        value++;
    }
    char *end = value + strlen(value);
    while (end != value &&
           (end[-1] == ' ' || end[-1] == '\t')) {
        *--end = '\0';
    }
    return value;
}

static bool receive_request(int fd, size_t max_body,
                            http_request *request, int *http_status,
                            char *error, size_t error_size) {
    memset(request, 0, sizeof(*request));
    *http_status = 400;
    char *headers = (char *)malloc(K3_SERVER_MAX_HEADERS + 1u);
    if (headers == NULL) {
        *http_status = 500;
        set_error(error, error_size, "allocating request headers failed");
        return false;
    }
    size_t received = 0u;
    char *terminator = NULL;
    while (received < K3_SERVER_MAX_HEADERS) {
        const ssize_t amount = recv(
            fd, headers + received,
            K3_SERVER_MAX_HEADERS - received, 0);
        if (amount > 0) {
            received += (size_t)amount;
            headers[received] = '\0';
            terminator = strstr(headers, "\r\n\r\n");
            if (terminator != NULL) {
                break;
            }
            continue;
        }
        if (amount < 0 && errno == EINTR) {
            continue;
        }
        set_error(error, error_size,
                  amount == 0 ? "connection closed before request headers" :
                  "reading request headers failed: %s",
                  strerror(errno));
        free(headers);
        return false;
    }
    if (terminator == NULL) {
        *http_status = 413;
        set_error(error, error_size,
                  "request headers exceed %u bytes",
                  K3_SERVER_MAX_HEADERS);
        free(headers);
        return false;
    }
    const size_t header_size =
        (size_t)(terminator - headers) + 4u;
    *terminator = '\0';
    char *line_end = strstr(headers, "\r\n");
    if (line_end == NULL) {
        set_error(error, error_size, "invalid HTTP request line");
        free(headers);
        return false;
    }
    *line_end = '\0';
    char version[16];
    char trailing = '\0';
    if (sscanf(headers, "%15s %2047s %15s %c",
               request->method, request->path,
               version, &trailing) != 3 ||
        (strcmp(version, "HTTP/1.1") != 0 &&
         strcmp(version, "HTTP/1.0") != 0)) {
        set_error(error, error_size, "invalid HTTP request line");
        free(headers);
        return false;
    }

    bool have_content_length = false;
    size_t content_length = 0u;
    bool expect_continue = false;
    for (char *line = line_end + 2u;
         line < terminator && *line != '\0';) {
        char *next = strstr(line, "\r\n");
        if (next == NULL) {
            next = terminator;
        }
        *next = '\0';
        char *colon = strchr(line, ':');
        if (colon == NULL) {
            set_error(error, error_size, "malformed HTTP header");
            free(headers);
            return false;
        }
        *colon = '\0';
        char *value = trim_header_value(colon + 1u);
        if (strcasecmp(line, "Content-Length") == 0) {
            if (have_content_length || value[0] == '\0') {
                set_error(error, error_size,
                          "invalid duplicate Content-Length");
                free(headers);
                return false;
            }
            errno = 0;
            char *end = NULL;
            const unsigned long long parsed =
                strtoull(value, &end, 10);
            if (errno != 0 || end == value || *end != '\0' ||
                parsed > SIZE_MAX) {
                set_error(error, error_size,
                          "invalid Content-Length");
                free(headers);
                return false;
            }
            content_length = (size_t)parsed;
            have_content_length = true;
        } else if (strcasecmp(line, "Transfer-Encoding") == 0 &&
                   strcasecmp(value, "identity") != 0) {
            *http_status = 501;
            set_error(error, error_size,
                      "chunked request bodies are not supported");
            free(headers);
            return false;
        } else if (strcasecmp(line, "Authorization") == 0) {
            free(request->authorization);
            request->authorization = strdup(value);
            if (request->authorization == NULL) {
                *http_status = 500;
                set_error(error, error_size,
                          "allocating authorization header failed");
                free(headers);
                return false;
            }
        } else if (strcasecmp(line, "Expect") == 0 &&
                   strcasecmp(value, "100-continue") == 0) {
            expect_continue = true;
        }
        line = next + 2u;
    }
    if (strcmp(request->method, "POST") == 0 &&
        !have_content_length) {
        *http_status = 411;
        set_error(error, error_size,
                  "POST requests require Content-Length");
        free(headers);
        return false;
    }
    if (content_length > max_body) {
        *http_status = 413;
        set_error(error, error_size,
                  "request body exceeds the %zu-byte limit", max_body);
        free(headers);
        return false;
    }
    if (content_length == SIZE_MAX) {
        *http_status = 413;
        set_error(error, error_size, "request body is too large");
        free(headers);
        return false;
    }
    request->body = (char *)malloc(content_length + 1u);
    if (request->body == NULL) {
        *http_status = 500;
        set_error(error, error_size, "allocating request body failed");
        free(headers);
        return false;
    }
    size_t body_received = received - header_size;
    if (body_received > content_length) {
        body_received = content_length;
    }
    if (body_received != 0u) {
        memcpy(request->body, headers + header_size, body_received);
    }
    free(headers);
    if (expect_continue && body_received < content_length &&
        !send_all(fd, "HTTP/1.1 100 Continue\r\n\r\n", 25u)) {
        set_error(error, error_size,
                  "sending 100 Continue failed");
        return false;
    }
    while (body_received < content_length) {
        const ssize_t amount = recv(
            fd, request->body + body_received,
            content_length - body_received, 0);
        if (amount > 0) {
            body_received += (size_t)amount;
            continue;
        }
        if (amount < 0 && errno == EINTR) {
            continue;
        }
        set_error(error, error_size,
                  amount == 0 ? "connection closed before request body" :
                  "reading request body failed: %s",
                  strerror(errno));
        return false;
    }
    request->body[content_length] = '\0';
    request->body_size = content_length;
    char *query = strchr(request->path, '?');
    if (query != NULL) {
        *query = '\0';
    }
    return true;
}

static bool authorized(const http_request *request, const char *api_key) {
    if (api_key == NULL) {
        return true;
    }
    if (request->authorization == NULL ||
        strncmp(request->authorization, "Bearer ", 7u) != 0) {
        return false;
    }
    const char *provided = request->authorization + 7u;
    const size_t expected_size = strlen(api_key);
    const size_t provided_size = strlen(provided);
    size_t difference = expected_size ^ provided_size;
    const size_t compared =
        expected_size < provided_size ?
            expected_size : provided_size;
    for (size_t i = 0u; i < compared; i++) {
        difference |= (unsigned char)(api_key[i] ^ provided[i]);
    }
    return difference == 0u;
}

static bool stream_send_event(stream_state *stream,
                              const char *event, size_t event_size) {
    if (stream->failed) {
        return false;
    }
    stream->failed =
        !send_all(stream->fd, "data: ", 6u) ||
        !send_all(stream->fd, event, event_size) ||
        !send_all(stream->fd, "\n\n", 2u);
    return !stream->failed;
}

static bool stream_send_text_field(stream_state *stream,
                                   const char *field,
                                   const char *bytes,
                                   size_t size) {
    char error[256];
    char *escaped = NULL;
    if (!k3_json_escape(
            bytes, size, &escaped, NULL,
            error, sizeof(error))) {
        stream->failed = true;
        return false;
    }
    const int required = snprintf(
        NULL, 0,
        "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"created\":%lld,\"model\":\"%s\","
        "\"system_fingerprint\":\"" MOONSHINE_SYSTEM_FINGERPRINT "\","
        "\"choices\":[{\"index\":0,\"delta\":{\"%s\":%s},"
        "\"finish_reason\":null}]}",
        stream->completion_id, (long long)stream->created,
        MOONSHINE_MODEL_ID, field, escaped);
    char *event = required < 0 ? NULL :
        (char *)malloc((size_t)required + 1u);
    if (event != NULL) {
        snprintf(
            event, (size_t)required + 1u,
            "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
            "\"created\":%lld,\"model\":\"%s\","
            "\"system_fingerprint\":\"" MOONSHINE_SYSTEM_FINGERPRINT "\","
            "\"choices\":[{\"index\":0,\"delta\":{\"%s\":%s},"
            "\"finish_reason\":null}]}",
            stream->completion_id, (long long)stream->created,
            MOONSHINE_MODEL_ID, field, escaped);
    }
    free(escaped);
    if (event == NULL) {
        stream->failed = true;
        return false;
    }
    const bool ok =
        stream_send_event(stream, event, (size_t)required);
    free(event);
    return ok;
}

static bool stream_send_pending(stream_state *stream) {
    if (stream->pending_size == 0u) {
        return true;
    }
    const char *field = stream->pending_reasoning ?
        "reasoning_content" : "content";
    const bool ok = stream_send_text_field(
        stream, field, stream->pending, stream->pending_size);
    if (ok) {
        stream->pending_size = 0u;
    }
    return ok;
}

static bool stream_send_tool_call(
        stream_state *stream,
        const k3_tool_call *call,
        size_t index) {
    char id[384];
    const int id_size = snprintf(
        id, sizeof(id), "call_%s_%zu",
        stream->completion_id, index + 1u);
    char error[256];
    char *escaped_id = NULL;
    char *escaped_name = NULL;
    char *escaped_arguments = NULL;
    bool ok = id_size > 0 && (size_t)id_size < sizeof(id) &&
        k3_json_escape(
            id, (size_t)id_size,
            &escaped_id, NULL, error, sizeof(error)) &&
        k3_json_escape(
            call->name, strlen(call->name),
            &escaped_name, NULL, error, sizeof(error)) &&
        k3_json_escape(
            call->arguments, strlen(call->arguments),
            &escaped_arguments, NULL, error, sizeof(error));
    if (!ok) {
        free(escaped_arguments);
        free(escaped_name);
        free(escaped_id);
        stream->failed = true;
        return false;
    }
    const int required = snprintf(
        NULL, 0,
        "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"created\":%lld,\"model\":\"%s\","
        "\"system_fingerprint\":\"" MOONSHINE_SYSTEM_FINGERPRINT "\","
        "\"choices\":[{\"index\":0,\"delta\":{"
        "\"tool_calls\":[{\"index\":%zu,\"id\":%s,"
        "\"type\":\"function\",\"function\":{"
        "\"name\":%s,\"arguments\":%s}}]},"
        "\"finish_reason\":null}]}",
        stream->completion_id, (long long)stream->created,
        MOONSHINE_MODEL_ID, index,
        escaped_id, escaped_name, escaped_arguments);
    char *event = required < 0 ? NULL :
        (char *)malloc((size_t)required + 1u);
    if (event != NULL) {
        snprintf(
            event, (size_t)required + 1u,
            "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
            "\"created\":%lld,\"model\":\"%s\","
            "\"system_fingerprint\":\"" MOONSHINE_SYSTEM_FINGERPRINT "\","
            "\"choices\":[{\"index\":0,\"delta\":{"
            "\"tool_calls\":[{\"index\":%zu,\"id\":%s,"
            "\"type\":\"function\",\"function\":{"
            "\"name\":%s,\"arguments\":%s}}]},"
            "\"finish_reason\":null}]}",
            stream->completion_id, (long long)stream->created,
            MOONSHINE_MODEL_ID, index,
            escaped_id, escaped_name, escaped_arguments);
    }
    free(escaped_arguments);
    free(escaped_name);
    free(escaped_id);
    if (event == NULL) {
        stream->failed = true;
        return false;
    }
    ok = stream_send_event(stream, event, (size_t)required);
    free(event);
    return ok;
}

static size_t complete_utf8_prefix(const char *bytes, size_t size) {
    size_t offset = 0u;
    while (offset < size) {
        const unsigned char first =
            (unsigned char)bytes[offset];
        size_t width = 1u;
        if (first < 0x80u) {
            width = 1u;
        } else if (first >= 0xc2u && first <= 0xdfu) {
            width = 2u;
        } else if (first >= 0xe0u && first <= 0xefu) {
            width = 3u;
        } else if (first >= 0xf0u && first <= 0xf4u) {
            width = 4u;
        }
        if (offset + width > size) {
            break;
        }
        bool valid = true;
        for (size_t i = 1u; i < width; i++) {
            const unsigned char continuation =
                (unsigned char)bytes[offset + i];
            if ((continuation & 0xc0u) != 0x80u) {
                valid = false;
                break;
            }
        }
        if (valid && width == 3u) {
            const unsigned char second =
                (unsigned char)bytes[offset + 1u];
            valid = !((first == 0xe0u && second < 0xa0u) ||
                      (first == 0xedu && second >= 0xa0u));
        } else if (valid && width == 4u) {
            const unsigned char second =
                (unsigned char)bytes[offset + 1u];
            valid = !((first == 0xf0u && second < 0x90u) ||
                      (first == 0xf4u && second >= 0x90u));
        }
        offset += valid ? width : 1u;
    }
    return offset;
}

static void stream_channel_callback(const char *bytes,
                                    size_t size,
                                    void *user_data,
                                    bool reasoning) {
    stream_state *stream = (stream_state *)user_data;
    if (stream->failed || size == 0u) {
        return;
    }
    if (stream->pending_size != 0u &&
        stream->pending_reasoning != reasoning &&
        !stream_send_pending(stream)) {
        return;
    }
    stream->pending_reasoning = reasoning;
    if (stream->pending_size > SIZE_MAX - size) {
        stream->failed = true;
        return;
    }
    const size_t required = stream->pending_size + size;
    if (required > stream->pending_capacity) {
        size_t capacity =
            stream->pending_capacity == 0u ?
                64u : stream->pending_capacity;
        while (capacity < required) {
            if (capacity > SIZE_MAX / 2u) {
                stream->failed = true;
                return;
            }
            capacity *= 2u;
        }
        char *pending = (char *)realloc(
            stream->pending, capacity);
        if (pending == NULL) {
            stream->failed = true;
            return;
        }
        stream->pending = pending;
        stream->pending_capacity = capacity;
    }
    memcpy(stream->pending + stream->pending_size, bytes, size);
    stream->pending_size += size;
    const size_t complete = complete_utf8_prefix(
        stream->pending, stream->pending_size);
    if (complete != 0u &&
        stream_send_text_field(
            stream,
            reasoning ? "reasoning_content" : "content",
            stream->pending, complete)) {
        memmove(
            stream->pending,
            stream->pending + complete,
            stream->pending_size - complete);
        stream->pending_size -= complete;
    }
}

static void stream_callback(const char *bytes, size_t size, void *user_data) {
    stream_channel_callback(bytes, size, user_data, false);
}

static void stream_reasoning_callback(
        const char *bytes, size_t size, void *user_data) {
    stream_channel_callback(bytes, size, user_data, true);
}

static bool stream_begin(stream_state *stream) {
    static const char headers[] =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache, no-store\r\n"
        "Connection: close\r\n"
        "X-Accel-Buffering: no\r\n"
        "\r\n";
    if (!send_all(stream->fd, headers, sizeof(headers) - 1u)) {
        stream->failed = true;
        return false;
    }
    char event[1024];
    const char *reasoning_field = stream->thinking ?
        ",\"reasoning_content\":\"\"" : "";
    const int size = snprintf(
        event, sizeof(event),
        "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"created\":%lld,\"model\":\"%s\","
        "\"system_fingerprint\":\"" MOONSHINE_SYSTEM_FINGERPRINT "\","
        "\"choices\":[{\"index\":0,\"delta\":{"
        "\"role\":\"assistant\",\"content\":\"\"%s},"
        "\"finish_reason\":null}]}",
        stream->completion_id, (long long)stream->created,
        MOONSHINE_MODEL_ID, reasoning_field);
    const bool ok =
        size > 0 && (size_t)size < sizeof(event) &&
        stream_send_event(stream, event, (size_t)size);
    if (ok) {
        clock_gettime(
            CLOCK_MONOTONIC, &stream->last_progress);
    }
    return ok;
}

static void stream_progress(
        k3_chat_prefill_progress_unit unit,
        uint32_t completed,
        uint32_t total,
        void *user_data) {
    stream_state *stream = (stream_state *)user_data;
    if (stream->failed) {
        return;
    }
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    if (completed != total &&
        elapsed_seconds(stream->last_progress, now) <
            K3_SERVER_KEEPALIVE_SECONDS) {
        return;
    }
    const char *unit_name =
        unit == K3_CHAT_PREFILL_PROGRESS_LAYERS ?
            "layer" : "token";
    char comment[160];
    const int size = snprintf(
        comment, sizeof(comment),
        ": moonshine prefill %s %u/%u\r\n\r\n",
        unit_name, completed, total);
    if (size <= 0 || (size_t)size >= sizeof(comment) ||
        !send_all(stream->fd, comment, (size_t)size)) {
        stream->failed = true;
        return;
    }
    stream->last_progress = now;
}

static void observe_prefill_progress(
        k3_chat_prefill_progress_unit unit,
        uint32_t completed,
        uint32_t total,
        void *user_data) {
    request_observer *observer = (request_observer *)user_data;
    if (observer->stream != NULL) {
        stream_progress(unit, completed, total, observer->stream);
    }
    if (completed == total) {
        return;
    }
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    if (elapsed_seconds(observer->last_prefill_log, now) <
        K3_SERVER_LOG_PROGRESS_SECONDS) {
        return;
    }
    server_log(
        SERVER_LOG_INFO, "request.prefill.progress",
        observer->completion_id,
        "unit=%s completed=%u total=%u elapsed=%.3fs",
        unit == K3_CHAT_PREFILL_PROGRESS_LAYERS ? "layer" : "token",
        completed, total,
        elapsed_seconds(observer->prefill_start, now));
    observer->last_prefill_log = now;
}

static void observe_lifecycle(
        k3_chat_lifecycle_event event,
        const k3_chat_turn_result *result,
        void *user_data) {
    request_observer *observer = (request_observer *)user_data;
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    if (event == K3_CHAT_LIFECYCLE_PREFILL_START) {
        observer->prefill_start = now;
        observer->last_prefill_log = now;
        const char *reuse = result->prompt_reused_tokens != 0u ? "hit" :
            (result->prompt_reuse_declined ? "miss" :
             (observer->clear_expert_cache ? "disabled" : "cold"));
        server_log(
            result->prompt_reuse_declined ?
                SERVER_LOG_WARN : SERVER_LOG_INFO,
            "request.prefill.start", observer->completion_id,
            "prompt=%u evaluated=%u reused=%u reuse=%s "
            "strategy=%s%s",
            result->prompt_tokens,
            result->prompt_evaluated_tokens,
            result->prompt_reused_tokens,
            reuse,
            prefill_strategy_name(result->prefill_strategy),
            result->prompt_reuse_declined ?
                " replacement=guarded" : "");
        if (result->prompt_reuse_declined) {
            server_log(
                SERVER_LOG_WARN, "request.prefix.miss",
                observer->completion_id,
                "retained=%u matched=%u candidate=%u",
                result->prompt_reuse_retained_tokens,
                result->prompt_reuse_matched_tokens,
                result->prompt_reuse_candidate_tokens);
        }
        return;
    }
    if (event == K3_CHAT_LIFECYCLE_PREFILL_COMPLETE) {
        const double rate = result->prompt_seconds > 0.0 ?
            (double)result->prompt_evaluated_tokens /
                result->prompt_seconds : 0.0;
        server_log(
            SERVER_LOG_INFO, "request.prefill.complete",
            observer->completion_id,
            "evaluated=%u reused=%u seconds=%.3f rate=%.3f_tok/s "
            "read=%.3f_GiB workspace_borrow=%.3f_GiB",
            result->prompt_evaluated_tokens,
            result->prompt_reused_tokens,
            result->prompt_seconds, rate,
            (double)result->range_stats.routed_physical_read_bytes /
                (1024.0 * 1024.0 * 1024.0),
            (double)result->range_stats.warm_cache_workspace_bytes /
                (1024.0 * 1024.0 * 1024.0));
        return;
    }
    if (event == K3_CHAT_LIFECYCLE_DECODE_START) {
        observer->decode_start = now;
        observer->decode_phase = observer->thinking ?
            "reasoning" : "response_or_tool";
        server_log(
            SERVER_LOG_INFO, "request.decode.start",
            observer->completion_id,
            "phase=%s max_output=%u",
            observer->decode_phase,
            observer->max_output_tokens);
        return;
    }
    if (event == K3_CHAT_LIFECYCLE_RESPONSE_START) {
        observer->decode_phase = "response_or_tool";
        server_log(
            SERVER_LOG_INFO, "request.decode.phase",
            observer->completion_id,
            "phase=%s generated=%u elapsed=%.3fs note=tool_payloads_buffered",
            observer->decode_phase,
            result->generated_tokens,
            elapsed_seconds(observer->decode_start, now));
        return;
    }
    if (event == K3_CHAT_LIFECYCLE_DECODE_PROGRESS) {
        server_log(
            SERVER_LOG_INFO, "request.decode.progress",
            observer->completion_id,
            "phase=%s generated=%u elapsed=%.3fs",
            observer->decode_phase == NULL ? "unknown" :
                observer->decode_phase,
            result->generated_tokens,
            elapsed_seconds(observer->decode_start, now));
    }
}

static void stream_end(stream_state *stream,
                       const k3_chat_turn_result *result) {
    if (stream->pending_size != 0u && !stream->failed) {
        (void)stream_send_pending(stream);
    }
    if (stream->defer_content && !stream->failed &&
        result->response.size != 0u) {
        (void)stream_send_text_field(
            stream, "content",
            result->response.data, result->response.size);
    }
    for (size_t i = 0u;
         i < result->tool_call_count && !stream->failed;
         i++) {
        (void)stream_send_tool_call(
            stream, &result->tool_calls[i], i);
    }
    if (!stream->failed) {
        char comment[192];
        const int size = snprintf(
            comment, sizeof(comment),
            ": moonshine prompt total=%u evaluated=%u "
            "reused=%u\r\n\r\n",
            result->prompt_tokens,
            result->prompt_evaluated_tokens,
            result->prompt_reused_tokens);
        if (size <= 0 || (size_t)size >= sizeof(comment) ||
            !send_all(stream->fd, comment, (size_t)size)) {
            stream->failed = true;
        }
    }
    if (!stream->failed) {
        const char *finish = result->finish_reason ==
            K3_CHAT_FINISH_TOOL_CALLS ? "tool_calls" :
            (result->finish_reason ==
                K3_CHAT_FINISH_END_OF_MESSAGE ? "stop" : "length");
        char event[1024];
        const int size = snprintf(
            event, sizeof(event),
            "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
            "\"created\":%lld,\"model\":\"%s\","
            "\"system_fingerprint\":\"" MOONSHINE_SYSTEM_FINGERPRINT "\","
            "\"choices\":[{\"index\":0,\"delta\":{},"
            "\"finish_reason\":\"%s\"}],"
            "\"usage\":{\"prompt_tokens\":%u,"
            "\"completion_tokens\":%u,\"total_tokens\":%u,"
            "\"prompt_tokens_details\":{\"cached_tokens\":%u}}}",
            stream->completion_id, (long long)stream->created,
            MOONSHINE_MODEL_ID, finish,
            result->prompt_tokens, result->generated_tokens,
            result->prompt_tokens + result->generated_tokens,
            result->prompt_reused_tokens);
        if (size > 0 && (size_t)size < sizeof(event)) {
            (void)stream_send_event(stream, event, (size_t)size);
        } else {
            stream->failed = true;
        }
    }
    if (!stream->failed) {
        (void)stream_send_event(stream, "[DONE]", 6u);
    }
    free(stream->pending);
    stream->pending = NULL;
    stream->pending_size = 0u;
    stream->pending_capacity = 0u;
}

static void stream_send_inference_error(stream_state *stream,
                                        const char *message) {
    if (!stream->failed) {
        char *escaped = NULL;
        char error[256];
        if (k3_json_escape(
                message, strlen(message), &escaped, NULL,
                error, sizeof(error))) {
            const int required = snprintf(
                NULL, 0, "{\"error\":{\"message\":%s,"
                "\"type\":\"server_error\",\"code\":null}}",
                escaped);
            char *event = required < 0 ? NULL :
                (char *)malloc((size_t)required + 1u);
            if (event != NULL) {
                snprintf(
                    event, (size_t)required + 1u,
                    "{\"error\":{\"message\":%s,"
                    "\"type\":\"server_error\",\"code\":null}}",
                    escaped);
                (void)stream_send_event(
                    stream, event, (size_t)required);
                free(event);
            }
            free(escaped);
        }
        if (!stream->failed) {
            (void)stream_send_event(stream, "[DONE]", 6u);
        }
    }
    free(stream->pending);
    stream->pending = NULL;
}

static void make_completion_id(char *id, size_t id_size, time_t created) {
    completion_counter++;
    snprintf(id, id_size, "chatcmpl-moonshine-%lld-%llu",
             (long long)created, completion_counter);
}

static void log_result(const request_observer *observer,
                       const k3_chat_turn_result *result,
                       bool streamed, bool client_ok) {
    const uint64_t cache_hits =
        result->cache_after.hits -
        result->cache_before.hits;
    const uint64_t cache_accesses =
        result->cache_after.accesses -
        result->cache_before.accesses;
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    if (result->state_digest_valid) {
        server_log(
            SERVER_LOG_INFO, "request.state.digest",
            observer->completion_id,
            "position=%u kda=%016llx conv=%016llx "
            "mla=%016llx attnres=%016llx",
            result->state_digest.token_position,
            (unsigned long long)result->state_digest.kda_state_hash,
            (unsigned long long)result->state_digest.kda_conv_hash,
            (unsigned long long)result->state_digest.mla_cache_hash,
            (unsigned long long)result->state_digest.attn_res_hash);
    }
    if (result->decode_stats.steps != 0u) {
        uint64_t hits = 0u;
        uint64_t accesses = 0u;
        uint64_t logical_bytes = 0u;
        uint64_t physical_bytes = 0u;
        uint64_t read_requests = 0u;
        double io_wait = 0.0;
        double expert_pipeline = 0.0;
        double layer_host_intervals = 0.0;
        uint32_t max_inflight = 0u;
        for (uint32_t i = 0u;
             i < K3_ENGINE_DECODE_LAYER_COUNT; i++) {
            const k3_engine_decode_layer_stats *layer =
                &result->decode_stats.layer[i];
            hits += layer->hits;
            accesses += layer->accesses;
            logical_bytes += layer->logical_expert_bytes;
            physical_bytes += layer->physical_read_bytes;
            read_requests += layer->read_requests;
            io_wait += layer->io_wait_seconds;
            expert_pipeline += layer->expert_pipeline_seconds;
            layer_host_intervals += layer->host_interval_seconds;
            if (layer->max_inflight > max_inflight) {
                max_inflight = layer->max_inflight;
            }
        }
        double unattributed =
            result->decode_stats.wall_seconds - layer_host_intervals;
        if (unattributed < 0.0) unattributed = 0.0;
        server_log(
            SERVER_LOG_INFO, "request.decode.io",
            observer->completion_id,
            "capture=%llu steps=%llu trace_rows=%llu "
            "cache=%llu/%llu reads=%llu logical=%.3f_GiB "
            "physical=%.3f_GiB io_wait=%.3fs "
            "expert_pipeline=%.3fs layer_host_intervals=%.3fs "
            "unattributed=%.3fs "
            "max_inflight=%u",
            (unsigned long long)result->decode_stats.capture,
            (unsigned long long)result->decode_stats.steps,
            (unsigned long long)result->decode_stats.trace_rows,
            (unsigned long long)hits,
            (unsigned long long)accesses,
            (unsigned long long)read_requests,
            (double)logical_bytes /
                (1024.0 * 1024.0 * 1024.0),
            (double)physical_bytes /
                (1024.0 * 1024.0 * 1024.0),
            io_wait, expert_pipeline, layer_host_intervals,
            unattributed,
            max_inflight);
    }
    if (result->range_stats.routed_layer_sweeps != 0u) {
        const double average_unique =
            (double)result->range_stats.unique_experts_across_layers /
            result->range_stats.routed_layer_sweeps;
        server_log(
            SERVER_LOG_INFO, "request.prefill.io",
            observer->completion_id,
            "unique_experts=%u/%.1f/%u routes=%llu read=%.3f_GiB "
            "workspace_borrow=%.3f_GiB read_wait=%.3fs "
            "expert_pipeline=%.3fs routed=%.3fs",
            result->range_stats.min_unique_experts_per_layer,
            average_unique,
            result->range_stats.max_unique_experts_per_layer,
            (unsigned long long)
                result->range_stats.selected_expert_routes,
            (double)result->range_stats.routed_physical_read_bytes /
                (1024.0 * 1024.0 * 1024.0),
            (double)result->range_stats.warm_cache_workspace_bytes /
                (1024.0 * 1024.0 * 1024.0),
            result->range_stats.routed_read_wait_seconds,
            result->range_stats.routed_expert_pipeline_seconds,
            result->range_stats.routed_stream_seconds);
    }
    server_log(
        SERVER_LOG_INFO, "request.complete",
        observer->completion_id,
        "prompt=%u evaluated=%u reused=%u prefill=%.3fs "
        "generated=%u decode=%.3fs rate=%.3f_tok/s total=%.3fs "
        "finish=%s forced_trailer=%u tool_calls=%zu "
        "reasoning_bytes=%zu content_bytes=%zu "
        "stream=%s client=%s cache=%llu/%llu cache_total=%llu/%llu",
        result->prompt_tokens,
        result->prompt_evaluated_tokens,
        result->prompt_reused_tokens,
        result->prompt_seconds,
        result->generated_tokens, result->decode_seconds,
        result->tokens_per_second,
        elapsed_seconds(observer->request_start, now),
        result->finish_reason == K3_CHAT_FINISH_TOOL_CALLS ?
            "tool_calls" :
            (result->finish_reason ==
                K3_CHAT_FINISH_END_OF_MESSAGE ? "stop" : "length"),
        result->forced_trailer_tokens,
        result->tool_call_count,
        result->reasoning_content.size,
        result->response.size,
        streamed ? "yes" : "no",
        client_ok ? "connected" : "disconnected",
        (unsigned long long)cache_hits,
        (unsigned long long)cache_accesses,
        (unsigned long long)result->cache_after.hits,
        (unsigned long long)result->cache_after.accesses);
}

static void handle_chat_completion(int fd, k3_chat_session *session,
                                   const http_request *http,
                                   const server_config *config,
                                   const char *peer) {
    char error[1024];
    const time_t created = time(NULL);
    char completion_id[128];
    make_completion_id(completion_id, sizeof(completion_id), created);
    request_observer observer = {
        .completion_id = completion_id,
        .clear_expert_cache =
            config->clear_expert_cache_per_request,
    };
    clock_gettime(CLOCK_MONOTONIC, &observer.request_start);
    k3_openai_chat_request request;
    memset(&request, 0, sizeof(request));
    if (!k3_openai_parse_chat_request(
            http->body, http->body_size,
            effective_max_output_tokens(config), &request,
            error, sizeof(error))) {
        server_log(
            SERVER_LOG_WARN, "request.reject", completion_id,
            "peer=%s status=400 error=\"%s\"",
            peer, error);
        (void)send_json_error(fd, 400, error);
        return;
    }
    if (strcmp(request.model, MOONSHINE_MODEL_ID) != 0) {
        snprintf(
            error, sizeof(error),
            "model \"%s\" is unavailable; use \"%s\"",
            request.model, MOONSHINE_MODEL_ID);
        server_log(
            SERVER_LOG_WARN, "request.reject", completion_id,
            "peer=%s status=400 error=\"%s\"",
            peer, error);
        (void)send_json_error(fd, 400, error);
        k3_openai_chat_request_free(&request);
        return;
    }

    observer.max_output_tokens = request.max_tokens;
    observer.thinking = request.thinking;
    server_log(
        SERVER_LOG_INFO, "request.start", completion_id,
        "peer=%s stream=%s messages=%zu max_output=%u reasoning=%s "
        "tools=%zu tool_choice=%s parallel_tools=%s format=%s",
        peer, request.stream ? "yes" : "no",
        request.message_count, request.max_tokens,
        request.reasoning_effort,
        request.tool_count, tool_choice_name(request.tool_choice),
        request.parallel_tool_calls ? "yes" : "no",
        response_format_name(request.response_format));
    k3_chat_turn_result result;
    memset(&result, 0, sizeof(result));
    bool ok;
    if (request.stream) {
        stream_state stream = {
            .fd = fd,
            .completion_id = completion_id,
            .created = created,
            .thinking = request.thinking,
            .defer_content = request.response_format !=
                K3_RESPONSE_FORMAT_TEXT,
        };
        observer.stream = &stream;
        if (!stream_begin(&stream)) {
            server_log(
                SERVER_LOG_WARN, "request.client_disconnect",
                completion_id,
                "peer=%s stage=stream_begin", peer);
            k3_openai_chat_request_free(&request);
            return;
        }
        const k3_chat_completion_options completion_options = {
            .reuse_prefix = true,
            .clear_expert_cache =
                config->clear_expert_cache_per_request,
            .progress_callback = observe_prefill_progress,
            .progress_data = &observer,
            .lifecycle_callback = observe_lifecycle,
            .lifecycle_data = &observer,
            .thinking = request.thinking,
            .thinking_effort = request.reasoning_effort,
            .reasoning_callback = stream_reasoning_callback,
            .reasoning_data = &stream,
            .tools_json = request.tools_json,
            .tool_choice = request.tool_choice,
            .enforce_single_tool_call =
                !request.parallel_tool_calls &&
                request.tool_count != 0u,
            .response_format = request.response_format,
            .response_schema_json = request.response_schema_json,
            .preserve_request_directive_history = true,
        };
        ok = k3_chat_session_complete_messages_with_options(
            session, request.messages, request.message_count,
            request.max_tokens, &completion_options,
            stream.defer_content ? NULL : stream_callback,
            &stream,
            &result, error, sizeof(error));
        if (ok) {
            ok = k3_openai_validate_tool_policy(
                &request, &result, error, sizeof(error));
        }
        if (ok) {
            ok = k3_openai_validate_response_format(
                &request, &result, error, sizeof(error));
        }
        if (ok) {
            stream_end(&stream, &result);
            log_result(&observer, &result, true, !stream.failed);
        } else {
            server_log(
                SERVER_LOG_ERROR, "request.failed", completion_id,
                "capture=%llu peer=%s stage=inference error=\"%s\"",
                (unsigned long long)result.decode_stats.capture,
                peer, error);
            stream_send_inference_error(&stream, error);
        }
    } else {
        const k3_chat_completion_options completion_options = {
            .reuse_prefix = true,
            .clear_expert_cache =
                config->clear_expert_cache_per_request,
            .progress_callback = observe_prefill_progress,
            .progress_data = &observer,
            .lifecycle_callback = observe_lifecycle,
            .lifecycle_data = &observer,
            .thinking = request.thinking,
            .thinking_effort = request.reasoning_effort,
            .tools_json = request.tools_json,
            .tool_choice = request.tool_choice,
            .enforce_single_tool_call =
                !request.parallel_tool_calls &&
                request.tool_count != 0u,
            .response_format = request.response_format,
            .response_schema_json = request.response_schema_json,
            .preserve_request_directive_history = true,
        };
        ok = k3_chat_session_complete_messages_with_options(
            session, request.messages, request.message_count,
            request.max_tokens, &completion_options,
            NULL, NULL,
            &result, error, sizeof(error));
        if (ok) {
            ok = k3_openai_validate_tool_policy(
                &request, &result, error, sizeof(error));
        }
        if (ok) {
            ok = k3_openai_validate_response_format(
                &request, &result, error, sizeof(error));
        }
        if (!ok) {
            server_log(
                SERVER_LOG_ERROR, "request.failed", completion_id,
                "capture=%llu peer=%s stage=inference error=\"%s\"",
                (unsigned long long)result.decode_stats.capture,
                peer, error);
            (void)send_json_error(fd, 500, error);
        } else {
            char *json = NULL;
            size_t json_size = 0u;
            ok = k3_openai_build_chat_response(
                completion_id, created, MOONSHINE_MODEL_ID,
                &result, &json, &json_size,
                error, sizeof(error));
            if (!ok) {
                server_log(
                    SERVER_LOG_ERROR, "request.failed",
                    completion_id,
                    "capture=%llu peer=%s "
                    "stage=response_build error=\"%s\"",
                    (unsigned long long)result.decode_stats.capture,
                    peer, error);
                (void)send_json_error(fd, 500, error);
            } else {
                char metrics[512];
                snprintf(
                    metrics, sizeof(metrics),
                    "X-Moonshine-Prompt-Seconds: %.6f\r\n"
                    "X-Moonshine-Prompt-Evaluated-Tokens: %u\r\n"
                    "X-Moonshine-Prompt-Reused-Tokens: %u\r\n"
                    "X-Moonshine-Decode-Seconds: %.6f\r\n"
                    "X-Moonshine-Decode-Tokens-Per-Second: %.6f\r\n",
                    result.prompt_seconds,
                    result.prompt_evaluated_tokens,
                    result.prompt_reused_tokens,
                    result.decode_seconds,
                    result.tokens_per_second);
                const bool client_ok = send_response(
                    fd, 200, "application/json",
                    json, json_size, metrics);
                log_result(&observer, &result, false, client_ok);
                free(json);
            }
        }
    }
    k3_chat_turn_result_free(&result);
    k3_openai_chat_request_free(&request);
}

static void handle_request(int fd, k3_chat_session *session,
                           const server_config *config,
                           const char *peer) {
    http_request request;
    int status = 400;
    char error[1024];
    if (!receive_request(
            fd, config->max_body, &request, &status,
            error, sizeof(error))) {
        server_log(
            SERVER_LOG_WARN, "http.reject", NULL,
            "peer=%s status=%d error=\"%s\"",
            peer, status, error);
        (void)send_json_error(fd, status, error);
        http_request_free(&request);
        return;
    }
    if (strcmp(request.path, "/health") == 0) {
        if (strcmp(request.method, "GET") != 0) {
            server_log(
                SERVER_LOG_WARN, "http.reject", NULL,
                "peer=%s status=405 method=%s path=%s",
                peer, request.method, request.path);
            (void)send_json_error(fd, 405, "method not allowed");
        } else {
            char body[512];
            const int body_size = snprintf(
                body, sizeof(body),
                "{\"status\":\"ok\",\"ready\":true,"
                "\"model\":\"" MOONSHINE_MODEL_ID "\","
                "\"engine\":\"" MOONSHINE_NAME "\","
                "\"version\":\"" MOONSHINE_VERSION "\","
                "\"expert_cache\":\"persistent\","
                "\"prefix_reuse\":\"automatic_exact_prefix\","
                "\"context_length\":%u,"
                "\"max_output_tokens\":%u,"
                "\"slots\":1}",
                config->context,
                effective_max_output_tokens(config));
            if (body_size < 0 ||
                (size_t)body_size >= sizeof(body)) {
                (void)send_json_error(
                    fd, 500, "health metadata overflow");
            } else {
                (void)send_response(
                    fd, 200, "application/json",
                    body, (size_t)body_size, NULL);
            }
        }
        http_request_free(&request);
        return;
    }
    if (!authorized(&request, config->api_key)) {
        server_log(
            SERVER_LOG_WARN, "http.unauthorized", NULL,
            "peer=%s path=%s", peer, request.path);
        (void)send_json_error(fd, 401, "invalid or missing API key");
        http_request_free(&request);
        return;
    }
    if (strcmp(request.path, "/v1/models") == 0) {
        if (strcmp(request.method, "GET") != 0) {
            server_log(
                SERVER_LOG_WARN, "http.reject", NULL,
                "peer=%s status=405 method=%s path=%s",
                peer, request.method, request.path);
            (void)send_json_error(fd, 405, "method not allowed");
        } else {
            char body[512];
            const int body_size = snprintf(
                body, sizeof(body),
                "{\"object\":\"list\",\"data\":[{"
                "\"id\":\"" MOONSHINE_MODEL_ID "\","
                "\"object\":\"model\",\"created\":0,"
                "\"owned_by\":\"local\","
                "\"context_length\":%u,"
                "\"max_output_tokens\":%u}]}",
                config->context,
                effective_max_output_tokens(config));
            if (body_size < 0 ||
                (size_t)body_size >= sizeof(body)) {
                (void)send_json_error(
                    fd, 500, "model metadata overflow");
            } else {
                (void)send_response(
                    fd, 200, "application/json",
                    body, (size_t)body_size, NULL);
            }
        }
    } else if (strcmp(
                   request.path,
                   "/v1/chat/completions") == 0) {
        if (strcmp(request.method, "POST") != 0) {
            server_log(
                SERVER_LOG_WARN, "http.reject", NULL,
                "peer=%s status=405 method=%s path=%s",
                peer, request.method, request.path);
            (void)send_json_error(fd, 405, "method not allowed");
        } else {
            handle_chat_completion(
                fd, session, &request, config, peer);
        }
    } else {
        server_log(
            SERVER_LOG_WARN, "http.reject", NULL,
            "peer=%s status=404 method=%s path=%s",
            peer, request.method, request.path);
        (void)send_json_error(fd, 404, "endpoint not found");
    }
    http_request_free(&request);
}

int main(int argc, char **argv) {
    interactive_log =
        isatty(STDERR_FILENO) && getenv("NO_COLOR") == NULL;
    server_config config;
    if (!parse_args(argc, argv, &config)) {
        usage(stderr, argv[0]);
        return 2;
    }
    if (!loopback_host(config.host) && config.api_key == NULL) {
        server_log(
            SERVER_LOG_ERROR, "server.config.reject", NULL,
            "error=\"refusing non-loopback bind without --api-key "
            "or MOONSHINE_API_KEY\"");
        return 2;
    }
    signal(SIGPIPE, SIG_IGN);
    const struct sigaction stop_action = {
        .sa_handler = stop_handler,
    };
    (void)sigaction(SIGINT, &stop_action, NULL);
    (void)sigaction(SIGTERM, &stop_action, NULL);

    char error[1024];
    const int listener =
        create_listener(&config, error, sizeof(error));
    if (listener < 0) {
        server_log(
            SERVER_LOG_ERROR, "server.listen.failed", NULL,
            "error=\"%s\"", error);
        return 1;
    }
    active_listener = listener;
    server_log(
        SERVER_LOG_INFO, "server.load.start", NULL,
        "model_root=%s context=%u experts=%u staging=%u q8=yes "
        "range_backend=%s",
        config.model_root, config.context,
        config.experts, config.staging,
        config.range_backend == K3_PREFILL_PROJECTION_DEFAULT ?
            "default" : "kda-blas");
    k3_chat_session *session = NULL;
    k3_engine_stats stats;
    memset(&stats, 0, sizeof(stats));
    const k3_chat_session_config session_config = {
        .model_root = config.model_root,
        .context = config.context,
        .sequential_prefill_limit = config.sequential_limit,
        .experts_per_layer = config.experts,
        .staging_slots = config.staging,
        .q8_projections = true,
        .range_backend = config.range_backend,
        .decode_diagnostics_prefix =
            config.decode_diagnostics_prefix,
        .capture_state_digest = config.capture_state_digest,
    };
    if (!k3_chat_session_create(
            &session, &session_config, &stats,
            error, sizeof(error))) {
        server_log(
            SERVER_LOG_ERROR, "server.load.failed", NULL,
            "error=\"%s\"", error);
        close(listener);
        return 1;
    }
    server_log(
        SERVER_LOG_INFO, "server.ready", NULL,
        "listen=http://%s:%u model=%s version=%s context=%u "
        "max_output=%u load=%.3fs static=%.3f_GiB cache=%.3f_GiB "
        "state=%.3f_GiB slots=1 auth=%s range_backend=%s",
        config.host, config.port, MOONSHINE_MODEL_ID,
        MOONSHINE_VERSION, config.context,
        effective_max_output_tokens(&config),
        stats.startup_seconds,
        (double)stats.static_store.resident_bytes /
            (1024.0 * 1024.0 * 1024.0),
        (double)stats.cache_bytes / (1024.0 * 1024.0 * 1024.0),
        (double)stats.state_bytes / (1024.0 * 1024.0 * 1024.0),
        config.api_key == NULL ? "off" : "on",
        config.range_backend == K3_PREFILL_PROJECTION_DEFAULT ?
            "default" : "kda-blas");

    while (!stop_requested) {
        struct sockaddr_storage peer_address;
        socklen_t peer_size = sizeof(peer_address);
        const int client = accept(
            listener, (struct sockaddr *)&peer_address, &peer_size);
        if (client < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (!stop_requested) {
                server_log(
                    SERVER_LOG_ERROR, "server.accept.failed", NULL,
                    "error=\"%s\"", strerror(errno));
            }
            break;
        }
        char peer[NI_MAXHOST] = "unknown";
        (void)getnameinfo(
            (struct sockaddr *)&peer_address, peer_size,
            peer, sizeof(peer), NULL, 0, NI_NUMERICHOST);
        handle_request(client, session, &config, peer);
        close(client);
    }
    server_log(SERVER_LOG_INFO, "server.stop", NULL,
               "reason=signal_or_listener_close");
    k3_chat_session_destroy(session);
    if (active_listener >= 0) {
        close(listener);
        active_listener = -1;
    }
    return 0;
}
