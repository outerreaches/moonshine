/*
 * An OpenAI-shaped text API over the MiMo GPU worker.
 *
 * Separate from k3_server rather than a fork of it: that file is 2,600 lines
 * wired to K3's engine, tokenizer and options, and the GLM lane's serving
 * path has to keep working untouched. What is shared is taken -- the JSON
 * parser, the slot discipline's shape, the local-only default.
 *
 * Scope is the plan's M5 text API and nothing beyond it. Tools, images,
 * audio and video are refused explicitly with a reason rather than ignored,
 * because a request that silently drops its tools gets a confidently wrong
 * answer instead of an error.
 *
 * One slot. One GPU, one worker, one KV cache holding one conversation, so a
 * second concurrent request has nowhere to run and is refused immediately
 * rather than queued behind a deadline the caller cannot see.
 *
 *   tools/mimo26_server ROOT [--port N] [--host H] [--slots N] [--context N]
 *       [--prefill-chunk N] [--expert-lookahead on|off]
 */
#include "k3_json.h"
#include "mimo26_gpu_worker.h"
#include "mimo26_server_options.h"
#include "mimo26_server_slot.h"
#include "mimo26_tokenizer.h"
#include <initializer_list>

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define MODEL_ID "mimo-v2.6-flash"
#define MAX_REQUEST_BYTES (1u << 20)      /* 1 MiB; a text turn is far less */
#define MAX_MESSAGES 64u
#define DEFAULT_MAX_TOKENS 512u
#define DEFAULT_DEADLINE_SECONDS 600.0

static volatile sig_atomic_t g_shutdown = 0;

static void on_signal(int signal_number)
{
    (void)signal_number;
    g_shutdown = 1;
}

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* ---- transport ---- */

#ifndef MIMO26_HTTP_TIMEOUT_SECONDS
#define MIMO26_HTTP_TIMEOUT_SECONDS 5.0
#endif

/* Absolute deadline: drip-fed bytes cannot renew the request budget.
 * Nonblocking calls also avoid a readiness race turning into an unbounded I/O. */
static bool transport_wait(int fd, short events, double deadline)
{
    while (!g_shutdown) {
        double remaining = deadline - now_seconds();
        if (remaining <= 0) { errno = ETIMEDOUT; return false; }
        int milliseconds = remaining < .1 ? (int)(remaining * 1000) + 1 : 100;
        struct pollfd waiting{fd, events, 0};
        int result = poll(&waiting, 1, milliseconds);
        if (result < 0) { if (errno == EINTR) continue; return false; }
        if (result && (waiting.revents & (events | POLLHUP | POLLERR))) return true;
        if (result && (waiting.revents & POLLNVAL)) return false;
    }
    errno = ECANCELED;
    return false;
}

static ssize_t transport_recv(int fd, void *data, size_t bytes, double deadline)
{
    while (transport_wait(fd, POLLIN, deadline)) {
        ssize_t got = recv(fd, data, bytes, MSG_DONTWAIT);
        if (got < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        return got;
    }
    return -1;
}

static bool send_all_until(int fd, const void *data, size_t size, double deadline)
{
    const char *cursor = (const char *)data;
    while (size > 0) {
        if (!transport_wait(fd, POLLOUT, deadline)) return false;
        const ssize_t written = send(fd, cursor, size, MSG_NOSIGNAL | MSG_DONTWAIT);
        if (written <= 0) {
            if (written < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
                continue;
            }
            return false;
        }
        cursor += written;
        size -= (size_t)written;
    }
    return true;
}

static bool send_all(int fd, const void *data, size_t size)
{
    return send_all_until(fd, data, size, now_seconds() + MIMO26_HTTP_TIMEOUT_SECONDS);
}

static bool send_response(int fd, int status, const char *reason,
                          const char *content_type, const char *body,
                          size_t body_size)
{
    char header[512];
    const int header_size = snprintf(
        header, sizeof header,
        "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
        "Cache-Control: no-store\r\nConnection: close\r\n\r\n",
        status, reason, content_type, body_size);
    return header_size > 0 &&
           send_all(fd, header, (size_t)header_size) &&
           (body_size == 0 || send_all(fd, body, body_size));
}

/*
 * Errors carry a machine-readable `code` as well as a message, because a
 * caller needs to distinguish "retry in a moment" from "this request will
 * never work" without parsing prose.
 */
static void send_error(int fd, int status, const char *reason,
                       const char *code, const char *message)
{
    /* k3_json_escape emits the surrounding quotes itself, so the format
     * below must not add its own -- doing that shipped `""Paris""` in the
     * first response this server ever produced. */
    char *escaped = NULL;
    size_t escaped_size = 0;
    char error[256];
    if (!k3_json_escape(message, strlen(message), &escaped, &escaped_size,
                        error, sizeof error)) {
        escaped = NULL;
    }
    char body[1024];
    const int size = snprintf(
        body, sizeof body,
        "{\"error\":{\"type\":\"%s\",\"code\":\"%s\",\"message\":%s}}",
        status >= 500 ? "server_error" : "invalid_request_error", code,
        escaped != NULL ? escaped : "\"request failed\"");
    free(escaped);
    if (size > 0 && (size_t)size < sizeof body) {
        send_response(fd, status, reason, "application/json", body,
                      (size_t)size);
    }
}

/* Has the peer gone away? Checked between tokens so a vanished client stops
 * costing GPU time. */
static bool peer_disconnected(int fd)
{
    struct pollfd probe;
    probe.fd = fd;
    probe.events = POLLRDHUP | POLLERR | POLLHUP;
    probe.revents = 0;
    if (poll(&probe, 1u, 0) <= 0) {
        return false;
    }
    return (probe.revents & (POLLRDHUP | POLLERR | POLLHUP)) != 0;
}

/* ---- request ---- */

typedef struct {
    char   *method;
    char   *path;
    char   *body;
    size_t  body_size;
} http_request;

static void request_free(http_request *request)
{
    free(request->method);
    free(request->path);
    free(request->body);
    memset(request, 0, sizeof *request);
}

static bool read_request(int fd, http_request *request, char *error,
                         size_t error_size)
{
    memset(request, 0, sizeof *request);
    char *buffer = (char *)malloc(MAX_REQUEST_BYTES + 1u);
    if (buffer == NULL) {
        snprintf(error, error_size, "out of memory");
        return false;
    }
    size_t used = 0;
    size_t header_end = 0;
    const double deadline = now_seconds() + MIMO26_HTTP_TIMEOUT_SECONDS;
    /* Headers first, bounded. */
    while (header_end == 0) {
        if (used >= 16384u) {
            free(buffer);
            snprintf(error, error_size, "request headers are too large");
            return false;
        }
        const ssize_t got = transport_recv(fd, buffer + used, 16384u - used, deadline);
        if (got <= 0) {
            free(buffer);
            snprintf(error, error_size, "connection closed while reading");
            return false;
        }
        used += (size_t)got;
        buffer[used] = '\0';
        const char *marker = strstr(buffer, "\r\n\r\n");
        if (marker != NULL) {
            header_end = (size_t)(marker - buffer) + 4u;
        }
    }

    /* One strict HTTP/1.x message; no transfer coding or pipelining. */
    auto reject = [&](const char *message) {
        free(buffer); request_free(request);
        snprintf(error, error_size, "%s", message); return false;
    };
    if (memchr(buffer, '\0', header_end)) return reject("NUL in headers");
    const char *line_end = strstr(buffer, "\r\n");
    if (!line_end) return reject("missing request line");
    for (size_t i=0;i<header_end;++i) {
        unsigned char c=(unsigned char)buffer[i];
        if ((c<32 && c!='\r' && c!='\n' && c!='\t') || c==127)
            return reject("control character in headers");
        if ((c=='\n' && (!i || buffer[i-1]!='\r')) ||
            (c=='\r' && (i+1>=header_end || buffer[i+1]!='\n')))
            return reject("invalid header line ending");
    }
    /* Request line. */
    const char *space = (const char *)memchr(buffer, ' ', header_end);
    if (space == NULL) {
        free(buffer);
        snprintf(error, error_size, "malformed request line");
        return false;
    }
    request->method = strndup(buffer, (size_t)(space - buffer));
    const char *path_start = space + 1;
    const char *path_end = (const char *)memchr(path_start, ' ',
                                                header_end -
                                                    (size_t)(path_start -
                                                             buffer));
    if (path_end == NULL) {
        free(buffer);
        request_free(request);
        snprintf(error, error_size, "malformed request line");
        return false;
    }
    request->path = strndup(path_start, (size_t)(path_end - path_start));
    if (!request->method || !request->path) return reject("out of memory");
    if (space>=line_end || path_end>=line_end || space==buffer || path_end==path_start ||
        path_start[0]!='/' || (size_t)(line_end-path_end)!=9 ||
        (memcmp(path_end+1,"HTTP/1.1",8) && memcmp(path_end+1,"HTTP/1.0",8)))
        return reject("invalid request line");
    for (const char *p=buffer;p<space;++p) if (*p<'A' || *p>'Z') return reject("invalid method");
    for (const char *p=path_start;p<path_end;++p) if ((unsigned char)*p<=32 || (unsigned char)*p==127) return reject("invalid target");

    size_t content_length = 0;
    bool has_length=false;
    for (const char *line=line_end+2; line<buffer+header_end-2;) {
        const char *end=strstr(line,"\r\n");
        if (!end || end==line || *line==' ' || *line=='\t') return reject("invalid header");
        const char *colon=(const char*)memchr(line,':',end-line);
        if (!colon || colon==line) return reject("invalid header name");
        for (const char *p=line;p<colon;++p) {
            unsigned char c=(unsigned char)*p;
            if (!((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||strchr("!#$%&'*+-.^_`|~",c)))
                return reject("invalid header name");
        }
        size_t name=colon-line;
        if ((name==17 && !strncasecmp(line,"Transfer-Encoding",name)) ||
            (name==6 && !strncasecmp(line,"Expect",name))) return reject("unsupported transfer coding or expectation");
        if (name==14 && !strncasecmp(line,"Content-Length",name)) {
            if (has_length) return reject("duplicate Content-Length");
            has_length=true;
            const char *p=colon+1,*last=end;
            while(p<last && (*p==' '||*p=='\t'))++p;
            while(last>p && (last[-1]==' '||last[-1]=='\t'))--last;
            if(p==last)return reject("empty Content-Length");
            for(;p<last;++p) {
                if(*p<'0'||*p>'9'||content_length>(MAX_REQUEST_BYTES-(size_t)(*p-'0'))/10)
                    return reject("invalid Content-Length");
                content_length=content_length*10+(size_t)(*p-'0');
            }
        }
        line=end+2;
    }
    if (!strcmp(request->method,"POST") && !has_length) return reject("POST requires Content-Length");
    if (used-header_end>content_length) return reject("unexpected bytes beyond request body");
    if (content_length > MAX_REQUEST_BYTES - header_end) {
        free(buffer);
        request_free(request);
        snprintf(error, error_size, "request body exceeds %u bytes",
                 MAX_REQUEST_BYTES);
        return false;
    }
    while (used - header_end < content_length) {
        const ssize_t got = transport_recv(fd, buffer + used,
                                 content_length - (used-header_end), deadline);
        if (got <= 0) {
            free(buffer);
            request_free(request);
            snprintf(error, error_size, "connection closed reading the body");
            return false;
        }
        used += (size_t)got;
    }
    request->body = (char *)malloc(content_length + 1u);
    if (request->body == NULL) {
        free(buffer);
        request_free(request);
        snprintf(error, error_size, "out of memory");
        return false;
    }
    memcpy(request->body, buffer + header_end, content_length);
    request->body[content_length] = '\0';
    request->body_size = content_length;
    free(buffer);
    return true;
}

/* Grow-and-append for assembling a JSON fragment. */
static bool append_json(char **buffer, size_t *used, size_t *capacity,
                        const char *text)
{
    const size_t length = strlen(text);
    if (*used + length + 1u > *capacity) {
        size_t grown = *capacity ? *capacity : 256u;
        while (*used + length + 1u > grown) {
            grown *= 2u;
        }
        char *next = (char *)realloc(*buffer, grown);
        if (next == NULL) {
            return false;
        }
        *buffer = next;
        *capacity = grown;
    }
    memcpy(*buffer + *used, text, length);
    *used += length;
    (*buffer)[*used] = '\0';
    return true;
}

/* ---- runtime ---- */

typedef struct {
    int                listener;
    mimo26_gpu_worker *worker;
    mimo26_tokenizer  *tokenizer;
    mimo26_slot        slot;
    const char        *model_root;
    size_t             context_capacity;
    uint16_t           prefill_chunk;
    uint16_t           expert_slots_per_layer;
    bool               expert_lookahead;
    bool               retain_experts;
    uint64_t           served;
    /* Consecutive supervised restarts that did not lead to a clean request.
     * Bounded so a persistently broken worker stops thrashing and stays
     * degraded for an operator to look at, rather than resetting forever. */
    unsigned           recovery_attempts;
} server_runtime;

/* Healthy request boundaries only. Never fall back to cold reset after a
 * retention refusal: pending I/O or a sticky execution fault requires a new
 * worker, even if the serving slot had appeared idle. */
static bool reset_request_worker(server_runtime *runtime, char *error, size_t size)
{
    if (!runtime->retain_experts) {
        mimo26_gpu_worker_reset(runtime->worker);
        return true;
    }
    if (mimo26_gpu_worker_reset_context(runtime->worker, error, size) != MIMO26_GPU_WORKER_OK) {
        mimo26_slot_fault(&runtime->slot);
        return false;
    }
    return true;
}

#define MAX_RECOVERY_ATTEMPTS 3u

/*
 * Supervised restart after a fault.
 *
 * Quarantine without a way out means one failed decode takes the server down
 * until someone notices. A metadata reset is not recovery from failed I/O:
 * registered reads may still be outstanding and copied payloads may disagree
 * with the old cache mappings. Until supervised worker/process recreation is
 * qualified, execution faults must stay quarantined. Only an already healthy
 * worker (e.g. after a request-side failure) can take the cold-reset path.
 *
 * Bounded on purpose. If resets keep being followed by faults the problem is
 * not transient, and continuing to reset would hide a hardware or checkpoint
 * problem behind an endless retry loop.
 */
static bool attempt_recovery(server_runtime *runtime)
{
    if (runtime->slot.phase != MIMO26_SLOT_QUARANTINED) {
        return true;
    }
    if (runtime->recovery_attempts >= MAX_RECOVERY_ATTEMPTS) {
        return false;
    }
    runtime->recovery_attempts++;
    char error[256]{};
    if (mimo26_gpu_worker_reset_context(runtime->worker, error, sizeof error)
            != MIMO26_GPU_WORKER_OK) {
        fprintf(stderr, "mimo26: recovery refused; worker recreation required: %s\n", error);
        return false;
    }
    fprintf(stderr,
            "mimo26: supervised restart %u of %u after a worker fault\n",
            runtime->recovery_attempts, MAX_RECOVERY_ATTEMPTS);
    mimo26_gpu_worker_reset(runtime->worker);
    return mimo26_slot_recover(&runtime->slot);
}

static void send_health(int fd, server_runtime *runtime)
{
    mimo26_gpu_worker_stats stats;
    mimo26_gpu_worker_get_stats(runtime->worker, &stats);
    const mimo26_slot *slot = &runtime->slot;
    char body[1024];
    /*
     * `ready` is the liveness answer a load balancer needs: healthy AND able
     * to start work now. A quarantined or draining worker reports not ready
     * while still answering this endpoint, which is the distinction that
     * makes the check useful.
     */
    const int size = snprintf(
        body, sizeof body,
        "{\"status\":\"%s\",\"ready\":%s,\"model\":\"%s\","
        "\"phase\":\"%s\",\"context\":%zu,\"prefill_chunk\":%u,\"expert_lookahead\":%s,\"expert_slots\":%u,"
        "\"retain_experts\":%s,\"served\":%llu,\"tokens\":%llu,"
        "\"expert_accesses\":%llu,\"expert_hits\":%llu,\"expert_uploads\":%llu,"
        "\"expert_hit_rate\":%.4f,\"resident_gib\":%.2f,"
        "\"admitted\":%llu,\"rejected_busy\":%llu,"
        "\"rejected_quarantined\":%llu,\"cancelled\":%llu,"
        "\"deadline_stops\":%llu,\"faults\":%llu,\"recoveries\":%llu}",
        slot->phase == MIMO26_SLOT_QUARANTINED ? "degraded" : "ok",
        mimo26_slot_ready(slot) ? "true" : "false", MODEL_ID,
        mimo26_slot_phase_name(slot->phase), runtime->context_capacity,
        (unsigned)runtime->prefill_chunk, runtime->expert_lookahead ? "true" : "false",
        (unsigned)runtime->expert_slots_per_layer,
        runtime->retain_experts ? "true" : "false",
        (unsigned long long)runtime->served,
        (unsigned long long)stats.tokens,
        (unsigned long long)stats.expert_accesses,
        (unsigned long long)stats.expert_hits,
        (unsigned long long)stats.expert_uploads,
        stats.expert_accesses
            ? (double)stats.expert_hits / (double)stats.expert_accesses
            : 0.0,
        (double)mimo26_gpu_worker_resident_bytes(runtime->worker) /
            1073741824.0,
        (unsigned long long)slot->admitted,
        (unsigned long long)slot->rejected_busy,
        (unsigned long long)slot->rejected_quarantined,
        (unsigned long long)slot->cancelled,
        (unsigned long long)slot->deadline_stops,
        (unsigned long long)slot->faults,
        (unsigned long long)slot->recoveries);
    if (size > 0) {
        send_response(fd, 200, "OK", "application/json", body, (size_t)size);
    }
}

/*
 * Refuse anything that arrives while the slot is occupied.
 *
 * Without this the accept loop leaves a second connection in the backlog
 * until the first finishes, so the caller waits with no signal -- which is
 * precisely what the slot discipline exists to prevent. Called between
 * tokens, so a client learns it was refused in roughly one token's time
 * rather than after a whole generation.
 *
 * The request is not read before refusing. A 503 with Connection: close is
 * a complete answer, and reading an unbounded body from a connection we are
 * about to reject would let a caller hold the loop open.
 */
static void refuse_backlog(server_runtime *runtime)
{
    /* One bounded batch per safe boundary; a flood must not starve the
     * requesting client, shutdown, deadlines or GPU progress. All busy writes
     * share this budget, rather than getting five seconds each. */
    const double deadline = now_seconds() + .1;
    for (unsigned count = 0; count < 8 && !g_shutdown && now_seconds() < deadline; ++count) {
        struct pollfd waiting;
        waiting.fd = runtime->listener;
        waiting.events = POLLIN;
        waiting.revents = 0;
        if (poll(&waiting, 1u, 0) <= 0 || !(waiting.revents & POLLIN)) {
            return;
        }
        const int client = accept(runtime->listener, NULL, NULL);
        if (client < 0) {
            return;
        }
        mimo26_slot_count_rejection(&runtime->slot, MIMO26_SLOT_REJECT_BUSY);
        static const char body[] = "{\"error\":{\"type\":\"server_error\","
            "\"code\":\"slot_busy\",\"message\":\"another request is using the single "
            "execution slot; retry shortly\"}}";
        char wire[512];
        int length = snprintf(wire, sizeof wire,
            "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n"
            "Content-Length: %zu\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n%s",
            sizeof body - 1, body);
        if (length > 0 && (size_t)length < sizeof wire)
            send_all_until(client, wire, (size_t)length, deadline);
        close(client);
    }
}

static void send_models(int fd)
{
    char body[512];
    const int size = snprintf(
        body, sizeof body,
        "{\"object\":\"list\",\"data\":[{\"id\":\"%s\",\"object\":\"model\","
        "\"owned_by\":\"moonshine\"}]}", MODEL_ID);
    if (size > 0) {
        send_response(fd, 200, "OK", "application/json", body, (size_t)size);
    }
}

/* ---- chat completions ---- */

typedef struct {
    mimo26_chat_message messages[MAX_MESSAGES];
    char   *owned[MAX_MESSAGES * 3u];
    size_t  owned_count;
    size_t  message_count;
    uint32_t max_tokens;
    bool     stream;
    bool     enable_thinking;
    char    *tools_json;
    /* Storage for assistant turns replayed with their tool calls. */
    mimo26_tool_call calls[MAX_MESSAGES][8];
} chat_request;

static void chat_request_free(chat_request *request)
{
    for (size_t i = 0; i < request->owned_count; i++) {
        free(request->owned[i]);
    }
    memset(request, 0, sizeof *request);
}

static char *take(chat_request *request, char *text)
{
    if (text != NULL && request->owned_count < MAX_MESSAGES * 3u) {
        request->owned[request->owned_count++] = text;
    }
    return text;
}

/*
 * Parse strictly. Anything the worker cannot honour is refused with a reason
 * rather than dropped, including sampling controls: this worker is greedy,
 * so accepting a temperature and ignoring it would make a caller believe
 * they had varied the output when they had not.
 */
static bool parse_chat(const char *body, size_t body_size,
                       chat_request *request, char *code, size_t code_size,
                       char *error, size_t error_size)
{
    memset(request, 0, sizeof *request);
    request->max_tokens = DEFAULT_MAX_TOKENS;
    request->enable_thinking = true;
    snprintf(code, code_size, "invalid_request");

    k3_json_document document;
    memset(&document, 0, sizeof document);
    if (!k3_json_parse(&document, body, body_size, error, error_size)) {
        return false;
    }
    #define REFUSE(reason_code, message)                                      \
        do {                                                                  \
            snprintf(code, code_size, "%s", (reason_code));                   \
            snprintf(error, error_size, "%s", (message));                     \
            k3_json_document_free(&document);                                 \
            chat_request_free(request);                                       \
            return false;                                                     \
        } while (0)

    const int32_t root = document.root;
    if (root < 0 || document.tokens[root].type != K3_JSON_OBJECT) {
        REFUSE("invalid_request", "body must be a JSON object");
    }
    if (k3_json_object_get(&document, root, "functions") >= 0) {
        REFUSE("option_unsupported",
               "the legacy 'functions' form is not supported; use 'tools'");
    }
    {
        /*
         * tool_choice is refused rather than ignored. The template offers no
         * way to force or forbid a call, so honouring "required" or a named
         * function would mean pretending -- and a caller who asked for a
         * forced call and got prose has no way to tell that from the model
         * declining.
         */
        const int32_t choice = k3_json_object_get(&document, root,
                                                  "tool_choice");
        if (choice >= 0 && !k3_json_string_equal(&document, choice, "auto")) {
            REFUSE("option_unsupported",
                   "only tool_choice 'auto' is supported; this template "
                   "cannot force or forbid a call");
        }
    }
    {
        /* Kept as raw source so the renderer serializes it the way the
         * template's tojson does, rather than however the client spaced it. */
        const int32_t tools = k3_json_object_get(&document, root, "tools");
        if (tools >= 0) {
            if (document.tokens[tools].type != K3_JSON_ARRAY) {
                REFUSE("invalid_request", "tools must be an array");
            }
            const size_t length = document.tokens[tools].end -
                                  document.tokens[tools].start;
            char *raw = (char *)malloc(length + 1u);
            if (raw == NULL) {
                REFUSE("invalid_request", "tools are too large");
            }
            memcpy(raw, body + document.tokens[tools].start, length);
            raw[length] = '\0';
            request->tools_json = take(request, raw);
        }
    }
    for (const char *unsupported : {"temperature", "top_p", "top_k", "n",
                                    "presence_penalty", "frequency_penalty",
                                    "logit_bias", "seed"}) {
        if (k3_json_object_get(&document, root, unsupported) >= 0) {
            char message[192];
            snprintf(message, sizeof message,
                     "'%s' is not supported: this backend decodes greedily, "
                     "and accepting the option would imply otherwise",
                     unsupported);
            REFUSE("option_unsupported", message);
        }
    }

    const int32_t model = k3_json_object_get(&document, root, "model");
    if (model >= 0 && !k3_json_string_equal(&document, model, MODEL_ID)) {
        REFUSE("model_not_found", "unknown model; this server serves "
               MODEL_ID);
    }
    const int32_t stream = k3_json_object_get(&document, root, "stream");
    if (stream >= 0) {
        k3_json_bool(&document, stream, &request->stream);
    }
    bool thinking_seen = false;
    const int32_t thinking = k3_json_object_get(&document, root,
                                                "enable_thinking");
    if (thinking >= 0) {
        if (!k3_json_bool(&document, thinking, &request->enable_thinking)) {
            REFUSE("invalid_request", "enable_thinking must be a boolean");
        }
        thinking_seen = true;
    }
    {
        /*
         * OpenAI-compatible clients -- llama.cpp and vLLM among them -- pass
         * template arguments nested under chat_template_kwargs rather than at
         * the top level, so dropping the object silently meant a caller who
         * asked for no thinking got thinking and nothing said so. The
         * capability was already there; only this spelling of it was missing.
         *
         * Keys inside are refused exactly as unsupported top-level options
         * are, because the same reasoning applies one level down: a misspelled
         * kwarg that quietly changes nothing is the failure this server
         * refuses 'temperature' to avoid.
         */
        const int32_t kwargs = k3_json_object_get(&document, root,
                                                  "chat_template_kwargs");
        if (kwargs >= 0) {
            if (document.tokens[kwargs].type != K3_JSON_OBJECT) {
                REFUSE("invalid_request",
                       "chat_template_kwargs must be an object");
            }
            for (int32_t key = document.tokens[kwargs].first_child; key >= 0;) {
                const int32_t value = document.tokens[key].next_sibling;
                if (value < 0) {
                    REFUSE("invalid_request",
                           "malformed chat_template_kwargs");
                }
                if (!k3_json_string_equal(&document, key, "enable_thinking")) {
                    /* Name the key back to the caller: the whole point is
                     * that a typo must not read as success. */
                    char name[64];
                    const size_t start = document.tokens[key].start;
                    size_t length = document.tokens[key].end - start;
                    if (length >= sizeof name) {
                        length = sizeof name - 1u;
                    }
                    memcpy(name, body + start, length);
                    name[length] = '\0';
                    char message[224];
                    snprintf(message, sizeof message,
                             "chat_template_kwargs '%s' is not supported; this "
                             "template accepts only 'enable_thinking'", name);
                    REFUSE("option_unsupported", message);
                }
                bool nested = true;
                if (!k3_json_bool(&document, value, &nested)) {
                    REFUSE("invalid_request",
                           "chat_template_kwargs.enable_thinking must be a "
                           "boolean");
                }
                /* Both spellings given and disagreeing: refuse rather than
                 * pick, since either choice silently discards what the caller
                 * asked for in the other field. */
                if (thinking_seen && nested != request->enable_thinking) {
                    REFUSE("invalid_request",
                           "enable_thinking and "
                           "chat_template_kwargs.enable_thinking disagree");
                }
                request->enable_thinking = nested;
                thinking_seen = true;
                key = document.tokens[value].next_sibling;
            }
        }
    }
    const int32_t max_tokens = k3_json_object_get(&document, root,
                                                  "max_tokens");
    if (max_tokens >= 0) {
        uint32_t value = 0;
        if (!k3_json_u32(&document, max_tokens, &value) || value == 0u) {
            REFUSE("invalid_request", "max_tokens must be a positive integer");
        }
        request->max_tokens = value;
    }

    const int32_t messages = k3_json_object_get(&document, root, "messages");
    if (messages < 0 || document.tokens[messages].type != K3_JSON_ARRAY) {
        REFUSE("invalid_request", "messages must be an array");
    }
    for (int32_t m = document.tokens[messages].first_child; m >= 0;
         m = document.tokens[m].next_sibling) {
        if (request->message_count >= MAX_MESSAGES) {
            REFUSE("too_many_messages", "conversation exceeds the message "
                   "limit for this backend");
        }
        const int32_t role = k3_json_object_get(&document, m, "role");
        const int32_t content = k3_json_object_get(&document, m, "content");
        if (role < 0 || content < 0) {
            REFUSE("invalid_request", "each message needs a role and content");
        }
        if (!k3_json_string_equal(&document, role, "system") &&
            !k3_json_string_equal(&document, role, "user") &&
            !k3_json_string_equal(&document, role, "assistant") &&
            !k3_json_string_equal(&document, role, "tool")) {
            REFUSE("role_unsupported",
                   "role must be system, user, assistant or tool");
        }
        char *role_text = NULL;
        if (!k3_json_string_dup(&document, role, &role_text, error,
                                error_size)) {
            REFUSE("invalid_request", "role must be a string");
        }
        take(request, role_text);

        char *content_text = NULL;
        if (document.tokens[content].type == K3_JSON_ARRAY) {
            /* Content parts: text is concatenated, anything else refused.
             * A dropped image part would silently change the question. */
            size_t used = 0, capacity = 1024u;
            content_text = (char *)calloc(1u, capacity);
            for (int32_t part = document.tokens[content].first_child;
                 part >= 0 && content_text != NULL;
                 part = document.tokens[part].next_sibling) {
                const int32_t type = k3_json_object_get(&document, part,
                                                        "type");
                if (type < 0 || !k3_json_string_equal(&document, type,
                                                      "text")) {
                    free(content_text);
                    REFUSE("modality_unsupported",
                           "only text content is supported; image, audio and "
                           "video parts are refused");
                }
                const int32_t text_field = k3_json_object_get(&document, part,
                                                              "text");
                char *piece = NULL;
                if (text_field < 0 ||
                    !k3_json_string_dup(&document, text_field, &piece, error,
                                        error_size)) {
                    free(content_text);
                    REFUSE("invalid_request", "a text part has no text");
                }
                const size_t length = strlen(piece);
                if (used + length + 1u > capacity) {
                    capacity = (used + length + 1u) * 2u;
                    char *grown = (char *)realloc(content_text, capacity);
                    if (grown == NULL) {
                        free(piece);
                        free(content_text);
                        REFUSE("invalid_request", "content is too large");
                    }
                    content_text = grown;
                }
                memcpy(content_text + used, piece, length);
                used += length;
                content_text[used] = '\0';
                free(piece);
            }
        } else if (!k3_json_string_dup(&document, content, &content_text,
                                       error, error_size)) {
            REFUSE("invalid_request", "content must be a string or an array "
                   "of text parts");
        }
        take(request, content_text);

        char *reasoning_text = NULL;
        const int32_t reasoning = k3_json_object_get(&document, m,
                                                     "reasoning_content");
        if (reasoning >= 0) {
            k3_json_string_dup(&document, reasoning, &reasoning_text, error,
                               error_size);
            take(request, reasoning_text);
        }

        request->messages[request->message_count].role = role_text;
        request->messages[request->message_count].content = content_text;
        request->messages[request->message_count].reasoning = reasoning_text;

        /* Replaying an assistant turn that made calls is how a caller
         * continues a tool conversation; dropping them would make the model
         * see a turn where it said nothing and call again. */
        const int32_t tool_calls = k3_json_object_get(&document, m,
                                                      "tool_calls");
        if (tool_calls >= 0) {
            size_t made = 0;
            for (int32_t call = document.tokens[tool_calls].first_child;
                 call >= 0 && made < 8u;
                 call = document.tokens[call].next_sibling) {
                const int32_t function = k3_json_object_get(&document, call,
                                                            "function");
                if (function < 0) {
                    REFUSE("invalid_request",
                           "a tool call has no function object");
                }
                const int32_t name_field = k3_json_object_get(&document,
                                                              function,
                                                              "name");
                const int32_t args = k3_json_object_get(&document, function,
                                                        "arguments");
                char *call_name = NULL;
                if (name_field < 0 ||
                    !k3_json_string_dup(&document, name_field, &call_name,
                                        error, error_size)) {
                    REFUSE("invalid_request", "a tool call has no name");
                }
                take(request, call_name);
                char *call_args = NULL;
                if (args >= 0) {
                    /* OpenAI sends arguments as a JSON string; the template
                     * wants the object. Both spellings are accepted. */
                    if (document.tokens[args].type == K3_JSON_STRING) {
                        k3_json_string_dup(&document, args, &call_args, error,
                                           error_size);
                    } else {
                        const size_t length = document.tokens[args].end -
                                              document.tokens[args].start;
                        call_args = (char *)malloc(length + 1u);
                        if (call_args != NULL) {
                            memcpy(call_args, body +
                                   document.tokens[args].start, length);
                            call_args[length] = '\0';
                        }
                    }
                    take(request, call_args);
                }
                request->calls[request->message_count][made].name = call_name;
                request->calls[request->message_count][made].arguments_json =
                    call_args;
                made++;
            }
            request->messages[request->message_count].tool_calls =
                request->calls[request->message_count];
            request->messages[request->message_count].tool_call_count = made;
        }
        request->message_count++;
    }
    #undef REFUSE
    if (request->message_count == 0u) {
        snprintf(code, code_size, "invalid_request");
        snprintf(error, error_size, "messages must not be empty");
        k3_json_document_free(&document);
        return false;
    }
    k3_json_document_free(&document);
    return true;
}

typedef struct {
    int    fd;
    bool   streaming;
    bool   headers_sent;
    char   id[64];
    long   created;
} response_state;

static bool stream_begin(response_state *state)
{
    const char *header =
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
        "Cache-Control: no-store\r\nConnection: close\r\n\r\n";
    state->headers_sent = true;
    return send_all(state->fd, header, strlen(header));
}

static bool stream_chunk(response_state *state, const char *text,
                         size_t length, const char *finish_reason)
{
    char *escaped = NULL;
    size_t escaped_size = 0;
    char error[256];
    if (length > 0 &&
        !k3_json_escape(text, length, &escaped, &escaped_size, error,
                        sizeof error)) {
        return false;
    }
    char frame[8192];
    int size;
    if (finish_reason != NULL) {
        size = snprintf(frame, sizeof frame,
                        "data: {\"id\":\"%s\",\"object\":"
                        "\"chat.completion.chunk\",\"created\":%ld,"
                        "\"model\":\"%s\",\"choices\":[{\"index\":0,"
                        "\"delta\":{},\"finish_reason\":\"%s\"}]}\n\n",
                        state->id, state->created, MODEL_ID, finish_reason);
    } else {
        size = snprintf(frame, sizeof frame,
                        "data: {\"id\":\"%s\",\"object\":"
                        "\"chat.completion.chunk\",\"created\":%ld,"
                        "\"model\":\"%s\",\"choices\":[{\"index\":0,"
                        "\"delta\":{\"content\":%s},"
                        "\"finish_reason\":null}]}\n\n",
                        state->id, state->created, MODEL_ID,
                        escaped != NULL ? escaped : "\"\"");
    }
    free(escaped);
    return size > 0 && send_all(state->fd, frame, (size_t)size);
}

/* Controls are checked only before work or at committed GPU boundaries.
 * Shutdown drains the slot; it is not a client cancellation. */
static mimo26_slot_step request_control(server_runtime *runtime, int client)
{
    if (g_shutdown) {
        mimo26_slot_drain(&runtime->slot);
        return MIMO26_SLOT_STOP_SHUTDOWN;
    }
    if (peer_disconnected(client)) mimo26_slot_cancel(&runtime->slot);
    return mimo26_slot_control_check(&runtime->slot, now_seconds());
}

/* Preserve why a prefill returned without final logits, including when the
 * final prompt chunk stopped before the output projection. */
typedef struct {
    server_runtime *runtime;
    int client;
    mimo26_slot_step stop;
} prefill_request_context;

static bool prefill_progress(void *context, size_t done, size_t total)
{
    prefill_request_context *request = (prefill_request_context *)context;
    server_runtime *runtime = request->runtime;
    (void)done;
    (void)total;
    if (request->stop == MIMO26_SLOT_CONTINUE) {
        request->stop = request_control(runtime, request->client);
        if (request->stop == MIMO26_SLOT_CONTINUE) {
            refuse_backlog(runtime);
            request->stop = request_control(runtime, request->client);
        }
    }
    return request->stop == MIMO26_SLOT_CONTINUE;
}

static void handle_chat(server_runtime *runtime, int fd,
                        const http_request *http)
{
    char code[64];
    char detail[512];
    chat_request request;
    if (!parse_chat(http->body, http->body_size, &request, code, sizeof code,
                    detail, sizeof detail)) {
        const int status = strcmp(code, "model_not_found") == 0 ? 404 : 400;
        send_error(fd, status, status == 404 ? "Not Found" : "Bad Request",
                   code, detail);
        return;
    }

    /* Admission before any work, so a busy or unhealthy server answers
     * immediately rather than after tokenizing. A quarantined worker gets
     * one supervised restart first, so a transient fault costs a request
     * rather than the process. */
    attempt_recovery(runtime);
    const double started = now_seconds();
    const mimo26_slot_admission admission =
        mimo26_slot_admit(&runtime->slot, started, DEFAULT_DEADLINE_SECONDS,
                          request.max_tokens);
    if (admission != MIMO26_SLOT_ADMIT_OK) {
        const char *reason_code =
            admission == MIMO26_SLOT_REJECT_BUSY ? "slot_busy"
            : admission == MIMO26_SLOT_REJECT_QUARANTINED ? "worker_degraded"
                                                          : "shutting_down";
        const char *message =
            admission == MIMO26_SLOT_REJECT_BUSY
                ? "another request is using the single execution slot; retry "
                  "shortly"
            : admission == MIMO26_SLOT_REJECT_QUARANTINED
                ? "the worker faulted repeatedly and supervised restart has "
                  "given up; it needs operator attention"
                : "the server is shutting down";
        send_error(fd, 503, "Service Unavailable", reason_code, message);
        chat_request_free(&request);
        return;
    }

    mimo26_token_buffer prompt;
    memset(&prompt, 0, sizeof prompt);
    char error[512];
    if (!mimo26_tokenizer_encode_chat(runtime->tokenizer, request.messages,
                                      request.message_count,
                                      request.tools_json, true,
                                      request.enable_thinking, &prompt, error,
                                      sizeof error)) {
        send_error(fd, 400, "Bad Request", "encode_failed", error);
        mimo26_slot_finish(&runtime->slot);
        chat_request_free(&request);
        return;
    }
    if (prompt.count + request.max_tokens > runtime->context_capacity) {
        snprintf(detail, sizeof detail,
                 "prompt of %zu tokens plus max_tokens %u exceeds the "
                 "qualified context of %zu",
                 prompt.count, request.max_tokens,
                 runtime->context_capacity);
        send_error(fd, 400, "Bad Request", "context_exceeded", detail);
        mimo26_token_buffer_free(&prompt);
        mimo26_slot_finish(&runtime->slot);
        chat_request_free(&request);
        return;
    }

    if (!reset_request_worker(runtime, error, sizeof error)) {
        fprintf(stderr, "mimo26: request reset refused: %s\n", error);
        send_error(fd, 500, "Internal Server Error", "decode_failed",
                   "the worker refused context reset and is quarantined");
        mimo26_token_buffer_free(&prompt);
        chat_request_free(&request);
        return;
    }
    response_state state;
    memset(&state, 0, sizeof state);
    state.fd = fd;
    state.streaming = request.stream;
    state.created = (long)time(NULL);
    snprintf(state.id, sizeof state.id, "chatcmpl-mimo26-%llu",
             (unsigned long long)runtime->slot.request_id);

    float *logits = (float *)malloc(152576u * sizeof *logits);
    char *collected = NULL;
    size_t collected_used = 0, collected_capacity = 0;
    if (logits == NULL) {
        send_error(fd, 500, "Internal Server Error", "out_of_memory",
                   "could not allocate logits");
        mimo26_token_buffer_free(&prompt);
        mimo26_slot_finish(&runtime->slot);
        chat_request_free(&request);
        return;
    }

    /*
     * The prompt goes through the layer-major prefill path, not the decode
     * loop. Measured at the 512 gate that is 0.098 s/token against 0.422,
     * because the BF16 projections are read once per chunk rather than once
     * per token.
     *
     * The progress hook keeps the admission guarantee intact. Prefill does
     * not return control per token, so without it a caller arriving during
     * a long prompt would wait for the whole prompt rather than being
     * refused in milliseconds -- at 8K that is fifteen minutes of silence.
     * It also lets a client that disconnects mid-prefill stop the work.
     */
    bool failed = false;
    uint32_t next = 0;
    prefill_request_context progress_context = {runtime, fd, request_control(runtime, fd)};
    if (progress_context.stop != MIMO26_SLOT_CONTINUE) {
        /* No work or logits to consume when already stopped before prefill. */
    } else if (mimo26_gpu_worker_prefill(runtime->worker, prompt.ids, prompt.count,
                                  logits, prefill_progress, &progress_context, error,
                                  sizeof error) != MIMO26_GPU_WORKER_OK) {
        failed = true;
    } else if (progress_context.stop == MIMO26_SLOT_CONTINUE) {
        /* A stopped prefill returns OK without producing final logits. */
        next = mimo26_gpu_worker_argmax(logits);
    }
    if (failed) {
        /* A decode failure means the worker's state is not trusted. */
        mimo26_slot_fault(&runtime->slot);
        fprintf(stderr, "mimo26 request %s: prefill failed: %s\n",
                state.id, error);
        send_error(fd, 500, "Internal Server Error", "decode_failed",
                   "the worker faulted during prefill and is quarantined");
        free(logits);
        mimo26_token_buffer_free(&prompt);
        chat_request_free(&request);
        return;
    }

    if (state.streaming && !stream_begin(&state) && !g_shutdown &&
        progress_context.stop == MIMO26_SLOT_CONTINUE) {
        mimo26_slot_cancel(&runtime->slot);
    }

    mimo26_decode_stream decoder;
    mimo26_decode_stream_init(&decoder);
    const char *finish_reason = "stop";
    size_t produced_tokens = 0;
    /*
     * Reasoning separation. The model opens its turn with <think>...</think>
     * and the tokens inside are not the answer. Routing them to a separate
     * field rather than into content is what the plan means by reasoning
     * separation -- concatenating them would make the visible reply begin
     * mid-deliberation, which is what the first draft of this server did
     * ("Hmm, the").
     *
     * Tracked on token ids rather than by scanning text, because the markers
     * are single tokens and a text scan would also match a user who typed
     * the words.
     */
    bool in_reasoning = false;
    char *reasoning = NULL;
    size_t reasoning_used = 0, reasoning_capacity = 0;
    while (true) {
        mimo26_slot_step step = progress_context.stop;
        if (step == MIMO26_SLOT_CONTINUE) step = request_control(runtime, fd);
        if (step == MIMO26_SLOT_CONTINUE) {
            refuse_backlog(runtime);
            step = request_control(runtime, fd);
        }
        if (step == MIMO26_SLOT_CONTINUE)
            step = mimo26_slot_step_check(&runtime->slot, now_seconds());
        if (step != MIMO26_SLOT_CONTINUE) {
            finish_reason = mimo26_slot_finish_reason(step);
            if (step == MIMO26_SLOT_STOP_DEADLINE) {
                runtime->slot.deadline_stops++;
            }
            break;
        }
        /* The model's own end-of-turn marker. */
        if (next == MIMO26_TOK_IM_END || next == MIMO26_TOK_ENDOFTEXT) {
            finish_reason = "stop";
            break;
        }
        if (next == MIMO26_TOK_THINK_OPEN) {
            in_reasoning = true;
        } else if (next == MIMO26_TOK_THINK_CLOSE) {
            in_reasoning = false;
        }
        /*
         * The tool-call wrapper is a special token pair, so the streaming
         * decoder drops it along with every other special -- which it has to,
         * or <|im_end|> would reach the client. Re-inserted here as text so
         * the parser sees a complete block. Without this the model's
         * <function=...> arrives naked and is shown to the user as prose,
         * which is exactly what the first live tool call did.
         */
        if (next == MIMO26_TOK_TOOL_CALL_OPEN ||
            next == MIMO26_TOK_TOOL_CALL_CLOSE) {
            const char *marker = next == MIMO26_TOK_TOOL_CALL_OPEN
                                     ? "<tool_call>" : "</tool_call>";
            const size_t marker_length = strlen(marker);
            if (!in_reasoning) {
                if (collected_used + marker_length + 1u > collected_capacity) {
                    collected_capacity =
                        (collected_used + marker_length + 1u) * 2u;
                    char *grown = (char *)realloc(collected,
                                                  collected_capacity);
                    if (grown == NULL) { failed = true; break; }
                    collected = grown;
                }
                memcpy(collected + collected_used, marker, marker_length);
                collected_used += marker_length;
                collected[collected_used] = '\0';
            }
            produced_tokens++;
            if (mimo26_gpu_worker_decode(runtime->worker, next, logits, error,
                                         sizeof error) !=
                MIMO26_GPU_WORKER_OK) {
                failed = true;
                break;
            }
            next = mimo26_gpu_worker_argmax(logits);
            continue;
        }
        if (next == MIMO26_TOK_THINK_OPEN ||
            next == MIMO26_TOK_THINK_CLOSE) {
            /* The markers themselves belong in neither field. */
            produced_tokens++;
            if (mimo26_gpu_worker_decode(runtime->worker, next, logits, error,
                                         sizeof error) !=
                MIMO26_GPU_WORKER_OK) {
                failed = true;
                break;
            }
            next = mimo26_gpu_worker_argmax(logits);
            continue;
        }

        char piece[64];
        size_t piece_size = 0;
        if (!mimo26_tokenizer_decode_stream(runtime->tokenizer, &decoder,
                                            next, piece, sizeof piece,
                                            &piece_size)) {
            failed = true;
            break;
        }
        if (piece_size > 0 && in_reasoning) {
            /* Collected either way: a streaming caller receives it in the
             * final frame rather than interleaved with the answer. */
            if (reasoning_used + piece_size + 1u > reasoning_capacity) {
                reasoning_capacity = (reasoning_used + piece_size + 1u) * 2u;
                char *grown = (char *)realloc(reasoning, reasoning_capacity);
                if (grown == NULL) {
                    failed = true;
                    break;
                }
                reasoning = grown;
            }
            memcpy(reasoning + reasoning_used, piece, piece_size);
            reasoning_used += piece_size;
            reasoning[reasoning_used] = '\0';
        } else if (piece_size > 0) {
            if (state.streaming) {
                if (!stream_chunk(&state, piece, piece_size, NULL)) {
                    mimo26_slot_cancel(&runtime->slot);
                }
            } else {
                if (collected_used + piece_size + 1u > collected_capacity) {
                    collected_capacity =
                        (collected_used + piece_size + 1u) * 2u;
                    char *grown = (char *)realloc(collected,
                                                  collected_capacity);
                    if (grown == NULL) {
                        failed = true;
                        break;
                    }
                    collected = grown;
                }
                memcpy(collected + collected_used, piece, piece_size);
                collected_used += piece_size;
                collected[collected_used] = '\0';
            }
        }
        produced_tokens++;

        if (mimo26_gpu_worker_decode(runtime->worker, next, logits, error,
                                     sizeof error) != MIMO26_GPU_WORKER_OK) {
            failed = true;
            break;
        }
        next = mimo26_gpu_worker_argmax(logits);
    }

    if (failed) {
        mimo26_slot_fault(&runtime->slot);
        fprintf(stderr, "mimo26 %s: generation failed: %s\n", state.id,
                error);
        if (!state.headers_sent) {
            send_error(fd, 500, "Internal Server Error", "decode_failed",
                       "the worker faulted during generation and is "
                       "quarantined");
        }
    } else if (!strcmp(finish_reason, "shutdown")) {
        /* Transport observes the signal too; don't attempt a new response
         * or publish a partially collected tool call during shutdown. */
        mimo26_slot_finish(&runtime->slot);
    } else if (state.streaming) {
        stream_chunk(&state, NULL, 0, finish_reason);
        const char *done = "data: [DONE]\n\n";
        send_all(fd, done, strlen(done));
        mimo26_slot_finish(&runtime->slot);
    } else {
        /*
         * Tool calls are parsed out of the completed text rather than
         * detected token by token. The block only means anything once it is
         * closed, and a partial <function= is not a call -- streaming a
         * half-formed one would put the model's internal syntax in front of
         * a user.
         */
        char *visible = NULL;
        mimo26_parsed_tool_call *parsed = NULL;
        size_t parsed_count = 0;
        if (collected != NULL &&
            mimo26_tokenizer_parse_tool_calls(collected, &visible, &parsed,
                                              &parsed_count, error,
                                              sizeof error)) {
            free(collected);
            collected = visible;
            collected_used = strlen(visible);
            if (parsed_count > 0) {
                finish_reason = "tool_calls";
            }
        }
        char *escaped = NULL;
        size_t escaped_size = 0;
        if (collected != NULL) {
            k3_json_escape(collected, collected_used, &escaped, &escaped_size,
                           error, sizeof error);
        }
        char *escaped_reasoning = NULL;
        size_t escaped_reasoning_size = 0;
        if (reasoning != NULL) {
            k3_json_escape(reasoning, reasoning_used, &escaped_reasoning,
                           &escaped_reasoning_size, error, sizeof error);
        }
        /* Render the calls in OpenAI's shape, with arguments as a JSON
         * string, which is what clients parse. */
        char *calls_text = NULL;
        size_t calls_used = 0, calls_capacity = 0;
        if (parsed_count > 0) {
            append_json(&calls_text, &calls_used, &calls_capacity, "[");
            for (size_t i = 0; i < parsed_count; i++) {
                char *escaped_name = NULL, *escaped_args = NULL;
                size_t ignored = 0;
                k3_json_escape(parsed[i].name, strlen(parsed[i].name),
                               &escaped_name, &ignored, error, sizeof error);
                k3_json_escape(parsed[i].arguments_json,
                               strlen(parsed[i].arguments_json),
                               &escaped_args, &ignored, error, sizeof error);
                char entry[512];
                snprintf(entry, sizeof entry,
                         "%s{\"id\":\"call_%s_%zu\",\"type\":\"function\","
                         "\"function\":{\"name\":%s,\"arguments\":%s}}",
                         i ? "," : "", state.id, i,
                         escaped_name != NULL ? escaped_name : "\"\"",
                         escaped_args != NULL ? escaped_args : "\"{}\"");
                append_json(&calls_text, &calls_used, &calls_capacity, entry);
                free(escaped_name);
                free(escaped_args);
            }
            append_json(&calls_text, &calls_used, &calls_capacity, "]");
        }
        const size_t body_capacity =
            escaped_size + escaped_reasoning_size + calls_used + 1024u;
        char *body = (char *)malloc(body_capacity);
        if (body != NULL) {
            const int size = snprintf(
                body, body_capacity,
                "{\"id\":\"%s\",\"object\":\"chat.completion\","
                "\"created\":%ld,\"model\":\"%s\",\"choices\":[{\"index\":0,"
                "\"message\":{\"role\":\"assistant\",\"content\":%s,"
                "\"reasoning_content\":%s,\"tool_calls\":%s},"
                "\"finish_reason\":\"%s\"}],\"usage\":{"
                "\"prompt_tokens\":%zu,\"completion_tokens\":%zu,"
                "\"total_tokens\":%zu}}",
                state.id, state.created, MODEL_ID,
                escaped != NULL ? escaped : "\"\"",
                escaped_reasoning != NULL ? escaped_reasoning : "null",
                calls_text != NULL ? calls_text : "null",
                finish_reason, prompt.count,
                produced_tokens, prompt.count + produced_tokens);
            if (size > 0) {
                send_response(fd, 200, "OK", "application/json", body,
                              (size_t)size);
            }
            free(body);
        }
        free(escaped);
        free(escaped_reasoning);
        free(calls_text);
        mimo26_tool_calls_free(parsed, parsed_count);
        mimo26_slot_finish(&runtime->slot);
    }

    /* Logged without content: a request log that contains the prompt is a
     * transcript of everything anyone ever asked. */
    fprintf(stderr,
            "mimo26 %s prompt=%zu completion=%zu reasoning=%zuB finish=%s "
            "%.2fs\n",
            state.id, prompt.count, produced_tokens, reasoning_used,
            finish_reason, now_seconds() - started);
    runtime->served++;
    if (!failed) {
        /* A request that completed cleanly proves the worker recovered, so
         * the next transient fault gets a full budget again. */
        runtime->recovery_attempts = 0u;
    }

    free(collected);
    free(reasoning);
    free(logits);
    mimo26_token_buffer_free(&prompt);
    chat_request_free(&request);
}

/* ---- main ---- */

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    setvbuf(stderr, NULL, _IOLBF, 0);
    if (argc < 2 || (argc == 2 && strcmp(argv[1], "--help") == 0)) {
        fprintf(stderr,
                "usage: %s ROOT [--port N] [--host H] [--slots N] "
                "[--context N] [--prefill-chunk 0..128] "
                "[--expert-lookahead on|off] [--retain-experts on|off]\n", argv[0]);
        return argc < 2 ? 2 : 0;
    }
    /* Local-only by default. Remote exposure needs an authentication and
     * network policy the plan requires before it is offered. */
    mimo26_server_options options{};
    mimo26_gpu_worker_config_defaults(&options.worker);
    char option_error[256];
    if (!mimo26_server_parse_options(argc, argv, &options, option_error, sizeof option_error)) {
        fprintf(stderr, "configuration: %s\n", option_error);
        return 2;
    }
    const char *root = options.root;
    const char *host = options.host;
    const int port = options.port;
    const mimo26_gpu_worker_config config = options.worker;

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, SIG_IGN);

    server_runtime runtime;
    memset(&runtime, 0, sizeof runtime);
    runtime.model_root = root;
    runtime.context_capacity = config.global_kv_capacity;
    runtime.prefill_chunk = config.prefill_chunk;
    runtime.expert_slots_per_layer = config.expert_slots_per_layer;
    runtime.expert_lookahead = config.expert_lookahead;
    runtime.retain_experts = options.retain_experts;
    mimo26_slot_init(&runtime.slot);

    char error[512];
    if (!mimo26_tokenizer_create(&runtime.tokenizer, root, error,
                                 sizeof error)) {
        fprintf(stderr, "tokenizer: %s\n", error);
        return 1;
    }
    printf("loading the worker (%u expert slots per layer, context %zu, "
           "prefill chunk %u, expert lookahead %s)\n",
           (unsigned)config.expert_slots_per_layer,
           config.global_kv_capacity, (unsigned)config.prefill_chunk,
           config.expert_lookahead ? "on" : "off");
    if (mimo26_gpu_worker_create(&runtime.worker, root, &config, error,
                                 sizeof error) != MIMO26_GPU_WORKER_OK) {
        fprintf(stderr, "worker: %s\n", error);
        return 1;
    }
    mimo26_gpu_worker_stats stats;
    mimo26_gpu_worker_get_stats(runtime.worker, &stats);
    printf("resident %.2f GiB after %.1f s\n",
           (double)mimo26_gpu_worker_resident_bytes(runtime.worker) /
               1073741824.0,
           stats.load_seconds);

    const int listener = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    int reuse = 1;
    setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof reuse);
    struct sockaddr_in address;
    memset(&address, 0, sizeof address);
    address.sin_family = AF_INET;
    address.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &address.sin_addr) != 1 ||
        bind(listener, (struct sockaddr *)&address, sizeof address) != 0 ||
        listen(listener, 16) != 0) {
        fprintf(stderr, "cannot listen on %s:%d: %s\n", host, port,
                strerror(errno));
        return 1;
    }
    runtime.listener = listener;
    printf("listening on http://%s:%d  (model %s)\n", host, port, MODEL_ID);

    while (!g_shutdown) {
        struct pollfd waiting;
        waiting.fd = listener;
        waiting.events = POLLIN;
        waiting.revents = 0;
        if (poll(&waiting, 1u, 200) <= 0) {
            continue;
        }
        const int client = accept(listener, NULL, NULL);
        if (client < 0) {
            continue;
        }
        http_request http;
        if (!read_request(client, &http, error, sizeof error)) {
            send_error(client, 400, "Bad Request", "malformed_request",
                       error);
            close(client);
            continue;
        }
        if (strcmp(http.method, "GET") == 0 &&
            strcmp(http.path, "/health") == 0) {
            send_health(client, &runtime);
        } else if (strcmp(http.method, "GET") == 0 &&
                   strcmp(http.path, "/v1/models") == 0) {
            send_models(client);
        } else if (strcmp(http.method, "POST") == 0 &&
                   strcmp(http.path, "/v1/chat/completions") == 0) {
            handle_chat(&runtime, client, &http);
        } else {
            send_error(client, 404, "Not Found", "no_such_route",
                       "unknown route; this server offers /health, "
                       "/v1/models and /v1/chat/completions");
        }
        request_free(&http);
        close(client);
    }

    /* Graceful shutdown: refuse new work, let anything in flight finish.
     * Single-threaded, so nothing can be in flight here -- the drain is
     * still done properly so the shape survives a threaded accept loop. */
    printf("\nshutting down\n");
    mimo26_slot_drain(&runtime.slot);
    close(listener);
    mimo26_gpu_worker_destroy(runtime.worker);
    mimo26_tokenizer_destroy(runtime.tokenizer);
    printf("served %llu requests\n", (unsigned long long)runtime.served);
    return 0;
}
