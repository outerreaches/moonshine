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
 *       [--retain-experts on|off] [--min-headroom-gib N]
 */
#include "k3_json.h"
#include "mimo26_gpu_worker.h"
#include "mimo26_server_options.h"
#include "moonshine_version.h"
#include "k3_prefix_reuse.h"
#include "k3_prefix_bundle.h"
#include <fcntl.h>
#include <limits.h>
#include <sys/stat.h>
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
/* Read by parse_chat, which has no runtime handle. Set once before the
 * accept loop and never written again. */
static uint32_t g_max_output_tokens = 8192u;

static volatile sig_atomic_t g_shutdown = 0;

static void on_signal(int signal_number)
{
    (void)signal_number;
    g_shutdown = 1;
}

/* Wall clock, for a reader in another process; now_seconds() is monotonic
 * and means nothing outside this one. */

/*
 * Structured log, in the shape k3_server.c emits and the dashboard already
 * parses: "<ISO8601 with ms> <LEVEL> <event.name> key=value ...".
 *
 * Ported rather than invented. The dashboard renders a Moonshine card from
 * exactly these records -- requests, in-flight state, engine facts -- without
 * ever opening a connection, which matters because this engine stops calling
 * accept() while it works and a poller's handshake sits in the backlog until
 * the kernel refuses real clients. Matching K3's vocabulary means both
 * engines feed one parser.
 */
static void server_log(const char *level, const char *event,
                       const char *completion_id, const char *fmt, ...)
{
    struct timespec now;
    struct tm utc;
    char stamp[40];
    clock_gettime(CLOCK_REALTIME, &now);
    gmtime_r(&now.tv_sec, &utc);
    if (strftime(stamp, sizeof stamp, "%Y-%m-%dT%H:%M:%S.000Z", &utc) == 0u) {
        memcpy(stamp, "1970-01-01T00:00:00.000Z", 25u);
    }
    const unsigned ms = (unsigned)((now.tv_nsec / 1000000L) % 1000L);
    stamp[20] = (char)('0' + ms / 100u);
    stamp[21] = (char)('0' + (ms / 10u) % 10u);
    stamp[22] = (char)('0' + ms % 10u);

    char message[1024] = {0};
    if (fmt != NULL && fmt[0] != '\0') {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(message, sizeof message, fmt, ap);
        va_end(ap);
    }
    if (completion_id != NULL && completion_id[0] != '\0') {
        fprintf(stderr, "%s %s %s id=%s%s%s\n", stamp, level, event,
                completion_id, message[0] ? " " : "", message);
    } else {
        fprintf(stderr, "%s %s %s%s%s\n", stamp, level, event,
                message[0] ? " " : "", message);
    }
}

static double wall_clock_epoch(void)
{
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
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
    /* RFC 7235 requires a challenge on 401; without it a conforming client
     * cannot know which scheme to retry with. */
    const int header_size = snprintf(
        header, sizeof header,
        "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
        "%sCache-Control: no-store\r\nConnection: close\r\n\r\n",
        status, reason, content_type, body_size,
        status == 401 ? "WWW-Authenticate: Bearer\r\n" : "");
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

/*
 * MemAvailable, the kernel's own estimate of what can be allocated without
 * swapping. Returns 0 if it cannot be read, which disables the guard rather
 * than refusing on a host whose /proc layout is unexpected.
 */
static uint64_t host_available_bytes(void)
{
    FILE *meminfo = fopen("/proc/meminfo", "re");
    if (meminfo == NULL) {
        return 0u;
    }
    char line[256];
    uint64_t kib = 0u;
    while (fgets(line, sizeof line, meminfo) != NULL) {
        if (strncmp(line, "MemAvailable:", 13) == 0) {
            kib = strtoull(line + 13, NULL, 10);
            break;
        }
    }
    fclose(meminfo);
    return kib * 1024ull;
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

/*
 * Both ported from k3_server.c rather than rewritten, because both are easy to
 * get subtly wrong: the comparison must not return early on the first
 * differing byte, and the loopback test must accept the whole 127/8 block, not
 * just 127.0.0.1.
 */
static bool loopback_host(const char *host)
{
    if (strcmp(host, "localhost") == 0 || strcmp(host, "::1") == 0) {
        return true;
    }
    struct in_addr address;
    return inet_pton(AF_INET, host, &address) == 1 &&
           (ntohl(address.s_addr) >> 24) == 127u;
}

/* Constant time in the compared prefix; the length difference is folded into
 * the accumulator rather than short-circuiting on it. */
static bool authorized(const char *authorization, const char *api_key)
{
    if (api_key == NULL) {
        return true;
    }
    if (authorization == NULL || strncmp(authorization, "Bearer ", 7u) != 0) {
        return false;
    }
    const char *provided = authorization + 7u;
    const size_t expected_size = strlen(api_key);
    const size_t provided_size = strlen(provided);
    size_t difference = expected_size ^ provided_size;
    const size_t compared =
        expected_size < provided_size ? expected_size : provided_size;
    for (size_t i = 0u; i < compared; i++) {
        difference |= (unsigned char)(api_key[i] ^ provided[i]);
    }
    return difference == 0u;
}

/* ---- request ---- */

typedef struct {
    char   *method;
    char   *path;
    char   *body;
    char   *authorization;
    size_t  body_size;
} http_request;

static void request_free(http_request *request)
{
    free(request->method);
    free(request->path);
    free(request->body);
    free(request->authorization);
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
        if (name==13 && !strncasecmp(line,"Authorization",name)) {
            if (request->authorization) return reject("duplicate Authorization");
            const char *p=colon+1,*last=end;
            while(p<last && (*p==' '||*p=='\t'))++p;
            while(last>p && (last[-1]==' '||last[-1]=='\t'))--last;
            request->authorization=(char*)malloc((size_t)(last-p)+1u);
            if (!request->authorization) return reject("out of memory");
            memcpy(request->authorization,p,(size_t)(last-p));
            request->authorization[last-p]='\0';
        }
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
    bool               expert_major;
    bool               expert_weight_reuse;
    bool               retain_experts;
    double             request_deadline_seconds;
    /*
     * The exact token sequence believed to be committed to the worker's KV,
     * so the next request can continue from it instead of re-prefilling.
     *
     * Published only after a request completes cleanly, and invalidated at the
     * start of every request. Any path that leaves the handler without
     * republishing therefore leaves the next request to reset and prefill in
     * full -- the failure mode is lost work, never reuse of damaged history.
     */
    bool               kv_prefix_reuse;
    /* Reported so a supervisor can verify the profile it transmitted rather
     * than only transmitting it. */
    uint32_t           min_headroom_gib;
    bool               api_key_set;
    /*
     * Whether the resolved profile is this build's compiled-in default.
     *
     * A qualification run against a non-default profile says nothing about
     * what an operator gets by passing no flags, and the packager has no way
     * to learn a binary's defaults without loading a model. So the binary
     * answers the question itself.
     */
    bool               stock_profile;
    /*
     * Activity, for an observer that must not open a connection. This engine
     * serves one request at a time and stops calling accept() while it works,
     * so a poller's TCP handshake completes in the kernel and sits in the
     * accept backlog; at a five-second tick that overflowed a 16-slot backlog
     * during one long prefill and the kernel began refusing real clients.
     * The dashboard therefore reads a file, and this is what fills it.
     */
    const char        *status_file;
    bool               inflight;
    double             inflight_since;
    double             inflight_since_epoch;
    size_t             inflight_prompt;
    char               inflight_id[64];
    /* Last completed request, and a small window for a rate that is not one
     * sample. */
    char               last_id[64];
    size_t             last_prompt_tokens;
    size_t             last_evaluated_tokens;   /* prefill actually computed */
    size_t             last_completion_tokens;
    size_t             last_reused_tokens;
    double             last_prefill_seconds;
    double             last_decode_seconds;
    double             last_total_seconds;
    char               last_finish[24];
    bool               last_from_disk;
    double             window_prefill_tokens;
    double             window_prefill_seconds;
    double             window_decode_tokens;
    double             window_decode_seconds;
    uint32_t          *resident_ids;
    size_t             resident_count;
    size_t             resident_capacity;
    bool               resident_valid;
    uint64_t           prefix_hits;
    uint64_t           prefix_misses;
    uint64_t           prefix_tokens_saved;
    /*
     * Second tier: prefixes persisted to disk, so a prompt survives the
     * request that computed it. The in-memory tier above only continues a
     * context that is still resident and extends exactly.
     */
    k3_prefix_bundle  *prefix_bundle;
    uint64_t           prefix_disk_hits;
    uint64_t           prefix_disk_publishes;
    double             prefix_disk_seconds;
    /*
     * Cached rather than asked for. /health used to call
     * k3_prefix_bundle_count(), which walks the index on every health poll and
     * dragged the bundle into the link of two gates that deliberately link
     * almost nothing. Refreshed wherever the count can change.
     */
    size_t             prefix_entries;
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

/*
 * The exact sequence committed to KV during one request: the prompt, then
 * every token handed to decode, in order. Built alongside the request so the
 * resident prefix published at the end is what the worker actually holds
 * rather than a reconstruction of it.
 *
 * `ok` goes false on any allocation failure, which only costs the next request
 * its reuse.
 */
typedef struct {
    uint32_t *ids;
    size_t    count;
    size_t    capacity;
    bool      ok;
} token_trail;

static void trail_push(token_trail *trail, uint32_t id)
{
    if (!trail->ok) {
        return;
    }
    if (trail->count == trail->capacity) {
        const size_t grown = trail->capacity ? trail->capacity * 2u : 256u;
        uint32_t *ids = (uint32_t *)realloc(trail->ids, grown * sizeof *ids);
        if (ids == NULL) {
            trail->ok = false;
            return;
        }
        trail->ids = ids;
        trail->capacity = grown;
    }
    trail->ids[trail->count++] = id;
}

/* The resident prefix is a claim about worker state; drop it whenever that
 * claim might no longer hold. */
static void resident_invalidate(server_runtime *runtime)
{
    runtime->resident_valid = false;
    runtime->resident_count = 0;
}

/*
 * Record the sequence now committed to KV. Called only on a clean completion,
 * with the prompt followed by every token actually generated.
 */
static void resident_publish(server_runtime *runtime, const uint32_t *ids,
                             size_t count)
{
    if (!runtime->kv_prefix_reuse || count == 0) {
        resident_invalidate(runtime);
        return;
    }
    if (count > runtime->resident_capacity) {
        uint32_t *grown = (uint32_t *)realloc(runtime->resident_ids,
                                              count * sizeof *grown);
        if (grown == NULL) {           /* keep going without reuse */
            resident_invalidate(runtime);
            return;
        }
        runtime->resident_ids = grown;
        runtime->resident_capacity = count;
    }
    memcpy(runtime->resident_ids, ids, count * sizeof *ids);
    runtime->resident_count = count;
    runtime->resident_valid = true;
}

/*
 * How many leading tokens of `ids` the worker can keep.
 *
 * Reuse requires the resident sequence to be a STRICT prefix of the new
 * prompt. A partial match would mean rewinding the divergent tail, and the KV
 * journals only 8 steps because windowed layers cannot otherwise restore what
 * they evicted -- so anything else resets and prefills in full.
 *
 * The worker's own committed position is the authority. If the server's
 * bookkeeping disagrees with it by even one token, the belief is wrong and
 * reuse is refused.
 */
static size_t plan_prefix_reuse(server_runtime *runtime, const uint32_t *ids,
                                size_t count)
{
    if (!runtime->kv_prefix_reuse || !runtime->resident_valid ||
        runtime->resident_count == 0) {
        return 0;
    }
    /* Needs at least one new token: the last one produces the logits this
     * request answers from, so a prompt already fully resident cannot be
     * continued without re-running its final position. */
    if (runtime->resident_count >= count) {
        return 0;
    }
    if (!mimo26_gpu_worker_idle(runtime->worker) ||
        mimo26_gpu_worker_position(runtime->worker) !=
            (uint64_t)runtime->resident_count) {
        return 0;
    }
    /*
     * K3's predicate rather than a second implementation of it. Its header
     * spells out why this has to be conservative: a wrong admission does not
     * fail loudly, it continues from causal state that does not correspond to
     * the supplied history and returns a plausible completion computed against
     * the wrong prefix.
     *
     * It insists on a two-token remainder where MiMo's prefill would accept
     * one. That declines the rare prompt that grew by exactly one token, which
     * is not worth forking a tested predicate to recover.
     */
    if (!k3_prefix_reuse_admits(runtime->resident_ids,
                                runtime->resident_count, ids, count)) {
        return 0;
    }
    return runtime->resident_count;
}

/*
 * Below this, a checkpoint is not worth its I/O: the windowed ring alone is a
 * fixed ~24 MiB whatever the prompt length, while a few hundred tokens of
 * prefill is seconds. 512 tokens is roughly 30 s of prefill against a ~35 MiB
 * write.
 */
#define PREFIX_PUBLISH_MIN_TOKENS 512u

/*
 * K3's bundle stores its own engine's state descriptor. MiMo's checkpoint
 * carries the same facts under different names, so the bundle can be reused
 * unchanged rather than forked: capacity is the context, the committed length
 * is the token position, and the KV's geometry identity stands in for the
 * model layout.
 */
static void prefix_state_info(const mimo26_kv_state_info *from,
                              k3_engine_state_file_info *to)
{
    memset(to, 0, sizeof *to);
    to->format_version = from->format_version;
    to->context = (uint32_t)from->global_capacity;
    to->token_position = (uint32_t)from->length;
    to->model_layout_crc64 = from->layout_crc64;
    to->payload_bytes = from->payload_bytes;
    to->file_bytes = from->file_bytes;
    to->payload_crc64 = from->payload_crc64;
    to->q8_projections = false;
    to->wall_seconds = from->wall_seconds;
}

/*
 * Longest stored prefix this prompt admits, restored into the worker.
 *
 * Returns the number of leading tokens now committed, or 0 if nothing was
 * usable -- in which case the worker may have been left empty by a failed
 * import, and the caller still performs its reset.
 */
static size_t restore_prefix_from_disk(server_runtime *runtime,
                                       const uint32_t *ids, size_t count)
{
    if (runtime->prefix_bundle == NULL ||
        !mimo26_gpu_worker_idle(runtime->worker)) {
        return 0;
    }
    size_t best_index = SIZE_MAX;
    size_t best_tokens = 0;
    const size_t entries = k3_prefix_bundle_count(runtime->prefix_bundle);
    for (size_t index = 0; index < entries; index++) {
        k3_prefix_bundle_entry entry;
        if (!k3_prefix_bundle_entry_at(runtime->prefix_bundle, index, &entry)) {
            continue;
        }
        if (entry.token_count > best_tokens &&
            k3_prefix_reuse_admits(entry.tokens, entry.token_count,
                                   ids, count)) {
            best_index = index;
            best_tokens = entry.token_count;
        }
    }
    if (best_index == SIZE_MAX) {
        return 0;
    }
    k3_prefix_bundle_entry entry;
    if (!k3_prefix_bundle_entry_at(runtime->prefix_bundle, best_index,
                                   &entry)) {
        return 0;
    }
    char error[512];
    mimo26_kv_state_info info;
    const double started = now_seconds();
    if (mimo26_gpu_worker_import_state(runtime->worker, entry.state_path,
                                       &info, error, sizeof error) !=
        MIMO26_GPU_WORKER_OK) {
        /* A checkpoint that will not load is worse than none: drop it so the
         * next request does not pay for it again. */
        fprintf(stderr, "mimo26: dropping unusable checkpoint %s: %s\n",
                entry.state_path, error);
        char remove_error[256];
        (void)k3_prefix_bundle_remove(runtime->prefix_bundle, best_index,
                                      remove_error, sizeof remove_error);
        runtime->prefix_entries =
            k3_prefix_bundle_count(runtime->prefix_bundle);
        return 0;
    }
    /* The file decides how much history exists; believe it over the index. */
    if (info.length != (uint64_t)entry.token_count) {
        fprintf(stderr,
                "mimo26: checkpoint %s holds %llu positions but its index "
                "claims %zu; dropping\n", entry.state_path,
                (unsigned long long)info.length, entry.token_count);
        char remove_error[256];
        (void)k3_prefix_bundle_remove(runtime->prefix_bundle, best_index,
                                      remove_error, sizeof remove_error);
        runtime->prefix_entries =
            k3_prefix_bundle_count(runtime->prefix_bundle);
        return 0;
    }
    runtime->prefix_disk_hits++;
    runtime->prefix_disk_seconds += now_seconds() - started;
    return entry.token_count;
}

/*
 * Store the state produced by evaluating exactly `count` prompt tokens.
 *
 * Published after prefill and before any generation, which is the point that
 * matters: chat templates are append-only, so a prompt is a prefix of every
 * later prompt in the same conversation. A checkpoint taken after generation
 * would be keyed by tokens the client never echoes back when the model's
 * reasoning is separated from its visible content, and would never match.
 */
/* Bump when the KV-producing arithmetic changes in any way; it names the
 * prefix store's directory, so a bump retires the old checkpoints. */
/*
 * A stop that produced NOTHING is not a truncation.
 *
 * Mapping "deadline" to the schema's "length" fixed a real defect -- the internal
 * name was rejected outright by strict clients -- but it introduced a subtler one.
 * With zero completion tokens the DeepSeek harness rendered "length" as:
 *
 *     Output token limit reached. The reply was cut off; earlier output is
 *     preserved in the conversation. Send "continue" to let the model resume.
 *
 * Every clause is false when a prefill deadline fired before generation began:
 * there was no output limit, nothing was cut off, nothing was preserved. A loud
 * wrong error had been traded for a quiet wrong message, which is worse -- the
 * operator is told to do something that does not address the cause.
 *
 * So the wire answer depends on whether any tokens were produced:
 *
 *   generated > 0, stopped at a limit  -> 200 with finish_reason "length".
 *                                         The reply genuinely was cut off.
 *   generated == 0, stopped at a limit -> an HTTP error naming the real cause.
 *                                         The request did not complete at all.
 *
 * This is not a return to the original bug. That was a schema violation, which a
 * client cannot interpret; this is a documented failure status with a structured
 * body, which is exactly what the convention has for a server-side timeout.
 */
static bool stop_produced_nothing(const char *internal, size_t generated)
{
    return generated == 0u && internal != NULL &&
           (!strcmp(internal, "deadline") || !strcmp(internal, "shutdown"));
}

/* Mirrors K3_PREFIX_REUSE_MIN_SUFFIX in k3_prefix_reuse.c: exact-prefix
 * admission requires this many tokens after the retained prefix. */
enum { PREFIX_RETRY_MIN_SUFFIX = 2u };

enum { MIMO26_KV_NUMERICS_VERSION = 1u };

/*
 * What happened when a checkpoint was offered, so a client is told the truth.
 *
 * The responses used to promise unconditionally that "the evaluated prefix was
 * checkpointed, so retrying resumes from it". Publication can decline for five
 * reasons, and even a successful publish is useless to an IDENTICAL retry if it
 * covers the whole prompt: exact-prefix admission requires at least
 * K3_PREFIX_REUSE_MIN_SUFFIX tokens after the retained prefix.
 */
typedef struct {
    bool   stored;        /* an entry now exists covering `retained` tokens */
    bool   resumable;     /* ...and an identical retry can actually use it */
    size_t retained;
} prefix_publish_outcome;

static prefix_publish_outcome publish_prefix_checkpoint(server_runtime *runtime,
                                      const uint32_t *ids, size_t count)
{
    /*
     * Say why, when the answer is no.
     *
     * These guards used to return silently, which is how resumable prefill first
     * appeared to do nothing at all: publishes stayed at 0 with no indication of
     * which precondition had failed. A checkpoint quietly not happening is
     * exactly the kind of thing that needs a line in the log.
     */
    const char *refusal = NULL;
    if (runtime->prefix_bundle == NULL) refusal = "no prefix store configured";
    else if (count < PREFIX_PUBLISH_MIN_TOKENS) refusal = "prefix shorter than the publish floor";
    else if (!mimo26_gpu_worker_idle(runtime->worker)) refusal = "worker not idle";
    else if (mimo26_gpu_worker_position(runtime->worker) != (uint64_t)count)
        refusal = "worker position does not match the prefix length";
    if (refusal != NULL) {
        server_log("INFO", "prefix.publish.declined", runtime->inflight_id,
                   "reason=%s tokens=%zu position=%llu floor=%u", refusal, count,
                   (unsigned long long)mimo26_gpu_worker_position(runtime->worker),
                   (unsigned)PREFIX_PUBLISH_MIN_TOKENS);
        return (prefix_publish_outcome){false, false, 0u};
    }
    size_t existing = SIZE_MAX;
    if (k3_prefix_bundle_find_exact(runtime->prefix_bundle, ids, count,
                                    &existing)) {
        return (prefix_publish_outcome){true, false, count};  /* already stored */
    }
    char id[33];
    char state_path[PATH_MAX];
    char error[512];
    if (!k3_prefix_bundle_allocate_state_path(runtime->prefix_bundle, id,
                                              sizeof id, state_path,
                                              sizeof state_path, error,
                                              sizeof error)) {
        fprintf(stderr, "mimo26: checkpoint path refused: %s\n", error);
        return (prefix_publish_outcome){false, false, 0u};
    }
    mimo26_kv_state_info info;
    if (mimo26_gpu_worker_export_state(runtime->worker, state_path, &info,
                                       error, sizeof error) !=
        MIMO26_GPU_WORKER_OK) {
        fprintf(stderr, "mimo26: checkpoint export failed: %s\n", error);
        (void)remove(state_path);
        return (prefix_publish_outcome){false, false, 0u};
    }
    k3_engine_state_file_info state_info;
    prefix_state_info(&info, &state_info);
    k3_prefix_bundle_snapshot snapshot;
    memset(&snapshot, 0, sizeof snapshot);
    snapshot.tokens = ids;
    snapshot.token_count = count;
    if (!k3_prefix_bundle_publish(runtime->prefix_bundle, id, state_path,
                                  &snapshot, &state_info, error,
                                  sizeof error)) {
        fprintf(stderr, "mimo26: checkpoint publish failed: %s\n", error);
        (void)remove(state_path);
        return (prefix_publish_outcome){false, false, 0u};
    }
    runtime->prefix_disk_publishes++;
    runtime->prefix_entries = k3_prefix_bundle_count(runtime->prefix_bundle);
    fprintf(stderr, "mimo26: checkpoint published, %zu tokens, %.1f MiB, "
            "%.3f s\n", count,
            (double)info.file_bytes / (1024.0 * 1024.0), info.wall_seconds);
    return (prefix_publish_outcome){true, false, count};
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

/*
 * Rendered once and used twice: /health for anything that can safely poll,
 * and the status file for the dashboard, which cannot. Two renderers would
 * drift, and the file is the one nobody would notice going stale.
 */
static int render_health(server_runtime *runtime, char *body, size_t limit)
{
    mimo26_gpu_worker_stats stats;
    mimo26_gpu_worker_get_stats(runtime->worker, &stats);
    const mimo26_slot *slot = &runtime->slot;
    /*
     * `ready` is the liveness answer a load balancer needs: healthy AND able
     * to start work now. A quarantined or draining worker reports not ready
     * while still answering this endpoint, which is the distinction that
     * makes the check useful.
     */
    const int size = snprintf(
        body, limit,
        "{\"status\":\"%s\",\"ready\":%s,\"model\":\"%s\","
        "\"version\":\"" MOONSHINE_VERSION "\",\"auth\":\"%s\","
        "\"pid\":%d,\"stock_profile\":%s,"
        "\"phase\":\"%s\",\"context\":%zu,\"prefill_chunk\":%u,\"expert_lookahead\":%s,"
        "\"expert_major\":%s,\"expert_weight_reuse\":%s,\"expert_slots\":%u,"
        "\"kv_prefix_reuse\":%s,\"prefix_hits\":%llu,"
        "\"prefix_misses\":%llu,\"prefix_tokens_saved\":%llu,"
        "\"prefix_disk_hits\":%llu,\"prefix_disk_publishes\":%llu,"
        "\"prefix_disk_entries\":%zu,"
        "\"request_deadline_seconds\":%u,\"min_headroom_gib\":%u,"
        "\"retain_experts\":%s,\"served\":%llu,\"tokens\":%llu,"
        "\"expert_accesses\":%llu,\"expert_hits\":%llu,\"expert_uploads\":%llu,"
        "\"expert_hit_rate\":%.4f,\"resident_gib\":%.2f,"
        "\"admitted\":%llu,\"rejected_busy\":%llu,"
        "\"rejected_quarantined\":%llu,\"cancelled\":%llu,"
        "\"deadline_stops\":%llu,\"faults\":%llu,\"recoveries\":%llu,"
        /* Activity. inflight is the only field that answers "is it working
         * right now"; everything else describes the last completed request. */
        /* inflight_since_epoch, not an elapsed value: the document is only
         * rewritten at request start and end, so a duration frozen at write
         * time reads 0.0 for the whole request. The reader subtracts. */
        "\"inflight\":%s,\"inflight_id\":\"%s\","
        "\"inflight_since_epoch\":%.3f,\"inflight_prompt_tokens\":%zu,"
        "\"now_epoch\":%.3f,"
        "\"last_id\":\"%s\",\"last_prompt_tokens\":%zu,"
        "\"last_evaluated_tokens\":%zu,"
        "\"last_completion_tokens\":%zu,\"last_reused_tokens\":%zu,"
        "\"last_from_disk\":%s,\"last_finish\":\"%s\","
        "\"last_prefill_seconds\":%.2f,\"last_decode_seconds\":%.2f,"
        "\"last_total_seconds\":%.2f,"
        "\"last_prefill_tps\":%.2f,\"last_decode_tps\":%.2f,"
        /* Rolling, so a rate is not one sample. Prefill excludes tokens that
         * were restored rather than evaluated, or reuse would read as an
         * impossibly fast prefill instead of as avoided work. */
        "\"prefill_tps\":%.2f,\"decode_tps\":%.2f}",
        slot->phase == MIMO26_SLOT_QUARANTINED ? "degraded" : "ok",
        mimo26_slot_ready(slot) ? "true" : "false", MODEL_ID,
        runtime->api_key_set ? "on" : "off",
        (int)getpid(), runtime->stock_profile ? "true" : "false",
        mimo26_slot_phase_name(slot->phase), runtime->context_capacity,
        (unsigned)runtime->prefill_chunk, runtime->expert_lookahead ? "true" : "false",
        runtime->expert_major ? "true" : "false",
        runtime->expert_weight_reuse ? "true" : "false",
        (unsigned)runtime->expert_slots_per_layer,
        runtime->kv_prefix_reuse ? "true" : "false",
        (unsigned long long)runtime->prefix_hits,
        (unsigned long long)runtime->prefix_misses,
        (unsigned long long)runtime->prefix_tokens_saved,
        (unsigned long long)runtime->prefix_disk_hits,
        (unsigned long long)runtime->prefix_disk_publishes,
        runtime->prefix_entries,
        (unsigned)runtime->request_deadline_seconds,
        runtime->min_headroom_gib,
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
        (unsigned long long)slot->recoveries,
        runtime->inflight ? "true" : "false",
        runtime->inflight ? runtime->inflight_id : "",
        runtime->inflight ? runtime->inflight_since_epoch : 0.0,
        runtime->inflight ? runtime->inflight_prompt : (size_t)0,
        wall_clock_epoch(),
        runtime->last_id, runtime->last_prompt_tokens,
        runtime->last_evaluated_tokens,
        runtime->last_completion_tokens, runtime->last_reused_tokens,
        runtime->last_from_disk ? "true" : "false", runtime->last_finish,
        runtime->last_prefill_seconds, runtime->last_decode_seconds,
        runtime->last_total_seconds,
        runtime->last_prefill_seconds > 0.0
            ? (double)runtime->last_evaluated_tokens
                  / runtime->last_prefill_seconds : 0.0,
        runtime->last_decode_seconds > 0.0
            ? (double)runtime->last_completion_tokens
                  / runtime->last_decode_seconds : 0.0,
        runtime->window_prefill_seconds > 0.0
            ? runtime->window_prefill_tokens / runtime->window_prefill_seconds
            : 0.0,
        runtime->window_decode_seconds > 0.0
            ? runtime->window_decode_tokens / runtime->window_decode_seconds
            : 0.0);
    return size;
}

static void send_health(int fd, server_runtime *runtime)
{
    char body[2048];
    const int size = render_health(runtime, body, sizeof body);
    if (size > 0 && (size_t)size < sizeof body) {
        send_response(fd, 200, "OK", "application/json", body, (size_t)size);
    } else {
        send_error(fd, 500, "Internal Server Error", "health_unavailable",
                   "could not render the health document");
    }
}

/*
 * Publish the same document to a file, atomically, for observers that must
 * not connect. Written at request start and end, so "working now" is visible
 * rather than inferred from a gap between samples. Failures are silent by
 * design: losing a status write must never disturb serving.
 */
static void write_status_file(server_runtime *runtime)
{
    if (runtime->status_file == NULL) {
        return;
    }
    char body[2048];
    const int size = render_health(runtime, body, sizeof body);
    if (size <= 0 || (size_t)size >= sizeof body) {
        return;
    }
    char temporary[PATH_MAX];
    if (snprintf(temporary, sizeof temporary, "%s.new",
                 runtime->status_file) >= (int)sizeof temporary) {
        return;
    }
    const int handle = open(temporary, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC,
                            0644);
    if (handle < 0) {
        return;
    }
    const bool wrote = write(handle, body, (size_t)size) == (ssize_t)size &&
                       write(handle, "\n", 1) == 1;
    close(handle);
    if (!wrote || rename(temporary, runtime->status_file) != 0) {
        (void)remove(temporary);
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
    if (request->max_tokens > g_max_output_tokens) {
        request->max_tokens = g_max_output_tokens;
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
        /*
         * An assistant turn that only made tool calls has no text, and the
         * OpenAI shape for it is content null or content absent. Both were
         * refused, so a caller could get tool calls out of this server and
         * then be unable to send the results back -- the whole point of
         * making them. Accepted as empty, and only when tool_calls is present:
         * a user message with no content is a client bug worth surfacing
         * rather than silently treating as blank.
         */
        const int32_t message_calls = k3_json_object_get(&document, m,
                                                         "tool_calls");
        const bool content_is_null =
            content >= 0 && document.tokens[content].type == K3_JSON_NULL;
        const bool calls_without_text =
            message_calls >= 0 && (content < 0 || content_is_null);
        if (role < 0 || (content < 0 && !calls_without_text)) {
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
        if (calls_without_text) {
            content_text = strdup("");
            if (content_text == NULL) {
                REFUSE("invalid_request", "content is too large");
            }
        } else if (document.tokens[content].type == K3_JSON_ARRAY) {
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
                        "\"delta\":{},\"finish_reason\":\"%s\","
                        "\"x_moonshine_finish\":\"%s\"}]}\n\n",
                        state->id, state->created, MODEL_ID,
                        mimo26_slot_wire_finish_reason(finish_reason), finish_reason);
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

/*
 * One delta frame carrying the parsed tool calls.
 *
 * A streaming turn used to forward the model's own call syntax as content and
 * never emit this, so a caller saw <function=...> as prose -- exactly what
 * the parser exists to prevent. `calls_json` is the same array the
 * non-streaming response builds, so both shapes agree.
 */
static bool stream_tool_calls(response_state *state, const char *calls_json)
{
    char *frame = NULL;
    size_t used = 0, capacity = 0;
    char error[256];
    char head[512];
    const int size = snprintf(head, sizeof head,
                              "data: {\"id\":\"%s\",\"object\":"
                              "\"chat.completion.chunk\",\"created\":%ld,"
                              "\"model\":\"%s\",\"choices\":[{\"index\":0,"
                              "\"delta\":{\"tool_calls\":",
                              state->id, state->created, MODEL_ID);
    if (size <= 0 || (size_t)size >= sizeof head) {
        return false;
    }
    if (!append_json(&frame, &used, &capacity, head) ||
        !append_json(&frame, &used, &capacity, calls_json) ||
        !append_json(&frame, &used, &capacity,
                     "},\"finish_reason\":null}]}\n\n")) {
        free(frame);
        return false;
    }
    (void)error;
    const bool ok = send_all(state->fd, frame, used);
    free(frame);
    return ok;
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
    /*
     * Tokens this prefill actually evaluated, as last reported by the progress
     * callback.
     *
     * Without it, request.prefill.complete logged the whole remaining prompt as
     * "evaluated" even when the deadline had cut the prefill short, and divided
     * by it: a 46,346-token prompt that reached 17,664 tokens in 1,819 s was
     * recorded as "evaluated=46346 rate=25.48" when the real rate was ~9.7 t/s.
     * That is the same error as dividing a prompt by a truncated wall clock and
     * calling it throughput -- it inflates the number precisely when the run
     * failed. See [[perf-screens-need-interleaved-baselines]].
     */
    size_t evaluated;
    /*
     * Present when the response is streaming, so prefill can keep the
     * connection alive. A cold prefill emits no tokens for minutes, and a
     * client that sees no bytes at all concludes the request is dead --
     * undici, which is what this lane's harness uses, gives up after 300 s of
     * silence. A 9,597-token prompt took 302 s and was cancelled one second
     * after prefill finished, having done all the work.
     */
    response_state *stream;
    double last_keepalive;
    /* Prefill's own start, not the request's: the dashboard divides evaluated
     * tokens by this to show a live rate, and folding in tokenization would
     * understate it. */
    double prefill_started;
} prefill_request_context;

static bool prefill_progress(void *context, size_t done, size_t total)
{
    prefill_request_context *request = (prefill_request_context *)context;
    server_runtime *runtime = request->runtime;
    /* Unconditionally, ahead of the throttle below: the last value seen is what
     * a stopped prefill has to report, and it must not depend on whether the
     * tick happened to fire on the final chunk. */
    request->evaluated = done;
    (void)total;
    if (request->stop == MIMO26_SLOT_CONTINUE) {
        request->stop = request_control(runtime, request->client);
        if (request->stop == MIMO26_SLOT_CONTINUE) {
            refuse_backlog(runtime);
            request->stop = request_control(runtime, request->client);
        }
    }
    /*
     * One throttled tick serves two consumers, because both need the same
     * thing at the same cadence and neither should drive the other.
     *
     *   the client -- an SSE comment, ignored by every parser but bytes on
     *   the wire, which is what resets a transport idle timer;
     *   the observer -- a progress record, so a prefill that runs for
     *   fifteen minutes is visible as it goes rather than only once it ends.
     *
     * Ten seconds: short enough to beat an idle timeout by a wide margin,
     * long enough that a long prefill costs tens of lines rather than
     * thousands.
     */
    if (request->stop == MIMO26_SLOT_CONTINUE) {
        const double now = now_seconds();
        if (now - request->last_keepalive >= 10.0) {
            request->last_keepalive = now;
            /* Logged whether or not the response streams: a non-streaming
             * request is exactly as slow and exactly as opaque. */
            server_log("INFO", "request.prefill.progress",
                       runtime->inflight_id,
                       "completed=%zu total=%zu elapsed=%.3f unit=token",
                       done, total, now - request->prefill_started);
            if (request->stream != NULL && request->stream->headers_sent) {
                char note[96];
                const int size = snprintf(note, sizeof note,
                                          ": prefill %zu/%zu\n\n", done,
                                          total);
                if (size > 0 && !send_all(request->stream->fd, note,
                                          (size_t)size)) {
                    /* The client is gone; stop rather than finish work
                     * nobody will read. */
                    request->stop = MIMO26_SLOT_STOP_CANCELLED;
                }
            }
        }
    }
    return request->stop == MIMO26_SLOT_CONTINUE;
}

/*
 * Take the tool calls out of a completed turn.
 *
 * Shared by both response shapes on purpose. This lived inline in the
 * non-streaming branch, which is why the streaming branch had no tool calls
 * at all and forwarded the model's own syntax as prose instead. One
 * implementation means the two shapes cannot disagree about what a call is.
 *
 * On success *text becomes the visible remainder, *finish becomes
 * "tool_calls" when any were found, and the returned string is the OpenAI
 * array. Returns NULL when there are none. Caller frees.
 */
static char *take_tool_calls(const char *id, char **text, size_t *text_used,
                             const char **finish, char *error,
                             size_t error_size)
{
    if (*text == NULL) {
        return NULL;
    }
    char *visible = NULL;
    mimo26_parsed_tool_call *parsed = NULL;
    size_t parsed_count = 0;
    if (!mimo26_tokenizer_parse_tool_calls(*text, &visible, &parsed,
                                           &parsed_count, error,
                                           error_size)) {
        return NULL;
    }
    free(*text);
    *text = visible;
    *text_used = strlen(visible);
    if (parsed_count == 0) {
        mimo26_tool_calls_free(parsed, parsed_count);
        return NULL;
    }
    *finish = "tool_calls";
    char *calls = NULL;
    size_t used = 0, capacity = 0;
    append_json(&calls, &used, &capacity, "[");
    for (size_t i = 0; i < parsed_count; i++) {
        char *escaped_name = NULL, *escaped_args = NULL;
        size_t ignored = 0;
        k3_json_escape(parsed[i].name, strlen(parsed[i].name), &escaped_name,
                       &ignored, error, error_size);
        k3_json_escape(parsed[i].arguments_json,
                       strlen(parsed[i].arguments_json), &escaped_args,
                       &ignored, error, error_size);
        char entry[512];
        snprintf(entry, sizeof entry,
                 "%s{\"id\":\"call_%s_%zu\",\"type\":\"function\","
                 "\"function\":{\"name\":%s,\"arguments\":%s}}",
                 i ? "," : "", id, i,
                 escaped_name != NULL ? escaped_name : "\"\"",
                 escaped_args != NULL ? escaped_args : "\"{}\"");
        append_json(&calls, &used, &capacity, entry);
        free(escaped_name);
        free(escaped_args);
    }
    append_json(&calls, &used, &capacity, "]");
    mimo26_tool_calls_free(parsed, parsed_count);
    return calls;
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
        mimo26_slot_admit(&runtime->slot, started,
                          runtime->request_deadline_seconds,
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

    /*
     * Decide reuse from the resident prefix, then drop the claim immediately.
     * It is republished only where this handler completes cleanly, so every
     * early return, fault and cancellation below leaves the next request to
     * reset and prefill in full.
     */
    size_t reuse = plan_prefix_reuse(runtime, prompt.ids, prompt.count);
    resident_invalidate(runtime);
    bool from_disk_reuse = false;
    if (reuse == 0) {
        /* Nothing resident extends this prompt; try the stored prefixes. */
        reuse = restore_prefix_from_disk(runtime, prompt.ids, prompt.count);
        from_disk_reuse = reuse > 0;
    }
    if (reuse > 0) {
        runtime->prefix_hits++;
        runtime->prefix_tokens_saved += (uint64_t)reuse;
    } else {
        runtime->prefix_misses++;
        if (!reset_request_worker(runtime, error, sizeof error)) {
            fprintf(stderr, "mimo26: request reset refused: %s\n", error);
            send_error(fd, 500, "Internal Server Error", "decode_failed",
                       "the worker refused context reset and is quarantined");
            mimo26_token_buffer_free(&prompt);
            chat_request_free(&request);
            return;
        }
    }
    runtime->inflight = true;
    runtime->inflight_since = started;
    runtime->inflight_since_epoch = wall_clock_epoch();
    runtime->inflight_prompt = prompt.count;
    snprintf(runtime->inflight_id, sizeof runtime->inflight_id,
             "chatcmpl-mimo26-%llu",
             (unsigned long long)runtime->slot.request_id);
    server_log("INFO", "request.start", runtime->inflight_id,
               "prompt=%zu tools=%d reasoning=%s", prompt.count,
               request.tools_json != NULL ? 1 : 0,
               request.enable_thinking ? "on" : "off");
    server_log("INFO", "request.prefill.start", runtime->inflight_id,
               "prompt=%zu evaluated=%zu reused=%zu reuse=%s", prompt.count,
               prompt.count - reuse, reuse,
               reuse == 0 ? "none" : (from_disk_reuse ? "disk" : "resident"));
    write_status_file(runtime);

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
    /* Split on purpose: one number cannot distinguish a slow prompt from slow
     * generation, and they have different causes and different fixes. */
    double prefill_seconds = 0.0;
    double decode_seconds = 0.0;
    const double prefill_started = now_seconds();
    /*
     * Open the stream BEFORE prefill, not after. Held until prefill finished,
     * the whole prompt was evaluated behind a connection that had received no
     * bytes at all, and a client with an ordinary idle timeout hung up just
     * as the work completed. The cost is that a prefill failure can no longer
     * carry an HTTP status, which the failure paths below now handle.
     */
    if (state.streaming && !stream_begin(&state)) {
        mimo26_slot_cancel(&runtime->slot);
    }
    /* Designated, not positional: this was positional, so adding a field in the
     * middle of the struct silently shifted &state into it. */
    prefix_publish_outcome checkpoint = {false, false, 0u};
    prefill_request_context progress_context = {
        .runtime = runtime,
        .client = fd,
        .stop = request_control(runtime, fd),
        .evaluated = 0u,
        .stream = state.streaming ? &state : NULL,
        .last_keepalive = now_seconds(),
        .prefill_started = now_seconds()};
    if (progress_context.stop != MIMO26_SLOT_CONTINUE) {
        /* No work or logits to consume when already stopped before prefill. */
    } else if (mimo26_gpu_worker_prefill(runtime->worker, prompt.ids + reuse,
                                  prompt.count - reuse,
                                  logits, prefill_progress, &progress_context, error,
                                  sizeof error) != MIMO26_GPU_WORKER_OK) {
        failed = true;
    } else if (progress_context.stop == MIMO26_SLOT_CONTINUE) {
        /* A stopped prefill returns OK without producing final logits. */
        next = mimo26_gpu_worker_argmax(logits);
        /* The prompt is now evaluated and nothing has been generated on top,
         * which is exactly the state a later turn can continue from. */
        /*
         * The WHOLE prompt, deliberately -- not capped.
         *
         * A capped prefix looks attractive, because a whole-prompt checkpoint is
         * inadmissible for an IDENTICAL retry (exact-prefix admission needs
         * PREFIX_RETRY_MIN_SUFFIX tokens of suffix). Measured at depth: such a
         * retry reused 0 and re-prefilled for 944 s.
         *
         * But capping is unsound here, and the publisher's own position guard
         * proves it: mimo26_gpu_worker_export_state takes no length, so it writes
         * whatever the worker holds. Publishing count = position - 2 would record
         * metadata claiming N-2 tokens against a state file holding N, and the
         * guard refuses -- which it did, silently declining every publish until
         * this was reverted.
         *
         * Serving an identical retry from a whole-prompt checkpoint needs
         * TRUNCATING import, which does not exist. Left open. The case that
         * actually matters is unaffected: the next TURN continues strictly after
         * the assistant opener, so this prefix admits it.
         */
        checkpoint = publish_prefix_checkpoint(runtime, prompt.ids, prompt.count);
        checkpoint.resumable = checkpoint.stored &&
                               checkpoint.retained + PREFIX_RETRY_MIN_SUFFIX <=
                                   prompt.count;
    } else if (progress_context.evaluated > 0u) {
        /*
         * RESUMABLE PREFILL. The prefill ran out of time (or the client left)
         * part-way, so publish what it DID evaluate instead of discarding it.
         *
         * Without this, prefix reuse could never help the case it exists for. A
         * checkpoint was published only on completion, so a prompt too long for
         * the deadline re-prefilled from zero on every retry and died at the same
         * token: the deadline prevented the very checkpoint that would beat the
         * deadline. Observed on 2026-09-27 with a 46,346-token prompt that
         * reached 17,664 tokens in 1,800 s, three times over, publishes = 0.
         *
         * With it, a long prompt is ingested across attempts -- ~17K, then
         * resuming to ~35K, then complete -- each one inside the deadline.
         *
         * Safe because a chunk is transactional across all 48 layers, so the
         * only yield point is BETWEEN chunks: `evaluated` is chunk-aligned and
         * the committed KV is exactly what a fresh prefill of that prefix
         * produces, which is the property mimo26_prefix_reuse_gate already
         * checks across FRESH, CONTINUE and RESTORE. And it is not taken on
         * trust -- publish_prefix_checkpoint refuses unless the worker is idle
         * and its position equals the count passed here, so a mistake in this
         * arithmetic declines to publish rather than storing a mislabelled
         * prefix.
         *
         * reuse + evaluated, because the worker was handed the prompt past the
         * reused prefix and counts from there, while a checkpoint is keyed by
         * the absolute token sequence.
         */
        /*
         * Exactly what was evaluated -- which equals the worker's position, so
         * the publisher's guard admits it.
         *
         * An earlier version capped this to leave PREFIX_RETRY_MIN_SUFFIX tokens,
         * so that an identical retry would be admissible. That is unsound for the
         * same reason as the completed path above: export writes the worker's
         * whole state, so a smaller recorded count contradicts the file. The cap
         * never fired in testing because `evaluated` rarely reaches the end, and
         * it would have silently declined the publish if it ever had.
         */
        checkpoint = publish_prefix_checkpoint(
            runtime, prompt.ids, reuse + progress_context.evaluated);
        checkpoint.resumable = checkpoint.stored && checkpoint.retained > reuse &&
                               checkpoint.retained + PREFIX_RETRY_MIN_SUFFIX <=
                                   prompt.count;   /* honest, not enforced */
    }
    prefill_seconds = now_seconds() - prefill_started;
    /*
     * A stopped prefill reports what it evaluated, not what it was asked to.
     * `stopped` is also logged so a reader never has to infer from a rate why
     * evaluated is short of the prompt.
     */
    const bool prefill_stopped = progress_context.stop != MIMO26_SLOT_CONTINUE;
    const size_t prefill_evaluated =
        prefill_stopped ? progress_context.evaluated : prompt.count - reuse;
    server_log("INFO", "request.prefill.complete", runtime->inflight_id,
               "evaluated=%zu of=%zu reused=%zu stopped=%s seconds=%.3f rate=%.2f",
               prefill_evaluated, prompt.count - reuse, reuse,
               prefill_stopped ? "true" : "false", prefill_seconds,
               prefill_seconds > 0.0
                   ? (double)prefill_evaluated / prefill_seconds : 0.0);
    server_log("INFO", "request.decode.start", runtime->inflight_id, "");
    const double decode_started = now_seconds();
    if (failed) {
        /* A decode failure means the worker's state is not trusted. */
        mimo26_slot_fault(&runtime->slot);
        fprintf(stderr, "mimo26 request %s: prefill failed: %s\n",
                state.id, error);
        if (state.headers_sent) {
            /*
             * The stream is already open, so the status line is spent. Close it
             * the way a client can detect rather than pretending.
             *
             * This used to send finish_reason "error", which is NOT a value the
             * chat-completions schema defines -- so it reproduced exactly the
             * strict-client rejection that the deadline repair was for, and the
             * slot-stop mapping tests never covered it because this string never
             * passes through the enum. An error FRAME says the same thing in a
             * shape clients parse, and matches the zero-token deadline path.
             */
            char frame[512];
            const int size = snprintf(frame, sizeof frame,
                "data: {\"error\":{\"message\":\"the worker faulted during "
                "prefill and is quarantined\",\"type\":\"server_error\","
                "\"code\":\"prefill_failed\"}}\n\n");
            if (size > 0) send_all(fd, frame, (size_t)size);
            const char *done = "data: [DONE]\n\n";
            send_all(fd, done, strlen(done));
        } else {
            send_error(fd, 500, "Internal Server Error", "decode_failed",
                       "the worker faulted during prefill and is quarantined");
        }
        free(logits);
        mimo26_token_buffer_free(&prompt);
        chat_request_free(&request);
        return;
    }

    mimo26_decode_stream decoder;
    mimo26_decode_stream_init(&decoder);
    const char *finish_reason = "stop";
    size_t produced_tokens = 0;
    /*
     * Seeded with the whole prompt -- including any part reused rather than
     * prefilled, since the KV holds it either way -- then extended with every
     * token decode commits.
     */
    token_trail trail = {NULL, 0, 0, true};
    for (size_t i = 0; i < prompt.count; i++) {
        trail_push(&trail, prompt.ids[i]);
    }
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
    /*
     * Text between the tool-call markers is the model's call syntax, not
     * prose. It still has to be collected so it can be parsed at the end, but
     * it must not be streamed as content.
     */
    bool in_tool_call = false;
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
            in_tool_call = next == MIMO26_TOK_TOOL_CALL_OPEN;
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
            trail_push(&trail, next);
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
            trail_push(&trail, next);
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
            /*
             * Collected whether or not the response streams. A streaming turn
             * used to forward each piece and keep nothing, which left no text
             * to parse tool calls out of -- so a streaming caller received the
             * model's raw call syntax and no tool_calls at all.
             */
            if (state.streaming && !in_tool_call) {
                if (!stream_chunk(&state, piece, piece_size, NULL)) {
                    mimo26_slot_cancel(&runtime->slot);
                }
            }
            {
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
        trail_push(&trail, next);

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
        /*
         * The call syntax was withheld from the content stream while it was
         * being generated, so this is where it becomes tool_calls. Emitted
         * before the terminal frame, which then carries finish_reason
         * tool_calls rather than stop.
         */
        char *calls_text = take_tool_calls(state.id, &collected,
                                           &collected_used, &finish_reason,
                                           error, sizeof error);
        if (calls_text != NULL) {
            if (!stream_tool_calls(&state, calls_text)) {
                mimo26_slot_cancel(&runtime->slot);
            }
            free(calls_text);
        }
        if (stop_produced_nothing(finish_reason, produced_tokens)) {
            /* A 200 was already committed with the headers, so the status cannot
             * carry this. An error frame can, and a client that ignores it is no
             * worse off than with a misleading finish_reason. */
            char frame[640];
            const int size = snprintf(frame, sizeof frame,
                "data: {\"error\":{\"message\":\"%s reached after %.0fs before "
                "any token was generated; evaluated %zu of %zu prompt tokens "
                "(%zu reused). %s\",\"type\":\"server_error\","
                "\"code\":\"%s\"}}\n\n",
                !strcmp(finish_reason, "deadline")
                    ? "request deadline" : "server shutdown",
                prefill_seconds, prefill_evaluated + reuse, prompt.count, reuse,
                checkpoint.resumable
                    ? "Retrying the same request resumes from the checkpoint."
                    : (checkpoint.stored
                           ? "A checkpoint was stored but an identical retry "
                             "cannot resume from it."
                           : "No checkpoint was stored."),
                !strcmp(finish_reason, "deadline")
                    ? "deadline_exceeded" : "server_shutdown");
            if (size > 0) send_all(fd, frame, (size_t)size);
        } else {
            stream_chunk(&state, NULL, 0, finish_reason);
        }
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
        if (stop_produced_nothing(finish_reason, produced_tokens)) {
            char detail[512];
            /* Only promise resumption when usable state actually exists. The
             * first version of this message promised it unconditionally, while
             * publication can decline for five reasons and a whole-prompt entry
             * is inadmissible for an identical retry. */
            snprintf(detail, sizeof detail,
                     "%s reached after %.0fs before any token was generated; "
                     "evaluated %zu of %zu prompt tokens (%zu reused). %s",
                     !strcmp(finish_reason, "deadline")
                         ? "request deadline" : "server shutdown",
                     prefill_seconds, prefill_evaluated + reuse, prompt.count,
                     reuse,
                     checkpoint.resumable
                         ? "The evaluated prefix was checkpointed, so retrying "
                           "the same request resumes from it rather than "
                           "restarting."
                         : (checkpoint.stored
                                ? "A checkpoint was stored but an identical "
                                  "retry cannot resume from it; a longer "
                                  "follow-up request can."
                                : "No checkpoint was stored, so a retry starts "
                                  "from the beginning."));
            send_error(fd, 504, "Gateway Timeout",
                       !strcmp(finish_reason, "deadline")
                           ? "deadline_exceeded" : "server_shutdown", detail);
            mimo26_slot_finish(&runtime->slot);
            /* collected and reasoning are released by the shared tail below,
             * which also records last_finish and the deadline counter -- this
             * path must not short-circuit either. */
        } else {
        char *calls_text = take_tool_calls(state.id, &collected,
                                           &collected_used, &finish_reason,
                                           error, sizeof error);
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
        const size_t calls_used = calls_text != NULL ? strlen(calls_text) : 0u;
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
                "\"finish_reason\":\"%s\","
                "\"x_moonshine_finish\":\"%s\"}],\"usage\":{"
                "\"prompt_tokens\":%zu,\"completion_tokens\":%zu,"
                "\"total_tokens\":%zu}}",
                state.id, state.created, MODEL_ID,
                escaped != NULL ? escaped : "\"\"",
                escaped_reasoning != NULL ? escaped_reasoning : "null",
                calls_text != NULL ? calls_text : "null",
                mimo26_slot_wire_finish_reason(finish_reason), finish_reason, prompt.count,
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
        mimo26_slot_finish(&runtime->slot);
        }
    }

    /* Logged without content: a request log that contains the prompt is a
     * transcript of everything anyone ever asked. */
    fprintf(stderr,
            "mimo26 %s prompt=%zu completion=%zu reasoning=%zuB finish=%s "
            "%.2fs\n",
            state.id, prompt.count, produced_tokens, reasoning_used,
            finish_reason, now_seconds() - started);
    decode_seconds = now_seconds() - decode_started;
    {
        const double total = now_seconds() - started;
        snprintf(runtime->last_id, sizeof runtime->last_id, "%s", state.id);
        snprintf(runtime->last_finish, sizeof runtime->last_finish, "%s",
                 finish_reason);
        runtime->last_prompt_tokens = prompt.count;
        /*
         * Evaluated, distinct from requested and from reused.
         *
         * The health rates divided (last_prompt_tokens - last_reused_tokens) by
         * the prefill wall, which is the tokens the request ASKED for. After an
         * early stop that overstates throughput -- by 2.62x on the recorded
         * 46,346-token request that evaluated 17,664 -- and it disagreed with the
         * structured logs, which were fixed to use the evaluated count while
         * these two surfaces were not. The comment two lines below already gives
         * the reason for excluding reused tokens; unevaluated ones are the same
         * argument.
         */
        runtime->last_evaluated_tokens = prefill_evaluated;
        runtime->last_completion_tokens = produced_tokens;
        runtime->last_reused_tokens = reuse;
        runtime->last_from_disk = reuse > 0 && from_disk_reuse;
        runtime->last_prefill_seconds = prefill_seconds;
        runtime->last_decode_seconds = decode_seconds;
        runtime->last_total_seconds = total;
        /* Only tokens actually evaluated count toward a prefill rate; the
         * reused ones were restored, and folding them in would report a rate
         * the engine never achieved. */
        if (prefill_evaluated > 0u && prefill_seconds > 0.0) {
            runtime->window_prefill_tokens += (double)prefill_evaluated;
            runtime->window_prefill_seconds += prefill_seconds;
        }
        if (produced_tokens > 0 && decode_seconds > 0.0) {
            runtime->window_decode_tokens += (double)produced_tokens;
            runtime->window_decode_seconds += decode_seconds;
        }
        runtime->inflight = false;
        mimo26_gpu_worker_stats final;
        mimo26_gpu_worker_get_stats(runtime->worker, &final);
        if (failed) {
            server_log("ERROR", "request.failed", state.id,
                       "prompt=%zu total=%.3f error=\"%s\"", prompt.count,
                       total, error);
        } else {
            server_log("INFO", "request.complete", state.id,
                       "prompt=%zu evaluated=%zu reused=%zu prefill=%.3f "
                       "generated=%zu decode=%.3f rate=%.2f total=%.3f "
                       "finish=%s cache=%llu/%llu",
                       /* prefill_evaluated, not prompt.count - reuse: this
                        * record carried the same fabricated total the
                        * prefill.complete one did, claiming a stopped prefill
                        * had evaluated the whole prompt. */
                       prompt.count, prefill_evaluated, reuse,
                       prefill_seconds, produced_tokens, decode_seconds,
                       decode_seconds > 0.0
                           ? (double)produced_tokens / decode_seconds : 0.0,
                       total, finish_reason,
                       (unsigned long long)final.expert_hits,
                       (unsigned long long)final.expert_accesses);
        }
    }
    runtime->served++;
    if (!failed) {
        /* A request that completed cleanly proves the worker recovered, so
         * the next transient fault gets a full budget again. */
        runtime->recovery_attempts = 0u;
    }

    /*
     * Publish the resident prefix only for a generation that ran to its own
     * end. A cancelled or deadlined request leaves KV consistent but short of
     * this trail -- prefill may have stopped mid-prompt -- so claiming it
     * would describe state the worker does not have. plan_prefix_reuse checks
     * the worker's committed position against this count as well, so a wrong
     * claim costs a reset rather than corrupt history.
     */
    const bool clean_end = !failed && trail.ok &&
                           (!strcmp(finish_reason, "stop") ||
                            !strcmp(finish_reason, "length") ||
                            !strcmp(finish_reason, "tool_calls"));
    if (clean_end && mimo26_gpu_worker_idle(runtime->worker) &&
        mimo26_gpu_worker_position(runtime->worker) ==
            (uint64_t)trail.count) {
        resident_publish(runtime, trail.ids, trail.count);
    } else {
        resident_invalidate(runtime);
    }
    free(trail.ids);

    write_status_file(runtime);
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
    if (argc == 2 && strcmp(argv[1], "--version") == 0) {
        printf("%s %s (MiMo V2.6 Flash serving lane)\n", MOONSHINE_NAME,
               MOONSHINE_VERSION);
        return 0;
    }
    if (argc < 2 || (argc == 2 && strcmp(argv[1], "--help") == 0)) {
        fprintf(stderr,
                "usage: %s ROOT [--port N] [--host H] [--slots N] "
                "[--context N] [--prefill-chunk 0..128] "
                "[--expert-lookahead on|off] [--retain-experts on|off] "
                "[--min-headroom-gib N] [--expert-major on|off] "
                "[--request-deadline-seconds N] [--kv-prefix-reuse on|off] "
                "[--api-key KEY] [--max-output-tokens N] [--version]\n",
                argv[0]);
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
    /*
     * Resolve the environment override before anything reports or admits the
     * profile, so the startup line, the memory guard, /health and execution
     * all describe one configuration.
     */
    {
        char resolve_error[256];
        if (mimo26_gpu_worker_resolve_overrides(&options.worker, resolve_error,
                                                sizeof resolve_error) !=
            MIMO26_GPU_WORKER_OK) {
            fprintf(stderr, "configuration: %s\n", resolve_error);
            return 2;
        }
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
    runtime.expert_major = config.expert_major;
    runtime.expert_weight_reuse = config.expert_weight_reuse;
    runtime.expert_lookahead = config.expert_lookahead;
    runtime.retain_experts = options.retain_experts;
    runtime.request_deadline_seconds = (double)options.request_deadline_seconds;
    runtime.kv_prefix_reuse = options.kv_prefix_reuse;
    runtime.min_headroom_gib = options.min_headroom_gib;
    runtime.status_file = options.status_file;
    g_max_output_tokens = options.max_output_tokens;
    runtime.api_key_set = options.api_key != NULL;
    {
        mimo26_gpu_worker_config stock;
        mimo26_gpu_worker_config_defaults(&stock);
        char ignored[256];
        (void)mimo26_gpu_worker_resolve_overrides(&stock, ignored,
                                                  sizeof ignored);
        runtime.stock_profile =
            config.expert_slots_per_layer == stock.expert_slots_per_layer &&
            config.global_kv_capacity == stock.global_kv_capacity &&
            config.prefill_chunk == stock.prefill_chunk &&
            config.expert_lookahead == stock.expert_lookahead &&
            config.expert_major == stock.expert_major &&
            options.retain_experts == true &&
            options.request_deadline_seconds == 600u &&
            options.min_headroom_gib == 8u;
    }

    mimo26_slot_init(&runtime.slot);

    char error[512];
    if (!mimo26_tokenizer_create(&runtime.tokenizer, root, error,
                                 sizeof error)) {
        fprintf(stderr, "tokenizer: %s\n", error);
        return 1;
    }
    server_log("INFO", "server.load.start", NULL,
               "experts=%u context=%zu chunk=%u",
               (unsigned)config.expert_slots_per_layer,
               config.global_kv_capacity, (unsigned)config.prefill_chunk);
    printf("loading the worker (%u expert slots per layer, context %zu, "
           "prefill chunk %u, expert lookahead %s, expert-major %s, "
           "request deadline %us)\n",
           (unsigned)config.expert_slots_per_layer,
           config.global_kv_capacity, (unsigned)config.prefill_chunk,
           config.expert_lookahead ? "on" : "off",
           config.expert_major ? "on" : "off",
           (unsigned)options.request_deadline_seconds);
    /*
     * Profile-aware memory guard, before anything is allocated.
     *
     * The footprint is predictable from the flags, so refuse a profile this
     * host cannot hold rather than discovering it by swapping -- a swapping
     * host invalidates every measurement taken on it, and the failure
     * otherwise arrives deep inside a multi-minute load.
     */
    {
        const uint64_t planned = mimo26_gpu_worker_planned_bytes(&config);
        const uint64_t available = host_available_bytes();
        const uint64_t floor_bytes =
            (uint64_t)options.min_headroom_gib * 1073741824ull;
        printf("profile needs ~%.1f GiB, host has %.1f GiB available, "
               "floor %u GiB\n", (double)planned / 1073741824.0,
               (double)available / 1073741824.0, options.min_headroom_gib);
        /*
         * An unreadable MemAvailable used to skip the check entirely, so the
         * one case where the host's state is unknown was the one case that
         * loaded unconditionally. Refuse instead: the operator can disable the
         * guard deliberately with --min-headroom-gib 0.
         */
        if (floor_bytes > 0u && available == 0u) {
            fprintf(stderr,
                    "configuration: cannot read MemAvailable, so the %u GiB "
                    "headroom floor cannot be checked against a profile "
                    "needing about %.1f GiB. Pass --min-headroom-gib 0 to load "
                    "without the check.\n",
                    options.min_headroom_gib,
                    (double)planned / 1073741824.0);
            return 1;
        }
        if (floor_bytes > 0u && (planned + floor_bytes) > available) {
            fprintf(stderr,
                    "configuration: this profile needs about %.1f GiB and "
                    "would leave less than the %u GiB floor of the %.1f GiB "
                    "available. Reduce --slots or --context, or pass a lower "
                    "--min-headroom-gib deliberately.\n",
                    (double)planned / 1073741824.0, options.min_headroom_gib,
                    (double)available / 1073741824.0);
            return 1;
        }
    }
    if (mimo26_gpu_worker_create(&runtime.worker, root, &config, error,
                                 sizeof error) != MIMO26_GPU_WORKER_OK) {
        fprintf(stderr, "worker: %s\n", error);
        return 1;
    }
    /*
     * Read the mode back rather than trusting the request. The worker resolves
     * the MIMO26_EXPERT_MAJOR override, so this is the only value that matches
     * execution -- and it is what /health will report.
     */
    {
        const bool effective = mimo26_gpu_worker_expert_major(runtime.worker);
        if (effective != runtime.expert_major) {
            printf("expert-major overridden by environment: %s -> %s\n",
                   runtime.expert_major ? "on" : "off",
                   effective ? "on" : "off");
        }
        runtime.expert_major = effective;
        const bool reuse = mimo26_gpu_worker_expert_weight_reuse(runtime.worker);
        if (reuse != runtime.expert_weight_reuse) {
            printf("expert weight reuse overridden by environment: %s -> %s\n",
                   runtime.expert_weight_reuse ? "on" : "off", reuse ? "on" : "off");
        }
        runtime.expert_weight_reuse = reuse;
    }
    /*
     * Opened after the worker, because the bundle's identity is this KV's
     * geometry -- layer count, per-layer heads, window and capacity. Gating on
     * it means a checkpoint written under a different profile is never even
     * offered, before the file's own identity check gets a chance to refuse it.
     *
     * Geometry alone is NOT a semantic identity, which the 2026-09-26 review
     * caught. Two workers can agree on every dimension and still write
     * different KV for the same tokens if they compute it differently -- the
     * tiled expert kernel does exactly that. Under a geometry-only identity a
     * checkpoint written by the GEMV would be offered to tiled execution and
     * silently restored, corrupting both serving and any A/B comparison.
     *
     * So mix the arithmetic into the identity the bundle gates on: an
     * implementation version that is bumped by hand whenever the numerics
     * change, plus the effective kernel mode. Different arithmetic then refuses
     * the store instead of reading it.
     *
     * STILL OPEN: model identity. Nothing here distinguishes two different
     * checkpoints of compatibly-shaped weights, so a cache directory must not
     * be shared between models. Tracked in the review note.
     */
    if (options.prefix_cache_dir != NULL) {
        /*
         * Separate stores per arithmetic mode, by DIRECTORY.
         *
         * This was a composite CRC mixed into the bundle identity, and that was
         * wrong in a way that disabled publishing entirely: the identity is
         * compared against the exported state file's own model_layout_crc64,
         * and that field is the KV's integrity check -- mimo26_kv import
         * requires it to equal the live cache's layout CRC (mimo26_kv.c:647).
         * So the store expected a composite the state file could never carry,
         * every k3_prefix_bundle_publish was refused with "checkpoint state
         * identity is invalid", and the only symptom was publishes stuck at 0.
         *
         * A directory needs no identity surgery, cannot disagree with the state
         * file, and is legible: an operator can see which store belongs to which
         * kernel. Checkpoints written by one mode are never offered to the other
         * because the other never looks in that directory.
         */
        char store[PATH_MAX];
        const int written = snprintf(store, sizeof store, "%s/k%u-%s",
                                     options.prefix_cache_dir,
                                     (unsigned)MIMO26_KV_NUMERICS_VERSION,
                                     runtime.expert_weight_reuse ? "tiled" : "gemv");
        if (written <= 0 || (size_t)written >= sizeof store) {
            fprintf(stderr, "prefix cache: directory path too long\n");
            return 1;
        }
        (void)mkdir(options.prefix_cache_dir, 0700);
        (void)mkdir(store, 0700);
        const k3_prefix_bundle_identity identity = {
            1u, (uint32_t)config.global_kv_capacity,
            mimo26_gpu_worker_layout_crc64(runtime.worker), false};
        char bundle_error[512];
        if (!k3_prefix_bundle_open(&runtime.prefix_bundle,
                                   store, &identity,
                                   options.prefix_cache_entries,
                                   (uint64_t)options.prefix_cache_gib *
                                       1073741824ull,
                                   bundle_error, sizeof bundle_error)) {
            /*
             * The commonest cause is a context change: the bundle's identity
             * includes the capacity its checkpoints were written at, so
             * every stored prefix becomes unreadable and the store refuses
             * itself. Since 2026-09-26 the expert kernel mode is in the
             * identity too, so toggling --expert-weight-reuse invalidates a
             * store for the same reason and by design. Refusing is right --
             * the alternative is silently restoring KV computed by different
             * arithmetic -- but the operator needs to be told which lever to
             * pull.
             */
            fprintf(stderr, "prefix cache: %s\n", bundle_error);
            fprintf(stderr,
                    "prefix cache: this store was written for a different "
                    "profile than %zu context. Each expert kernel already gets "
                    "its own subdirectory (%s), so this is a geometry change: "
                    "point --prefix-cache-dir somewhere new, or remove %s to "
                    "discard the stored prefixes and start fresh.\n",
                    config.global_kv_capacity, store, store);
            return 1;
        }
        runtime.prefix_entries = k3_prefix_bundle_count(runtime.prefix_bundle);
        printf("prefix cache at %s: %zu stored, %u entries and %u GiB at "
               "most\n", store, runtime.prefix_entries,
               options.prefix_cache_entries, options.prefix_cache_gib);
    }

    mimo26_gpu_worker_stats stats;
    mimo26_gpu_worker_get_stats(runtime.worker, &stats);
    printf("resident %.2f GiB after %.1f s, expert-major %s\n",
           (double)mimo26_gpu_worker_resident_bytes(runtime.worker) /
               1073741824.0,
           stats.load_seconds, runtime.expert_major ? "on" : "off");

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
    write_status_file(&runtime);
    server_log("INFO", "server.ready", NULL,
               "version=\"%s\" experts=%u context=%zu max_output=%u "
               "load=%.1f state=%.2f auth=%s reuse=%s",
               MOONSHINE_VERSION, (unsigned)config.expert_slots_per_layer,
               config.global_kv_capacity, options.max_output_tokens,
               stats.load_seconds,
               (double)mimo26_gpu_worker_resident_bytes(runtime.worker) /
                   1073741824.0,
               runtime.api_key_set ? "on" : "off",
               runtime.kv_prefix_reuse ? "on" : "off");
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
        if (!authorized(http.authorization, options.api_key)) {
            send_error(client, 401, "Unauthorized", "unauthorized",
                       "this server requires Authorization: Bearer <key>");
            request_free(&http);
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
