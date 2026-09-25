#ifndef MIMO26_SERVER_OPTIONS_H
#define MIMO26_SERVER_OPTIONS_H

/* CPU-only parsing. The caller supplies worker defaults; reject ambiguous
 * configuration before loading weights or opening a GPU context. */
#include "mimo26_gpu_worker.h"
#include <arpa/inet.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    const char *root;
    const char *host;
    uint16_t port;
    bool retain_experts;
    /*
     * Minimum host memory, in GiB, that must remain available after the
     * profile's predicted resident footprint. 0 disables the check.
     *
     * A profile's memory cost is a property of the flags, not something to
     * discover by swapping. 10 GiB as of 2026-09-24, matching the operator's
     * stated tolerance on a host that runs nothing else: it admits the
     * default 160-slot profile (104.9 GiB predicted, ~16 GiB left) and still
     * refuses 176 slots (114.3 GiB, ~7 GiB left), which is the configuration
     * measured to swap during allocation.
     */
    uint32_t min_headroom_gib;
    mimo26_gpu_worker_config worker;
} mimo26_server_options;

static inline bool mimo26_server_decimal(const char *s, uint64_t low,
                                         uint64_t high, uint64_t *out)
{
    if (!s || !*s) return false;
    uint64_t value = 0;
    for (; *s; ++s) {
        if (*s < '0' || *s > '9') return false;
        unsigned digit = (unsigned)(*s - '0');
        if (digit > high || value > (high - digit) / 10u) return false;
        value = value * 10u + digit;
    }
    if (value < low) return false;
    *out = value;
    return true;
}

static inline bool mimo26_server_parse_options(int argc, char **argv,
        mimo26_server_options *options, char *error, size_t error_size)
{
    if (argc < 2 || !argv[1][0] || argv[1][0] == '-') {
        snprintf(error, error_size, "a model root is required"); return false;
    }
    mimo26_server_options parsed = *options;
    parsed.root = argv[1]; parsed.host = "127.0.0.1"; parsed.port = 8640;
    parsed.retain_experts = false;
    parsed.min_headroom_gib = 10u;
    unsigned seen = 0;
    for (int i = 2; i < argc; i += 2) {
        const char *key = argv[i];
        unsigned bit = 0;
        if (!strcmp(key,"--host")) bit=1;
        else if (!strcmp(key,"--port")) bit=2;
        else if (!strcmp(key,"--slots")) bit=4;
        else if (!strcmp(key,"--context")) bit=8;
        else if (!strcmp(key,"--prefill-chunk")) bit=16;
        else if (!strcmp(key,"--expert-lookahead")) bit=32;
        else if (!strcmp(key,"--retain-experts")) bit=64;
        else if (!strcmp(key,"--min-headroom-gib")) bit=128;
        else if (!strcmp(key,"--expert-major")) bit=256;
        if (!bit || (seen & bit) || i + 1 >= argc) {
            snprintf(error,error_size,"unknown, duplicate or missing-value option: %s",key); return false;
        }
        seen |= bit;
        const char *value = argv[i+1]; uint64_t number = 0;
        if (bit == 1) {
            struct in_addr address;
            if (inet_pton(AF_INET,value,&address) != 1) {
                snprintf(error,error_size,"--host requires an IPv4 address"); return false;
            }
            parsed.host=value;
        } else if (bit == 32 || bit == 64 || bit == 256) {
            if (strcmp(value,"on") && strcmp(value,"off")) {
                snprintf(error,error_size,"%s requires on or off",key); return false;
            }
            if (bit == 32) parsed.worker.expert_lookahead=!strcmp(value,"on");
            else if (bit == 256) parsed.worker.expert_major=!strcmp(value,"on");
            else parsed.retain_experts=!strcmp(value,"on");
        } else {
            uint64_t low=1,high=UINT32_MAX;
            if (bit == 2) high=65535;
            if (bit == 4) {low=8;high=256;}
            if (bit == 16) {low=0;high=128;}
            if (bit == 128) {low=0;high=512;}
            if (!mimo26_server_decimal(value,low,high,&number)) {
                snprintf(error,error_size,"invalid integer for %s",key); return false;
            }
            if (bit == 2) parsed.port=(uint16_t)number;
            if (bit == 4) parsed.worker.expert_slots_per_layer=(uint16_t)number;
            if (bit == 8) parsed.worker.global_kv_capacity=(size_t)number;
            if (bit == 16) parsed.worker.prefill_chunk=(uint16_t)number;
            if (bit == 128) parsed.min_headroom_gib=(uint32_t)number;
        }
    }
    /*
     * Grouped prefill needs the remaining-group schedule; without lookahead it
     * measured slower than per-token. So the default follows lookahead rather
     * than fighting it -- turning lookahead off turns grouping off too -- but
     * asking for both explicitly is a contradiction and is refused. The
     * resolved profile is always printed and exposed on /health, so neither
     * case is silent.
     */
    if (parsed.worker.expert_major && !parsed.worker.expert_lookahead) {
        if (seen & 256u) {
            snprintf(error,error_size,
                     "--expert-major on needs --expert-lookahead on: grouped "
                     "prefill without the remaining-group schedule is slower");
            return false;
        }
        parsed.worker.expert_major = false;
    }
    if (parsed.worker.expert_major && !parsed.worker.prefill_chunk) {
        snprintf(error,error_size,"--expert-major requires layer-major prefill (chunk > 0)");
        return false;
    }
    if (parsed.worker.expert_lookahead && !parsed.worker.prefill_chunk) {
        snprintf(error,error_size,"expert lookahead requires layer-major prefill (chunk > 0)"); return false;
    }
    *options=parsed;
    return true;
}
#endif
