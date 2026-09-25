#ifndef MIMO26_SERVER_OPTIONS_H
#define MIMO26_SERVER_OPTIONS_H

/* CPU-only parsing. The caller supplies worker defaults; reject ambiguous
 * configuration before loading weights or opening a GPU context. */
#include "mimo26_gpu_worker.h"
#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *root;
    const char *host;
    uint16_t port;
    /*
     * Keep expert payloads and cache mappings across request boundaries.
     * Context is reset regardless -- this is NOT KV reuse, which is
     * kv_prefix_reuse below.
     *
     * On by default as of 2026-09-25, from the serving qualification rather
     * than the warm benchmark: it costs nothing measurable in memory (GTT max
     * and swap identical to the retention-off run, to the MiB) and took the
     * 20-request soak median from 22.4 s to 13.4 s. All seven functional
     * checks pass identically either way.
     *
     * It does change the failure path: a retention refusal quarantines the
     * slot rather than falling back to a cold reset, because pending I/O or a
     * sticky execution fault needs a new worker.
     */
    bool retain_experts;
    /*
     * Minimum host memory, in GiB, that must remain available after the
     * profile's predicted resident footprint. 0 disables the check.
     *
     * A profile's memory cost is a property of the flags, not something to
     * discover by swapping. 8 GiB as of 2026-09-24, the operator's stated
     * minimum for this host, which runs nothing else. It admits the 160-slot
     * 262144-context profile (111.4 GiB predicted, ~8.3 GiB left) and still
     * refuses 176 slots, the configuration measured to swap during allocation.
     *
     * 8 is deliberately close to the bone and is a decision about THIS box. A
     * public default should be higher: the 160-slot profile clears it by only
     * ~0.3 GiB, so a host with less free memory will be refused -- correctly,
     * but the conservative 128-slot profile leaves ~27 GiB at the same context
     * and is the one to ship where the host is unknown.
     */
    uint32_t min_headroom_gib;
    /*
     * Wall-clock budget for one request, in seconds. Checked between prefill
     * chunks and decode steps, so it bounds a slow request rather than a hung
     * one.
     *
     * It is the real ceiling on prompt length, and until 2026-09-24 it was a
     * hardcoded 600 that no flag could reach. Cold prefill measured ~18 tok/s,
     * so 600 s admits about 11K prompt tokens -- an 11,053-token probe spent
     * its entire budget in prefill and returned zero completion tokens. A
     * 262144 context is unreachable in one cold request at any deadline worth
     * setting (~4 hours). It is only reachable by a session that accumulates
     * across turns, which requires --kv-prefix-reuse below; without that every
     * turn re-prefills the whole history and the ceiling is the deadline.
     */
    uint32_t request_deadline_seconds;
    /*
     * Continue from the KV already resident when the new prompt begins with
     * exactly the token sequence it holds, prefilling only the remainder.
     *
     * This is a DIFFERENT feature from --retain-experts, which keeps expert
     * payloads and still clears KV at every request boundary. Without this,
     * a multi-turn client re-prefills its whole history every turn.
     *
     * Default off until reused continuations are shown to match fresh
     * full-prefill results, per the 2026-09-25 review.
     */
    bool kv_prefix_reuse;
    /*
     * Directory holding persisted prefix checkpoints, and the budget they may
     * occupy. NULL disables the on-disk tier, leaving only the in-memory one
     * that continues an exactly-matching resident context.
     *
     * The budget must be set deliberately. The drive that makes sense for this
     * also holds the model weights and another lane's working set, and a
     * checkpoint is not small: about 24 MiB of fixed windowed ring plus
     * 22.5 KiB per token, so roughly 244 MiB at 10K tokens.
     */
    /*
     * Bearer credential. NULL disables the check, which is only tolerable on
     * loopback -- a non-loopback bind without one is refused at startup, the
     * policy K3's server already enforces. MIMO26_API_KEY is consulted when
     * the flag is absent, so the key need not appear in a process listing.
     */
    const char *api_key;
    /*
     * Per-request output ceiling, independent of context. Without it a client
     * can ask for as many tokens as the context allows and occupy the single
     * slot for hours; only the request deadline bounded it.
     */
    uint32_t max_output_tokens;
    const char *prefix_cache_dir;
    uint32_t prefix_cache_gib;
    uint32_t prefix_cache_entries;
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
    parsed.retain_experts = true;
    parsed.min_headroom_gib = 8u;
    parsed.request_deadline_seconds = 600u;
    parsed.kv_prefix_reuse = false;
    parsed.api_key = getenv("MIMO26_API_KEY");
    parsed.max_output_tokens = 8192u;
    parsed.prefix_cache_dir = NULL;
    parsed.prefix_cache_gib = 16u;
    parsed.prefix_cache_entries = 32u;
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
        else if (!strcmp(key,"--request-deadline-seconds")) bit=512;
        else if (!strcmp(key,"--kv-prefix-reuse")) bit=1024;
        else if (!strcmp(key,"--prefix-cache-dir")) bit=2048;
        else if (!strcmp(key,"--prefix-cache-gib")) bit=4096;
        else if (!strcmp(key,"--prefix-cache-entries")) bit=8192;
        else if (!strcmp(key,"--api-key")) bit=16384;
        else if (!strcmp(key,"--max-output-tokens")) bit=32768;
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
        } else if (bit == 2048) {
            if (value[0] != '/') {
                snprintf(error,error_size,
                         "--prefix-cache-dir requires an absolute path");
                return false;
            }
            parsed.prefix_cache_dir=value;
        } else if (bit == 16384) {
            /* An empty key means no key, matching K3, rather than a
             * credential every caller can guess. */
            parsed.api_key = value[0] ? value : NULL;
        } else if (bit == 32 || bit == 64 || bit == 256 || bit == 1024) {
            if (strcmp(value,"on") && strcmp(value,"off")) {
                snprintf(error,error_size,"%s requires on or off",key); return false;
            }
            if (bit == 32) parsed.worker.expert_lookahead=!strcmp(value,"on");
            else if (bit == 256) parsed.worker.expert_major=!strcmp(value,"on");
            else if (bit == 1024) parsed.kv_prefix_reuse=!strcmp(value,"on");
            else parsed.retain_experts=!strcmp(value,"on");
        } else {
            uint64_t low=1,high=UINT32_MAX;
            if (bit == 2) high=65535;
            if (bit == 4) {low=8;high=256;}
            if (bit == 16) {low=0;high=128;}
            if (bit == 128) {low=0;high=512;}
            if (bit == 512) {low=1;high=86400;}
            if (bit == 4096) {low=1;high=4096;}
            if (bit == 8192) {low=1;high=64;}   /* the bundle's own ceiling */
            if (bit == 32768) {low=1;high=65536;}
            if (!mimo26_server_decimal(value,low,high,&number)) {
                snprintf(error,error_size,"invalid integer for %s",key); return false;
            }
            if (bit == 2) parsed.port=(uint16_t)number;
            if (bit == 4) parsed.worker.expert_slots_per_layer=(uint16_t)number;
            if (bit == 8) parsed.worker.global_kv_capacity=(size_t)number;
            if (bit == 16) parsed.worker.prefill_chunk=(uint16_t)number;
            if (bit == 128) parsed.min_headroom_gib=(uint32_t)number;
            if (bit == 512) parsed.request_deadline_seconds=(uint32_t)number;
            if (bit == 4096) parsed.prefix_cache_gib=(uint32_t)number;
            if (bit == 8192) parsed.prefix_cache_entries=(uint32_t)number;
            if (bit == 32768) parsed.max_output_tokens=(uint32_t)number;
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
    /* Refuse a reachable bind with no credential, rather than warning.
     *
     * The whole 127/8 block is loopback, not just 127.0.0.1 -- the first
     * version of this check compared strings and refused a legitimate
     * 127.0.0.5 bind. --host has already validated the address as IPv4, so
     * parsing it again here cannot fail. */
    struct in_addr bound;
    const bool loopback = inet_pton(AF_INET, parsed.host, &bound) == 1 &&
                          (ntohl(bound.s_addr) >> 24) == 127u;
    if (!loopback && parsed.api_key == NULL) {
        snprintf(error,error_size,
                 "a non-loopback bind requires --api-key or MIMO26_API_KEY");
        return false;
    }
    /* A checkpoint directory without reuse would write files nothing reads. */
    if (parsed.prefix_cache_dir != NULL && !parsed.kv_prefix_reuse) {
        snprintf(error,error_size,
                 "--prefix-cache-dir needs --kv-prefix-reuse on");
        return false;
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
