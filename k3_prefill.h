#ifndef K3_PREFILL_H
#define K3_PREFILL_H

#include "k3_safetensors.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    /*
     * Keep every decode-cache byte live. Routed experts pass through the
     * existing fixed QD2 staging window and are consumed expert-major.
     */
    K3_PREFILL_CACHE_RETAIN = 0,

    /*
     * Cold-start only: lend exactly the planned batch workspace from the
     * empty decode cache. The caller must prove that no warm cache contents
     * exist before selecting this lease.
     */
    K3_PREFILL_CACHE_BORROW_COLD_WORKSPACE = 1,

    /*
     * Warm-memory fallback: lend a slot-aligned tail of the decode cache to
     * the batch workspace after invalidating only the overlapping mappings.
     * The runtime may select this when a separate allocation would violate
     * the host/CMA guard.
     */
    K3_PREFILL_CACHE_BORROW_WARM_WORKSPACE = 2,
} k3_prefill_cache_lease;

typedef struct {
    uint32_t chunk_tokens;
    uint32_t context_remaining;
    uint32_t routed_layers;
    uint32_t routed_layer_sweeps;
    /* Full-store ceilings; actual routed reads depend on router unions. */
    uint32_t expert_read_requests;
    uint32_t shard_segments;

    uint64_t hidden_row_bytes;
    uint64_t attn_res_row_bytes;
    uint64_t main_scratch_bytes;
    uint64_t moe_scratch_bytes;
    uint64_t mla_scratch_bytes;
    uint64_t route_bytes;
    uint64_t expert_index_bytes;
    uint64_t auxiliary_workspace_bytes;
    uint64_t batch_workspace_bytes;
    uint64_t incremental_workspace_bytes;
    uint64_t mla_append_bytes;

    /* Full-store ceilings; runtime selected-union totals are in engine stats. */
    uint64_t routed_layer_payload_bytes;
    uint64_t routed_store_payload_bytes;
    uint64_t routed_store_requested_read_bytes;
    uint64_t routed_store_physical_read_bytes;
    uint64_t staging_window_bytes;
    uint64_t registered_staging_bytes;

    uint64_t decode_cache_bytes;
    uint64_t retained_cache_bytes;
    uint64_t borrowed_cache_bytes;
    uint64_t new_layer_buffer_bytes;
    uint64_t new_device_bytes;
} k3_prefill_plan;

/*
 * Build a payload-free memory and I/O plan for one layer-major K3 chunk.
 *
 * This validates every routed expert's six-tensor physical span while
 * deriving full-store I/O ceilings, the exact sweep count, per-token
 * workspace, MLA cache growth, and cache-lease ownership. Runtime reads only
 * the physical-order union selected by each layer and reports its exact
 * dynamic request/byte ledger in k3_engine_prefill_stats.
 */
bool k3_prefill_plan_build(
        const k3_st_model       *model,
        uint32_t                 chunk_tokens,
        uint32_t                 context_remaining,
        uint16_t                 experts_per_layer,
        uint16_t                 staging_slots,
        k3_prefill_cache_lease   lease,
        k3_prefill_plan         *plan,
        char                    *error,
        size_t                   error_size);

/*
 * Diagnostic experiments may reserve one fixed device scratch region while
 * preserving the same per-token workspace and cache-lease accounting.
 */
bool k3_prefill_plan_build_with_aux_workspace(
        const k3_st_model       *model,
        uint32_t                 chunk_tokens,
        uint32_t                 context_remaining,
        uint16_t                 experts_per_layer,
        uint16_t                 staging_slots,
        k3_prefill_cache_lease   lease,
        uint64_t                 auxiliary_workspace_bytes,
        k3_prefill_plan         *plan,
        char                    *error,
        size_t                   error_size);

#ifdef __cplusplus
}
#endif

#endif
