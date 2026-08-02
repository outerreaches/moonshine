#include "k3_prefill.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    K3_PREFILL_CONTEXT = 1048576,
    K3_PREFILL_HIDDEN = 7168,
    K3_PREFILL_KDA_INNER = 12288,
    K3_PREFILL_KDA_HEAD_DIM = 128,
    K3_PREFILL_DENSE_INTERMEDIATE = 33792,
    K3_PREFILL_LATENT = 3584,
    K3_PREFILL_EXPERT_HIDDEN = 3072,
    K3_PREFILL_SHARED_HIDDEN = 6144,
    K3_PREFILL_TOP_K = 16,
    K3_PREFILL_EXPERTS = 896,
    K3_PREFILL_MOE_LAYERS = 92,
    K3_PREFILL_HEADS = 96,
    K3_PREFILL_MLA_Q_LORA = 1536,
    K3_PREFILL_MLA_Q = 96 * 192,
    K3_PREFILL_MLA_CACHE_DIM = 576,
    K3_PREFILL_MLA_LAYERS = 24,
    K3_PREFILL_EXPERT_TENSORS = 6,
    K3_PREFILL_STAGING_BYTES = 17551360,
};

static const uint64_t K3_PREFILL_EXPERT_BYTES = UINT64_C(17547264);

typedef struct {
    uint64_t start;
    uint64_t end;
    uint16_t shard;
} k3_prefill_expert_span;

static void prefill_error(char *error,
                          size_t error_size,
                          const char *format,
                          ...) {
    if (!error || error_size == 0u) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static bool add_u64(uint64_t *total, uint64_t value) {
    if (*total > UINT64_MAX - value) return false;
    *total += value;
    return true;
}

static bool multiply_u64(uint64_t left,
                         uint64_t right,
                         uint64_t *product) {
    if (left != 0u && right > UINT64_MAX / left) return false;
    *product = left * right;
    return true;
}

static bool expert_span(
        const k3_st_model *model,
        uint32_t layer,
        uint32_t expert,
        uint16_t *shard,
        uint64_t *start,
        uint64_t *end,
        char *error,
        size_t error_size) {
    static const char *suffix[K3_PREFILL_EXPERT_TENSORS] = {
        "w1.weight_packed", "w1.weight_scale",
        "w2.weight_packed", "w2.weight_scale",
        "w3.weight_packed", "w3.weight_scale",
    };
    const k3_st_tensor *tensor[K3_PREFILL_EXPERT_TENSORS];
    char name[256];
    for (uint32_t index = 0;
         index < K3_PREFILL_EXPERT_TENSORS; index++) {
        const int length = snprintf(
            name, sizeof(name),
            "language_model.model.layers.%u.block_sparse_moe."
            "experts.%u.%s", layer, expert, suffix[index]);
        if (length < 0 || (size_t)length >= sizeof(name)) {
            prefill_error(error, error_size,
                          "layer-%u expert-%u tensor name overflow",
                          layer, expert);
            return false;
        }
        tensor[index] = k3_st_find(model, name);
        if (!tensor[index]) {
            prefill_error(error, error_size,
                          "missing layer-%u expert-%u tensor %s",
                          layer, expert, suffix[index]);
            return false;
        }
        if (tensor[index]->dtype != K3_ST_DTYPE_U8 ||
            (index > 0u &&
             (tensor[index]->shard != tensor[0]->shard ||
              tensor[index - 1u]->physical_offset +
                  tensor[index - 1u]->byte_length !=
                      tensor[index]->physical_offset))) {
            prefill_error(error, error_size,
                          "layer-%u expert-%u tensors are not one "
                          "contiguous U8 span", layer, expert);
            return false;
        }
    }
    *shard = tensor[0]->shard;
    *start = tensor[0]->physical_offset;
    *end =
        tensor[K3_PREFILL_EXPERT_TENSORS - 1u]->physical_offset +
        tensor[K3_PREFILL_EXPERT_TENSORS - 1u]->byte_length;
    if (*end < *start ||
        *end - *start != K3_PREFILL_EXPERT_BYTES) {
        prefill_error(error, error_size,
                      "layer-%u expert-%u physical span is invalid",
                      layer, expert);
        return false;
    }
    return true;
}

static int compare_expert_span(const void *left, const void *right) {
    const k3_prefill_expert_span *a =
        (const k3_prefill_expert_span *)left;
    const k3_prefill_expert_span *b =
        (const k3_prefill_expert_span *)right;
    if (a->shard != b->shard) {
        return a->shard < b->shard ? -1 : 1;
    }
    if (a->start == b->start) return 0;
    return a->start < b->start ? -1 : 1;
}

static bool derive_workspace(uint32_t chunk_tokens,
                             k3_prefill_plan *plan) {
    const uint64_t bf16 = sizeof(uint16_t);
    const uint64_t hidden = K3_PREFILL_HIDDEN * bf16;
    const uint64_t inner = K3_PREFILL_KDA_INNER * bf16;
    const uint64_t dense =
        K3_PREFILL_DENSE_INTERMEDIATE * bf16;

    const uint64_t main_per_token =
        5u * hidden +
        10u * inner +
        K3_PREFILL_KDA_HEAD_DIM * bf16 +
        K3_PREFILL_HEADS * bf16 +
        3u * dense;
    const uint64_t hidden_per_token = 2u * hidden;
    const uint64_t attn_res_per_token = 8u * hidden;
    const uint64_t moe_per_token =
        (uint64_t)K3_PREFILL_EXPERTS * sizeof(float) +
        ((uint64_t)K3_PREFILL_LATENT +
         3u * K3_PREFILL_EXPERT_HIDDEN +
         (uint64_t)K3_PREFILL_TOP_K * K3_PREFILL_LATENT +
         2u * K3_PREFILL_LATENT +
         K3_PREFILL_HIDDEN +
         3u * K3_PREFILL_SHARED_HIDDEN +
         K3_PREFILL_HIDDEN) * bf16;
    const uint64_t mla_per_token =
        (2u * K3_PREFILL_MLA_Q_LORA +
         K3_PREFILL_MLA_Q +
         K3_PREFILL_MLA_CACHE_DIM +
         (uint64_t)K3_PREFILL_HEADS *
             K3_PREFILL_MLA_CACHE_DIM +
         3u * K3_PREFILL_KDA_INNER) * bf16;
    const uint64_t route_per_token =
        (uint64_t)K3_PREFILL_TOP_K *
        (sizeof(uint32_t) + sizeof(float));
    const uint64_t expert_index_per_token =
        2u * sizeof(uint32_t);

    if (!multiply_u64(
            hidden_per_token, chunk_tokens,
            &plan->hidden_row_bytes) ||
        !multiply_u64(
            attn_res_per_token, chunk_tokens,
            &plan->attn_res_row_bytes) ||
        !multiply_u64(
            main_per_token, chunk_tokens,
            &plan->main_scratch_bytes) ||
        !multiply_u64(
            moe_per_token, chunk_tokens,
            &plan->moe_scratch_bytes) ||
        !multiply_u64(
            mla_per_token, chunk_tokens,
            &plan->mla_scratch_bytes) ||
        !multiply_u64(
            route_per_token, chunk_tokens,
            &plan->route_bytes) ||
        !multiply_u64(
            expert_index_per_token, chunk_tokens,
            &plan->expert_index_bytes)) {
        return false;
    }
    plan->batch_workspace_bytes = 0u;
    if (!add_u64(
            &plan->batch_workspace_bytes,
            plan->hidden_row_bytes) ||
        !add_u64(
            &plan->batch_workspace_bytes,
            plan->attn_res_row_bytes) ||
        !add_u64(
            &plan->batch_workspace_bytes,
            plan->main_scratch_bytes) ||
        !add_u64(
            &plan->batch_workspace_bytes,
            plan->moe_scratch_bytes) ||
        !add_u64(
            &plan->batch_workspace_bytes,
            plan->mla_scratch_bytes) ||
        !add_u64(
            &plan->batch_workspace_bytes,
            plan->route_bytes) ||
        !add_u64(
            &plan->batch_workspace_bytes,
            plan->expert_index_bytes)) {
        return false;
    }
    const uint64_t one_token_workspace =
        hidden_per_token + attn_res_per_token +
        main_per_token + moe_per_token +
        mla_per_token + route_per_token +
        expert_index_per_token;
    plan->incremental_workspace_bytes =
        plan->batch_workspace_bytes > one_token_workspace ?
            plan->batch_workspace_bytes - one_token_workspace : 0u;
    return multiply_u64(
        (uint64_t)chunk_tokens *
            K3_PREFILL_MLA_LAYERS,
        K3_PREFILL_MLA_CACHE_DIM * bf16,
        &plan->mla_append_bytes);
}

bool k3_prefill_plan_build_with_aux_workspace(
        const k3_st_model *model,
        uint32_t chunk_tokens,
        uint32_t context_remaining,
        uint16_t experts_per_layer,
        uint16_t staging_slots,
        k3_prefill_cache_lease lease,
        uint64_t auxiliary_workspace_bytes,
        k3_prefill_plan *plan,
        char *error,
        size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (plan) memset(plan, 0, sizeof(*plan));
    if (!model || !plan || model->shard_count == 0u ||
        chunk_tokens == 0u ||
        chunk_tokens > context_remaining ||
        context_remaining > K3_PREFILL_CONTEXT ||
        experts_per_layer == 0u ||
        staging_slots < 2u ||
        (lease != K3_PREFILL_CACHE_RETAIN &&
         lease != K3_PREFILL_CACHE_BORROW_COLD_WORKSPACE &&
         lease != K3_PREFILL_CACHE_BORROW_WARM_WORKSPACE)) {
        prefill_error(error, error_size,
                      "invalid K3 prefill-plan arguments");
        return false;
    }

    plan->chunk_tokens = chunk_tokens;
    plan->context_remaining = context_remaining;
    plan->routed_layers = K3_PREFILL_MOE_LAYERS;
    plan->routed_layer_sweeps = K3_PREFILL_MOE_LAYERS;
    if (!derive_workspace(chunk_tokens, plan)) {
        prefill_error(error, error_size,
                      "K3 prefill workspace byte count overflow");
        return false;
    }
    plan->auxiliary_workspace_bytes =
        auxiliary_workspace_bytes;
    if (!add_u64(
            &plan->batch_workspace_bytes,
            plan->auxiliary_workspace_bytes)) {
        prefill_error(error, error_size,
                      "K3 auxiliary workspace byte count overflow");
        return false;
    }

    for (uint32_t layer = 1u;
         layer <= K3_PREFILL_MOE_LAYERS; layer++) {
        k3_prefill_expert_span spans[K3_PREFILL_EXPERTS];
        uint64_t layer_bytes = 0u;
        for (uint32_t expert = 0u;
             expert < K3_PREFILL_EXPERTS; expert++) {
            uint16_t shard = 0u;
            uint64_t start = 0u;
            uint64_t end = 0u;
            if (!expert_span(
                    model, layer, expert, &shard, &start, &end,
                    error, error_size)) {
                return false;
            }
            if (shard >= model->shard_count) {
                prefill_error(error, error_size,
                              "layer-%u expert-%u has invalid shard %u",
                              layer, expert, shard);
                return false;
            }
            spans[expert].shard = shard;
            spans[expert].start = start;
            spans[expert].end = end;
            const uint64_t aligned_start =
                start & ~UINT64_C(4095);
            const uint64_t aligned_end =
                (end + UINT64_C(4095)) & ~UINT64_C(4095);
            const uint64_t physical_end =
                aligned_end >
                    model->shards[shard].file_bytes ?
                model->shards[shard].file_bytes :
                aligned_end;
            if (aligned_end < aligned_start ||
                physical_end < end ||
                aligned_end - aligned_start >
                    K3_PREFILL_STAGING_BYTES ||
                !add_u64(
                    &plan->routed_store_requested_read_bytes,
                    aligned_end - aligned_start) ||
                !add_u64(
                    &plan->routed_store_physical_read_bytes,
                    physical_end - aligned_start) ||
                !add_u64(&layer_bytes, end - start)) {
                prefill_error(error, error_size,
                              "layer-%u expert-%u read ledger overflow",
                              layer, expert);
                return false;
            }
            plan->expert_read_requests++;
        }
        qsort(spans, K3_PREFILL_EXPERTS, sizeof(spans[0]),
              compare_expert_span);
        uint16_t previous_shard = UINT16_MAX;
        uint64_t previous_end = 0u;
        for (uint32_t index = 0u;
             index < K3_PREFILL_EXPERTS; index++) {
            if (spans[index].shard == previous_shard &&
                spans[index].start < previous_end) {
                prefill_error(
                    error, error_size,
                    "layer-%u routed expert spans overlap in shard %u",
                    layer, spans[index].shard);
                return false;
            }
            if (spans[index].shard != previous_shard ||
                spans[index].start != previous_end) {
                plan->shard_segments++;
            }
            previous_shard = spans[index].shard;
            previous_end = spans[index].end;
        }
        if (layer == 1u) {
            plan->routed_layer_payload_bytes = layer_bytes;
        } else if (layer_bytes !=
                   plan->routed_layer_payload_bytes) {
            prefill_error(error, error_size,
                          "layer-%u routed payload differs from layer 1",
                          layer);
            return false;
        }
        if (!add_u64(
                &plan->routed_store_payload_bytes,
                layer_bytes)) {
            prefill_error(error, error_size,
                          "K3 routed-store byte count overflow");
            return false;
        }
    }

    if (!multiply_u64(
            K3_PREFILL_STAGING_BYTES, 2u,
            &plan->staging_window_bytes) ||
        !multiply_u64(
            K3_PREFILL_STAGING_BYTES, staging_slots,
            &plan->registered_staging_bytes) ||
        !multiply_u64(
            (uint64_t)K3_PREFILL_MOE_LAYERS *
                experts_per_layer,
            K3_PREFILL_EXPERT_BYTES,
            &plan->decode_cache_bytes)) {
        prefill_error(error, error_size,
                      "K3 prefill cache byte count overflow");
        return false;
    }

    if (lease != K3_PREFILL_CACHE_RETAIN) {
        if (plan->decode_cache_bytes <
            plan->batch_workspace_bytes) {
            prefill_error(
                error, error_size,
                "%s K3 prefill cache %.3f GiB cannot lend "
                "%.3f GiB batch workspace",
                lease == K3_PREFILL_CACHE_BORROW_COLD_WORKSPACE ?
                    "cold" : "warm",
                (double)plan->decode_cache_bytes / 1073741824.0,
                (double)plan->batch_workspace_bytes /
                    1073741824.0);
            return false;
        }
        if (lease == K3_PREFILL_CACHE_BORROW_WARM_WORKSPACE) {
            if (plan->batch_workspace_bytes >
                UINT64_MAX - (K3_PREFILL_EXPERT_BYTES - 1u)) {
                prefill_error(
                    error, error_size,
                    "warm K3 prefill cache slot rounding overflow");
                return false;
            }
            const uint64_t slots =
                (plan->batch_workspace_bytes +
                 K3_PREFILL_EXPERT_BYTES - 1u) /
                K3_PREFILL_EXPERT_BYTES;
            if (!multiply_u64(
                    slots, K3_PREFILL_EXPERT_BYTES,
                    &plan->borrowed_cache_bytes) ||
                plan->borrowed_cache_bytes >
                    plan->decode_cache_bytes) {
                prefill_error(
                    error, error_size,
                    "warm K3 prefill cache slot rounding overflow");
                return false;
            }
        } else {
            plan->borrowed_cache_bytes =
                plan->batch_workspace_bytes;
        }
        plan->retained_cache_bytes =
            plan->decode_cache_bytes -
            plan->borrowed_cache_bytes;
        plan->new_device_bytes = 0u;
    } else {
        plan->retained_cache_bytes =
            plan->decode_cache_bytes;
        plan->new_device_bytes =
            plan->batch_workspace_bytes;
    }

    /*
     * Every mode executes from the fixed QD staging window. Cache leases are
     * ownership transfers, never implicit allocations. The warm lease is
     * selected only when retaining the complete cache would violate the
     * guarded separate-allocation bound.
     */
    plan->new_layer_buffer_bytes = 0u;
    return true;
}

bool k3_prefill_plan_build(
        const k3_st_model *model,
        uint32_t chunk_tokens,
        uint32_t context_remaining,
        uint16_t experts_per_layer,
        uint16_t staging_slots,
        k3_prefill_cache_lease lease,
        k3_prefill_plan *plan,
        char *error,
        size_t error_size) {
    return k3_prefill_plan_build_with_aux_workspace(
        model, chunk_tokens, context_remaining,
        experts_per_layer, staging_slots, lease, 0u,
        plan, error, error_size);
}
