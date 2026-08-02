/*
 * Layer-major K3 prefill implementation.
 *
 * This file is included at the end of k3_engine.cu so it can share the
 * engine's private state and locked model constants without exposing them as
 * a second ABI. Keep k3_engine_forward_token independent as the decode
 * regression path.
 */

static const uint64_t K3_ENGINE_KDA_DEQUANT_WORKSPACE_BYTES =
    (uint64_t)K3_ENGINE_KDA_INNER *
    K3_ENGINE_HIDDEN * sizeof(uint16_t);

typedef struct {
    void *allocation;
    uint64_t allocation_bytes;
    bool owns_allocation;
    void *hidden[2];
    void *scratch[K3_ENGINE_SCRATCH_COUNT];
    void *moe_scratch[K3_ENGINE_MOE_SCRATCH_COUNT];
    void *mla_scratch[K3_ENGINE_MLA_SCRATCH_COUNT];
    void *attn_res_blocks;
    void *auxiliary_workspace;

    uint32_t *route_ids_host;
    uint32_t *route_ids_device;
    float *route_weights_host;
    float *route_weights_device;

    uint32_t *expert_tokens_host;
    uint32_t *expert_tokens_device;
    uint32_t *expert_outputs_host;
    uint32_t *expert_outputs_device;

    uint32_t token_count;
} k3_prefill_workspace;

typedef struct {
    k3_engine_expert_layout layout;
    uint16_t expert;
} k3_prefill_expert_layout;

static bool prefill_finish_default_subphase(
        double *accumulator,
        struct timespec start,
        uint32_t layer,
        const char *label,
        char *error,
        size_t error_size) {
    const hipError_t status = hipStreamSynchronize(NULL);
    if (status != hipSuccess) {
        engine_error(
            error, error_size,
            "layer-%u K3 prefill %s sync failed: %s",
            layer, label, hipGetErrorString(status));
        return false;
    }
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    *accumulator += elapsed_seconds(start, end);
    return true;
}

static void prefill_add_host_subphase(
        double *accumulator,
        struct timespec start) {
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    *accumulator += elapsed_seconds(start, end);
}

static void *prefill_slice(void *base,
                           uint32_t row,
                           uint64_t elements_per_row) {
    return (uint8_t *)base +
        (uint64_t)row * elements_per_row * sizeof(uint16_t);
}

static const void *prefill_const_slice(
        const void *base,
        uint32_t row,
        uint64_t elements_per_row) {
    return (const uint8_t *)base +
        (uint64_t)row * elements_per_row * sizeof(uint16_t);
}

static void *prefill_take(uint8_t **cursor,
                          uint64_t *remaining,
                          uint64_t bytes) {
    if (bytes > *remaining) return NULL;
    void *result = *cursor;
    *cursor += bytes;
    *remaining -= bytes;
    return result;
}

static void prefill_workspace_destroy(
        k3_prefill_workspace *workspace) {
    if (!workspace) return;
    free(workspace->expert_outputs_host);
    free(workspace->expert_tokens_host);
    free(workspace->route_weights_host);
    free(workspace->route_ids_host);
    if (workspace->allocation &&
        workspace->owns_allocation) {
        (void)hipFree(workspace->allocation);
    }
    memset(workspace, 0, sizeof(*workspace));
}

static bool prefill_workspace_create(
        k3_engine *engine,
        k3_prefill_workspace *workspace,
        const k3_prefill_plan *plan,
        k3_prefill_cache_lease lease,
        char *error,
        size_t error_size) {
    memset(workspace, 0, sizeof(*workspace));
    workspace->token_count = plan->chunk_tokens;

    workspace->allocation_bytes =
        plan->batch_workspace_bytes;
    hipError_t status = hipSuccess;
    if (lease != K3_PREFILL_CACHE_RETAIN) {
        if (!engine || !engine->cache ||
            plan->borrowed_cache_bytes <
                workspace->allocation_bytes) {
            engine_error(
                error, error_size,
                "K3 prefill cache lease is too small");
            return false;
        }
        workspace->allocation = engine->cache +
            (lease == K3_PREFILL_CACHE_BORROW_WARM_WORKSPACE ?
                plan->retained_cache_bytes : 0u);
        workspace->owns_allocation = false;
    } else {
        uint64_t available = 0u;
        uint64_t cma_free = 0u;
        if (!host_memory_bytes(&available, &cma_free) ||
            plan->new_device_bytes >
                UINT64_MAX - cma_free -
                K3_ENGINE_HOST_GUARD_BYTES ||
            available <
                plan->new_device_bytes + cma_free +
                K3_ENGINE_HOST_GUARD_BYTES) {
            engine_error(
                error, error_size,
                "K3 prefill workspace rejected: "
                "MemAvailable %.3f GiB, workspace %.3f GiB, "
                "CMA reserve %.3f GiB, guard %.3f GiB",
                (double)available / 1073741824.0,
                (double)plan->new_device_bytes /
                    1073741824.0,
                (double)cma_free / 1073741824.0,
                (double)K3_ENGINE_HOST_GUARD_BYTES /
                    1073741824.0);
            return false;
        }
        status = hipMalloc(
            &workspace->allocation,
            workspace->allocation_bytes);
        if (status != hipSuccess) {
            engine_error(error, error_size,
                         "K3 prefill device workspace failed: %s",
                         hipGetErrorString(status));
            return false;
        }
        workspace->owns_allocation = true;
    }
    uint8_t *cursor = (uint8_t *)workspace->allocation;
    uint64_t remaining = workspace->allocation_bytes;
    const uint32_t tokens = workspace->token_count;
    const uint64_t hidden_bytes =
        (uint64_t)tokens * K3_ENGINE_HIDDEN *
        sizeof(uint16_t);
    const uint64_t inner_bytes =
        (uint64_t)tokens * K3_ENGINE_KDA_INNER *
        sizeof(uint16_t);
    const uint64_t dense_bytes =
        (uint64_t)tokens * K3_ENGINE_DENSE_INTERMEDIATE *
        sizeof(uint16_t);
    const uint64_t scratch_sizes[K3_ENGINE_SCRATCH_COUNT] = {
        hidden_bytes, hidden_bytes, hidden_bytes,
        hidden_bytes, hidden_bytes,
        inner_bytes, inner_bytes, inner_bytes,
        inner_bytes, inner_bytes, inner_bytes,
        inner_bytes, inner_bytes, inner_bytes, inner_bytes,
        (uint64_t)tokens * K3_ENGINE_KDA_HEAD_DIM *
            sizeof(uint16_t),
        (uint64_t)tokens * K3_ENGINE_HEADS *
            sizeof(uint16_t),
        dense_bytes, dense_bytes, dense_bytes,
    };
    for (uint32_t index = 0u;
         index < K3_ENGINE_SCRATCH_COUNT; index++) {
        workspace->scratch[index] = prefill_take(
            &cursor, &remaining, scratch_sizes[index]);
    }
    for (uint32_t index = 0u; index < 2u; index++) {
        workspace->hidden[index] = prefill_take(
            &cursor, &remaining, hidden_bytes);
    }
    workspace->attn_res_blocks = prefill_take(
        &cursor, &remaining,
        (uint64_t)tokens * 8u * K3_ENGINE_HIDDEN *
            sizeof(uint16_t));

    const uint64_t moe_sizes[K3_ENGINE_MOE_SCRATCH_COUNT] = {
        (uint64_t)tokens * K3_ENGINE_EXPERTS *
            sizeof(float),
        (uint64_t)tokens * K3_ENGINE_LATENT *
            sizeof(uint16_t),
        (uint64_t)tokens * K3_ENGINE_EXPERT_HIDDEN *
            sizeof(uint16_t),
        (uint64_t)tokens * K3_ENGINE_EXPERT_HIDDEN *
            sizeof(uint16_t),
        (uint64_t)tokens * K3_ENGINE_EXPERT_HIDDEN *
            sizeof(uint16_t),
        (uint64_t)tokens * K3_ENGINE_TOP_K *
            K3_ENGINE_LATENT * sizeof(uint16_t),
        (uint64_t)tokens * K3_ENGINE_LATENT *
            sizeof(uint16_t),
        (uint64_t)tokens * K3_ENGINE_LATENT *
            sizeof(uint16_t),
        hidden_bytes,
        (uint64_t)tokens * K3_ENGINE_SHARED_HIDDEN *
            sizeof(uint16_t),
        (uint64_t)tokens * K3_ENGINE_SHARED_HIDDEN *
            sizeof(uint16_t),
        (uint64_t)tokens * K3_ENGINE_SHARED_HIDDEN *
            sizeof(uint16_t),
        hidden_bytes,
    };
    for (uint32_t index = 0u;
         index < K3_ENGINE_MOE_SCRATCH_COUNT; index++) {
        workspace->moe_scratch[index] = prefill_take(
            &cursor, &remaining, moe_sizes[index]);
    }

    const uint64_t mla_sizes[K3_ENGINE_MLA_SCRATCH_COUNT] = {
        (uint64_t)tokens * K3_ENGINE_MLA_Q_LORA *
            sizeof(uint16_t),
        (uint64_t)tokens * K3_ENGINE_MLA_Q_LORA *
            sizeof(uint16_t),
        (uint64_t)tokens * K3_ENGINE_MLA_Q *
            sizeof(uint16_t),
        (uint64_t)tokens * K3_ENGINE_MLA_CACHE_DIM *
            sizeof(uint16_t),
        (uint64_t)tokens * K3_ENGINE_HEADS *
            K3_ENGINE_MLA_CACHE_DIM * sizeof(uint16_t),
        inner_bytes, inner_bytes, inner_bytes,
    };
    for (uint32_t index = 0u;
         index < K3_ENGINE_MLA_SCRATCH_COUNT; index++) {
        workspace->mla_scratch[index] = prefill_take(
            &cursor, &remaining, mla_sizes[index]);
    }
    const uint64_t route_count =
        (uint64_t)tokens * K3_ENGINE_TOP_K;
    workspace->route_ids_device =
        (uint32_t *)prefill_take(
            &cursor, &remaining,
            route_count * sizeof(uint32_t));
    workspace->route_weights_device =
        (float *)prefill_take(
            &cursor, &remaining,
            route_count * sizeof(float));
    workspace->expert_tokens_device =
        (uint32_t *)prefill_take(
            &cursor, &remaining,
            (uint64_t)tokens * sizeof(uint32_t));
    workspace->expert_outputs_device =
        (uint32_t *)prefill_take(
            &cursor, &remaining,
            (uint64_t)tokens * sizeof(uint32_t));
    workspace->auxiliary_workspace = prefill_take(
        &cursor, &remaining,
        plan->auxiliary_workspace_bytes);
    if (remaining != 0u) {
        engine_error(
            error, error_size,
            "K3 prefill workspace ledger mismatch: "
            "%llu unassigned bytes",
            (unsigned long long)remaining);
        prefill_workspace_destroy(workspace);
        return false;
    }

    workspace->route_ids_host = (uint32_t *)malloc(
        (size_t)route_count * sizeof(uint32_t));
    workspace->route_weights_host = (float *)malloc(
        (size_t)route_count * sizeof(float));
    workspace->expert_tokens_host = (uint32_t *)malloc(
        (size_t)tokens * sizeof(uint32_t));
    workspace->expert_outputs_host = (uint32_t *)malloc(
        (size_t)tokens * sizeof(uint32_t));
    if (!workspace->route_ids_host ||
        !workspace->route_weights_host ||
        !workspace->expert_tokens_host ||
        !workspace->expert_outputs_host) {
        engine_error(error, error_size,
                     "K3 prefill host scheduler workspace failed");
        prefill_workspace_destroy(workspace);
        return false;
    }
    return true;
}

static bool prefill_kda_projection(
        k3_engine *engine,
        k3_prefill_workspace *workspace,
        const k3_static_weight *weight,
        void *output,
        const void *input,
        k3_prefill_projection_backend backend,
        k3_engine_prefill_stats *stats,
        uint32_t layer,
        char *error,
        size_t error_size) {
    if (backend == K3_PREFILL_PROJECTION_DEFAULT) {
        return k3_static_weight_gemm_bf16(
            weight, output, input,
            workspace->token_count, NULL);
    }
    if (backend !=
            K3_PREFILL_PROJECTION_KDA_DEQUANT_BLAS_EXPERIMENT ||
        !weight || !weight->tensor ||
        weight->kind != K3_STATIC_WEIGHT_Q8_128 ||
        weight->tensor->ndim != 2u ||
        weight->tensor->shape[0] > UINT32_MAX ||
        weight->tensor->shape[1] > UINT32_MAX ||
        !workspace->auxiliary_workspace) {
        engine_error(
            error, error_size,
            "layer-%u KDA dequant/BLAS projection is not eligible",
            layer);
        return false;
    }
    const uint32_t rows =
        (uint32_t)weight->tensor->shape[0];
    const uint32_t columns =
        (uint32_t)weight->tensor->shape[1];
    const uint64_t weight_bytes =
        (uint64_t)rows * columns * sizeof(uint16_t);
    if (weight_bytes >
        K3_ENGINE_KDA_DEQUANT_WORKSPACE_BYTES) {
        engine_error(
            error, error_size,
            "layer-%u KDA dequant/BLAS weight exceeds scratch",
            layer);
        return false;
    }

    struct timespec start;
    if (stats) clock_gettime(CLOCK_MONOTONIC, &start);
    if (!k3_rocm_q8_128_dequantize_bf16(
            workspace->auxiliary_workspace,
            weight->data, weight->scales,
            rows, columns, NULL)) {
        engine_error(
            error, error_size,
            "layer-%u KDA weight dequantization failed",
            layer);
        return false;
    }
    if (stats && !prefill_finish_default_subphase(
            &stats->kda_dequantize_seconds,
            start, layer, "KDA weight dequantization",
            error, error_size)) {
        return false;
    }

    if (stats) clock_gettime(CLOCK_MONOTONIC, &start);
    if (!k3_rocm_blas_bf16_gemm_bf16(
            engine->blas, output,
            workspace->auxiliary_workspace, input,
            workspace->token_count,
            rows, columns, NULL)) {
        engine_error(
            error, error_size,
            "layer-%u KDA hipBLAS projection failed",
            layer);
        return false;
    }
    return !stats || prefill_finish_default_subphase(
        &stats->kda_blas_seconds,
        start, layer, "KDA hipBLAS projection",
        error, error_size);
}

static bool prefill_kda_attention(
        k3_engine *engine,
        k3_prefill_workspace *workspace,
        uint32_t layer,
        uint32_t position,
        const void *normalized,
        void *attention,
        k3_prefill_projection_backend backend,
        k3_engine_prefill_stats *stats,
        char *error,
        size_t error_size) {
    (void)position;
#define PREFILL_KDA_WEIGHT(variable, suffix)                                \
    const k3_static_weight *variable = required_layer_weight(               \
        engine, layer, suffix, error, error_size);                           \
    if (!variable) return false
    PREFILL_KDA_WEIGHT(q_proj, "self_attn.q_proj.weight");
    PREFILL_KDA_WEIGHT(k_proj, "self_attn.k_proj.weight");
    PREFILL_KDA_WEIGHT(v_proj, "self_attn.v_proj.weight");
    PREFILL_KDA_WEIGHT(q_conv, "self_attn.q_conv1d.weight");
    PREFILL_KDA_WEIGHT(k_conv, "self_attn.k_conv1d.weight");
    PREFILL_KDA_WEIGHT(v_conv, "self_attn.v_conv1d.weight");
    PREFILL_KDA_WEIGHT(f_a, "self_attn.f_a_proj.weight");
    PREFILL_KDA_WEIGHT(f_b, "self_attn.f_b_proj.weight");
    PREFILL_KDA_WEIGHT(b_proj, "self_attn.b_proj.weight");
    PREFILL_KDA_WEIGHT(a_log, "self_attn.A_log");
    PREFILL_KDA_WEIGHT(dt_bias, "self_attn.dt_bias");
    PREFILL_KDA_WEIGHT(g_proj, "self_attn.g_proj.weight");
    PREFILL_KDA_WEIGHT(o_norm, "self_attn.o_norm.weight");
    PREFILL_KDA_WEIGHT(o_proj, "self_attn.o_proj.weight");
#undef PREFILL_KDA_WEIGHT
    const uint32_t tokens = workspace->token_count;
    const uint32_t kda_index = layer - layer / 4u;
    const uint64_t conv_vector_bytes =
        (uint64_t)K3_ENGINE_KDA_INNER * 4u *
        sizeof(uint16_t);
    const uint64_t conv_layer_bytes =
        3u * conv_vector_bytes;
    const uint64_t state_layer_bytes =
        (uint64_t)K3_ENGINE_HEADS *
        K3_ENGINE_KDA_HEAD_DIM *
        K3_ENGINE_KDA_HEAD_DIM * sizeof(float);
    uint8_t *conv = (uint8_t *)engine->kda_conv +
        (uint64_t)kda_index * conv_layer_bytes;
    uint8_t *state = (uint8_t *)engine->kda_state +
        (uint64_t)kda_index * state_layer_bytes;

    struct timespec subphase_start;
    if (stats) {
        clock_gettime(CLOCK_MONOTONIC, &subphase_start);
    }
    if (!prefill_kda_projection(
            engine, workspace, q_proj,
            workspace->scratch[S_Q_PROJECTION],
            normalized, backend, stats, layer,
            error, error_size) ||
        !prefill_kda_projection(
            engine, workspace, k_proj,
            workspace->scratch[S_K_PROJECTION],
            normalized, backend, stats, layer,
            error, error_size) ||
        !prefill_kda_projection(
            engine, workspace, v_proj,
            workspace->scratch[S_V_PROJECTION],
            normalized, backend, stats, layer,
            error, error_size) ||
        !prefill_kda_projection(
            engine, workspace, f_a,
            workspace->scratch[S_F_A],
            normalized, backend, stats, layer,
            error, error_size) ||
        !prefill_kda_projection(
            engine, workspace, f_b,
            workspace->scratch[S_RAW_GATE],
            workspace->scratch[S_F_A],
            backend, stats, layer, error, error_size) ||
        !prefill_kda_projection(
            engine, workspace, b_proj,
            workspace->scratch[S_RAW_BETA],
            normalized, backend, stats, layer,
            error, error_size) ||
        !prefill_kda_projection(
            engine, workspace, g_proj,
            workspace->scratch[S_OUTPUT_GATE],
            normalized, backend, stats, layer,
            error, error_size)) {
        engine_error(error, error_size,
                     "layer-%u KDA batch projection failed",
                     layer);
        return false;
    }
    if (stats && !prefill_finish_default_subphase(
            &stats->kda_projection_seconds,
            subphase_start, layer, "KDA projection",
            error, error_size)) {
        return false;
    }

    if (stats) {
        clock_gettime(CLOCK_MONOTONIC, &subphase_start);
    }
    if (!k3_rocm_short_conv4_silu_sequence_bf16_f32_weight(
            workspace->scratch[S_Q], conv,
            workspace->scratch[S_Q_PROJECTION],
            q_conv->data, tokens, K3_ENGINE_KDA_INNER,
            NULL) ||
        !k3_rocm_short_conv4_silu_sequence_bf16_f32_weight(
            workspace->scratch[S_K],
            conv + conv_vector_bytes,
            workspace->scratch[S_K_PROJECTION],
            k_conv->data, tokens, K3_ENGINE_KDA_INNER,
            NULL) ||
        !k3_rocm_short_conv4_silu_sequence_bf16_f32_weight(
            workspace->scratch[S_V],
            conv + 2u * conv_vector_bytes,
            workspace->scratch[S_V_PROJECTION],
            v_conv->data, tokens, K3_ENGINE_KDA_INNER,
            NULL)) {
        engine_error(
            error, error_size,
            "layer-%u KDA convolution scan failed",
            layer);
        return false;
    }
    if (stats && !prefill_finish_default_subphase(
            &stats->kda_convolution_seconds,
            subphase_start, layer, "KDA convolution",
            error, error_size)) {
        return false;
    }

    if (stats) {
        clock_gettime(CLOCK_MONOTONIC, &subphase_start);
    }
    for (uint32_t token = 0u; token < tokens; token++) {
        if (!k3_rocm_kda_recurrent_bf16_f32_state(
                prefill_slice(
                    workspace->scratch[S_KDA], token,
                    K3_ENGINE_KDA_INNER),
                state,
                prefill_const_slice(
                    workspace->scratch[S_Q], token,
                    K3_ENGINE_KDA_INNER),
                prefill_const_slice(
                    workspace->scratch[S_K], token,
                    K3_ENGINE_KDA_INNER),
                prefill_const_slice(
                    workspace->scratch[S_V], token,
                    K3_ENGINE_KDA_INNER),
                prefill_const_slice(
                    workspace->scratch[S_RAW_GATE], token,
                    K3_ENGINE_KDA_INNER),
                prefill_const_slice(
                    workspace->scratch[S_RAW_BETA], token,
                    K3_ENGINE_HEADS),
                a_log->data, dt_bias->data,
                K3_ENGINE_HEADS, K3_ENGINE_KDA_HEAD_DIM,
                -5.0f, NULL)) {
            engine_error(
                error, error_size,
                "layer-%u KDA causal scan failed at token %u",
                layer, token);
            return false;
        }
    }
    if (stats && !prefill_finish_default_subphase(
            &stats->kda_recurrent_seconds,
            subphase_start, layer, "KDA recurrent scan",
            error, error_size)) {
        return false;
    }

    if (stats) {
        clock_gettime(CLOCK_MONOTONIC, &subphase_start);
    }
    if (!k3_rocm_rms_norm_sigmoid_gate_bf16_f32_weight(
            workspace->scratch[S_GATED],
            workspace->scratch[S_KDA],
            workspace->scratch[S_OUTPUT_GATE],
            o_norm->data, tokens * K3_ENGINE_HEADS,
            K3_ENGINE_KDA_HEAD_DIM, 1e-5f, NULL)) {
        engine_error(error, error_size,
                     "layer-%u KDA batch gate/norm failed", layer);
        return false;
    }
    if (stats && !prefill_finish_default_subphase(
            &stats->kda_gate_norm_seconds,
            subphase_start, layer, "KDA gate/norm",
            error, error_size)) {
        return false;
    }

    if (stats) {
        clock_gettime(CLOCK_MONOTONIC, &subphase_start);
    }
    if (!prefill_kda_projection(
            engine, workspace, o_proj, attention,
            workspace->scratch[S_GATED],
            backend, stats, layer, error, error_size)) {
        engine_error(error, error_size,
                     "layer-%u KDA output projection failed", layer);
        return false;
    }
    if (stats && !prefill_finish_default_subphase(
            &stats->kda_output_projection_seconds,
            subphase_start, layer, "KDA output projection",
            error, error_size)) {
        return false;
    }
    return true;
}

static bool prefill_mla_attention(
        k3_engine *engine,
        k3_prefill_workspace *workspace,
        uint32_t layer,
        uint32_t position,
        const void *normalized,
        void *attention,
        char *error,
        size_t error_size) {
#define PREFILL_MLA_WEIGHT(variable, suffix)                                \
    const k3_static_weight *variable = required_layer_weight(               \
        engine, layer, suffix, error, error_size);                           \
    if (!variable) return false
    PREFILL_MLA_WEIGHT(q_a, "self_attn.q_a_proj.weight");
    PREFILL_MLA_WEIGHT(q_a_norm, "self_attn.q_a_layernorm.weight");
    PREFILL_MLA_WEIGHT(q_b, "self_attn.q_b_proj.weight");
    PREFILL_MLA_WEIGHT(kv_a, "self_attn.kv_a_proj_with_mqa.weight");
    PREFILL_MLA_WEIGHT(kv_a_norm, "self_attn.kv_a_layernorm.weight");
    PREFILL_MLA_WEIGHT(kv_b, "self_attn.kv_b_proj.weight");
    PREFILL_MLA_WEIGHT(gate, "self_attn.g_proj.weight");
    PREFILL_MLA_WEIGHT(o_proj, "self_attn.o_proj.weight");
#undef PREFILL_MLA_WEIGHT
    const uint32_t tokens = workspace->token_count;
    const uint32_t mla_index =
        layer == 92u ? 23u : layer / 4u;
    const uint64_t cache_layer_bytes =
        (uint64_t)engine->context *
        K3_ENGINE_MLA_CACHE_DIM * sizeof(uint16_t);
    uint8_t *cache = (uint8_t *)engine->mla_cache +
        (uint64_t)mla_index * cache_layer_bytes;
    uint8_t *packed_key =
        engine->mla_packed_keys +
        (uint64_t)mla_index *
            K3_ENGINE_MLA_PACKED_ELEMENTS *
            sizeof(uint16_t);
    uint8_t *attention_workspace =
        (uint8_t *)engine->workspace;
    void *scores = attention_workspace;
    void *probabilities = attention_workspace +
        (uint64_t)K3_ENGINE_HEADS *
            engine->context * sizeof(float);
    void *latent = attention_workspace +
        (uint64_t)K3_ENGINE_HEADS *
            engine->context *
            (sizeof(float) + sizeof(uint16_t));

    if (!k3_static_weight_gemm_bf16(
            q_a, workspace->mla_scratch[A_MLA_Q_A],
            normalized, tokens, NULL) ||
        !k3_rocm_rms_norm_bf16(
            workspace->mla_scratch[A_MLA_Q_A_NORM],
            workspace->mla_scratch[A_MLA_Q_A],
            q_a_norm->data, tokens,
            K3_ENGINE_MLA_Q_LORA, 1e-5f, NULL) ||
        !k3_static_weight_gemm_bf16(
            q_b, workspace->mla_scratch[A_MLA_Q],
            workspace->mla_scratch[A_MLA_Q_A_NORM],
            tokens, NULL) ||
        !k3_static_weight_gemm_bf16(
            kv_a, workspace->mla_scratch[A_MLA_KV],
            normalized, tokens, NULL) ||
        !k3_static_weight_gemm_bf16(
            gate, workspace->mla_scratch[A_MLA_GATE],
            normalized, tokens, NULL)) {
        engine_error(error, error_size,
                     "layer-%u MLA batch projection failed",
                     layer);
        return false;
    }
    for (uint32_t token = 0u; token < tokens; token++) {
        uint16_t *cache_row = (uint16_t *)cache +
            (uint64_t)(position + token) *
                K3_ENGINE_MLA_CACHE_DIM;
        const uint16_t *kv_row =
            (const uint16_t *)prefill_const_slice(
                workspace->mla_scratch[A_MLA_KV],
                token, K3_ENGINE_MLA_CACHE_DIM);
        if (!k3_rocm_rms_norm_bf16(
                cache_row, kv_row, kv_a_norm->data,
                1u, K3_ENGINE_MLA_LATENT, 1e-5f, NULL) ||
            hipMemcpyAsync(
                cache_row + K3_ENGINE_MLA_LATENT,
                kv_row + K3_ENGINE_MLA_LATENT,
                64u * sizeof(uint16_t),
                hipMemcpyDeviceToDevice, NULL) != hipSuccess ||
            !k3_rocm_mla_absorb_q_bf16(
                prefill_slice(
                    workspace->mla_scratch[A_MLA_ABSORBED_Q],
                    token,
                    (uint64_t)K3_ENGINE_HEADS *
                        K3_ENGINE_MLA_CACHE_DIM),
                prefill_const_slice(
                    workspace->mla_scratch[A_MLA_Q],
                    token, K3_ENGINE_MLA_Q),
                packed_key, K3_ENGINE_HEADS, NULL) ||
            !k3_rocm_blas_mla_attention_bf16(
                engine->blas, latent, scores, probabilities,
                prefill_const_slice(
                    workspace->mla_scratch[A_MLA_ABSORBED_Q],
                    token,
                    (uint64_t)K3_ENGINE_HEADS *
                        K3_ENGINE_MLA_CACHE_DIM),
                cache, K3_ENGINE_HEADS,
                position + token + 1u, NULL) ||
            !k3_rocm_mla_decompress_v_bf16(
                prefill_slice(
                    workspace->mla_scratch[A_MLA_VALUE],
                    token, K3_ENGINE_KDA_INNER),
                latent, kv_b->data, K3_ENGINE_HEADS, NULL)) {
            engine_error(
                error, error_size,
                "layer-%u MLA causal scan failed at token %u",
                layer, token);
            return false;
        }
    }
    if (!k3_rocm_sigmoid_mul_bf16(
            workspace->mla_scratch[A_MLA_GATED],
            workspace->mla_scratch[A_MLA_VALUE],
            workspace->mla_scratch[A_MLA_GATE],
            (uint64_t)tokens * K3_ENGINE_KDA_INNER,
            NULL) ||
        !k3_static_weight_gemm_bf16(
            o_proj, attention,
            workspace->mla_scratch[A_MLA_GATED],
            tokens, NULL)) {
        engine_error(error, error_size,
                     "layer-%u MLA batch output failed", layer);
        return false;
    }
    return true;
}

static bool prefill_attention(
        k3_engine *engine,
        k3_prefill_workspace *workspace,
        uint32_t layer,
        uint32_t position,
        const void *normalized,
        void *attention,
        k3_prefill_projection_backend backend,
        k3_engine_prefill_stats *stats,
        char *error,
        size_t error_size) {
    const bool is_mla =
        layer % 4u == 3u || layer == 92u;
    return is_mla ?
        prefill_mla_attention(
            engine, workspace, layer, position,
            normalized, attention, error, error_size) :
        prefill_kda_attention(
            engine, workspace, layer, position,
            normalized, attention, backend, stats,
            error, error_size);
}

static bool prefill_layer0(
        k3_engine *engine,
        k3_prefill_workspace *workspace,
        uint32_t position,
        const uint32_t *input_tokens,
        char *error,
        size_t error_size,
        uint64_t *embedding_physical_bytes) {
#define PREFILL_LAYER0_WEIGHT(variable, suffix)                             \
    const k3_static_weight *variable = required_weight(                     \
        engine, "language_model.model.layers.0." suffix,                    \
        error, error_size);                                                 \
    if (!variable) return false
    PREFILL_LAYER0_WEIGHT(input_norm, "input_layernorm.weight");
    PREFILL_LAYER0_WEIGHT(mlp_res_norm, "mlp_res_norm.weight");
    PREFILL_LAYER0_WEIGHT(mlp_res_proj, "mlp_res_proj.weight");
    PREFILL_LAYER0_WEIGHT(post_norm, "post_attention_layernorm.weight");
    PREFILL_LAYER0_WEIGHT(dense_gate, "mlp.gate_proj.weight");
    PREFILL_LAYER0_WEIGHT(dense_up, "mlp.up_proj.weight");
    PREFILL_LAYER0_WEIGHT(dense_down, "mlp.down_proj.weight");
#undef PREFILL_LAYER0_WEIGHT
    const uint32_t tokens = workspace->token_count;
    const uint64_t hidden_bytes =
        (uint64_t)K3_ENGINE_HIDDEN * sizeof(uint16_t);
    const k3_st_tensor *embedding = k3_st_find(
        &engine->model,
        "language_model.model.embed_tokens.weight");
    if (!embedding || embedding->ndim != 2u ||
        embedding->shape[0] != K3_ENGINE_VOCAB ||
        embedding->shape[1] != K3_ENGINE_HIDDEN) {
        engine_error(error, error_size,
                     "invalid K3 embedding tensor");
        return false;
    }
    for (uint32_t token = 0u; token < tokens; token++) {
        if (input_tokens[token] >= K3_ENGINE_VOCAB) {
            engine_error(error, error_size,
                         "invalid K3 prefill token %u at row %u",
                         input_tokens[token], token);
            return false;
        }
        k3_st_read_view view;
        if (!k3_st_read_span_into(
                &engine->model, embedding->shard,
                embedding->physical_offset +
                    (uint64_t)input_tokens[token] *
                        hidden_bytes,
                hidden_bytes, 4096u,
                engine->staging_host[0],
                K3_ENGINE_STAGING_BYTES,
                &view, error, error_size)) {
            return false;
        }
        const uint64_t source_offset =
            (uint64_t)(view.data -
                       (uint8_t *)engine->staging_host[0]);
        hipError_t status = hipMemcpyAsync(
            prefill_slice(
                workspace->hidden[0], token,
                K3_ENGINE_HIDDEN),
            (uint8_t *)engine->staging_device[0] +
                source_offset,
            hidden_bytes, hipMemcpyDeviceToDevice, NULL);
        if (status != hipSuccess ||
            hipStreamSynchronize(NULL) != hipSuccess) {
            engine_error(error, error_size,
                         "embedding row %u upload failed: %s",
                         token, hipGetErrorString(status));
            return false;
        }
        *embedding_physical_bytes +=
            view.allocation_bytes;
        status = hipMemcpyAsync(
            prefill_slice(
                workspace->attn_res_blocks,
                token * 8u, K3_ENGINE_HIDDEN),
            prefill_const_slice(
                workspace->hidden[0], token,
                K3_ENGINE_HIDDEN),
            hidden_bytes, hipMemcpyDeviceToDevice, NULL);
        if (status != hipSuccess) {
            engine_error(error, error_size,
                         "layer-0 AttnRes row save failed: %s",
                         hipGetErrorString(status));
            return false;
        }
    }

    if (!k3_rocm_rms_norm_bf16(
            workspace->scratch[S_NORM],
            workspace->hidden[0], input_norm->data,
            tokens, K3_ENGINE_HIDDEN, 1e-5f, NULL) ||
        !prefill_kda_attention(
            engine, workspace, 0u, position,
            workspace->scratch[S_NORM],
            workspace->scratch[S_ATTENTION],
            K3_PREFILL_PROJECTION_DEFAULT,
            NULL, error, error_size) ||
        !k3_rocm_attn_res_bf16(
            workspace->scratch[S_MLP_INPUT],
            workspace->scratch[S_ATTENTION],
            workspace->attn_res_blocks,
            mlp_res_norm->data, mlp_res_proj->data,
            tokens, 8u, 1u,
            K3_ENGINE_HIDDEN, 1e-5f, NULL) ||
        !k3_rocm_rms_norm_bf16(
            workspace->scratch[S_POST_NORM],
            workspace->scratch[S_MLP_INPUT],
            post_norm->data, tokens,
            K3_ENGINE_HIDDEN, 1e-5f, NULL) ||
        !k3_static_weight_gemm_bf16(
            dense_gate, workspace->scratch[S_DENSE_GATE],
            workspace->scratch[S_POST_NORM],
            tokens, NULL) ||
        !k3_static_weight_gemm_bf16(
            dense_up, workspace->scratch[S_DENSE_UP],
            workspace->scratch[S_POST_NORM],
            tokens, NULL) ||
        !k3_rocm_situ_bf16(
            workspace->scratch[S_DENSE_ACTIVATION],
            workspace->scratch[S_DENSE_GATE],
            workspace->scratch[S_DENSE_UP],
            (uint64_t)tokens *
                K3_ENGINE_DENSE_INTERMEDIATE,
            4.0f, 25.0f, NULL) ||
        !k3_static_weight_gemm_bf16(
            dense_down, workspace->scratch[S_MLP_OUTPUT],
            workspace->scratch[S_DENSE_ACTIVATION],
            tokens, NULL) ||
        !k3_rocm_add_bf16(
            workspace->hidden[1],
            workspace->scratch[S_ATTENTION],
            workspace->scratch[S_MLP_OUTPUT],
            (uint64_t)tokens * K3_ENGINE_HIDDEN, NULL)) {
        if (!error || !error[0]) {
            engine_error(error, error_size,
                         "layer-0 K3 prefill failed");
        }
        return false;
    }
    return true;
}

static int compare_prefill_expert_layout(
        const void *left, const void *right) {
    const k3_prefill_expert_layout *a =
        (const k3_prefill_expert_layout *)left;
    const k3_prefill_expert_layout *b =
        (const k3_prefill_expert_layout *)right;
    if (a->layout.shard != b->layout.shard) {
        return a->layout.shard < b->layout.shard ?
            -1 : 1;
    }
    if (a->layout.physical_start ==
        b->layout.physical_start) {
        return 0;
    }
    return a->layout.physical_start <
           b->layout.physical_start ? -1 : 1;
}

static void abort_prefill_io(k3_engine *engine) {
    char discard[128];
    while (k3_io_uring_outstanding(
               engine->staging_ring) > 0u) {
        k3_io_completion completions[K3_ENGINE_STREAM_QD];
        uint16_t count = 0u;
        if (!k3_io_uring_wait(
                engine->staging_ring, completions,
                K3_ENGINE_STREAM_QD, &count,
                discard, sizeof(discard))) {
            break;
        }
    }
    (void)hipStreamSynchronize(engine->expert_stream);
    (void)hipStreamSynchronize(engine->shared_stream);
}

static bool launch_prefill_expert(
        k3_engine *engine,
        k3_prefill_workspace *workspace,
        uint8_t *base,
        const k3_engine_expert_layout *layout,
        uint32_t selected_count) {
    const uint64_t index_bytes =
        (uint64_t)selected_count * sizeof(uint32_t);
    return hipMemcpyAsync(
               workspace->expert_tokens_device,
               workspace->expert_tokens_host,
               index_bytes, hipMemcpyHostToDevice,
               engine->expert_stream) == hipSuccess &&
           hipMemcpyAsync(
               workspace->expert_outputs_device,
               workspace->expert_outputs_host,
               index_bytes, hipMemcpyHostToDevice,
               engine->expert_stream) == hipSuccess &&
           k3_rocm_gather_rows_bf16(
               workspace->moe_scratch[M_ROUTED_NORM],
               workspace->moe_scratch[M_LATENT],
               workspace->expert_tokens_device,
               selected_count, K3_ENGINE_LATENT,
               engine->expert_stream) &&
           k3_rocm_mxfp4_gemm_bf16(
               workspace->moe_scratch[M_EXPERT_GATE],
               base + layout->relative[0],
               base + layout->relative[1],
               workspace->moe_scratch[M_ROUTED_NORM],
               selected_count, K3_ENGINE_EXPERT_HIDDEN,
               K3_ENGINE_LATENT, engine->expert_stream) &&
           k3_rocm_mxfp4_gemm_bf16(
               workspace->moe_scratch[M_EXPERT_UP],
               base + layout->relative[4],
               base + layout->relative[5],
               workspace->moe_scratch[M_ROUTED_NORM],
               selected_count, K3_ENGINE_EXPERT_HIDDEN,
               K3_ENGINE_LATENT, engine->expert_stream) &&
           k3_rocm_situ_bf16(
               workspace->moe_scratch[M_EXPERT_ACTIVATION],
               workspace->moe_scratch[M_EXPERT_GATE],
               workspace->moe_scratch[M_EXPERT_UP],
               (uint64_t)selected_count *
                   K3_ENGINE_EXPERT_HIDDEN,
               4.0f, 25.0f, engine->expert_stream) &&
           k3_rocm_mxfp4_gemm_bf16(
               workspace->moe_scratch[M_ROUTED],
               base + layout->relative[2],
               base + layout->relative[3],
               workspace->moe_scratch[M_EXPERT_ACTIVATION],
               selected_count, K3_ENGINE_LATENT,
               K3_ENGINE_EXPERT_HIDDEN,
               engine->expert_stream) &&
           k3_rocm_scatter_rows_bf16(
               workspace->moe_scratch[M_EXPERT_OUTPUTS],
               workspace->moe_scratch[M_ROUTED],
               workspace->expert_outputs_device,
               selected_count, K3_ENGINE_LATENT,
               engine->expert_stream);
}

static bool prefill_stream_routed_layer(
        k3_engine *engine,
        k3_prefill_workspace *workspace,
        uint32_t layer,
        k3_engine_prefill_stats *stats,
        char *error,
        size_t error_size) {
    bool selected_experts[K3_ENGINE_EXPERTS] = { false };
    const uint64_t route_count =
        (uint64_t)workspace->token_count * K3_ENGINE_TOP_K;
    for (uint64_t route = 0u; route < route_count; route++) {
        const uint32_t expert = workspace->route_ids_host[route];
        if (expert >= K3_ENGINE_EXPERTS) {
            engine_error(error, error_size,
                         "layer-%u selected expert %u is invalid",
                         layer, expert);
            return false;
        }
        selected_experts[expert] = true;
    }
    k3_prefill_expert_layout layouts[K3_ENGINE_EXPERTS];
    for (uint32_t expert = 0u;
         expert < K3_ENGINE_EXPERTS; expert++) {
        layouts[expert].expert = (uint16_t)expert;
        if (!find_expert_layout(
                engine, layer, expert,
                &layouts[expert].layout,
                error, error_size)) {
            return false;
        }
    }
    qsort(layouts, K3_ENGINE_EXPERTS,
          sizeof(layouts[0]),
          compare_prefill_expert_layout);
    for (uint32_t index = 1u;
         index < K3_ENGINE_EXPERTS; index++) {
        const k3_engine_expert_layout *previous =
            &layouts[index - 1u].layout;
        const k3_engine_expert_layout *current =
            &layouts[index].layout;
        if (previous->shard != current->shard ||
            previous->physical_start +
                K3_ENGINE_EXPERT_BYTES !=
                    current->physical_start) {
            engine_error(
                error, error_size,
                "layer-%u is not one contiguous physical "
                "expert sweep at index %u",
                layer, index);
            return false;
        }
    }

    uint32_t layout_count = 0u;
    for (uint32_t index = 0u;
         index < K3_ENGINE_EXPERTS; index++) {
        if (selected_experts[layouts[index].expert]) {
            layouts[layout_count++] = layouts[index];
        }
    }
    if (layout_count == 0u) {
        engine_error(error, error_size,
                     "layer-%u prefill selected no experts", layer);
        return false;
    }
    const uint32_t requests_before = stats->expert_read_requests;
    const uint64_t bytes_before = stats->routed_physical_read_bytes;
    uint64_t expected_physical_bytes = 0u;
    for (uint32_t index = 0u; index < layout_count; index++) {
        const k3_engine_expert_layout *layout =
            &layouts[index].layout;
        if (layout->aligned_start >
            UINT64_MAX - layout->aligned_bytes) {
            engine_error(error, error_size,
                         "layer-%u selected request end overflow", layer);
            return false;
        }
        const uint64_t request_end =
            layout->aligned_start + layout->aligned_bytes;
        const uint64_t physical_end =
            request_end > engine->model.shards[layout->shard].file_bytes ?
                engine->model.shards[layout->shard].file_bytes :
                request_end;
        if (physical_end < layout->aligned_start ||
            expected_physical_bytes >
                UINT64_MAX - (physical_end - layout->aligned_start)) {
            engine_error(error, error_size,
                         "layer-%u selected-read ledger overflow", layer);
            return false;
        }
        expected_physical_bytes +=
            physical_end - layout->aligned_start;
    }

    uint32_t next = 0u;
    k3_io_request requests[K3_ENGINE_STREAM_QD];
    uint16_t initial = 0u;
    while (next < layout_count &&
           initial < K3_ENGINE_STREAM_QD) {
        const k3_engine_expert_layout *layout =
            &layouts[next].layout;
        requests[initial].fd =
            engine->model.shards[layout->shard].direct_fd;
        requests[initial].offset = layout->aligned_start;
        requests[initial].bytes = layout->aligned_bytes;
        requests[initial].buffer_index = initial;
        requests[initial].user_data = next;
        next++;
        initial++;
    }
    struct timespec subphase_start;
    clock_gettime(CLOCK_MONOTONIC, &subphase_start);
    if (!k3_io_uring_submit(
            engine->staging_ring, requests, initial,
            error, error_size)) {
        return false;
    }
    prefill_add_host_subphase(
        &stats->routed_submit_seconds,
        subphase_start);

    uint32_t completed = 0u;
    while (completed < layout_count) {
        k3_io_completion completions[K3_ENGINE_STREAM_QD];
        uint16_t completion_count = 0u;
        clock_gettime(CLOCK_MONOTONIC, &subphase_start);
        if (!k3_io_uring_wait(
                engine->staging_ring, completions,
                K3_ENGINE_STREAM_QD, &completion_count,
                error, error_size)) {
            abort_prefill_io(engine);
            return false;
        }
        prefill_add_host_subphase(
            &stats->routed_read_wait_seconds,
            subphase_start);
        k3_io_request refills[K3_ENGINE_STREAM_QD];
        uint16_t refill_count = 0u;
        for (uint16_t completion = 0u;
             completion < completion_count; completion++) {
            const uint32_t index =
                (uint32_t)completions[completion].user_data;
            if (index >= layout_count) {
                engine_error(error, error_size,
                             "layer-%u prefill completion "
                             "index %u is invalid",
                             layer, index);
                abort_prefill_io(engine);
                return false;
            }
            const k3_prefill_expert_layout *selected =
                &layouts[index];
            const uint16_t buffer =
                completions[completion].buffer_index;
            const uint64_t required =
                selected->layout.physical_start -
                selected->layout.aligned_start +
                K3_ENGINE_EXPERT_BYTES;
            if (buffer >= K3_ENGINE_STREAM_QD ||
                completions[completion].result < 0 ||
                (uint64_t)completions[completion].result <
                    required) {
                engine_error(
                    error, error_size,
                    "layer-%u prefill expert %u read failed: "
                    "buffer=%u result=%d required=%llu",
                    layer, selected->expert, buffer,
                    completions[completion].result,
                    (unsigned long long)required);
                abort_prefill_io(engine);
                return false;
            }
            stats->routed_physical_read_bytes +=
                (uint64_t)completions[completion].result;
            stats->expert_read_requests++;

            uint32_t selected_count = 0u;
            clock_gettime(CLOCK_MONOTONIC, &subphase_start);
            for (uint32_t token = 0u;
                 token < workspace->token_count; token++) {
                for (uint32_t rank = 0u;
                     rank < K3_ENGINE_TOP_K; rank++) {
                    const uint32_t route =
                        token * K3_ENGINE_TOP_K + rank;
                    if (workspace->route_ids_host[route] ==
                        selected->expert) {
                        workspace->
                            expert_tokens_host[selected_count] =
                                token;
                        workspace->
                            expert_outputs_host[selected_count] =
                                route;
                        selected_count++;
                    }
                }
            }
            prefill_add_host_subphase(
                &stats->routed_index_seconds,
                subphase_start);
            if (selected_count == 0u) {
                engine_error(
                    error, error_size,
                    "layer-%u selected expert %u has no routes",
                    layer, selected->expert);
                abort_prefill_io(engine);
                return false;
            }
            uint8_t *base =
                (uint8_t *)engine->staging_device[buffer] +
                (selected->layout.physical_start -
                 selected->layout.aligned_start);
            clock_gettime(CLOCK_MONOTONIC, &subphase_start);
            if (!launch_prefill_expert(
                    engine, workspace, base,
                    &selected->layout, selected_count) ||
                hipStreamSynchronize(
                    engine->expert_stream) != hipSuccess) {
                engine_error(
                    error, error_size,
                    "layer-%u prefill expert %u batch failed",
                    layer, selected->expert);
                abort_prefill_io(engine);
                return false;
            }
            prefill_add_host_subphase(
                &stats->routed_expert_pipeline_seconds,
                subphase_start);
            completed++;

            if (next < layout_count) {
                const k3_engine_expert_layout *layout =
                    &layouts[next].layout;
                refills[refill_count].fd =
                    engine->model.shards[
                        layout->shard].direct_fd;
                refills[refill_count].offset =
                    layout->aligned_start;
                refills[refill_count].bytes =
                    layout->aligned_bytes;
                refills[refill_count].buffer_index = buffer;
                refills[refill_count].user_data = next;
                next++;
                refill_count++;
            }
        }
        if (refill_count != 0u) {
            clock_gettime(
                CLOCK_MONOTONIC,
                &subphase_start);
            if (!k3_io_uring_submit(
                    engine->staging_ring,
                    refills, refill_count,
                    error, error_size)) {
                abort_prefill_io(engine);
                return false;
            }
            prefill_add_host_subphase(
                &stats->routed_submit_seconds,
                subphase_start);
        }
    }
    if (k3_io_uring_outstanding(
            engine->staging_ring) != 0u) {
        engine_error(error, error_size,
                     "layer-%u retained prefill reads",
                     layer);
        abort_prefill_io(engine);
        return false;
    }
    if (stats->expert_read_requests - requests_before !=
            layout_count ||
        stats->routed_physical_read_bytes - bytes_before !=
            expected_physical_bytes) {
        engine_error(
            error, error_size,
            "layer-%u selected-read ledger mismatch: "
            "requests %u/%u bytes %llu/%llu",
            layer,
            stats->expert_read_requests - requests_before,
            layout_count,
            (unsigned long long)(
                stats->routed_physical_read_bytes - bytes_before),
            (unsigned long long)expected_physical_bytes);
        return false;
    }
    return true;
}

static bool prefill_routed_layer(
        k3_engine *engine,
        k3_prefill_workspace *workspace,
        uint32_t layer,
        uint32_t position,
        const void *input,
        void *output,
        k3_prefill_projection_backend backend,
        k3_engine_prefill_stats *stats,
        char *error,
        size_t error_size) {
#define PREFILL_LAYER_WEIGHT(variable, suffix)                              \
    const k3_static_weight *variable = required_layer_weight(               \
        engine, layer, suffix, error, error_size);                           \
    if (!variable) return false
    PREFILL_LAYER_WEIGHT(
        self_res_norm, "self_attention_res_norm.weight");
    PREFILL_LAYER_WEIGHT(
        self_res_proj, "self_attention_res_proj.weight");
    PREFILL_LAYER_WEIGHT(input_norm, "input_layernorm.weight");
    PREFILL_LAYER_WEIGHT(mlp_res_norm, "mlp_res_norm.weight");
    PREFILL_LAYER_WEIGHT(mlp_res_proj, "mlp_res_proj.weight");
    PREFILL_LAYER_WEIGHT(
        post_norm, "post_attention_layernorm.weight");
    PREFILL_LAYER_WEIGHT(
        router_bias,
        "block_sparse_moe.gate.e_score_correction_bias");
    PREFILL_LAYER_WEIGHT(
        router_weight, "block_sparse_moe.gate.weight");
    PREFILL_LAYER_WEIGHT(
        routed_down,
        "block_sparse_moe.routed_expert_down_proj.weight");
    PREFILL_LAYER_WEIGHT(
        routed_norm,
        "block_sparse_moe.routed_expert_norm.weight");
    PREFILL_LAYER_WEIGHT(
        routed_up,
        "block_sparse_moe.routed_expert_up_proj.weight");
    PREFILL_LAYER_WEIGHT(
        shared_down,
        "block_sparse_moe.shared_experts.down_proj.weight");
    PREFILL_LAYER_WEIGHT(
        shared_gate,
        "block_sparse_moe.shared_experts.gate_proj.weight");
    PREFILL_LAYER_WEIGHT(
        shared_up,
        "block_sparse_moe.shared_experts.up_proj.weight");
#undef PREFILL_LAYER_WEIGHT
    const uint32_t tokens = workspace->token_count;
    const uint32_t pre_attention_blocks =
        (layer + 11u) / 12u;
    const bool block_boundary = layer % 12u == 0u;
    const uint64_t hidden_bytes =
        (uint64_t)K3_ENGINE_HIDDEN * sizeof(uint16_t);
    struct timespec phase_start;
    struct timespec phase_end;
    clock_gettime(CLOCK_MONOTONIC, &phase_start);

    if (!k3_rocm_attn_res_bf16(
            workspace->scratch[S_MLP_INPUT], input,
            workspace->attn_res_blocks,
            self_res_norm->data, self_res_proj->data,
            tokens, 8u, pre_attention_blocks,
            K3_ENGINE_HIDDEN, 1e-5f, NULL)) {
        engine_error(error, error_size,
                     "layer-%u prefill pre-attention "
                     "AttnRes failed", layer);
        return false;
    }
    if (block_boundary) {
        for (uint32_t token = 0u;
             token < tokens; token++) {
            hipError_t status = hipMemcpyAsync(
                prefill_slice(
                    workspace->attn_res_blocks,
                    token * 8u + pre_attention_blocks,
                    K3_ENGINE_HIDDEN),
                prefill_const_slice(
                    input, token, K3_ENGINE_HIDDEN),
                hidden_bytes,
                hipMemcpyDeviceToDevice, NULL);
            if (status != hipSuccess) {
                engine_error(
                    error, error_size,
                    "layer-%u prefill block save failed: %s",
                    layer, hipGetErrorString(status));
                return false;
            }
        }
    }
    if (!k3_rocm_rms_norm_bf16(
            workspace->scratch[S_NORM],
            workspace->scratch[S_MLP_INPUT],
            input_norm->data, tokens,
            K3_ENGINE_HIDDEN, 1e-5f, NULL) ||
        !prefill_attention(
            engine, workspace, layer, position,
            workspace->scratch[S_NORM],
            workspace->scratch[S_ATTENTION],
            backend, stats, error, error_size)) {
        return false;
    }
    if (block_boundary) {
        hipError_t status = hipMemcpyAsync(
            workspace->scratch[S_MLP_OUTPUT],
            workspace->scratch[S_ATTENTION],
            (uint64_t)tokens * hidden_bytes,
            hipMemcpyDeviceToDevice, NULL);
        if (status != hipSuccess) {
            engine_error(error, error_size,
                         "layer-%u prefill boundary copy "
                         "failed: %s", layer,
                         hipGetErrorString(status));
            return false;
        }
    } else if (!k3_rocm_add_bf16(
                   workspace->scratch[S_MLP_OUTPUT],
                   input,
                   workspace->scratch[S_ATTENTION],
                   (uint64_t)tokens *
                       K3_ENGINE_HIDDEN, NULL)) {
        engine_error(error, error_size,
                     "layer-%u prefill attention residual "
                     "failed", layer);
        return false;
    }
    hipError_t status = hipStreamSynchronize(NULL);
    if (status != hipSuccess) {
        engine_error(error, error_size,
                     "layer-%u prefill attention timing "
                     "sync failed: %s", layer,
                     hipGetErrorString(status));
        return false;
    }
    clock_gettime(CLOCK_MONOTONIC, &phase_end);
    const double attention_seconds =
        elapsed_seconds(phase_start, phase_end);
    stats->attention_seconds += attention_seconds;
    if (layer % 4u == 3u || layer == 92u) {
        stats->mla_attention_seconds +=
            attention_seconds;
    } else {
        stats->kda_attention_seconds +=
            attention_seconds;
    }
    clock_gettime(CLOCK_MONOTONIC, &phase_start);

    if (!k3_rocm_attn_res_bf16(
            workspace->scratch[S_MLP_INPUT],
            workspace->scratch[S_MLP_OUTPUT],
            workspace->attn_res_blocks,
            mlp_res_norm->data, mlp_res_proj->data,
            tokens, 8u,
            pre_attention_blocks +
                (block_boundary ? 1u : 0u),
            K3_ENGINE_HIDDEN, 1e-5f, NULL) ||
        !k3_rocm_rms_norm_bf16(
            workspace->scratch[S_POST_NORM],
            workspace->scratch[S_MLP_INPUT],
            post_norm->data, tokens,
            K3_ENGINE_HIDDEN, 1e-5f, NULL) ||
        !k3_rocm_bf16_gemm_f32(
            workspace->moe_scratch[M_LOGITS],
            router_weight->data,
            workspace->scratch[S_POST_NORM],
            tokens, K3_ENGINE_EXPERTS,
            K3_ENGINE_HIDDEN, NULL) ||
        !k3_rocm_router_topk_f32_batch(
            workspace->route_ids_device,
            workspace->route_weights_device,
            workspace->moe_scratch[M_LOGITS],
            router_bias->data, tokens,
            K3_ENGINE_EXPERTS, K3_ENGINE_TOP_K,
            1.0f, NULL)) {
        engine_error(error, error_size,
                     "layer-%u prefill router failed", layer);
        return false;
    }
    status = hipStreamSynchronize(NULL);
    if (status != hipSuccess) {
        engine_error(error, error_size,
                     "layer-%u prefill router sync failed: %s",
                     layer, hipGetErrorString(status));
        return false;
    }
    const uint64_t route_count =
        (uint64_t)tokens * K3_ENGINE_TOP_K;
    status = hipMemcpy(
        workspace->route_ids_host,
        workspace->route_ids_device,
        route_count * sizeof(uint32_t),
        hipMemcpyDeviceToHost);
    if (status == hipSuccess) {
        status = hipMemcpy(
            workspace->route_weights_host,
            workspace->route_weights_device,
            route_count * sizeof(float),
            hipMemcpyDeviceToHost);
    }
    if (status != hipSuccess) {
        engine_error(error, error_size,
                     "layer-%u prefill route copy failed: %s",
                     layer, hipGetErrorString(status));
        return false;
    }
    bool selected_experts[K3_ENGINE_EXPERTS] = { false };
    uint32_t unique_experts = 0u;
    for (uint32_t route = 0u;
         route < tokens * K3_ENGINE_TOP_K; route++) {
        if (workspace->route_ids_host[route] >=
            K3_ENGINE_EXPERTS) {
            engine_error(error, error_size,
                         "layer-%u prefill route %u is invalid",
                         layer,
                         workspace->route_ids_host[route]);
            return false;
        }
        const uint32_t expert = workspace->route_ids_host[route];
        if (!selected_experts[expert]) {
            selected_experts[expert] = true;
            unique_experts++;
        }
    }
    stats->selected_expert_routes +=
        (uint64_t)tokens * K3_ENGINE_TOP_K;
    if (stats->unique_experts_across_layers == 0u ||
        unique_experts < stats->min_unique_experts_per_layer) {
        stats->min_unique_experts_per_layer = unique_experts;
    }
    if (unique_experts > stats->max_unique_experts_per_layer) {
        stats->max_unique_experts_per_layer = unique_experts;
    }
    stats->unique_experts_across_layers += unique_experts;
    clock_gettime(CLOCK_MONOTONIC, &phase_end);
    stats->router_seconds +=
        elapsed_seconds(phase_start, phase_end);
    clock_gettime(CLOCK_MONOTONIC, &phase_start);

    if (!k3_static_weight_gemm_bf16(
            routed_down,
            workspace->moe_scratch[M_LATENT],
            workspace->scratch[S_POST_NORM],
            tokens, engine->expert_stream) ||
        !k3_static_weight_gemm_bf16(
            shared_gate,
            workspace->moe_scratch[M_SHARED_GATE],
            workspace->scratch[S_POST_NORM],
            tokens, engine->shared_stream) ||
        !k3_static_weight_gemm_bf16(
            shared_up,
            workspace->moe_scratch[M_SHARED_UP],
            workspace->scratch[S_POST_NORM],
            tokens, engine->shared_stream) ||
        !k3_rocm_situ_bf16(
            workspace->moe_scratch[M_SHARED_ACTIVATION],
            workspace->moe_scratch[M_SHARED_GATE],
            workspace->moe_scratch[M_SHARED_UP],
            (uint64_t)tokens *
                K3_ENGINE_SHARED_HIDDEN,
            4.0f, 25.0f, engine->shared_stream) ||
        !k3_static_weight_gemm_bf16(
            shared_down,
            workspace->moe_scratch[M_SHARED_OUTPUT],
            workspace->moe_scratch[M_SHARED_ACTIVATION],
            tokens, engine->shared_stream)) {
        engine_error(error, error_size,
                     "layer-%u prefill static MoE failed",
                     layer);
        return false;
    }
    status = hipMemsetAsync(
        workspace->moe_scratch[M_EXPERT_OUTPUTS], 0,
        (uint64_t)tokens * K3_ENGINE_TOP_K *
            K3_ENGINE_LATENT * sizeof(uint16_t),
        engine->expert_stream);
    if (status != hipSuccess ||
        !prefill_stream_routed_layer(
            engine, workspace, layer, stats,
            error, error_size)) {
        return false;
    }
    clock_gettime(CLOCK_MONOTONIC, &phase_end);
    stats->routed_stream_seconds +=
        elapsed_seconds(phase_start, phase_end);
    clock_gettime(CLOCK_MONOTONIC, &phase_start);

    status = hipStreamSynchronize(engine->expert_stream);
    if (status != hipSuccess ||
        !k3_rocm_weighted_sum_bf16_batch(
            workspace->moe_scratch[M_ROUTED],
            workspace->moe_scratch[M_EXPERT_OUTPUTS],
            workspace->route_weights_device,
            tokens, K3_ENGINE_TOP_K,
            K3_ENGINE_LATENT, NULL) ||
        !k3_rocm_rms_norm_bf16(
            workspace->moe_scratch[M_ROUTED_NORM],
            workspace->moe_scratch[M_ROUTED],
            routed_norm->data, tokens,
            K3_ENGINE_LATENT, 1e-5f, NULL) ||
        !k3_static_weight_gemm_bf16(
            routed_up,
            workspace->moe_scratch[M_ROUTED_FULL],
            workspace->moe_scratch[M_ROUTED_NORM],
            tokens, NULL)) {
        engine_error(error, error_size,
                     "layer-%u prefill routed MoE tail failed",
                     layer);
        return false;
    }
    status = hipStreamSynchronize(engine->shared_stream);
    if (status != hipSuccess ||
        !k3_rocm_add_bf16(
            workspace->scratch[S_ATTENTION],
            workspace->moe_scratch[M_ROUTED_FULL],
            workspace->moe_scratch[M_SHARED_OUTPUT],
            (uint64_t)tokens * K3_ENGINE_HIDDEN,
            NULL) ||
        !k3_rocm_add_bf16(
            output,
            workspace->scratch[S_MLP_OUTPUT],
            workspace->scratch[S_ATTENTION],
            (uint64_t)tokens * K3_ENGINE_HIDDEN,
            NULL)) {
        engine_error(error, error_size,
                     "layer-%u prefill final residual failed",
                     layer);
        return false;
    }
    status = hipStreamSynchronize(NULL);
    if (status != hipSuccess) {
        engine_error(error, error_size,
                     "layer-%u prefill MoE-tail timing "
                     "sync failed: %s", layer,
                     hipGetErrorString(status));
        return false;
    }
    clock_gettime(CLOCK_MONOTONIC, &phase_end);
    stats->moe_tail_seconds +=
        elapsed_seconds(phase_start, phase_end);
    return true;
}

static bool k3_engine_plan_prefill_for_remaining(
        const k3_engine *engine,
        uint32_t token_count,
        uint32_t context_remaining,
        k3_prefill_cache_lease lease,
        k3_prefill_projection_backend backend,
        k3_prefill_plan *plan,
        char *error,
        size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!engine || !engine->causal_state_valid || !plan ||
        engine->decoded_layers != 0u ||
        context_remaining > engine->context ||
        (backend != K3_PREFILL_PROJECTION_DEFAULT &&
         backend !=
             K3_PREFILL_PROJECTION_KDA_DEQUANT_BLAS_EXPERIMENT)) {
        engine_error(error, error_size,
                     "invalid K3 engine prefill-plan request");
        return false;
    }
    uint64_t auxiliary_workspace_bytes = 0u;
    if (backend ==
        K3_PREFILL_PROJECTION_KDA_DEQUANT_BLAS_EXPERIMENT) {
        const k3_static_weight *probe =
            k3_static_store_find(
                engine->static_store,
                "language_model.model.layers.1."
                "self_attn.q_proj.weight");
        if (!probe ||
            probe->kind != K3_STATIC_WEIGHT_Q8_128) {
            engine_error(
                error, error_size,
                "KDA dequant/BLAS prefill experiment "
                "requires Q8 static projections");
            return false;
        }
        auxiliary_workspace_bytes =
            K3_ENGINE_KDA_DEQUANT_WORKSPACE_BYTES;
    }
    return k3_prefill_plan_build_with_aux_workspace(
        &engine->model, token_count,
        context_remaining,
        engine->experts_per_layer,
        engine->staging_slots, lease,
        auxiliary_workspace_bytes,
        plan, error, error_size);
}

extern "C" bool k3_engine_plan_prefill_with_projection_backend(
        const k3_engine *engine,
        uint32_t token_count,
        k3_prefill_cache_lease lease,
        k3_prefill_projection_backend backend,
        k3_prefill_plan *plan,
        char *error,
        size_t error_size) {
    if (!engine || engine->token_position > engine->context) {
        if (error && error_size) {
            engine_error(error, error_size,
                         "invalid K3 engine prefill-plan request");
        }
        return false;
    }
    return k3_engine_plan_prefill_for_remaining(
        engine, token_count,
        engine->context - engine->token_position,
        lease, backend, plan, error, error_size);
}

extern "C" bool k3_engine_plan_prefill(
        const k3_engine *engine,
        uint32_t token_count,
        k3_prefill_cache_lease lease,
        k3_prefill_plan *plan,
        char *error,
        size_t error_size) {
    return k3_engine_plan_prefill_with_projection_backend(
        engine, token_count, lease,
        K3_PREFILL_PROJECTION_DEFAULT,
        plan, error, error_size);
}

extern "C" bool k3_engine_plan_reset_prefill(
        const k3_engine *engine,
        uint32_t token_count,
        bool clear_expert_cache,
        k3_prefill_projection_backend backend,
        k3_prefill_plan *plan,
        char *error,
        size_t error_size) {
    if (!engine) {
        if (error && error_size) {
            engine_error(error, error_size,
                         "invalid K3 reset-prefill plan request");
        }
        return false;
    }
    return k3_engine_plan_prefill_for_remaining(
        engine, token_count, engine->context,
        clear_expert_cache ?
            K3_PREFILL_CACHE_BORROW_COLD_WORKSPACE :
            K3_PREFILL_CACHE_BORROW_WARM_WORKSPACE,
        backend, plan, error, error_size);
}

static bool k3_engine_forward_range_impl(
        k3_engine *engine,
        const uint32_t *input_tokens,
        uint32_t token_count,
        k3_prefill_projection_backend backend,
        uint32_t *next_token,
        float *token_value,
        k3_engine_prefill_stats *stats,
        k3_engine_prefill_progress_callback progress_callback,
        void *progress_data,
        char *error,
        size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (stats) memset(stats, 0, sizeof(*stats));
    if (!engine || !engine->causal_state_valid ||
        !input_tokens || !next_token ||
        token_count < 2u ||
        engine->decoded_layers != 0u ||
        engine->token_position > engine->context ||
        token_count >
            engine->context - engine->token_position ||
        engine->staging_slots < K3_ENGINE_TOP_K ||
        (backend != K3_PREFILL_PROJECTION_DEFAULT &&
         backend !=
             K3_PREFILL_PROJECTION_KDA_DEQUANT_BLAS_EXPERIMENT)) {
        engine_error(error, error_size,
                     "invalid K3 token-range request");
        return false;
    }
    k3_expert_cache_stats cache_stats;
    k3_expert_cache_get_stats(
        engine->cache_policy, &cache_stats);
    k3_prefill_cache_lease lease =
        engine->token_position == 0u &&
        cache_stats.accesses == 0u ?
            K3_PREFILL_CACHE_BORROW_COLD_WORKSPACE :
            K3_PREFILL_CACHE_RETAIN;
    k3_prefill_plan plan;
    if (!k3_engine_plan_prefill_with_projection_backend(
            engine, token_count, lease, backend,
            &plan, error, error_size)) {
        return false;
    }
    if (lease == K3_PREFILL_CACHE_RETAIN) {
        uint64_t available = 0u;
        uint64_t cma_free = 0u;
        const bool memory_known =
            host_memory_bytes(&available, &cma_free);
        const bool reserve_fits =
            cma_free <= UINT64_MAX -
                K3_ENGINE_HOST_GUARD_BYTES;
        const uint64_t reserve = reserve_fits ?
            cma_free + K3_ENGINE_HOST_GUARD_BYTES :
            UINT64_MAX;
        const bool retain_fits =
            memory_known && reserve_fits &&
            plan.new_device_bytes <= UINT64_MAX - reserve &&
            available >= plan.new_device_bytes + reserve;
        if (!retain_fits) {
            lease = K3_PREFILL_CACHE_BORROW_WARM_WORKSPACE;
            if (!k3_engine_plan_prefill_with_projection_backend(
                    engine, token_count, lease, backend,
                    &plan, error, error_size)) {
                return false;
            }
            const uint32_t first_slot = (uint32_t)(
                plan.retained_cache_bytes /
                    K3_ENGINE_EXPERT_BYTES);
            const uint32_t borrowed_slots = (uint32_t)(
                plan.borrowed_cache_bytes /
                    K3_ENGINE_EXPERT_BYTES);
            /*
             * Range prefill streams routed weights directly and never plans
             * decode-cache admissions. The workspace is destroyed before
             * token decode resumes, so invalidating these mappings makes the
             * borrowed physical range exclusive for the complete loan.
             */
            if (!k3_expert_cache_invalidate_slots(
                    engine->cache_policy,
                    first_slot, borrowed_slots,
                    error, error_size)) {
                return false;
            }
        }
    }
    k3_prefill_workspace workspace;
    if (!prefill_workspace_create(
            engine, &workspace, &plan, lease,
            error, error_size)) {
        return false;
    }
    k3_engine_prefill_stats measured;
    memset(&measured, 0, sizeof(measured));
    if (lease == K3_PREFILL_CACHE_BORROW_WARM_WORKSPACE) {
        measured.warm_cache_workspace_bytes =
            plan.borrowed_cache_bytes;
    }
    struct timespec start;
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    const uint32_t position = engine->token_position;
    struct timespec phase_start;
    struct timespec phase_end;
    clock_gettime(CLOCK_MONOTONIC, &phase_start);
    bool ok = prefill_layer0(
        engine, &workspace, position,
        input_tokens, error, error_size,
        &measured.embedding_physical_read_bytes);
    if (ok) {
        const hipError_t status = hipStreamSynchronize(NULL);
        if (status != hipSuccess) {
            engine_error(error, error_size,
                         "K3 prefill layer-0 timing sync "
                         "failed: %s",
                         hipGetErrorString(status));
            ok = false;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &phase_end);
    measured.layer0_seconds =
        elapsed_seconds(phase_start, phase_end);
    if (ok && progress_callback) {
        progress_callback(1u, K3_ENGINE_LAYERS, progress_data);
    }
    void *input = workspace.hidden[1];
    void *output = workspace.hidden[0];
    for (uint32_t layer = 1u;
         ok && layer < K3_ENGINE_LAYERS; layer++) {
        ok = prefill_routed_layer(
            engine, &workspace, layer, position,
            input, output, backend, &measured,
            error, error_size);
        if (ok) {
            measured.routed_layer_sweeps++;
            void *swap = input;
            input = output;
            output = swap;
            if (progress_callback) {
                progress_callback(
                    layer + 1u, K3_ENGINE_LAYERS,
                    progress_data);
            }
        }
    }
    if (ok &&
        (measured.routed_layer_sweeps !=
             plan.routed_layer_sweeps ||
         measured.expert_read_requests >
             plan.expert_read_requests ||
         measured.routed_physical_read_bytes >
             plan.routed_store_physical_read_bytes)) {
        engine_error(
            error, error_size,
            "K3 prefill I/O ceiling violation: "
            "sweeps %u/%u requests %u/%u bytes %llu/%llu",
            measured.routed_layer_sweeps,
            plan.routed_layer_sweeps,
            measured.expert_read_requests,
            plan.expert_read_requests,
            (unsigned long long)
                measured.routed_physical_read_bytes,
            (unsigned long long)
                plan.routed_store_physical_read_bytes);
        ok = false;
    }
    const uint32_t last = token_count - 1u;
    clock_gettime(CLOCK_MONOTONIC, &phase_start);
    if (ok) {
        hipError_t status = hipMemcpyAsync(
            engine->token_buffer[0],
            prefill_const_slice(
                input, last, K3_ENGINE_HIDDEN),
            (uint64_t)K3_ENGINE_HIDDEN *
                sizeof(uint16_t),
            hipMemcpyDeviceToDevice, NULL);
        for (uint32_t block = 0u;
             status == hipSuccess && block < 8u; block++) {
            status = hipMemcpyAsync(
                prefill_slice(
                    engine->attn_res_blocks,
                    block, K3_ENGINE_HIDDEN),
                prefill_const_slice(
                    workspace.attn_res_blocks,
                    last * 8u + block,
                    K3_ENGINE_HIDDEN),
                (uint64_t)K3_ENGINE_HIDDEN *
                    sizeof(uint16_t),
                hipMemcpyDeviceToDevice, NULL);
        }
        if (status != hipSuccess) {
            engine_error(error, error_size,
                         "K3 prefill final state copy failed: %s",
                         hipGetErrorString(status));
            ok = false;
        }
    }
    if (ok) {
        engine->decoded_layers = K3_ENGINE_LAYERS;
        engine->token_position += token_count - 1u;
        ok = k3_engine_decode_greedy(
            engine, engine->token_buffer[0],
            next_token, token_value, error, error_size);
    }
    clock_gettime(CLOCK_MONOTONIC, &phase_end);
    measured.output_seconds =
        elapsed_seconds(phase_start, phase_end);
    clock_gettime(CLOCK_MONOTONIC, &end);
    measured.wall_seconds = elapsed_seconds(start, end);
    if (stats) *stats = measured;
    prefill_workspace_destroy(&workspace);
    return ok;
}

extern "C" bool k3_engine_forward_range_with_projection_backend(
        k3_engine *engine,
        const uint32_t *input_tokens,
        uint32_t token_count,
        k3_prefill_projection_backend backend,
        uint32_t *next_token,
        float *token_value,
        k3_engine_prefill_stats *stats,
        char *error,
        size_t error_size) {
    return k3_engine_forward_range_impl(
        engine, input_tokens, token_count, backend,
        next_token, token_value, stats,
        NULL, NULL, error, error_size);
}

extern "C" bool
k3_engine_forward_range_with_projection_backend_and_progress(
        k3_engine *engine,
        const uint32_t *input_tokens,
        uint32_t token_count,
        k3_prefill_projection_backend backend,
        uint32_t *next_token,
        float *token_value,
        k3_engine_prefill_stats *stats,
        k3_engine_prefill_progress_callback progress_callback,
        void *progress_data,
        char *error,
        size_t error_size) {
    return k3_engine_forward_range_impl(
        engine, input_tokens, token_count, backend,
        next_token, token_value, stats,
        progress_callback, progress_data,
        error, error_size);
}

extern "C" bool k3_engine_forward_range_with_progress(
        k3_engine *engine,
        const uint32_t *input_tokens,
        uint32_t token_count,
        uint32_t *next_token,
        float *token_value,
        k3_engine_prefill_stats *stats,
        k3_engine_prefill_progress_callback progress_callback,
        void *progress_data,
        char *error,
        size_t error_size) {
    return k3_engine_forward_range_impl(
        engine, input_tokens, token_count,
        K3_PREFILL_PROJECTION_DEFAULT,
        next_token, token_value, stats,
        progress_callback, progress_data,
        error, error_size);
}

extern "C" bool k3_engine_forward_range(
        k3_engine *engine,
        const uint32_t *input_tokens,
        uint32_t token_count,
        uint32_t *next_token,
        float *token_value,
        k3_engine_prefill_stats *stats,
        char *error,
        size_t error_size) {
    return k3_engine_forward_range_impl(
        engine, input_tokens, token_count,
        K3_PREFILL_PROJECTION_DEFAULT,
        next_token, token_value, stats,
        NULL, NULL, error, error_size);
}

static bool prefill_hash_device_bytes(
        const void *device,
        uint64_t bytes,
        uint64_t *hash,
        char *error,
        size_t error_size) {
    const uint64_t chunk_bytes =
        UINT64_C(16) * UINT64_C(1024) * UINT64_C(1024);
    uint8_t *host = (uint8_t *)malloc((size_t)chunk_bytes);
    if (!host) {
        engine_error(error, error_size,
                     "K3 state-digest host allocation failed");
        return false;
    }
    uint64_t value = UINT64_C(1469598103934665603);
    for (uint64_t offset = 0u; offset < bytes;) {
        const uint64_t remaining = bytes - offset;
        const uint64_t count =
            remaining < chunk_bytes ? remaining : chunk_bytes;
        hipError_t status = hipMemcpy(
            host, (const uint8_t *)device + offset,
            count, hipMemcpyDeviceToHost);
        if (status != hipSuccess) {
            engine_error(error, error_size,
                         "K3 state-digest copy failed at %llu: %s",
                         (unsigned long long)offset,
                         hipGetErrorString(status));
            free(host);
            return false;
        }
        for (uint64_t index = 0u; index < count; index++) {
            value ^= host[index];
            value *= UINT64_C(1099511628211);
        }
        offset += count;
    }
    free(host);
    *hash = value;
    return true;
}

extern "C" bool k3_engine_get_state_digest(
        const k3_engine *engine,
        k3_engine_state_digest *digest,
        char *error,
        size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (digest) memset(digest, 0, sizeof(*digest));
    if (!engine || !engine->causal_state_valid || !digest ||
        engine->decoded_layers != 0u) {
        engine_error(error, error_size,
                     "invalid K3 state-digest request");
        return false;
    }
    hipError_t status = hipDeviceSynchronize();
    if (status != hipSuccess) {
        engine_error(error, error_size,
                     "K3 state-digest synchronization failed: %s",
                     hipGetErrorString(status));
        return false;
    }
    if (!prefill_hash_device_bytes(
            engine->kda_state, K3_ENGINE_KDA_STATE_BYTES,
            &digest->kda_state_hash, error, error_size) ||
        !prefill_hash_device_bytes(
            engine->kda_conv, K3_ENGINE_KDA_CONV_BYTES,
            &digest->kda_conv_hash, error, error_size) ||
        !prefill_hash_device_bytes(
            engine->mla_cache, engine->mla_cache_bytes,
            &digest->mla_cache_hash, error, error_size) ||
        !prefill_hash_device_bytes(
            engine->attn_res_blocks,
            8u * K3_ENGINE_HIDDEN * sizeof(uint16_t),
            &digest->attn_res_hash, error, error_size)) {
        return false;
    }
    digest->token_position = engine->token_position;
    return true;
}
