#include "k3_static_store.h"

#include "k3_rocm_ops.h"

#include <hip/hip_runtime.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct k3_static_store {
    k3_static_weight *weights;
    uint32_t count;
};

static void k3_store_error(char *error,
                           size_t error_size,
                           const char *format,
                           ...) {
    if (!error || error_size == 0) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static double elapsed_seconds(struct timespec start,
                              struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) +
           (double)(end.tv_nsec - start.tv_nsec) / 1e9;
}

static bool is_static_text_tensor(const k3_st_tensor *tensor) {
    return tensor && tensor->name &&
           strncmp(tensor->name, "language_model.", 15u) == 0 &&
           strstr(tensor->name, ".block_sparse_moe.experts.") == NULL &&
           strstr(tensor->name, "embed_tokens.weight") == NULL;
}

static bool q8_layout(const k3_st_tensor *tensor,
                      uint64_t *data_bytes,
                      uint64_t *scale_bytes) {
    if (!k3_static_weight_is_q8_candidate(tensor) ||
        tensor->shape[0] > UINT32_MAX ||
        tensor->shape[1] > UINT32_MAX) {
        return false;
    }
    const uint64_t rows = tensor->shape[0];
    const uint64_t columns = tensor->shape[1];
    *data_bytes = rows * columns;
    *scale_bytes =
        rows * (columns / 128u) * sizeof(float);
    return true;
}

extern "C" bool k3_static_weight_is_q8_candidate(
        const k3_st_tensor *tensor) {
    if (!tensor || tensor->dtype != K3_ST_DTYPE_BF16 ||
        tensor->ndim != 2u || tensor->shape[1] % 128u != 0u ||
        strstr(tensor->name, ".block_sparse_moe.experts.") ||
        strstr(tensor->name, "embed_tokens.weight") ||
        strstr(tensor->name, "lm_head.weight") ||
        strstr(tensor->name, ".block_sparse_moe.gate.") ||
        strstr(tensor->name, "_res_proj.weight") ||
        strstr(tensor->name, "output_attn_res_proj.weight")) {
        return false;
    }
    const size_t length = strlen(tensor->name);
    static const char suffix[] = ".self_attn.kv_b_proj.weight";
    const size_t suffix_length = sizeof(suffix) - 1u;
    return length < suffix_length ||
           strcmp(tensor->name + length - suffix_length, suffix) != 0;
}

extern "C" bool k3_static_store_plan(
        const k3_st_model      *model,
        bool                    q8_projections,
        k3_static_store_stats  *stats,
        char                   *error,
        size_t                  error_size) {
    if (error && error_size) error[0] = '\0';
    if (stats) memset(stats, 0, sizeof(*stats));
    if (!model || !model->tensors || model->tensor_count == 0u ||
        !stats) {
        k3_store_error(error, error_size,
                       "invalid static-store plan arguments");
        return false;
    }
    k3_static_store_stats planned;
    memset(&planned, 0, sizeof(planned));
    for (size_t i = 0; i < model->tensor_count; i++) {
        const k3_st_tensor *tensor = &model->tensors[i];
        if (!is_static_text_tensor(tensor)) continue;
        planned.source_bytes += tensor->byte_length;
        planned.weight_count++;

        uint64_t resident_bytes = tensor->byte_length;
        uint64_t transient_bytes = tensor->byte_length;
        uint64_t data_bytes = 0;
        uint64_t scale_bytes = 0;
        if (q8_projections &&
            q8_layout(tensor, &data_bytes, &scale_bytes)) {
            resident_bytes = data_bytes + scale_bytes;
            planned.q8_source_bytes += tensor->byte_length;
            planned.q8_resident_bytes += resident_bytes;
            planned.q8_weight_count++;
            /*
             * The Q8 loader briefly owns both the aligned host read and a
             * BF16 device source while the packed destination is already
             * represented in resident_bytes.
             */
            if (tensor->byte_length <= UINT64_MAX / 2u) {
                transient_bytes = tensor->byte_length * 2u;
            } else {
                transient_bytes = UINT64_MAX;
            }
        }
        if (planned.resident_bytes >
            UINT64_MAX - resident_bytes) {
            k3_store_error(error, error_size,
                           "static-store plan byte count overflow");
            return false;
        }
        planned.resident_bytes += resident_bytes;
        if (transient_bytes > planned.peak_transient_bytes) {
            planned.peak_transient_bytes = transient_bytes;
        }
    }
    *stats = planned;
    return true;
}

static void release_weight(k3_static_weight *weight) {
    if (weight->scales) (void)hipFree(weight->scales);
    if (weight->data) (void)hipFree(weight->data);
    memset(weight, 0, sizeof(*weight));
}

extern "C" void k3_static_store_destroy(k3_static_store *store) {
    if (!store) return;
    for (uint32_t i = 0; i < store->count; i++) {
        release_weight(&store->weights[i]);
    }
    free(store->weights);
    free(store);
}

extern "C" bool k3_static_store_load(
        k3_static_store       **out,
        const k3_st_model      *model,
        bool                    q8_projections,
        k3_static_store_stats  *stats,
        char                   *error,
        size_t                  error_size) {
    if (error && error_size) error[0] = '\0';
    if (out) *out = NULL;
    if (stats) memset(stats, 0, sizeof(*stats));
    if (!out || !model || !model->tensors || model->tensor_count == 0u) {
        k3_store_error(error, error_size,
                       "invalid static-store load arguments");
        return false;
    }
    k3_static_store_stats planned;
    if (!k3_static_store_plan(
            model, q8_projections, &planned,
            error, error_size)) {
        return false;
    }
    const uint32_t count = planned.weight_count;
    k3_static_store *store =
        (k3_static_store *)calloc(1, sizeof(*store));
    if (!store) {
        k3_store_error(error, error_size,
                       "static-store allocation failed");
        return false;
    }
    store->weights = (k3_static_weight *)calloc(
        count, sizeof(*store->weights));
    if (!store->weights) {
        free(store);
        k3_store_error(error, error_size,
                       "static-store weight table allocation failed");
        return false;
    }

    k3_static_store_stats measured;
    memset(&measured, 0, sizeof(measured));
    for (size_t tensor_index = 0;
         tensor_index < model->tensor_count; tensor_index++) {
        const k3_st_tensor *tensor = &model->tensors[tensor_index];
        if (!is_static_text_tensor(tensor)) continue;
        k3_static_weight *weight =
            &store->weights[store->count];
        weight->tensor = tensor;

        struct timespec read_start;
        struct timespec read_end;
        k3_st_read read;
        memset(&read, 0, sizeof(read));
        clock_gettime(CLOCK_MONOTONIC, &read_start);
        if (!k3_st_read_span(model, tensor->shard,
                             tensor->physical_offset,
                             tensor->byte_length, 4096u,
                             &read, error, error_size)) {
            k3_static_store_destroy(store);
            return false;
        }
        clock_gettime(CLOCK_MONOTONIC, &read_end);
        measured.read_seconds +=
            elapsed_seconds(read_start, read_end);
        measured.source_bytes += tensor->byte_length;
        measured.physical_read_bytes += read.allocation_bytes;

        struct timespec convert_start;
        struct timespec convert_end;
        clock_gettime(CLOCK_MONOTONIC, &convert_start);
        hipError_t hip_status = hipSuccess;
        if (q8_projections &&
            k3_static_weight_is_q8_candidate(tensor)) {
            const uint32_t rows = (uint32_t)tensor->shape[0];
            const uint32_t columns = (uint32_t)tensor->shape[1];
            uint64_t data_bytes = 0;
            uint64_t scale_bytes = 0;
            (void)q8_layout(
                tensor, &data_bytes, &scale_bytes);
            void *source = NULL;
            hip_status = hipMalloc(&source, tensor->byte_length);
            if (hip_status == hipSuccess) {
                hip_status = hipMemcpy(
                    source, read.data, tensor->byte_length,
                    hipMemcpyHostToDevice);
            }
            if (hip_status == hipSuccess) {
                hip_status = hipMalloc(&weight->data, data_bytes);
            }
            if (hip_status == hipSuccess) {
                hip_status = hipMalloc(&weight->scales, scale_bytes);
            }
            if (hip_status == hipSuccess &&
                !k3_rocm_bf16_quantize_q8_128(
                    weight->data, weight->scales, source,
                    rows, columns, NULL)) {
                hip_status = hipErrorLaunchFailure;
            }
            if (hip_status == hipSuccess) {
                hip_status = hipDeviceSynchronize();
            }
            if (source) (void)hipFree(source);
            weight->kind = K3_STATIC_WEIGHT_Q8_128;
            weight->resident_bytes = data_bytes + scale_bytes;
            measured.q8_source_bytes += tensor->byte_length;
            measured.q8_resident_bytes += weight->resident_bytes;
            measured.q8_weight_count++;
        } else {
            hip_status = hipMalloc(&weight->data,
                                   tensor->byte_length);
            if (hip_status == hipSuccess) {
                hip_status = hipMemcpy(
                    weight->data, read.data, tensor->byte_length,
                    hipMemcpyHostToDevice);
            }
            weight->kind = tensor->dtype == K3_ST_DTYPE_BF16 ?
                K3_STATIC_WEIGHT_BF16 : K3_STATIC_WEIGHT_RAW;
            weight->resident_bytes = tensor->byte_length;
        }
        clock_gettime(CLOCK_MONOTONIC, &convert_end);
        measured.upload_quantize_seconds +=
            elapsed_seconds(convert_start, convert_end);
        k3_st_read_release(&read);
        if (hip_status != hipSuccess) {
            k3_store_error(
                error, error_size,
                "ROCm static load failed for %s: %s",
                tensor->name, hipGetErrorString(hip_status));
            release_weight(weight);
            k3_static_store_destroy(store);
            return false;
        }
        measured.resident_bytes += weight->resident_bytes;
        measured.weight_count++;
        store->count++;
    }
    if (store->count != count) {
        k3_store_error(error, error_size,
                       "static-store count changed during load");
        k3_static_store_destroy(store);
        return false;
    }
    measured.peak_transient_bytes =
        planned.peak_transient_bytes;
    if (stats) *stats = measured;
    *out = store;
    return true;
}

extern "C" const k3_static_weight *k3_static_store_find(
        const k3_static_store *store,
        const char *name) {
    if (!store || !name) return NULL;
    uint32_t low = 0;
    uint32_t high = store->count;
    while (low < high) {
        const uint32_t middle = low + (high - low) / 2u;
        const int comparison =
            strcmp(name, store->weights[middle].tensor->name);
        if (comparison == 0) return &store->weights[middle];
        if (comparison < 0) high = middle;
        else low = middle + 1u;
    }
    return NULL;
}

extern "C" bool k3_static_weight_gemv_bf16(
        const k3_static_weight *weight,
        void *output,
        const void *input,
        void *stream) {
    if (!weight || !output || !input || !weight->tensor ||
        weight->tensor->dtype != K3_ST_DTYPE_BF16 ||
        weight->tensor->ndim != 2u ||
        weight->tensor->shape[0] > UINT32_MAX ||
        weight->tensor->shape[1] > UINT32_MAX) {
        return false;
    }
    const uint32_t rows =
        (uint32_t)weight->tensor->shape[0];
    const uint32_t columns =
        (uint32_t)weight->tensor->shape[1];
    if (weight->kind == K3_STATIC_WEIGHT_Q8_128) {
        return k3_rocm_q8_128_gemv_bf16(
            output, weight->data, weight->scales, input,
            rows, columns, stream);
    }
    return weight->kind == K3_STATIC_WEIGHT_BF16 &&
           k3_rocm_bf16_gemv_bf16(
               output, weight->data, input,
               rows, columns, stream);
}

extern "C" bool k3_static_weight_gemm_bf16(
        const k3_static_weight *weight,
        void *output,
        const void *input,
        uint32_t vector_count,
        void *stream) {
    if (!weight || !output || !input || vector_count == 0u ||
        !weight->tensor ||
        weight->tensor->dtype != K3_ST_DTYPE_BF16 ||
        weight->tensor->ndim != 2u ||
        weight->tensor->shape[0] > UINT32_MAX ||
        weight->tensor->shape[1] > UINT32_MAX) {
        return false;
    }
    const uint32_t rows =
        (uint32_t)weight->tensor->shape[0];
    const uint32_t columns =
        (uint32_t)weight->tensor->shape[1];
    if (weight->kind == K3_STATIC_WEIGHT_Q8_128) {
        return k3_rocm_q8_128_gemm_bf16(
            output, weight->data, weight->scales, input,
            vector_count, rows, columns, stream);
    }
    return weight->kind == K3_STATIC_WEIGHT_BF16 &&
           k3_rocm_bf16_gemm_bf16(
               output, weight->data, input,
               vector_count, rows, columns, stream);
}
