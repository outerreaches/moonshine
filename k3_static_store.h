#ifndef K3_STATIC_STORE_H
#define K3_STATIC_STORE_H

#include "k3_safetensors.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    K3_STATIC_WEIGHT_RAW = 0,
    K3_STATIC_WEIGHT_BF16,
    K3_STATIC_WEIGHT_Q8_128,
} k3_static_weight_kind;

typedef struct {
    const k3_st_tensor    *tensor;
    void                  *data;
    void                  *scales;
    uint64_t               resident_bytes;
    k3_static_weight_kind  kind;
} k3_static_weight;

typedef struct {
    uint64_t source_bytes;
    uint64_t physical_read_bytes;
    uint64_t resident_bytes;
    uint64_t q8_source_bytes;
    uint64_t q8_resident_bytes;
    uint64_t peak_transient_bytes;
    uint32_t weight_count;
    uint32_t q8_weight_count;
    double   read_seconds;
    double   upload_quantize_seconds;
} k3_static_store_stats;

typedef struct k3_static_store k3_static_store;

/*
 * Derive the exact static residency and a conservative peak load transient
 * without allocating host or device tensor storage.
 */
bool k3_static_store_plan(const k3_st_model      *model,
                          bool                    q8_projections,
                          k3_static_store_stats  *stats,
                          char                   *error,
                          size_t                  error_size);

/*
 * Load the permanent text tier while excluding routed MXFP4 experts and the
 * input embedding table. With q8_projections enabled, eligible BF16 matrices
 * are quantized one at a time and their transient source allocations are
 * released immediately.
 *
 * The SafeTensors model must outlive the store because entries retain pointers
 * into its sorted tensor directory.
 */
bool k3_static_store_load(k3_static_store       **out,
                          const k3_st_model      *model,
                          bool                    q8_projections,
                          k3_static_store_stats  *stats,
                          char                   *error,
                          size_t                  error_size);

const k3_static_weight *k3_static_store_find(
    const k3_static_store *store,
    const char            *name);

bool k3_static_weight_gemv_bf16(const k3_static_weight *weight,
                                void                   *output,
                                const void             *input,
                                void                   *stream);

bool k3_static_weight_gemm_bf16(const k3_static_weight *weight,
                                void                   *output,
                                const void             *input,
                                uint32_t                vector_count,
                                void                   *stream);

bool k3_static_weight_is_q8_candidate(const k3_st_tensor *tensor);

void k3_static_store_destroy(k3_static_store *store);

#ifdef __cplusplus
}
#endif

#endif
