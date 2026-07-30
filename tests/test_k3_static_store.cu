#include "k3_static_store.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t status_ = (call);                                        \
        if (status_ != hipSuccess) {                                        \
            fprintf(stderr, "FAIL: %s: %s\n", #call,                        \
                    hipGetErrorString(status_));                            \
            return 1;                                                       \
        }                                                                   \
    } while (0)

#define CHECK(condition, message)                                           \
    do {                                                                    \
        if (!(condition)) {                                                 \
            fprintf(stderr, "FAIL: %s\n", (message));                       \
            return 1;                                                       \
        }                                                                   \
    } while (0)

enum {
    K3_HIDDEN = 7168,
    K3_KDA_INNER = 12288,
    K3_EXPECTED_WEIGHTS = 2459,
    K3_EXPECTED_Q8_WEIGHTS = 1135,
};

static const uint64_t K3_EXPECTED_SOURCE_BYTES =
    UINT64_C(111160730624);
static const uint64_t K3_EXPECTED_RESIDENT_BYTES =
    UINT64_C(59345729536);
static const uint64_t K3_EXPECTED_Q8_SOURCE_BYTES =
    UINT64_C(106972905472);
static const uint64_t K3_EXPECTED_Q8_RESIDENT_BYTES =
    UINT64_C(55157904384);
static const uint64_t K3_EXPECTED_PEAK_TRANSIENT_BYTES =
    UINT64_C(2348810240);
static const uint64_t K3_CACHE_32_BYTES =
    UINT64_C(51659145216);
static const uint64_t K3_RUNTIME_8K_BYTES =
    UINT64_C(1268580352);
static const uint64_t K3_REQUIRED_GUARD_BYTES =
    UINT64_C(4) * UINT64_C(1024) * UINT64_C(1024) * UINT64_C(1024);

static double elapsed_seconds(struct timespec start,
                              struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) +
           (double)(end.tv_nsec - start.tv_nsec) / 1e9;
}

static uint64_t mem_available_bytes(void) {
    FILE *source = fopen("/proc/meminfo", "r");
    if (!source) return 0;
    char key[64];
    unsigned long long value = 0;
    char unit[32];
    uint64_t result = 0;
    while (fscanf(source, "%63s %llu %31s",
                  key, &value, unit) == 3) {
        if (strcmp(key, "MemAvailable:") == 0) {
            result = (uint64_t)value * 1024u;
            break;
        }
    }
    fclose(source);
    return result;
}

static uint32_t rng_state = UINT32_C(0x4b335354);

static float random_input(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return ((float)(rng_state & UINT32_C(0x00ffffff)) /
            (float)UINT32_C(0x01000000) * 2.0f - 1.0f) * 0.125f;
}

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] :
        "/srv/modelstore/models/moonshotai__Kimi-K3";
    const bool plan_only =
        argc > 2 && strcmp(argv[2], "plan") == 0;
    if (argc > 3 || (argc > 2 && !plan_only)) {
        fprintf(stderr, "usage: %s [MODEL_ROOT [plan]]\n", argv[0]);
        return 2;
    }
    if (plan_only) {
        char error[512];
        k3_st_model model;
        CHECK(k3_st_model_open(
                  &model, root, 96u, error, sizeof(error)),
              error);
        k3_static_store_stats bf16_plan;
        k3_static_store_stats q8_plan;
        CHECK(k3_static_store_plan(
                  &model, false, &bf16_plan,
                  error, sizeof(error)),
              error);
        CHECK(k3_static_store_plan(
                  &model, true, &q8_plan,
                  error, sizeof(error)),
              error);
        CHECK(bf16_plan.source_bytes ==
                  K3_EXPECTED_SOURCE_BYTES &&
              bf16_plan.resident_bytes ==
                  K3_EXPECTED_SOURCE_BYTES &&
              bf16_plan.q8_weight_count == 0u &&
              bf16_plan.peak_transient_bytes ==
                  K3_EXPECTED_PEAK_TRANSIENT_BYTES,
              "BF16 static plan");
        CHECK(q8_plan.source_bytes ==
                  K3_EXPECTED_SOURCE_BYTES &&
              q8_plan.resident_bytes ==
                  K3_EXPECTED_RESIDENT_BYTES &&
              q8_plan.q8_source_bytes ==
                  K3_EXPECTED_Q8_SOURCE_BYTES &&
              q8_plan.q8_resident_bytes ==
                  K3_EXPECTED_Q8_RESIDENT_BYTES &&
              q8_plan.q8_weight_count ==
                  K3_EXPECTED_Q8_WEIGHTS &&
              q8_plan.peak_transient_bytes ==
                  K3_EXPECTED_PEAK_TRANSIENT_BYTES,
              "Q8 static plan");
        printf("K3 static plan: BF16 %.3f GiB; "
               "Q8 %.3f GiB; peak transient %.3f GiB: PASS\n",
               bf16_plan.resident_bytes / 1073741824.0,
               q8_plan.resident_bytes / 1073741824.0,
               q8_plan.peak_transient_bytes / 1073741824.0);
        k3_st_model_close(&model);
        return 0;
    }
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("K3 static store: SKIP (no ROCm device)\n");
        return 0;
    }
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t properties;
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    printf("K3 static store on %s (%s)\n",
           properties.name, properties.gcnArchName);

    size_t hip_free_before = 0;
    size_t hip_total = 0;
    HIP_CHECK(hipMemGetInfo(&hip_free_before, &hip_total));
    const uint64_t mem_before = mem_available_bytes();

    char error[512];
    k3_st_model model;
    CHECK(k3_st_model_open(&model, root, 96u,
                           error, sizeof(error)),
          error);
    struct timespec load_start;
    struct timespec load_end;
    k3_static_store *store = NULL;
    k3_static_store_stats stats;
    clock_gettime(CLOCK_MONOTONIC, &load_start);
    CHECK(k3_static_store_load(
              &store, &model, true, &stats,
              error, sizeof(error)),
          error);
    clock_gettime(CLOCK_MONOTONIC, &load_end);

    CHECK(stats.weight_count == K3_EXPECTED_WEIGHTS,
          "static-store weight count");
    CHECK(stats.q8_weight_count == K3_EXPECTED_Q8_WEIGHTS,
          "static-store Q8 weight count");
    CHECK(stats.source_bytes == K3_EXPECTED_SOURCE_BYTES,
          "static-store source-byte ledger");
    CHECK(stats.resident_bytes == K3_EXPECTED_RESIDENT_BYTES,
          "static-store resident-byte ledger");
    CHECK(stats.q8_source_bytes == K3_EXPECTED_Q8_SOURCE_BYTES,
          "static-store Q8 source-byte ledger");
    CHECK(stats.q8_resident_bytes ==
              K3_EXPECTED_Q8_RESIDENT_BYTES,
          "static-store Q8 resident-byte ledger");

    const k3_static_weight *q_projection =
        k3_static_store_find(
            store,
            "language_model.model.layers.0.self_attn.q_proj.weight");
    const k3_static_weight *kv_b =
        k3_static_store_find(
            store,
            "language_model.model.layers.3.self_attn.kv_b_proj.weight");
    const k3_static_weight *router =
        k3_static_store_find(
            store,
            "language_model.model.layers.1.block_sparse_moe.gate.weight");
    const k3_static_weight *attn_res =
        k3_static_store_find(
            store,
            "language_model.model.layers.92.self_attention_res_proj.weight");
    const k3_static_weight *head =
        k3_static_store_find(
            store, "language_model.lm_head.weight");
    CHECK(q_projection && kv_b && router && attn_res && head,
          "static-store representative lookup");
    CHECK(q_projection->kind == K3_STATIC_WEIGHT_Q8_128,
          "KDA projection should be Q8");
    CHECK(kv_b->kind == K3_STATIC_WEIGHT_BF16,
          "MLA kv_b must stay BF16");
    CHECK(router->kind == K3_STATIC_WEIGHT_BF16,
          "router must stay BF16");
    CHECK(attn_res->kind == K3_STATIC_WEIGHT_BF16,
          "AttnRes score vector must stay BF16");
    CHECK(head->kind == K3_STATIC_WEIGHT_BF16,
          "output head must stay BF16");

    const size_t input_bytes =
        (size_t)K3_HIDDEN * sizeof(hip_bfloat16);
    const size_t output_bytes =
        (size_t)K3_KDA_INNER * sizeof(hip_bfloat16);
    hip_bfloat16 *input =
        (hip_bfloat16 *)malloc(input_bytes);
    hip_bfloat16 *output =
        (hip_bfloat16 *)malloc(output_bytes);
    CHECK(input && output, "static-store GEMV host allocation");
    for (uint32_t i = 0; i < K3_HIDDEN; i++) {
        input[i] = hip_bfloat16(random_input());
    }
    void *d_input = NULL;
    void *d_output = NULL;
    HIP_CHECK(hipMalloc(&d_input, input_bytes));
    HIP_CHECK(hipMalloc(&d_output, output_bytes));
    HIP_CHECK(hipMemcpy(d_input, input, input_bytes,
                        hipMemcpyHostToDevice));
    CHECK(k3_static_weight_gemv_bf16(
              q_projection, d_output, d_input, NULL),
          "post-source-release Q8 GEMV launch");
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(output, d_output, output_bytes,
                        hipMemcpyDeviceToHost));
    bool finite_nonzero = false;
    for (uint32_t i = 0; i < K3_KDA_INNER; i++) {
        const float value = (float)output[i];
        CHECK(isfinite(value), "static-store GEMV produced non-finite output");
        if (value != 0.0f) finite_nonzero = true;
    }
    CHECK(finite_nonzero,
          "static-store GEMV output is identically zero");
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_input));
    free(output);
    free(input);

    size_t hip_free_loaded = 0;
    HIP_CHECK(hipMemGetInfo(&hip_free_loaded, &hip_total));
    const uint64_t mem_loaded = mem_available_bytes();
    printf("  source: %.3f GiB across %u weights\n",
           stats.source_bytes / 1073741824.0,
           stats.weight_count);
    printf("  Q8: %.3f GiB from %.3f GiB across %u matrices\n",
           stats.q8_resident_bytes / 1073741824.0,
           stats.q8_source_bytes / 1073741824.0,
           stats.q8_weight_count);
    printf("  resident static tier: %.3f GiB "
           "(ledger exact)\n",
           stats.resident_bytes / 1073741824.0);
    printf("  load wall %.3f s: reads %.3f s at %.3f GiB/s; "
           "upload/quantize %.3f s\n",
           elapsed_seconds(load_start, load_end),
           stats.read_seconds,
           stats.physical_read_bytes /
               fmax(stats.read_seconds, 1e-9) / 1073741824.0,
           stats.upload_quantize_seconds);
    printf("  MemAvailable: %.3f GiB -> %.3f GiB; "
           "HIP free: %.3f GiB -> %.3f GiB\n",
           mem_before / 1073741824.0,
           mem_loaded / 1073741824.0,
           hip_free_before / 1073741824.0,
           hip_free_loaded / 1073741824.0);

    void *cache_device = NULL;
    void *runtime_device = NULL;
    struct timespec reserve_start;
    struct timespec reserve_end;
    clock_gettime(CLOCK_MONOTONIC, &reserve_start);
    HIP_CHECK(hipMalloc(
        &cache_device, K3_CACHE_32_BYTES));
    HIP_CHECK(hipMalloc(
        &runtime_device, K3_RUNTIME_8K_BYTES));
    HIP_CHECK(hipMemset(
        cache_device, 0, K3_CACHE_32_BYTES));
    HIP_CHECK(hipMemset(
        runtime_device, 0, K3_RUNTIME_8K_BYTES));
    HIP_CHECK(hipDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &reserve_end);
    CHECK(cache_device != NULL,
          "device cache allocation");
    const uint64_t mem_planned = mem_available_bytes();
    size_t hip_free_planned = 0;
    HIP_CHECK(hipMemGetInfo(&hip_free_planned, &hip_total));
    printf("  full plan: +%.3f GiB device cache + "
           "%.3f GiB 8K runtime in %.3f s\n",
           K3_CACHE_32_BYTES / 1073741824.0,
           K3_RUNTIME_8K_BYTES / 1073741824.0,
           elapsed_seconds(reserve_start, reserve_end));
    printf("  observed full-plan headroom: MemAvailable %.3f GiB; "
           "HIP free %.3f GiB\n",
           mem_planned / 1073741824.0,
           hip_free_planned / 1073741824.0);
    CHECK(mem_planned >= K3_REQUIRED_GUARD_BYTES,
          "full 32/layer plan violates 4 GiB MemAvailable guard");

    HIP_CHECK(hipFree(runtime_device));
    HIP_CHECK(hipFree(cache_device));
    k3_static_store_destroy(store);
    k3_st_model_close(&model);
    const uint64_t mem_released = mem_available_bytes();
    size_t hip_free_released = 0;
    HIP_CHECK(hipMemGetInfo(&hip_free_released, &hip_total));
    printf("  released: MemAvailable %.3f GiB; HIP free %.3f GiB\n",
           mem_released / 1073741824.0,
           hip_free_released / 1073741824.0);
    printf("K3 static store: PASS\n");
    return 0;
}
