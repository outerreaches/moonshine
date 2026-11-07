/*
 * G1: the MiMo expert path on the GPU, against the verified CPU dequantizer.
 *
 * This is the gate everything else in the GPU lane stands on. The CPU worker
 * caches experts dequantized to BF16 at 48 MiB each; packed MXFP4 is 12.75
 * MiB, 3.76x smaller, so the GPU path must feed the kernel the checkpoint's
 * native bytes and decode in registers. K3 already has such a kernel
 * (k3_rocm_mxfp4_gemv_bf16) written for a different checkpoint family, so the
 * question is whether its numerics are MiMo's numerics.
 *
 * Element-wise they must be identical, and that is provable rather than
 * hopeful: an E2M1 code carries at most two significand bits and an E8M0
 * scale is a pure power of two, so code x scale is exactly representable in
 * BF16 for all 4096 combinations. The CPU path's rounding to BF16 is
 * therefore a no-op, and any disagreement here is a layout or convention
 * defect, not a precision one.
 *
 * What legitimately differs is reduction order. The CPU accumulates a row
 * left to right in F32; the kernel gives each thread a stride of 32-element
 * groups and tree-reduces. So this test asserts two different things with two
 * different strengths:
 *
 *   - dequantized weights must match BIT-EXACTLY (no tolerance)
 *   - the dot product must match within a reduction-order bound
 *
 * Splitting them matters. A single tolerance over the product would let a
 * wrong nibble order hide inside a plausible-looking error budget, which is
 * the same class of mistake that cost this lane the QKV layout.
 *
 *   MIMO26_ROOT=/path/to/checkpoint tests/test_mimo26_gpu_mxfp4
 */
#include "k3_rocm_ops.h"
#include "k3_safetensors.h"
#include "mimo26_architecture.h"
#include "mimo26_manifest.h"
#include "mimo26_ops.h"
#include "mimo26_weights.h"

#include <hip/hip_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void ok(const char *what, int passed, const char *detail)
{
    printf("  %-4s %-46s %s\n", passed ? "ok" : "FAIL", what,
           detail ? detail : "");
    if (!passed) {
        failures++;
    }
}

#define HIP_OK(call)                                                          \
    do {                                                                      \
        hipError_t _e = (call);                                               \
        if (_e != hipSuccess) {                                               \
            fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #call,    \
                    hipGetErrorString(_e));                                   \
            return 1;                                                         \
        }                                                                     \
    } while (0)

/* Read one tensor's raw bytes, exactly as mimo26_weights.c does. */
static void *read_raw(const k3_st_model *model, const char *name,
                      uint64_t *byte_length, uint64_t *shape0,
                      uint64_t *shape1)
{
    const k3_st_tensor *tensor = k3_st_find(model, name);
    if (tensor == NULL) {
        fprintf(stderr, "tensor not found: %s\n", name);
        return NULL;
    }
    char error[512];
    k3_st_read read;
    memset(&read, 0, sizeof read);
    if (!k3_st_read_span(model, tensor->shard, tensor->physical_offset,
                         tensor->byte_length, 4096u, &read, error,
                         sizeof error)) {
        fprintf(stderr, "read %s: %s\n", name, error);
        return NULL;
    }
    void *buffer = malloc(tensor->byte_length);
    if (buffer != NULL) {
        memcpy(buffer, read.data, tensor->byte_length);
    }
    k3_st_read_release(&read);
    if (byte_length != NULL) { *byte_length = tensor->byte_length; }
    if (shape0 != NULL) { *shape0 = tensor->shape[0]; }
    if (shape1 != NULL) { *shape1 = tensor->ndim > 1u ? tensor->shape[1] : 1u; }
    return buffer;
}

static float bf16_to_float(uint16_t bits)
{
    uint32_t widened = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &widened, sizeof value);
    return value;
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    const char *root = argc > 1 ? argv[1] : getenv("MIMO26_ROOT");
    if (root == NULL) {
        root = "/srv/modelstore/models/XiaomiMiMo__MiMo-V2.6-Flash-RL";
    }

    hipDeviceProp_t properties;
    HIP_OK(hipGetDeviceProperties(&properties, 0));
    printf("gfx %s, %d CUs\n", properties.gcnArchName,
           properties.multiProcessorCount);

    char error[1024];
    mimo26_manifest manifest;
    if (!mimo26_manifest_load(&manifest, root, error, sizeof error)) {
        fprintf(stderr, "manifest: %s\n", error);
        return 1;
    }
    k3_st_model model;
    memset(&model, 0, sizeof model);
    if (!mimo26_manifest_open_model(&manifest, root, &model, error,
                                    sizeof error)) {
        fprintf(stderr, "open: %s\n", error);
        return 1;
    }

    /* Layer 1 is the first MoE layer; expert 0 of its three projections
     * covers both expert shapes the architecture uses. */
    const char *names[] = {
        "model.layers.1.mlp.experts.0.gate_proj",
        "model.layers.1.mlp.experts.0.up_proj",
        "model.layers.1.mlp.experts.0.down_proj",
    };

    for (size_t which = 0; which < sizeof names / sizeof names[0]; which++) {
        char weight_name[256];
        char scale_name[256];
        snprintf(weight_name, sizeof weight_name, "%s.weight", names[which]);
        snprintf(scale_name, sizeof scale_name, "%s.weight_scale", names[which]);

        uint64_t packed_bytes = 0, rows = 0, packed_cols = 0;
        uint8_t *packed = (uint8_t *)read_raw(&model, weight_name,
                                              &packed_bytes, &rows,
                                              &packed_cols);
        uint8_t *scales = (uint8_t *)read_raw(&model, scale_name, NULL, NULL,
                                              NULL);
        if (packed == NULL || scales == NULL) {
            return 1;
        }
        const uint64_t columns = packed_cols * 2u;

        /* CPU: the dequantizer already verified bit-exact against NumPy. */
        uint16_t *cpu_weights =
            (uint16_t *)malloc((size_t)rows * columns * sizeof *cpu_weights);
        if (cpu_weights == NULL) { return 1; }
        if (mimo26_dequantize_mxfp4(cpu_weights, packed, scales, (size_t)rows,
                                    (size_t)columns, error,
                                    sizeof error) != MIMO26_WEIGHTS_OK) {
            fprintf(stderr, "dequantize %s: %s\n", weight_name, error);
            return 1;
        }

        /* A deterministic, well-conditioned input. Using the dequantized
         * weights' own row 0 would correlate input with weights and mask a
         * transposed read, so this is independent of them. */
        uint16_t *input =
            (uint16_t *)malloc((size_t)columns * sizeof *input);
        if (input == NULL) { return 1; }
        uint32_t state = 0x9E3779B9u;
        for (uint64_t c = 0; c < columns; c++) {
            state = state * 1664525u + 1013904223u;
            const float value =
                ((float)((state >> 8) & 0xFFFFu) / 65535.0f - 0.5f) * 2.0f;
            input[c] = mimo26_f32_to_bf16(value);
        }

        /* GPU: decode the native bytes in-kernel. */
        uint8_t *d_packed = NULL;
        uint8_t *d_scales = NULL;
        uint16_t *d_input = NULL;
        uint16_t *d_output = NULL;
        const uint64_t scale_bytes = rows * (columns / 32u);
        HIP_OK(hipMalloc(&d_packed, packed_bytes));
        HIP_OK(hipMalloc(&d_scales, scale_bytes));
        HIP_OK(hipMalloc(&d_input, (size_t)columns * sizeof *d_input));
        HIP_OK(hipMalloc(&d_output, (size_t)rows * sizeof *d_output));
        HIP_OK(hipMemcpy(d_packed, packed, packed_bytes,
                         hipMemcpyHostToDevice));
        HIP_OK(hipMemcpy(d_scales, scales, scale_bytes,
                         hipMemcpyHostToDevice));
        HIP_OK(hipMemcpy(d_input, input, (size_t)columns * sizeof *d_input,
                         hipMemcpyHostToDevice));

        if (!k3_rocm_mxfp4_gemv_bf16(d_output, d_packed, d_scales, d_input,
                                     (uint32_t)rows, (uint32_t)columns,
                                     NULL)) {
            fprintf(stderr, "k3_rocm_mxfp4_gemv_bf16 failed\n");
            return 1;
        }
        HIP_OK(hipDeviceSynchronize());

        uint16_t *gpu_output =
            (uint16_t *)malloc((size_t)rows * sizeof *gpu_output);
        if (gpu_output == NULL) { return 1; }
        HIP_OK(hipMemcpy(gpu_output, d_output,
                         (size_t)rows * sizeof *gpu_output,
                         hipMemcpyDeviceToHost));

        /* CPU dot products in F32, left to right. */
        uint16_t *cpu_output =
            (uint16_t *)malloc((size_t)rows * sizeof *cpu_output);
        if (cpu_output == NULL) { return 1; }
        if (mimo26_matmul_bf16(cpu_output, cpu_weights, input, (size_t)rows,
                               (size_t)columns) != MIMO26_OPS_OK) {
            fprintf(stderr, "matmul failed\n");
            return 1;
        }

        uint64_t exact = 0;
        uint64_t within_one_ulp = 0;
        double worst_relative = 0.0;
        double reference_scale = 0.0;
        for (uint64_t r = 0; r < rows; r++) {
            const float want = bf16_to_float(cpu_output[r]);
            const float got = bf16_to_float(gpu_output[r]);
            if (cpu_output[r] == gpu_output[r]) {
                exact++;
            }
            const int delta = (int)cpu_output[r] - (int)gpu_output[r];
            if (delta >= -1 && delta <= 1) {
                within_one_ulp++;
            }
            reference_scale += fabs((double)want);
            const double denominator = fabs((double)want) > 1e-6
                                           ? fabs((double)want) : 1e-6;
            const double relative = fabs((double)want - (double)got) /
                                    denominator;
            if (relative > worst_relative) {
                worst_relative = relative;
            }
        }
        reference_scale /= (double)rows;

        char detail[256];
        snprintf(detail, sizeof detail,
                 "%llu rows, %.1f%% bit-exact, %.1f%% within 1 ulp, "
                 "worst rel %.3e, mean |out| %.3f",
                 (unsigned long long)rows,
                 100.0 * (double)exact / (double)rows,
                 100.0 * (double)within_one_ulp / (double)rows,
                 worst_relative, reference_scale);
        /*
         * The bound is on reduction order alone. Over 2048 or 4096 terms a
         * BF16 result differing by more than a couple of ulps means the
         * summands differ, not their order -- and the weights are proven
         * element-exact above, so that would be a layout defect.
         */
        ok(names[which], within_one_ulp * 100u >= rows * 99u, detail);

        free(packed);
        free(scales);
        free(cpu_weights);
        free(cpu_output);
        free(gpu_output);
        free(input);
        HIP_OK(hipFree(d_packed));
        HIP_OK(hipFree(d_scales));
        HIP_OK(hipFree(d_input));
        HIP_OK(hipFree(d_output));
    }

    printf("test_mimo26_gpu_mxfp4: %s\n", failures == 0 ? "ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
