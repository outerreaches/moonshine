/*
 * What the GPU expert path costs, measured rather than assumed.
 *
 * The MoE weights dominate a MiMo decode step: 47 routed layers x 8 experts
 * of 12.75 MiB packed is 4.68 GiB read per token, against a few tens of MiB
 * for everything else. So the achievable rate of the MXFP4 GEMV over GTT
 * sets the ceiling for the whole worker, and it is worth knowing before
 * building residency management around it rather than after.
 *
 * The resident set is deliberately sized like a real per-layer cache. A
 * single expert measured in a loop would sit in cache and report a bandwidth
 * the worker will never see.
 *
 * Reports bytes moved, not FLOPs. This path is entirely memory bound -- one
 * multiply-add per weight byte pair -- so a FLOP figure would flatter it and
 * describe nothing anyone can act on.
 */
#include "k3_rocm_ops.h"

#include <hip/hip_runtime.h>

#include <stdio.h>
#include <stdlib.h>

#define MOE_LAYERS 47u
#define TOP_K 8u
#define TOTAL_EXPERT_IDENTITIES 12032u
#define STATIC_TEXT_GIB 8.2837

typedef struct { const char *name; uint32_t rows, cols; } projection;

int main(int argc, char **argv)
{
    unsigned resident_experts = 64u;
    if (argc > 1) {
        resident_experts = (unsigned)strtoul(argv[1], NULL, 10);
    }

    const projection projections[] = {
        {"gate_proj", 2048u, 4096u},
        {"up_proj",   2048u, 4096u},
        {"down_proj", 4096u, 2048u},
    };
    const size_t count = sizeof projections / sizeof projections[0];

    double packed_per_expert = 0.0;
    for (size_t j = 0; j < count; j++) {
        const double elements =
            (double)projections[j].rows * (double)projections[j].cols;
        packed_per_expert += elements / 2.0     /* two E2M1 codes per byte */
                           + elements / 32.0;   /* one E8M0 per 32 elements */
    }

    hipDeviceProp_t properties;
    if (hipGetDeviceProperties(&properties, 0) != hipSuccess) {
        fprintf(stderr, "no device\n");
        return 1;
    }
    size_t free_bytes = 0, total_bytes = 0;
    hipMemGetInfo(&free_bytes, &total_bytes);
    printf("gfx %s, %d CUs, %.2f GiB addressable\n", properties.gcnArchName,
           properties.multiProcessorCount, (double)total_bytes / 1073741824.0);
    printf("packed bytes per expert: %.2f MiB\n",
           packed_per_expert / 1048576.0);

    uint8_t **weights = (uint8_t **)calloc(resident_experts * count,
                                           sizeof *weights);
    uint8_t **scales = (uint8_t **)calloc(resident_experts * count,
                                          sizeof *scales);
    uint16_t *input = NULL, *output = NULL;
    if (weights == NULL || scales == NULL ||
        hipMalloc(&input, 4096u * sizeof *input) != hipSuccess ||
        hipMalloc(&output, 4096u * sizeof *output) != hipSuccess) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }
    hipMemset(input, 0x3c, 4096u * sizeof *input);
    for (unsigned e = 0; e < resident_experts; e++) {
        for (size_t j = 0; j < count; j++) {
            const size_t elements =
                (size_t)projections[j].rows * projections[j].cols;
            const size_t index = e * count + j;
            if (hipMalloc(&weights[index], elements / 2u) != hipSuccess ||
                hipMalloc(&scales[index], elements / 32u) != hipSuccess) {
                fprintf(stderr, "out of device memory at expert %u\n", e);
                return 1;
            }
            hipMemset(weights[index], 0x52, elements / 2u);
            hipMemset(scales[index], 127, elements / 32u);
        }
    }
    printf("resident packed cache: %.2f GiB across %u experts\n",
           packed_per_expert * resident_experts / 1073741824.0,
           resident_experts);
    hipDeviceSynchronize();

    double achieved = 0.0;
    for (int pass = 0; pass < 2; pass++) {   /* first pass warms */
        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);
        hipEventRecord(start, 0);
        const int iterations = 4;
        for (int it = 0; it < iterations; it++) {
            for (unsigned e = 0; e < resident_experts; e++) {
                for (size_t j = 0; j < count; j++) {
                    const size_t index = e * count + j;
                    k3_rocm_mxfp4_gemv_bf16(output, weights[index],
                                            scales[index], input,
                                            projections[j].rows,
                                            projections[j].cols, NULL);
                }
            }
        }
        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);
        float milliseconds = 0.0f;
        hipEventElapsedTime(&milliseconds, start, stop);
        const double moved =
            packed_per_expert * resident_experts * (double)iterations;
        achieved = moved / 1e9 / ((double)milliseconds / 1000.0);
        if (pass == 1) {
            printf("achieved: %.1f GB/s over %.2f GiB in %.1f ms\n", achieved,
                   moved / 1073741824.0, (double)milliseconds);
        }
    }

    const double per_token_bytes =
        packed_per_expert * MOE_LAYERS * TOP_K;
    const double seconds = per_token_bytes / (achieved * 1e9);
    printf("\nper token: %.2f GiB of packed expert reads -> %.1f ms, "
           "%.1f tok/s on this path alone\n",
           per_token_bytes / 1073741824.0, seconds * 1000.0, 1.0 / seconds);

    /* The comparison that justifies caching packed bytes rather than
     * dequantized ones, which is the single most consequential choice in the
     * GPU design. */
    const double expanded_per_expert = 48.0 * 1048576.0;
    printf("if experts were cached dequantized to BF16 (%.0f MiB each): "
           "%.2f GiB per token, %.1f tok/s\n",
           expanded_per_expert / 1048576.0,
           expanded_per_expert * MOE_LAYERS * TOP_K / 1073741824.0,
           achieved * 1e9 / (expanded_per_expert * MOE_LAYERS * TOP_K));

    const double budget =
        (double)total_bytes / 1073741824.0 - STATIC_TEXT_GIB;
    printf("\n%.2f GiB after static text holds %.0f of %u expert identities "
           "(%.0f%%) packed, %.0f (%.0f%%) expanded\n",
           budget, budget * 1073741824.0 / packed_per_expert,
           TOTAL_EXPERT_IDENTITIES,
           100.0 * budget * 1073741824.0 / packed_per_expert /
               TOTAL_EXPERT_IDENTITIES,
           budget * 1073741824.0 / expanded_per_expert,
           100.0 * budget * 1073741824.0 / expanded_per_expert /
               TOTAL_EXPERT_IDENTITIES);
    return 0;
}
