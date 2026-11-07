/*
 * Dump operator inputs and this implementation's outputs so the vendored
 * reference modules can be run against them.
 *
 *   mimo26_dump_ops OUTPUT.bin
 *
 * Little-endian. Sections are length-prefixed and tagged so the reader can
 * skip anything it does not understand:
 *
 *   u32 section_count
 *   per section: char tag[8], u32 payload_bytes, payload
 *
 * "RMSNORM ": u32 count, u16 input[count], u16 weight[count], u16 output[count]
 * "ROUTER  ": u32 experts, u32 topk, f32 logits[experts], f32 bias[experts],
 *             u32 indices[topk], f32 weights[topk]
 * "SILUPROD": u32 count, f32 gate[count], f32 up[count], f32 output[count]
 * "EXPERTACC": u32 experts, u32 count, f32 weights[experts],
 *              u16 outputs[experts][count], u16 combined[count]
 */
#include "mimo26_attention.h"
#include "mimo26_ops.h"
#include "mimo26_router.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RMSNORM_COUNT 4096u
#define SILU_COUNT 2048u
#define EXPERT_COUNT 8u
#define EXPERT_WIDTH 256u

static uint32_t next_random(uint32_t *state)
{
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static float unit_random(uint32_t *state)
{
    return (float)(next_random(state) >> 8) / (float)(1u << 24);
}

static void write_section(FILE *out, const char *tag, const void *payload,
                          uint32_t bytes)
{
    char padded[8];
    memset(padded, ' ', sizeof padded);
    const size_t length = strlen(tag);
    memcpy(padded, tag, length < sizeof padded ? length : sizeof padded);
    fwrite(padded, 1u, sizeof padded, out);
    fwrite(&bytes, sizeof bytes, 1u, out);
    fwrite(payload, 1u, bytes, out);
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s OUTPUT.bin\n", argv[0]);
        return 2;
    }
    FILE *out = fopen(argv[1], "wb");
    if (out == NULL) {
        fprintf(stderr, "cannot write %s\n", argv[1]);
        return 2;
    }
    const uint32_t section_count = 4u;
    fwrite(&section_count, sizeof section_count, 1u, out);
    uint32_t state = 20260923u;
    int result = 1;

    /* RMSNorm */
    {
        const uint32_t count = RMSNORM_COUNT;
        uint16_t *input = malloc(count * sizeof *input);
        uint16_t *weight = malloc(count * sizeof *weight);
        uint16_t *output = malloc(count * sizeof *output);
        if (!input || !weight || !output) {
            goto done;
        }
        for (uint32_t i = 0; i < count; i++) {
            input[i] = mimo26_f32_to_bf16((unit_random(&state) - 0.5f) * 6.0f);
            weight[i] = mimo26_f32_to_bf16(0.5f + unit_random(&state));
        }
        if (mimo26_rmsnorm_bf16(output, input, weight, count,
                                MIMO26_LAYERNORM_EPSILON) != MIMO26_OPS_OK) {
            free(input); free(weight); free(output);
            goto done;
        }
        const uint32_t bytes = (uint32_t)(sizeof count + 3u * count * sizeof *input);
        unsigned char *payload = malloc(bytes);
        if (!payload) { free(input); free(weight); free(output); goto done; }
        size_t offset = 0;
        memcpy(payload + offset, &count, sizeof count); offset += sizeof count;
        memcpy(payload + offset, input, count * sizeof *input);
        offset += count * sizeof *input;
        memcpy(payload + offset, weight, count * sizeof *weight);
        offset += count * sizeof *weight;
        memcpy(payload + offset, output, count * sizeof *output);
        write_section(out, "RMSNORM", payload, bytes);
        free(payload); free(input); free(weight); free(output);
    }

    /* Router */
    {
        const uint32_t experts = MIMO26_ROUTER_EXPERTS;
        const uint32_t topk = MIMO26_ROUTER_TOP_K;
        float *logits = malloc(experts * sizeof *logits);
        float *bias = malloc(experts * sizeof *bias);
        uint32_t indices[MIMO26_ROUTER_TOP_K];
        float weights[MIMO26_ROUTER_TOP_K];
        if (!logits || !bias) { free(logits); free(bias); goto done; }
        for (uint32_t e = 0; e < experts; e++) {
            logits[e] = (unit_random(&state) - 0.5f) * 10.0f;
            bias[e] = (unit_random(&state) - 0.5f) * 0.4f;
        }
        if (mimo26_router_top8_256_f32(indices, weights, logits, bias) !=
            MIMO26_ROUTER_OK) {
            free(logits); free(bias); goto done;
        }
        const uint32_t bytes = (uint32_t)(2u * sizeof experts +
                                          2u * experts * sizeof(float) +
                                          topk * sizeof(uint32_t) +
                                          topk * sizeof(float));
        unsigned char *payload = malloc(bytes);
        if (!payload) { free(logits); free(bias); goto done; }
        size_t offset = 0;
        memcpy(payload + offset, &experts, sizeof experts); offset += sizeof experts;
        memcpy(payload + offset, &topk, sizeof topk); offset += sizeof topk;
        memcpy(payload + offset, logits, experts * sizeof *logits);
        offset += experts * sizeof *logits;
        memcpy(payload + offset, bias, experts * sizeof *bias);
        offset += experts * sizeof *bias;
        memcpy(payload + offset, indices, sizeof indices);
        offset += sizeof indices;
        memcpy(payload + offset, weights, sizeof weights);
        write_section(out, "ROUTER", payload, bytes);
        free(payload); free(logits); free(bias);
    }

    /* SiLU product */
    {
        const uint32_t count = SILU_COUNT;
        float *gate = malloc(count * sizeof *gate);
        float *up = malloc(count * sizeof *up);
        float *output = malloc(count * sizeof *output);
        if (!gate || !up || !output) { free(gate); free(up); free(output); goto done; }
        for (uint32_t i = 0; i < count; i++) {
            gate[i] = (unit_random(&state) - 0.5f) * 16.0f;
            up[i] = (unit_random(&state) - 0.5f) * 16.0f;
        }
        if (mimo26_silu_product_f32(output, gate, up, count) !=
            MIMO26_OPS_OK) {
            free(gate); free(up); free(output); goto done;
        }
        const uint32_t bytes =
            (uint32_t)(sizeof count + 3u * count * sizeof(float));
        unsigned char *payload = malloc(bytes);
        if (!payload) { free(gate); free(up); free(output); goto done; }
        size_t offset = 0;
        memcpy(payload + offset, &count, sizeof count); offset += sizeof count;
        memcpy(payload + offset, gate, count * sizeof *gate);
        offset += count * sizeof *gate;
        memcpy(payload + offset, up, count * sizeof *up);
        offset += count * sizeof *up;
        memcpy(payload + offset, output, count * sizeof *output);
        write_section(out, "SILUPROD", payload, bytes);
        free(payload); free(gate); free(up); free(output);
    }

    /* Weighted expert accumulation */
    {
        const uint32_t experts = EXPERT_COUNT;
        const uint32_t count = EXPERT_WIDTH;
        float weights[EXPERT_COUNT];
        uint16_t *outputs = malloc(experts * count * sizeof *outputs);
        uint16_t *combined = malloc(count * sizeof *combined);
        float *accumulator = calloc(count, sizeof *accumulator);
        if (!outputs || !combined || !accumulator) {
            free(outputs); free(combined); free(accumulator); goto done;
        }
        float total = 0.0f;
        for (uint32_t e = 0; e < experts; e++) {
            weights[e] = unit_random(&state);
            total += weights[e];
        }
        for (uint32_t e = 0; e < experts; e++) {
            weights[e] /= total;
            for (uint32_t i = 0; i < count; i++) {
                outputs[e * count + i] =
                    mimo26_f32_to_bf16((unit_random(&state) - 0.5f) * 4.0f);
            }
        }
        for (uint32_t e = 0; e < experts; e++) {
            if (mimo26_expert_accumulate_f32(accumulator,
                                             outputs + e * count, weights[e],
                                             count) != MIMO26_OPS_OK) {
                free(outputs); free(combined); free(accumulator); goto done;
            }
        }
        if (mimo26_expert_finalize_bf16(combined, accumulator, count) !=
            MIMO26_OPS_OK) {
            free(outputs); free(combined); free(accumulator); goto done;
        }
        const uint32_t bytes =
            (uint32_t)(2u * sizeof experts + experts * sizeof(float) +
                       experts * count * sizeof(uint16_t) +
                       count * sizeof(uint16_t));
        unsigned char *payload = malloc(bytes);
        if (!payload) { free(outputs); free(combined); free(accumulator); goto done; }
        size_t offset = 0;
        memcpy(payload + offset, &experts, sizeof experts); offset += sizeof experts;
        memcpy(payload + offset, &count, sizeof count); offset += sizeof count;
        memcpy(payload + offset, weights, sizeof weights); offset += sizeof weights;
        memcpy(payload + offset, outputs, experts * count * sizeof *outputs);
        offset += experts * count * sizeof *outputs;
        memcpy(payload + offset, combined, count * sizeof *combined);
        write_section(out, "EXPERTACC", payload, bytes);
        free(payload); free(outputs); free(combined); free(accumulator);
    }

    printf("wrote %u operator sections to %s\n", section_count, argv[1]);
    result = 0;
done:
    fclose(out);
    if (result != 0) {
        fprintf(stderr, "dump failed\n");
    }
    return result;
}
