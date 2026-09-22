/*
 * Dump RoPE inputs and outputs so an independent implementation can check
 * them. Deterministic: the same seed always produces the same vectors.
 *
 *   mimo26_dump_rope OUTPUT.bin
 *
 * Format, little-endian throughout:
 *   u32 case_count
 *   u32 head_dim
 *   u32 rope_dim
 *   then per case: u64 position, f32 theta,
 *                  u16 input[head_dim], u16 output[head_dim],
 *                  u16 cos[rope_dim], u16 sin[rope_dim]
 */
#include "mimo26_attention.h"
#include "mimo26_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t next_random(uint32_t *state)
{
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static float unit_random(uint32_t *state)
{
    return (float)(next_random(state) >> 8) / (float)(1u << 24);
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

    struct { uint64_t position; float theta; } cases[] = {
        {0u, MIMO26_SWA_ROPE_THETA},
        {1u, MIMO26_SWA_ROPE_THETA},
        {127u, MIMO26_SWA_ROPE_THETA},
        {128u, MIMO26_SWA_ROPE_THETA},
        {129u, MIMO26_SWA_ROPE_THETA},
        {4096u, MIMO26_SWA_ROPE_THETA},
        {0u, MIMO26_GLOBAL_ROPE_THETA},
        {1u, MIMO26_GLOBAL_ROPE_THETA},
        {4096u, MIMO26_GLOBAL_ROPE_THETA},
        {65535u, MIMO26_GLOBAL_ROPE_THETA},
        {1048575u, MIMO26_GLOBAL_ROPE_THETA},
    };
    const uint32_t case_count = (uint32_t)(sizeof cases / sizeof cases[0]);
    const uint32_t head_dim = MIMO26_QK_HEAD_DIM;
    const uint32_t rope_dim = MIMO26_ROPE_DIM;
    fwrite(&case_count, sizeof case_count, 1u, out);
    fwrite(&head_dim, sizeof head_dim, 1u, out);
    fwrite(&rope_dim, sizeof rope_dim, 1u, out);

    uint32_t state = 20260922u;
    for (uint32_t c = 0; c < case_count; c++) {
        uint16_t input[MIMO26_QK_HEAD_DIM];
        uint16_t output[MIMO26_QK_HEAD_DIM];
        uint16_t cos_table[MIMO26_ROPE_DIM];
        uint16_t sin_table[MIMO26_ROPE_DIM];

        for (size_t d = 0; d < MIMO26_QK_HEAD_DIM; d++) {
            input[d] = mimo26_f32_to_bf16((unit_random(&state) - 0.5f) * 4.0f);
        }
        if (mimo26_rope_table(cos_table, sin_table, cases[c].position,
                              cases[c].theta) != MIMO26_ATTENTION_OK) {
            fprintf(stderr, "rope table failed at case %u\n", c);
            fclose(out);
            return 1;
        }
        memcpy(output, input, sizeof output);
        if (mimo26_rope_apply(output, cos_table, sin_table) !=
            MIMO26_ATTENTION_OK) {
            fprintf(stderr, "rope apply failed at case %u\n", c);
            fclose(out);
            return 1;
        }

        fwrite(&cases[c].position, sizeof cases[c].position, 1u, out);
        fwrite(&cases[c].theta, sizeof cases[c].theta, 1u, out);
        fwrite(input, sizeof input, 1u, out);
        fwrite(output, sizeof output, 1u, out);
        fwrite(cos_table, sizeof cos_table, 1u, out);
        fwrite(sin_table, sizeof sin_table, 1u, out);
    }
    fclose(out);
    printf("wrote %u RoPE cases to %s\n", case_count, argv[1]);
    return 0;
}
