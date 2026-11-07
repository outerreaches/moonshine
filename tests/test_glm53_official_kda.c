#include "../glm53_state_oracle.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define T 5u
#define D 128u
#define HEADER_BYTES 36u
#define ARRAY_COUNT 8u
#define FLOAT_COUNT (4u*T*D + T + D*D + T*D + D*D)
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); goto done; \
} } while (0)

static uint32_t load_u32le(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static float load_f32le(const unsigned char *p) {
    uint32_t bits = load_u32le(p);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int close_enough(float got, float expected, float *worst) {
    float error = fabsf(got - expected);
    float scale = fmaxf(fabsf(got), fabsf(expected));
    float limit = 3.0e-6f + 4.0e-6f * scale;
    if (error > *worst) *worst = error;
    return isfinite(got) && isfinite(expected) && error <= limit;
}

int main(int argc, char **argv) {
    const char *path = argc == 2 ? argv[1] :
        "tests/fixtures/glm53_phase4_kda_v1.bin";
    static const unsigned char magic[8] = {'G','5','3','K','D','A','4',0};
    unsigned char header[HEADER_BYTES], *raw = NULL;
    float *fixture = NULL, *full_state = NULL, *split_state = NULL;
    float *full_out = NULL, *split_out = NULL, *full_scratch = NULL;
    float *split_scratch = NULL;
    const float *q, *k, *v, *g, *beta, *initial, *expected_out, *expected_state;
    size_t state_n = D*D, out_n = T*D, scratch_n, split_scratch_n;
    size_t i, offset = 0u;
    float worst_out = 0.0f, worst_state = 0.0f;
    FILE *file = NULL;
    long file_size;
    int result = 1;

    if (argc > 2) {
        fprintf(stderr, "usage: %s [FIXTURE]\n", argv[0]);
        return 2;
    }
    CHECK(sizeof(float) == 4u);
    file = fopen(path, "rb");
    CHECK(file != NULL);
    CHECK(fseek(file, 0, SEEK_END) == 0);
    file_size = ftell(file);
    CHECK(file_size == (long)(HEADER_BYTES + FLOAT_COUNT*sizeof(float)));
    CHECK(fseek(file, 0, SEEK_SET) == 0);
    CHECK(fread(header, 1u, sizeof(header), file) == sizeof(header));
    CHECK(memcmp(header, magic, sizeof(magic)) == 0);
    CHECK(load_u32le(header + 8u) == 1u);
    CHECK(load_u32le(header + 12u) == UINT32_C(0x01020304));
    CHECK(load_u32le(header + 16u) == T);
    CHECK(load_u32le(header + 20u) == D);
    CHECK(load_u32le(header + 24u) == D);
    CHECK(load_u32le(header + 28u) == ARRAY_COUNT);
    CHECK(load_u32le(header + 32u) == FLOAT_COUNT);

    raw = (unsigned char *)malloc(FLOAT_COUNT*sizeof(float));
    fixture = (float *)malloc(FLOAT_COUNT*sizeof(float));
    CHECK(raw != NULL && fixture != NULL);
    CHECK(fread(raw, sizeof(float), FLOAT_COUNT, file) == FLOAT_COUNT);
    CHECK(fgetc(file) == EOF);
    CHECK(fclose(file) == 0); file = NULL;
    for (i = 0u; i < FLOAT_COUNT; ++i) {
        fixture[i] = load_f32le(raw + i*sizeof(float));
        CHECK(isfinite(fixture[i]));
    }
    q = fixture + offset; offset += T*D;
    k = fixture + offset; offset += T*D;
    v = fixture + offset; offset += T*D;
    g = fixture + offset; offset += T*D;
    beta = fixture + offset; offset += T;
    initial = fixture + offset; offset += state_n;
    expected_out = fixture + offset; offset += out_n;
    expected_state = fixture + offset; offset += state_n;
    CHECK(offset == FLOAT_COUNT);
    for (i = 0u; i < T; ++i) CHECK(beta[i] >= 0.0f && beta[i] <= 1.0f);
    for (i = 0u; i < T*D; ++i) CHECK(g[i] >= -5.0f && g[i] <= 0.0f);

    scratch_n = glm53_kda_scratch_floats(D, D, T);
    split_scratch_n = glm53_kda_scratch_floats(D, D, 3u);
    CHECK(scratch_n != 0u && split_scratch_n != 0u);
    full_state = (float *)malloc(state_n*sizeof(float));
    split_state = (float *)malloc(state_n*sizeof(float));
    full_out = (float *)malloc(out_n*sizeof(float));
    split_out = (float *)malloc(out_n*sizeof(float));
    full_scratch = (float *)malloc(scratch_n*sizeof(float));
    split_scratch = (float *)malloc(split_scratch_n*sizeof(float));
    CHECK(full_state && split_state && full_out && split_out &&
          full_scratch && split_scratch);

    memcpy(full_state, initial, state_n*sizeof(float));
    CHECK(glm53_kda_chunk_f32(q, k, v, g, beta, T, full_state, D, D,
                              full_out, full_scratch, scratch_n));
    for (i = 0u; i < out_n; ++i)
        CHECK(close_enough(full_out[i], expected_out[i], &worst_out));
    for (i = 0u; i < state_n; ++i)
        CHECK(close_enough(full_state[i], expected_state[i], &worst_state));

    memcpy(split_state, initial, state_n*sizeof(float));
    CHECK(glm53_kda_chunk_f32(q, k, v, g, beta, 2u, split_state, D, D,
                              split_out, split_scratch, split_scratch_n));
    CHECK(glm53_kda_chunk_f32(q + 2u*D, k + 2u*D, v + 2u*D,
                              g + 2u*D, beta + 2u, 3u, split_state, D, D,
                              split_out + 2u*D, split_scratch,
                              split_scratch_n));
    CHECK(memcmp(full_out, split_out, out_n*sizeof(float)) == 0);
    CHECK(memcmp(full_state, split_state, state_n*sizeof(float)) == 0);

    printf("PASS official layer0 head0 KDA T=%u D=%u split=2+3 "
           "max_output_error=%g max_state_error=%g\n",
           (unsigned)T, (unsigned)D, worst_out, worst_state);
    result = 0;

done:
    if (file != NULL) (void)fclose(file);
    free(split_scratch); free(full_scratch); free(split_out); free(full_out);
    free(split_state); free(full_state); free(fixture); free(raw);
    return result;
}
