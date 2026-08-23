#include "k3_mzg.h"
#include "k3_safetensors.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(condition, message)                                           \
    do {                                                                    \
        if (!(condition)) {                                                 \
            fprintf(stderr, "FAIL: %s\n", (message));                      \
            return 1;                                                       \
        }                                                                   \
    } while (0)

enum {
    K3_SHARDS = 96,
    K3_EXPERT_BYTES = 17547264,
    K3_PACKED_BYTES = 5505024,
    K3_ALIGNMENT = 4096,
};

static bool pread_full(int fd, void *buffer, size_t bytes, uint64_t offset) {
    uint8_t *cursor = (uint8_t *)buffer;
    size_t remaining = bytes;
    while (remaining != 0u) {
        ssize_t got = pread(fd, cursor, remaining, (off_t)offset);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return false;
        cursor += (size_t)got;
        offset += (uint64_t)got;
        remaining -= (size_t)got;
    }
    return true;
}

static void canonicalize_packed(uint8_t *data, size_t bytes) {
    for (size_t index = 0u; index < bytes; index++) {
        uint8_t low = data[index] & UINT8_C(0x0f);
        uint8_t high = data[index] >> 4u;
        if (low == UINT8_C(0x08)) low = 0u;
        if (high == UINT8_C(0x08)) high = 0u;
        data[index] = low | (uint8_t)(high << 4u);
    }
}

static void fill_pattern(uint8_t *destination, size_t bytes,
                         const uint8_t *pattern, size_t pattern_bytes) {
    for (size_t index = 0u; index < bytes; index++) {
        destination[index] = pattern[index % pattern_bytes];
    }
}

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] :
        "/srv/modelstore/models/moonshotai__Kimi-K3";
    const uint32_t layers =
        argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 10) : 92u;
    const uint32_t experts =
        argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 10) : 896u;
    char error[512];
    const char *configured_store = getenv("MOONSHINE_EXPERT_STORE");
    char *saved_store = configured_store ? strdup(configured_store) : NULL;
    CHECK(!configured_store || saved_store,
          "copy MZG store selection");
    CHECK(setenv(
              "MOONSHINE_EXPERT_STORE",
              "/nonexistent/moonshine-mzg-store", 1) == 0,
          "set missing MZG store selection");
    k3_mzg_store *missing = NULL;
    CHECK(!k3_mzg_store_open_optional(
              &missing, root, layers, experts,
              error, sizeof(error)) &&
              !missing && error[0] != '\0',
          "explicit missing MZG store did not fail closed");
    CHECK(saved_store ?
              setenv("MOONSHINE_EXPERT_STORE", saved_store, 1) == 0 :
              unsetenv("MOONSHINE_EXPERT_STORE") == 0,
          "restore MZG store selection");
    free(saved_store);

    k3_mzg_store *store = NULL;
    CHECK(k3_mzg_store_open_optional(
              &store, root, layers, experts,
              error, sizeof(error)),
          error);
    if (!store) {
        printf("K3 MZG native store: SKIP (no expert-store-mzg1)\n");
        return 0;
    }

    k3_mzg_span span;
    CHECK(k3_mzg_store_span(store, 1u, 0u, &span),
          "MZG span lookup");
    CHECK(span.offset % K3_ALIGNMENT == 0u &&
              span.bytes % K3_ALIGNMENT == 0u &&
              span.bytes <= K3_EXPERT_BYTES + K3_ALIGNMENT,
          "MZG span bounds");
    void *block = NULL;
    CHECK(posix_memalign(&block, K3_ALIGNMENT, span.bytes) == 0,
          "MZG aligned block allocation");
    CHECK(pread_full(span.direct_fd, block, span.bytes, span.offset),
          "MZG O_DIRECT block read");
    uint8_t *decoded = (uint8_t *)malloc(K3_EXPERT_BYTES);
    CHECK(decoded != NULL, "MZG decoded allocation");
    CHECK(k3_mzg_store_decode(
              store, 1u, 0u, block, span.bytes,
              decoded, error, sizeof(error)),
          error);

    k3_st_model model;
    memset(&model, 0, sizeof(model));
    k3_st_read source;
    memset(&source, 0, sizeof(source));
    bool source_open = false;
    uint8_t *expected = (uint8_t *)malloc(K3_EXPERT_BYTES);
    CHECK(expected != NULL, "source canonical allocation");
    if (layers == 1u && experts == 1u) {
        static const uint8_t packed[] =
            { 0x00u, 0x01u, 0x20u, 0xf0u, 0x37u, 0x00u };
        static const uint8_t scale1[] = { 121u, 121u, 122u };
        static const uint8_t scale2[] = { 123u, 123u, 124u };
        static const uint8_t scale3[] = { 125u, 125u, 126u };
        fill_pattern(expected, K3_PACKED_BYTES,
                     packed, sizeof(packed));
        fill_pattern(expected + UINT64_C(5505024), UINT64_C(344064),
                     scale1, sizeof(scale1));
        fill_pattern(expected + UINT64_C(5849088), K3_PACKED_BYTES,
                     packed, sizeof(packed));
        fill_pattern(expected + UINT64_C(11354112), UINT64_C(344064),
                     scale2, sizeof(scale2));
        fill_pattern(expected + UINT64_C(11698176), K3_PACKED_BYTES,
                     packed, sizeof(packed));
        fill_pattern(expected + UINT64_C(17203200), UINT64_C(344064),
                     scale3, sizeof(scale3));
    } else {
        CHECK(k3_st_model_open(&model, root, K3_SHARDS,
                               error, sizeof(error)), error);
        source_open = true;
        static const char *first_name =
            "language_model.model.layers.1.block_sparse_moe."
            "experts.0.w1.weight_packed";
        static const char *last_name =
            "language_model.model.layers.1.block_sparse_moe."
            "experts.0.w3.weight_scale";
        const k3_st_tensor *first = k3_st_find(&model, first_name);
        const k3_st_tensor *last = k3_st_find(&model, last_name);
        CHECK(first && last && first->shard == last->shard &&
                  last->physical_offset + last->byte_length -
                      first->physical_offset == K3_EXPERT_BYTES,
              "source expert layout");
        CHECK(k3_st_read_span(
                  &model, first->shard, first->physical_offset,
                  K3_EXPERT_BYTES, K3_ALIGNMENT,
                  &source, error, sizeof(error)),
              error);
        memcpy(expected, source.data, K3_EXPERT_BYTES);
        canonicalize_packed(expected, K3_PACKED_BYTES);
        canonicalize_packed(
            expected + UINT64_C(5849088), K3_PACKED_BYTES);
        canonicalize_packed(
            expected + UINT64_C(11698176), K3_PACKED_BYTES);
    }
    CHECK(memcmp(decoded, expected, K3_EXPERT_BYTES) == 0,
          "native MZG decode differs from zero-canonical source");

    uint8_t *corrupt = (uint8_t *)block;
    corrupt[192u + 1024u] ^= UINT8_C(0x01);
    CHECK(!k3_mzg_store_decode(
              store, 1u, 0u, block, span.bytes,
              decoded, error, sizeof(error)),
          "corrupt MZG frame was accepted");

    printf("K3 MZG native store: PASS span=%u bytes\n", span.bytes);
    free(expected);
    if (source.allocation) k3_st_read_release(&source);
    if (source_open) k3_st_model_close(&model);
    free(decoded);
    free(block);
    k3_mzg_store_destroy(store);
    return 0;
}
