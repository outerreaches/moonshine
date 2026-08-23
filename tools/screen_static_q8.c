#include "k3_q8_codec.h"
#include "k3_safetensors.h"
#include "k3_static_store.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <zstd.h>

enum {
    K3_STATIC_Q8_SHARDS = 96,
    K3_STATIC_Q8_EXPECTED_MATRICES = 1135,
    K3_STATIC_Q8_READ_BYTES = 64 * 1024 * 1024,
};

static const uint64_t K3_STATIC_Q8_EXPECTED_SOURCE_BYTES =
    UINT64_C(106972905472);
static const uint64_t K3_STATIC_Q8_EXPECTED_VALUE_BYTES =
    UINT64_C(53486452736);
static const uint64_t K3_STATIC_Q8_EXPECTED_SCALE_BYTES =
    UINT64_C(1671451648);

typedef struct {
    size_t tile_bytes;
    size_t fill;
    size_t compressed_capacity;
    uint8_t *tile;
    uint8_t *compressed;
    uint64_t raw_bytes;
    uint64_t stored_bytes;
    uint64_t tiles;
    uint64_t raw_escape_tiles;
} tile_screen;

typedef struct {
    tile_screen values;
    tile_screen scales;
} screen_pair;

static double elapsed_seconds(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) +
           (double)(end.tv_nsec - start.tv_nsec) / 1e9;
}

static bool add_u64(uint64_t *value, uint64_t increment) {
    if (!value || increment > UINT64_MAX - *value) return false;
    *value += increment;
    return true;
}

static bool tile_screen_init(tile_screen *screen, size_t tile_bytes) {
    if (!screen || tile_bytes == 0u) return false;
    memset(screen, 0, sizeof(*screen));
    screen->tile_bytes = tile_bytes;
    screen->compressed_capacity = ZSTD_compressBound(tile_bytes);
    screen->tile = (uint8_t *)malloc(tile_bytes);
    screen->compressed = (uint8_t *)malloc(screen->compressed_capacity);
    if (!screen->tile || !screen->compressed) {
        free(screen->tile);
        free(screen->compressed);
        memset(screen, 0, sizeof(*screen));
        return false;
    }
    return true;
}

static void tile_screen_destroy(tile_screen *screen) {
    if (!screen) return;
    free(screen->tile);
    free(screen->compressed);
    memset(screen, 0, sizeof(*screen));
}

static bool tile_screen_flush(tile_screen *screen) {
    if (!screen || !screen->tile || !screen->compressed) return false;
    if (screen->fill == 0u) return true;
    const size_t compressed_bytes = ZSTD_compress(
        screen->compressed, screen->compressed_capacity,
        screen->tile, screen->fill, 1);
    if (ZSTD_isError(compressed_bytes)) return false;
    const bool raw_escape = compressed_bytes >= screen->fill;
    const uint64_t stored = raw_escape ?
        (uint64_t)screen->fill : (uint64_t)compressed_bytes;
    if (!add_u64(&screen->raw_bytes, (uint64_t)screen->fill) ||
        !add_u64(&screen->stored_bytes, stored) ||
        !add_u64(&screen->tiles, 1u) ||
        (raw_escape && !add_u64(&screen->raw_escape_tiles, 1u))) {
        return false;
    }
    screen->fill = 0u;
    return true;
}

static bool tile_screen_add(tile_screen *screen,
                            const void *data,
                            size_t bytes) {
    if (!screen || (!data && bytes != 0u)) return false;
    const uint8_t *source = (const uint8_t *)data;
    while (bytes != 0u) {
        size_t available = screen->tile_bytes - screen->fill;
        size_t chunk = bytes < available ? bytes : available;
        memcpy(screen->tile + screen->fill, source, chunk);
        screen->fill += chunk;
        source += chunk;
        bytes -= chunk;
        if (screen->fill == screen->tile_bytes &&
            !tile_screen_flush(screen)) {
            return false;
        }
    }
    return true;
}

static bool pread_full(int fd, void *buffer, size_t bytes, uint64_t offset) {
    uint8_t *destination = (uint8_t *)buffer;
    size_t completed = 0u;
    while (completed < bytes) {
        const ssize_t result = pread(
            fd, destination + completed, bytes - completed,
            (off_t)(offset + completed));
        if (result < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (result == 0) return false;
        completed += (size_t)result;
    }
    return true;
}

static void json_string(FILE *output, const char *value) {
    fputc('"', output);
    for (const unsigned char *cursor = (const unsigned char *)value;
         cursor && *cursor; cursor++) {
        switch (*cursor) {
        case '"': fputs("\\\"", output); break;
        case '\\': fputs("\\\\", output); break;
        case '\n': fputs("\\n", output); break;
        case '\r': fputs("\\r", output); break;
        case '\t': fputs("\\t", output); break;
        default:
            if (*cursor < 0x20u) {
                fprintf(output, "\\u%04x", (unsigned)*cursor);
            } else {
                fputc(*cursor, output);
            }
            break;
        }
    }
    fputc('"', output);
}

static double reduction(uint64_t raw, uint64_t stored) {
    return raw == 0u ? 0.0 : 1.0 - (double)stored / (double)raw;
}

static bool write_result(FILE *output,
                         const char *model_root,
                         const screen_pair *screens,
                         size_t screen_count,
                         const k3_q8_histogram *value_histogram,
                         const k3_q8_histogram *scale_histogram,
                         uint32_t matrix_count,
                         uint64_t source_bytes,
                         uint64_t value_bytes,
                         uint64_t scale_bytes,
                         double seconds) {
    if (!output || !model_root || !screens || !value_histogram ||
        !scale_histogram) {
        return false;
    }
    fputs("{\n  \"schema\": \"moonshine-static-q8-screen-v1\",\n", output);
    fputs("  \"model_root\": ", output);
    json_string(output, model_root);
    fprintf(output,
            ",\n  \"matrix_count\": %u,\n"
            "  \"source_bf16_bytes\": %" PRIu64 ",\n"
            "  \"q8_value_bytes\": %" PRIu64 ",\n"
            "  \"q8_scale_bytes\": %" PRIu64 ",\n"
            "  \"q8_resident_bytes\": %" PRIu64 ",\n"
            "  \"value_entropy_bits_per_symbol\": %.9f,\n"
            "  \"scale_byte_entropy_bits_per_symbol\": %.9f,\n"
            "  \"screen_seconds\": %.6f,\n"
            "  \"zstd1_independent_tiles\": [\n",
            matrix_count, source_bytes, value_bytes, scale_bytes,
            value_bytes + scale_bytes,
            k3_q8_histogram_entropy(value_histogram),
            k3_q8_histogram_entropy(scale_histogram), seconds);
    for (size_t index = 0u; index < screen_count; index++) {
        const tile_screen *values = &screens[index].values;
        const tile_screen *scales = &screens[index].scales;
        const uint64_t raw = values->raw_bytes + scales->raw_bytes;
        const uint64_t stored = values->stored_bytes + scales->stored_bytes;
        fprintf(output,
                "    {\"tile_bytes\": %zu, "
                "\"value_raw_bytes\": %" PRIu64 ", "
                "\"value_stored_bytes\": %" PRIu64 ", "
                "\"value_tiles\": %" PRIu64 ", "
                "\"value_raw_escape_tiles\": %" PRIu64 ", "
                "\"scale_raw_bytes\": %" PRIu64 ", "
                "\"scale_stored_bytes\": %" PRIu64 ", "
                "\"scale_tiles\": %" PRIu64 ", "
                "\"scale_raw_escape_tiles\": %" PRIu64 ", "
                "\"payload_reduction\": %.9f}%s\n",
                values->tile_bytes,
                values->raw_bytes, values->stored_bytes,
                values->tiles, values->raw_escape_tiles,
                scales->raw_bytes, scales->stored_bytes,
                scales->tiles, scales->raw_escape_tiles,
                reduction(raw, stored),
                index + 1u == screen_count ? "" : ",");
    }
    fputs("  ],\n  \"notes\": [\n"
          "    \"Zstd frames and raw escapes are independent per tensor and tile.\",\n"
          "    \"Payload reduction excludes a future format descriptor, checksum, and alignment overhead.\",\n"
          "    \"CPU Q8 bytes require a later full-corpus GPU-quantizer identity gate before publication.\"\n"
          "  ]\n}\n", output);
    return !ferror(output);
}

static bool run_self_test(void) {
    tile_screen screen;
    uint8_t data[20u * 1024u] = {0};
    if (!tile_screen_init(&screen, 16u * 1024u)) return false;
    const bool ok =
        tile_screen_add(&screen, data, sizeof(data)) &&
        tile_screen_flush(&screen) &&
        screen.raw_bytes == sizeof(data) &&
        screen.stored_bytes < screen.raw_bytes &&
        screen.tiles == 2u &&
        screen.raw_escape_tiles == 0u;
    tile_screen_destroy(&screen);
    return ok;
}

static void usage(FILE *output, const char *program) {
    fprintf(output,
            "usage: %s MODEL_ROOT OUTPUT_JSON\n"
            "       %s --self-test\n\n"
            "Offline-only full K3 static Q8 entropy/Zstd screen. The command "
            "reads about 100 GiB from MODEL_ROOT and must not run beside a "
            "latency-sensitive model service. OUTPUT_JSON is published "
            "atomically.\n",
            program, program);
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        usage(stdout, argv[0]);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--self-test") == 0) {
        if (!run_self_test()) {
            fprintf(stderr, "screen_static_q8: self-test failed\n");
            return 1;
        }
        printf("K3 static Q8 screen: PASS\n");
        return 0;
    }
    if (argc != 3) {
        usage(stderr, argv[0]);
        return 2;
    }
    const char *model_root = argv[1];
    const char *output_path = argv[2];
    if (access(output_path, F_OK) == 0) {
        fprintf(stderr,
                "screen_static_q8: output already exists: %s\n",
                output_path);
        return 1;
    }

    k3_st_model model;
    char error[512] = {0};
    if (!k3_st_model_open(
            &model, model_root, K3_STATIC_Q8_SHARDS,
            error, sizeof(error))) {
        fprintf(stderr, "screen_static_q8: %s\n", error);
        return 1;
    }

    static const size_t tile_sizes[] = {16u * 1024u, 32u * 1024u, 64u * 1024u};
    screen_pair screens[sizeof(tile_sizes) / sizeof(tile_sizes[0])];
    memset(screens, 0, sizeof(screens));
    bool initialized = true;
    for (size_t index = 0u; index < sizeof(screens) / sizeof(screens[0]); index++) {
        if (!tile_screen_init(&screens[index].values, tile_sizes[index]) ||
            !tile_screen_init(&screens[index].scales, tile_sizes[index])) {
            initialized = false;
            break;
        }
    }
    if (!initialized) {
        fprintf(stderr, "screen_static_q8: allocating tile screens failed\n");
        for (size_t index = 0u; index < sizeof(screens) / sizeof(screens[0]); index++) {
            tile_screen_destroy(&screens[index].values);
            tile_screen_destroy(&screens[index].scales);
        }
        k3_st_model_close(&model);
        return 1;
    }

    k3_q8_histogram value_histogram;
    k3_q8_histogram scale_histogram;
    k3_q8_histogram_reset(&value_histogram);
    k3_q8_histogram_reset(&scale_histogram);
    uint32_t matrix_count = 0u;
    uint64_t source_bytes = 0u;
    uint64_t value_bytes = 0u;
    uint64_t scale_bytes = 0u;
    bool ok = true;
    struct timespec started;
    struct timespec finished;
    clock_gettime(CLOCK_MONOTONIC, &started);

    for (size_t tensor_index = 0u;
         ok && tensor_index < model.tensor_count; tensor_index++) {
        const k3_st_tensor *tensor = &model.tensors[tensor_index];
        if (!k3_static_weight_is_text_tensor(tensor) ||
            !k3_static_weight_is_q8_candidate(tensor)) {
            continue;
        }
        if (tensor->shape[0] == 0u || tensor->shape[0] > UINT32_MAX ||
            tensor->shape[1] == 0u || tensor->shape[1] > UINT32_MAX ||
            tensor->shape[1] % K3_Q8_CODEC_BLOCK != 0u ||
            tensor->byte_length != tensor->shape[0] * tensor->shape[1] * 2u) {
            fprintf(stderr, "screen_static_q8: invalid Q8 tensor %s\n", tensor->name);
            ok = false;
            break;
        }
        const uint32_t rows = (uint32_t)tensor->shape[0];
        const uint32_t columns = (uint32_t)tensor->shape[1];
        const size_t row_bytes = (size_t)columns * sizeof(uint16_t);
        uint32_t rows_per_chunk = (uint32_t)(K3_STATIC_Q8_READ_BYTES / row_bytes);
        if (rows_per_chunk == 0u) rows_per_chunk = 1u;
        if (rows_per_chunk > rows) rows_per_chunk = rows;
        const size_t allocation_bytes = (size_t)rows_per_chunk * row_bytes;
        uint16_t *input = (uint16_t *)malloc(allocation_bytes);
        if (!input) {
            fprintf(stderr, "screen_static_q8: allocating source buffer failed\n");
            ok = false;
            break;
        }

        const k3_st_shard *shard = &model.shards[tensor->shard];
        for (uint32_t first_row = 0u; ok && first_row < rows;) {
            uint32_t chunk_rows = rows - first_row;
            if (chunk_rows > rows_per_chunk) chunk_rows = rows_per_chunk;
            const size_t chunk_bytes = (size_t)chunk_rows * row_bytes;
            const uint64_t offset = tensor->physical_offset +
                (uint64_t)first_row * row_bytes;
            if (!pread_full(shard->fd, input, chunk_bytes, offset)) {
                fprintf(stderr,
                        "screen_static_q8: reading %s failed: %s\n",
                        tensor->name, strerror(errno));
                ok = false;
                break;
            }
            const uint64_t blocks =
                (uint64_t)chunk_rows * columns / K3_Q8_CODEC_BLOCK;
            for (uint64_t block = 0u; ok && block < blocks; block++) {
                int8_t quantized[K3_Q8_CODEC_BLOCK];
                float scale;
                if (!k3_q8_quantize_bf16_block(
                        input + block * K3_Q8_CODEC_BLOCK,
                        quantized, &scale) ||
                    !k3_q8_histogram_add(
                        &value_histogram, quantized,
                        K3_Q8_CODEC_BLOCK)) {
                    fprintf(stderr,
                            "screen_static_q8: quantizing %s failed\n",
                            tensor->name);
                    ok = false;
                    break;
                }
                uint32_t scale_bits;
                memcpy(&scale_bits, &scale, sizeof(scale_bits));
                const uint8_t scale_data[4] = {
                    (uint8_t)(scale_bits & 0xffu),
                    (uint8_t)((scale_bits >> 8u) & 0xffu),
                    (uint8_t)((scale_bits >> 16u) & 0xffu),
                    (uint8_t)((scale_bits >> 24u) & 0xffu),
                };
                if (!k3_q8_histogram_add(
                        &scale_histogram,
                        (const int8_t *)scale_data,
                        sizeof(scale_data))) {
                    ok = false;
                    break;
                }
                for (size_t screen = 0u;
                     ok && screen < sizeof(screens) / sizeof(screens[0]);
                     screen++) {
                    ok = tile_screen_add(
                             &screens[screen].values,
                             quantized, sizeof(quantized)) &&
                         tile_screen_add(
                             &screens[screen].scales,
                             scale_data, sizeof(scale_data));
                }
            }
            first_row += chunk_rows;
        }
        free(input);
        for (size_t screen = 0u;
             ok && screen < sizeof(screens) / sizeof(screens[0]); screen++) {
            ok = tile_screen_flush(&screens[screen].values) &&
                 tile_screen_flush(&screens[screen].scales);
        }
        if (!ok) break;
        matrix_count++;
        ok = add_u64(&source_bytes, tensor->byte_length) &&
             add_u64(&value_bytes,
                     (uint64_t)rows * columns) &&
             add_u64(&scale_bytes,
                     (uint64_t)rows * (columns / K3_Q8_CODEC_BLOCK) *
                         sizeof(float));
        if (matrix_count % 64u == 0u) {
            fprintf(stderr,
                    "screen_static_q8: matrices=%u source=%.3f GiB\n",
                    matrix_count, (double)source_bytes / 1073741824.0);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &finished);

    if (ok &&
        (matrix_count != K3_STATIC_Q8_EXPECTED_MATRICES ||
         source_bytes != K3_STATIC_Q8_EXPECTED_SOURCE_BYTES ||
         value_bytes != K3_STATIC_Q8_EXPECTED_VALUE_BYTES ||
         scale_bytes != K3_STATIC_Q8_EXPECTED_SCALE_BYTES)) {
        fprintf(stderr,
                "screen_static_q8: pinned ledger mismatch: "
                "matrices=%u/%u source=%" PRIu64 "/%" PRIu64
                " values=%" PRIu64 "/%" PRIu64
                " scales=%" PRIu64 "/%" PRIu64 "\n",
                matrix_count, K3_STATIC_Q8_EXPECTED_MATRICES,
                source_bytes, K3_STATIC_Q8_EXPECTED_SOURCE_BYTES,
                value_bytes, K3_STATIC_Q8_EXPECTED_VALUE_BYTES,
                scale_bytes, K3_STATIC_Q8_EXPECTED_SCALE_BYTES);
        ok = false;
    }

    char *temporary_path = NULL;
    FILE *output = NULL;
    if (ok) {
        const size_t length = strlen(output_path);
        temporary_path = (char *)malloc(length + 9u);
        if (!temporary_path) {
            ok = false;
        } else {
            memcpy(temporary_path, output_path, length);
            memcpy(temporary_path + length, ".partial", 9u);
            output = fopen(temporary_path, "wx");
            if (!output) {
                fprintf(stderr,
                        "screen_static_q8: opening %s failed: %s\n",
                        temporary_path, strerror(errno));
                ok = false;
            }
        }
    }
    const double seconds = elapsed_seconds(started, finished);
    if (ok) {
        ok = write_result(
            output, model_root, screens,
            sizeof(screens) / sizeof(screens[0]),
            &value_histogram, &scale_histogram,
            matrix_count, source_bytes, value_bytes, scale_bytes,
            seconds);
    }
    if (output) {
        if (ok && (fflush(output) != 0 || fsync(fileno(output)) != 0)) {
            ok = false;
        }
        if (fclose(output) != 0) ok = false;
    }
    if (ok && rename(temporary_path, output_path) != 0) {
        fprintf(stderr,
                "screen_static_q8: publishing %s failed: %s\n",
                output_path, strerror(errno));
        ok = false;
    }
    if (!ok && temporary_path) unlink(temporary_path);

    for (size_t index = 0u; index < sizeof(screens) / sizeof(screens[0]); index++) {
        tile_screen_destroy(&screens[index].values);
        tile_screen_destroy(&screens[index].scales);
    }
    free(temporary_path);
    k3_st_model_close(&model);

    if (!ok) return 1;
    fprintf(stderr,
            "screen_static_q8: PASS matrices=%u q8=%.3f GiB seconds=%.3f\n",
            matrix_count,
            (double)(value_bytes + scale_bytes) / 1073741824.0,
            seconds);
    return 0;
}
