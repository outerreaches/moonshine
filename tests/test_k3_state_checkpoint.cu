#include "k3_engine.h"

#include <hip/hip_runtime.h>

#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static bool digest_equal(
        const k3_engine_state_digest *left,
        const k3_engine_state_digest *right) {
    return
        left->kda_state_hash == right->kda_state_hash &&
        left->kda_conv_hash == right->kda_conv_hash &&
        left->mla_cache_hash == right->mla_cache_hash &&
        left->attn_res_hash == right->attn_res_hash &&
        left->token_position == right->token_position;
}

static bool flip_file_byte(const char *path,
                           uint64_t offset) {
    int fd = open(path, O_RDWR | O_CLOEXEC);
    if (fd < 0) return false;
    uint8_t value = 0u;
    bool ok =
        pread(fd, &value, 1u, (off_t)offset) == 1;
    value ^= UINT8_C(0x5a);
    if (ok) {
        ok = pwrite(
            fd, &value, 1u, (off_t)offset) == 1;
    }
    if (ok) ok = fsync(fd) == 0;
    if (close(fd) != 0) ok = false;
    return ok;
}

static void put_u64_le(uint8_t *destination,
                       uint64_t value) {
    for (uint32_t index = 0u; index < 8u; index++) {
        destination[index] =
            (uint8_t)(value >> (index * 8u));
    }
}

static uint64_t header_crc64(
        const uint8_t *data,
        uint64_t bytes) {
    uint64_t table[256];
    const uint64_t polynomial =
        UINT64_C(0x42f0e1eba9ea3693);
    for (uint32_t value = 0u; value < 256u; value++) {
        uint64_t crc = (uint64_t)value << 56u;
        for (uint32_t bit = 0u; bit < 8u; bit++) {
            crc = crc & (UINT64_C(1) << 63u) ?
                (crc << 1u) ^ polynomial : crc << 1u;
        }
        table[value] = crc;
    }
    uint64_t crc = 0u;
    for (uint64_t index = 0u; index < bytes; index++) {
        const uint8_t slot =
            (uint8_t)((crc >> 56u) ^ data[index]);
        crc = table[slot] ^ (crc << 8u);
    }
    return crc;
}

static bool write_file_bytes(const char *path,
                             const void *data,
                             uint64_t bytes,
                             uint64_t offset) {
    int fd = open(path, O_RDWR | O_CLOEXEC);
    if (fd < 0) return false;
    bool ok = pwrite(
        fd, data, (size_t)bytes, (off_t)offset) ==
            (ssize_t)bytes;
    if (ok) ok = fsync(fd) == 0;
    if (close(fd) != 0) ok = false;
    return ok;
}

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] :
        "/srv/modelstore/models/moonshotai__Kimi-K3";
    const char *state_directory =
        argc > 2 ? argv[2] : "/tmp";
    if (argc > 3) {
        fprintf(
            stderr,
            "usage: %s [MODEL_ROOT [STATE_DIRECTORY]]\n",
            argv[0]);
        return 2;
    }

    k3_engine *source = NULL;
    k3_engine *restored = NULL;
    char state_path[PATH_MAX] = { 0 };
    char error[512] = { 0 };

#define CLEANUP()                                                           \
    do {                                                                    \
        k3_engine_destroy(restored);                                        \
        restored = NULL;                                                    \
        k3_engine_destroy(source);                                          \
        source = NULL;                                                      \
        if (state_path[0] != '\0') (void)unlink(state_path);                 \
    } while (0)

#define CHECK(condition, message)                                           \
    do {                                                                    \
        if (!(condition)) {                                                 \
            fprintf(stderr, "FAIL: %s", (message));                         \
            if (error[0] != '\0') fprintf(stderr, ": %s", error);           \
            fprintf(stderr, "\n");                                         \
            CLEANUP();                                                      \
            return 1;                                                       \
        }                                                                   \
    } while (0)

#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t status_ = (call);                                        \
        if (status_ != hipSuccess) {                                        \
            snprintf(                                                       \
                error, sizeof(error), "%s",                                 \
                hipGetErrorString(status_));                                \
            CHECK(false, #call);                                            \
        }                                                                   \
    } while (0)

    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("K3 state checkpoint: SKIP (no ROCm device)\n");
        return 0;
    }
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t properties;
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    printf("K3 state checkpoint on %s (%s)\n",
           properties.name, properties.gcnArchName);

    int path_length = snprintf(
        state_path, sizeof(state_path),
        "%s/k3-state-test-XXXXXX", state_directory);
    CHECK(path_length > 0 &&
              (size_t)path_length < sizeof(state_path),
          "state test path overflow");
    int placeholder = mkstemp(state_path);
    CHECK(placeholder >= 0,
          "state test path creation failed");
    CHECK(close(placeholder) == 0 &&
              unlink(state_path) == 0,
          "state test placeholder cleanup failed");

    k3_engine_stats source_stats;
    CHECK(k3_engine_create(
              &source, root, 8192u, 32u, 16u, true,
              &source_stats, error, sizeof(error)),
          "source engine creation");

    static const uint32_t prompt_tokens[2] = {
        163587u, 2778u,
    };
    uint32_t checkpoint_next = 0u;
    float checkpoint_value = 0.0f;
    for (uint32_t index = 0u; index < 2u; index++) {
        CHECK(k3_engine_forward_token(
                  source, prompt_tokens[index],
                  &checkpoint_next, &checkpoint_value,
                  error, sizeof(error)),
              "source checkpoint prefix");
    }
    k3_engine_state_digest checkpoint_digest;
    CHECK(k3_engine_get_state_digest(
              source, &checkpoint_digest,
              error, sizeof(error)),
          "source checkpoint digest");

    k3_engine_state_file_info exported;
    CHECK(k3_engine_export_state(
              source, state_path, &exported,
              error, sizeof(error)),
          "state export");
    const uint64_t expected_payload_bytes =
        UINT64_C(434110464) +
        UINT64_C(20348928) +
        UINT64_C(24) * UINT64_C(2) *
            UINT64_C(576) * sizeof(uint16_t) +
        UINT64_C(8) * UINT64_C(7168) *
            sizeof(uint16_t);
    CHECK(exported.format_version == 1u &&
              exported.token_position == 2u &&
              exported.q8_projections &&
              exported.payload_bytes ==
                  expected_payload_bytes &&
              exported.file_bytes ==
                  exported.payload_bytes + 256u &&
              exported.model_layout_crc64 != 0u &&
              exported.payload_crc64 != 0u,
          "state export ledger");
    struct stat state_stat;
    CHECK(stat(state_path, &state_stat) == 0 &&
              (uint64_t)state_stat.st_size ==
                  exported.file_bytes,
          "state export file size");
    k3_engine_state_digest post_export_digest;
    CHECK(k3_engine_get_state_digest(
              source, &post_export_digest,
              error, sizeof(error)) &&
              digest_equal(
                  &checkpoint_digest,
                  &post_export_digest),
          "state export changed live state");

    enum { CONTINUATION_TOKENS = 3 };
    uint32_t expected_tokens[CONTINUATION_TOKENS] = { 0u };
    float expected_values[CONTINUATION_TOKENS] = { 0.0f };
    uint32_t continuation_input = checkpoint_next;
    for (uint32_t index = 0u;
         index < CONTINUATION_TOKENS; index++) {
        CHECK(k3_engine_forward_token(
                  source, continuation_input,
                  &expected_tokens[index],
                  &expected_values[index],
                  error, sizeof(error)),
              "source uninterrupted continuation");
        continuation_input = expected_tokens[index];
    }
    k3_engine_state_digest expected_final_digest;
    CHECK(k3_engine_get_state_digest(
              source, &expected_final_digest,
              error, sizeof(error)),
          "source final digest");
    k3_engine_destroy(source);
    source = NULL;
    HIP_CHECK(hipDeviceSynchronize());

    k3_engine_stats restored_stats;
    CHECK(k3_engine_create(
              &restored, root, 8192u, 32u, 16u, true,
              &restored_stats, error, sizeof(error)),
          "restored engine creation");
    k3_engine_state_digest blank_digest;
    CHECK(k3_engine_get_state_digest(
              restored, &blank_digest,
              error, sizeof(error)),
          "blank restored-engine digest");

    const uint64_t header_probe = 40u;
    CHECK(flip_file_byte(state_path, header_probe),
          "header corruption write");
    error[0] = '\0';
    CHECK(!k3_engine_import_state(
              restored, state_path, NULL,
              error, sizeof(error)) &&
              strstr(error, "header CRC64") != NULL,
          "corrupt header was not rejected");
    CHECK(flip_file_byte(state_path, header_probe),
          "header corruption restore");

    /*
     * Format-v1 offsets are duplicated here intentionally: this produces a
     * checksum-valid but stale model identity instead of generic corruption.
     */
    uint8_t original_header[256];
    int header_fd =
        open(state_path, O_RDONLY | O_CLOEXEC);
    CHECK(header_fd >= 0 &&
              pread(
                  header_fd, original_header,
                  sizeof(original_header), 0) ==
                  (ssize_t)sizeof(original_header) &&
              close(header_fd) == 0,
          "stale-identity header read");
    uint8_t stale_header[256];
    memcpy(
        stale_header, original_header,
        sizeof(stale_header));
    stale_header[40] ^= UINT8_C(0x01);
    memset(stale_header + 104, 0, sizeof(uint64_t));
    put_u64_le(
        stale_header + 104,
        header_crc64(
            stale_header, sizeof(stale_header)));
    CHECK(write_file_bytes(
              state_path, stale_header,
              sizeof(stale_header), 0u),
          "stale-identity header write");
    error[0] = '\0';
    CHECK(!k3_engine_import_state(
              restored, state_path, NULL,
              error, sizeof(error)) &&
              strstr(error, "identity mismatch") != NULL,
          "stale model identity was not rejected");
    CHECK(write_file_bytes(
              state_path, original_header,
              sizeof(original_header), 0u),
          "stale-identity header restore");

    const uint64_t payload_probe =
        exported.file_bytes - exported.payload_bytes +
        exported.payload_bytes / 2u;
    CHECK(flip_file_byte(state_path, payload_probe),
          "payload corruption write");
    error[0] = '\0';
    CHECK(!k3_engine_import_state(
              restored, state_path, NULL,
              error, sizeof(error)) &&
              strstr(error, "payload CRC64") != NULL,
          "corrupt payload was not rejected");
    CHECK(flip_file_byte(state_path, payload_probe),
          "payload corruption restore");

    int truncate_fd =
        open(state_path, O_RDWR | O_CLOEXEC);
    CHECK(truncate_fd >= 0,
          "truncate test open");
    uint8_t final_byte = 0u;
    CHECK(pread(
              truncate_fd, &final_byte, 1u,
              (off_t)(exported.file_bytes - 1u)) == 1 &&
              ftruncate(
                  truncate_fd,
                  (off_t)(exported.file_bytes - 1u)) == 0,
          "truncate test mutation");
    error[0] = '\0';
    CHECK(!k3_engine_import_state(
              restored, state_path, NULL,
              error, sizeof(error)),
          "truncated payload was not rejected");
    CHECK(ftruncate(
              truncate_fd,
              (off_t)exported.file_bytes) == 0 &&
              pwrite(
                  truncate_fd, &final_byte, 1u,
                  (off_t)(exported.file_bytes - 1u)) == 1 &&
              fsync(truncate_fd) == 0 &&
              close(truncate_fd) == 0,
          "truncate test restoration");

    k3_engine_state_digest post_rejection_digest;
    CHECK(k3_engine_get_state_digest(
              restored, &post_rejection_digest,
              error, sizeof(error)) &&
              digest_equal(
                  &blank_digest,
                  &post_rejection_digest),
          "rejected file changed engine state");

    k3_engine_state_file_info imported;
    error[0] = '\0';
    CHECK(k3_engine_import_state(
              restored, state_path, &imported,
              error, sizeof(error)),
          "valid state import");
    CHECK(imported.format_version ==
              exported.format_version &&
              imported.token_position ==
                  exported.token_position &&
              imported.model_layout_crc64 ==
                  exported.model_layout_crc64 &&
              imported.payload_bytes ==
                  exported.payload_bytes &&
              imported.file_bytes ==
                  exported.file_bytes &&
              imported.payload_crc64 ==
                  exported.payload_crc64 &&
              imported.q8_projections ==
                  exported.q8_projections,
          "imported state metadata");
    k3_engine_state_digest imported_digest;
    CHECK(k3_engine_get_state_digest(
              restored, &imported_digest,
              error, sizeof(error)) &&
              digest_equal(
                  &checkpoint_digest,
                  &imported_digest),
          "imported causal-state digest");
    k3_engine_cache_stats imported_cache;
    k3_engine_get_cache_stats(
        restored, &imported_cache);
    CHECK(imported_cache.accesses == 0u &&
              imported_cache.hits == 0u,
          "semantic import unexpectedly restored expert cache");

    uint32_t actual_tokens[CONTINUATION_TOKENS] = { 0u };
    float actual_values[CONTINUATION_TOKENS] = { 0.0f };
    continuation_input = checkpoint_next;
    for (uint32_t index = 0u;
         index < CONTINUATION_TOKENS; index++) {
        CHECK(k3_engine_forward_token(
                  restored, continuation_input,
                  &actual_tokens[index],
                  &actual_values[index],
                  error, sizeof(error)),
              "restored continuation");
        CHECK(actual_tokens[index] ==
                  expected_tokens[index] &&
                  memcmp(
                      &actual_values[index],
                      &expected_values[index],
                      sizeof(actual_values[index])) == 0,
              "restored continuation output drift");
        continuation_input = actual_tokens[index];
    }
    k3_engine_state_digest actual_final_digest;
    CHECK(k3_engine_get_state_digest(
              restored, &actual_final_digest,
              error, sizeof(error)) &&
              digest_equal(
                  &expected_final_digest,
                  &actual_final_digest),
          "restored continuation state drift");

    printf("K3 state checkpoint: PASS\n");
    printf("  format=%u position=%u file=%.3f MiB "
           "payload_crc=%016" PRIx64
           " model_crc=%016" PRIx64 "\n",
           exported.format_version,
           exported.token_position,
           exported.file_bytes / 1048576.0,
           exported.payload_crc64,
           exported.model_layout_crc64);
    printf("  export=%.3f s import=%.3f s "
           "(import validates before upload)\n",
           exported.wall_seconds,
           imported.wall_seconds);
    printf("  rejected: header-bitflip, stale-model, "
           "payload-bitflip, truncated-file\n");
    printf("  imported state: kda=%016" PRIx64
           " conv=%016" PRIx64 " mla=%016" PRIx64
           " attnres=%016" PRIx64 "\n",
           imported_digest.kda_state_hash,
           imported_digest.kda_conv_hash,
           imported_digest.mla_cache_hash,
           imported_digest.attn_res_hash);
    printf("  exact continuation IDs:");
    for (uint32_t index = 0u;
         index < CONTINUATION_TOKENS; index++) {
        printf(" %u", actual_tokens[index]);
    }
    printf("\n");

    CLEANUP();
#undef HIP_CHECK
#undef CHECK
#undef CLEANUP
    return 0;
}
