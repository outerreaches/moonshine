#ifndef K3_MZG2_H
#define K3_MZG2_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    K3_MZG2_VERSION = 2,
    K3_MZG2_FILE_HEADER_BYTES = 4096,
    K3_MZG2_ALIGNMENT = 4096,
    K3_MZG2_EXPERT_BYTES = 17547264,
    K3_MZG2_MODEL_TOTAL = 4096,
    K3_MZG2_TILE_RAW = 0x80000000u,
    K3_MZG2_MODEL_MASK = 0xffu,
    K3_MZG2_MODEL_PACKED = 0u,
    K3_MZG2_MODEL_SCALE = 1u,
    K3_MZG2_ERROR_BOUNDS = 1u,
    K3_MZG2_ERROR_STATE = 2u,
    K3_MZG2_ERROR_CHECKSUM = 4u,
    K3_MZG2_ERROR_TERMINAL = 8u,
};

#pragma pack(push, 1)
typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t alignment;
    uint32_t layer;
    uint32_t expert_count;
    uint32_t index_entry_bytes;
    uint32_t model_entry_bytes;
    uint32_t tile_bytes;
    uint32_t flags;
    uint64_t index_offset;
    uint64_t index_bytes;
    uint64_t model_offset;
    uint64_t model_bytes;
    uint64_t data_offset;
    uint64_t max_block_bytes;
    uint8_t source_manifest_sha256[32];
} k3_mzg2_file_header;

typedef struct {
    uint16_t layer;
    uint16_t expert;
    uint32_t block_bytes;
    uint64_t block_offset;
} k3_mzg2_index_entry;

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t layer;
    uint32_t expert;
    uint32_t tile_bytes;
    uint32_t tile_count;
    uint32_t output_bytes;
    uint32_t descriptor_offset;
    uint32_t descriptor_bytes;
    uint32_t payload_offset;
    uint32_t block_bytes;
    uint32_t flags;
    uint32_t model_offset;
    uint32_t model_bytes;
} k3_mzg2_block_header;

typedef struct {
    uint32_t payload_offset;
    uint32_t payload_bytes;
    uint32_t output_offset;
    uint32_t output_bytes;
    uint32_t kind;
    uint32_t reserved;
    uint64_t checksum;
} k3_mzg2_tile_descriptor;
#pragma pack(pop)

typedef struct {
    uint8_t decode[K3_MZG2_MODEL_TOTAL];
    uint8_t values[15];
    uint8_t symbol_count;
    uint8_t padding[16];
    uint32_t frequency[15];
    uint32_t cumulative[15];
} k3_mzg2_model;

typedef struct {
    k3_mzg2_model model[2];
} k3_mzg2_model_entry;

typedef struct k3_mzg2_store k3_mzg2_store;

typedef struct {
    int direct_fd;
    uint64_t offset;
    uint32_t bytes;
} k3_mzg2_span;

/*
 * MOONSHINE_MZG2_EXPERIMENT selects an absolute partial-layer sidecar or a
 * complete sidecar directory. Unset/off leaves the qualified SafeTensor path
 * unchanged. MZG2 and MZG1 selection remain mutually exclusive.
 */
bool k3_mzg2_store_open_optional(k3_mzg2_store **out,
                                 char *error,
                                 size_t error_size);
void k3_mzg2_store_destroy(k3_mzg2_store *store);
uint32_t k3_mzg2_store_max_block_bytes(const k3_mzg2_store *store);
bool k3_mzg2_store_span(const k3_mzg2_store *store,
                        uint32_t layer,
                        uint32_t expert,
                        k3_mzg2_span *span);

/*
 * Validate one host-visible block header and enqueue its bounded wave32 decode.
 * BLOCK_DEVICE aliases BLOCK_HOST through hipHostMallocMapped. DESTINATION and
 * ERROR_DEVICE are device pointers. STREAM is a hipStream_t passed opaquely.
 */
bool k3_mzg2_store_launch(k3_mzg2_store *store,
                          uint32_t layer,
                          uint32_t expert,
                          const void *block_host,
                          const void *block_device,
                          uint32_t block_bytes,
                          void *destination,
                          uint32_t *error_device,
                          void *stream,
                          char *error,
                          size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
