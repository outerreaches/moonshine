#ifndef K3_BUNDLE_H
#define K3_BUNDLE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum {
    K3_BUNDLE_STATIC_TENSORS = 2460,
    K3_BUNDLE_LAYERS = 92,
    K3_BUNDLE_EXPERTS_PER_LAYER = 896,
    K3_BUNDLE_EXPERTS = 82432,
    K3_BUNDLE_TILE_BYTES = 16384,
    K3_BUNDLE_EXPERT_TENSORS = 6,
};

#define K3_BUNDLE_STATIC_PAYLOAD_BYTES UINT64_C(113509540864)
#define K3_BUNDLE_EXPERT_BYTES UINT64_C(17547264)
#define K3_BUNDLE_MODEL_LAYOUT_CRC64 UINT64_C(0xd17f7f2aad23c9c9)

typedef struct {
    char static_path[PATH_MAX];
    char mzg2_path[PATH_MAX];
    uint64_t source_model_layout_crc64;
    uint64_t static_payload_bytes;
    uint64_t static_file_bytes;
    uint32_t static_tensor_count;
} k3_bundle;

bool k3_bundle_detect(const char *root,
                      bool *present,
                      char *error,
                      size_t error_size);

bool k3_bundle_load(k3_bundle *bundle,
                    const char *root,
                    char *error,
                    size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
