#ifndef K3_MZG_H
#define K3_MZG_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct k3_mzg_store k3_mzg_store;

typedef struct {
    int       direct_fd;
    uint64_t  offset;
    uint32_t  bytes;
} k3_mzg_span;

/*
 * Open the expert store when selected. MOONSHINE_EXPERT_STORE accepts `auto`
 * (MODEL_ROOT/expert-store-mzg1), `off`, or an absolute store path. Unset is
 * the qualified source-SafeTensors default and leaves *OUT null. A selected
 * but incomplete/corrupt store fails closed.
 */
bool k3_mzg_store_open_optional(k3_mzg_store **out,
                                const char    *model_root,
                                uint32_t       layers,
                                uint32_t       experts,
                                char          *error,
                                size_t         error_size);

void k3_mzg_store_destroy(k3_mzg_store *store);

uint32_t k3_mzg_store_max_block_bytes(const k3_mzg_store *store);

bool k3_mzg_store_span(const k3_mzg_store *store,
                       uint32_t            layer,
                       uint32_t            expert,
                       k3_mzg_span         *span);

/*
 * Decode one aligned block into the native 17,547,264-byte expert layout.
 * DESTINATION must be host-writable; six persistent workers decode twelve
 * two-stripe frames into disjoint ranges and join before return.
 */
bool k3_mzg_store_decode(k3_mzg_store *store,
                         uint32_t      layer,
                         uint32_t      expert,
                         const void   *block,
                         uint32_t      block_bytes,
                         void         *destination,
                         char         *error,
                         size_t        error_size);

#ifdef __cplusplus
}
#endif

#endif
