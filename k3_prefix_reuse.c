#include "k3_prefix_reuse.h"

#include <string.h>

enum {
    /* The turn executor rejects rendered prompts shorter than two tokens, so
     * an admitted reuse must leave at least that much suffix to evaluate. */
    K3_PREFIX_REUSE_MIN_SUFFIX = 2u,
};

bool k3_prefix_reuse_admits(
        const uint32_t *retained,
        size_t          retained_count,
        const uint32_t *candidate,
        size_t          candidate_count) {
    if (retained == NULL || candidate == NULL) {
        return false;
    }
    if (retained_count == 0u) {
        return false;
    }
    if (retained_count > candidate_count ||
        candidate_count - retained_count <
            K3_PREFIX_REUSE_MIN_SUFFIX) {
        return false;
    }
    if (retained_count >
        SIZE_MAX / sizeof(*retained)) {
        return false;
    }
    return memcmp(
        candidate, retained,
        retained_count * sizeof(*retained)) == 0;
}
