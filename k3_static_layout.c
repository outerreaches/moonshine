#include "k3_static_store.h"

#include <string.h>
bool k3_static_weight_is_text_tensor(const k3_st_tensor *tensor) {
    return tensor && tensor->name &&
           strncmp(tensor->name, "language_model.", 15u) == 0 &&
           strstr(tensor->name, ".block_sparse_moe.experts.") == NULL &&
           strstr(tensor->name, "embed_tokens.weight") == NULL;
}


bool k3_static_weight_is_q8_candidate(const k3_st_tensor *tensor) {
    if (!tensor || tensor->dtype != K3_ST_DTYPE_BF16 ||
        tensor->ndim != 2u || tensor->shape[1] % 128u != 0u ||
        strstr(tensor->name, ".block_sparse_moe.experts.") ||
        strstr(tensor->name, "embed_tokens.weight") ||
        strstr(tensor->name, "lm_head.weight") ||
        strstr(tensor->name, ".block_sparse_moe.gate.") ||
        strstr(tensor->name, "_res_proj.weight") ||
        strstr(tensor->name, "output_attn_res_proj.weight")) {
        return false;
    }
    const size_t length = strlen(tensor->name);
    static const char suffix[] = ".self_attn.kv_b_proj.weight";
    const size_t suffix_length = sizeof(suffix) - 1u;
    return length < suffix_length ||
           strcmp(tensor->name + length - suffix_length, suffix) != 0;
}
