#define _POSIX_C_SOURCE 200809L

#include "k31_model.h"

#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(condition, message) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "FAIL: %s\n", (message)); \
            return 1; \
        } \
    } while (0)

#define FIXTURE_CAPACITY 16384u

typedef struct {
    char data[FIXTURE_CAPACITY];
    size_t size;
} fixture_buffer;

static bool fixture_append(fixture_buffer *fixture, const char *fmt, ...) {
    if (fixture->size >= sizeof(fixture->data)) {
        return false;
    }
    va_list ap;
    va_start(ap, fmt);
    const int written = vsnprintf(fixture->data + fixture->size,
                                  sizeof(fixture->data) - fixture->size,
                                  fmt, ap);
    va_end(ap);
    if (written < 0 ||
        (size_t)written >= sizeof(fixture->data) - fixture->size) {
        return false;
    }
    fixture->size += (size_t)written;
    return true;
}

static bool fixture_layers(fixture_buffer *fixture, bool dsa) {
    bool first = true;
    for (uint32_t layer = 1u; layer <= K31_MODEL_LAYER_COUNT; layer++) {
        const bool selected = (layer % 4u == 0u || layer == 93u) == dsa;
        if (selected &&
            !fixture_append(fixture, "%s%u", first ? "" : ",", layer)) {
            return false;
        }
        if (selected) {
            first = false;
        }
    }
    return true;
}

static char *valid_config(void) {
    fixture_buffer fixture;
    memset(&fixture, 0, sizeof(fixture));
    bool ok = fixture_append(
        &fixture,
        "{\"architectures\":[\"%s\"],\"model_type\":\"%s\","
        "\"text_config\":{\"num_hidden_layers\":93,"
        "\"linear_attn_config\":{\"full_attn_layers\":[",
        K31_MODEL_ARCHITECTURE, K31_MODEL_TYPE);
    ok = ok && fixture_layers(&fixture, true) && fixture_append(
        &fixture, "],\"kda_layers\":[");
    ok = ok && fixture_layers(&fixture, false) && fixture_append(
        &fixture,
        "]},\"dsa_config\":{\"dsa_layers\":[");
    ok = ok && fixture_layers(&fixture, true) && fixture_append(
        &fixture,
        "],\"index_topk\":2048,\"index_head_dim\":128,"
        "\"index_num_heads\":32,\"index_rope_dim\":0,"
        "\"index_k_norm_type\":\"layernorm\",\"index_topk_freq\":1,"
        "\"index_skip_topk_offset\":1}}}");
    if (!ok) {
        return NULL;
    }
    char *copy = (char *)malloc(fixture.size + 1u);
    if (copy != NULL) {
        memcpy(copy, fixture.data, fixture.size + 1u);
    }
    return copy;
}

static char *replace_once(const char *source, const char *from,
                          const char *to) {
    const char *match = strstr(source, from);
    if (match == NULL || strstr(match + strlen(from), from) != NULL) {
        return NULL;
    }
    const size_t prefix = (size_t)(match - source);
    const size_t from_size = strlen(from);
    const size_t to_size = strlen(to);
    const size_t suffix_size = strlen(match + from_size);
    char *edited = (char *)malloc(prefix + to_size + suffix_size + 1u);
    if (edited == NULL) {
        return NULL;
    }
    memcpy(edited, source, prefix);
    memcpy(edited + prefix, to, to_size);
    memcpy(edited + prefix + to_size, match + from_size, suffix_size + 1u);
    return edited;
}

static int expect_rejection(const char *valid, const char *from,
                            const char *to, const char *label) {
    char *edited = replace_once(valid, from, to);
    CHECK(edited != NULL, "construct rejection fixture");
    k31_model_profile profile;
    memset(&profile, 0xa5, sizeof(profile));
    const k31_model_profile before = profile;
    char error[256];
    const bool parsed = k31_model_profile_parse(
        &profile, edited, strlen(edited), error, sizeof(error));
    free(edited);
    CHECK(!parsed, label);
    CHECK(memcmp(&profile, &before, sizeof(profile)) == 0,
          "rejection changed output profile");
    CHECK(error[0] != '\0', "rejection omitted diagnostic");
    return 0;
}

static int test_valid_config(const char *valid) {
    const size_t size = strlen(valid);
    char *unterminated = (char *)malloc(size);
    CHECK(unterminated != NULL, "allocate unterminated fixture");
    memcpy(unterminated, valid, size);
    k31_model_profile profile;
    memset(&profile, 0, sizeof(profile));
    CHECK(k31_model_profile_parse(&profile, unterminated, size, NULL, 0u),
          "parse valid in-memory config");
    free(unterminated);

    CHECK(strcmp(profile.model_id, K31_MODEL_ID) == 0 &&
              strcmp(profile.model_revision, K31_MODEL_REVISION) == 0 &&
              strcmp(K31_MODEL_ID, "moonshotai/Kimi-K3.1-Pro") == 0 &&
              strcmp(K31_MODEL_REVISION,
                     "6d0a0202956796543364d666ab19c6ad1a494ee1") == 0,
          "upstream model identity");
    CHECK(profile.layer_count == 93u &&
              profile.kda_layer_count == 69u &&
              profile.dsa_layer_count == 24u,
          "layer counts");
    uint32_t kda_count = 0u;
    uint32_t dsa_count = 0u;
    for (uint32_t layer = 1u; layer <= K31_MODEL_LAYER_COUNT; layer++) {
        const k31_attention_type expected =
            layer % 4u == 0u || layer == 93u ?
                K31_ATTENTION_DSA : K31_ATTENTION_KDA;
        CHECK(profile.attention_by_layer[layer - 1u] == expected,
              "validated attention map");
        if (expected == K31_ATTENTION_DSA) {
            dsa_count++;
        } else {
            kda_count++;
        }
    }
    CHECK(kda_count == 69u && dsa_count == 24u &&
              profile.attention_by_layer[0] == K31_ATTENTION_KDA &&
              profile.attention_by_layer[3] == K31_ATTENTION_DSA &&
              profile.attention_by_layer[90] == K31_ATTENTION_KDA &&
              profile.attention_by_layer[91] == K31_ATTENTION_DSA &&
              profile.attention_by_layer[92] == K31_ATTENTION_DSA,
          "attention map boundaries");
    CHECK(profile.dsa.index_topk == 2048u &&
              profile.dsa.index_head_dim == 128u &&
              profile.dsa.index_num_heads == 32u &&
              profile.dsa.index_rope_dim == 0u &&
              profile.dsa.index_k_norm_type == K31_INDEX_K_NORM_LAYERNORM &&
              profile.dsa.index_topk_freq == 1u &&
              profile.dsa.index_skip_topk_offset == 1u,
          "DSA geometry");
    return 0;
}

static int test_unknown_fields(const char *valid) {
    char *edited = replace_once(
        valid, "{\"architectures\"",
        "{\"future_config\":{\"nested\":[true,null,3.5]},\"architectures\"");
    CHECK(edited != NULL, "construct unknown-field fixture");
    k31_model_profile profile;
    CHECK(k31_model_profile_parse(&profile, edited, strlen(edited), NULL, 0u),
          "unknown non-DSA field was rejected");
    CHECK(profile.layer_count == 93u &&
              profile.attention_by_layer[92] == K31_ATTENTION_DSA,
          "unknown field changed parsed profile");
    free(edited);
    return 0;
}

static int test_identity_and_layer_rejections(const char *valid) {
    static const struct {
        const char *from;
        const char *to;
        const char *label;
    } cases[] = {
        { "\"architectures\":[\"KimiK31ForConditionalGeneration\"],",
          "", "missing architecture" },
        { "\"architectures\":[\"KimiK31ForConditionalGeneration\"]",
          "\"architectures\":[\"Other\"]", "wrong architecture" },
        { "\"architectures\":[\"KimiK31ForConditionalGeneration\"]",
          "\"architectures\":[\"KimiK31ForConditionalGeneration\",\"Other\"]",
          "multiple architectures" },
        { "\"architectures\":[\"KimiK31ForConditionalGeneration\"]",
          "\"architectures\":[\"KimiK31ForConditionalGeneration\"],"
          "\"architectures\":[\"KimiK31ForConditionalGeneration\"]",
          "duplicate architecture" },
        { "\"model_type\":\"kimi_k31\",", "", "missing model type" },
        { "\"model_type\":\"kimi_k31\"",
          "\"model_type\":\"kimi_k3\"", "wrong model type" },
        { "\"model_type\":\"kimi_k31\"",
          "\"model_type\":\"kimi_k31\",\"model_type\":\"kimi_k31\"",
          "duplicate model type" },
        { "\"text_config\":{", "\"text_config\":[],", "non-object text config" },
        { "\"text_config\":{", "\"ignored_text_config\":{",
          "missing text config" },
        { "\"num_hidden_layers\":93,",
          "", "missing layer count" },
        { "\"num_hidden_layers\":93",
          "\"num_hidden_layers\":92", "wrong layer count" },
        { "\"num_hidden_layers\":93",
          "\"num_hidden_layers\":93,\"num_hidden_layers\":93",
          "duplicate layer count" },
        { "\"linear_attn_config\":{", "\"ignored_linear_config\":{",
          "missing linear attention config" },
        { "\"dsa_config\":{", "\"ignored_dsa_config\":{",
          "missing DSA config" },
    };
    for (size_t i = 0u; i < sizeof(cases) / sizeof(cases[0]); i++) {
        CHECK(expect_rejection(valid, cases[i].from, cases[i].to,
                               cases[i].label) == 0,
              "identity/layer rejection case");
    }
    return 0;
}

static int test_dsa_rejections(const char *valid) {
    static const struct {
        const char *from;
        const char *to;
        const char *label;
    } cases[] = {
        { "\"dsa_layers\":", "\"ignored_layers\":", "missing DSA layers" },
        { "\"dsa_layers\":", "\"dsa_layers\":[],\"dsa_layers\":",
          "duplicate DSA layer property" },
        { "\"index_topk\":2048,", "", "missing index_topk" },
        { "\"index_topk\":2048", "\"index_topk\":2047", "wrong index_topk" },
        { "\"index_topk\":2048", "\"index_topk\":\"2048\"",
          "non-integer index_topk" },
        { "\"index_head_dim\":128,", "", "missing index_head_dim" },
        { "\"index_head_dim\":128", "\"index_head_dim\":127",
          "wrong index_head_dim" },
        { "\"index_num_heads\":32,", "", "missing index_num_heads" },
        { "\"index_num_heads\":32", "\"index_num_heads\":31",
          "wrong index_num_heads" },
        { "\"index_rope_dim\":0,", "", "missing index_rope_dim" },
        { "\"index_rope_dim\":0", "\"index_rope_dim\":1",
          "wrong index_rope_dim" },
        { "\"index_k_norm_type\":\"layernorm\",", "",
          "missing index_k_norm_type" },
        { "\"index_k_norm_type\":\"layernorm\"",
          "\"index_k_norm_type\":\"rmsnorm\"", "wrong index_k_norm_type" },
        { "\"index_topk_freq\":1,", "", "missing index_topk_freq" },
        { "\"index_topk_freq\":1", "\"index_topk_freq\":2",
          "wrong index_topk_freq" },
        { ",\"index_skip_topk_offset\":1", "",
          "missing index_skip_topk_offset" },
        { "\"index_skip_topk_offset\":1", "\"index_skip_topk_offset\":0",
          "wrong index_skip_topk_offset" },
        { "\"index_skip_topk_offset\":1",
          "\"index_skip_topk_offset\":1,\"index_skip_topk_offset\":1",
          "duplicate DSA geometry field" },
    };
    for (size_t i = 0u; i < sizeof(cases) / sizeof(cases[0]); i++) {
        CHECK(expect_rejection(valid, cases[i].from, cases[i].to,
                               cases[i].label) == 0,
              "DSA rejection case");
    }
    return 0;
}

static int test_attention_rejections(const char *valid) {
    static const struct {
        const char *from;
        const char *to;
        const char *label;
    } cases[] = {
        { "\"full_attn_layers\":", "\"ignored_full_layers\":",
          "missing full attention layers" },
        { "\"kda_layers\":", "\"ignored_kda_layers\":",
          "missing KDA layers" },
        { "\"dsa_layers\":[4,8", "\"dsa_layers\":[4,4",
          "duplicate DSA placement" },
        { "\"full_attn_layers\":[4,8", "\"full_attn_layers\":[4,4",
          "duplicate full attention placement" },
        { "\"kda_layers\":[1,2,3", "\"kda_layers\":[1,2,2",
          "duplicate KDA placement" },
        { "\"dsa_layers\":[4,", "\"dsa_layers\":[1,",
          "DSA/KDA conflict" },
        { "\"full_attn_layers\":[4,", "\"full_attn_layers\":[1,",
          "full/DSA conflict" },
        { "\"kda_layers\":[1,", "\"kda_layers\":[4,",
          "KDA/DSA conflict" },
        { "\"dsa_layers\":[4,", "\"dsa_layers\":[0,",
          "zero DSA placement" },
        { "\"full_attn_layers\":[4,", "\"full_attn_layers\":[94,",
          "out-of-range full attention placement" },
        { ",90,91]", ",90,94]", "out-of-range KDA placement" },
        { ",93],\"index_topk\"", "],\"index_topk\"",
          "missing final DSA placement" },
        { ",93],\"kda_layers\"", "],\"kda_layers\"",
          "missing final full attention placement" },
        { ",90,91]}", ",90]}", "missing final KDA placement" },
        { "\"kda_layers\":[1,", "\"kda_layers\":[\"1\",",
          "non-numeric attention placement" },
    };
    for (size_t i = 0u; i < sizeof(cases) / sizeof(cases[0]); i++) {
        CHECK(expect_rejection(valid, cases[i].from, cases[i].to,
                               cases[i].label) == 0,
              "attention rejection case");
    }
    return 0;
}

static int test_file_loading(const char *valid) {
    char path[] = "/tmp/k31-model-profile-XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0, "create temporary config");
    FILE *file = fdopen(fd, "wb");
    CHECK(file != NULL, "open temporary config");
    CHECK(fwrite(valid, 1u, strlen(valid), file) == strlen(valid) &&
              fclose(file) == 0,
          "write temporary config");

    k31_model_profile profile;
    memset(&profile, 0, sizeof(profile));
    char error[256];
    CHECK(k31_model_profile_load(&profile, path, error, sizeof(error)),
          "load valid config file");
    CHECK(profile.layer_count == 93u &&
              profile.dsa.index_topk == 2048u &&
              profile.attention_by_layer[92] == K31_ATTENTION_DSA,
          "file-loaded profile");

    file = fopen(path, "wb");
    CHECK(file != NULL, "reopen temporary config");
    CHECK(fwrite("{}", 1u, 2u, file) == 2u && fclose(file) == 0,
          "write invalid config file");
    const k31_model_profile before = profile;
    CHECK(!k31_model_profile_load(&profile, path, error, sizeof(error)) &&
              memcmp(&profile, &before, sizeof(profile)) == 0,
          "invalid file changed output profile");
    CHECK(unlink(path) == 0, "remove temporary config");
    CHECK(!k31_model_profile_load(&profile, path, error, sizeof(error)) &&
              memcmp(&profile, &before, sizeof(profile)) == 0,
          "missing config file was accepted");
    return 0;
}

static int test_invalid_inputs(const char *valid) {
    k31_model_profile profile;
    memset(&profile, 0xa5, sizeof(profile));
    const k31_model_profile before = profile;
    CHECK(!k31_model_profile_parse(NULL, valid, strlen(valid), NULL, 0u) &&
              !k31_model_profile_parse(&profile, NULL, 0u, NULL, 0u) &&
              !k31_model_profile_load(NULL, "unused", NULL, 0u) &&
              !k31_model_profile_load(&profile, NULL, NULL, 0u),
          "invalid API arguments were accepted");
    CHECK(!k31_model_profile_parse(&profile, "", 0u, NULL, 0u) &&
              !k31_model_profile_parse(&profile, "[]", 2u, NULL, 0u) &&
              !k31_model_profile_parse(&profile, "{", 1u, NULL, 0u),
          "malformed or non-object JSON was accepted");
    CHECK(memcmp(&profile, &before, sizeof(profile)) == 0,
          "invalid input changed output profile");
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 2) {
        k31_model_profile profile;
        char error[512];
        if (!k31_model_profile_load(&profile, argv[1], error, sizeof(error))) {
            fprintf(stderr, "FAIL: %s\n", error);
            return 1;
        }
        printf("K3.1 model profile: PASS %s@%s layers=%u dsa=%u topk=%u\n",
               profile.model_id, profile.model_revision,
               profile.layer_count, profile.dsa_layer_count,
               profile.dsa.index_topk);
        return 0;
    }
    if (argc != 1) {
        fprintf(stderr, "usage: %s [CONFIG_JSON]\n", argv[0]);
        return 2;
    }
    char *valid = valid_config();
    CHECK(valid != NULL, "build synthetic valid config");
    CHECK(test_valid_config(valid) == 0, "valid config behavior");
    CHECK(test_unknown_fields(valid) == 0, "unknown field behavior");
    CHECK(test_identity_and_layer_rejections(valid) == 0,
          "identity/layer rejection behavior");
    CHECK(test_dsa_rejections(valid) == 0, "DSA rejection behavior");
    CHECK(test_attention_rejections(valid) == 0,
          "attention rejection behavior");
    CHECK(test_file_loading(valid) == 0, "file loading behavior");
    CHECK(test_invalid_inputs(valid) == 0, "invalid input behavior");
    free(valid);
    printf("K3.1 model profile parser: PASS\n");
    return 0;
}
