#include "../glm53_manifest.h"
#include "../glm53_process_memory.h"
#include "../glm53_static_layout.h"
#include "../glm53_static_loader.h"
#include "../glm53_static_bindings.h"
#include "../glm53_phase5c.h"
#include "../glm53_weights.h"
#include "../glm53_architecture.h"

#include <hip/hip_runtime.h>

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <vector>

#define GIB (UINT64_C(1024) * UINT64_C(1024) * UINT64_C(1024))
#define EXPECTED_ENTRIES ((size_t)37713)
#define EXPECTED_LOGICAL UINT64_C(15300311288)
#define EXPECTED_PADDED UINT64_C(15300353024)
#define EXPECTED_FULL_CRC64 UINT64_C(0xe29a16329b0270f3)
#define EXPECTED_SMOKE_CRC64 UINT64_C(0xbae1a998574b9223)
#define EXPECTED_DIAGNOSTIC_CRC64 UINT64_C(0x46cc7f15c53bb356)
#define SAMPLE_BYTES ((size_t)65536)

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
    goto done; \
} } while (0)
#define HIP_CHECK(c) do { hipError_t e_ = (c); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP FAIL %s:%d: %s: %s\n", __FILE__, __LINE__, \
            #c, hipGetErrorString(e_)); goto done; \
} } while (0)

static double monotonic_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static uint64_t align256(uint64_t value) {
    return (value + UINT64_C(255)) & ~UINT64_C(255);
}

static uint64_t crc_update(uint64_t crc, const void *source, size_t bytes) {
    const unsigned char *data = (const unsigned char *)source;
    size_t i;
    unsigned bit;
    for (i = 0u; i < bytes; ++i) {
        crc ^= (uint64_t)data[i] << 56u;
        for (bit = 0u; bit < 8u; ++bit)
            crc = (crc & (UINT64_C(1) << 63u)) ?
                (crc << 1u) ^ UINT64_C(0x42f0e1eba9ea3693) : crc << 1u;
    }
    return crc;
}

static bool exact_pread(int fd, void *destination, size_t bytes,
                        uint64_t offset) {
    unsigned char *out = (unsigned char *)destination;
    size_t done = 0u;
    while (done < bytes) {
        ssize_t got = pread(fd, out + done, bytes - done,
                            (off_t)(offset + (uint64_t)done));
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return false;
        done += (size_t)got;
    }
    return true;
}

static bool memory_clean(const glm53_process_memory *m) {
    return m->vm_swap_bytes == 0u && m->smaps_swap_bytes == 0u &&
           m->smaps_swap_pss_bytes == 0u && m->model_vma_count == 0u &&
           m->model_vma_bytes == 0u && m->largest_model_vma_bytes == 0u;
}

static void print_memory(const char *label, const glm53_process_memory *m) {
    printf("memory[%s]: MemAvailable=%" PRIu64 " MiB SwapFree=%" PRIu64
           " MiB VmSwap=%" PRIu64 " smaps_swap=%" PRIu64
           " smaps_swap_pss=%" PRIu64 " model_vmas=%" PRIu64
           " model_vma_bytes=%" PRIu64 "\n",
           label, m->mem_available_bytes >> 20u, m->swap_free_bytes >> 20u,
           m->vm_swap_bytes, m->smaps_swap_bytes, m->smaps_swap_pss_bytes,
           m->model_vma_count, m->model_vma_bytes);
}

static bool sample_memory(const char *label, const char *prefix,
                          glm53_process_memory *m) {
    glm53_process_memory_status status = glm53_process_memory_sample(prefix, m);
    if (status != GLM53_PROCESS_MEMORY_OK) {
        fprintf(stderr, "FAIL memory[%s]: %s\n", label,
                glm53_process_memory_status_string(status));
        return false;
    }
    print_memory(label, m);
    if (!memory_clean(m)) {
        fprintf(stderr, "FAIL memory[%s]: process swap or model VMA present\n",
                label);
        return false;
    }
    return true;
}

/* This deliberately uses pread, never mmap, and compares every smoke byte. */
static bool compare_whole_entry(const k3_st_model *model,
                                const glm53_static_store *store,
                                const glm53_static_layout_entry *layout_entry,
                                uint64_t *crc_out) {
    const glm53_static_runtime_entry *entry;
    unsigned char *source = NULL, *device = NULL;
    uint64_t done = 0u, crc = 0u;
    bool ok = false;
    size_t capacity = (size_t)(8u * 1024u * 1024u);
    entry = glm53_static_store_find(store, layout_entry->tensor->name);
    if (entry == NULL || entry->logical_bytes != layout_entry->logical_bytes ||
        entry->device_offset != layout_entry->device_offset ||
        layout_entry->source_shard >= model->shard_count) return false;
    source = (unsigned char *)malloc(capacity);
    device = (unsigned char *)malloc(capacity);
    if (source == NULL || device == NULL) goto finish;
    while (done < entry->logical_bytes) {
        size_t amount = (size_t)((entry->logical_bytes - done > capacity) ?
                                capacity : entry->logical_bytes - done);
        if (!exact_pread(model->shards[layout_entry->source_shard].fd, source,
                         amount, layout_entry->source_physical_offset + done))
            goto finish;
        if (hipMemcpy(device,
                (unsigned char *)glm53_static_store_device_pointer(store, entry) + done,
                amount, hipMemcpyDeviceToHost) != hipSuccess) goto finish;
        if (memcmp(source, device, amount) != 0) goto finish;
        crc = crc_update(crc, source, amount);
        done += amount;
    }
    if (crc != entry->crc64_ecma) goto finish;
    *crc_out = crc;
    ok = true;
finish:
    free(device);
    free(source);
    return ok;
}

/* Full mode samples one exact span in each of first/middle/last entries. */
static bool compare_full_samples(const k3_st_model *model,
                                 const glm53_static_layout *layout,
                                 const glm53_static_store *store,
                                 uint64_t *crc_out) {
    const size_t indices[3] = {0u, layout->entry_count / 2u,
                               layout->entry_count - 1u};
    unsigned char source[SAMPLE_BYTES], device[SAMPLE_BYTES];
    uint64_t crc = 0u;
    size_t which;
    for (which = 0u; which < 3u; ++which) {
        const glm53_static_layout_entry *le = &layout->entries[indices[which]];
        const glm53_static_runtime_entry *re =
            glm53_static_store_find(store, le->tensor->name);
        size_t amount = le->logical_bytes < SAMPLE_BYTES ?
                        (size_t)le->logical_bytes : SAMPLE_BYTES;
        uint64_t within;
        if (amount == 0u || re == NULL || le->source_shard >= model->shard_count)
            return false;
        if (which == 0u) within = 0u;
        else if (which == 1u) within = (le->logical_bytes - amount) / 2u;
        else within = le->logical_bytes - amount;
        if (!exact_pread(model->shards[le->source_shard].fd, source, amount,
                         le->source_physical_offset + within)) return false;
        if (hipMemcpy(device,
                (unsigned char *)glm53_static_store_device_pointer(store, re) + within,
                amount, hipMemcpyDeviceToHost) != hipSuccess) return false;
        if (memcmp(source, device, amount) != 0) return false;
        crc = crc_update(crc, source, amount);
        printf("sample[%zu]: entry=%zu name=%s within=%" PRIu64
               " bytes=%zu crc64=%016" PRIx64 "\n", which, indices[which],
               le->tensor->name, within, amount,
               crc_update(0u, source, amount));
    }
    *crc_out = crc;
    return true;
}

int main(int argc, char **argv) {
    glm53_manifest manifest;
    k3_st_model all, main_model;
    glm53_architecture_report architecture;
    glm53_weight_plan weights;
    glm53_static_layout full_layout, smoke_layout;
    glm53_static_layout_entry smoke_entries[2];
    glm53_static_store *store = NULL;
    glm53_static_bindings bindings;
    glm53_phase5c_session *diagnostic_session = NULL;
    glm53_static_loader_stats stats;
    glm53_process_memory before, loaded, after;
    char error[512], root_prefix[PATH_MAX];
    hipDeviceProp_t properties;
    const glm53_static_layout *selected_layout = NULL;
    const glm53_static_layout_entry *smallest_big = NULL;
    size_t i, main_count = 0u;
    int device_count = 0, result = 1;
    bool full = false, metadata_open = false, after_sampled = false;
    uint64_t crc0 = 0u, crc1 = 0u, sample_crc = 0u, diagnostic_crc = 0u;
    double load_start = 0.0, load_end = 0.0, verify_end = 0.0;
    glm53_static_loader_status loader_status;

    memset(&manifest, 0, sizeof(manifest));
    memset(&all, 0, sizeof(all));
    memset(&main_model, 0, sizeof(main_model));
    memset(&architecture, 0, sizeof(architecture));
    memset(&weights, 0, sizeof(weights));
    memset(&full_layout, 0, sizeof(full_layout));
    memset(&smoke_layout, 0, sizeof(smoke_layout));
    memset(&bindings, 0, sizeof(bindings));
    memset(smoke_entries, 0, sizeof(smoke_entries));
    memset(&stats, 0, sizeof(stats));
    memset(error, 0, sizeof(error));

    if (argc < 2 || argc > 3 ||
        (argc == 3 && strcmp(argv[2], "full") != 0)) {
        fprintf(stderr, "usage: %s OFFICIAL_ROOT [full]\n", argv[0]);
        return 2;
    }
    full = argc == 3;
    if (realpath(argv[1], root_prefix) == NULL) {
        fprintf(stderr, "FAIL realpath(%s): %s\n", argv[1], strerror(errno));
        return 1;
    }

    CHECK(glm53_manifest_load(&manifest, argv[1], error, sizeof(error)));
    CHECK(k3_st_model_open_5digit_total(&all, argv[1], GLM53_SHARD_COUNT,
                                        error, sizeof(error)));
    metadata_open = true;
    CHECK(glm53_manifest_reconcile(&manifest, &all, error, sizeof(error)));
    CHECK(glm53_architecture_validate(&all, &architecture, error, sizeof(error)));
    main_model.tensors = (k3_st_tensor *)calloc(GLM53_MAIN_TENSOR_COUNT,
                                                sizeof(*main_model.tensors));
    CHECK(main_model.tensors != NULL);
    main_model.shards = all.shards;
    main_model.shard_count = all.shard_count;
    main_model.routed_span = all.routed_span;
    main_model.routed_span_context = all.routed_span_context;
    for (i = 0u; i < all.tensor_count; ++i) {
        if (glm53_architecture_validate_main_tensor(&all.tensors[i], NULL)) {
            CHECK(main_count < GLM53_MAIN_TENSOR_COUNT);
            main_model.tensors[main_count++] = all.tensors[i];
        }
    }
    CHECK(main_count == GLM53_MAIN_TENSOR_COUNT);
    main_model.tensor_count = main_count;
    main_model.tensor_capacity = main_count;
    CHECK(glm53_architecture_validate_main(&main_model, &architecture,
                                           error, sizeof(error)));
    CHECK(glm53_weight_plan_build_manifest(&weights, &manifest, &main_model,
                                            error, sizeof(error)));
    CHECK(glm53_static_layout_build(&full_layout, &weights,
                                     error, sizeof(error)) == GLM53_STATIC_LAYOUT_OK);
    CHECK(full_layout.entry_count == EXPECTED_ENTRIES);
    CHECK(full_layout.tensor_count == EXPECTED_ENTRIES);
    CHECK(full_layout.logical_bytes == EXPECTED_LOGICAL);
    CHECK(full_layout.padded_bytes == EXPECTED_PADDED);
    CHECK(sample_memory("before", root_prefix, &before));

    if (full) {
        bool enough = before.mem_available_bytes >= UINT64_C(100) * GIB &&
                      glm53_process_memory_policy_allows(&before,
                                                        full_layout.padded_bytes);
        if (!enough) {
            printf("SKIP full: requires MemAvailable>=100 GiB and policy "
                   "allowance for padded=%" PRIu64 " plus 16 GiB reserve\n",
                   full_layout.padded_bytes);
            result = 77;
            goto done;
        }
        selected_layout = &full_layout;
    } else {
        for (i = 0u; i < full_layout.entry_count; ++i) {
            const glm53_static_layout_entry *candidate = &full_layout.entries[i];
            if (candidate->logical_bytes > GLM53_STATIC_LOADER_CHUNK_BYTES &&
                (smallest_big == NULL ||
                 candidate->logical_bytes < smallest_big->logical_bytes ||
                 (candidate->logical_bytes == smallest_big->logical_bytes &&
                  strcmp(candidate->tensor->name,
                         smallest_big->tensor->name) < 0)))
                smallest_big = candidate;
        }
        CHECK(smallest_big != NULL && smallest_big != &full_layout.entries[0]);
        smoke_entries[0] = full_layout.entries[0];
        smoke_entries[1] = *smallest_big;
        if (strcmp(smoke_entries[0].tensor->name,
                   smoke_entries[1].tensor->name) > 0) {
            glm53_static_layout_entry temporary = smoke_entries[0];
            smoke_entries[0] = smoke_entries[1];
            smoke_entries[1] = temporary;
        }
        smoke_entries[0].device_offset = 0u;
        smoke_entries[1].device_offset = align256(smoke_entries[0].logical_bytes);
        smoke_layout.entries = smoke_entries;
        smoke_layout.entry_count = 2u;
        smoke_layout.tensor_count = 2u;
        smoke_layout.logical_bytes = smoke_entries[0].logical_bytes +
                                     smoke_entries[1].logical_bytes;
        smoke_layout.padded_bytes = align256(smoke_entries[1].device_offset +
                                             smoke_entries[1].logical_bytes);
        smoke_layout.max_tensor_bytes = smoke_entries[0].logical_bytes >
                                         smoke_entries[1].logical_bytes ?
                                         smoke_entries[0].logical_bytes :
                                         smoke_entries[1].logical_bytes;
        smoke_layout.built = true;
        CHECK(glm53_process_memory_policy_allows(&before,
                                                  smoke_layout.padded_bytes));
        selected_layout = &smoke_layout;
        printf("smoke subset: %s (%" PRIu64 " bytes), %s (%" PRIu64
               " bytes), padded=%" PRIu64 "\n",
               smoke_entries[0].tensor->name, smoke_entries[0].logical_bytes,
               smoke_entries[1].tensor->name, smoke_entries[1].logical_bytes,
               smoke_layout.padded_bytes);
    }

    HIP_CHECK(hipGetDeviceCount(&device_count));
    CHECK(device_count > 0);
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    load_start = monotonic_seconds();
    loader_status = glm53_static_loader_load(&store, &all, selected_layout,
                                              NULL, NULL, &stats,
                                              error, sizeof(error));
    load_end = monotonic_seconds();
    if (loader_status != GLM53_STATIC_LOADER_OK)
        fprintf(stderr, "loader failed: %s: %s\n",
                glm53_static_loader_status_string(loader_status), error);
    CHECK(loader_status == GLM53_STATIC_LOADER_OK);
    CHECK(store != NULL);
    CHECK(stats.entry_count == selected_layout->entry_count);
    CHECK(stats.logical_read_bytes == selected_layout->logical_bytes);
    CHECK(stats.logical_submitted_bytes == selected_layout->logical_bytes);
    CHECK(stats.logical_completed_bytes == selected_layout->logical_bytes);
    CHECK(stats.device_allocation_bytes == selected_layout->padded_bytes);
    CHECK(glm53_static_store_entry_count(store) == selected_layout->entry_count);
    CHECK(glm53_static_store_device_bytes(store) == selected_layout->padded_bytes);
    CHECK(sample_memory("loaded", root_prefix, &loaded));

    if (full) {
        const glm53_static_binding *binding;
        glm53_static_bindings_status binding_status;
        CHECK(stats.logical_read_bytes == EXPECTED_LOGICAL);
        CHECK(stats.device_allocation_bytes == EXPECTED_PADDED);
        CHECK(stats.aggregate_crc64_ecma == EXPECTED_FULL_CRC64);
        binding_status = glm53_static_bindings_build(
            &bindings, &weights, &full_layout, store, error, sizeof(error));
        CHECK(binding_status == GLM53_STATIC_BINDINGS_OK);
        CHECK(bindings.binding_count == GLM53_STATIC_BINDING_COUNT);
        binding = glm53_static_bindings_global(&bindings, GLM53_GLOBAL_LM_HEAD);
        CHECK(binding != NULL && binding->device ==
              glm53_static_store_device_pointer(store, binding->runtime));
        binding = glm53_static_bindings_layer(&bindings, 0u, GLM53_ROLE_MLP_GATE);
        CHECK(binding != NULL && binding->shape[0] == 12288u &&
              binding->shape[1] == 4096u);
        binding = glm53_static_bindings_expert_scale(
            &bindings, 44u, 287u, GLM53_ROUTED_SCALE_UP);
        CHECK(binding != NULL && binding->shape[0] == 16u &&
              binding->shape[1] == 32u);
        {
            glm53_phase5c_attention_provider zero_provider = {
                glm53_phase5c_zero_attention_provider, NULL};
            glm53_phase5c_layer0_early_head_diagnostic first, second;
            std::vector<unsigned char> first_logits(
                GLM53_PHASE5C_LAYER0_VOCAB * 2u);
            std::vector<unsigned char> second_logits(
                GLM53_PHASE5C_LAYER0_VOCAB * 2u);
            CHECK(glm53_phase5c_session_create(&diagnostic_session, &bindings,
                  error, sizeof(error)) == GLM53_PHASE5C_OK);
            CHECK(glm53_phase5c_layer0_early_head_diagnostic_step(
                  diagnostic_session, 1u, 0u, &zero_provider) == GLM53_PHASE5C_OK);
            CHECK(glm53_phase5c_session_get_layer0_early_head_diagnostic(
                  diagnostic_session, &first) && first.available &&
                  first.position == 0u &&
                  first.layer0_early_head_diagnostic_logits_count ==
                      GLM53_PHASE5C_LAYER0_VOCAB);
            HIP_CHECK(hipMemcpy(first_logits.data(),
                      first.layer0_early_head_diagnostic_logits,
                      GLM53_PHASE5C_LAYER0_VOCAB * 2u, hipMemcpyDeviceToHost));
            CHECK(glm53_phase5c_session_reset(diagnostic_session) ==
                  GLM53_PHASE5C_OK);
            CHECK(glm53_phase5c_layer0_early_head_diagnostic_step(
                  diagnostic_session, 1u, 0u, &zero_provider) == GLM53_PHASE5C_OK);
            CHECK(glm53_phase5c_session_get_layer0_early_head_diagnostic(
                  diagnostic_session, &second) && second.available &&
                  second.position == 0u);
            HIP_CHECK(hipMemcpy(second_logits.data(),
                      second.layer0_early_head_diagnostic_logits,
                      GLM53_PHASE5C_LAYER0_VOCAB * 2u, hipMemcpyDeviceToHost));
            CHECK(memcmp(first_logits.data(), second_logits.data(),
                         GLM53_PHASE5C_LAYER0_VOCAB * 2u) == 0);
            diagnostic_crc = crc_update(0u, first_logits.data(),
                             GLM53_PHASE5C_LAYER0_VOCAB * 2u);
            CHECK(diagnostic_crc == EXPECTED_DIAGNOSTIC_CRC64);
            glm53_phase5c_session_destroy(diagnostic_session);
            diagnostic_session = NULL;
        }
        CHECK(compare_full_samples(&all, selected_layout, store, &sample_crc));
    } else {
        CHECK(stats.aggregate_crc64_ecma == EXPECTED_SMOKE_CRC64);
        CHECK(compare_whole_entry(&all, store, &smoke_entries[0], &crc0));
        CHECK(compare_whole_entry(&all, store, &smoke_entries[1], &crc1));
        sample_crc = crc_update(crc0, &crc1, sizeof(crc1));
    }
    verify_end = monotonic_seconds();

    glm53_phase5c_session_destroy(diagnostic_session);
    diagnostic_session = NULL;
    glm53_phase5c_session_destroy(diagnostic_session);
    glm53_static_bindings_free(&bindings);
    glm53_static_store_destroy(store);
    store = NULL;
    HIP_CHECK(hipDeviceSynchronize());
    CHECK(sample_memory("after-destroy", root_prefix, &after));
    after_sampled = true;
    printf("PASS phase5b official static loader %s on %s (%s): entries=%zu "
           "logical=%" PRIu64 " allocation=%" PRIu64
           " requests=%" PRIu64 " direct=%" PRIu64 " buffered=%" PRIu64
           " aggregate_crc64=%016" PRIx64 " verify_crc64=%016" PRIx64
           " diagnostic_crc64=%016" PRIx64
           " load=%.3fs verify=%.3fs MemAvailable(before/loaded/after)="
           "%" PRIu64 "/%" PRIu64 "/%" PRIu64 " MiB\n",
           full ? "full" : "smoke", properties.name, properties.gcnArchName,
           stats.entry_count, stats.logical_read_bytes,
           stats.device_allocation_bytes, stats.read_requests,
           stats.direct_requests, stats.buffered_requests,
           stats.aggregate_crc64_ecma, sample_crc, diagnostic_crc,
           load_end - load_start,
           verify_end - load_end, before.mem_available_bytes >> 20u,
           loaded.mem_available_bytes >> 20u, after.mem_available_bytes >> 20u);
    result = 0;

done:
    if (result != 0 && error[0] != '\0') fprintf(stderr, "detail: %s\n", error);
    glm53_static_store_destroy(store);
    if (store != NULL && metadata_open && !after_sampled) {
        if (sample_memory("cleanup", root_prefix, &after)) after_sampled = true;
    }
    glm53_static_layout_free(&full_layout);
    glm53_weight_plan_free(&weights);
    free(main_model.tensors);
    k3_st_model_close(&all);
    glm53_manifest_free(&manifest);
    return result;
}
