#include "k3_mzg.h"

#include <zstd.h>

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

enum {
    K3_MZG_SHARDS = 96,
    K3_MZG_FILE_HEADER_BYTES = 4096,
    K3_MZG_ALIGNMENT = 4096,
    K3_MZG_CODEC_ZSTD1 = 1,
    K3_MZG_FILE_FLAGS = 3,
    K3_MZG_BLOCK_FLAGS = 1,
    K3_MZG_FRAMES = 12,
    K3_MZG_WORKERS = 6,
    K3_MZG_BLOCK_HEADER_BYTES = 192,
    K3_MZG_EXPERT_BYTES = 17547264,
    K3_MZG_FRAME_RAW = 0x80000000u,
    K3_MZG_FRAME_SIZE_MASK = 0x7fffffffu,
};

static const uint32_t k_output_bytes[K3_MZG_FRAMES] = {
    2752512u, 2752512u, 172032u, 172032u,
    2752512u, 2752512u, 172032u, 172032u,
    2752512u, 2752512u, 172032u, 172032u,
};
static const uint32_t k_output_offset[K3_MZG_FRAMES] = {
    0u, 2752512u, 5505024u, 5677056u,
    5849088u, 8601600u, 11354112u, 11526144u,
    11698176u, 14450688u, 17203200u, 17375232u,
};

#pragma pack(push, 1)
typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t alignment;
    uint32_t codec;
    uint32_t flags;
    uint32_t shard_index;
    uint32_t shard_count;
    uint32_t expert_count;
    uint32_t index_entry_bytes;
    uint64_t index_offset;
    uint64_t index_bytes;
    uint64_t data_offset;
    uint64_t max_block_bytes;
    uint8_t source_manifest_sha256[32];
    uint8_t source_shard_sha256[32];
} k3_mzg_file_header;

typedef struct {
    uint16_t layer;
    uint16_t expert;
    uint32_t block_bytes;
    uint64_t block_offset;
} k3_mzg_index_entry;

typedef struct {
    char magic[4];
    uint16_t layer;
    uint16_t expert;
    uint32_t payload_bytes;
    uint32_t flags;
    uint32_t frame_count;
    uint32_t header_bytes;
} k3_mzg_block_prefix;

typedef struct {
    uint32_t encoded_bytes;
    uint32_t output_offset;
    uint32_t output_bytes;
} k3_mzg_frame_descriptor;
#pragma pack(pop)

typedef struct {
    int fd;
    int direct_fd;
    uint64_t file_bytes;
} k3_mzg_file;

typedef struct {
    uint16_t file;
    uint64_t offset;
    uint32_t bytes;
    bool present;
} k3_mzg_map_entry;

typedef struct k3_mzg_worker k3_mzg_worker;

struct k3_mzg_store {
    uint32_t layers;
    uint32_t experts;
    uint32_t max_block_bytes;
    uint16_t file_count;
    k3_mzg_file files[K3_MZG_SHARDS];
    bool source_shard_seen[K3_MZG_SHARDS];
    bool source_manifest_set;
    uint8_t source_manifest_sha256[32];
    bool expected_shard_seen[K3_MZG_SHARDS];
    uint8_t expected_shard_sha256[K3_MZG_SHARDS][32];
    k3_mzg_map_entry *map;

    pthread_t threads[K3_MZG_WORKERS];
    k3_mzg_worker *workers;
    pthread_mutex_t mutex;
    pthread_cond_t start;
    pthread_cond_t done;
    uint32_t workers_started;
    bool mutex_initialized;
    bool start_initialized;
    bool done_initialized;
    uint64_t generation;
    uint32_t completed;
    uint32_t frame_count;
    bool stop;
    bool failed;
    const uint8_t *source[K3_MZG_FRAMES];
    uint32_t stored_bytes[K3_MZG_FRAMES];
    bool raw[K3_MZG_FRAMES];
    uint8_t *destination[K3_MZG_FRAMES];
    uint32_t output_bytes[K3_MZG_FRAMES];
};

struct k3_mzg_worker {
    k3_mzg_store *store;
    uint32_t index;
    ZSTD_DCtx *context;
};

static void k3_mzg_error(char *error, size_t error_size,
                         const char *format, ...) {
    if (!error || error_size == 0u) return;
    va_list arguments;
    va_start(arguments, format);
    (void)vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static bool k3_mzg_pread_full(int fd, void *buffer,
                              size_t bytes, uint64_t offset) {
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

static int k3_mzg_hex_nibble(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

static bool k3_mzg_load_source_manifest(k3_mzg_store *store,
                                        const char *model_root,
                                        char *error,
                                        size_t error_size) {
    char path[4096];
    int length = snprintf(path, sizeof(path), "%s/manifest.tsv", model_root);
    if (length < 0 || (size_t)length >= sizeof(path)) {
        k3_mzg_error(error, error_size, "source manifest path overflow");
        return false;
    }
    FILE *file = fopen(path, "r");
    if (!file) {
        k3_mzg_error(error, error_size,
                     "open %s: %s", path, strerror(errno));
        return false;
    }
    char *line = NULL;
    size_t capacity = 0u;
    uint32_t count = 0u;
    while (getline(&line, &capacity, file) >= 0) {
        unsigned shard = 0u;
        unsigned total = 0u;
        unsigned long long bytes = 0u;
        char digest[65];
        char trailing = '\0';
        const int fields = sscanf(
            line,
            "model-%5u-of-%6u.safetensors\t%llu\t%64[0-9A-Fa-f]%c",
            &shard, &total, &bytes, digest, &trailing);
        (void)bytes;
        if (fields < 4 || shard == 0u || shard > K3_MZG_SHARDS ||
            total != K3_MZG_SHARDS || strlen(digest) != 64u ||
            store->expected_shard_seen[shard - 1u]) {
            free(line);
            (void)fclose(file);
            k3_mzg_error(error, error_size,
                         "invalid source manifest line for shard %u", shard);
            return false;
        }
        for (uint32_t byte = 0u; byte < 32u; byte++) {
            const int high = k3_mzg_hex_nibble(digest[byte * 2u]);
            const int low = k3_mzg_hex_nibble(digest[byte * 2u + 1u]);
            if (high < 0 || low < 0) {
                free(line);
                (void)fclose(file);
                k3_mzg_error(error, error_size,
                             "invalid source manifest digest");
                return false;
            }
            store->expected_shard_sha256[shard - 1u][byte] =
                (uint8_t)((high << 4) | low);
        }
        store->expected_shard_seen[shard - 1u] = true;
        count++;
    }
    free(line);
    if (fclose(file) != 0 || count != K3_MZG_SHARDS) {
        k3_mzg_error(error, error_size,
                     "source manifest has %u/%u shards",
                     count, K3_MZG_SHARDS);
        return false;
    }
    return true;
}

static void *k3_mzg_worker_main(void *opaque) {
    k3_mzg_worker *worker = (k3_mzg_worker *)opaque;
    k3_mzg_store *store = worker->store;
    uint64_t observed = 0u;
    for (;;) {
        (void)pthread_mutex_lock(&store->mutex);
        while (!store->stop && store->generation == observed) {
            (void)pthread_cond_wait(&store->start, &store->mutex);
        }
        if (store->stop) {
            (void)pthread_mutex_unlock(&store->mutex);
            return NULL;
        }
        observed = store->generation;
        const uint32_t frame_count = store->frame_count;
        (void)pthread_mutex_unlock(&store->mutex);

        bool worker_ok = true;
        for (uint32_t frame = worker->index;
             frame < frame_count;
             frame += K3_MZG_WORKERS) {
            bool ok = false;
            if (store->raw[frame]) {
                ok = store->stored_bytes[frame] ==
                    store->output_bytes[frame];
                if (ok) {
                    memcpy(store->destination[frame],
                           store->source[frame],
                           store->output_bytes[frame]);
                }
            } else {
                const size_t result = ZSTD_decompressDCtx(
                    worker->context,
                    store->destination[frame],
                    store->output_bytes[frame],
                    store->source[frame],
                    store->stored_bytes[frame]);
                ok = !ZSTD_isError(result) &&
                    result == store->output_bytes[frame];
            }
            if (!ok) worker_ok = false;
        }

        (void)pthread_mutex_lock(&store->mutex);
        if (!worker_ok) store->failed = true;
        store->completed++;
        if (store->completed == K3_MZG_WORKERS) {
            (void)pthread_cond_signal(&store->done);
        }
        (void)pthread_mutex_unlock(&store->mutex);
    }
}

static bool k3_mzg_start_workers(k3_mzg_store *store,
                                 char *error, size_t error_size) {
    store->workers = (k3_mzg_worker *)calloc(
        K3_MZG_WORKERS, sizeof(*store->workers));
    if (!store->workers) {
        k3_mzg_error(error, error_size, "MZG worker allocation failed");
        return false;
    }
    if (pthread_mutex_init(&store->mutex, NULL) != 0) {
        k3_mzg_error(error, error_size, "MZG worker mutex creation failed");
        return false;
    }
    store->mutex_initialized = true;
    if (pthread_cond_init(&store->start, NULL) != 0) {
        k3_mzg_error(error, error_size, "MZG worker start condition failed");
        return false;
    }
    store->start_initialized = true;
    if (pthread_cond_init(&store->done, NULL) != 0) {
        k3_mzg_error(error, error_size, "MZG worker done condition failed");
        return false;
    }
    store->done_initialized = true;
    for (uint32_t worker = 0u; worker < K3_MZG_WORKERS; worker++) {
        store->workers[worker].store = store;
        store->workers[worker].index = worker;
        store->workers[worker].context = ZSTD_createDCtx();
        if (!store->workers[worker].context ||
            pthread_create(&store->threads[worker], NULL,
                           k3_mzg_worker_main,
                           &store->workers[worker]) != 0) {
            k3_mzg_error(error, error_size,
                         "MZG worker %u creation failed", worker);
            return false;
        }
        store->workers_started++;
    }
    return true;
}

void k3_mzg_store_destroy(k3_mzg_store *store) {
    if (!store) return;
    if (store->workers_started != 0u && store->mutex_initialized) {
        (void)pthread_mutex_lock(&store->mutex);
        store->stop = true;
        if (store->start_initialized) {
            (void)pthread_cond_broadcast(&store->start);
        }
        (void)pthread_mutex_unlock(&store->mutex);
        for (uint32_t worker = 0u;
             worker < store->workers_started; worker++) {
            (void)pthread_join(store->threads[worker], NULL);
        }
    }
    if (store->workers) {
        for (uint32_t worker = 0u; worker < K3_MZG_WORKERS; worker++) {
            ZSTD_freeDCtx(store->workers[worker].context);
        }
    }
    if (store->done_initialized) {
        (void)pthread_cond_destroy(&store->done);
    }
    if (store->start_initialized) {
        (void)pthread_cond_destroy(&store->start);
    }
    if (store->mutex_initialized) {
        (void)pthread_mutex_destroy(&store->mutex);
    }
    for (uint16_t file = 0u; file < store->file_count; file++) {
        if (store->files[file].direct_fd >= 0) {
            (void)close(store->files[file].direct_fd);
        }
        if (store->files[file].fd >= 0) {
            (void)close(store->files[file].fd);
        }
    }
    free(store->workers);
    free(store->map);
    free(store);
}

static bool k3_mzg_load_file(k3_mzg_store *store,
                             const char *path,
                             char *error, size_t error_size) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    int direct_fd = open(path, O_RDONLY | O_CLOEXEC | O_DIRECT);
    if (fd < 0 || direct_fd < 0) {
        const int saved = errno;
        if (fd >= 0) (void)close(fd);
        if (direct_fd >= 0) (void)close(direct_fd);
        k3_mzg_error(error, error_size, "open %s: %s", path, strerror(saved));
        return false;
    }
    struct stat status;
    k3_mzg_file_header header;
    if (fstat(fd, &status) != 0 ||
        !k3_mzg_pread_full(fd, &header, sizeof(header), 0u) ||
        memcmp(header.magic, "K3MZG1\0", 8u) != 0 ||
        header.version != 1u ||
        header.header_bytes != K3_MZG_FILE_HEADER_BYTES ||
        header.alignment != K3_MZG_ALIGNMENT ||
        header.codec != K3_MZG_CODEC_ZSTD1 ||
        header.flags != K3_MZG_FILE_FLAGS ||
        header.shard_count != K3_MZG_SHARDS ||
        header.shard_index >= K3_MZG_SHARDS ||
        header.index_entry_bytes != sizeof(k3_mzg_index_entry) ||
        header.index_offset != K3_MZG_FILE_HEADER_BYTES ||
        header.index_bytes !=
            (uint64_t)header.expert_count * sizeof(k3_mzg_index_entry) ||
        header.data_offset < header.index_offset + header.index_bytes ||
        header.data_offset % K3_MZG_ALIGNMENT != 0u ||
        header.max_block_bytes == 0u ||
        header.max_block_bytes % K3_MZG_ALIGNMENT != 0u ||
        header.max_block_bytes > K3_MZG_EXPERT_BYTES + K3_MZG_ALIGNMENT) {
        k3_mzg_error(error, error_size, "invalid MZG header %s", path);
        (void)close(direct_fd);
        (void)close(fd);
        return false;
    }
    if (store->source_shard_seen[header.shard_index]) {
        k3_mzg_error(error, error_size,
                     "duplicate MZG source shard %u",
                     header.shard_index);
        (void)close(direct_fd);
        (void)close(fd);
        return false;
    }
    if (store->source_manifest_set) {
        if (memcmp(
                store->source_manifest_sha256,
                header.source_manifest_sha256,
                sizeof(store->source_manifest_sha256)) != 0) {
            k3_mzg_error(error, error_size,
                         "MZG source manifest mismatch %s", path);
            (void)close(direct_fd);
            (void)close(fd);
            return false;
        }
    } else {
        memcpy(store->source_manifest_sha256,
               header.source_manifest_sha256,
               sizeof(store->source_manifest_sha256));
        store->source_manifest_set = true;
    }
    store->source_shard_seen[header.shard_index] = true;
    if (!store->expected_shard_seen[header.shard_index] ||
        memcmp(
            store->expected_shard_sha256[header.shard_index],
            header.source_shard_sha256, 32u) != 0) {
        k3_mzg_error(error, error_size,
                     "MZG/source shard identity mismatch %s", path);
        (void)close(direct_fd);
        (void)close(fd);
        return false;
    }
    if (store->file_count >= K3_MZG_SHARDS) {
        k3_mzg_error(error, error_size, "too many MZG sidecars");
        (void)close(direct_fd);
        (void)close(fd);
        return false;
    }
    k3_mzg_index_entry *index = (k3_mzg_index_entry *)malloc(
        (size_t)header.index_bytes);
    if (!index || !k3_mzg_pread_full(
            fd, index, (size_t)header.index_bytes,
            header.index_offset)) {
        k3_mzg_error(error, error_size, "read MZG index %s", path);
        free(index);
        (void)close(direct_fd);
        (void)close(fd);
        return false;
    }
    const uint16_t file_index = store->file_count;
    uint32_t loaded = 0u;
    for (uint32_t item = 0u; item < header.expert_count; item++) {
        const k3_mzg_index_entry *entry = &index[item];
        if (entry->layer == 0u || entry->layer > store->layers ||
            entry->expert >= store->experts ||
            entry->block_offset % K3_MZG_ALIGNMENT != 0u ||
            entry->block_bytes == 0u ||
            entry->block_bytes % K3_MZG_ALIGNMENT != 0u ||
            entry->block_bytes > header.max_block_bytes ||
            entry->block_offset > (uint64_t)status.st_size ||
            entry->block_bytes >
                (uint64_t)status.st_size - entry->block_offset) {
            k3_mzg_error(error, error_size,
                         "invalid MZG index entry %s item %u", path, item);
            free(index);
            (void)close(direct_fd);
            (void)close(fd);
            return false;
        }
        const uint64_t map_index =
            (uint64_t)(entry->layer - 1u) * store->experts + entry->expert;
        k3_mzg_map_entry *mapping = &store->map[map_index];
        if (mapping->present) {
            k3_mzg_error(error, error_size,
                         "duplicate MZG layer %u expert %u",
                         entry->layer, entry->expert);
            free(index);
            (void)close(direct_fd);
            (void)close(fd);
            return false;
        }
        mapping->file = file_index;
        mapping->offset = entry->block_offset;
        mapping->bytes = entry->block_bytes;
        mapping->present = true;
        loaded++;
    }
    free(index);
    store->files[file_index].fd = fd;
    store->files[file_index].direct_fd = direct_fd;
    store->files[file_index].file_bytes = (uint64_t)status.st_size;
    store->file_count++;
    if (header.max_block_bytes > store->max_block_bytes) {
        store->max_block_bytes = (uint32_t)header.max_block_bytes;
    }
    return loaded == header.expert_count;
}

bool k3_mzg_store_open_optional(k3_mzg_store **out,
                                const char *model_root,
                                uint32_t layers,
                                uint32_t experts,
                                char *error,
                                size_t error_size) {
    if (out) *out = NULL;
    if (error && error_size) error[0] = '\0';
    if (!out || !model_root || layers == 0u || experts == 0u ||
        (uint64_t)layers * experts > SIZE_MAX / sizeof(k3_mzg_map_entry)) {
        k3_mzg_error(error, error_size, "invalid MZG store configuration");
        return false;
    }
    const char *selection = getenv("MOONSHINE_EXPERT_STORE");
    if (!selection || selection[0] == '\0' ||
        strcmp(selection, "off") == 0) {
        return true;
    }
    char root[4096];
    int length = 0;
    if (strcmp(selection, "auto") == 0) {
        length = snprintf(root, sizeof(root),
                          "%s/expert-store-mzg1", model_root);
    } else if (selection[0] == '/') {
        length = snprintf(root, sizeof(root), "%s", selection);
    } else {
        k3_mzg_error(
            error, error_size,
            "MOONSHINE_EXPERT_STORE must be auto, off, or absolute");
        return false;
    }
    if (length < 0 || (size_t)length >= sizeof(root)) {
        k3_mzg_error(error, error_size, "MZG store path overflow");
        return false;
    }
    struct stat root_status;
    if (stat(root, &root_status) != 0) {
        k3_mzg_error(error, error_size, "stat %s: %s", root, strerror(errno));
        return false;
    }
    if (!S_ISDIR(root_status.st_mode)) {
        k3_mzg_error(error, error_size, "%s is not a directory", root);
        return false;
    }

    k3_mzg_store *store = (k3_mzg_store *)calloc(1, sizeof(*store));
    if (!store) {
        k3_mzg_error(error, error_size, "MZG store allocation failed");
        return false;
    }
    store->layers = layers;
    store->experts = experts;
    for (uint32_t file = 0u; file < K3_MZG_SHARDS; file++) {
        store->files[file].fd = -1;
        store->files[file].direct_fd = -1;
    }
    store->map = (k3_mzg_map_entry *)calloc(
        (size_t)layers * experts, sizeof(*store->map));
    if (!store->map) {
        k3_mzg_error(error, error_size, "MZG map allocation failed");
        k3_mzg_store_destroy(store);
        return false;
    }

    if (!k3_mzg_load_source_manifest(
            store, model_root, error, error_size)) {
        k3_mzg_store_destroy(store);
        return false;
    }
    for (uint32_t shard = 1u; shard <= K3_MZG_SHARDS; shard++) {
        char path[4096];
        length = snprintf(
            path, sizeof(path),
            "%s/model-%05u-of-000096.mzg", root, shard);
        if (length < 0 || (size_t)length >= sizeof(path)) {
            k3_mzg_error(error, error_size, "MZG sidecar path overflow");
            k3_mzg_store_destroy(store);
            return false;
        }
        if (access(path, F_OK) != 0) {
            if (errno == ENOENT) continue;
            k3_mzg_error(error, error_size,
                         "access %s: %s", path, strerror(errno));
            k3_mzg_store_destroy(store);
            return false;
        }
        if (!k3_mzg_load_file(store, path, error, error_size)) {
            k3_mzg_store_destroy(store);
            return false;
        }
    }

    uint64_t present = 0u;
    for (uint64_t index = 0u; index < (uint64_t)layers * experts; index++) {
        if (store->map[index].present) present++;
    }
    if (present != (uint64_t)layers * experts ||
        store->file_count == 0u || store->max_block_bytes == 0u) {
        k3_mzg_error(error, error_size,
                     "MZG store is incomplete: %llu/%llu experts",
                     (unsigned long long)present,
                     (unsigned long long)((uint64_t)layers * experts));
        k3_mzg_store_destroy(store);
        return false;
    }
    if (!k3_mzg_start_workers(store, error, error_size)) {
        k3_mzg_store_destroy(store);
        return false;
    }
    *out = store;
    return true;
}

uint32_t k3_mzg_store_max_block_bytes(const k3_mzg_store *store) {
    return store ? store->max_block_bytes : 0u;
}

bool k3_mzg_store_span(const k3_mzg_store *store,
                       uint32_t layer,
                       uint32_t expert,
                       k3_mzg_span *span) {
    if (span) memset(span, 0, sizeof(*span));
    if (!store || !span || layer == 0u || layer > store->layers ||
        expert >= store->experts) {
        return false;
    }
    const k3_mzg_map_entry *mapping =
        &store->map[(uint64_t)(layer - 1u) * store->experts + expert];
    if (!mapping->present || mapping->file >= store->file_count) return false;
    span->direct_fd = store->files[mapping->file].direct_fd;
    span->offset = mapping->offset;
    span->bytes = mapping->bytes;
    return true;
}

bool k3_mzg_store_decode(k3_mzg_store *store,
                         uint32_t layer,
                         uint32_t expert,
                         const void *block,
                         uint32_t block_bytes,
                         void *destination,
                         char *error,
                         size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!store || !block || !destination ||
        block_bytes < K3_MZG_BLOCK_HEADER_BYTES) {
        k3_mzg_error(error, error_size, "invalid MZG decode arguments");
        return false;
    }
    const uint8_t *bytes = (const uint8_t *)block;
    const k3_mzg_block_prefix *prefix =
        (const k3_mzg_block_prefix *)bytes;
    if (memcmp(prefix->magic, "MZGB", 4u) != 0 ||
        prefix->layer != layer || prefix->expert != expert ||
        prefix->flags != K3_MZG_BLOCK_FLAGS ||
        prefix->frame_count != K3_MZG_FRAMES ||
        prefix->header_bytes != K3_MZG_BLOCK_HEADER_BYTES) {
        k3_mzg_error(error, error_size,
                     "MZG block header mismatch for layer %u expert %u",
                     layer, expert);
        return false;
    }
    const k3_mzg_frame_descriptor *descriptor =
        (const k3_mzg_frame_descriptor *)(bytes + sizeof(*prefix));
    uint64_t cursor = prefix->header_bytes;
    uint64_t payload = 0u;
    (void)pthread_mutex_lock(&store->mutex);
    store->completed = 0u;
    store->frame_count = K3_MZG_FRAMES;
    store->failed = false;
    for (uint32_t frame = 0u; frame < K3_MZG_FRAMES; frame++) {
        const uint32_t encoded = descriptor[frame].encoded_bytes;
        const uint32_t stored = encoded & K3_MZG_FRAME_SIZE_MASK;
        if (descriptor[frame].output_offset != k_output_offset[frame] ||
            descriptor[frame].output_bytes != k_output_bytes[frame] ||
            cursor + stored > block_bytes) {
            (void)pthread_mutex_unlock(&store->mutex);
            k3_mzg_error(error, error_size,
                         "MZG frame %u bounds mismatch", frame);
            return false;
        }
        store->source[frame] = bytes + cursor;
        store->stored_bytes[frame] = stored;
        store->raw[frame] = (encoded & K3_MZG_FRAME_RAW) != 0u;
        store->destination[frame] =
            (uint8_t *)destination + descriptor[frame].output_offset;
        store->output_bytes[frame] = descriptor[frame].output_bytes;
        cursor += stored;
        payload += stored;
    }
    if (payload != prefix->payload_bytes) {
        (void)pthread_mutex_unlock(&store->mutex);
        k3_mzg_error(error, error_size, "MZG block payload mismatch");
        return false;
    }
    store->generation++;
    (void)pthread_cond_broadcast(&store->start);
    while (store->completed != K3_MZG_WORKERS) {
        (void)pthread_cond_wait(&store->done, &store->mutex);
    }
    const bool ok = !store->failed;
    (void)pthread_mutex_unlock(&store->mutex);
    if (!ok) {
        k3_mzg_error(error, error_size,
                     "MZG Zstd decode failed for layer %u expert %u",
                     layer, expert);
    }
    return ok;
}
