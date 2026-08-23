#include "k3_mzg2.h"

#include <hip/hip_runtime.h>

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr uint32_t kScaleBits = 12u;
constexpr uint32_t kRansLowerBound = 1u << 16u;
constexpr uint32_t kWaveSize = 32u;
constexpr uint32_t kThreads = 256u;
constexpr uint32_t kWavesPerBlock = kThreads / kWaveSize;
constexpr uint32_t kLayers = 92u;
constexpr uint32_t kExperts = 896u;
constexpr uint32_t kMaxFiles = kLayers;

const uint8_t kPinnedManifestSha256[32] = {
    0x47, 0x6f, 0xa0, 0xba, 0x64, 0xe3, 0x23, 0x3c,
    0xbb, 0x9c, 0xa0, 0x64, 0x23, 0x27, 0x36, 0x1a,
    0x73, 0xf6, 0x80, 0x7e, 0x75, 0x1e, 0xdb, 0x07,
    0x1c, 0x92, 0xfa, 0x22, 0x16, 0xb2, 0x02, 0xa4,
};

struct StoreFile {
    int fd;
    int direct_fd;
    uint64_t bytes;
};

struct MapEntry {
    uint16_t file;
    bool present;
    uint64_t offset;
    uint32_t bytes;
};

struct StoreImpl {
    StoreFile files[kMaxFiles];
    uint16_t file_count;
    uint32_t max_block_bytes;
    MapEntry *map;
};

void set_error(char *error, size_t error_size, const char *format, ...) {
    if (!error || error_size == 0u) return;
    va_list arguments;
    va_start(arguments, format);
    (void)vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

bool pread_full(int fd, void *buffer, size_t bytes, uint64_t offset) {
    uint8_t *cursor = static_cast<uint8_t *>(buffer);
    while (bytes != 0u) {
        const ssize_t result = pread(fd, cursor, bytes, static_cast<off_t>(offset));
        if (result < 0 && errno == EINTR) continue;
        if (result <= 0) return false;
        cursor += static_cast<size_t>(result);
        bytes -= static_cast<size_t>(result);
        offset += static_cast<uint64_t>(result);
    }
    return true;
}


__host__ __device__ uint32_t word_checksum(
        uint32_t word, uint32_t dword_index) {
    const uint32_t rotation = dword_index & 31u;
    const uint32_t rotated = rotation
        ? (word << rotation) | (word >> (32u - rotation))
        : word;
    return rotated ^ (dword_index * UINT32_C(0x9e3779b9));
}

bool validate_model(const k3_mzg2_model &model, bool packed) {
    const uint32_t count = model.symbol_count;
    if (count == 0u || count > 15u) return false;
    uint32_t total = 0u;
    for (uint32_t symbol = 0u; symbol < count; ++symbol) {
        const uint32_t frequency = model.frequency[symbol];
        if (model.cumulative[symbol] != total || frequency == 0u ||
            frequency > K3_MZG2_MODEL_TOTAL - total) {
            return false;
        }
        const uint8_t value = model.values[symbol];
        if (packed && (value > 15u || value == 8u)) return false;
        for (uint32_t previous = 0u; previous < symbol; ++previous) {
            if (model.values[previous] == value) return false;
        }
        total += frequency;
    }
    if (total != K3_MZG2_MODEL_TOTAL) return false;
    for (uint32_t slot = 0u; slot < K3_MZG2_MODEL_TOTAL; ++slot) {
        const uint32_t symbol = model.decode[slot];
        if (symbol >= count || slot < model.cumulative[symbol] ||
            slot >= model.cumulative[symbol] + model.frequency[symbol]) {
            return false;
        }
    }
    return true;
}

bool validate_model_entry(const k3_mzg2_model_entry &entry) {
    return validate_model(entry.model[K3_MZG2_MODEL_PACKED], true) &&
           validate_model(entry.model[K3_MZG2_MODEL_SCALE], false);
}

__global__ void decode_tiles_kernel(
        const uint8_t *block,
        uint32_t block_bytes,
        const k3_mzg2_tile_descriptor *descriptors,
        uint32_t tile_count,
        const k3_mzg2_model_entry *entry,
        uint8_t *output,
        uint32_t *error_flags) {
    __shared__ uint8_t decode_lut[2][K3_MZG2_MODEL_TOTAL];
    __shared__ uint8_t model_values[2][15];
    __shared__ uint8_t symbol_count[2];
    __shared__ uint32_t frequency[2][15];
    __shared__ uint32_t cumulative[2][15];
    __shared__ k3_mzg2_tile_descriptor tile_descriptors[kWavesPerBlock];
    for (uint32_t index = threadIdx.x;
         index < 2u * K3_MZG2_MODEL_TOTAL; index += blockDim.x) {
        decode_lut[index / K3_MZG2_MODEL_TOTAL]
                  [index % K3_MZG2_MODEL_TOTAL] =
            entry->model[index / K3_MZG2_MODEL_TOTAL]
                .decode[index % K3_MZG2_MODEL_TOTAL];
    }
    for (uint32_t index = threadIdx.x; index < 30u; index += blockDim.x) {
        const uint32_t model = index / 15u;
        const uint32_t symbol = index % 15u;
        model_values[model][symbol] = entry->model[model].values[symbol];
        frequency[model][symbol] = entry->model[model].frequency[symbol];
        cumulative[model][symbol] = entry->model[model].cumulative[symbol];
    }
    if (threadIdx.x < 2u) {
        symbol_count[threadIdx.x] =
            entry->model[threadIdx.x].symbol_count;
    }
    if (threadIdx.x < kWavesPerBlock) {
        const uint32_t index = blockIdx.x * kWavesPerBlock + threadIdx.x;
        if (index < tile_count) {
            tile_descriptors[threadIdx.x] = descriptors[index];
        }
    }
    __syncthreads();

    const uint32_t lane = threadIdx.x & (kWaveSize - 1u);
    const uint32_t wave = threadIdx.x / kWaveSize;
    const uint32_t tile_index = blockIdx.x * kWavesPerBlock + wave;
    if (tile_index >= tile_count) return;
    const k3_mzg2_tile_descriptor descriptor = tile_descriptors[wave];
    const uint32_t model = descriptor.kind & K3_MZG2_MODEL_MASK;
    bool valid = model <= K3_MZG2_MODEL_SCALE &&
        (descriptor.kind &
         ~(K3_MZG2_TILE_RAW | K3_MZG2_MODEL_MASK)) == 0u &&
        descriptor.reserved == 0u &&
        descriptor.output_bytes == 16384u &&
        descriptor.output_offset == tile_index * 16384u &&
        descriptor.output_offset <= K3_MZG2_EXPERT_BYTES &&
        descriptor.output_bytes <=
            K3_MZG2_EXPERT_BYTES - descriptor.output_offset &&
        descriptor.payload_offset % 4u == 0u &&
        descriptor.payload_offset <= block_bytes &&
        descriptor.payload_bytes <= block_bytes - descriptor.payload_offset;
    if (!valid) {
        if (lane == 0u) atomicOr(error_flags, K3_MZG2_ERROR_BOUNDS);
        return;
    }

    const uint8_t *payload = block + descriptor.payload_offset;
    uint8_t *destination = output + descriptor.output_offset;
    uint32_t checksum = 0u;
    if (descriptor.kind & K3_MZG2_TILE_RAW) {
        if (descriptor.payload_bytes != descriptor.output_bytes) {
            if (lane == 0u) atomicOr(error_flags, K3_MZG2_ERROR_BOUNDS);
            return;
        }
        for (uint32_t offset = lane * 4u; offset < descriptor.output_bytes;
             offset += kWaveSize * 4u) {
            const uint32_t word =
                *reinterpret_cast<const uint32_t *>(payload + offset);
            *reinterpret_cast<uint32_t *>(destination + offset) = word;
            checksum ^= word_checksum(word, offset / 4u);
        }
    } else {
        if (descriptor.payload_bytes < kWaveSize * sizeof(uint32_t) ||
            (descriptor.payload_bytes - kWaveSize * sizeof(uint32_t)) % 2u) {
            if (lane == 0u) atomicOr(error_flags, K3_MZG2_ERROR_BOUNDS);
            return;
        }
        const uint32_t *initial_states =
            reinterpret_cast<const uint32_t *>(payload);
        const uint16_t *words = reinterpret_cast<const uint16_t *>(
            payload + kWaveSize * sizeof(uint32_t));
        const uint32_t word_count =
            (descriptor.payload_bytes - kWaveSize * sizeof(uint32_t)) / 2u;
        uint32_t state = initial_states[lane];
        if (__ballot(state < kRansLowerBound)) valid = false;
        uint32_t cursor = 0u;
        uint32_t assembled = 0u;
        const uint32_t symbols_per_byte =
            model == K3_MZG2_MODEL_PACKED ? 2u : 1u;
        const uint32_t rounds =
            descriptor.output_bytes * symbols_per_byte / kWaveSize;
        const unsigned long long lane_mask = lane == 0u
            ? 0u : ((UINT64_C(1) << lane) - 1u);
        for (uint32_t round = 0u; round < rounds && valid; ++round) {
            const uint32_t slot = state & (K3_MZG2_MODEL_TOTAL - 1u);
            const uint32_t symbol = decode_lut[model][slot];
            if (symbol >= 15u || symbol >= symbol_count[model]) {
                valid = false;
                break;
            }
            state = frequency[model][symbol] * (state >> kScaleBits)
                + slot - cumulative[model][symbol];
            const bool need = state < kRansLowerBound;
            const unsigned long long mask = __ballot(need);
            const uint32_t needed = __popcll(mask);
            if (needed > word_count - cursor) {
                valid = false;
                break;
            }
            if (need) {
                state = (state << 16u)
                    | words[cursor + __popcll(mask & lane_mask)];
            }
            cursor += needed;
            const uint32_t value = model_values[model][symbol];
            if (model == K3_MZG2_MODEL_PACKED) {
                assembled |= value << ((round & 7u) * 4u);
                if ((round & 7u) == 7u) {
                    const uint32_t offset =
                        ((round >> 3u) * kWaveSize + lane) * 4u;
                    *reinterpret_cast<uint32_t *>(destination + offset) =
                        assembled;
                    checksum ^= word_checksum(assembled, offset / 4u);
                    assembled = 0u;
                }
            } else {
                assembled |= value << ((round & 3u) * 8u);
                if ((round & 3u) == 3u) {
                    const uint32_t offset =
                        ((round >> 2u) * kWaveSize + lane) * 4u;
                    *reinterpret_cast<uint32_t *>(destination + offset) =
                        assembled;
                    checksum ^= word_checksum(assembled, offset / 4u);
                    assembled = 0u;
                }
            }
        }
        if (__ballot(!valid || cursor != word_count ||
                     state != kRansLowerBound)) {
            if (lane == 0u) {
                atomicOr(error_flags,
                         valid ? K3_MZG2_ERROR_TERMINAL
                               : K3_MZG2_ERROR_STATE);
            }
            return;
        }
    }
    for (uint32_t delta = 16u; delta > 0u; delta >>= 1u) {
        checksum ^= __shfl_xor(checksum, delta, kWaveSize);
    }
    if (lane == 0u &&
        checksum != static_cast<uint32_t>(descriptor.checksum)) {
        atomicOr(error_flags, K3_MZG2_ERROR_CHECKSUM);
    }
}

bool load_file(StoreImpl *store,
               const char *path,
               char *error,
               size_t error_size) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    int direct_fd = open(path, O_RDONLY | O_CLOEXEC | O_DIRECT);
    struct stat status{};
    k3_mzg2_file_header header{};
    if (fd < 0 || direct_fd < 0 || fstat(fd, &status) != 0 ||
        !pread_full(fd, &header, sizeof(header), 0u) ||
        memcmp(header.magic, "K3MZG2\0", 8u) != 0 ||
        header.version != K3_MZG2_VERSION ||
        header.header_bytes != K3_MZG2_FILE_HEADER_BYTES ||
        header.alignment != K3_MZG2_ALIGNMENT ||
        header.layer == 0u || header.layer > kLayers ||
        header.expert_count == 0u || header.expert_count > kExperts ||
        header.index_entry_bytes != sizeof(k3_mzg2_index_entry) ||
        header.model_entry_bytes != 0u ||
        header.tile_bytes != 16384u || header.flags != 0u ||
        header.index_offset != K3_MZG2_FILE_HEADER_BYTES ||
        header.index_bytes !=
            static_cast<uint64_t>(header.expert_count)
                * sizeof(k3_mzg2_index_entry) ||
        header.model_offset != 0u || header.model_bytes != 0u ||
        header.data_offset < header.index_offset + header.index_bytes ||
        header.data_offset % K3_MZG2_ALIGNMENT != 0u ||
        header.max_block_bytes == 0u ||
        header.max_block_bytes > UINT32_MAX ||
        header.max_block_bytes % K3_MZG2_ALIGNMENT != 0u ||
        header.data_offset > static_cast<uint64_t>(status.st_size) ||
        memcmp(header.source_manifest_sha256,
               kPinnedManifestSha256, sizeof(kPinnedManifestSha256)) != 0) {
        const int saved = errno;
        if (direct_fd >= 0) close(direct_fd);
        if (fd >= 0) close(fd);
        set_error(error, error_size, "invalid MZG2 store %s: %s",
                  path, strerror(saved));
        return false;
    }
    if (store->file_count >= kMaxFiles) {
        close(direct_fd);
        close(fd);
        set_error(error, error_size, "too many MZG2 sidecars");
        return false;
    }
    auto *index = static_cast<k3_mzg2_index_entry *>(
        malloc(static_cast<size_t>(header.index_bytes)));
    if (!index || !pread_full(
            fd, index, static_cast<size_t>(header.index_bytes),
            header.index_offset)) {
        free(index);
        close(direct_fd);
        close(fd);
        set_error(error, error_size, "read MZG2 index %s", path);
        return false;
    }
    const uint16_t file_index = store->file_count;
    for (uint32_t item = 0u; item < header.expert_count; ++item) {
        const k3_mzg2_index_entry &entry = index[item];
        if (entry.layer != header.layer || entry.expert >= kExperts ||
            entry.block_bytes == 0u ||
            entry.block_bytes % K3_MZG2_ALIGNMENT != 0u ||
            entry.block_bytes > header.max_block_bytes ||
            entry.block_offset % K3_MZG2_ALIGNMENT != 0u ||
            entry.block_offset < header.data_offset ||
            entry.block_offset > static_cast<uint64_t>(status.st_size) ||
            entry.block_bytes >
                static_cast<uint64_t>(status.st_size) - entry.block_offset) {
            free(index);
            close(direct_fd);
            close(fd);
            set_error(error, error_size,
                      "invalid MZG2 index %s item %u", path, item);
            return false;
        }
        MapEntry &mapping = store->map[
            static_cast<uint64_t>(entry.layer - 1u) * kExperts
            + entry.expert];
        if (mapping.present) {
            free(index);
            close(direct_fd);
            close(fd);
            set_error(error, error_size,
                      "duplicate MZG2 layer %u expert %u",
                      entry.layer, entry.expert);
            return false;
        }
        mapping.file = file_index;
        mapping.present = true;
        mapping.offset = entry.block_offset;
        mapping.bytes = entry.block_bytes;
    }
    free(index);
    store->files[file_index].fd = fd;
    store->files[file_index].direct_fd = direct_fd;
    store->files[file_index].bytes = static_cast<uint64_t>(status.st_size);
    store->file_count++;
    if (header.max_block_bytes > store->max_block_bytes) {
        store->max_block_bytes = static_cast<uint32_t>(header.max_block_bytes);
    }
    return true;
}

}  // namespace

struct k3_mzg2_store : StoreImpl {};

extern "C" bool k3_mzg2_store_open_optional(
        k3_mzg2_store **out, char *error, size_t error_size) {
    if (out) *out = nullptr;
    if (error && error_size) error[0] = '\0';
    if (!out) {
        set_error(error, error_size, "invalid MZG2 store output");
        return false;
    }
    const char *path = getenv("MOONSHINE_MZG2_EXPERIMENT");
    if (!path || path[0] == '\0' || strcmp(path, "off") == 0) return true;
    if (path[0] != '/') {
        set_error(error, error_size,
                  "MOONSHINE_MZG2_EXPERIMENT must be off or absolute");
        return false;
    }
    struct stat root_status{};
    if (stat(path, &root_status) != 0 ||
        (!S_ISREG(root_status.st_mode) && !S_ISDIR(root_status.st_mode))) {
        set_error(error, error_size, "invalid MZG2 selection %s: %s",
                  path, strerror(errno));
        return false;
    }
    k3_mzg2_store *store = static_cast<k3_mzg2_store *>(
        calloc(1u, sizeof(*store)));
    if (!store) {
        set_error(error, error_size, "MZG2 store allocation failed");
        return false;
    }
    for (uint32_t file = 0u; file < kMaxFiles; ++file) {
        store->files[file].fd = -1;
        store->files[file].direct_fd = -1;
    }
    store->map = static_cast<MapEntry *>(
        calloc(static_cast<size_t>(kLayers) * kExperts,
               sizeof(*store->map)));
    if (!store->map) {
        k3_mzg2_store_destroy(store);
        set_error(error, error_size, "MZG2 map allocation failed");
        return false;
    }
    if (S_ISREG(root_status.st_mode)) {
        if (!load_file(store, path, error, error_size)) {
            k3_mzg2_store_destroy(store);
            return false;
        }
    } else {
        for (uint32_t layer = 1u; layer <= kLayers; ++layer) {
            char sidecar[4096];
            const int length = snprintf(
                sidecar, sizeof(sidecar), "%s/layer-%03u.mzg2",
                path, layer);
            if (length < 0 || static_cast<size_t>(length) >= sizeof(sidecar) ||
                !load_file(store, sidecar, error, error_size)) {
                k3_mzg2_store_destroy(store);
                return false;
            }
        }
        uint64_t present = 0u;
        for (uint64_t index = 0u;
             index < static_cast<uint64_t>(kLayers) * kExperts; ++index) {
            if (store->map[index].present) ++present;
        }
        if (present != static_cast<uint64_t>(kLayers) * kExperts) {
            k3_mzg2_store_destroy(store);
            set_error(error, error_size,
                      "MZG2 directory incomplete: %llu/%llu experts",
                      static_cast<unsigned long long>(present),
                      static_cast<unsigned long long>(
                          static_cast<uint64_t>(kLayers) * kExperts));
            return false;
        }
    }
    *out = store;
    return true;
}

extern "C" void k3_mzg2_store_destroy(k3_mzg2_store *store) {
    if (!store) return;
    for (uint16_t file = 0u; file < store->file_count; ++file) {
        if (store->files[file].direct_fd >= 0) {
            (void)close(store->files[file].direct_fd);
        }
        if (store->files[file].fd >= 0) {
            (void)close(store->files[file].fd);
        }
    }
    free(store->map);
    free(store);
}

extern "C" uint32_t k3_mzg2_store_max_block_bytes(
        const k3_mzg2_store *store) {
    return store ? store->max_block_bytes : 0u;
}

extern "C" bool k3_mzg2_store_span(
        const k3_mzg2_store *store,
        uint32_t layer,
        uint32_t expert,
        k3_mzg2_span *span) {
    if (span) memset(span, 0, sizeof(*span));
    if (!store || !span || layer == 0u || layer > kLayers ||
        expert >= kExperts) {
        return false;
    }
    const MapEntry &mapping = store->map[
        static_cast<uint64_t>(layer - 1u) * kExperts + expert];
    if (!mapping.present || mapping.file >= store->file_count) return false;
    span->direct_fd = store->files[mapping.file].direct_fd;
    span->offset = mapping.offset;
    span->bytes = mapping.bytes;
    return true;
}

extern "C" bool k3_mzg2_store_launch(
        k3_mzg2_store *store,
        uint32_t layer,
        uint32_t expert,
        const void *block_host,
        const void *block_device,
        uint32_t block_bytes,
        void *destination,
        uint32_t *error_device,
        void *stream_opaque,
        char *error,
        size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!store || layer == 0u || layer > kLayers || expert >= kExperts ||
        !block_host || !block_device || !destination || !error_device ||
        block_bytes < sizeof(k3_mzg2_block_header)) {
        set_error(error, error_size, "invalid MZG2 launch arguments");
        return false;
    }
    const auto *header = static_cast<const k3_mzg2_block_header *>(block_host);
    if (memcmp(header->magic, "MZ2BLOCK", 8u) != 0 ||
        header->version != K3_MZG2_VERSION ||
        header->header_bytes != sizeof(*header) ||
        header->layer != layer || header->expert != expert ||
        header->tile_bytes != 16384u ||
        header->output_bytes != K3_MZG2_EXPERT_BYTES ||
        header->tile_count != K3_MZG2_EXPERT_BYTES / 16384u ||
        header->flags != 0u || header->block_bytes != block_bytes ||
        header->model_offset < header->header_bytes ||
        header->model_offset % alignof(k3_mzg2_model_entry) != 0u ||
        header->model_bytes != sizeof(k3_mzg2_model_entry) ||
        header->model_offset > block_bytes ||
        header->model_bytes > block_bytes - header->model_offset ||
        header->descriptor_offset <
            header->model_offset + header->model_bytes ||
        header->descriptor_offset %
            alignof(k3_mzg2_tile_descriptor) != 0u ||
        header->descriptor_bytes !=
            header->tile_count * sizeof(k3_mzg2_tile_descriptor) ||
        header->descriptor_offset > block_bytes ||
        header->descriptor_bytes >
            block_bytes - header->descriptor_offset ||
        header->payload_offset <
            header->descriptor_offset + header->descriptor_bytes ||
        header->payload_offset % 4u != 0u ||
        header->payload_offset > block_bytes) {
        set_error(error, error_size,
                  "MZG2 block header mismatch layer %u expert %u",
                  layer, expert);
        return false;
    }
    const auto *host_models =
        reinterpret_cast<const k3_mzg2_model_entry *>(
            static_cast<const uint8_t *>(block_host) + header->model_offset);
    if (!validate_model_entry(*host_models)) {
        set_error(error, error_size,
                  "MZG2 model table mismatch layer %u expert %u",
                  layer, expert);
        return false;
    }

    hipStream_t stream = static_cast<hipStream_t>(stream_opaque);
    hipError_t status = hipMemsetAsync(
        error_device, 0, sizeof(*error_device), stream);
    if (status != hipSuccess) {
        set_error(error, error_size, "clear MZG2 error flag: %s",
                  hipGetErrorString(status));
        return false;
    }
    const auto *device_bytes = static_cast<const uint8_t *>(block_device);
    const auto *device_models =
        reinterpret_cast<const k3_mzg2_model_entry *>(
            device_bytes + header->model_offset);
    const auto *descriptors =
        reinterpret_cast<const k3_mzg2_tile_descriptor *>(
            device_bytes + header->descriptor_offset);
    const uint32_t blocks =
        (header->tile_count + kWavesPerBlock - 1u) / kWavesPerBlock;
    hipLaunchKernelGGL(
        decode_tiles_kernel,
        dim3(blocks), dim3(kThreads), 0, stream,
        device_bytes, block_bytes, descriptors, header->tile_count,
        device_models, static_cast<uint8_t *>(destination), error_device);
    status = hipGetLastError();
    if (status != hipSuccess) {
        set_error(error, error_size, "launch MZG2 decode: %s",
                  hipGetErrorString(status));
        return false;
    }
    return true;
}

static_assert(sizeof(k3_mzg2_file_header) == 124u,
              "MZG2 file header ABI");
static_assert(sizeof(k3_mzg2_index_entry) == 16u,
              "MZG2 index ABI");
static_assert(sizeof(k3_mzg2_block_header) == 64u,
              "MZG2 block header ABI");
static_assert(sizeof(k3_mzg2_tile_descriptor) == 32u,
              "MZG2 tile descriptor ABI");
static_assert(sizeof(k3_mzg2_model) == 4248u,
              "MZG2 model ABI");
