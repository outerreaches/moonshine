#include "k3_mzg2.h"
#include "k3_safetensors.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr uint32_t kScaleBits = 12u;
constexpr uint32_t kRansLowerBound = 1u << 16u;
constexpr uint32_t kWaveSize = 32u;
constexpr uint32_t kTileBytes = 16384u;
constexpr uint32_t kPackedBytes = 5'505'024u;
constexpr uint32_t kScaleBytes = 344'064u;

struct HostModel {
    k3_mzg2_model disk{};
    std::array<uint8_t, 256> symbol_for_value{};
};

struct Plane {
    uint32_t offset;
    uint32_t bytes;
    uint32_t model;
};

struct EncodedTile {
    k3_mzg2_tile_descriptor descriptor{};
    std::vector<uint8_t> payload;
};

struct EncodedBlock {
    std::vector<uint8_t> bytes;
    uint32_t tile_count = 0u;
    uint32_t raw_tiles = 0u;
};

uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

uint32_t word_checksum(uint32_t word, uint32_t dword_index) {
    const uint32_t rotation = dword_index & 31u;
    const uint32_t rotated = rotation
        ? (word << rotation) | (word >> (32u - rotation))
        : word;
    return rotated ^ (dword_index * UINT32_C(0x9e3779b9));
}

uint32_t tile_checksum(const uint8_t *data, uint32_t bytes) {
    uint32_t checksum = 0u;
    for (uint32_t offset = 0u; offset < bytes; offset += 4u) {
        uint32_t word;
        std::memcpy(&word, data + offset, sizeof(word));
        checksum ^= word_checksum(word, offset / 4u);
    }
    return checksum;
}

uint64_t canonicalize_packed(uint8_t *data, uint32_t bytes) {
    uint64_t changed = 0u;
    for (uint32_t index = 0u; index < bytes; ++index) {
        uint8_t low = data[index] & 0x0fu;
        uint8_t high = data[index] >> 4u;
        if (low == 8u) {
            low = 0u;
            ++changed;
        }
        if (high == 8u) {
            high = 0u;
            ++changed;
        }
        data[index] = static_cast<uint8_t>(low | (high << 4u));
    }
    return changed;
}

bool build_model(const std::array<uint64_t, 256> &counts,
                 const std::vector<uint8_t> &values,
                 HostModel *model) {
    if (!model || values.empty() || values.size() > 15u) return false;
    model->symbol_for_value.fill(0xffu);
    std::memset(&model->disk, 0, sizeof(model->disk));
    uint64_t total = 0u;
    for (uint8_t value : values) {
        if (counts[value] == 0u) return false;
        total += counts[value];
    }
    std::array<uint64_t, 15> remainder{};
    uint32_t frequency_sum = 0u;
    for (uint32_t symbol = 0u; symbol < values.size(); ++symbol) {
        const uint8_t value = values[symbol];
        uint32_t frequency = static_cast<uint32_t>(
            counts[value] * K3_MZG2_MODEL_TOTAL / total);
        if (frequency == 0u) frequency = 1u;
        model->disk.frequency[symbol] = frequency;
        remainder[symbol] = counts[value] * K3_MZG2_MODEL_TOTAL % total;
        frequency_sum += frequency;
        model->disk.values[symbol] = value;
        model->symbol_for_value[value] = static_cast<uint8_t>(symbol);
    }
    while (frequency_sum < K3_MZG2_MODEL_TOTAL) {
        uint32_t best = 0u;
        for (uint32_t symbol = 1u; symbol < values.size(); ++symbol) {
            if (remainder[symbol] > remainder[best]) best = symbol;
        }
        ++model->disk.frequency[best];
        remainder[best] = 0u;
        ++frequency_sum;
    }
    while (frequency_sum > K3_MZG2_MODEL_TOTAL) {
        uint32_t best = UINT32_MAX;
        for (uint32_t symbol = 0u; symbol < values.size(); ++symbol) {
            if (model->disk.frequency[symbol] > 1u &&
                (best == UINT32_MAX ||
                 remainder[symbol] < remainder[best])) {
                best = symbol;
            }
        }
        if (best == UINT32_MAX) return false;
        --model->disk.frequency[best];
        --frequency_sum;
    }
    if (frequency_sum != K3_MZG2_MODEL_TOTAL) return false;
    uint32_t cumulative = 0u;
    for (uint32_t symbol = 0u; symbol < values.size(); ++symbol) {
        model->disk.cumulative[symbol] = cumulative;
        const uint32_t end = cumulative + model->disk.frequency[symbol];
        if (end > K3_MZG2_MODEL_TOTAL) return false;
        std::fill(
            model->disk.decode + cumulative,
            model->disk.decode + end,
            static_cast<uint8_t>(symbol));
        cumulative = end;
    }
    model->disk.symbol_count = static_cast<uint8_t>(values.size());
    return cumulative == K3_MZG2_MODEL_TOTAL;
}

bool validate_model(const k3_mzg2_model &model, bool packed) {
    if (model.symbol_count == 0u || model.symbol_count > 15u) return false;
    bool values[256]{};
    uint32_t cumulative = 0u;
    for (uint32_t symbol = 0u; symbol < model.symbol_count; ++symbol) {
        const uint8_t value = model.values[symbol];
        if (values[value] ||
            (packed && (value > 15u || value == 8u)) ||
            model.frequency[symbol] == 0u ||
            model.cumulative[symbol] != cumulative ||
            model.frequency[symbol] >
                K3_MZG2_MODEL_TOTAL - cumulative) {
            return false;
        }
        values[value] = true;
        cumulative += model.frequency[symbol];
    }
    if (cumulative != K3_MZG2_MODEL_TOTAL) return false;
    for (uint32_t slot = 0u; slot < K3_MZG2_MODEL_TOTAL; ++slot) {
        const uint32_t symbol = model.decode[slot];
        if (symbol >= model.symbol_count ||
            slot < model.cumulative[symbol] ||
            slot >= model.cumulative[symbol] +
                model.frequency[symbol]) {
            return false;
        }
    }
    return true;
}

uint8_t tile_value(const uint8_t *data,
                   uint32_t model,
                   uint32_t round,
                   uint32_t lane) {
    if (model == K3_MZG2_MODEL_PACKED) {
        const uint32_t dword = round / 8u;
        const uint32_t nibble = round % 8u;
        const uint32_t byte_index =
            (dword * kWaveSize + lane) * 4u + nibble / 2u;
        return static_cast<uint8_t>(
            (data[byte_index] >> ((nibble & 1u) * 4u)) & 0x0fu);
    }
    const uint32_t dword = round / 4u;
    const uint32_t byte = round % 4u;
    return data[(dword * kWaveSize + lane) * 4u + byte];
}

bool encode_tile(const uint8_t *data,
                 uint32_t output_bytes,
                 uint32_t model_index,
                 const HostModel &model,
                 EncodedTile *tile) {
    if (!data || !tile || output_bytes == 0u || output_bytes % 128u) {
        return false;
    }
    std::array<uint32_t, kWaveSize> states{};
    states.fill(kRansLowerBound);
    std::vector<uint16_t> words;
    const uint32_t symbols_per_byte =
        model_index == K3_MZG2_MODEL_PACKED ? 2u : 1u;
    const uint32_t rounds = output_bytes * symbols_per_byte / kWaveSize;
    words.reserve(output_bytes / 2u);
    for (uint32_t reverse_round = rounds; reverse_round > 0u; --reverse_round) {
        const uint32_t round = reverse_round - 1u;
        for (uint32_t reverse_lane = kWaveSize;
             reverse_lane > 0u; --reverse_lane) {
            const uint32_t lane = reverse_lane - 1u;
            const uint8_t value = tile_value(data, model_index, round, lane);
            const uint8_t symbol = model.symbol_for_value[value];
            if (symbol == 0xffu || symbol >= model.disk.symbol_count) {
                return false;
            }
            const uint32_t frequency = model.disk.frequency[symbol];
            const uint32_t cumulative = model.disk.cumulative[symbol];
            uint32_t state = states[lane];
            const uint64_t maximum = static_cast<uint64_t>(frequency) << 20u;
            while (state >= maximum) {
                words.push_back(static_cast<uint16_t>(state));
                state >>= 16u;
            }
            const uint64_t encoded =
                (static_cast<uint64_t>(state / frequency) << kScaleBits)
                + state % frequency + cumulative;
            if (encoded > UINT32_MAX) return false;
            states[lane] = static_cast<uint32_t>(encoded);
        }
    }
    std::reverse(words.begin(), words.end());
    const uint32_t encoded_bytes =
        kWaveSize * sizeof(uint32_t)
        + static_cast<uint32_t>(words.size() * sizeof(uint16_t));
    tile->descriptor.output_bytes = output_bytes;
    tile->descriptor.kind = model_index;
    tile->descriptor.checksum = tile_checksum(data, output_bytes);
    if (encoded_bytes >= output_bytes) {
        tile->descriptor.kind |= K3_MZG2_TILE_RAW;
        tile->payload.assign(data, data + output_bytes);
    } else {
        tile->payload.resize(encoded_bytes);
        std::memcpy(tile->payload.data(), states.data(),
                    kWaveSize * sizeof(uint32_t));
        std::memcpy(
            tile->payload.data() + kWaveSize * sizeof(uint32_t),
            words.data(), words.size() * sizeof(uint16_t));
    }
    return true;
}

bool encode_block(const std::vector<uint8_t> &expert,
                  const std::array<Plane, 6> &planes,
                  uint32_t layer,
                  uint32_t expert_index,
                  const HostModel models[2],
                  const k3_mzg2_model_entry &model_entry,
                  EncodedBlock *block) {
    std::vector<EncodedTile> tiles;
    for (const Plane &plane : planes) {
        for (uint32_t position = 0u; position < plane.bytes;
             position += kTileBytes) {
            const uint32_t bytes =
                std::min(kTileBytes, plane.bytes - position);
            EncodedTile tile;
            tile.descriptor.output_offset = plane.offset + position;
            if (!encode_tile(
                    expert.data() + plane.offset + position,
                    bytes, plane.model, models[plane.model], &tile)) {
                return false;
            }
            if (tile.descriptor.kind & K3_MZG2_TILE_RAW) ++block->raw_tiles;
            tiles.push_back(std::move(tile));
        }
    }
    const uint32_t model_offset = sizeof(k3_mzg2_block_header);
    const uint32_t model_bytes = sizeof(k3_mzg2_model_entry);
    const uint32_t descriptor_offset = model_offset + model_bytes;
    const uint32_t descriptor_bytes = static_cast<uint32_t>(
        tiles.size() * sizeof(k3_mzg2_tile_descriptor));
    uint32_t cursor = static_cast<uint32_t>(align_up(
        descriptor_offset + descriptor_bytes, K3_MZG2_ALIGNMENT));
    for (EncodedTile &tile : tiles) {
        cursor = static_cast<uint32_t>(align_up(cursor, 4u));
        tile.descriptor.payload_offset = cursor;
        tile.descriptor.payload_bytes = static_cast<uint32_t>(tile.payload.size());
        if (UINT32_MAX - cursor < tile.payload.size()) return false;
        cursor += static_cast<uint32_t>(tile.payload.size());
    }
    const uint32_t block_bytes = static_cast<uint32_t>(
        align_up(cursor, K3_MZG2_ALIGNMENT));
    block->bytes.assign(block_bytes, 0u);
    block->tile_count = static_cast<uint32_t>(tiles.size());
    auto *header = reinterpret_cast<k3_mzg2_block_header *>(block->bytes.data());
    std::memcpy(header->magic, "MZ2BLOCK", 8u);
    header->version = K3_MZG2_VERSION;
    header->header_bytes = sizeof(*header);
    header->layer = layer;
    header->expert = expert_index;
    header->tile_bytes = kTileBytes;
    header->tile_count = block->tile_count;
    header->output_bytes = K3_MZG2_EXPERT_BYTES;
    header->descriptor_offset = descriptor_offset;
    header->descriptor_bytes = descriptor_bytes;
    header->payload_offset = static_cast<uint32_t>(align_up(
        descriptor_offset + descriptor_bytes, K3_MZG2_ALIGNMENT));
    header->block_bytes = block_bytes;
    header->model_offset = model_offset;
    header->model_bytes = model_bytes;
    std::memcpy(
        block->bytes.data() + model_offset,
        &model_entry, sizeof(model_entry));
    auto *descriptors = reinterpret_cast<k3_mzg2_tile_descriptor *>(
        block->bytes.data() + descriptor_offset);
    for (uint32_t index = 0u; index < tiles.size(); ++index) {
        descriptors[index] = tiles[index].descriptor;
        std::memcpy(
            block->bytes.data() + tiles[index].descriptor.payload_offset,
            tiles[index].payload.data(), tiles[index].payload.size());
    }
    return true;
}

bool decode_block(const EncodedBlock &block,
                  const k3_mzg2_model_entry &models,
                  std::vector<uint8_t> *output) {
    if (block.bytes.size() < sizeof(k3_mzg2_block_header) ||
        !validate_model(models.model[0], true) ||
        !validate_model(models.model[1], false)) {
        return false;
    }
    const auto *header = reinterpret_cast<const k3_mzg2_block_header *>(
        block.bytes.data());
    if (std::memcmp(header->magic, "MZ2BLOCK", 8u) != 0 ||
        header->version != K3_MZG2_VERSION ||
        header->header_bytes != sizeof(*header) ||
        header->tile_bytes != kTileBytes ||
        header->tile_count != K3_MZG2_EXPERT_BYTES / kTileBytes ||
        header->output_bytes != K3_MZG2_EXPERT_BYTES ||
        header->model_offset < header->header_bytes ||
        header->model_bytes != sizeof(k3_mzg2_model_entry) ||
        header->model_offset > block.bytes.size() ||
        header->model_bytes >
            block.bytes.size() - header->model_offset ||
        header->descriptor_offset <
            header->model_offset + header->model_bytes ||
        header->descriptor_bytes !=
            header->tile_count * sizeof(k3_mzg2_tile_descriptor) ||
        header->descriptor_offset > block.bytes.size() ||
        header->descriptor_bytes >
            block.bytes.size() - header->descriptor_offset ||
        header->payload_offset <
            header->descriptor_offset + header->descriptor_bytes ||
        header->payload_offset > block.bytes.size() ||
        header->block_bytes != block.bytes.size()) {
        return false;
    }
    const auto *descriptors =
        reinterpret_cast<const k3_mzg2_tile_descriptor *>(
            block.bytes.data() + header->descriptor_offset);
    output->assign(K3_MZG2_EXPERT_BYTES, 0u);
    for (uint32_t tile = 0u; tile < header->tile_count; ++tile) {
        const auto &descriptor = descriptors[tile];
        const uint32_t model_index = descriptor.kind & K3_MZG2_MODEL_MASK;
        if (model_index > K3_MZG2_MODEL_SCALE ||
            (descriptor.kind &
             ~(K3_MZG2_TILE_RAW | K3_MZG2_MODEL_MASK)) != 0u ||
            descriptor.reserved != 0u ||
            descriptor.output_bytes != kTileBytes ||
            descriptor.output_offset != tile * kTileBytes ||
            descriptor.output_offset > output->size() ||
            descriptor.output_bytes >
                output->size() - descriptor.output_offset ||
            descriptor.payload_offset > block.bytes.size() ||
            descriptor.payload_bytes >
                block.bytes.size() - descriptor.payload_offset) {
            return false;
        }
        const uint8_t *payload =
            block.bytes.data() + descriptor.payload_offset;
        uint8_t *destination = output->data() + descriptor.output_offset;
        if (descriptor.kind & K3_MZG2_TILE_RAW) {
            if (descriptor.payload_bytes != descriptor.output_bytes) return false;
            std::memcpy(destination, payload, descriptor.output_bytes);
        } else {
            if (descriptor.payload_bytes < kWaveSize * sizeof(uint32_t) ||
                (descriptor.payload_bytes - kWaveSize * sizeof(uint32_t)) % 2u) {
                return false;
            }
            std::array<uint32_t, kWaveSize> states{};
            std::memcpy(states.data(), payload, kWaveSize * sizeof(uint32_t));
            const uint16_t *words = reinterpret_cast<const uint16_t *>(
                payload + kWaveSize * sizeof(uint32_t));
            const uint32_t word_count =
                (descriptor.payload_bytes - kWaveSize * sizeof(uint32_t)) / 2u;
            uint32_t cursor = 0u;
            const auto &model = models.model[model_index];
            const uint32_t symbols_per_byte =
                model_index == K3_MZG2_MODEL_PACKED ? 2u : 1u;
            const uint32_t rounds =
                descriptor.output_bytes * symbols_per_byte / kWaveSize;
            for (uint32_t round = 0u; round < rounds; ++round) {
                for (uint32_t lane = 0u; lane < kWaveSize; ++lane) {
                    uint32_t state = states[lane];
                    if (state < kRansLowerBound) return false;
                    const uint32_t slot = state & (K3_MZG2_MODEL_TOTAL - 1u);
                    const uint32_t symbol = model.decode[slot];
                    if (symbol >= model.symbol_count) return false;
                    state = model.frequency[symbol] * (state >> kScaleBits)
                        + slot - model.cumulative[symbol];
                    if (state < kRansLowerBound) {
                        if (cursor >= word_count) return false;
                        state = (state << 16u) | words[cursor++];
                    }
                    states[lane] = state;
                    const uint8_t value = model.values[symbol];
                    if (model_index == K3_MZG2_MODEL_PACKED) {
                        const uint32_t dword = round / 8u;
                        const uint32_t nibble = round % 8u;
                        const uint32_t byte_index =
                            (dword * kWaveSize + lane) * 4u + nibble / 2u;
                        destination[byte_index] |=
                            static_cast<uint8_t>(value << ((nibble & 1u) * 4u));
                    } else {
                        const uint32_t dword = round / 4u;
                        const uint32_t byte = round % 4u;
                        destination[(dword * kWaveSize + lane) * 4u + byte] = value;
                    }
                }
            }
            if (cursor != word_count ||
                std::any_of(states.begin(), states.end(),
                            [](uint32_t state) {
                                return state != kRansLowerBound;
                            })) {
                return false;
            }
        }
        if (tile_checksum(destination, descriptor.output_bytes) !=
            static_cast<uint32_t>(descriptor.checksum)) {
            return false;
        }
    }
    return true;
}

bool parse_digest(const char *text, uint8_t digest[32]) {
    if (!text || std::strlen(text) != 64u) return false;
    for (uint32_t index = 0u; index < 32u; ++index) {
        unsigned value = 0u;
        if (std::sscanf(text + index * 2u, "%2x", &value) != 1) return false;
        digest[index] = static_cast<uint8_t>(value);
    }
    return true;
}

bool write_full(FILE *file, const void *data, size_t bytes) {
    return std::fwrite(data, 1u, bytes, file) == bytes;
}

bool pread_full_fd(int fd, void *buffer, size_t bytes, uint64_t offset) {
    uint8_t *cursor = static_cast<uint8_t *>(buffer);
    while (bytes != 0u) {
        const ssize_t result =
            pread(fd, cursor, bytes, static_cast<off_t>(offset));
        if (result < 0 && errno == EINTR) continue;
        if (result <= 0) return false;
        cursor += static_cast<size_t>(result);
        bytes -= static_cast<size_t>(result);
        offset += static_cast<uint64_t>(result);
    }
    return true;
}

int verify_layer_store(const char *model_root, const char *store_path) {
    int fd = open(store_path, O_RDONLY | O_CLOEXEC);
    struct stat status{};
    k3_mzg2_file_header file_header{};
    if (fd < 0 || fstat(fd, &status) != 0 ||
        !pread_full_fd(fd, &file_header, sizeof(file_header), 0u) ||
        std::memcmp(file_header.magic, "K3MZG2\0", 8u) != 0 ||
        file_header.version != K3_MZG2_VERSION ||
        file_header.header_bytes != K3_MZG2_FILE_HEADER_BYTES ||
        file_header.alignment != K3_MZG2_ALIGNMENT ||
        file_header.layer == 0u || file_header.layer > 92u ||
        file_header.expert_count == 0u ||
        file_header.expert_count > 896u ||
        file_header.index_entry_bytes !=
            sizeof(k3_mzg2_index_entry) ||
        file_header.model_entry_bytes != 0u ||
        file_header.index_offset != K3_MZG2_FILE_HEADER_BYTES ||
        file_header.index_bytes !=
            static_cast<uint64_t>(file_header.expert_count) *
                sizeof(k3_mzg2_index_entry) ||
        file_header.model_offset != 0u ||
        file_header.model_bytes != 0u ||
        file_header.data_offset <
            file_header.index_offset + file_header.index_bytes ||
        file_header.data_offset % K3_MZG2_ALIGNMENT != 0u ||
        file_header.max_block_bytes == 0u ||
        file_header.max_block_bytes > UINT32_MAX ||
        file_header.data_offset >
            static_cast<uint64_t>(status.st_size)) {
        std::fprintf(stderr, "invalid MZG2 layer store %s\n", store_path);
        if (fd >= 0) close(fd);
        return 1;
    }
    std::vector<k3_mzg2_index_entry> index(file_header.expert_count);
    if (!pread_full_fd(
            fd, index.data(), static_cast<size_t>(file_header.index_bytes),
            file_header.index_offset)) {
        std::fprintf(stderr, "read MZG2 index %s\n", store_path);
        close(fd);
        return 1;
    }
    char error[512];
    k3_st_model model;
    if (!k3_st_model_open(&model, model_root, 96u, error, sizeof(error))) {
        std::fprintf(stderr, "open source model: %s\n", error);
        close(fd);
        return 1;
    }
    static const char *suffix[6] = {
        "w1.weight_packed", "w1.weight_scale",
        "w2.weight_packed", "w2.weight_scale",
        "w3.weight_packed", "w3.weight_scale",
    };
    uint64_t verified_bytes = 0u;
    for (uint32_t expert = 0u;
         expert < file_header.expert_count; ++expert) {
        const k3_mzg2_index_entry &entry = index[expert];
        if (entry.layer != file_header.layer ||
            entry.expert != expert ||
            entry.block_bytes == 0u ||
            entry.block_bytes % K3_MZG2_ALIGNMENT != 0u ||
            entry.block_bytes > file_header.max_block_bytes ||
            entry.block_offset < file_header.data_offset ||
            entry.block_offset % K3_MZG2_ALIGNMENT != 0u ||
            entry.block_offset > static_cast<uint64_t>(status.st_size) ||
            entry.block_bytes >
                static_cast<uint64_t>(status.st_size) -
                    entry.block_offset) {
            std::fprintf(stderr, "invalid MZG2 index expert %u\n", expert);
            return 1;
        }
        EncodedBlock block;
        block.bytes.resize(entry.block_bytes);
        if (!pread_full_fd(
                fd, block.bytes.data(), block.bytes.size(),
                entry.block_offset)) {
            std::fprintf(stderr, "read MZG2 expert %u\n", expert);
            return 1;
        }
        const auto *block_header =
            reinterpret_cast<const k3_mzg2_block_header *>(
                block.bytes.data());
        if (block_header->layer != file_header.layer ||
            block_header->expert != expert ||
            block_header->model_offset > block.bytes.size() ||
            block_header->model_bytes !=
                sizeof(k3_mzg2_model_entry) ||
            block_header->model_bytes >
                block.bytes.size() - block_header->model_offset) {
            std::fprintf(stderr, "invalid MZG2 block expert %u\n", expert);
            return 1;
        }
        const auto *models =
            reinterpret_cast<const k3_mzg2_model_entry *>(
                block.bytes.data() + block_header->model_offset);
        std::vector<uint8_t> decoded;
        if (!decode_block(block, *models, &decoded)) {
            std::fprintf(stderr, "decode MZG2 expert %u failed\n", expert);
            return 1;
        }

        const k3_st_tensor *tensor[6]{};
        char name[256];
        for (uint32_t item = 0u; item < 6u; ++item) {
            std::snprintf(
                name, sizeof(name),
                "language_model.model.layers.%u.block_sparse_moe."
                "experts.%u.%s",
                file_header.layer, expert, suffix[item]);
            tensor[item] = k3_st_find(&model, name);
            if (!tensor[item]) {
                std::fprintf(stderr, "missing source %s\n", name);
                return 1;
            }
        }
        const uint64_t physical_start = tensor[0]->physical_offset;
        const uint64_t physical_end =
            tensor[5]->physical_offset + tensor[5]->byte_length;
        if (physical_end - physical_start != K3_MZG2_EXPERT_BYTES) {
            std::fprintf(stderr, "source span mismatch expert %u\n", expert);
            return 1;
        }
        k3_st_read source;
        if (!k3_st_read_span(
                &model, tensor[0]->shard, physical_start,
                K3_MZG2_EXPERT_BYTES, K3_MZG2_ALIGNMENT,
                &source, error, sizeof(error))) {
            std::fprintf(stderr, "read source expert %u: %s\n",
                         expert, error);
            return 1;
        }
        std::vector<uint8_t> canonical(
            source.data, source.data + K3_MZG2_EXPERT_BYTES);
        for (uint32_t item = 0u; item < 6u; item += 2u) {
            canonicalize_packed(
                canonical.data() +
                    (tensor[item]->physical_offset - physical_start),
                static_cast<uint32_t>(tensor[item]->byte_length));
        }
        k3_st_read_release(&source);
        if (decoded != canonical) {
            std::fprintf(stderr, "source mismatch expert %u\n", expert);
            return 1;
        }
        verified_bytes += entry.block_bytes;
        if ((expert + 1u) % 64u == 0u ||
            expert + 1u == file_header.expert_count) {
            std::printf(
                "MZG2 verify layer=%u experts=%u/%u\n",
                file_header.layer, expert + 1u,
                file_header.expert_count);
            std::fflush(stdout);
        }
    }
    k3_st_model_close(&model);
    close(fd);
    std::printf(
        "MZG2 verify: PASS layer=%u experts=%u stored=%llu input=%s\n",
        file_header.layer, file_header.expert_count,
        static_cast<unsigned long long>(verified_bytes), store_path);
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc == 4 && std::strcmp(argv[1], "--verify") == 0) {
        return verify_layer_store(argv[2], argv[3]);
    }
    const char *model_root = argc > 1 ? argv[1]
        : "/srv/modelstore/models/moonshotai__Kimi-K3";
    const char *output_path = argc > 2 ? argv[2] : "layer-010.mzg2.partial";
    const uint32_t layer = argc > 3
        ? static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10)) : 10u;
    const uint32_t expert_count = argc > 4
        ? static_cast<uint32_t>(std::strtoul(argv[4], nullptr, 10)) : 896u;
    const char *manifest_digest = argc > 5 ? argv[5]
        : "476fa0ba64e3233cbb9ca0642327361a73f6807e751edb071c92fa2216b202a4";
    if (layer == 0u || layer > 92u || expert_count == 0u ||
        expert_count > 896u) {
        std::fprintf(stderr, "invalid layer/expert count\n");
        return 1;
    }

    char error[512];
    k3_st_model model;
    if (!k3_st_model_open(&model, model_root, 96u, error, sizeof(error))) {
        std::fprintf(stderr, "open source model: %s\n", error);
        return 1;
    }
    std::vector<k3_mzg2_index_entry> index(expert_count);
    k3_mzg2_file_header header{};
    std::memcpy(header.magic, "K3MZG2\0", 8u);
    header.version = K3_MZG2_VERSION;
    header.header_bytes = K3_MZG2_FILE_HEADER_BYTES;
    header.alignment = K3_MZG2_ALIGNMENT;
    header.layer = layer;
    header.expert_count = expert_count;
    header.index_entry_bytes = sizeof(k3_mzg2_index_entry);
    header.model_entry_bytes = 0u;
    header.tile_bytes = kTileBytes;
    header.index_offset = K3_MZG2_FILE_HEADER_BYTES;
    header.index_bytes =
        static_cast<uint64_t>(expert_count) * sizeof(k3_mzg2_index_entry);
    header.model_offset = 0u;
    header.model_bytes = 0u;
    header.data_offset = align_up(
        header.index_offset + header.index_bytes, K3_MZG2_ALIGNMENT);
    if (!parse_digest(manifest_digest, header.source_manifest_sha256)) {
        std::fprintf(stderr, "invalid source manifest digest\n");
        k3_st_model_close(&model);
        return 1;
    }

    FILE *file = std::fopen(output_path, "w+b");
    if (!file || ftruncate(fileno(file), static_cast<off_t>(header.data_offset)) != 0 ||
        fseeko(file, static_cast<off_t>(header.data_offset), SEEK_SET) != 0) {
        std::fprintf(stderr, "create %s: %s\n", output_path, strerror(errno));
        if (file) std::fclose(file);
        k3_st_model_close(&model);
        return 1;
    }

    static const char *suffix[6] = {
        "w1.weight_packed", "w1.weight_scale",
        "w2.weight_packed", "w2.weight_scale",
        "w3.weight_packed", "w3.weight_scale",
    };
    uint64_t raw_bytes = 0u;
    uint64_t stored_bytes = 0u;
    uint64_t negative_zero_codes = 0u;
    uint64_t raw_tiles = 0u;
    uint64_t cursor = header.data_offset;
    for (uint32_t expert = 0u; expert < expert_count; ++expert) {
        const k3_st_tensor *tensor[6]{};
        char name[256];
        for (uint32_t item = 0u; item < 6u; ++item) {
            std::snprintf(
                name, sizeof(name),
                "language_model.model.layers.%u.block_sparse_moe.experts.%u.%s",
                layer, expert, suffix[item]);
            tensor[item] = k3_st_find(&model, name);
            if (!tensor[item]) {
                std::fprintf(stderr, "missing %s\n", name);
                return 1;
            }
        }
        const uint64_t physical_start = tensor[0]->physical_offset;
        const uint64_t physical_end =
            tensor[5]->physical_offset + tensor[5]->byte_length;
        if (physical_end - physical_start != K3_MZG2_EXPERT_BYTES) {
            std::fprintf(stderr, "expert %u span mismatch\n", expert);
            return 1;
        }
        k3_st_read source;
        if (!k3_st_read_span(
                &model, tensor[0]->shard, physical_start,
                K3_MZG2_EXPERT_BYTES, K3_MZG2_ALIGNMENT,
                &source, error, sizeof(error))) {
            std::fprintf(stderr, "read expert %u: %s\n", expert, error);
            return 1;
        }
        std::vector<uint8_t> canonical(
            source.data, source.data + K3_MZG2_EXPERT_BYTES);
        std::array<Plane, 6> planes{};
        for (uint32_t item = 0u; item < 6u; ++item) {
            planes[item].offset = static_cast<uint32_t>(
                tensor[item]->physical_offset - physical_start);
            planes[item].bytes = static_cast<uint32_t>(tensor[item]->byte_length);
            planes[item].model = (item & 1u)
                ? K3_MZG2_MODEL_SCALE : K3_MZG2_MODEL_PACKED;
            if (planes[item].model == K3_MZG2_MODEL_PACKED) {
                negative_zero_codes += canonicalize_packed(
                    canonical.data() + planes[item].offset,
                    planes[item].bytes);
            }
        }
        if (planes[0].bytes != kPackedBytes || planes[1].bytes != kScaleBytes) {
            std::fprintf(stderr, "expert %u plane geometry mismatch\n", expert);
            return 1;
        }

        std::array<uint64_t, 256> counts[2]{};
        for (const Plane &plane : planes) {
            const uint8_t *data = canonical.data() + plane.offset;
            if (plane.model == K3_MZG2_MODEL_PACKED) {
                for (uint32_t byte = 0u; byte < plane.bytes; ++byte) {
                    ++counts[0][data[byte] & 0x0fu];
                    ++counts[0][data[byte] >> 4u];
                }
            } else {
                for (uint32_t byte = 0u; byte < plane.bytes; ++byte) {
                    ++counts[1][data[byte]];
                }
            }
        }
        std::vector<uint8_t> values[2];
        for (uint32_t value = 0u; value < 16u; ++value) {
            if (value != 8u && counts[0][value]) {
                values[0].push_back(static_cast<uint8_t>(value));
            }
        }
        for (uint32_t value = 0u; value < 256u; ++value) {
            if (counts[1][value]) values[1].push_back(static_cast<uint8_t>(value));
        }
        HostModel host_models[2];
        if (values[0].size() != 15u || values[1].size() > 15u ||
            !build_model(counts[0], values[0], &host_models[0]) ||
            !build_model(counts[1], values[1], &host_models[1])) {
            std::fprintf(stderr, "expert %u model build failed scales=%zu\n",
                         expert, values[1].size());
            return 1;
        }
        k3_mzg2_model_entry model_entry{};
        model_entry.model[0] = host_models[0].disk;
        model_entry.model[1] = host_models[1].disk;
        EncodedBlock block;
        if (!encode_block(
                canonical, planes, layer, expert, host_models,
                model_entry, &block)) {
            std::fprintf(stderr, "expert %u encode failed\n", expert);
            return 1;
        }
        std::vector<uint8_t> decoded;
        if (!decode_block(block, model_entry, &decoded) ||
            decoded != canonical) {
            std::fprintf(stderr, "expert %u roundtrip failed\n", expert);
            return 1;
        }
        index[expert].layer = (uint16_t)layer;
        index[expert].expert = (uint16_t)expert;
        index[expert].block_bytes = static_cast<uint32_t>(block.bytes.size());
        index[expert].block_offset = cursor;
        if (!write_full(file, block.bytes.data(), block.bytes.size())) {
            std::fprintf(stderr, "write expert %u: %s\n", expert, strerror(errno));
            return 1;
        }
        cursor += block.bytes.size();
        raw_bytes += K3_MZG2_EXPERT_BYTES;
        stored_bytes += block.bytes.size();
        raw_tiles += block.raw_tiles;
        header.max_block_bytes = std::max<uint64_t>(
            header.max_block_bytes, block.bytes.size());
        k3_st_read_release(&source);
        if ((expert + 1u) % 16u == 0u || expert + 1u == expert_count) {
            std::printf(
                "MZG2 layer=%u experts=%u/%u stored=%.3fGiB reduction=%.3f%%\n",
                layer, expert + 1u, expert_count,
                static_cast<double>(stored_bytes) / 1073741824.0,
                100.0 * (1.0 - static_cast<double>(stored_bytes) / raw_bytes));
            std::fflush(stdout);
        }
    }

    if (fseeko(file, 0, SEEK_SET) != 0 ||
        !write_full(file, &header, sizeof(header)) ||
        fseeko(file, static_cast<off_t>(header.index_offset), SEEK_SET) != 0 ||
        !write_full(file, index.data(), index.size() * sizeof(index[0])) ||
        std::fflush(file) != 0 || fsync(fileno(file)) != 0 ||
        std::fclose(file) != 0) {
        std::fprintf(stderr, "finalize %s: %s\n", output_path, strerror(errno));
        return 1;
    }
    k3_st_model_close(&model);
    std::printf(
        "MZG2 transcode: PASS layer=%u experts=%u raw=%llu stored=%llu "
        "reduction=%.9f%% max_block=%llu raw_tiles=%llu negative_zero=%llu "
        "output=%s\n",
        layer, expert_count,
        static_cast<unsigned long long>(raw_bytes),
        static_cast<unsigned long long>(stored_bytes),
        100.0 * (1.0 - static_cast<double>(stored_bytes) / raw_bytes),
        static_cast<unsigned long long>(header.max_block_bytes),
        static_cast<unsigned long long>(raw_tiles),
        static_cast<unsigned long long>(negative_zero_codes), output_path);
    return 0;
}
