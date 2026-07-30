enum {
    K3_STATE_FORMAT_VERSION = 1,
    K3_STATE_HEADER_BYTES = 256,
    K3_STATE_ENDIAN_MARKER = 0x01020304,
    K3_STATE_FLAG_Q8_PROJECTIONS = 1,
    K3_STATE_TRANSFER_BYTES = 16 * 1024 * 1024,
};

enum {
    K3_STATE_OFF_MAGIC = 0,
    K3_STATE_OFF_VERSION = 8,
    K3_STATE_OFF_HEADER_BYTES = 12,
    K3_STATE_OFF_ENDIAN = 16,
    K3_STATE_OFF_FLAGS = 20,
    K3_STATE_OFF_CONTEXT = 24,
    K3_STATE_OFF_TOKEN_POSITION = 28,
    K3_STATE_OFF_LAYERS = 32,
    K3_STATE_OFF_MLA_LAYERS = 36,
    K3_STATE_OFF_MODEL_LAYOUT_CRC = 40,
    K3_STATE_OFF_PAYLOAD_BYTES = 48,
    K3_STATE_OFF_PAYLOAD_CRC = 56,
    K3_STATE_OFF_KDA_STATE_BYTES = 64,
    K3_STATE_OFF_KDA_CONV_BYTES = 72,
    K3_STATE_OFF_MLA_BYTES = 80,
    K3_STATE_OFF_ATTN_RES_BYTES = 88,
    K3_STATE_OFF_HEADER_CRC = 104,
};

static const uint8_t K3_STATE_MAGIC[8] = {
    'K', '3', 'S', 'T', 'A', 'T', 'E', '\0',
};

typedef struct {
    uint32_t flags;
    uint32_t context;
    uint32_t token_position;
    uint64_t model_layout_crc64;
    uint64_t payload_bytes;
    uint64_t payload_crc64;
    uint64_t kda_state_bytes;
    uint64_t kda_conv_bytes;
    uint64_t mla_bytes;
    uint64_t attn_res_bytes;
    uint64_t file_bytes;
} k3_state_metadata;

typedef struct {
    int fd;
    uint8_t *host;
    uint64_t crc_table[256];
    uint64_t payload_crc64;
} k3_state_writer;

typedef struct {
    int fd;
    uint8_t *host;
    uint64_t offset;
} k3_state_reader;

static void state_put_u16(uint8_t *destination,
                          uint16_t value) {
    destination[0] = (uint8_t)value;
    destination[1] = (uint8_t)(value >> 8);
}

static void state_put_u32(uint8_t *destination,
                          uint32_t value) {
    for (uint32_t index = 0u; index < 4u; index++) {
        destination[index] =
            (uint8_t)(value >> (index * 8u));
    }
}

static void state_put_u64(uint8_t *destination,
                          uint64_t value) {
    for (uint32_t index = 0u; index < 8u; index++) {
        destination[index] =
            (uint8_t)(value >> (index * 8u));
    }
}

static uint32_t state_get_u32(const uint8_t *source) {
    uint32_t value = 0u;
    for (uint32_t index = 0u; index < 4u; index++) {
        value |= (uint32_t)source[index] << (index * 8u);
    }
    return value;
}

static uint64_t state_get_u64(const uint8_t *source) {
    uint64_t value = 0u;
    for (uint32_t index = 0u; index < 8u; index++) {
        value |= (uint64_t)source[index] << (index * 8u);
    }
    return value;
}

static void state_crc64_table(uint64_t table[256]) {
    const uint64_t polynomial =
        UINT64_C(0x42f0e1eba9ea3693);
    for (uint32_t value = 0u; value < 256u; value++) {
        uint64_t crc = (uint64_t)value << 56u;
        for (uint32_t bit = 0u; bit < 8u; bit++) {
            crc = crc & (UINT64_C(1) << 63u) ?
                (crc << 1u) ^ polynomial : crc << 1u;
        }
        table[value] = crc;
    }
}

static uint64_t state_crc64_update(
        const uint64_t table[256],
        uint64_t crc,
        const void *data,
        uint64_t bytes) {
    const uint8_t *source = (const uint8_t *)data;
    for (uint64_t index = 0u; index < bytes; index++) {
        const uint8_t slot =
            (uint8_t)((crc >> 56u) ^ source[index]);
        crc = table[slot] ^ (crc << 8u);
    }
    return crc;
}

static void state_crc_u16(
        const uint64_t table[256],
        uint64_t *crc,
        uint16_t value) {
    uint8_t encoded[2];
    state_put_u16(encoded, value);
    *crc = state_crc64_update(
        table, *crc, encoded, sizeof(encoded));
}

static void state_crc_u32(
        const uint64_t table[256],
        uint64_t *crc,
        uint32_t value) {
    uint8_t encoded[4];
    state_put_u32(encoded, value);
    *crc = state_crc64_update(
        table, *crc, encoded, sizeof(encoded));
}

static void state_crc_u64(
        const uint64_t table[256],
        uint64_t *crc,
        uint64_t value) {
    uint8_t encoded[8];
    state_put_u64(encoded, value);
    *crc = state_crc64_update(
        table, *crc, encoded, sizeof(encoded));
}

/*
 * Paths are intentionally excluded: a byte-identical model tree may move
 * between hosts. Shard sizes/header boundaries and the complete sorted tensor
 * directory define the state ABI identity.
 */
static uint64_t engine_model_layout_crc64(
        const k3_st_model *model) {
    uint64_t table[256];
    state_crc64_table(table);
    uint64_t crc = 0u;
    state_crc_u64(table, &crc, model->shard_count);
    for (size_t index = 0u;
         index < model->shard_count; index++) {
        state_crc_u64(
            table, &crc, model->shards[index].file_bytes);
        state_crc_u64(
            table, &crc, model->shards[index].data_offset);
    }
    state_crc_u64(table, &crc, model->tensor_count);
    for (size_t index = 0u;
         index < model->tensor_count; index++) {
        const k3_st_tensor *tensor = &model->tensors[index];
        const size_t name_bytes = strlen(tensor->name);
        state_crc_u64(table, &crc, name_bytes);
        crc = state_crc64_update(
            table, crc, tensor->name, name_bytes);
        state_crc_u64(
            table, &crc, tensor->physical_offset);
        state_crc_u64(
            table, &crc, tensor->byte_length);
        state_crc_u16(table, &crc, tensor->shard);
        state_crc_u32(table, &crc, tensor->ndim);
        state_crc_u32(table, &crc, tensor->dtype);
        for (uint32_t dimension = 0u;
             dimension < tensor->ndim; dimension++) {
            state_crc_u64(
                table, &crc, tensor->shape[dimension]);
        }
    }
    return crc;
}

static bool state_write_full(int fd,
                             const void *data,
                             uint64_t bytes) {
    const uint8_t *source = (const uint8_t *)data;
    while (bytes > 0u) {
        const size_t count =
            bytes > SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        ssize_t written = write(fd, source, count);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) return false;
        source += written;
        bytes -= (uint64_t)written;
    }
    return true;
}

static bool state_pwrite_full(int fd,
                              const void *data,
                              uint64_t bytes,
                              uint64_t offset) {
    const uint8_t *source = (const uint8_t *)data;
    while (bytes > 0u) {
        const size_t count =
            bytes > SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        ssize_t written = pwrite(
            fd, source, count, (off_t)offset);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) return false;
        source += written;
        bytes -= (uint64_t)written;
        offset += (uint64_t)written;
    }
    return true;
}

static bool state_pread_full(int fd,
                             void *data,
                             uint64_t bytes,
                             uint64_t offset) {
    uint8_t *destination = (uint8_t *)data;
    while (bytes > 0u) {
        const size_t count =
            bytes > SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        ssize_t got = pread(
            fd, destination, count, (off_t)offset);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return false;
        destination += got;
        bytes -= (uint64_t)got;
        offset += (uint64_t)got;
    }
    return true;
}

static uint64_t state_mla_payload_bytes(
        uint32_t token_position) {
    return
        (uint64_t)K3_ENGINE_MLA_LAYERS *
        token_position *
        K3_ENGINE_MLA_CACHE_DIM *
        sizeof(uint16_t);
}

static void state_metadata_build(
        const k3_engine *engine,
        k3_state_metadata *metadata) {
    memset(metadata, 0, sizeof(*metadata));
    metadata->flags =
        engine->q8_projections ?
            K3_STATE_FLAG_Q8_PROJECTIONS : 0u;
    metadata->context = engine->context;
    metadata->token_position = engine->token_position;
    metadata->model_layout_crc64 =
        engine->model_layout_crc64;
    metadata->kda_state_bytes =
        K3_ENGINE_KDA_STATE_BYTES;
    metadata->kda_conv_bytes =
        K3_ENGINE_KDA_CONV_BYTES;
    metadata->mla_bytes =
        state_mla_payload_bytes(engine->token_position);
    metadata->attn_res_bytes =
        (uint64_t)8u * K3_ENGINE_HIDDEN *
        sizeof(uint16_t);
    metadata->payload_bytes =
        metadata->kda_state_bytes +
        metadata->kda_conv_bytes +
        metadata->mla_bytes +
        metadata->attn_res_bytes;
    metadata->file_bytes =
        K3_STATE_HEADER_BYTES + metadata->payload_bytes;
}

static void state_info_set(
        k3_engine_state_file_info *info,
        const k3_state_metadata *metadata) {
    if (!info) return;
    memset(info, 0, sizeof(*info));
    info->format_version = K3_STATE_FORMAT_VERSION;
    info->token_position = metadata->token_position;
    info->model_layout_crc64 =
        metadata->model_layout_crc64;
    info->payload_bytes = metadata->payload_bytes;
    info->file_bytes = metadata->file_bytes;
    info->payload_crc64 = metadata->payload_crc64;
    info->q8_projections =
        (metadata->flags &
         K3_STATE_FLAG_Q8_PROJECTIONS) != 0u;
}

static void state_header_encode(
        const k3_state_metadata *metadata,
        uint8_t header[K3_STATE_HEADER_BYTES]) {
    memset(header, 0, K3_STATE_HEADER_BYTES);
    memcpy(
        header + K3_STATE_OFF_MAGIC,
        K3_STATE_MAGIC, sizeof(K3_STATE_MAGIC));
    state_put_u32(
        header + K3_STATE_OFF_VERSION,
        K3_STATE_FORMAT_VERSION);
    state_put_u32(
        header + K3_STATE_OFF_HEADER_BYTES,
        K3_STATE_HEADER_BYTES);
    state_put_u32(
        header + K3_STATE_OFF_ENDIAN,
        K3_STATE_ENDIAN_MARKER);
    state_put_u32(
        header + K3_STATE_OFF_FLAGS,
        metadata->flags);
    state_put_u32(
        header + K3_STATE_OFF_CONTEXT,
        metadata->context);
    state_put_u32(
        header + K3_STATE_OFF_TOKEN_POSITION,
        metadata->token_position);
    state_put_u32(
        header + K3_STATE_OFF_LAYERS,
        K3_ENGINE_LAYERS);
    state_put_u32(
        header + K3_STATE_OFF_MLA_LAYERS,
        K3_ENGINE_MLA_LAYERS);
    state_put_u64(
        header + K3_STATE_OFF_MODEL_LAYOUT_CRC,
        metadata->model_layout_crc64);
    state_put_u64(
        header + K3_STATE_OFF_PAYLOAD_BYTES,
        metadata->payload_bytes);
    state_put_u64(
        header + K3_STATE_OFF_PAYLOAD_CRC,
        metadata->payload_crc64);
    state_put_u64(
        header + K3_STATE_OFF_KDA_STATE_BYTES,
        metadata->kda_state_bytes);
    state_put_u64(
        header + K3_STATE_OFF_KDA_CONV_BYTES,
        metadata->kda_conv_bytes);
    state_put_u64(
        header + K3_STATE_OFF_MLA_BYTES,
        metadata->mla_bytes);
    state_put_u64(
        header + K3_STATE_OFF_ATTN_RES_BYTES,
        metadata->attn_res_bytes);
    uint64_t table[256];
    state_crc64_table(table);
    const uint64_t header_crc = state_crc64_update(
        table, 0u, header, K3_STATE_HEADER_BYTES);
    state_put_u64(
        header + K3_STATE_OFF_HEADER_CRC,
        header_crc);
}

static bool state_header_decode(
        const uint8_t header[K3_STATE_HEADER_BYTES],
        uint64_t file_bytes,
        k3_state_metadata *metadata,
        char *error,
        size_t error_size) {
    if (memcmp(
            header + K3_STATE_OFF_MAGIC,
            K3_STATE_MAGIC, sizeof(K3_STATE_MAGIC)) != 0) {
        engine_error(error, error_size,
                     "K3 state file has invalid magic");
        return false;
    }
    if (state_get_u32(
            header + K3_STATE_OFF_VERSION) !=
            K3_STATE_FORMAT_VERSION ||
        state_get_u32(
            header + K3_STATE_OFF_HEADER_BYTES) !=
            K3_STATE_HEADER_BYTES ||
        state_get_u32(
            header + K3_STATE_OFF_ENDIAN) !=
            K3_STATE_ENDIAN_MARKER) {
        engine_error(error, error_size,
                     "K3 state file format is unsupported");
        return false;
    }
    uint8_t checked[K3_STATE_HEADER_BYTES];
    memcpy(checked, header, sizeof(checked));
    const uint64_t expected_header_crc =
        state_get_u64(
            checked + K3_STATE_OFF_HEADER_CRC);
    memset(
        checked + K3_STATE_OFF_HEADER_CRC,
        0, sizeof(uint64_t));
    uint64_t table[256];
    state_crc64_table(table);
    const uint64_t measured_header_crc =
        state_crc64_update(
            table, 0u, checked, sizeof(checked));
    if (measured_header_crc != expected_header_crc) {
        engine_error(error, error_size,
                     "K3 state header CRC64 mismatch");
        return false;
    }
    memset(metadata, 0, sizeof(*metadata));
    metadata->flags = state_get_u32(
        header + K3_STATE_OFF_FLAGS);
    metadata->context = state_get_u32(
        header + K3_STATE_OFF_CONTEXT);
    metadata->token_position = state_get_u32(
        header + K3_STATE_OFF_TOKEN_POSITION);
    metadata->model_layout_crc64 = state_get_u64(
        header + K3_STATE_OFF_MODEL_LAYOUT_CRC);
    metadata->payload_bytes = state_get_u64(
        header + K3_STATE_OFF_PAYLOAD_BYTES);
    metadata->payload_crc64 = state_get_u64(
        header + K3_STATE_OFF_PAYLOAD_CRC);
    metadata->kda_state_bytes = state_get_u64(
        header + K3_STATE_OFF_KDA_STATE_BYTES);
    metadata->kda_conv_bytes = state_get_u64(
        header + K3_STATE_OFF_KDA_CONV_BYTES);
    metadata->mla_bytes = state_get_u64(
        header + K3_STATE_OFF_MLA_BYTES);
    metadata->attn_res_bytes = state_get_u64(
        header + K3_STATE_OFF_ATTN_RES_BYTES);
    metadata->file_bytes = file_bytes;
    const uint64_t expected_mla_bytes =
        state_mla_payload_bytes(
            metadata->token_position);
    const uint64_t expected_attn_res_bytes =
        (uint64_t)8u * K3_ENGINE_HIDDEN *
        sizeof(uint16_t);
    const uint64_t expected_payload_bytes =
        K3_ENGINE_KDA_STATE_BYTES +
        K3_ENGINE_KDA_CONV_BYTES +
        expected_mla_bytes +
        expected_attn_res_bytes;
    if ((metadata->flags &
         ~K3_STATE_FLAG_Q8_PROJECTIONS) != 0u ||
        metadata->context < K3_ENGINE_MIN_CONTEXT ||
        metadata->context > K3_ENGINE_MAX_CONTEXT ||
        metadata->token_position > metadata->context ||
        state_get_u32(
            header + K3_STATE_OFF_LAYERS) !=
            K3_ENGINE_LAYERS ||
        state_get_u32(
            header + K3_STATE_OFF_MLA_LAYERS) !=
            K3_ENGINE_MLA_LAYERS ||
        metadata->kda_state_bytes !=
            K3_ENGINE_KDA_STATE_BYTES ||
        metadata->kda_conv_bytes !=
            K3_ENGINE_KDA_CONV_BYTES ||
        metadata->mla_bytes != expected_mla_bytes ||
        metadata->attn_res_bytes !=
            expected_attn_res_bytes ||
        metadata->payload_bytes !=
            expected_payload_bytes ||
        metadata->payload_bytes >
            UINT64_MAX - K3_STATE_HEADER_BYTES ||
        file_bytes !=
            K3_STATE_HEADER_BYTES +
                metadata->payload_bytes) {
        engine_error(error, error_size,
                     "K3 state file metadata is inconsistent");
        return false;
    }
    return true;
}

static bool state_writer_device(
        k3_state_writer *writer,
        const void *device,
        uint64_t bytes,
        const char *label,
        char *error,
        size_t error_size) {
    for (uint64_t offset = 0u; offset < bytes;) {
        const uint64_t count =
            bytes - offset < K3_STATE_TRANSFER_BYTES ?
                bytes - offset : K3_STATE_TRANSFER_BYTES;
        hipError_t status = hipMemcpy(
            writer->host,
            (const uint8_t *)device + offset,
            count, hipMemcpyDeviceToHost);
        if (status != hipSuccess) {
            engine_error(
                error, error_size,
                "K3 state export %s copy failed at %llu: %s",
                label, (unsigned long long)offset,
                hipGetErrorString(status));
            return false;
        }
        writer->payload_crc64 = state_crc64_update(
            writer->crc_table,
            writer->payload_crc64,
            writer->host, count);
        if (!state_write_full(
                writer->fd, writer->host, count)) {
            engine_error(
                error, error_size,
                "K3 state export %s write failed: %s",
                label, strerror(errno));
            return false;
        }
        offset += count;
    }
    return true;
}

static bool state_reader_device(
        k3_state_reader *reader,
        void *device,
        uint64_t bytes,
        const char *label,
        char *error,
        size_t error_size) {
    for (uint64_t offset = 0u; offset < bytes;) {
        const uint64_t count =
            bytes - offset < K3_STATE_TRANSFER_BYTES ?
                bytes - offset : K3_STATE_TRANSFER_BYTES;
        if (!state_pread_full(
                reader->fd, reader->host, count,
                reader->offset)) {
            engine_error(
                error, error_size,
                "K3 state import %s read failed: %s",
                label, strerror(errno));
            return false;
        }
        hipError_t status = hipMemcpy(
            (uint8_t *)device + offset,
            reader->host, count,
            hipMemcpyHostToDevice);
        if (status != hipSuccess) {
            engine_error(
                error, error_size,
                "K3 state import %s copy failed at %llu: %s",
                label, (unsigned long long)offset,
                hipGetErrorString(status));
            return false;
        }
        reader->offset += count;
        offset += count;
    }
    return true;
}

extern "C" bool k3_engine_export_state(
        const k3_engine *engine,
        const char *path,
        k3_engine_state_file_info *info,
        char *error,
        size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (info) memset(info, 0, sizeof(*info));
    if (!engine || !path || path[0] == '\0' ||
        !engine->causal_state_valid ||
        engine->decoded_layers != 0u ||
        engine->token_position > engine->context ||
        k3_io_uring_outstanding(
            engine->staging_ring) != 0u) {
        engine_error(error, error_size,
                     "invalid K3 state-export request");
        return false;
    }
    struct timespec operation_start;
    struct timespec operation_end;
    clock_gettime(
        CLOCK_MONOTONIC, &operation_start);
    hipError_t status = hipDeviceSynchronize();
    if (status != hipSuccess) {
        engine_error(
            error, error_size,
            "K3 state-export synchronization failed: %s",
            hipGetErrorString(status));
        return false;
    }
    const size_t path_bytes = strlen(path);
    static const char suffix[] = ".tmp.XXXXXX";
    if (path_bytes > SIZE_MAX - sizeof(suffix)) {
        engine_error(error, error_size,
                     "K3 state-export path is too long");
        return false;
    }
    char *temporary =
        (char *)malloc(path_bytes + sizeof(suffix));
    if (!temporary) {
        engine_error(error, error_size,
                     "K3 state-export path allocation failed");
        return false;
    }
    memcpy(temporary, path, path_bytes);
    memcpy(
        temporary + path_bytes,
        suffix, sizeof(suffix));
    int fd = mkstemp(temporary);
    if (fd < 0) {
        engine_error(
            error, error_size,
            "create K3 state temporary file: %s",
            strerror(errno));
        free(temporary);
        return false;
    }
    (void)fcntl(fd, F_SETFD, FD_CLOEXEC);
    bool ok = false;
    uint8_t zero_header[K3_STATE_HEADER_BYTES] = { 0 };
    k3_state_writer writer;
    memset(&writer, 0, sizeof(writer));
    writer.fd = fd;
    writer.host =
        (uint8_t *)malloc(K3_STATE_TRANSFER_BYTES);
    state_crc64_table(writer.crc_table);
    k3_state_metadata metadata;
    state_metadata_build(engine, &metadata);
    const uint64_t mla_layer_bytes =
        (uint64_t)engine->context *
        K3_ENGINE_MLA_CACHE_DIM * sizeof(uint16_t);
    const uint64_t mla_occupied_bytes =
        (uint64_t)engine->token_position *
        K3_ENGINE_MLA_CACHE_DIM * sizeof(uint16_t);
    uint8_t header[K3_STATE_HEADER_BYTES];
    if (!writer.host) {
        engine_error(error, error_size,
                     "K3 state-export transfer allocation failed");
        goto done;
    }
    if (!state_write_full(
            fd, zero_header, sizeof(zero_header))) {
        engine_error(
            error, error_size,
            "K3 state-export header reservation failed: %s",
            strerror(errno));
        goto done;
    }
    if (!state_writer_device(
            &writer, engine->kda_state,
            K3_ENGINE_KDA_STATE_BYTES, "KDA state",
            error, error_size) ||
        !state_writer_device(
            &writer, engine->kda_conv,
            K3_ENGINE_KDA_CONV_BYTES, "KDA convolution",
            error, error_size)) {
        goto done;
    }
    for (uint32_t layer = 0u;
         layer < K3_ENGINE_MLA_LAYERS; layer++) {
        if (!state_writer_device(
                &writer,
                (const uint8_t *)engine->mla_cache +
                    (uint64_t)layer * mla_layer_bytes,
                mla_occupied_bytes, "MLA rows",
                error, error_size)) {
            goto done;
        }
    }
    if (!state_writer_device(
            &writer, engine->attn_res_blocks,
            metadata.attn_res_bytes, "AttnRes",
            error, error_size)) {
        goto done;
    }
    metadata.payload_crc64 =
        writer.payload_crc64;
    state_header_encode(&metadata, header);
    if (!state_pwrite_full(
            fd, header, sizeof(header), 0u)) {
        engine_error(
            error, error_size,
            "K3 state-export header write failed: %s",
            strerror(errno));
        goto done;
    }
    if (fsync(fd) != 0) {
        engine_error(
            error, error_size,
            "K3 state-export fsync failed: %s",
            strerror(errno));
        goto done;
    }
    if (close(fd) != 0) {
        fd = -1;
        engine_error(
            error, error_size,
            "K3 state-export close failed: %s",
            strerror(errno));
        goto done;
    }
    fd = -1;
    if (rename(temporary, path) != 0) {
        engine_error(
            error, error_size,
            "K3 state-export rename failed: %s",
            strerror(errno));
        goto done;
    }
    state_info_set(info, &metadata);
    clock_gettime(
        CLOCK_MONOTONIC, &operation_end);
    if (info) {
        info->wall_seconds = elapsed_seconds(
            operation_start, operation_end);
    }
    ok = true;

done:
    if (fd >= 0) (void)close(fd);
    if (!ok) (void)unlink(temporary);
    free(writer.host);
    free(temporary);
    return ok;
}

extern "C" bool k3_engine_import_state(
        k3_engine *engine,
        const char *path,
        k3_engine_state_file_info *info,
        char *error,
        size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (info) memset(info, 0, sizeof(*info));
    if (!engine || !path || path[0] == '\0' ||
        engine->decoded_layers != 0u ||
        k3_io_uring_outstanding(
            engine->staging_ring) != 0u) {
        engine_error(error, error_size,
                     "invalid K3 state-import request");
        return false;
    }
    struct timespec operation_start;
    struct timespec operation_end;
    clock_gettime(
        CLOCK_MONOTONIC, &operation_start);
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        engine_error(
            error, error_size,
            "open K3 state file: %s", strerror(errno));
        return false;
    }
    bool ok = false;
    uint8_t *host =
        (uint8_t *)malloc(K3_STATE_TRANSFER_BYTES);
    struct stat status_buffer;
    uint64_t file_bytes = 0u;
    uint8_t header[K3_STATE_HEADER_BYTES];
    k3_state_metadata metadata;
    bool file_q8 = false;
    uint64_t table[256];
    uint64_t payload_crc64 = 0u;
    uint64_t offset = 0u;
    uint64_t remaining = 0u;
    hipError_t hip_status = hipSuccess;
    k3_state_reader reader;
    memset(&reader, 0, sizeof(reader));
    uint64_t mla_layer_bytes = 0u;
    uint64_t mla_occupied_bytes = 0u;
    if (!host) {
        engine_error(error, error_size,
                     "K3 state-import transfer allocation failed");
        goto done;
    }
    if (fstat(fd, &status_buffer) != 0 ||
        status_buffer.st_size < 0) {
        engine_error(
            error, error_size,
            "stat K3 state file: %s", strerror(errno));
        goto done;
    }
    file_bytes = (uint64_t)status_buffer.st_size;
    if (file_bytes < sizeof(header) ||
        !state_pread_full(
            fd, header, sizeof(header), 0u)) {
        engine_error(error, error_size,
                     "K3 state file is truncated");
        goto done;
    }
    if (!state_header_decode(
            header, file_bytes, &metadata,
            error, error_size)) {
        goto done;
    }
    file_q8 =
        (metadata.flags &
         K3_STATE_FLAG_Q8_PROJECTIONS) != 0u;
    if (metadata.context != engine->context ||
        metadata.model_layout_crc64 !=
            engine->model_layout_crc64 ||
        file_q8 != engine->q8_projections) {
        engine_error(
            error, error_size,
            "K3 state model/static identity mismatch");
        goto done;
    }
    state_crc64_table(table);
    payload_crc64 = 0u;
    offset = K3_STATE_HEADER_BYTES;
    remaining = metadata.payload_bytes;
    while (remaining > 0u) {
        const uint64_t count =
            remaining < K3_STATE_TRANSFER_BYTES ?
                remaining : K3_STATE_TRANSFER_BYTES;
        if (!state_pread_full(
                fd, host, count, offset)) {
            engine_error(error, error_size,
                         "K3 state payload is truncated");
            goto done;
        }
        payload_crc64 = state_crc64_update(
            table, payload_crc64, host, count);
        offset += count;
        remaining -= count;
    }
    if (payload_crc64 != metadata.payload_crc64) {
        engine_error(error, error_size,
                     "K3 state payload CRC64 mismatch");
        goto done;
    }

    hip_status = hipDeviceSynchronize();
    if (hip_status != hipSuccess) {
        engine_error(
            error, error_size,
            "K3 state-import synchronization failed: %s",
            hipGetErrorString(hip_status));
        goto done;
    }
    engine->causal_state_valid = false;
    hip_status = hipMemset(
        engine->mla_cache, 0,
        engine->mla_cache_bytes);
    if (hip_status != hipSuccess) {
        engine_error(
            error, error_size,
            "K3 state-import MLA clear failed: %s",
            hipGetErrorString(hip_status));
        goto done;
    }
    reader.fd = fd;
    reader.host = host;
    reader.offset = K3_STATE_HEADER_BYTES;
    if (!state_reader_device(
            &reader, engine->kda_state,
            K3_ENGINE_KDA_STATE_BYTES, "KDA state",
            error, error_size) ||
        !state_reader_device(
            &reader, engine->kda_conv,
            K3_ENGINE_KDA_CONV_BYTES, "KDA convolution",
            error, error_size)) {
        goto done;
    }
    mla_layer_bytes =
        (uint64_t)engine->context *
        K3_ENGINE_MLA_CACHE_DIM * sizeof(uint16_t);
    mla_occupied_bytes =
        (uint64_t)metadata.token_position *
        K3_ENGINE_MLA_CACHE_DIM * sizeof(uint16_t);
    for (uint32_t layer = 0u;
         layer < K3_ENGINE_MLA_LAYERS; layer++) {
        if (!state_reader_device(
                &reader,
                (uint8_t *)engine->mla_cache +
                    (uint64_t)layer * mla_layer_bytes,
                mla_occupied_bytes, "MLA rows",
                error, error_size)) {
            goto done;
        }
    }
    if (!state_reader_device(
            &reader, engine->attn_res_blocks,
            metadata.attn_res_bytes, "AttnRes",
            error, error_size)) {
        goto done;
    }
    if (reader.offset != metadata.file_bytes) {
        engine_error(error, error_size,
                     "K3 state-import payload ledger mismatch");
        goto done;
    }
    hip_status = hipDeviceSynchronize();
    if (hip_status != hipSuccess) {
        engine_error(
            error, error_size,
            "K3 state-import completion failed: %s",
            hipGetErrorString(hip_status));
        goto done;
    }
    engine->token_position = metadata.token_position;
    engine->decoded_layers = 0u;
    engine->causal_state_valid = true;
    state_info_set(info, &metadata);
    clock_gettime(
        CLOCK_MONOTONIC, &operation_end);
    if (info) {
        info->wall_seconds = elapsed_seconds(
            operation_start, operation_end);
    }
    ok = true;

done:
    free(host);
    (void)close(fd);
    return ok;
}
