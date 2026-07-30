#ifndef K3_SAFETENSORS_H
#define K3_SAFETENSORS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define K3_ST_MAX_DIMS 8

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    K3_ST_DTYPE_UNKNOWN = 0,
    K3_ST_DTYPE_BF16,
    K3_ST_DTYPE_F32,
    K3_ST_DTYPE_U8,
} k3_st_dtype;

typedef struct {
    char        *name;
    uint64_t     physical_offset;
    uint64_t     byte_length;
    uint64_t     shape[K3_ST_MAX_DIMS];
    uint16_t     shard;
    uint8_t      ndim;
    k3_st_dtype  dtype;
} k3_st_tensor;

typedef struct {
    char     *path;
    int       fd;
    int       direct_fd;
    uint64_t  file_bytes;
    uint64_t  data_offset;
} k3_st_shard;

typedef struct {
    k3_st_shard  *shards;
    size_t        shard_count;
    k3_st_tensor *tensors;
    size_t        tensor_count;
    size_t        tensor_capacity;
} k3_st_model;

typedef struct {
    void    *allocation;
    uint8_t *data;
    uint64_t allocation_bytes;
    uint64_t data_bytes;
    uint64_t physical_offset;
    bool     used_direct_io;
} k3_st_read;

typedef struct {
    uint8_t *data;
    uint64_t allocation_bytes;
    uint64_t data_bytes;
    uint64_t physical_offset;
    bool     used_direct_io;
} k3_st_read_view;

/*
 * Open model-NNNNN-of-NNNNN.safetensors shards, parse their headers, and
 * build a sorted tensor directory. The model retains read-only file
 * descriptors until k3_st_model_close().
 */
bool k3_st_model_open(k3_st_model *model,
                      const char  *root,
                      size_t       shard_count,
                      char        *error,
                      size_t       error_size);

void k3_st_model_close(k3_st_model *model);

const k3_st_tensor *k3_st_find(const k3_st_model *model, const char *name);

/*
 * Read an arbitrary byte span. The underlying request is expanded to the
 * requested power-of-two alignment and uses O_DIRECT when the filesystem
 * permits it. read->data points at the exact requested first byte.
 */
bool k3_st_read_span(const k3_st_model *model,
                     uint16_t           shard,
                     uint64_t           physical_offset,
                     uint64_t           byte_length,
                     uint64_t           alignment,
                     k3_st_read        *read,
                     char              *error,
                     size_t             error_size);

/*
 * Read the same aligned span into caller-owned storage. This is the streaming
 * path for page-aligned, GPU-visible fixed buffers; the caller retains
 * ownership and must keep the storage alive while the returned view is used.
 */
bool k3_st_read_span_into(const k3_st_model *model,
                          uint16_t           shard,
                          uint64_t           physical_offset,
                          uint64_t           byte_length,
                          uint64_t           alignment,
                          void              *allocation,
                          uint64_t           allocation_capacity,
                          k3_st_read_view   *view,
                          char              *error,
                          size_t             error_size);

void k3_st_read_release(k3_st_read *read);

#ifdef __cplusplus
}
#endif

#endif
