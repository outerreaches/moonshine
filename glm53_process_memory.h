#ifndef GLM53_PROCESS_MEMORY_H
#define GLM53_PROCESS_MEMORY_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GLM53_PROCESS_MEMORY_OK = 0,
    GLM53_PROCESS_MEMORY_INVALID_ARGUMENT,
    GLM53_PROCESS_MEMORY_OPEN_FAILED,
    GLM53_PROCESS_MEMORY_IO_ERROR,
    GLM53_PROCESS_MEMORY_MISSING_FIELD,
    GLM53_PROCESS_MEMORY_DUPLICATE_FIELD,
    GLM53_PROCESS_MEMORY_MALFORMED,
    GLM53_PROCESS_MEMORY_OVERFLOW
} glm53_process_memory_status;

typedef struct {
    uint64_t mem_available_bytes;
    uint64_t swap_free_bytes;
    uint64_t vm_swap_bytes;
    uint64_t smaps_swap_bytes;
    uint64_t smaps_swap_pss_bytes;
    uint64_t model_vma_count;
    uint64_t model_vma_bytes;
    uint64_t largest_model_vma_bytes;
} glm53_process_memory;

/* FILE parsers are exposed so admission behavior can be tested without
 * depending on the machine running the test. Destinations are transactional. */
glm53_process_memory_status glm53_process_memory_parse_meminfo(
    FILE *file, uint64_t *mem_available_bytes, uint64_t *swap_free_bytes);
glm53_process_memory_status glm53_process_memory_parse_status(
    FILE *file, uint64_t *vm_swap_bytes);
glm53_process_memory_status glm53_process_memory_parse_smaps_rollup(
    FILE *file, uint64_t *swap_bytes, uint64_t *swap_pss_bytes);
glm53_process_memory_status glm53_process_memory_parse_maps(
    FILE *file, const char *model_path_prefix, uint64_t *vma_count,
    uint64_t *vma_bytes, uint64_t *largest_vma_bytes);

/* proc_root is normally "/proc". The process files are read from
 * meminfo, self/status, self/smaps_rollup, and self/maps below it. */
glm53_process_memory_status glm53_process_memory_sample_root(
    const char *proc_root, const char *model_path_prefix,
    glm53_process_memory *out);
glm53_process_memory_status glm53_process_memory_sample(
    const char *model_path_prefix, glm53_process_memory *out);

/* Admission requires 16 GiB beyond remaining_uncommitted_bytes, no process
 * swap, and no already mapped model VMA. Arithmetic overflow rejects. */
bool glm53_process_memory_policy_allows(
    const glm53_process_memory *sample, uint64_t remaining_uncommitted_bytes);

const char *glm53_process_memory_status_string(
    glm53_process_memory_status status);

#ifdef __cplusplus
}
#endif
#endif
