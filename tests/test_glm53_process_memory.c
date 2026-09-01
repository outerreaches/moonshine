#include "glm53_process_memory.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
    return false; } } while (0)
#define OK(x) CHECK((x) == GLM53_PROCESS_MEMORY_OK)

static FILE *text_file(const char *text) {
    FILE *f = tmpfile();
    if (!f) return NULL;
    if (fputs(text, f) == EOF || fflush(f) != 0) {
        fclose(f);
        return NULL;
    }
    rewind(f);
    return f;
}

static bool test_kb_parsers(void) {
    FILE *f;
    uint64_t a = 7, b = 9;
    f = text_file("MemTotal: 999 kB\nSwapFree: 3 kB\nMemAvailable: 5 kB\n");
    CHECK(f != NULL);
    OK(glm53_process_memory_parse_meminfo(f, &a, &b));
    CHECK(fclose(f) == 0);
    CHECK(a == 5120 && b == 3072);

    a = 77; b = 88;
    f = text_file("MemAvailable: 1 kB\nMemAvailable: 2 kB\nSwapFree: 0 kB\n");
    CHECK(f != NULL);
    CHECK(glm53_process_memory_parse_meminfo(f, &a, &b) ==
          GLM53_PROCESS_MEMORY_DUPLICATE_FIELD);
    CHECK(fclose(f) == 0);
    CHECK(a == 77 && b == 88);

    f = text_file("MemAvailable: 1 MB\nSwapFree: 0 kB\n");
    CHECK(f != NULL);
    CHECK(glm53_process_memory_parse_meminfo(f, &a, &b) ==
          GLM53_PROCESS_MEMORY_MALFORMED);
    CHECK(fclose(f) == 0);
    CHECK(a == 77 && b == 88);

    f = text_file("MemAvailable: 18014398509481984 kB\nSwapFree: 0 kB\n");
    CHECK(f != NULL);
    CHECK(glm53_process_memory_parse_meminfo(f, &a, &b) ==
          GLM53_PROCESS_MEMORY_OVERFLOW);
    CHECK(fclose(f) == 0);

    f = text_file("Name:\ttest\nVmSwap:\t12 kB\n");
    CHECK(f != NULL);
    OK(glm53_process_memory_parse_status(f, &a));
    CHECK(fclose(f) == 0);
    CHECK(a == 12288);

    a = 1; b = 2;
    f = text_file("Rss: 4 kB\nSwap: 0 kB\nSwapPss: 1 kB\n");
    CHECK(f != NULL);
    OK(glm53_process_memory_parse_smaps_rollup(f, &a, &b));
    CHECK(fclose(f) == 0);
    CHECK(a == 0 && b == 1024);

    a = 123;
    f = text_file("Name: no-swap-field\n");
    CHECK(f != NULL);
    CHECK(glm53_process_memory_parse_status(f, &a) ==
          GLM53_PROCESS_MEMORY_MISSING_FIELD);
    CHECK(fclose(f) == 0);
    CHECK(a == 123);
    return true;
}

static bool test_maps(void) {
    static const char maps[] =
        "1000-3000 r--p 00000000 08:01 1 /models/a.bin\n"
        "4000-5000 rw-p 00002000 08:01 1 /models/a.bin\n"
        "5000-6000 r--p 00000000 08:01 9 /models-other/no.bin\n"
        "8000-a000 r--p 00000000 08:01 2 /other/b.safetensors\n"
        "b000-c000 r--p 00000000 08:01 4 /gone/c.safetensors (deleted)\n"
        "a000-b000 rw-p 00000000 00:00 0 [heap]\n"
        "c000-d000 r--p 00000000 08:01 3 /other/not-model.bin\n";
    FILE *f = text_file(maps);
    uint64_t count = 99, bytes = 88, largest = 77;
    CHECK(f != NULL);
    OK(glm53_process_memory_parse_maps(f, "/models/", &count, &bytes,
                                       &largest));
    CHECK(fclose(f) == 0);
    CHECK(count == 4);
    CHECK(bytes == UINT64_C(0x6000));
    CHECK(largest == UINT64_C(0x2000));

    count = 99; bytes = 88; largest = 77;
    f = text_file("1000-1000 r--p 0 00:00 0 /x.safetensors\n");
    CHECK(f != NULL);
    CHECK(glm53_process_memory_parse_maps(f, NULL, &count, &bytes,
                                          &largest) ==
          GLM53_PROCESS_MEMORY_MALFORMED);
    CHECK(fclose(f) == 0);
    CHECK(count == 99 && bytes == 88 && largest == 77);

    f = text_file("not a maps line\n");
    CHECK(f != NULL);
    CHECK(glm53_process_memory_parse_maps(f, NULL, &count, &bytes,
                                          &largest) ==
          GLM53_PROCESS_MEMORY_MALFORMED);
    CHECK(fclose(f) == 0);
    return true;
}

static bool test_policy(void) {
    glm53_process_memory s;
    memset(&s, 0, sizeof(s));
    s.mem_available_bytes = UINT64_C(20) * UINT64_C(1073741824);
    CHECK(glm53_process_memory_policy_allows(
        &s, UINT64_C(4) * UINT64_C(1073741824)));
    CHECK(!glm53_process_memory_policy_allows(
        &s, UINT64_C(4) * UINT64_C(1073741824) + 1));
    s.vm_swap_bytes = 1;
    CHECK(!glm53_process_memory_policy_allows(&s, 0));
    s.vm_swap_bytes = 0; s.smaps_swap_bytes = 1;
    CHECK(!glm53_process_memory_policy_allows(&s, 0));
    s.smaps_swap_bytes = 0; s.model_vma_count = 1;
    CHECK(!glm53_process_memory_policy_allows(&s, 0));
    s.model_vma_count = 0; s.mem_available_bytes = UINT64_MAX;
    CHECK(!glm53_process_memory_policy_allows(&s, UINT64_MAX));
    CHECK(!glm53_process_memory_policy_allows(NULL, 0));
    return true;
}

static bool test_live_proc(void) {
    glm53_process_memory s;
    glm53_process_memory_status status =
        glm53_process_memory_sample("/definitely/not/our/model/", &s);
    if (status != GLM53_PROCESS_MEMORY_OK) {
        fprintf(stderr, "live /proc sample: %s\n",
                glm53_process_memory_status_string(status));
        return false;
    }
    CHECK(s.mem_available_bytes > 0);
    CHECK(s.largest_model_vma_bytes <= s.model_vma_bytes);
    return true;
}

int main(void) {
    CHECK(test_kb_parsers());
    CHECK(test_maps());
    CHECK(test_policy());
    CHECK(test_live_proc());
    puts("glm53_process_memory: all tests passed");
    return 0;
}
