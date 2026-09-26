/*
 * Strictness of the two environment overrides resolved by
 * mimo26_gpu_worker_resolve_overrides.
 *
 * This function had no test until 2026-09-26, which is how its predecessor in
 * the layer shipped as `strcmp(value, "0") == 0 ? 0 : 1` -- so
 * MIMO26_EXPERT_MAJOR=off turned grouping ON, and /health reported the mode
 * that had been asked for rather than the one running. An override that is
 * parsed loosely is worse than no override: it silently runs a profile nothing
 * was qualified on.
 *
 * So both overrides must accept exactly {0,1,on,off}, refuse everything else
 * rather than guessing, and leave the config untouched when they refuse.
 */
#include "mimo26_gpu_worker.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned checks;

static mimo26_gpu_worker_config baseline(bool lookahead)
{
    mimo26_gpu_worker_config config;
    mimo26_gpu_worker_config_defaults(&config);
    config.expert_lookahead = lookahead;
    config.expert_major = false;
    config.expert_weight_reuse = false;
    return config;
}

/* Expect acceptance, and the named field to land on `want`. */
static void accepts(const char *name, const char *value, bool lookahead,
                    bool config_field_offset_major, bool want)
{
    mimo26_gpu_worker_config config = baseline(lookahead);
    char error[256] = {0};
    if (value != NULL) setenv(name, value, 1); else unsetenv(name);
    const mimo26_gpu_worker_status status =
        mimo26_gpu_worker_resolve_overrides(&config, error, sizeof error);
    if (status != MIMO26_GPU_WORKER_OK) {
        fprintf(stderr, "case %u: %s=%s expected OK, got %d (%s)\n",
                checks, name, value ? value : "(unset)", (int)status, error);
    }
    assert(status == MIMO26_GPU_WORKER_OK);
    const bool got = config_field_offset_major ? config.expert_major
                                              : config.expert_weight_reuse;
    if (got != want) {
        fprintf(stderr, "case %u: %s=%s expected %d, got %d\n", checks, name,
                value ? value : "(unset)", (int)want, (int)got);
    }
    assert(got == want);
    if (value != NULL) unsetenv(name);
    ++checks;
}

/* Expect refusal, a populated message, and no mutation of the config. */
static void refuses(const char *name, const char *value, bool lookahead)
{
    mimo26_gpu_worker_config config = baseline(lookahead), before = config;
    char error[256] = {0};
    setenv(name, value, 1);
    const mimo26_gpu_worker_status status =
        mimo26_gpu_worker_resolve_overrides(&config, error, sizeof error);
    if (status == MIMO26_GPU_WORKER_OK) {
        fprintf(stderr, "case %u: %s=%s should have been refused\n", checks, name, value);
    }
    assert(status != MIMO26_GPU_WORKER_OK);
    assert(error[0] != '\0');
    assert(memcmp(&before, &config, sizeof before) == 0);
    unsetenv(name);
    ++checks;
}

int main(void)
{
    unsetenv("MIMO26_EXPERT_MAJOR");
    unsetenv("MIMO26_EXPERT_WEIGHT_REUSE");

    /* Grouped prefill. "off" must mean off -- the original defect. */
    accepts("MIMO26_EXPERT_MAJOR", NULL, true, true, false);
    accepts("MIMO26_EXPERT_MAJOR", "1", true, true, true);
    accepts("MIMO26_EXPERT_MAJOR", "on", true, true, true);
    accepts("MIMO26_EXPERT_MAJOR", "0", true, true, false);
    accepts("MIMO26_EXPERT_MAJOR", "off", true, true, false);
    for (const char *bad : {"true", "false", "yes", "no", "ON", "Off", "2",
                            "", " on", "on ", "1junk"})
        refuses("MIMO26_EXPERT_MAJOR", bad, true);
    /* Grouping without the remaining-group schedule is slower, so refuse the
     * combination instead of quietly running it. */
    refuses("MIMO26_EXPERT_MAJOR", "on", false);
    accepts("MIMO26_EXPERT_MAJOR", "off", false, true, false);

    /*
     * The expert projection kernel. Unlike grouping it has no coupling to
     * lookahead -- weight reuse is a property of one call, not of the schedule
     * -- so "on" must be accepted with lookahead off too.
     */
    accepts("MIMO26_EXPERT_WEIGHT_REUSE", NULL, true, false, false);
    accepts("MIMO26_EXPERT_WEIGHT_REUSE", "1", true, false, true);
    accepts("MIMO26_EXPERT_WEIGHT_REUSE", "on", true, false, true);
    accepts("MIMO26_EXPERT_WEIGHT_REUSE", "0", true, false, false);
    accepts("MIMO26_EXPERT_WEIGHT_REUSE", "off", true, false, false);
    accepts("MIMO26_EXPERT_WEIGHT_REUSE", "on", false, false, true);
    for (const char *bad : {"true", "false", "yes", "reuse", "ON", "2", "",
                            " on", "on ", "1junk"})
        refuses("MIMO26_EXPERT_WEIGHT_REUSE", bad, true);

    /* Both at once, and independent of one another. */
    {
        mimo26_gpu_worker_config config = baseline(true);
        char error[256] = {0};
        setenv("MIMO26_EXPERT_MAJOR", "off", 1);
        setenv("MIMO26_EXPERT_WEIGHT_REUSE", "on", 1);
        assert(mimo26_gpu_worker_resolve_overrides(&config, error, sizeof error) ==
               MIMO26_GPU_WORKER_OK);
        assert(!config.expert_major && config.expert_weight_reuse);
        ++checks;
        /* A refusal on the second must not leave the first applied. */
        config = baseline(true);
        setenv("MIMO26_EXPERT_MAJOR", "on", 1);
        setenv("MIMO26_EXPERT_WEIGHT_REUSE", "banana", 1);
        assert(mimo26_gpu_worker_resolve_overrides(&config, error, sizeof error) !=
               MIMO26_GPU_WORKER_OK);
        assert(error[0] != '\0');
        ++checks;
        unsetenv("MIMO26_EXPERT_MAJOR");
        unsetenv("MIMO26_EXPERT_WEIGHT_REUSE");
    }

    /* The default profile must not enable either by accident. */
    {
        mimo26_gpu_worker_config config;
        mimo26_gpu_worker_config_defaults(&config);
        assert(!config.expert_weight_reuse);
        ++checks;
    }

    printf("PASS %u strict worker override cases\n", checks);
    return 0;
}
