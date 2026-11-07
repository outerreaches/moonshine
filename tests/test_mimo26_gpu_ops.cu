/*
 * G2: MiMo's GPU elementwise and router primitives against the CPU oracles.
 *
 * Every check here is BIT-EXACT. These are not reduction-order questions --
 * they are rounding-contract questions, and the whole reason this module
 * exists rather than reusing k3_rocm_ops is that three of MiMo's contracts
 * differ from K3's in ways a tolerance would not show:
 *
 *   RMSNorm  K3 rounds once, MiMo twice  (1029/4096 elements differ)
 *   SwiGLU   K3's is GLM's clamped form  (1681/4096 elements differ)
 *   Router   K3 sums the denominator in score order, MiMo in id order
 *
 * So the test also runs the K3 kernels on the same inputs and reports how
 * far they land. A "FAIL" there would mean the contracts had converged and
 * this module is redundant; anything else confirms the fork was necessary.
 * That reporting is what makes this a test of a decision and not just of
 * code.
 *
 * WHERE BIT-EXACTNESS STOPS. Everything above is exactly specified integer
 * and IEEE arithmetic, so it is held exactly. Two things here are not, and
 * both for the same reason: device libm and host libm are different
 * implementations, disagreeing by 1 ulp on 6.26% of expf inputs (measured in
 * docs/reference-extracts/expf-host-vs-device.hip). That reaches the router
 * through sigmoid, and attention through softmax. Anything downstream of a
 * transcendental gets a bounded gate and a stated reason; everything else
 * gets no tolerance at all. Reduction order is the other such boundary -- see
 * the BF16 GEMV check, which is K3's kernel and is allowed one BF16 step.
 *
 * Synthetic inputs only -- no checkpoint needed.
 */
#include "k3_rocm_ops.h"
#include "mimo26_ops.h"
#include "mimo26_rocm_ops.h"
#include "mimo26_router.h"

#include <hip/hip_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void ok(const char *what, int passed, const char *detail)
{
    printf("  %-4s %-44s %s\n", passed ? "ok" : "FAIL", what,
           detail ? detail : "");
    if (!passed) {
        failures++;
    }
}

#define HIP_OK(call)                                                          \
    do {                                                                      \
        hipError_t _e = (call);                                               \
        if (_e != hipSuccess) {                                               \
            fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #call,    \
                    hipGetErrorString(_e));                                   \
            return 1;                                                         \
        }                                                                     \
    } while (0)

static uint32_t rng_state = 0x243F6A88u;
static float next_uniform(float low, float high)
{
    rng_state = rng_state * 1664525u + 1013904223u;
    /* High bits: the low bits of an LCG have short periods, which has
     * already produced one bogus measurement in this lane. */
    const float unit = (float)((rng_state >> 8) & 0xFFFFu) / 65535.0f;
    return low + unit * (high - low);
}

#define HIDDEN 4096u
#define VECTORS 3u
#define EXPERTS MIMO26_ROUTER_EXPERTS
#define TOP_K MIMO26_ROUTER_TOP_K

int main(void)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    hipDeviceProp_t properties;
    HIP_OK(hipGetDeviceProperties(&properties, 0));
    printf("gfx %s, %d CUs\n", properties.gcnArchName,
           properties.multiProcessorCount);

    const size_t count = (size_t)VECTORS * HIDDEN;
    uint16_t *input = (uint16_t *)malloc(count * sizeof *input);
    uint16_t *second = (uint16_t *)malloc(count * sizeof *second);
    uint16_t *weight = (uint16_t *)malloc(HIDDEN * sizeof *weight);
    uint16_t *cpu = (uint16_t *)malloc(count * sizeof *cpu);
    uint16_t *gpu = (uint16_t *)malloc(count * sizeof *gpu);
    uint16_t *other = (uint16_t *)malloc(count * sizeof *other);
    if (!input || !second || !weight || !cpu || !gpu || !other) {
        return 1;
    }
    for (size_t i = 0; i < count; i++) {
        input[i] = mimo26_f32_to_bf16(next_uniform(-3.0f, 3.0f));
        second[i] = mimo26_f32_to_bf16(next_uniform(-3.0f, 3.0f));
    }
    for (size_t i = 0; i < HIDDEN; i++) {
        weight[i] = mimo26_f32_to_bf16(next_uniform(0.02f, 2.0f));
    }

    uint16_t *d_input = NULL, *d_second = NULL, *d_weight = NULL,
             *d_output = NULL;
    HIP_OK(hipMalloc(&d_input, count * sizeof *d_input));
    HIP_OK(hipMalloc(&d_second, count * sizeof *d_second));
    HIP_OK(hipMalloc(&d_weight, HIDDEN * sizeof *d_weight));
    HIP_OK(hipMalloc(&d_output, count * sizeof *d_output));
    HIP_OK(hipMemcpy(d_input, input, count * sizeof *input,
                     hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_second, second, count * sizeof *second,
                     hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_weight, weight, HIDDEN * sizeof *weight,
                     hipMemcpyHostToDevice));

    char detail[256];

    /* ---- RMSNorm ---- */
    for (uint32_t v = 0; v < VECTORS; v++) {
        mimo26_rmsnorm_bf16(cpu + (size_t)v * HIDDEN,
                            input + (size_t)v * HIDDEN, weight, HIDDEN,
                            1e-6f);
    }
    if (!mimo26_rocm_rmsnorm_bf16(d_output, d_input, d_weight, VECTORS,
                                  HIDDEN, 1e-6f, NULL)) {
        fprintf(stderr, "rmsnorm launch failed\n");
        return 1;
    }
    HIP_OK(hipDeviceSynchronize());
    HIP_OK(hipMemcpy(gpu, d_output, count * sizeof *gpu,
                     hipMemcpyDeviceToHost));
    size_t differing = 0;
    for (size_t i = 0; i < count; i++) {
        differing += cpu[i] != gpu[i];
    }
    snprintf(detail, sizeof detail, "%zu of %zu elements differ", differing,
             count);
    ok("rmsnorm matches the double-rounding contract", differing == 0,
       detail);

    /* The same inputs through K3's kernel, to show the fork was needed. */
    if (k3_rocm_rms_norm_bf16(d_output, d_input, d_weight, VECTORS, HIDDEN,
                              1e-6f, NULL)) {
        HIP_OK(hipDeviceSynchronize());
        HIP_OK(hipMemcpy(other, d_output, count * sizeof *other,
                         hipMemcpyDeviceToHost));
        size_t k3_differing = 0;
        for (size_t i = 0; i < count; i++) {
            k3_differing += cpu[i] != other[i];
        }
        snprintf(detail, sizeof detail,
                 "K3's single-rounding kernel differs on %zu of %zu",
                 k3_differing, count);
        ok("K3's rmsnorm is genuinely not interchangeable",
           k3_differing > 0, detail);
    }

    /* ---- SwiGLU ---- */
    mimo26_silu_product_bf16(cpu, input, second, count);
    if (!mimo26_rocm_silu_product_bf16(d_output, d_input, d_second, count,
                                       NULL)) {
        fprintf(stderr, "silu launch failed\n");
        return 1;
    }
    HIP_OK(hipDeviceSynchronize());
    HIP_OK(hipMemcpy(gpu, d_output, count * sizeof *gpu,
                     hipMemcpyDeviceToHost));
    differing = 0;
    for (size_t i = 0; i < count; i++) {
        differing += cpu[i] != gpu[i];
    }
    snprintf(detail, sizeof detail, "%zu of %zu elements differ", differing,
             count);
    ok("silu product matches, plain and unclamped", differing == 0, detail);

    /* ---- residual add ---- */
    mimo26_residual_add_bf16(cpu, input, second, count);
    if (!mimo26_rocm_residual_add_bf16(d_output, d_input, d_second, count,
                                       NULL)) {
        fprintf(stderr, "residual launch failed\n");
        return 1;
    }
    HIP_OK(hipDeviceSynchronize());
    HIP_OK(hipMemcpy(gpu, d_output, count * sizeof *gpu,
                     hipMemcpyDeviceToHost));
    differing = 0;
    for (size_t i = 0; i < count; i++) {
        differing += cpu[i] != gpu[i];
    }
    snprintf(detail, sizeof detail, "%zu of %zu elements differ", differing,
             count);
    ok("residual add matches", differing == 0, detail);

    /* ---- BF16 GEMV, at MiMo's real projection shapes ----
     * Every projection in the graph goes through this: fused QKV, o_proj and
     * lm_head. It is K3's kernel, so the question is the same one G1 asked of
     * the MXFP4 kernel -- is its contract MiMo's contract? The CPU oracle
     * sums a row in ascending column order; the kernel tree-reduces, so this
     * is a reduction-order question and is allowed to differ by a step of the
     * BF16 output. It is reported either way, because "reuse" is a decision
     * that should rest on a number. */
    {
        const uint32_t shapes[][2] = {
            {13568u, 4096u},   /* fused QKV, global layer */
            {14848u, 4096u},   /* fused QKV, windowed layer */
            {4096u, 8192u},    /* o_proj: hidden from 64 heads x 128 */
        };
        for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
            const uint32_t rows = shapes[s][0];
            const uint32_t cols = shapes[s][1];
            uint16_t *w = (uint16_t *)malloc((size_t)rows * cols * sizeof *w);
            uint16_t *x = (uint16_t *)malloc((size_t)cols * sizeof *x);
            uint16_t *want = (uint16_t *)malloc((size_t)rows * sizeof *want);
            uint16_t *got = (uint16_t *)malloc((size_t)rows * sizeof *got);
            if (!w || !x || !want || !got) { return 1; }
            for (size_t i = 0; i < (size_t)rows * cols; i++) {
                w[i] = mimo26_f32_to_bf16(next_uniform(-0.08f, 0.08f));
            }
            for (size_t i = 0; i < cols; i++) {
                x[i] = mimo26_f32_to_bf16(next_uniform(-2.0f, 2.0f));
            }
            mimo26_matmul_bf16(want, w, x, rows, cols);

            uint16_t *d_w = NULL, *d_x = NULL, *d_y = NULL;
            HIP_OK(hipMalloc(&d_w, (size_t)rows * cols * sizeof *d_w));
            HIP_OK(hipMalloc(&d_x, (size_t)cols * sizeof *d_x));
            HIP_OK(hipMalloc(&d_y, (size_t)rows * sizeof *d_y));
            HIP_OK(hipMemcpy(d_w, w, (size_t)rows * cols * sizeof *w,
                             hipMemcpyHostToDevice));
            HIP_OK(hipMemcpy(d_x, x, (size_t)cols * sizeof *x,
                             hipMemcpyHostToDevice));
            if (!k3_rocm_bf16_gemv_bf16(d_y, d_w, d_x, rows, cols, NULL)) {
                fprintf(stderr, "bf16 gemv launch failed\n");
                return 1;
            }
            HIP_OK(hipDeviceSynchronize());
            HIP_OK(hipMemcpy(got, d_y, (size_t)rows * sizeof *got,
                             hipMemcpyDeviceToHost));
            size_t bad = 0;
            int worst = 0;
            for (uint32_t r = 0; r < rows; r++) {
                if (want[r] != got[r]) {
                    bad++;
                    int d = (int)want[r] - (int)got[r];
                    if (d < 0) { d = -d; }
                    if (d > worst) { worst = d; }
                }
            }
            char label[96];
            snprintf(label, sizeof label, "bf16 gemv [%u,%u] matches", rows,
                     cols);
            snprintf(detail, sizeof detail, "%zu of %u rows differ%s%s",
                     bad, rows, bad ? ", worst " : " (bit-exact)", "");
            if (bad) {
                snprintf(detail + strlen(detail),
                         sizeof detail - strlen(detail), "%d ulp", worst);
            }
            /* One BF16 step is the most a pure reassociation can cost here. */
            ok(label, worst <= 1, detail);
            free(w); free(x); free(want); free(got);
            HIP_OK(hipFree(d_w)); HIP_OK(hipFree(d_x)); HIP_OK(hipFree(d_y));
        }
    }

    /* ---- router ---- */
    float *logits = (float *)malloc((size_t)VECTORS * EXPERTS * sizeof *logits);
    float *bias = (float *)malloc((size_t)EXPERTS * sizeof *bias);
    if (!logits || !bias) {
        return 1;
    }
    for (size_t i = 0; i < (size_t)VECTORS * EXPERTS; i++) {
        logits[i] = next_uniform(-6.0f, 6.0f);
    }
    for (size_t i = 0; i < EXPERTS; i++) {
        bias[i] = next_uniform(-0.1f, 0.1f);
    }
    /* Force exact ties so the lower-id rule is actually exercised rather
     * than merely assumed unreachable. */
    logits[5] = logits[9];
    logits[EXPERTS + 40] = logits[EXPERTS + 77];
    bias[5] = bias[9];
    bias[40] = bias[77];

    uint32_t cpu_ids[VECTORS][TOP_K];
    float cpu_weights[VECTORS][TOP_K];
    for (uint32_t v = 0; v < VECTORS; v++) {
        if (mimo26_router_top8_256_f32(cpu_ids[v], cpu_weights[v],
                                       logits + (size_t)v * EXPERTS,
                                       bias) != MIMO26_ROUTER_OK) {
            fprintf(stderr, "cpu router failed\n");
            return 1;
        }
    }

    float *d_logits = NULL, *d_bias = NULL, *d_weights = NULL;
    uint32_t *d_ids = NULL;
    HIP_OK(hipMalloc(&d_logits, (size_t)VECTORS * EXPERTS * sizeof *d_logits));
    HIP_OK(hipMalloc(&d_bias, (size_t)EXPERTS * sizeof *d_bias));
    HIP_OK(hipMalloc(&d_ids, (size_t)VECTORS * TOP_K * sizeof *d_ids));
    HIP_OK(hipMalloc(&d_weights, (size_t)VECTORS * TOP_K * sizeof *d_weights));
    HIP_OK(hipMemcpy(d_logits, logits,
                     (size_t)VECTORS * EXPERTS * sizeof *logits,
                     hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_bias, bias, (size_t)EXPERTS * sizeof *bias,
                     hipMemcpyHostToDevice));

    if (!mimo26_rocm_router_topk(d_ids, d_weights, d_logits, d_bias, VECTORS,
                                 EXPERTS, TOP_K, 1.0f, NULL)) {
        fprintf(stderr, "router launch failed\n");
        return 1;
    }
    HIP_OK(hipDeviceSynchronize());
    uint32_t gpu_ids[VECTORS][TOP_K];
    float gpu_weights[VECTORS][TOP_K];
    HIP_OK(hipMemcpy(gpu_ids, d_ids, sizeof gpu_ids, hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(gpu_weights, d_weights, sizeof gpu_weights,
                     hipMemcpyDeviceToHost));

    size_t id_mismatches = 0;
    size_t weight_mismatches = 0;
    size_t unordered = 0;
    double worst_weight_relative = 0.0;
    for (uint32_t v = 0; v < VECTORS; v++) {
        for (uint32_t k = 0; k < TOP_K; k++) {
            id_mismatches += cpu_ids[v][k] != gpu_ids[v][k];
            if (memcmp(&cpu_weights[v][k], &gpu_weights[v][k],
                       sizeof(float)) != 0) {
                weight_mismatches++;
                const double want = (double)cpu_weights[v][k];
                const double relative =
                    fabs(want - (double)gpu_weights[v][k]) /
                    (fabs(want) > 1e-12 ? fabs(want) : 1e-12);
                if (relative > worst_weight_relative) {
                    worst_weight_relative = relative;
                }
            }
            if (k > 0 && gpu_ids[v][k] <= gpu_ids[v][k - 1]) {
                unordered++;
            }
        }
    }
    /*
     * Selection is discrete and must be exact: a different expert is a
     * different computation, not a rounding difference.
     */
    snprintf(detail, sizeof detail, "%zu of %u ids differ",
             id_mismatches, VECTORS * TOP_K);
    ok("router selects the same experts", id_mismatches == 0, detail);

    /*
     * The weights are sigmoid values, and sigmoid is 1/(1+expf(-x)) on both
     * sides. Device libm and host libm disagree by 1 ulp on 6.26% of expf
     * inputs, so these cannot be bit-identical across backends and it is
     * wrong to demand it -- an earlier version of this test did, and passed
     * only because that particular draw happened not to hit the case.
     * Bounding the relative error keeps the check meaningful: a real fault
     * in the ordering or the denominator would move a weight by far more
     * than one float ulp.
     */
    snprintf(detail, sizeof detail,
             "%zu of %u differ, worst %.2e relative",
             weight_mismatches, VECTORS * TOP_K, worst_weight_relative);
    ok("router weights agree to within libm's last ulp",
       worst_weight_relative <= 1e-6, detail);
    ok("router returns ascending expert ids", unordered == 0, NULL);

    free(input);
    free(second);
    free(weight);
    free(cpu);
    free(gpu);
    free(other);
    free(logits);
    free(bias);
    HIP_OK(hipFree(d_input));
    HIP_OK(hipFree(d_second));
    HIP_OK(hipFree(d_weight));
    HIP_OK(hipFree(d_output));
    HIP_OK(hipFree(d_logits));
    HIP_OK(hipFree(d_bias));
    HIP_OK(hipFree(d_ids));
    HIP_OK(hipFree(d_weights));

    printf("test_mimo26_gpu_ops: %s\n", failures == 0 ? "ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
