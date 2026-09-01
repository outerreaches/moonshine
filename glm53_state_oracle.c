#include "glm53_state_oracle.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

static bool mul_size(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) return false;
    *out = a * b;
    return true;
}
static bool add_size(size_t a, size_t b, size_t *out) {
    if (b > SIZE_MAX - a) return false;
    *out = a + b;
    return true;
}
static bool byte_count(size_t count, size_t element_size, size_t *out) {
    return mul_size(count, element_size, out);
}
static bool ranges_overlap(const void *a, size_t an, const void *b, size_t bn) {
    uintptr_t aa, bb;
    if (an == 0 || bn == 0) return false;
    aa = (uintptr_t)a;
    bb = (uintptr_t)b;
    /* Overflow is treated conservatively as overlap/invalid. */
    if (aa > UINTPTR_MAX - an || bb > UINTPTR_MAX - bn) return true;
    return aa < bb + bn && bb < aa + an;
}

size_t glm53_kda_scratch_floats(size_t key_dim, size_t value_dim,
                                size_t steps) {
    size_t state_n, output_n, n;
    if (key_dim == 0 || value_dim == 0 || steps == 0) return 0;
    if (!mul_size(key_dim, value_dim, &state_n) ||
        !mul_size(steps, value_dim, &output_n) ||
        !add_size(state_n, key_dim, &n) ||
        !add_size(n, key_dim, &n) ||
        !add_size(n, output_n, &n)) return 0;
    return n;
}

static bool finite_array(const float *x, size_t n) {
    size_t i;
    if (!x) return false;
    for (i = 0; i < n; ++i) if (!isfinite(x[i])) return false;
    return true;
}

static bool kda_chunk_impl(const float *q, const float *k, const float *v,
                           const float *g, const float *beta, size_t steps,
                           float *state, size_t kd, size_t vd, float *output,
                           float *scratch, size_t scratch_floats) {
    size_t sk, sv, sg, sb, state_n, out_n, need, scratch_bytes;
    size_t q_bytes, v_bytes, state_bytes, out_bytes, beta_bytes;
    size_t t, i, j;
    float *shadow, *qn, *kn, *staged;

    need = glm53_kda_scratch_floats(kd, vd, steps);
    if (need == 0 || scratch_floats < need || !state || !output || !scratch ||
        !q || !k || !v || !g || !beta) return false;
    if (!mul_size(steps, kd, &sk) || !mul_size(steps, vd, &sv) ||
        !mul_size(kd, vd, &state_n) || !mul_size(steps, vd, &out_n) ||
        !byte_count(sk, sizeof(float), &q_bytes) ||
        !byte_count(sv, sizeof(float), &v_bytes) ||
        !byte_count(state_n, sizeof(float), &state_bytes) ||
        !byte_count(out_n, sizeof(float), &out_bytes) ||
        !byte_count(steps, sizeof(float), &beta_bytes) ||
        !byte_count(need, sizeof(float), &scratch_bytes)) return false;
    sg = sk; sb = steps;

    /* Writable regions must be unambiguous. Scratch must be private. */
    if (ranges_overlap(state, state_bytes, output, out_bytes) ||
        ranges_overlap(scratch, scratch_bytes, state, state_bytes) ||
        ranges_overlap(scratch, scratch_bytes, output, out_bytes) ||
        ranges_overlap(scratch, scratch_bytes, q, q_bytes) ||
        ranges_overlap(scratch, scratch_bytes, k, q_bytes) ||
        ranges_overlap(scratch, scratch_bytes, v, v_bytes) ||
        ranges_overlap(scratch, scratch_bytes, g, q_bytes) ||
        ranges_overlap(scratch, scratch_bytes, beta, beta_bytes)) return false;

    if (!finite_array(q, sk) || !finite_array(k, sk) ||
        !finite_array(v, sv) || !finite_array(g, sg) ||
        !finite_array(beta, sb) || !finite_array(state, state_n)) return false;

    shadow = scratch;
    qn = shadow + state_n;
    kn = qn + kd;
    staged = kn + kd;
    memcpy(shadow, state, state_bytes);

    for (t = 0; t < steps; ++t) {
        float qss = 0.0f, kss = 0.0f, qden, kden, qscale;
        for (i = 0; i < kd; ++i) {
            float qx = q[t * kd + i], kx = k[t * kd + i];
            qss += qx * qx;
            kss += kx * kx;
        }
        qden = sqrtf(qss + 1.0e-6f);
        kden = sqrtf(kss + 1.0e-6f);
        qscale = sqrtf((float)kd);
        if (!isfinite(qden) || !isfinite(kden) || !isfinite(qscale) ||
            qden == 0.0f || kden == 0.0f || qscale == 0.0f) return false;
        for (i = 0; i < kd; ++i) {
            qn[i] = (q[t * kd + i] / qden) / qscale;
            kn[i] = k[t * kd + i] / kden;
            if (!isfinite(qn[i]) || !isfinite(kn[i])) return false;
        }
        /* HF recurrent KDA: each key row has its own forget gate. */
        for (i = 0; i < kd; ++i) {
            float decay = expf(g[t * kd + i]);
            if (!isfinite(decay)) return false;
            for (j = 0; j < vd; ++j) {
                float x = shadow[i * vd + j] * decay;
                if (!isfinite(x)) return false;
                shadow[i * vd + j] = x;
            }
        }
        for (j = 0; j < vd; ++j) {
            float prediction = 0.0f, delta, y = 0.0f;
            for (i = 0; i < kd; ++i)
                prediction += shadow[i * vd + j] * kn[i];
            delta = (v[t * vd + j] - prediction) * beta[t];
            if (!isfinite(prediction) || !isfinite(delta)) return false;
            for (i = 0; i < kd; ++i) {
                float x = shadow[i * vd + j] + kn[i] * delta;
                if (!isfinite(x)) return false;
                shadow[i * vd + j] = x;
            }
            for (i = 0; i < kd; ++i) y += qn[i] * shadow[i * vd + j];
            if (!isfinite(y)) return false;
            staged[t * vd + j] = y;
        }
    }
    memcpy(state, shadow, state_bytes);
    memcpy(output, staged, out_bytes);
    return true;
}

bool glm53_kda_step_f32(const float *q, const float *k, const float *v,
                        const float *g, float beta, float *state,
                        size_t key_dim, size_t value_dim, float *output,
                        float *scratch, size_t scratch_floats) {
    return kda_chunk_impl(q, k, v, g, &beta, 1, state, key_dim, value_dim,
                          output, scratch, scratch_floats);
}

bool glm53_kda_chunk_f32(const float *q, const float *k, const float *v,
                         const float *g, const float *beta, size_t steps,
                         float *state, size_t key_dim, size_t value_dim,
                         float *output, float *scratch,
                         size_t scratch_floats) {
    return kda_chunk_impl(q, k, v, g, beta, steps, state, key_dim,
                          value_dim, output, scratch, scratch_floats);
}

static bool mla_index(size_t heads, size_t kd, size_t vd, size_t rank,
                      size_t head, size_t channel, size_t rank_channel,
                      bool value, size_t *flat) {
    size_t width, n, x;
    if (!flat || heads == 0 || kd == 0 || vd == 0 || rank == 0 ||
        head >= heads || rank_channel >= rank ||
        (!value && channel >= kd) || (value && channel >= vd) ||
        !add_size(kd, vd, &width)) return false;
    if (value && !add_size(kd, channel, &channel)) return false;
    if (!mul_size(head, width, &x) || !add_size(x, channel, &x) ||
        !mul_size(x, rank, &x) || !add_size(x, rank_channel, &x) ||
        !mul_size(heads, width, &n) || !mul_size(n, rank, &n) || x >= n)
        return false;
    *flat = x;
    return true;
}
bool glm53_mla_kv_b_key_index(size_t heads, size_t kd, size_t vd,
                              size_t rank, size_t head, size_t channel,
                              size_t r, size_t *flat) {
    return mla_index(heads, kd, vd, rank, head, channel, r, false, flat);
}
bool glm53_mla_kv_b_value_index(size_t heads, size_t kd, size_t vd,
                                size_t rank, size_t head, size_t channel,
                                size_t r, size_t *flat) {
    return mla_index(heads, kd, vd, rank, head, channel, r, true, flat);
}

bool glm53_index_pool4_build(const uint8_t *valid, size_t length,
                             size_t visible_through,
                             glm53_index_pool4 *pools, size_t pool_capacity,
                             size_t *tail_indices, size_t tail_capacity,
                             glm53_index_pool4_metadata *metadata) {
    size_t first = length, pool_count = 0, visible_count = 0;
    size_t tail_count, tail_start, i, start, pi;
    size_t pools_bytes, tail_bytes, valid_bytes = length;
    if (!metadata || (length != 0 && !valid)) return false;
    for (i = 0; i < length; ++i) {
        if (valid[i] > 1) return false;
        if (valid[i] && first == length) first = i;
        if (valid[i] && visible_through != SIZE_MAX && i <= visible_through)
            ++visible_count;
    }
    if (first != length) {
        for (start = first; start <= length - 1 && length - start >= 4;
             start += 4) {
            bool all = true;
            for (i = 0; i < 4; ++i) all = all && valid[start + i] != 0;
            if (all) ++pool_count;
        }
    }
    tail_count = visible_count % 4;
    tail_start = first;
    if (first != length) {
        if (!add_size(first, visible_count - tail_count, &tail_start)) return false;
        /* HF masks invalid/non-visible tail slots. Emit only real ones. */
        for (i = 0; i < tail_count; ++i) {
            size_t x;
            if (!add_size(tail_start, i, &x) || x >= length || !valid[x] ||
                visible_through == SIZE_MAX || x > visible_through) {
                tail_count = i;
                break;
            }
        }
    } else tail_count = 0;
    if (pool_count > pool_capacity || tail_count > tail_capacity ||
        (pool_count && !pools) || (tail_count && !tail_indices)) return false;
    if (!byte_count(pool_count, sizeof(*pools), &pools_bytes) ||
        !byte_count(tail_count, sizeof(*tail_indices), &tail_bytes) ||
        ranges_overlap(metadata, sizeof(*metadata), pools, pools_bytes) ||
        ranges_overlap(metadata, sizeof(*metadata), tail_indices, tail_bytes) ||
        ranges_overlap(pools, pools_bytes, tail_indices, tail_bytes) ||
        ranges_overlap(pools, pools_bytes, valid, valid_bytes) ||
        ranges_overlap(tail_indices, tail_bytes, valid, valid_bytes)) return false;

    pi = 0;
    if (first != length) {
        for (start = first; start <= length - 1 && length - start >= 4;
             start += 4) {
            bool all = true;
            for (i = 0; i < 4; ++i) all = all && valid[start + i] != 0;
            if (all) {
                pools[pi].start = start;
                pools[pi].end = start + 3;
                pools[pi].end_visible = visible_through != SIZE_MAX &&
                                         start + 3 <= visible_through;
                ++pi;
            }
        }
    }
    for (i = 0; i < tail_count; ++i) tail_indices[i] = tail_start + i;
    metadata->first_valid = first;
    metadata->pool_count = pool_count;
    metadata->tail_count = tail_count;
    return true;
}
