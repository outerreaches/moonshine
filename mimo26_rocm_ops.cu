#include "mimo26_rocm_ops.h"

#include <hip/hip_runtime.h>

#include <stdio.h>
#include <stdlib.h>

#define MIMO26_ROCM_THREADS 256u
#define MIMO26_ROCM_MAX_EXPERTS 1024u

/*
 * BF16 conversion is written out rather than taken from hip_bfloat16 so the
 * rounding is visibly round-to-nearest-even and identical to the CPU path's
 * mimo26_f32_to_bf16. The contracts this file exists to honour are rounding
 * contracts; hiding the rounding behind an operator would defeat the point.
 */
__device__ static inline float mimo26_bf16_to_f32_d(uint16_t bits)
{
    const uint32_t widened = (uint32_t)bits << 16;
    return __uint_as_float(widened);
}

__device__ static inline uint16_t mimo26_f32_to_bf16_d(float value)
{
    const uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u) {
        return (uint16_t)((bits >> 16) | 0x0040u);   /* quiet NaN */
    }
    const uint32_t rounding = ((bits >> 16) & 1u) + 0x7FFFu;
    return (uint16_t)((bits + rounding) >> 16);
}

__global__ static void mimo26_rmsnorm_kernel(uint16_t *output,
                                             const uint16_t *input,
                                             const uint16_t *weight,
                                             uint32_t hidden_size,
                                             float epsilon)
{
    const uint32_t vector = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint64_t base = (uint64_t)vector * hidden_size;
    __shared__ float reduction[MIMO26_ROCM_THREADS];

    /*
     * Summed by one thread in ascending order, not tree-reduced.
     *
     * The sum itself is well conditioned -- every term is a square -- so a
     * tree reduction lands within an ulp, and in isolation that is
     * invisible: this kernel passed a 12,288-element bit-exactness check
     * against the CPU while tree-reducing. It stopped being invisible in
     * composition. A single differing ulp in the reciprocal moves a handful
     * of normalized values by one BF16 step, and the router's dot product
     * cancels heavily enough to turn that into a 4.5e-04 shift in the
     * mixing weights, which then reaches every element of the expert sum.
     *
     * 4096 serial adds, twice per layer per token: ~400k operations against
     * the 4.68 GiB of expert reads in the same token. The exactness is free
     * and the amplification is not hypothetical.
     */
    if (tid == 0u) {
        float sum_squares = 0.0f;
        for (uint32_t d = 0; d < hidden_size; d++) {
            const float value = mimo26_bf16_to_f32_d(input[base + d]);
            sum_squares += value * value;
        }
        reduction[0] = sum_squares;
    }
    __syncthreads();
    const float reciprocal =
        rsqrtf(reduction[0] / (float)hidden_size + epsilon);

    for (uint32_t d = tid; d < hidden_size; d += blockDim.x) {
        const float scaled = mimo26_bf16_to_f32_d(input[base + d]) * reciprocal;
        /* The rounding that distinguishes this from K3's kernel: the
         * normalized value becomes BF16 before it meets the weight. */
        const float rounded = mimo26_bf16_to_f32_d(mimo26_f32_to_bf16_d(scaled));
        const float product = rounded * mimo26_bf16_to_f32_d(weight[d]);
        output[base + d] = mimo26_f32_to_bf16_d(product);
    }
}

__global__ static void mimo26_silu_product_kernel(uint16_t *output,
                                                  const uint16_t *gate,
                                                  const uint16_t *up,
                                                  uint64_t count)
{
    const uint64_t index =
        (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const float g = mimo26_bf16_to_f32_d(gate[index]);
    /* expf, not __expf: the fast intrinsic is a different function and this
     * module exists to match the CPU path's arithmetic exactly. */
    const float activated = g / (1.0f + expf(-g));
    /* silu rounds to BF16, then the product rounds again. */
    const float rounded =
        mimo26_bf16_to_f32_d(mimo26_f32_to_bf16_d(activated));
    output[index] =
        mimo26_f32_to_bf16_d(rounded * mimo26_bf16_to_f32_d(up[index]));
}

__global__ static void mimo26_residual_add_kernel(uint16_t *output,
                                                  const uint16_t *residual,
                                                  const uint16_t *delta,
                                                  uint64_t count)
{
    const uint64_t index =
        (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    output[index] = mimo26_f32_to_bf16_d(
        mimo26_bf16_to_f32_d(residual[index]) +
        mimo26_bf16_to_f32_d(delta[index]));
}

__global__ static void mimo26_router_topk_kernel(uint32_t *expert_ids,
                                                 float *expert_weights,
                                                 const float *logits,
                                                 const float *correction_bias,
                                                 uint32_t expert_count,
                                                 uint32_t top_k, float scale)
{
    const uint32_t vector = blockIdx.x;
    expert_ids += (uint64_t)vector * top_k;
    expert_weights += (uint64_t)vector * top_k;
    logits += (uint64_t)vector * expert_count;

    __shared__ float scores[MIMO26_ROCM_MAX_EXPERTS];
    __shared__ float choices[MIMO26_ROCM_MAX_EXPERTS];
    __shared__ uint32_t ids[MIMO26_ROCM_MAX_EXPERTS];

    for (uint32_t slot = threadIdx.x; slot < MIMO26_ROCM_MAX_EXPERTS;
         slot += blockDim.x) {
        if (slot < expert_count) {
            const float score = 1.0f / (1.0f + expf(-logits[slot]));
            scores[slot] = score;
            choices[slot] = score + correction_bias[slot];
            ids[slot] = slot;
        } else {
            scores[slot] = 0.0f;
            choices[slot] = -INFINITY;
            ids[slot] = UINT32_MAX;
        }
    }
    __syncthreads();

    /* Deterministic bitonic sort, ascending by (choice, then id descending),
     * so the tail holds the selection in choice-descending, id-ascending
     * order and a tie resolves to the lower id. */
    for (uint32_t width = 2u; width <= MIMO26_ROCM_MAX_EXPERTS;
         width <<= 1u) {
        for (uint32_t stride = width >> 1u; stride > 0u; stride >>= 1u) {
            for (uint32_t left = threadIdx.x; left < MIMO26_ROCM_MAX_EXPERTS;
                 left += blockDim.x) {
                const uint32_t right = left ^ stride;
                if (right <= left) {
                    continue;
                }
                const bool ascending = (left & width) == 0u;
                const float left_choice = choices[left];
                const float right_choice = choices[right];
                const uint32_t left_id = ids[left];
                const uint32_t right_id = ids[right];
                const bool left_after =
                    left_choice > right_choice ||
                    (left_choice == right_choice && left_id < right_id);
                const bool left_before =
                    left_choice < right_choice ||
                    (left_choice == right_choice && left_id > right_id);
                if (ascending ? left_after : left_before) {
                    choices[left] = right_choice;
                    choices[right] = left_choice;
                    const float swap_score = scores[left];
                    scores[left] = scores[right];
                    scores[right] = swap_score;
                    ids[left] = right_id;
                    ids[right] = left_id;
                }
            }
            __syncthreads();
        }
    }

    if (threadIdx.x != 0u) {
        return;
    }
    /*
     * MiMo declares ascending expert id for the returned indices AND for the
     * order the denominator accumulates in. Sorting the selection by id
     * before summing is the whole difference from K3's kernel, and it is a
     * real one: float addition is not associative, so summing in
     * score-descending order gives a denominator that can differ in the last
     * ulp, and that difference reaches every mixing weight.
     */
    uint32_t chosen[64];
    float chosen_score[64];
    for (uint32_t rank = 0; rank < top_k; rank++) {
        const uint32_t source = MIMO26_ROCM_MAX_EXPERTS - 1u - rank;
        uint32_t id = ids[source];
        float score = scores[source];
        uint32_t slot = rank;
        while (slot > 0u && chosen[slot - 1u] > id) {
            chosen[slot] = chosen[slot - 1u];
            chosen_score[slot] = chosen_score[slot - 1u];
            slot--;
        }
        chosen[slot] = id;
        chosen_score[slot] = score;
    }

    float denominator = 1e-20f;
    for (uint32_t rank = 0; rank < top_k; rank++) {
        denominator += chosen_score[rank];
    }
    for (uint32_t rank = 0; rank < top_k; rank++) {
        expert_ids[rank] = chosen[rank];
        expert_weights[rank] = chosen_score[rank] / denominator * scale;
    }
}

__global__ static void mimo26_expert_accumulate_kernel(
        float *accumulator, const uint16_t *expert_output, float weight,
        uint64_t count)
{
    const uint64_t index =
        (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    accumulator[index] += weight * mimo26_bf16_to_f32_d(expert_output[index]);
}

__global__ static void mimo26_expert_finalize_kernel(uint16_t *output,
                                                     const float *accumulator,
                                                     uint64_t count)
{
    const uint64_t index =
        (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    output[index] = mimo26_f32_to_bf16_d(accumulator[index]);
}

__global__ static void mimo26_zero_kernel(float *accumulator, uint64_t count)
{
    const uint64_t index =
        (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        accumulator[index] = 0.0f;
    }
}

#define MIMO26_ROCM_QUERY_HEADS 64u
#define MIMO26_ROCM_QK_DIM 192u
#define MIMO26_ROCM_V_DIM 128u

__device__ static inline bool mimo26_visible_d(uint64_t kv_position,
                                               uint64_t query_position,
                                               uint32_t window)
{
    if (kv_position > query_position) {
        return false;                       /* causal */
    }
    if (window == 0u) {
        return true;                        /* full attention */
    }
    /* kv > q - window, written to avoid unsigned wrap at small q. */
    return query_position - kv_position < (uint64_t)window;
}

/*
 * One block per query head. See the header for why the decomposition is what
 * it is: every reduction that the CPU performs in a specific order is
 * performed in that same order here.
 */
/*
 * Prefill reuses this kernel unchanged, with blockIdx.y selecting the query.
 *
 * That is deliberate rather than convenient: the masking already works on
 * absolute positions, so a chunk of tokens whose keys and values are simply
 * appended to the history is masked causally by the same predicate that
 * masks a single decode step. Writing a second kernel would mean two copies
 * of the cast points and the summation order, and the guarantee that
 * prefilling N tokens equals decoding them one at a time would become a hope.
 */
__global__ static void mimo26_attention_decode_kernel(
        uint16_t *out, const uint16_t *query, const uint16_t *keys,
        const uint16_t *values, const uint16_t *current_keys,
        const uint16_t *current_values, const uint16_t *sink_bias,
        float *scratch, uint32_t kv_heads, uint32_t kv_groups,
        uint32_t window, uint64_t history, uint64_t first_position,
        uint64_t query_position, float scale, uint32_t have_current,
        uint32_t has_sink, uint32_t scores_only)
{
    /* Batch index: 0 for decode, the token's offset within the chunk for
     * prefill. Each advances the query, the output and the scratch row, and
     * shifts this query's absolute position. */
    const uint32_t batch = blockIdx.y;
    query += (uint64_t)batch * MIMO26_ROCM_QUERY_HEADS * MIMO26_ROCM_QK_DIM;
    out += (uint64_t)batch * MIMO26_ROCM_QUERY_HEADS * MIMO26_ROCM_V_DIM;
    scratch += (uint64_t)batch * MIMO26_ROCM_QUERY_HEADS * (history + 2u);
    query_position += batch;

    const uint32_t head = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t kv_head = head / kv_groups;
    const uint64_t slot_total = history + (have_current ? 1u : 0u);
    const uint64_t slots = slot_total + (has_sink ? 1u : 0u);

    const uint16_t *q_head = query + (uint64_t)head * MIMO26_ROCM_QK_DIM;
    float *row = scratch + (uint64_t)head * (history + 2u);

    __shared__ float reduction[MIMO26_ROCM_THREADS];
    __shared__ float shared_max;
    __shared__ double shared_total;

    /* Pass 1: scores. Each thread owns whole slots, so each 192-term dot is
     * summed sequentially in ascending coordinate order, as on the CPU. */
    float local_max = -INFINITY;
    for (uint64_t t = tid; t < slot_total; t += blockDim.x) {
        const bool is_current = (have_current && t == history);
        const uint64_t kv_position =
            is_current ? query_position : first_position + t;
        if (!mimo26_visible_d(kv_position, query_position, window)) {
            row[t] = -INFINITY;
            continue;
        }
        const uint16_t *k_head =
            is_current
                ? current_keys + (uint64_t)kv_head * MIMO26_ROCM_QK_DIM
                : keys + (t * kv_heads + kv_head) * MIMO26_ROCM_QK_DIM;
        float dot = 0.0f;
        for (uint32_t d = 0; d < MIMO26_ROCM_QK_DIM; d++) {
            dot += mimo26_bf16_to_f32_d(q_head[d]) *
                   mimo26_bf16_to_f32_d(k_head[d]);
        }
        /* The reference's matmul output is BF16 and the scaling stays in that
         * dtype, so round here rather than carrying F32 forward. */
        const float scaled =
            mimo26_bf16_to_f32_d(mimo26_f32_to_bf16_d(dot * scale));
        row[t] = scaled;
        if (scaled > local_max) {
            local_max = scaled;
        }
    }
    /* Max is associative and commutative, so a tree reduction is safe here
     * in a way the additions are not. */
    reduction[tid] = local_max;
    __syncthreads();
    for (uint32_t width = blockDim.x / 2u; width > 0u; width /= 2u) {
        if (tid < width && reduction[tid + width] > reduction[tid]) {
            reduction[tid] = reduction[tid + width];
        }
        __syncthreads();
    }

    if (tid == 0u) {
        float maximum = reduction[0];
        if (has_sink) {
            const float sink = mimo26_bf16_to_f32_d(sink_bias[head]);
            row[slot_total] = sink;
            if (sink > maximum) {
                maximum = sink;
            }
        }
        shared_max = maximum;
    }
    __syncthreads();

    /*
     * Stop here when the caller only wants the scores. Everything up to this
     * point -- dot products, the scale and its BF16 rounding, masking, the
     * sink logit and the row max -- is exactly specified arithmetic with no
     * transcendental in it, so it can be held to bit-exactness against the
     * CPU. What follows calls expf, and device libm differs from host libm by
     * up to 1 ulp on about 6% of inputs, so it cannot be. Separating the two
     * keeps a strict gate on every layout and ordering decision instead of
     * hiding them all behind one tolerance.
     */
    if (scores_only) {
        return;
    }

    /* Pass 2: exponentials. Elementwise, so order is irrelevant. */
    const float maximum = shared_max;
    for (uint64_t t = tid; t < slots; t += blockDim.x) {
        if (row[t] == -INFINITY) {
            row[t] = 0.0f;
            continue;
        }
        const float shifted =
            mimo26_bf16_to_f32_d(mimo26_f32_to_bf16_d(row[t] - maximum));
        row[t] = expf(shifted);
    }
    __syncthreads();

    /* The denominator accumulates in double, in ascending slot order, by a
     * single thread. Reassociating this is the one shortcut that would cost
     * bit-exactness for a saving of microseconds. */
    if (tid == 0u) {
        double total = 0.0;
        for (uint64_t t = 0; t < slots; t++) {
            total += (double)row[t];
        }
        shared_total = total;
    }
    __syncthreads();
    const double total = shared_total;

    /* Pass 3: value accumulation. Splitting over the 128 output dimensions
     * rather than over history keeps every thread's summation in ascending
     * history order. The sink slot is excluded, so the probabilities that
     * survive deliberately sum to less than one. */
    for (uint32_t d = tid; d < MIMO26_ROCM_V_DIM; d += blockDim.x) {
        float accumulator = 0.0f;
        for (uint64_t t = 0; t < slot_total; t++) {
            if (row[t] == 0.0f) {
                continue;
            }
            const float probability = mimo26_bf16_to_f32_d(
                mimo26_f32_to_bf16_d((float)((double)row[t] / total)));
            if (probability == 0.0f) {
                continue;
            }
            const uint16_t *v_head =
                (have_current && t == history)
                    ? current_values + (uint64_t)kv_head * MIMO26_ROCM_V_DIM
                    : values + (t * kv_heads + kv_head) * MIMO26_ROCM_V_DIM;
            accumulator += probability * mimo26_bf16_to_f32_d(v_head[d]);
        }
        out[(uint64_t)head * MIMO26_ROCM_V_DIM + d] =
            mimo26_f32_to_bf16_d(accumulator);
    }
}


#define MIMO26_ROCM_ROPE_DIM 64u
#define MIMO26_ROCM_ROPE_PAIRS (MIMO26_ROCM_ROPE_DIM / 2u)
#define MIMO26_ROCM_VALUE_SCALE 0.707f

/*
 * Split the fused QKV output into per-head Q, K and V, scaling V on the way
 * so that anything cached downstream is already pre-scaled -- which is what
 * the reference does, and getting it wrong would make cached and fresh V
 * disagree.
 */
__global__ static void mimo26_split_qkv_kernel(uint16_t *q, uint16_t *k,
                                               uint16_t *v,
                                               const uint16_t *fused,
                                               uint32_t kv_heads)
{
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t q_total = MIMO26_ROCM_QUERY_HEADS * MIMO26_ROCM_QK_DIM;
    const uint32_t k_total = kv_heads * MIMO26_ROCM_QK_DIM;
    const uint32_t v_total = kv_heads * MIMO26_ROCM_V_DIM;
    if (index < q_total) {
        q[index] = fused[index];
    } else if (index < q_total + k_total) {
        k[index - q_total] = fused[index];
    } else if (index < q_total + k_total + v_total) {
        const uint32_t offset = index - q_total - k_total;
        v[offset] = mimo26_f32_to_bf16_d(
            mimo26_bf16_to_f32_d(fused[index]) * MIMO26_ROCM_VALUE_SCALE);
    }
}

/*
 * Rotate the first 64 coordinates of each head, leaving the other 128 alone.
 * Split-half (NeoX):
 *   out[j]    = x[j]*cos[j]    - x[j+32]*sin[j]
 *   out[j+32] = x[j+32]*cos[j] + x[j]*sin[j]
 *
 * Three roundings, as the CPU does: each product rounds to BF16, and so does
 * the sum. The cos/sin tables are built on the host -- they need powf, cosf
 * and sinf, and host and device libm differ in the last ulp, so computing
 * them here would import that difference into every rotated coordinate for
 * no benefit. Built once per position, they are tiny.
 */
__global__ static void mimo26_rope_apply_kernel(uint16_t *heads,
                                                const uint16_t *cos_table,
                                                const uint16_t *sin_table,
                                                uint32_t head_count)
{
    const uint32_t head = blockIdx.x;
    if (head >= head_count) {
        return;
    }
    uint16_t *vector = heads + (uint64_t)head * MIMO26_ROCM_QK_DIM;
    for (uint32_t j = threadIdx.x; j < MIMO26_ROCM_ROPE_PAIRS;
         j += blockDim.x) {
        const float low = mimo26_bf16_to_f32_d(vector[j]);
        const float high =
            mimo26_bf16_to_f32_d(vector[j + MIMO26_ROCM_ROPE_PAIRS]);
        const float c = mimo26_bf16_to_f32_d(cos_table[j]);
        const float s = mimo26_bf16_to_f32_d(sin_table[j]);
        const float lc = mimo26_bf16_to_f32_d(mimo26_f32_to_bf16_d(low * c));
        const float hs = mimo26_bf16_to_f32_d(mimo26_f32_to_bf16_d(high * s));
        const float hc = mimo26_bf16_to_f32_d(mimo26_f32_to_bf16_d(high * c));
        const float ls = mimo26_bf16_to_f32_d(mimo26_f32_to_bf16_d(low * s));
        vector[j] = mimo26_f32_to_bf16_d(lc - hs);
        vector[j + MIMO26_ROCM_ROPE_PAIRS] = mimo26_f32_to_bf16_d(hc + ls);
    }
}


/*
 * F32 GEMV whose rows are summed in ascending column order, matching the CPU.
 *
 * Two rows in this model need that and K3's tree-reducing GEMV cannot give
 * it. Router logits land near -2 while their summands total far more in
 * absolute value, so the sum cancels and association matters: measured
 * through the mixing weights, tree reduction sat 4.5e-04 from the CPU, about
 * 3800x libm's last ulp and enough to move every element of an
 * expert-weighted sum. The lm_head rows have the same shape of problem and
 * their output is the answer itself, where a reassociation can flip a
 * near-tie into a different token.
 *
 * One block per row. Every thread helps stage the row coalesced into shared
 * memory, then one thread sums it in order. The naive alternative -- one
 * thread per row walking global memory -- strides by a whole row between
 * neighbouring threads and wastes almost all of the bandwidth, which matters
 * at lm_head's 152,576 rows and 1.25 GiB per token.
 */
__global__ static void mimo26_ordered_gemv_f32_kernel(float *output,
                                                      const uint16_t *weights,
                                                      const uint16_t *input,
                                                      uint32_t columns,
                                                      uint32_t round_bf16)
{
    extern __shared__ float staged[];
    const uint32_t row = blockIdx.x;
    const uint16_t *w = weights + (uint64_t)row * columns;

    for (uint32_t c = threadIdx.x; c < columns; c += blockDim.x) {
        staged[c] = mimo26_bf16_to_f32_d(w[c]) * mimo26_bf16_to_f32_d(input[c]);
    }
    __syncthreads();

    if (threadIdx.x == 0u) {
        float sum = 0.0f;
        for (uint32_t c = 0; c < columns; c++) {
            sum += staged[c];
        }
        output[row] = round_bf16
                          ? mimo26_bf16_to_f32_d(mimo26_f32_to_bf16_d(sum))
                          : sum;
    }
}

/*
 * Batched QKV split. blockIdx.y selects the token; each token's fused row is
 * split into that token's slice of the q, k and v arrays.
 */
__global__ static void mimo26_split_qkv_batch_kernel(
        uint16_t *q, uint16_t *k, uint16_t *v, const uint16_t *fused,
        uint32_t kv_heads, uint32_t qkv_width)
{
    const uint32_t token = blockIdx.y;
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t q_total = MIMO26_ROCM_QUERY_HEADS * MIMO26_ROCM_QK_DIM;
    const uint32_t k_total = kv_heads * MIMO26_ROCM_QK_DIM;
    const uint32_t v_total = kv_heads * MIMO26_ROCM_V_DIM;
    fused += (uint64_t)token * qkv_width;
    if (index < q_total) {
        q[(uint64_t)token * q_total + index] = fused[index];
    } else if (index < q_total + k_total) {
        k[(uint64_t)token * k_total + (index - q_total)] = fused[index];
    } else if (index < q_total + k_total + v_total) {
        const uint32_t offset = index - q_total - k_total;
        v[(uint64_t)token * v_total + offset] = mimo26_f32_to_bf16_d(
            mimo26_bf16_to_f32_d(fused[index]) * MIMO26_ROCM_VALUE_SCALE);
    }
}

/*
 * Batched RoPE. Every token in a chunk sits at a different position and so
 * needs its own cos/sin pair; the tables are [tokens][64], built on the host
 * for the same reason the single-token ones are -- powf, cosf and sinf
 * differ between host and device libm in the last ulp, and computing them
 * on device would import that into every rotated coordinate.
 */
__global__ static void mimo26_rope_apply_batch_kernel(
        uint16_t *heads, const uint16_t *cos_tables,
        const uint16_t *sin_tables, uint32_t head_count)
{
    const uint32_t token = blockIdx.y;
    const uint32_t head = blockIdx.x;
    if (head >= head_count) {
        return;
    }
    uint16_t *vector = heads + ((uint64_t)token * head_count + head) *
                                   MIMO26_ROCM_QK_DIM;
    const uint16_t *cos_table = cos_tables +
                                (uint64_t)token * MIMO26_ROCM_ROPE_DIM;
    const uint16_t *sin_table = sin_tables +
                                (uint64_t)token * MIMO26_ROCM_ROPE_DIM;
    for (uint32_t j = threadIdx.x; j < MIMO26_ROCM_ROPE_PAIRS;
         j += blockDim.x) {
        const float low = mimo26_bf16_to_f32_d(vector[j]);
        const float high =
            mimo26_bf16_to_f32_d(vector[j + MIMO26_ROCM_ROPE_PAIRS]);
        const float c = mimo26_bf16_to_f32_d(cos_table[j]);
        const float s = mimo26_bf16_to_f32_d(sin_table[j]);
        const float lc = mimo26_bf16_to_f32_d(mimo26_f32_to_bf16_d(low * c));
        const float hs = mimo26_bf16_to_f32_d(mimo26_f32_to_bf16_d(high * s));
        const float hc = mimo26_bf16_to_f32_d(mimo26_f32_to_bf16_d(high * c));
        const float ls = mimo26_bf16_to_f32_d(mimo26_f32_to_bf16_d(low * s));
        vector[j] = mimo26_f32_to_bf16_d(lc - hs);
        vector[j + MIMO26_ROCM_ROPE_PAIRS] = mimo26_f32_to_bf16_d(hc + ls);
    }
}

/* Append a chunk's keys and values into the history arrays at `offset`. */
__global__ static void mimo26_append_kv_kernel(
        uint16_t *keys, uint16_t *values, const uint16_t *chunk_keys,
        const uint16_t *chunk_values, uint32_t kv_heads, uint64_t offset,
        uint32_t count)
{
    const uint32_t token = blockIdx.y;
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= count) {
        return;
    }
    const uint32_t k_width = kv_heads * MIMO26_ROCM_QK_DIM;
    const uint32_t v_width = kv_heads * MIMO26_ROCM_V_DIM;
    if (index < k_width) {
        keys[(offset + token) * k_width + index] =
            chunk_keys[(uint64_t)token * k_width + index];
    }
    if (index < v_width) {
        values[(offset + token) * v_width + index] =
            chunk_values[(uint64_t)token * v_width + index];
    }
}

__global__ static void mimo26_ordered_gemv_f32_batch_kernel(
        float *output, const uint16_t *weights, const uint16_t *input,
        uint32_t columns, uint32_t rows, uint32_t round_bf16)
{
    extern __shared__ float staged[];
    const uint32_t row = blockIdx.x;
    const uint32_t token = blockIdx.y;
    const uint16_t *w = weights + (uint64_t)row * columns;
    const uint16_t *x = input + (uint64_t)token * columns;

    for (uint32_t c = threadIdx.x; c < columns; c += blockDim.x) {
        staged[c] = mimo26_bf16_to_f32_d(w[c]) * mimo26_bf16_to_f32_d(x[c]);
    }
    __syncthreads();
    if (threadIdx.x == 0u) {
        float sum = 0.0f;
        for (uint32_t c = 0; c < columns; c++) {
            sum += staged[c];
        }
        /*
         * round_bf16 exists to settle moe_router_dtype by experiment rather
         * than by argument. The config declares bfloat16; the shipped
         * reference ignores the field and hard-codes an F32 linear; vLLM
         * honours it. Both cannot be what training did, and the shipped
         * file is the one that also got the QKV layout wrong, so its
         * behaviour is weak evidence.
         *
         * A BF16 router differs from an F32 one in exactly one observable
         * way: the logit is rounded to BF16 before sigmoid and the
         * correction bias, so a near-tie in the top-k selection can resolve
         * differently. Accumulation is F32 on real hardware either way.
         */
        output[(uint64_t)token * rows + row] =
            round_bf16 ? mimo26_bf16_to_f32_d(mimo26_f32_to_bf16_d(sum))
                       : sum;
    }
}

static inline uint32_t blocks_for(uint64_t count)
{
    return (uint32_t)((count + MIMO26_ROCM_THREADS - 1u) /
                      MIMO26_ROCM_THREADS);
}

/*
 * Off by default: F32 is what the shipped reference executes and what every
 * gate in this lane was qualified against. Flipping it is an experiment, not
 * a tuning knob, so it is read once and reported rather than silently
 * changing behaviour between runs.
 */
static int g_router_bf16 = -1;

extern "C" uint32_t mimo26_rocm_router_bf16_enabled(void)
{
    if (g_router_bf16 < 0) {
        const char *setting = getenv("MIMO26_ROUTER_BF16");
        g_router_bf16 = (setting != NULL && setting[0] == '1') ? 1 : 0;
        if (g_router_bf16) {
            fprintf(stderr, "mimo26: router logits rounded to BF16 "
                            "(moe_router_dtype experiment)\n");
        }
    }
    return (uint32_t)g_router_bf16;
}

extern "C" {

bool mimo26_rocm_rmsnorm_bf16(void *output, const void *input,
                              const void *weight, uint32_t vectors,
                              uint32_t hidden_size, float epsilon,
                              void *stream)
{
    if (output == NULL || input == NULL || weight == NULL || vectors == 0u ||
        hidden_size == 0u) {
        return false;
    }
    hipLaunchKernelGGL(mimo26_rmsnorm_kernel, dim3(vectors),
                       dim3(MIMO26_ROCM_THREADS), 0, (hipStream_t)stream,
                       (uint16_t *)output, (const uint16_t *)input,
                       (const uint16_t *)weight, hidden_size, epsilon);
    return hipGetLastError() == hipSuccess;
}

bool mimo26_rocm_silu_product_bf16(void *output, const void *gate,
                                   const void *up, uint64_t count,
                                   void *stream)
{
    if (output == NULL || gate == NULL || up == NULL || count == 0u) {
        return false;
    }
    hipLaunchKernelGGL(mimo26_silu_product_kernel, dim3(blocks_for(count)),
                       dim3(MIMO26_ROCM_THREADS), 0, (hipStream_t)stream,
                       (uint16_t *)output, (const uint16_t *)gate,
                       (const uint16_t *)up, count);
    return hipGetLastError() == hipSuccess;
}

bool mimo26_rocm_residual_add_bf16(void *output, const void *residual,
                                   const void *delta, uint64_t count,
                                   void *stream)
{
    if (output == NULL || residual == NULL || delta == NULL || count == 0u) {
        return false;
    }
    hipLaunchKernelGGL(mimo26_residual_add_kernel, dim3(blocks_for(count)),
                       dim3(MIMO26_ROCM_THREADS), 0, (hipStream_t)stream,
                       (uint16_t *)output, (const uint16_t *)residual,
                       (const uint16_t *)delta, count);
    return hipGetLastError() == hipSuccess;
}

bool mimo26_rocm_router_topk(uint32_t *expert_ids, float *expert_weights,
                             const float *logits, const float *correction_bias,
                             uint32_t vectors, uint32_t expert_count,
                             uint32_t top_k, float scale, void *stream)
{
    if (expert_ids == NULL || expert_weights == NULL || logits == NULL ||
        correction_bias == NULL || vectors == 0u || top_k == 0u ||
        top_k > 64u || expert_count == 0u ||
        expert_count > MIMO26_ROCM_MAX_EXPERTS) {
        return false;
    }
    hipLaunchKernelGGL(mimo26_router_topk_kernel, dim3(vectors),
                       dim3(MIMO26_ROCM_THREADS), 0, (hipStream_t)stream,
                       expert_ids, expert_weights, logits, correction_bias,
                       expert_count, top_k, scale);
    return hipGetLastError() == hipSuccess;
}

bool mimo26_rocm_expert_accumulate_f32(float *accumulator,
                                       const void *expert_output, float weight,
                                       uint64_t count, void *stream)
{
    if (accumulator == NULL || expert_output == NULL || count == 0u) {
        return false;
    }
    hipLaunchKernelGGL(mimo26_expert_accumulate_kernel,
                       dim3(blocks_for(count)), dim3(MIMO26_ROCM_THREADS), 0,
                       (hipStream_t)stream, accumulator,
                       (const uint16_t *)expert_output, weight, count);
    return hipGetLastError() == hipSuccess;
}

bool mimo26_rocm_expert_finalize_bf16(void *output, const float *accumulator,
                                      uint64_t count, void *stream)
{
    if (output == NULL || accumulator == NULL || count == 0u) {
        return false;
    }
    hipLaunchKernelGGL(mimo26_expert_finalize_kernel, dim3(blocks_for(count)),
                       dim3(MIMO26_ROCM_THREADS), 0, (hipStream_t)stream,
                       (uint16_t *)output, accumulator, count);
    return hipGetLastError() == hipSuccess;
}

bool mimo26_rocm_zero_f32(float *accumulator, uint64_t count, void *stream)
{
    if (accumulator == NULL || count == 0u) {
        return false;
    }
    hipLaunchKernelGGL(mimo26_zero_kernel, dim3(blocks_for(count)),
                       dim3(MIMO26_ROCM_THREADS), 0, (hipStream_t)stream,
                       accumulator, count);
    return hipGetLastError() == hipSuccess;
}

uint64_t mimo26_rocm_attention_scratch_floats(uint64_t history)
{
    return (uint64_t)MIMO26_ROCM_QUERY_HEADS * (history + 2u);
}

static bool attention_launch(void *out, const void *query, const void *keys,
                             const void *values, const void *current_keys,
                             const void *current_values,
                             const void *sink_bias, float *scratch,
                             uint32_t kv_heads, uint32_t kv_groups,
                             uint32_t window, uint64_t history,
                             uint64_t first_position, uint64_t query_position,
                             float scale, bool scores_only, void *stream,
                             uint32_t batch)
{
    const bool have_current =
        (current_keys != NULL && current_values != NULL);
    /* Same admission checks as the CPU entry point, and for the same reason:
     * an invalid geometry should be refused, not silently attended over. */
    if ((out == NULL && !scores_only) || query == NULL || scratch == NULL ||
        kv_heads == 0u ||
        kv_groups == 0u ||
        kv_groups * kv_heads != MIMO26_ROCM_QUERY_HEADS) {
        return false;
    }
    if ((current_keys == NULL) != (current_values == NULL)) {
        return false;
    }
    if (history > 0u && (keys == NULL || values == NULL)) {
        return false;
    }
    if (history == 0u && !have_current) {
        return false;
    }
    if (history > 0u) {
        if (first_position > query_position) {
            return false;
        }
        /*
         * The span is measured to the LAST query in the batch, not the
         * first. A prefill chunk deliberately carries keys for tokens that
         * come after the earliest query -- the causal predicate excludes
         * them per query -- so checking against the first would reject
         * every batch larger than one.
         */
        const uint64_t last_position = query_position + (batch - 1u);
        const uint64_t span = last_position - first_position + 1u;
        if (span < history) {
            return false;
        }
        if (have_current && span - 1u < history) {
            return false;
        }
    }
    hipLaunchKernelGGL(mimo26_attention_decode_kernel,
                       dim3(MIMO26_ROCM_QUERY_HEADS, batch),
                       dim3(MIMO26_ROCM_THREADS), 0, (hipStream_t)stream,
                       (uint16_t *)out, (const uint16_t *)query,
                       (const uint16_t *)keys, (const uint16_t *)values,
                       (const uint16_t *)current_keys,
                       (const uint16_t *)current_values,
                       (const uint16_t *)sink_bias, scratch, kv_heads,
                       kv_groups, window, history, first_position,
                       query_position, scale, have_current ? 1u : 0u,
                       sink_bias != NULL ? 1u : 0u, scores_only ? 1u : 0u);
    return hipGetLastError() == hipSuccess;
}


bool mimo26_rocm_attention_decode(void *out, const void *query,
                                  const void *keys, const void *values,
                                  const void *current_keys,
                                  const void *current_values,
                                  const void *sink_bias, float *scratch,
                                  uint32_t kv_heads, uint32_t kv_groups,
                                  uint32_t window, uint64_t history,
                                  uint64_t first_position,
                                  uint64_t query_position, float scale,
                                  void *stream)
{
    return attention_launch(out, query, keys, values, current_keys,
                            current_values, sink_bias, scratch, kv_heads,
                            kv_groups, window, history, first_position,
                            query_position, scale, false, stream, 1u);
}

bool mimo26_rocm_attention_prefill(void *out, const void *query,
                                   const void *keys, const void *values,
                                   const void *sink_bias, float *scratch,
                                   uint64_t scratch_floats,
                                   uint32_t kv_heads, uint32_t kv_groups,
                                   uint32_t window, uint64_t history,
                                   uint64_t first_position,
                                   uint64_t first_query_position,
                                   uint32_t query_count, float scale,
                                   void *stream)
{
    if (query_count == 0u || scratch == NULL) {
        return false;
    }
    /*
     * The chunk's own keys and values are expected to be appended to the
     * history the caller passes, so `history` counts both. Causal masking
     * inside the chunk then falls out of the absolute-position predicate --
     * token b sees history plus chunk entries 0..b and nothing after.
     */
    if (history < query_count) {
        return false;
    }
    /* Keys for chunk token 0 sit at this slot, so a sub-batch ending at chunk
     * offset e must be given prior + e slots. */
    const uint64_t prior = history - query_count;

    /*
     * Widest sub-batch the buffer allows at the deepest history, which is what
     * the last sub-batch asks for. Sizing off that worst case keeps every
     * launch the same width and the bound honest.
     */
    const uint64_t row_floats = mimo26_rocm_attention_scratch_floats(history);
    if (row_floats == 0u || scratch_floats < row_floats) {
        return false;
    }
    uint64_t width = scratch_floats / row_floats;
    if (width > query_count) {
        width = query_count;
    }

    for (uint64_t offset = 0u; offset < query_count; offset += width) {
        uint64_t n = query_count - offset;
        if (n > width) {
            n = width;
        }
        /* Exactly the slots this sub-batch's last query can see. Everything
         * past it was masked out of the full-width call regardless. */
        const uint64_t sub_history = prior + offset + n;
        if (!attention_launch(
                (uint16_t *)out +
                    offset * MIMO26_ROCM_QUERY_HEADS * MIMO26_ROCM_V_DIM,
                (const uint16_t *)query +
                    offset * MIMO26_ROCM_QUERY_HEADS * MIMO26_ROCM_QK_DIM,
                keys, values, NULL, NULL, sink_bias, scratch, kv_heads,
                kv_groups, window, sub_history, first_position,
                first_query_position + offset, scale, false, stream,
                (uint32_t)n)) {
            return false;
        }
    }
    return true;
}

bool mimo26_rocm_attention_scores(const void *query, const void *keys,
                                  const void *current_keys,
                                  const void *sink_bias, float *scratch,
                                  uint32_t kv_heads, uint32_t kv_groups,
                                  uint32_t window, uint64_t history,
                                  uint64_t first_position,
                                  uint64_t query_position, float scale,
                                  void *stream)
{
    /* values are unused on this path, but the shared admission checks want a
     * non-NULL pair whenever there is history to attend to. */
    return attention_launch(NULL, query, keys, keys, current_keys,
                            current_keys, sink_bias, scratch, kv_heads,
                            kv_groups, window, history, first_position,
                            query_position, scale, true, stream, 1u);
}

bool mimo26_rocm_split_qkv(void *q, void *k, void *v, const void *fused,
                           uint32_t kv_heads, void *stream)
{
    if (q == NULL || k == NULL || v == NULL || fused == NULL ||
        kv_heads == 0u) {
        return false;
    }
    const uint32_t total = MIMO26_ROCM_QUERY_HEADS * MIMO26_ROCM_QK_DIM +
                           kv_heads * MIMO26_ROCM_QK_DIM +
                           kv_heads * MIMO26_ROCM_V_DIM;
    hipLaunchKernelGGL(mimo26_split_qkv_kernel, dim3(blocks_for(total)),
                       dim3(MIMO26_ROCM_THREADS), 0, (hipStream_t)stream,
                       (uint16_t *)q, (uint16_t *)k, (uint16_t *)v,
                       (const uint16_t *)fused, kv_heads);
    return hipGetLastError() == hipSuccess;
}

bool mimo26_rocm_split_qkv_batch(void *q, void *k, void *v,
                                 const void *fused, uint32_t kv_heads,
                                 uint32_t qkv_width, uint32_t count,
                                 void *stream)
{
    if (q == NULL || k == NULL || v == NULL || fused == NULL ||
        kv_heads == 0u || count == 0u) {
        return false;
    }
    const uint32_t total = MIMO26_ROCM_QUERY_HEADS * MIMO26_ROCM_QK_DIM +
                           kv_heads * MIMO26_ROCM_QK_DIM +
                           kv_heads * MIMO26_ROCM_V_DIM;
    hipLaunchKernelGGL(mimo26_split_qkv_batch_kernel,
                       dim3(blocks_for(total), count),
                       dim3(MIMO26_ROCM_THREADS), 0, (hipStream_t)stream,
                       (uint16_t *)q, (uint16_t *)k, (uint16_t *)v,
                       (const uint16_t *)fused, kv_heads, qkv_width);
    return hipGetLastError() == hipSuccess;
}

bool mimo26_rocm_rope_apply_batch(void *heads, const void *cos_tables,
                                  const void *sin_tables,
                                  uint32_t head_count, uint32_t count,
                                  void *stream)
{
    if (heads == NULL || cos_tables == NULL || sin_tables == NULL ||
        head_count == 0u || count == 0u) {
        return false;
    }
    hipLaunchKernelGGL(mimo26_rope_apply_batch_kernel,
                       dim3(head_count, count),
                       dim3(MIMO26_ROCM_ROPE_PAIRS), 0, (hipStream_t)stream,
                       (uint16_t *)heads, (const uint16_t *)cos_tables,
                       (const uint16_t *)sin_tables, head_count);
    return hipGetLastError() == hipSuccess;
}

bool mimo26_rocm_append_kv(void *keys, void *values, const void *chunk_keys,
                           const void *chunk_values, uint32_t kv_heads,
                           uint64_t offset, uint32_t count, void *stream)
{
    if (keys == NULL || values == NULL || chunk_keys == NULL ||
        chunk_values == NULL || kv_heads == 0u || count == 0u) {
        return false;
    }
    const uint32_t width = kv_heads * MIMO26_ROCM_QK_DIM;
    hipLaunchKernelGGL(mimo26_append_kv_kernel,
                       dim3(blocks_for(width), count),
                       dim3(MIMO26_ROCM_THREADS), 0, (hipStream_t)stream,
                       (uint16_t *)keys, (uint16_t *)values,
                       (const uint16_t *)chunk_keys,
                       (const uint16_t *)chunk_values, kv_heads, offset,
                       count);
    return hipGetLastError() == hipSuccess;
}

bool mimo26_rocm_rope_apply(void *heads, const void *cos_table,
                            const void *sin_table, uint32_t head_count,
                            void *stream)
{
    if (heads == NULL || cos_table == NULL || sin_table == NULL ||
        head_count == 0u) {
        return false;
    }
    hipLaunchKernelGGL(mimo26_rope_apply_kernel, dim3(head_count),
                       dim3(MIMO26_ROCM_ROPE_PAIRS), 0, (hipStream_t)stream,
                       (uint16_t *)heads, (const uint16_t *)cos_table,
                       (const uint16_t *)sin_table, head_count);
    return hipGetLastError() == hipSuccess;
}

bool mimo26_rocm_ordered_gemv_f32(float *output, const void *weights,
                                  const void *input, uint32_t rows,
                                  uint32_t columns, void *stream)
{
    if (output == NULL || weights == NULL || input == NULL || rows == 0u ||
        columns == 0u) {
        return false;
    }
    /* One float of shared memory per column; 4096 columns is 16 KiB of the
     * 64 KiB available, so both users fit comfortably. */
    const size_t shared = (size_t)columns * sizeof(float);
    if (shared > 65536u) {
        return false;
    }
    /* Never rounds: this entry point also serves lm_head under
     * MIMO26_GPU_EXACT_HEAD, and the router experiment must not reach it. */
    hipLaunchKernelGGL(mimo26_ordered_gemv_f32_kernel, dim3(rows),
                       dim3(MIMO26_ROCM_THREADS), shared, (hipStream_t)stream,
                       output, (const uint16_t *)weights,
                       (const uint16_t *)input, columns, 0u);
    return hipGetLastError() == hipSuccess;
}

bool mimo26_rocm_ordered_gemv_f32_batch(float *output, const void *weights,
                                        const void *input, uint32_t rows,
                                        uint32_t columns, uint32_t count,
                                        void *stream)
{
    if (output == NULL || weights == NULL || input == NULL || rows == 0u ||
        columns == 0u || count == 0u) {
        return false;
    }
    const size_t shared = (size_t)columns * sizeof(float);
    if (shared > 65536u) {
        return false;
    }
    /* blockIdx.y selects the token; each keeps its own row order. */
    hipLaunchKernelGGL(mimo26_ordered_gemv_f32_batch_kernel,
                       dim3(rows, count), dim3(MIMO26_ROCM_THREADS), shared,
                       (hipStream_t)stream, output,
                       (const uint16_t *)weights, (const uint16_t *)input,
                       columns, rows, mimo26_rocm_router_bf16_enabled());
    return hipGetLastError() == hipSuccess;
}

bool mimo26_rocm_router_logits_f32(float *logits, const void *weight,
                                   const void *hidden, uint32_t experts,
                                   uint32_t hidden_size, void *stream)
{
    if (logits == NULL || weight == NULL || hidden == NULL ||
        experts == 0u || hidden_size == 0u) {
        return false;
    }
    const size_t shared = (size_t)hidden_size * sizeof(float);
    if (shared > 65536u) {
        return false;
    }
    /* Its own launch rather than a call through the generic entry point, so
     * the moe_router_dtype experiment applies here and nowhere else. */
    hipLaunchKernelGGL(mimo26_ordered_gemv_f32_kernel, dim3(experts),
                       dim3(MIMO26_ROCM_THREADS), shared, (hipStream_t)stream,
                       logits, (const uint16_t *)weight,
                       (const uint16_t *)hidden, hidden_size,
                       mimo26_rocm_router_bf16_enabled());
    return hipGetLastError() == hipSuccess;
}

}  /* extern "C" */
