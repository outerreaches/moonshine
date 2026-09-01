#include "glm53_vector_ops.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <stdint.h>

namespace {
constexpr uint32_t kThreads = 256u;

static bool span(const void *p, size_t count, size_t element_size,
                 uintptr_t *begin, uintptr_t *end) {
    if (p == nullptr || count == 0u || count > SIZE_MAX / element_size)
        return false;
    const size_t bytes = count * element_size;
    const uintptr_t first = reinterpret_cast<uintptr_t>(p);
    if (first > UINTPTR_MAX - bytes) return false;
    *begin = first;
    *end = first + bytes;
    return true;
}

static bool buffers(const void *const *p, const size_t *count,
                    const size_t *element_size, size_t n) {
    uintptr_t begin[3], end[3];
    if (n == 0u || n > 3u) return false;
    for (size_t i = 0; i < n; ++i)
        if (!span(p[i], count[i], element_size[i], &begin[i], &end[i]))
            return false;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1u; j < n; ++j)
            if (begin[i] < end[j] && begin[j] < end[i]) return false;
    return true;
}

static bool grid_for(size_t count, uint32_t *grid) {
    if (count == 0u || count > SIZE_MAX - (kThreads - 1u)) return false;
    const size_t blocks = (count + kThreads - 1u) / kThreads;
    if (blocks == 0u || blocks > UINT32_MAX) return false;
    *grid = static_cast<uint32_t>(blocks);
    return true;
}

__device__ __forceinline__ float stable_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
}

__global__ void gather_kernel(float *out, const hip_bfloat16 *table,
                              size_t base, size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (i < count) out[i] = static_cast<float>(table[base + i]);
}

template <bool Bf16Weight>
__global__ void rmsnorm_kernel(float *out, const float *in,
                               const void *weight, size_t count, float eps) {
    __shared__ float partial[kThreads];
    float sum = 0.0f;
    for (size_t i = threadIdx.x; i < count; i += kThreads)
        sum += in[i] * in[i];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t width = kThreads / 2u; width; width >>= 1u) {
        if (threadIdx.x < width) partial[threadIdx.x] += partial[threadIdx.x + width];
        __syncthreads();
    }
    const float scale = 1.0f / sqrtf(partial[0] / static_cast<float>(count) + eps);
    for (size_t i = threadIdx.x; i < count; i += kThreads) {
        const float w = Bf16Weight
            ? static_cast<float>(static_cast<const hip_bfloat16 *>(weight)[i])
            : static_cast<const float *>(weight)[i];
        out[i] = in[i] * scale * w;
    }
}

enum class BinaryOp { Add, Multiply, SwiGLU };
template <BinaryOp Op>
__global__ void binary_kernel(float *out, const float *a, const float *b,
                              size_t count, float limit) {
    const size_t i = static_cast<size_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (i >= count) return;
    if (Op == BinaryOp::Add) out[i] = a[i] + b[i];
    else if (Op == BinaryOp::Multiply) out[i] = a[i] * b[i];
    else {
        float gate = a[i], up = b[i];
        if (gate > limit) gate = limit;
        if (up > limit) up = limit;
        else if (up < -limit) up = -limit;
        out[i] = (gate * stable_sigmoid(gate)) * up;
    }
}

enum class UnaryOp { Sigmoid, Silu };
template <UnaryOp Op>
__global__ void unary_kernel(float *out, const float *in, size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (i >= count) return;
    const float s = stable_sigmoid(in[i]);
    out[i] = Op == UnaryOp::Sigmoid ? s : in[i] * s;
}

__global__ void f32_bf16_kernel(hip_bfloat16 *out, const float *in, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (i < n) out[i] = hip_bfloat16(in[i]);
}
__global__ void bf16_f32_kernel(float *out, const hip_bfloat16 *in, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (i < n) out[i] = static_cast<float>(in[i]);
}

__global__ void dot_kernel(float *out, const float *a, const float *b, size_t n) {
    __shared__ float partial[kThreads];
    float sum = 0.0f;
    for (size_t i = threadIdx.x; i < n; i += kThreads) sum += a[i] * b[i];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t width = kThreads / 2u; width; width >>= 1u) {
        if (threadIdx.x < width) partial[threadIdx.x] += partial[threadIdx.x + width];
        __syncthreads();
    }
    if (threadIdx.x == 0u) out[0] = partial[0];
}

template <bool Bf16Weight>
__global__ void matvec_kernel(float *out, const void *matrix,
                              const float *in, uint32_t rows, uint32_t cols) {
    const uint32_t row = blockIdx.x * kThreads + threadIdx.x;
    if (row >= rows) return;
    const size_t base = static_cast<size_t>(row) * cols;
    float sum = 0.0f;
    for (uint32_t c = 0; c < cols; ++c) {
        const float weight = Bf16Weight
            ? static_cast<float>(static_cast<const hip_bfloat16 *>(matrix)[base + c])
            : static_cast<const float *>(matrix)[base + c];
        sum += weight * in[c];
    }
    out[row] = sum;
}

static bool launch_ok() { return hipGetLastError() == hipSuccess; }
static void clear_status() { (void)hipGetLastError(); }
} // namespace

extern "C" bool glm53_vector_gather_bf16_f32(
    void *output, size_t output_count, const void *table, size_t table_count,
    uint32_t rows, uint32_t columns, uint32_t row_index, void *stream) {
    if (rows == 0u || columns == 0u || row_index >= rows) return false;
    const size_t r = rows, c = columns;
    if (r > SIZE_MAX / c) return false;
    const size_t needed = r * c;
    if (output_count < c || table_count < needed) return false;
    const void *p[] = {output, table};
    const size_t counts[] = {output_count, table_count};
    const size_t sizes[] = {sizeof(float), sizeof(hip_bfloat16)};
    uint32_t grid;
    if (!buffers(p, counts, sizes, 2u) || !grid_for(c, &grid)) return false;
    clear_status();
    hipLaunchKernelGGL(gather_kernel, dim3(grid), dim3(kThreads), 0,
                       static_cast<hipStream_t>(stream), static_cast<float *>(output),
                       static_cast<const hip_bfloat16 *>(table),
                       static_cast<size_t>(row_index) * c, c);
    return launch_ok();
}

template <bool Bf16Weight>
static bool rmsnorm(void *output, size_t output_count, const void *input,
                    size_t input_count, const void *weight, size_t weight_count,
                    size_t count, float eps, void *stream) {
    if (count == 0u || output_count < count || input_count < count ||
        weight_count < count || !std::isfinite(eps) || eps < 0.0f) return false;
    const void *p[] = {output, input, weight};
    const size_t counts[] = {output_count, input_count, weight_count};
    const size_t sizes[] = {sizeof(float), sizeof(float),
                            Bf16Weight ? sizeof(hip_bfloat16) : sizeof(float)};
    if (!buffers(p, counts, sizes, 3u)) return false;
    clear_status();
    hipLaunchKernelGGL((rmsnorm_kernel<Bf16Weight>), dim3(1), dim3(kThreads), 0,
                       static_cast<hipStream_t>(stream), static_cast<float *>(output),
                       static_cast<const float *>(input), weight, count, eps);
    return launch_ok();
}

extern "C" bool glm53_vector_rmsnorm_f32(
    void *o,size_t oc,const void *i,size_t ic,const void *w,size_t wc,
    size_t n,float eps,void *s) { return rmsnorm<false>(o,oc,i,ic,w,wc,n,eps,s); }
extern "C" bool glm53_vector_rmsnorm_bf16_weight_f32(
    void *o,size_t oc,const void *i,size_t ic,const void *w,size_t wc,
    size_t n,float eps,void *s) { return rmsnorm<true>(o,oc,i,ic,w,wc,n,eps,s); }

template <BinaryOp Op>
static bool binary(void *o,size_t oc,const void *a,size_t ac,const void *b,size_t bc,
                   size_t n,float limit,void *s) {
    if (n == 0u || oc < n || ac < n || bc < n || !std::isfinite(limit) || limit < 0.0f) return false;
    const void *p[] = {o,a,b}; const size_t cs[] = {oc,ac,bc};
    const size_t zs[] = {sizeof(float),sizeof(float),sizeof(float)}; uint32_t grid;
    if (!buffers(p,cs,zs,3u) || !grid_for(n,&grid)) return false;
    clear_status();
    hipLaunchKernelGGL((binary_kernel<Op>),dim3(grid),dim3(kThreads),0,
        static_cast<hipStream_t>(s),static_cast<float *>(o),
        static_cast<const float *>(a),static_cast<const float *>(b),n,limit);
    return launch_ok();
}
extern "C" bool glm53_vector_add_f32(void*o,size_t oc,const void*a,size_t ac,const void*b,size_t bc,size_t n,void*s){return binary<BinaryOp::Add>(o,oc,a,ac,b,bc,n,0.0f,s);}
extern "C" bool glm53_vector_multiply_f32(void*o,size_t oc,const void*a,size_t ac,const void*b,size_t bc,size_t n,void*s){return binary<BinaryOp::Multiply>(o,oc,a,ac,b,bc,n,0.0f,s);}
extern "C" bool glm53_vector_swiglu_f32(void*o,size_t oc,const void*a,size_t ac,const void*b,size_t bc,size_t n,float limit,void*s){return binary<BinaryOp::SwiGLU>(o,oc,a,ac,b,bc,n,limit,s);}

template <UnaryOp Op>
static bool unary(void *o,size_t oc,const void *i,size_t ic,size_t n,void *s) {
    if (n == 0u || oc < n || ic < n) return false;
    const void *p[]={o,i}; const size_t cs[]={oc,ic};
    const size_t zs[]={sizeof(float),sizeof(float)}; uint32_t grid;
    if (!buffers(p,cs,zs,2u)||!grid_for(n,&grid)) return false;
    clear_status();
    hipLaunchKernelGGL((unary_kernel<Op>),dim3(grid),dim3(kThreads),0,
        static_cast<hipStream_t>(s),static_cast<float *>(o),static_cast<const float *>(i),n);
    return launch_ok();
}
extern "C" bool glm53_vector_sigmoid_f32(void*o,size_t oc,const void*i,size_t ic,size_t n,void*s){return unary<UnaryOp::Sigmoid>(o,oc,i,ic,n,s);}
extern "C" bool glm53_vector_silu_f32(void*o,size_t oc,const void*i,size_t ic,size_t n,void*s){return unary<UnaryOp::Silu>(o,oc,i,ic,n,s);}

extern "C" bool glm53_vector_cast_f32_bf16(void *o,size_t oc,const void *i,size_t ic,size_t n,void *s) {
    if(n==0u||oc<n||ic<n)return false; const void*p[]={o,i};const size_t cs[]={oc,ic};
    const size_t zs[]={sizeof(hip_bfloat16),sizeof(float)};uint32_t grid;
    if(!buffers(p,cs,zs,2u)||!grid_for(n,&grid))return false;clear_status();
    hipLaunchKernelGGL(f32_bf16_kernel,dim3(grid),dim3(kThreads),0,static_cast<hipStream_t>(s),static_cast<hip_bfloat16*>(o),static_cast<const float*>(i),n);return launch_ok();
}
extern "C" bool glm53_vector_cast_bf16_f32(void *o,size_t oc,const void *i,size_t ic,size_t n,void *s) {
    if(n==0u||oc<n||ic<n)return false; const void*p[]={o,i};const size_t cs[]={oc,ic};
    const size_t zs[]={sizeof(float),sizeof(hip_bfloat16)};uint32_t grid;
    if(!buffers(p,cs,zs,2u)||!grid_for(n,&grid))return false;clear_status();
    hipLaunchKernelGGL(bf16_f32_kernel,dim3(grid),dim3(kThreads),0,static_cast<hipStream_t>(s),static_cast<float*>(o),static_cast<const hip_bfloat16*>(i),n);return launch_ok();
}
extern "C" bool glm53_vector_dot_f32(void*o,size_t oc,const void*a,size_t ac,const void*b,size_t bc,size_t n,void*s) {
    if(n==0u||oc<1u||ac<n||bc<n)return false;const void*p[]={o,a,b};const size_t cs[]={oc,ac,bc};const size_t zs[]={sizeof(float),sizeof(float),sizeof(float)};
    if(!buffers(p,cs,zs,3u))return false;clear_status();hipLaunchKernelGGL(dot_kernel,dim3(1),dim3(kThreads),0,static_cast<hipStream_t>(s),static_cast<float*>(o),static_cast<const float*>(a),static_cast<const float*>(b),n);return launch_ok();
}
template <bool Bf16Weight>
static bool matvec(void *o, size_t oc, const void *m, size_t mc,
                   const void *i, size_t ic, uint32_t rows, uint32_t cols,
                   void *s) {
    if (rows == 0u || cols == 0u ||
        static_cast<size_t>(rows) > SIZE_MAX / static_cast<size_t>(cols))
        return false;
    const size_t needed = static_cast<size_t>(rows) * cols;
    if (oc < rows || mc < needed || ic < cols) return false;
    const void *p[] = {o, m, i};
    const size_t cs[] = {oc, mc, ic};
    const size_t zs[] = {sizeof(float),
        Bf16Weight ? sizeof(hip_bfloat16) : sizeof(float), sizeof(float)};
    uint32_t grid;
    if (!buffers(p, cs, zs, 3u) || !grid_for(rows, &grid)) return false;
    clear_status();
    hipLaunchKernelGGL((matvec_kernel<Bf16Weight>), dim3(grid),
        dim3(kThreads), 0, static_cast<hipStream_t>(s),
        static_cast<float *>(o), m, static_cast<const float *>(i), rows, cols);
    return launch_ok();
}
extern "C" bool glm53_vector_matvec_f32(void *o, size_t oc, const void *m,
    size_t mc, const void *i, size_t ic, uint32_t rows, uint32_t cols, void *s) {
    return matvec<false>(o, oc, m, mc, i, ic, rows, cols, s);
}
extern "C" bool glm53_vector_matvec_bf16_weight_f32(
    void *o, size_t oc, const void *m, size_t mc, const void *i, size_t ic,
    uint32_t rows, uint32_t cols, void *s) {
    return matvec<true>(o, oc, m, mc, i, ic, rows, cols, s);
}
