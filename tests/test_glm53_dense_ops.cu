#include "glm53_dense_ops.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); return false; } } while (0)
#define HIP(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "HIP %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); return false; } } while (0)

static uint16_t bf16_bits(float value) {
    uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu))
        return static_cast<uint16_t>((bits >> 16u) | 0x0040u);
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return static_cast<uint16_t>(bits >> 16u);
}
static float from_bf16(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16u;
    float result; std::memcpy(&result, &bits, sizeof(result)); return result;
}
static float decode(uint8_t code) {
    const uint32_t sign = static_cast<uint32_t>(code & 0x80u) << 24u;
    const uint32_t exponent = (code >> 3u) & 15u;
    const uint32_t fraction = code & 7u;
    uint32_t bits;
    if (exponent == 0u) {
        if (fraction == 0u) bits = sign;
        else {
            const uint32_t top = fraction >= 4u ? 2u : (fraction >= 2u ? 1u : 0u);
            bits = sign | ((118u + top) << 23u) |
                   ((fraction - (1u << top)) << (23u - top));
        }
    } else bits = sign | ((exponent + 120u) << 23u) | (fraction << 20u);
    float value; std::memcpy(&value, &bits, sizeof(value)); return value;
}
static uint8_t encode(float value) {
    const float magnitude = std::fabs(value);
    uint8_t best = 0u; float distance = magnitude;
    for (uint32_t code = 1u; code <= 0x7eu; ++code) {
        const float candidate = decode(static_cast<uint8_t>(code));
        const float d = std::fabs(magnitude - candidate);
        if (d < distance || (d == distance && !(code & 1u) && (best & 1u))) {
            best = static_cast<uint8_t>(code); distance = d;
        }
    }
    uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint8_t>(best | ((bits >> 24u) & 0x80u));
}

struct Matrix { uint32_t rows, cols; std::vector<uint8_t> w; std::vector<float> s; };
static Matrix matrix(uint32_t rows, uint32_t cols, uint32_t seed) {
    Matrix m{rows, cols, std::vector<uint8_t>((size_t)rows * cols),
             std::vector<float>((size_t)((rows + 127u) / 128u) * (cols / 128u))};
    for (size_t i = 0; i < m.s.size(); ++i) m.s[i] = 0.0015f + 0.00017f * float((i + seed) % 7u);
    for (uint32_t r = 0; r < rows; ++r) for (uint32_t c = 0; c < cols; ++c) {
        int v = int((r * 19u + c * 13u + seed * 11u) % 31u) - 15;
        m.w[(size_t)r * cols + c] = encode(float(v) * 9.0f);
    }
    return m;
}
static std::vector<float> project(const Matrix &m, const std::vector<float> &x) {
    std::vector<uint8_t> q(x.size()); std::vector<float> xs(x.size() / 128u);
    for (size_t group = 0; group < xs.size(); ++group) {
        float maximum = 0.0f;
        for (size_t j = 0; j < 128u; ++j) maximum = std::max(maximum, std::fabs(x[group * 128u + j]));
        xs[group] = std::max(maximum, 1.0e-10f) * (1.0f / 448.0f);
        for (size_t j = 0; j < 128u; ++j) {
            float z = x[group * 128u + j] / xs[group];
            z = std::max(-448.0f, std::min(448.0f, z));
            q[group * 128u + j] = encode(z);
        }
    }
    std::vector<float> y(m.rows);
    const size_t scale_cols = m.cols / 128u;
    for (uint32_t r = 0; r < m.rows; ++r) {
        float sum = 0.0f;
        for (uint32_t c = 0; c < m.cols; ++c)
            sum += decode(m.w[(size_t)r * m.cols + c]) *
                   m.s[(size_t)(r / 128u) * scale_cols + c / 128u] *
                   decode(q[c]) * xs[c / 128u];
        y[r] = sum;
    }
    return y;
}
static std::vector<uint16_t> oracle(const Matrix &gate, const Matrix &up,
                                    const Matrix &down,
                                    const std::vector<uint16_t> &input, float limit) {
    std::vector<float> x(input.size());
    for (size_t i = 0; i < x.size(); ++i) x[i] = from_bf16(input[i]);
    std::vector<float> g = project(gate, x), u = project(up, x), a(g.size());
    for (size_t i = 0; i < a.size(); ++i) {
        const float capped_gate = std::min(g[i], limit);
        const float capped_up = std::max(-limit, std::min(u[i], limit));
        const float sigmoid = capped_gate >= 0.0f ? 1.0f / (1.0f + std::exp(-capped_gate)) :
            std::exp(capped_gate) / (1.0f + std::exp(capped_gate));
        a[i] = capped_gate * sigmoid * capped_up;
    }
    std::vector<float> y = project(down, a); std::vector<uint16_t> result(y.size());
    for (size_t i = 0; i < y.size(); ++i) result[i] = bf16_bits(y[i]);
    return result;
}

template<class T> static bool device_alloc(T **pointer, size_t count) {
    return hipMalloc(reinterpret_cast<void **>(pointer), sizeof(T) * count) == hipSuccess;
}

static bool rejections() {
    glm53_dense_fp8_matrix m{(void *)0x100000u, 32768u, (void *)0x200000u, 2u};
    glm53_dense_scratch s{(void *)0x300000u,128,(void *)0x400000u,256,
        (void *)0x500000u,256,(void *)0x600000u,256,(void *)0x700000u,128,
        (void *)0x800000u,256,(void *)0x900000u,2};
    CHECK(!glm53_dense_mlp_fp8_bf16((void*)0xa00000u,128,(void*)0xb00000u,128,&m,&m,&m,127,256,4.0f,&s,nullptr));
    CHECK(!glm53_dense_mlp_fp8_bf16((void*)0xa00000u,128,(void*)0xb00000u,128,&m,&m,&m,128,256,0.0f,&s,nullptr));
    glm53_dense_fp8_matrix gate=m, up=m, down=m;
    gate.weights_count=32768u; up.weights=(void*)0x1100000u; up.inverse_scales=(void*)0x1200000u;
    down.weights=(void*)0x1300000u; down.weights_count=32768u; down.inverse_scales=(void*)0x1400000u;
    s.q8_count=255u;
    CHECK(!glm53_dense_mlp_fp8_bf16((void*)0xa00000u,128,(void*)0xb00000u,128,&gate,&up,&down,128,256,4.0f,&s,nullptr));
    s.q8_count=256u; s.activation_f32=s.up_f32;
    CHECK(!glm53_dense_mlp_fp8_bf16((void*)0xa00000u,128,(void*)0xb00000u,128,&gate,&up,&down,128,256,4.0f,&s,nullptr));
    s.activation_f32=(void*)0x600000u; gate.weights=(void*)(UINTPTR_MAX-7u);
    CHECK(!glm53_dense_mlp_fp8_bf16((void*)0xa00000u,128,(void*)0xb00000u,128,&gate,&up,&down,128,256,4.0f,&s,nullptr));
    return true;
}

static bool integration() {
    constexpr uint32_t h=128u, n=256u; constexpr float limit=0.35f;
    Matrix hg=matrix(n,h,1), hu=matrix(n,h,2), hd=matrix(h,n,3);
    std::vector<uint16_t> x1(h),x2(h);
    for (uint32_t i=0;i<h;++i) { x1[i]=bf16_bits((float(int(i%23)-11))/32.0f); x2[i]=bf16_bits((float(int((i*7)%29)-14))/40.0f); }
    const auto want1=oracle(hg,hu,hd,x1,limit), want2=oracle(hg,hu,hd,x2,limit);
    uint8_t *wg=nullptr,*wu=nullptr,*wd=nullptr,*q8=nullptr; float *sg=nullptr,*su=nullptr,*sd=nullptr;
    float *xf=nullptr,*gf=nullptr,*uf=nullptr,*af=nullptr,*df=nullptr,*dyn=nullptr;
    hip_bfloat16 *in=nullptr,*out=nullptr; hipStream_t stream=nullptr;
    CHECK(device_alloc(&wg,hg.w.size())&&device_alloc(&wu,hu.w.size())&&device_alloc(&wd,hd.w.size()));
    CHECK(device_alloc(&sg,hg.s.size())&&device_alloc(&su,hu.s.size())&&device_alloc(&sd,hd.s.size()));
    CHECK(device_alloc(&xf,h)&&device_alloc(&gf,n)&&device_alloc(&uf,n)&&device_alloc(&af,n)&&device_alloc(&df,h));
    CHECK(device_alloc(&q8,n)&&device_alloc(&dyn,n/128u)&&device_alloc(&in,h)&&device_alloc(&out,h));
    HIP(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    HIP(hipMemcpyAsync(wg,hg.w.data(),hg.w.size(),hipMemcpyHostToDevice,stream)); HIP(hipMemcpyAsync(wu,hu.w.data(),hu.w.size(),hipMemcpyHostToDevice,stream)); HIP(hipMemcpyAsync(wd,hd.w.data(),hd.w.size(),hipMemcpyHostToDevice,stream));
    HIP(hipMemcpyAsync(sg,hg.s.data(),hg.s.size()*4,hipMemcpyHostToDevice,stream)); HIP(hipMemcpyAsync(su,hu.s.data(),hu.s.size()*4,hipMemcpyHostToDevice,stream)); HIP(hipMemcpyAsync(sd,hd.s.data(),hd.s.size()*4,hipMemcpyHostToDevice,stream));
    glm53_dense_fp8_matrix gate{wg,hg.w.size(),sg,hg.s.size()},up{wu,hu.w.size(),su,hu.s.size()},down{wd,hd.w.size(),sd,hd.s.size()};
    glm53_dense_scratch scratch{xf,h,gf,n,uf,n,af,n,df,h,q8,n,dyn,n/128u};
    std::vector<uint16_t> got1(h),got2(h);
    /* Copies and two complete MLPs deliberately share one nondefault stream.
     * No host synchronization separates them; the results prove queue order. */
    HIP(hipMemcpyAsync(in,x1.data(),h*2,hipMemcpyHostToDevice,stream)); CHECK(glm53_dense_mlp_fp8_bf16(out,h,in,h,&gate,&up,&down,h,n,limit,&scratch,stream)); HIP(hipMemcpyAsync(got1.data(),out,h*2,hipMemcpyDeviceToHost,stream));
    HIP(hipMemcpyAsync(in,x2.data(),h*2,hipMemcpyHostToDevice,stream)); CHECK(glm53_dense_mlp_fp8_bf16(out,h,in,h,&gate,&up,&down,h,n,limit,&scratch,stream)); HIP(hipMemcpyAsync(got2.data(),out,h*2,hipMemcpyDeviceToHost,stream));
    HIP(hipStreamSynchronize(stream));
    for (size_t i=0;i<h;++i) { float a=from_bf16(got1[i]),b=from_bf16(want1[i]); CHECK(std::fabs(a-b)<=0.012f*(1.0f+std::fabs(b))); a=from_bf16(got2[i]);b=from_bf16(want2[i]);CHECK(std::fabs(a-b)<=0.012f*(1.0f+std::fabs(b))); }
    HIP(hipStreamDestroy(stream));
    HIP(hipFree(out)); HIP(hipFree(in)); HIP(hipFree(dyn)); HIP(hipFree(q8));
    HIP(hipFree(df)); HIP(hipFree(af)); HIP(hipFree(uf)); HIP(hipFree(gf));
    HIP(hipFree(xf)); HIP(hipFree(sd)); HIP(hipFree(su)); HIP(hipFree(sg));
    HIP(hipFree(wd)); HIP(hipFree(wu)); HIP(hipFree(wg));
    return true;
}
int main() {
    int device=0; hipDeviceProp_t props{};
    if (hipGetDevice(&device)!=hipSuccess || hipGetDeviceProperties(&props,device)!=hipSuccess) { std::fprintf(stderr,"no HIP device\n"); return 1; }
    if (std::strstr(props.gcnArchName,"gfx1151")==nullptr) { std::fprintf(stderr,"expected gfx1151, got %s\n",props.gcnArchName); return 1; }
    if (!rejections() || !integration()) return 1;
    std::printf("glm53 dense ops: PASS on %s (128/256, nondefault queue)\n",props.gcnArchName); return 0;
}
