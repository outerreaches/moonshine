#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
__global__ void k(float *out, const float *in, int n){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) out[i] = expf(in[i]);
}
int main(){
    const int N = 200000;
    float *h = (float*)malloc(N*sizeof(float));
    float *g = (float*)malloc(N*sizeof(float));
    float *c = (float*)malloc(N*sizeof(float));
    // The range that arises after max-subtraction in a softmax: (-inf, 0].
    for (int i = 0; i < N; i++) h[i] = -40.0f * (float)i / (float)N;
    float *d_in, *d_out;
    hipMalloc(&d_in, N*sizeof(float)); hipMalloc(&d_out, N*sizeof(float));
    hipMemcpy(d_in, h, N*sizeof(float), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(k, dim3((N+255)/256), dim3(256), 0, 0, d_out, d_in, N);
    hipDeviceSynchronize();
    hipMemcpy(g, d_out, N*sizeof(float), hipMemcpyDeviceToHost);
    int differ = 0, worst = 0;
    for (int i = 0; i < N; i++){
        c[i] = expf(h[i]);
        unsigned a, b; memcpy(&a,&c[i],4); memcpy(&b,&g[i],4);
        if (a != b){ differ++; int d = (int)a-(int)b; if (d<0) d=-d; if (d>worst) worst=d; }
    }
    printf("device expf vs host expf over %d samples in [-40,0]: %d differ (%.3f%%), worst %d ulp\n",
           N, differ, 100.0*differ/N, worst);
    return 0;
}
