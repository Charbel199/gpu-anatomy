#include "../common/check.cuh"
#include "../common/timer.cuh"
#include "../common/bandwidth.cuh"
#include "../common/device_info.cuh"
#include <algorithm>
#include <numeric>
#include <random>


#define N (1 << 24)  // 16M (2^24) elements
#define BLOCK_SIZE 256
#define ITERS 4096
#define ACCS 8

__global__ void fp32_throughput(const float* __restrict__ data, float* __restrict__ out, int n){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= n) return;

    float seed = data[idx];
    float acc0 = seed;
    float acc1 = seed +1.0f;
    float acc2 = seed +2.0f;
    float acc3 = seed +3.0f;
    float acc4 = seed +4.0f;
    float acc5 = seed +5.0f;
    float acc6 = seed +6.0f;
    float acc7 = seed +7.0f;
    
    const float c = 1.001f;
    const float d = 0.5f;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        acc0 = acc0 * c + d;
        acc1 = acc1 * c + d;
        acc2 = acc2 * c + d;
        acc3 = acc3 * c + d;
        acc4 = acc4 * c + d;
        acc5 = acc5 * c + d;
        acc6 = acc6 * c + d;
        acc7 = acc7 * c + d;
    }
    out[idx] = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
}


int main() {
    print_device_info();

    size_t N_float_bytes = N * sizeof(float);  // ~ 64 MB (16M * 4 bytes/float)

    // device memory
    float *d_data, *d_out;
    CUDA_CHECK(cudaMalloc(&d_data, N_float_bytes));
    CUDA_CHECK(cudaMemset(d_data, 0, N_float_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, N_float_bytes));

    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("\nRunning fp32 throughput kernel ...");
    fp32_throughput<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);
    float ms_fp32 = benchmark([&]() {
        fp32_throughput<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);
    });
    
    printf("\nKernel time: %.4f ms\n", ms_fp32);
    double total_ffmas = (double)N * ITERS * ACCS;
    double total_flops = total_ffmas * 2.0;
    double seconds = ms_fp32 / 1000.0;
    double gflops = total_flops / seconds / 1e9;

    printf("\nFP32: %.1f GFLOP/s (%.2f TFLOP/s)\n", gflops, gflops / 1000.0);

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_out));
}