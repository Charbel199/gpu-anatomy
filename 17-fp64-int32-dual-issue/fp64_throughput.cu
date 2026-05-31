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

__global__ void fp64_throughput(const double* __restrict__ data, double* __restrict__ out, int n){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= n) return;

    double seed = data[idx];
    double acc0 = seed;
    double acc1 = seed +1.0;
    double acc2 = seed +2.0;
    double acc3 = seed +3.0;
    double acc4 = seed +4.0;
    double acc5 = seed +5.0;
    double acc6 = seed +6.0;
    double acc7 = seed +7.0;
    
    const double c = 1.001;
    const double d = 0.5;

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

    size_t N_double_bytes = N * sizeof(double);  // ~ 128 MB (16M * 8 bytes/double)

    // device memory
    double *d_data, *d_out;
    CUDA_CHECK(cudaMalloc(&d_data, N_double_bytes));
    CUDA_CHECK(cudaMemset(d_data, 0, N_double_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, N_double_bytes));

    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("\nRunning fp64 throughput kernel ...");
    fp64_throughput<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);
    float ms_fp64 = benchmark([&]() {
        fp64_throughput<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);
    });
    
    printf("\nKernel time: %.4f ms\n", ms_fp64);
    double total_ffmas = (double)N * ITERS * ACCS;
    double total_flops = total_ffmas * 2.0;
    double seconds = ms_fp64 / 1000.0;
    double gflops = total_flops / seconds / 1e9;

    printf("\nFP64: %.1f GFLOP/s (%.2f TFLOP/s)\n", gflops, gflops / 1000.0);

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_out));
}