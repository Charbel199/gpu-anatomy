#include "../common/check.cuh"
#include "../common/timer.cuh"
#include "../common/bandwidth.cuh"
#include "../common/device_info.cuh"
#include <algorithm>
#include <numeric>
#include <random>

#define N (1 << 24)  // 16M (2^24) elements
#define BLOCK_SIZE 32

__global__ void shmem_static(const float* __restrict__ data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // shared mem across a block
    __shared__ float shmem_data[32];

    // each thread writes 1 value into shared mem (HBM -> L2 -> L1 -> registers -> SHMEM, TODO: we will look at a more efficient way to do this later)
    if (idx < n) shmem_data[idx%32] = data[idx];
    __syncthreads();

    // read from shmem
    volatile float _ = shmem_data[idx%32];
}

__global__ void shmem_dynamic(const float* __restrict__ data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // shared mem across a block
    extern __shared__ float shmem_data[];

    // each thread writes 1 value into shared mem (HBM -> L2 -> L1 -> registers -> SHMEM, TODO: we will look at a more efficient way to do this later)
    if (idx < n) shmem_data[idx%32] = data[idx];
    __syncthreads();

    // read from shmem
    volatile float _ = shmem_data[idx%32];
}

int main() {
    print_device_info();

    size_t N_float_bytes = N * sizeof(float);  // ~ 64 MB (16M * 4 bytes/float)

    // device memory
    float *d_data, *d_out;
    CUDA_CHECK(cudaMalloc(&d_data, N_float_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, N_float_bytes));
    CUDA_CHECK(cudaMemset(d_data, 0, N_float_bytes));
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;


    printf("\nRunning static shmem  ...");
    shmem_static<<<grid, BLOCK_SIZE>>>(d_data, N);

    printf("\nRunning dynamic shmem  ...");
    size_t shmem_size = BLOCK_SIZE * sizeof(float);
    shmem_dynamic<<<grid, BLOCK_SIZE, shmem_size>>>(d_data, N);

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_out));
}


/*
After running
ncu --metrics launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic ./bin/02-shared-memory/dynamic_vs_static

I got the following results:

==PROF== Profiling "shmem_static" - 0: 0%....50%....100% - 1 pass
Running static shmem  ...
==PROF== Profiling "shmem_dynamic" - 1: 0%....50%....100% - 1 pass
Running dynamic shmem  ...==PROF== Disconnected from process 3285204
[3285204] dynamic_vs_static@127.0.0.1
  shmem_static(const float *, int) (524288, 1, 1)x(32, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    ------------------------------------ ----------- ------------
    Metric Name                          Metric Unit Metric Value
    ------------------------------------ ----------- ------------
    launch__shared_mem_per_block_dynamic  byte/block            0
    launch__shared_mem_per_block_static   byte/block          128
    ------------------------------------ ----------- ------------

  shmem_dynamic(const float *, int) (524288, 1, 1)x(32, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    ------------------------------------ ----------- ------------
    Metric Name                          Metric Unit Metric Value
    ------------------------------------ ----------- ------------
    launch__shared_mem_per_block_dynamic  byte/block          128
    launch__shared_mem_per_block_static   byte/block            0
    ------------------------------------ ----------- ------------

Pretty self-explanatory. 32 * sizeof(float) = 128 bytes/block
*/