#include "../common/check.cuh"
#include "../common/timer.cuh"
#include "../common/bandwidth.cuh"
#include "../common/device_info.cuh"
#include <algorithm>
#include <numeric>
#include <random>

#define N (1 << 24)  // 16M (2^24) elements
#define BLOCK_SIZE 32

__global__ void shared_atomic(const float* __restrict__ data, float* __restrict__ output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // shared mem across a block
    __shared__ float shmem_data;
    if (threadIdx.x == 0) shmem_data = 0;
    __syncthreads();
    

    if (idx < n) atomicAdd(&shmem_data, data[idx]);
    __syncthreads();

    // read from shmem
    if (threadIdx.x == 0) atomicAdd(&output[0], shmem_data);
}

__global__ void global_atomic(const float* __restrict__ data, float* __restrict__ output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) atomicAdd(&output[0], data[idx]);
}


int main() {
    print_device_info();

    size_t N_float_bytes = N * sizeof(float);  // ~ 64 MB (16M * 4 bytes/float)

    // device memory
    float *d_data, *d_out;
    CUDA_CHECK(cudaMalloc(&d_data, N_float_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_data, 0, N_float_bytes));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    bool ncu = true;

    if (!ncu){
        float ms_global = benchmark([&]() {
            global_atomic<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);
        });
        float ms_shared = benchmark([&]() {
            shared_atomic<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);
        });
        print_speedup("Global Atomic", ms_global, "Shared Atomic", ms_shared);
    } else {
        global_atomic<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);
        shared_atomic<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);
    }

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_out));
}


/*
Output of the code:

--- Comparison ---
Global Atomic                     20.595 ms
Shared Atomic                      0.641 ms
Speedup                            32.11x

And when we run: 
ncu --metrics l1tex__t_requests_pipe_lsu_mem_global_op_red.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum ./bin/02-shared-memory/shared_atomics

We get the following results:
==PROF== Profiling "global_atomic" - 0: 0%....50%....100% - 3 passes
==PROF== Profiling "shared_atomic" - 1: 0%....50%....100% - 3 passes
==PROF== Disconnected from process 3436556
[3436556] shared_atomics@127.0.0.1
  global_atomic(const float *, float *, int) (524288, 1, 1)x(32, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    ------------------------------------------------ ----------- ------------
    Metric Name                                      Metric Unit Metric Value
    ------------------------------------------------ ----------- ------------
    l1tex__t_requests_pipe_lsu_mem_global_op_red.sum                  524,288
    l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum       sector   16,777,216
    ------------------------------------------------ ----------- ------------

  shared_atomic(const float *, float *, int) (524288, 1, 1)x(32, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    ------------------------------------------------ ----------- ------------
    Metric Name                                      Metric Unit Metric Value
    ------------------------------------------------ ----------- ------------
    l1tex__t_requests_pipe_lsu_mem_global_op_red.sum                  524,288
    l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum       sector      524,288
    ------------------------------------------------ ----------- ------------

Let's first understand these metrics
    l1tex__t_requests_pipe_lsu_mem_global_op_red.sum -> warp-level atomic instructions issued
    l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum -> individual 32-byte memory transactions to global memory

For the global_atomic kernel, every warp's 32 threads do their own trip to main memory (every thread is performing an aotmic add with an HBM float)
    524,288 * 32 = 16,777,216
For the shared_atomic kernel, every warp works in shmem, and finally 1  thread talks with HBM
    524,288 * 1 = 524,288

Hence the x32 speedup
*/