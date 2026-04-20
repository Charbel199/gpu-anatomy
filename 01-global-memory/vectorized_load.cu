#include "../common/check.cuh"
#include "../common/timer.cuh"
#include "../common/bandwidth.cuh"
#include "../common/device_info.cuh"
#include <algorithm>
#include <numeric>
#include <random>

#define BLOCK_SIZE 256

__global__ void only_read_coalesced(const float* __restrict__ data,int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) volatile float _ = data[idx];
}

__global__ void only_read_vectorized_float4_coalesced(const float* __restrict__ data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const float4* data_float4 = reinterpret_cast<const float4*>(data);
    
    if (idx < n) volatile float4 _ = data_float4[idx];
}

int main() {
    print_device_info();

    size_t sizes[] = {                                                            
        1 * 1024 * 1024,    //   1 MB
        4 * 1024 * 1024,    //   4 MB
        16 * 1024 * 1024,   //  16 MB
        64 * 1024 * 1024,   //  64 MB
        256 * 1024 * 1024,  // 256 MB
        512 * 1024 * 1024,  // 512 MB
        1024 * 1024 * 1024, // 1024 MB
    };
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for(int i = 0; i < n_sizes; i++){
        size_t N = sizes[i];
        size_t N_float_bytes = N * sizeof(float);

        // device memory
        float *d_data, *d_data2;
        CUDA_CHECK(cudaMalloc(&d_data, N_float_bytes));
        CUDA_CHECK(cudaMalloc(&d_data2, N_float_bytes));

        int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        size_t moved = N_float_bytes;  // read 
        
        printf("\n===========================\nFor size: %zu\n", N); 
        float ms_coal = benchmark([&]() {
            only_read_coalesced<<<grid, BLOCK_SIZE>>>(d_data, N);
        });
        print_bandwidth("Coalesced", moved, ms_coal);

        int grid_vectorized = (N/4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        float ms_vectorized = benchmark([&]() {
            only_read_vectorized_float4_coalesced<<<grid_vectorized, BLOCK_SIZE>>>(d_data2, N/4);
        });
        
        print_bandwidth("Coalesced Vectorized", moved, ms_vectorized);
        
        print_speedup("Coalesced", ms_coal, "Coalesced Vectorized", ms_vectorized);

        CUDA_CHECK(cudaFree(d_data));
        CUDA_CHECK(cudaFree(d_data2));
    }
}

// float4 vectorized loads reduce the instruction overhead, which only helps when the kernel is compute bound
// as this will reduce the number of instructions
// once the kernel is memory bound (example 128 MB+, bigger than the size of L2), then both approaches saturate the memory bus
// every kernel has a max load size of 128 bits (using float4 and int4)