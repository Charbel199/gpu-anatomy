#include "../common/check.cuh"
#include "../common/timer.cuh"
#include "../common/bandwidth.cuh"
#include "../common/device_info.cuh"
#include <algorithm>
#include <numeric>
#include <random>

#define BLOCK_SIZE 256

__global__ void read_coalesced(const float* __restrict__ data,
                               float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = data[idx];
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
        float *d_data, *d_out;
        int *d_idx;
        CUDA_CHECK(cudaMalloc(&d_data, N_float_bytes));
        CUDA_CHECK(cudaMalloc(&d_out, N_float_bytes));

        CUDA_CHECK(cudaMemset(d_data, 0, N_float_bytes));

        int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        size_t moved = 2 * N_float_bytes;  // read + write

        float ms_coal = benchmark([&]() {
            read_coalesced<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);
        });

        char label[64];
        sprintf(label, "Coalesced %zu MB", N/(1024*1024));
        print_bandwidth(label, moved, ms_coal);


        CUDA_CHECK(cudaFree(d_data));
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_idx));
    }

}
