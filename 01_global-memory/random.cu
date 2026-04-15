#include "../common/check.cuh"
#include "../common/timer.cuh"
#include "../common/bandwidth.cuh"
#include "../common/device_info.cuh"
#include <algorithm>
#include <numeric>
#include <random>

#define N (1 << 24)  // 16M (2^24) elements
#define BLOCK_SIZE 256

__global__ void read_coalesced(const float* __restrict__ data,
                               float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = data[idx];
}

__global__ void read_random(const float* __restrict__ data,
                            const int* __restrict__ indices,
                            float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = data[indices[idx]];
}

int main() {
    print_device_info();

    size_t N_float_bytes = N * sizeof(float);  // ~ 64 MB (16M * 4 bytes/float)

    // shuffled indices on host
    int* h_idx = new int[N];
    std::iota(h_idx, h_idx + N, 0);
    std::shuffle(h_idx, h_idx + N, std::mt19937{42});

    // device memory
    float *d_data, *d_out;
    int *d_idx;
    CUDA_CHECK(cudaMalloc(&d_data, N_float_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, N_float_bytes));
    CUDA_CHECK(cudaMalloc(&d_idx, N * sizeof(int)));

    CUDA_CHECK(cudaMemset(d_data, 0, N_float_bytes));
    CUDA_CHECK(cudaMemcpy(d_idx, h_idx, N * sizeof(int), cudaMemcpyHostToDevice));

    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t moved = 2 * N_float_bytes;  // read + write

    float ms_coal = benchmark([&]() {
        read_coalesced<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);
    });
    print_bandwidth("Coalesced", moved, ms_coal);

    float ms_rand = benchmark([&]() {
        read_random<<<grid, BLOCK_SIZE>>>(d_data, d_idx, d_out, N);
    });
    
    print_bandwidth("Random", moved, ms_rand);

    print_speedup("Coalesced", ms_coal, "Random", ms_rand);

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_idx));
    delete[] h_idx;
}
