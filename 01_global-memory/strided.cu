#include "../common/check.cuh"
#include "../common/timer.cuh"
#include "../common/bandwidth.cuh"
#include "../common/device_info.cuh"

#define N (1 << 24)  // 16M (2^24) elements
#define BLOCK_SIZE 256

__global__ void read_strided(const float* __restrict__ data, float* __restrict__ out, int n, int stride){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = data[idx*stride];
}



int main() {
    print_device_info();

    int strides[] = {1, 2, 4, 8, 16, 32, 64};
    int n_strides = sizeof(strides) / sizeof(strides[0]);
    int max_N = N * strides[n_strides-1]; // num of elements = max stride * N

    for(int i = 0; i<n_strides; i++){
        // our array is max_N size, but we are always moving N foats
        size_t N_float_bytes = max_N  * sizeof(float);
        size_t N_float_bytes_moved = N  * sizeof(float);

        // device memory
        float *d_data, *d_out;
        CUDA_CHECK(cudaMalloc(&d_data, N_float_bytes));
        CUDA_CHECK(cudaMalloc(&d_out, N_float_bytes));
        CUDA_CHECK(cudaMemset(d_data, 0, N_float_bytes));

        int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        size_t moved = 2 * N_float_bytes_moved;  // read + write

        float ms_coal = benchmark([&]() {
            read_strided<<<grid, BLOCK_SIZE>>>(d_data, d_out, max_N, strides[i]);
        });

        char label[64];
        sprintf(label, "Strided %d ", strides[i]);
        print_bandwidth(label, moved, ms_coal);

        CUDA_CHECK(cudaFree(d_data));
        CUDA_CHECK(cudaFree(d_out));
    }

}


// bandwidth drops as stride increases, as expected. After stride 8 the bandwidth is kind of the same, this is because
// once the stride is large enough, each thread's access hits a different cache line, so we arrive already at the worst case performance