#include "../common/timer.cuh"
#include "../common/bandwidth.cuh"
#include "../common/check.cuh"
#include "../common/device_info.cuh"

#define N (1 << 26)  // 64M (2^26) elements
#define BLOCK_SIZE 256

__global__ void sin_precise(const float* __restrict__ data, float* __restrict__ out, int n){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        float v = data[idx];
        #pragma unroll
        for (int i = 0; i < 32; i++) v = sinf(v);
        out[idx] = v;
    }
}

__global__ void sin_fast(const float* __restrict__ data, float* __restrict__ out, int n){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        float v = data[idx];
        #pragma unroll
        for (int i = 0; i < 32; i++) v = __sinf(v);
        out[idx] = v;
    }
}

int main() {
    print_device_info();

    size_t N_float_bytes = N * sizeof(float);

    // device memory
    float *d_data, *d_out;
    CUDA_CHECK(cudaMalloc(&d_data, N_float_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, N_float_bytes));
    CUDA_CHECK(cudaMemset(d_data, 0, N_float_bytes));
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    bool ncu = true;

    if (!ncu) {
        printf("\nRunning sin_precise ...");
        float ms_sin_precise = benchmark([&]() {
            sin_precise<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);
        });

        printf("\nRunning sin_fast ...");    
        float ms_sin_fast = benchmark([&]() {
            sin_fast<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);
        });

        print_speedup("Sin precise", ms_sin_precise, "Sin fast", ms_sin_fast);
    } else {
        printf("\nRunning sin_precise ...");
        sin_precise<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);
        printf("\nRunning sin_fast ...");
        sin_fast<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);
    }
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_out));
}


/*
Running the code:

    Running sin_precise ...
    Running sin_fast ...
    --- Comparison ---
    Sin precise                        0.847 ms
    Sin fast                           0.357 ms
    Speedup                             2.37x

A clear performance gain just by using the special function units.

Note: By compiling with --use_fast_math (and without changing the source code), the compiler would automatically swap
the regular math functions to special function units operations.

Let's take a look at the SASS dump:
cuobjdump --dump-sass bin/16-special-function-units/sfu_sincos | grep -E "MUFU|FFMA|FMUL|FADD" > sfu_sincos.sass    

We can see that:
- sin_fast mainly executes "MUFU.SIN" operations
- sin_precise mainly executes "FFMA" operations
*/
