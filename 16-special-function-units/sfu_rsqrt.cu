#include "../common/timer.cuh"
#include "../common/bandwidth.cuh"
#include "../common/check.cuh"
#include "../common/device_info.cuh"

#define N (1 << 26)  // 64M (2^26) elements
#define BLOCK_SIZE 256


//   precise = sqrt + divide       (2 SFU ops + refinement on each)
//   fast    = combined 1/sqrt(x)  (1 SFU op + refinement)
__global__ void rsqrt_precise(const float* __restrict__ data, float* __restrict__ out, int n){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        float v = data[idx];
        #pragma unroll
        for (int i = 0; i < 32; i++) v = __fdiv_rn(1.0f, __fsqrt_rn(v));
        out[idx] = v;
    }
}

__global__ void rsqrt_fast(const float* __restrict__ data, float* __restrict__ out, int n){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        float v = data[idx];
        #pragma unroll
        for (int i = 0; i < 32; i++) v = __frsqrt_rn(v);
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

    bool ncu = false;

    if (!ncu) {
        printf("\nRunning rsqrt_precise ...");
        float ms_rsqrt_precise = benchmark([&]() {
            rsqrt_precise<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);
        });

        printf("\nRunning rsqrt_fast ...");    
        float ms_rsqrt_fast = benchmark([&]() {
            rsqrt_fast<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);
        });

        print_speedup("Rsqrt precise", ms_rsqrt_precise, "Rsqrt fast", ms_rsqrt_fast);
    } else {
        printf("\nRunning rsqrt_precise ...");
        rsqrt_precise<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);
        printf("\nRunning rsqrt_fast ...");
        rsqrt_fast<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);
    }
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_out));
}


/*
Running the code:

    Running rsqrt_precise ...
    Running rsqrt_fast ...
    --- Comparison ---
    Rsqrt precise                      2.400 ms
    Rsqrt fast                         0.462 ms
    Speedup                             5.19x

Both kernels are correctly-rounded (IEEE) and produce bit-identical output, but the
fast version computes 1/sqrt(x) in a single combined SFU operation while the precise
version runs sqrt then divide as two separate SFU operations + refinement on each.

Let's take a look at the SASS dump:
cuobjdump --dump-sass bin/16-special-function-units/sfu_rsqrt | grep -E "MUFU|FFMA|FMUL|FADD" > sfu_rsqrt.sass

We can see that:
- rsqrt_fast: one MUFU.RSQ per iteration + a few refinement FFMAs
- rsqrt_precise: MUFU.RSQ (for sqrt) + MUFU.RCP (for divide) per iteration, each followed by their own refinement FFMAs
*/
