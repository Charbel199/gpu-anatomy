#include "../common/timer.cuh"
#include "../common/bandwidth.cuh"
#include "../common/check.cuh"
#include "../common/device_info.cuh"

#define BLOCK_SIZE 256
#define N (1 << 24)  // 16M (2^24) elements

#define ARR_SIZE 16
__global__ void low_registers(const float* __restrict__ data, float* __restrict__ out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float arr[ARR_SIZE]; // to emulate ~ARR_SIZE register use

    #pragma unroll
    for (int i = 0; i < ARR_SIZE; i++) arr[i] = data[(idx + i * ARR_SIZE) % n];   // fill registers

    float sum = 0;
    #pragma unroll
    for (int i = 0; i < ARR_SIZE; i++) {
        sum += arr[i] * arr[(i * 37) % ARR_SIZE];  // pseudo-random partner forces distant slots to stay live
    }

    if(idx < n) out[idx] = sum;
}

#define ARR_SIZE2 128
__global__ void high_registers(const float* __restrict__ data, float* __restrict__ out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float arr[ARR_SIZE2]; // to emulate ~ARR_SIZE2 register use

    #pragma unroll
    for (int i = 0; i < ARR_SIZE2; i++) arr[i] = data[(idx + i * ARR_SIZE2) % n];   // fill registers

    float sum = 0;
    #pragma unroll
    for (int i = 0; i < ARR_SIZE2; i++) {
        sum += arr[i] * arr[(i * 37) % ARR_SIZE2];  // pseudo-random partner forces distant slots to stay live
    }

    if(idx < n) out[idx] = sum;
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

    printf("\nRunning low registers kernel ...");

    low_registers<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);

    printf("\nRunning high registers kernel ...");

    high_registers<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);
  


    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_out));
}


/*
We run 
ncu --metrics launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active ./bin/03-registers/high_register

And get the following results

==PROF== Profiling "low_registers" - 0: 0%....50%....100% - 1 pass
Running low registers kernel ...
==PROF== Profiling "high_registers" - 1: 0%....50%....100% - 1 pass
Running high registers kernel ...==PROF== Disconnected from process 1199424
[1199424] high_register@127.0.0.1
  low_registers(const float *, float *, int) (65536, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    ------------------------------------------------- --------------- ------------
    Metric Name                                           Metric Unit Metric Value
    ------------------------------------------------- --------------- ------------
    launch__registers_per_thread                      register/thread           36
    sm__warps_active.avg.pct_of_peak_sustained_active               %        91.87
    ------------------------------------------------- --------------- ------------

  high_registers(const float *, float *, int) (65536, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    ------------------------------------------------- --------------- ------------
    Metric Name                                           Metric Unit Metric Value
    ------------------------------------------------- --------------- ------------
    launch__registers_per_thread                      register/thread           80
    sm__warps_active.avg.pct_of_peak_sustained_active               %        47.92
    ------------------------------------------------- --------------- ------------

Everything seems in order, we are using more registers in every thread, and since every thread requires 80 registers,
we can only fit 4 warps per SM which results in a significant drop in occupancy and consequently in the achieved active warps.

Doing the math, we have a BLOCK SIZE of 256, and 80 registers per thread -> 80x256 = 20,480 registers per block

And on an RTX6000pro, there are 65,536 registers per SM -> 65,536/20,480 = 3.2 ~3 blocks per SM, active warps = 3 * (256/32) = 24 warps active
Theoretical max warps per sm = Max threads per SM / 32 = 1,5636 / 32 = 48

24/48 = 50% active warps, very close to 47.92 that we got.
*/
