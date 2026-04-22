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

#define ARR_SIZE3 1024
__global__ void force_spill(const float* __restrict__ data, float* __restrict__ out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float arr[ARR_SIZE3]; // to emulate ~ARR_SIZE3 register use

    #pragma unroll
    for (int i = 0; i < ARR_SIZE3; i++) arr[i] = data[(idx + i * ARR_SIZE3) % n];   // fill registers

    float sum = 0;
    #pragma unroll
    for (int i = 0; i < ARR_SIZE3; i++) {
        sum += arr[i] * arr[(i * 37) % ARR_SIZE3];  // pseudo-random partner forces distant slots to stay live
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
  
    printf("\nRunning forced spill kernel ...");

    force_spill<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_out));
}


/*
We run 
ncu --metrics launch__registers_per_thread,l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum ./bin/03-registers/forced_spill

[2927457] forced_spill@127.0.0.1
  low_registers(const float *, float *, int) (65536, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    ------------------------------------------- --------------- ------------
    Metric Name                                     Metric Unit Metric Value
    ------------------------------------------- --------------- ------------
    l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum            byte            0
    l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum            byte            0
    launch__registers_per_thread                register/thread           36
    ------------------------------------------- --------------- ------------

  high_registers(const float *, float *, int) (65536, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    ------------------------------------------- --------------- ------------
    Metric Name                                     Metric Unit Metric Value
    ------------------------------------------- --------------- ------------
    l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum            byte            0
    l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum            byte            0
    launch__registers_per_thread                register/thread           80
    ------------------------------------------- --------------- ------------

  force_spill(const float *, float *, int) (65536, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    ------------------------------------------- --------------- ------------
    Metric Name                                     Metric Unit Metric Value
    ------------------------------------------- --------------- ------------
    l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum           Gbyte        83.15
    l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum           Gbyte        77.58
    launch__registers_per_thread                register/thread          255
    ------------------------------------------- --------------- ------------
TODO: Add info here

*/
