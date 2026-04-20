#include "../common/timer.cuh"
#include "../common/bandwidth.cuh"
#include "../common/check.cuh"
#include "../common/device_info.cuh"

#define BLOCK_SIZE 256
#define N (1 << 24)  // 16M (2^24) elements

__global__ void no_registers(const float* __restrict__ data, float* __restrict__ out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        out[idx] = data[idx];
    }
}

// the goal is to force the compiler to keep the ARR_SIZE floats alive simultaneously so the register count scales with ARR_SIZE
// we should at least use ARR_SIZE + address computation overhead registers
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


int main() {
    print_device_info();

    size_t N_float_bytes = N * sizeof(float);  // ~ 64 MB (16M * 4 bytes/float)

    // device memory
    float *d_data, *d_out;
    CUDA_CHECK(cudaMalloc(&d_data, N_float_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, N_float_bytes));
    CUDA_CHECK(cudaMemset(d_data, 0, N_float_bytes));
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("\nRunning no registers kernel ...");
    no_registers<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);

    printf("\nRunning low registers kernel ...");
    low_registers<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_out));
}


/*
Let's first start by profiling these kernels
ncu --metrics launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active ./bin/03-registers/low_register

==PROF== Profiling "no_registers" - 0: 0%....50%....100% - 1 pass
Running no registers kernel ...
==PROF== Profiling "low_registers" - 1: 0%....50%....100% - 1 pass
Running low registers kernel ...==PROF== Disconnected from process 1581249
[1581249] low_register@127.0.0.1
  no_registers(const float *, float *, int) (65536, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    ------------------------------------------------- --------------- ------------
    Metric Name                                           Metric Unit Metric Value
    ------------------------------------------------- --------------- ------------
    launch__registers_per_thread                      register/thread           16
    sm__warps_active.avg.pct_of_peak_sustained_active               %        80.45
    ------------------------------------------------- --------------- ------------

  low_registers(const float *, float *, int) (65536, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    ------------------------------------------------- --------------- ------------
    Metric Name                                           Metric Unit Metric Value
    ------------------------------------------------- --------------- ------------
    launch__registers_per_thread                      register/thread           36
    sm__warps_active.avg.pct_of_peak_sustained_active               %        91.80
    ------------------------------------------------- --------------- ------------

We get 16 registers/thread for the no_registers kernel and 36 registers/thread for the low_registers kernel which makes a lot of sense.
To go a bit lower, we can look at the PTX code but it only holds virtual registers.
To view the actual real hardware allocations, we might as well go to the SASS level directly.

By running the following command we can take a look at the SASS code 
cuobjdump --dump-sass bin/03-registers/low_register > low_register.sass

        LDC R1, c[0x0][0x37c] ;
        S2R R0, SR_TID.X ;
        S2UR UR4, SR_CTAID.X ;
        LDCU UR5, c[0x0][0x390] ;
        LDC R7, c[0x0][0x360] ;
        IMAD R7, R7, UR4, R0 ;
        ISETP.GE.AND P0, PT, R7, UR5, PT ;                                                       
    @P0 EXIT ;
        LDC.64 R2, c[0x0][0x380] ;
        LDCU.64 UR4, c[0x0][0x358] ;
        LDC.64 R4, c[0x0][0x388] ;
        IMAD.WIDE R2, R7, 0x4, R2 ;
        LDG.E.CONSTANT R3, desc[UR4][R2.64] ;
        IMAD.WIDE R4, R7, 0x4, R4 ;
        STG.E desc[UR4][R4.64], R3 ;
        EXIT ;   

The no registers kernel has 16 registers allocated, in reality it only uses 6
  - R1 - stack pointer (always reserved)
  - R0 - threadIdx.x
  - R7 - blockIdx.x * blockDim.x + threadIdx.x (idx)
  - R2, R3 - 64-bit pointer to data + the float you load
  - R4, R5 - 64-bit pointer to out (every register is 32-bit wide, for 64-bit values the compiler uses 2 adjacent registers as a pair)
ncu shows 16 registers/thread, this could simply be the minimum number of registers allocated per thread (to be checked)

The low registers kernel is a bit too big in SASS to print here, but it uses around 34 registers and ncu rounds it up to 36.
*/