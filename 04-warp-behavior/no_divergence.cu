#include "../common/timer.cuh"
#include "../common/bandwidth.cuh"
#include "../common/check.cuh"
#include "../common/device_info.cuh"


#define BLOCK_SIZE 256
#define N ((1 << 24)+7)  // 16M (2^24) elements

__global__ void no_divergence(const float* __restrict__ data, float* __restrict__ out, int n){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx<n) out[idx] = data[idx];
}


int main(){
    print_device_info();

    size_t N_float_bytes = N * sizeof(float);  // ~ 64 MB (16M * 4 bytes/float)

    // device memory
    float *d_data, *d_out;
    CUDA_CHECK(cudaMalloc(&d_data, N_float_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, N_float_bytes));
    CUDA_CHECK(cudaMemset(d_data, 0, N_float_bytes));
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("\nRunning no divergence kernel ...");
    no_divergence<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_out));

}


/*
They key metrics here are

- smsp__thread_inst_executed_per_inst_executed.ratio
    It measures the average number of active threads per issued instruction in a warp, the goal is to hit all 32 threads in a warp:
    - 32 -> no divergence
    - 16 -> half divergence
    - 1 -> full divergence (1 thread active per instruction)

- smsp__sass_average_branch_targets_threads_uniform.pct
    It measures branch unofrmity (for every branch the kernel hit, did all 32 threads of the warp pick the same direction?)
    - 100 % -> no divergence
    - 0 % -> full divergence

After running
ncu --metrics smsp__thread_inst_executed_per_inst_executed.ratio,smsp__sass_average_branch_targets_threads_uniform.pct ./bin/04-warp-behavior/no_divergence

We got:
==PROF== Profiling "no_divergence" - 0: 0%....50%....100% - 3 passes
Running no divergence kernel ...==PROF== Disconnected from process 589727
[589727] no_divergence@127.0.0.1
  no_divergence(const float *, float *, int) (65536, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    ----------------------------------------------------- ----------- ------------
    Metric Name                                           Metric Unit Metric Value
    ----------------------------------------------------- ----------- ------------
    smsp__sass_average_branch_targets_threads_uniform.pct           %            0
    smsp__thread_inst_executed_per_inst_executed.ratio                          32
    ----------------------------------------------------- ----------- ------------

The 32 for the average number of active threads per issued instruction in a warp makes complete sense. But why did we get 0% for smsp__sass_average_branch_targets_threads_uniform.pct ?

If we go down to SASS
cuobjdump --dump-sass bin/04-warp-behavior/no_divergence > no_divergence.sass

                           LDC R1, c[0x0][0x37c] ;
                           S2R R7, SR_TID.X ;
                           LDCU UR4, c[0x0][0x360] ;
                           S2R R0, SR_CTAID.X ;
                           LDCU UR5, c[0x0][0x390] ;
                           IMAD R7, R0, UR4, R7 ;
                           ISETP.GE.AND P0, PT, R7, UR5, PT ;
                       @P0 EXIT ;
                           LDC.64 R2, c[0x0][0x380] ;
                           LDCU.64 UR4, c[0x0][0x358] ;
                           LDC.64 R4, c[0x0][0x388] ;
                           IMAD.WIDE R2, R7, 0x4, R2 ;
                           LDG.E.CONSTANT R2, desc[UR4][R2.64] ;
                           HFMA2 R9, -RZ, RZ, 2, 0 ;
                           IMAD.WIDE R4, R7, 0x4, R4 ;
                           FFMA R0, R2, R9, 1 ;
                           FMUL R7, R0, R0 ;
                           STG.E desc[UR4][R4.64], R7 ;
                           EXIT ;
Address 0130               BRA 0x130;

The compiler optimized the if statement into predicated instructions:
        ISETP.GE.AND P0, PT, R7, UR5, PT ;
    @P0 EXIT ;

In simple terms, P0 = (R7 >= UR5) = (idx >= N), the compiler actually flipped the condition
The EXIT instruction is issued to the warp scheduler regardless, but each lane's internal pipeline only 
commits it when P0 is true for that lane. Lanes where P0 is false make the EXIT a no-op.

In even simpler terms, predications is the compiler's way to optimize branching away (when it is beneficial), every thread runs every instruction
But every thread has a per-lane bit (like P0, a predicate register) that decides whether the instruction's result actually takes effect.
*/