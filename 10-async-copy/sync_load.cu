#include "../common/timer.cuh"
#include "../common/bandwidth.cuh"
#include "../common/check.cuh"
#include "../common/device_info.cuh"

#define N (1 << 26)  // 64M (2^26) elements
#define BLOCK_SIZE 256

__global__ void sync_load(const float* __restrict__ data, float* __restrict__ out){
    int idx = threadIdx.x + blockDim.x * blockIdx.x; // general idx
    __shared__ float tile[BLOCK_SIZE];
    
    tile[threadIdx.x] = data[idx]; // HBM -> L2 -> L1 -> register -> SMEM
    __syncthreads();

    float v = tile[(threadIdx.x + 1) % BLOCK_SIZE]; // read a different slot so that the shared write can't be eliminated by the compiler
    out[idx] = v;
}


int main() {
    print_device_info();

    size_t N_float_bytes = N * sizeof(float);  // ~ 256 MB (64M * 4 bytes/float)

    // device memory
    float *d_data, *d_out;
    CUDA_CHECK(cudaMalloc(&d_data, N_float_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, N_float_bytes));
    CUDA_CHECK(cudaMemset(d_data, 0, N_float_bytes));
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("\nRunning sync_load kernel ...");
    sync_load<<<grid, BLOCK_SIZE>>>(d_data, d_out);

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_out));
}


/*
Let's take a look at the SASS code, and specifically the Load and Store instructions
cuobjdump --dump-sass bin/10-async-copy/sync_load | grep -E "LDG|STS|LDS|STG" > sync_load.sass

    LDG.E.CONSTANT R0, desc[UR6][R2.64] ;  
    STS [R9], R0 ;                         
    LDS R11, [R6+UR4] ;                    
    STG.E desc[UR6][R2.64], R11 ;          

Now we profile with
ncu --metrics launch__registers_per_thread,sm__cycles_elapsed.avg,dram__bytes_op_read.sum,dram__bytes_op_write.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,smsp__inst_executed_pipe_lsu.sum ./bin/10-async-copy/sync_load 

==PROF== Profiling "sync_load" - 0: 0%....50%....100% - 1 pass
Running sync_load kernel ...==PROF== Disconnected from process 3635261
[3635261] sync_load@127.0.0.1
  sync_load(const float *, float *) (262144, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    -------------------------------------------- --------------- ------------
    Metric Name                                      Metric Unit Metric Value
    -------------------------------------------- --------------- ------------
    dram__bytes_op_read.sum                                Mbyte       268.65
    dram__bytes_op_write.sum                               Mbyte       234.34
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum           Mbyte       268.44
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum           Mbyte       268.44
    launch__registers_per_thread                 register/thread           16
    sm__cycles_elapsed.avg                                 cycle   529,381.71
    smsp__inst_executed_pipe_lsu.sum                        inst   10,485,760
    -------------------------------------------- --------------- ------------
*/
