#include "../common/timer.cuh"
#include "../common/bandwidth.cuh"
#include "../common/check.cuh"
#include "../common/device_info.cuh"


#include <cuda/pipeline>
#include <cuda/barrier>
#include <cooperative_groups/memcpy_async.h>

#define N (1 << 26)  // 64M (2^26) elements
#define BLOCK_SIZE 256

__global__ void cp_async(const float* __restrict__ data, float* __restrict__ out){
    int idx = threadIdx.x + blockDim.x * blockIdx.x; // general idx
    __shared__ float tile[BLOCK_SIZE];

    auto block = cooperative_groups::this_thread_block();
    cooperative_groups::memcpy_async(block, tile, data + blockIdx.x * BLOCK_SIZE, sizeof(float) * BLOCK_SIZE);
    // block -> which group is cooperating
    // tile -> DESTINATION address (shared mem)
    // data + blockIdx.x * BLOCK_SIZE -> SOURCE address (HBM) (equivalent to &data[blockIdx.x * BLOCK_SIZE])
    // sizeof(float) * BLOCK_SIZE -> number of bytes to copy
    
    // we are now going HBM -> L2 -> SMEM
    cooperative_groups::wait(block); // wait for all lanes in the block to finish their copies
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

    printf("\nRunning cp_async kernel ...");
    cp_async<<<grid, BLOCK_SIZE>>>(d_data, d_out);

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_out));
}


/*
We run
ncu --metrics launch__registers_per_thread,sm__cycles_elapsed.avg,dram__bytes_op_read.sum,dram__bytes_op_write.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,smsp__inst_executed_pipe_lsu.sum ./bin/10-async-copy/cp_async

[3762141] cp_async@127.0.0.1
  cp_async(const float *, float *) (262144, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    -------------------------------------------- --------------- ------------
    Metric Name                                      Metric Unit Metric Value
    -------------------------------------------- --------------- ------------
    dram__bytes_op_read.sum                                Mbyte       268.49
    dram__bytes_op_write.sum                               Mbyte       258.75
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum           Mbyte       268.44
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum           Mbyte       268.44
    launch__registers_per_thread                 register/thread           32
    sm__cycles_elapsed.avg                                 cycle   535,052.11
    smsp__inst_executed_pipe_lsu.sum                        inst   20,971,520
    -------------------------------------------- --------------- ------------

And compare against sync_load:

    Metric                     sync_load       cp_async
    ------------------------   ------------    ------------
    registers/thread                   16              32
    SM cycles                     529,381         535,052
    LSU instructions           10,485,760      20,971,520
    DRAM read                    268.65 MB       268.49 MB
    DRAM write                   234.34 MB       258.75 MB

Without cp.async (sync version):
  HBM -> L2 -> L1 -> register -> SMEM
           [LDG]    [STS]
Two instructions, value transits through a register.

With cp.async (async version):
  HBM -> L2 -> L1 -> SMEM 
        [LDGSTS] 

Let's also take a look at some SASS lines through:
cuobjdump --dump-sass bin/10-async-copy/cp_async | grep -E "LDGSTS|LDG|STS|STG|LDS|DEPBAR" | head -40

              @!PT LDS RZ, [RZ] ;
              @!PT LDS RZ, [RZ] ;
              @!PT LDS RZ, [RZ] ;
                   LDGSTS.E [R14], desc[UR6][R10.64] ;
              @!PT LDS RZ, [RZ] ;
              @!PT LDS RZ, [RZ] ;
              @!PT LDS RZ, [RZ] ;
                   LDGSTS.E [R6], desc[UR6][R26.64] ;
                   LDGSTS.E [R16], desc[UR6][R28.64] ;
                   LDGSTS.E [R18], desc[UR6][R22.64] ;
                   LDGSTS.E [R19], desc[UR6][R24.64] ;
                   LDGDEPBAR ;
                   DEPBAR.LE SB0, 0x0 ;
                   LDS R5, [R4+UR4] ;
                   STG.E desc[UR6][R2.64], R5 ;

Counter-intuitively, cp_async is WORSE here. The SASS does contain LDGSTS instructions
(the fused HBM -> SMEM async copy that skips registers), but cooperative_groups::memcpy_async
also emits:
    - multiple LDGSTS per thread (for alignment / chunking)
    - LDGDEPBAR pipeline barriers
    - @!PT LDS no-op loads used as hazard synchronization
    - extra registers to track pipeline state.

For a trivial pure-copy kernel like this one, that overhead outweighs the benefit of
skipping the register round-trip. cp.async only pays off when you can actually OVERLAP
the copy with compute (load tile N+1 while computing on tile N) - that's the point of
cp_async_pipelined.cu. Using cp.async without overlap is all cost, no benefit.
*/
