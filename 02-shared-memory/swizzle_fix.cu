#include "../common/check.cuh"
#include "../common/timer.cuh"
#include "../common/bandwidth.cuh"
#include "../common/device_info.cuh"
#include <algorithm>
#include <numeric>
#include <random>

#define ROW (1 << 10)  // 1024 elements
#define COL (1 << 10)  // 1024 elements
#define BLOCK_SIZE 32

__global__ void shmem_no_swizzle(const float* __restrict__ data, int row, int col) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // shared mem across a block
    __shared__ float shmem_data[32][32];

    if (idx < row) if (idy <col) shmem_data[threadIdx.x%32][threadIdx.y%32] = data[idx*col+idy];
    __syncthreads();

    // read from shmem
    volatile float _ = shmem_data[threadIdx.x%32][threadIdx.y%32];
}

__global__ void shmem_swizzle(const float* __restrict__ data, int row, int col) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // shared mem across a block
    __shared__ float shmem_data[32][32];

    int col_idx = threadIdx.y ^ threadIdx.x; // swizzling
    if (idx < row) if (idy <col) shmem_data[threadIdx.x%32][col_idx] = data[idx*col+idy];
    __syncthreads();

    // read from shmem
    volatile float _ = shmem_data[threadIdx.x%32][col_idx];
}



int main() {
    print_device_info();

    size_t N_float_bytes = ROW * COL * sizeof(float);

    // device memory
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N_float_bytes));
    CUDA_CHECK(cudaMemset(d_data, 0, N_float_bytes));
    dim3 grid((ROW + BLOCK_SIZE - 1) / BLOCK_SIZE, (COL + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    printf("\nRunning shmem no swizzle tile kernel ...");
    shmem_no_swizzle<<<grid, block>>>(d_data, ROW, COL);

    printf("\nRunning shmem swizzle tile kernel ...");
    shmem_swizzle<<<grid, block>>>(d_data, ROW, COL);

    CUDA_CHECK(cudaFree(d_data));
}


/*
After running
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum ./bin/02-shared-memory/swizzle_fix

I got the following results:
==PROF== Profiling "shmem_no_swizzle" - 0: 0%....50%....100% - 1 pass
Running shmem no swizzle tile kernel ...
==PROF== Profiling "shmem_swizzle" - 1: 0%....50%....100% - 1 pass
Running shmem swizzle tile kernel ...==PROF== Disconnected from process 3189860
[3189860] swizzle_fix@127.0.0.1
  shmem_no_swizzle(const float *, int, int) (32, 32, 1)x(32, 32, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                1,015,808
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                1,015,808
    -------------------------------------------------------- ----------- ------------

  shmem_swizzle(const float *, int, int) (32, 32, 1)x(32, 32, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                        0
    -------------------------------------------------------- ----------- ------------

The goal of swizzling is to remap column indices so that consecutive threads (in a warp) don't all land on the same bank.
XOR costs a single instruction, it follows the bijection rule if x1 != x2, then y^x1 != y^x2 meaning that we won't have collisions

Visualization of the indexing:

NO SWIZZLE (shmem[32][32]) column access, warp 0 (threadIdx.y=0, threadIdx.x=0..31):

  row\col   0    1    2  ...  31
    0     [  0][  1][  2]...[ 31]   <- thread 0 reads [0][0] = flat  0, bank  0
    1     [ 32][ 33][ 34]...[ 63]   <- thread 1 reads [1][0] = flat 32, bank  0
    2     [ 64][ 65][ 66]...[ 95]   <- thread 2 reads [2][0] = flat 64, bank  0
    ...
    31    [992][993][994]...[1023]  <- thread31 reads [31][0]= flat 992,bank 0

  All 32 threads hit bank 0 -> 32-way conflict.


SWIZZLE (shmem[32][32], col_idx = threadIdx.y ^ threadIdx.x), column access, warp 0 (threadIdx.y=0):

  For warp 0 (y=0), col_idx = 0 ^ threadIdx.x = threadIdx.x, so each thread picks a different column.

  row\col   0    1    2  ...  31
    0     [  *][   ][   ]...[   ]  <- thread 0: col = 0^0 =  0 -> [0][ 0] = flat   0, bank  0
    1     [   ][  *][   ]...[   ]  <- thread 1: col = 0^1 =  1 -> [1][ 1] = flat  33, bank  1
    2     [   ][   ][  *]...[   ]  <- thread 2: col = 0^2 =  2 -> [2][ 2] = flat  66, bank  2
    ...
    31    [   ][   ][   ]...[  *]  <- thread31: col = 0^31= 31 -> [31][31]= flat 1023,bank 31

  All 32 threads land on distinct banks (0,1,2,...,31). Zero conflicts.

For warp 1 (threadIdx.y=1), col_idx = 1 ^ threadIdx.x produces a different permutation (1,0,3,2,5,4,...)
but still a bijection so again, 32 unique banks, no conflicts.

Unlike padding, swizzling wastes zero memory the array is still [32][32].
*/