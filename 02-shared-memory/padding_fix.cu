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

__global__ void shmem_no_padding(const float* __restrict__ data, int row, int col) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // shared mem across a block
    __shared__ float shmem_data[32][32];

    if (idx < row) if (idy <col) shmem_data[threadIdx.x%32][threadIdx.y%32] = data[idx*col+idy];
    __syncthreads();

    // read from shmem
    volatile float _ = shmem_data[threadIdx.x%32][threadIdx.y%32];
}

__global__ void shmem_padding(const float* __restrict__ data, int row, int col) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // shared mem across a block
    __shared__ float shmem_data[32][33];

    if (idx < row) if (idy <col) shmem_data[threadIdx.x%32][threadIdx.y%32] = data[idx*col+idy];
    __syncthreads();

    // read from shmem
    volatile float _ = shmem_data[threadIdx.x%32][threadIdx.y%32];
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
    printf("\nRunning shmem no padding tile kernel ...");
    shmem_no_padding<<<grid, block>>>(d_data, ROW, COL);

    printf("\nRunning shmem padding tile kernel ...");
    shmem_padding<<<grid, block>>>(d_data, ROW, COL);

    CUDA_CHECK(cudaFree(d_data));
}


/*
After running
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum ./bin/02-shared-memory/padding_fix

I got the following results:
==PROF== Profiling "shmem_no_padding" - 0: 0%....50%....100% - 1 pass
Running shmem no padding tile kernel ...
==PROF== Profiling "shmem_padding" - 1: 0%....50%....100% - 1 pass
Running shmem padding tile kernel ...==PROF== Disconnected from process 3163534
[3163534] padding_fix@127.0.0.1
  shmem_no_padding(const float *, int, int) (32, 32, 1)x(32, 32, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                1,015,808
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                1,015,808
    -------------------------------------------------------- ----------- ------------

  shmem_padding(const float *, int, int) (32, 32, 1)x(32, 32, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                        0
    -------------------------------------------------------- ----------- ------------

If we do the math, for the no padding version:
- Warp #0 (threadIdx.y = 0, threadIdx.x 0..31) (Warps are 32 consecutive threads in row-major, hence why threadIdx.x changes)
    Access pattern is ->
    - Thread 0: shmem[0][0] → flat index 0 → bank 0
    - Thread 1: shmem[1][0] → flat index 32 → bank 0
    - Thread 2: shmem[2][0] → flat index 64 → bank 0
    ...
    So we can see a 32-way bank conflict for every warp. Meaning we need 32-1=31 extra passes per warp
    Our blocks have 32x32=1024 threads -> 1024/32 = 32 warps per block
    For a 2D array of 1024x1024, we have a grid size of 32x32

    TOTAL -> 32x32 grid x 32 warps per block x 31 extra passes = 1,015,808 which perfectly aligns with the ncu report

Naturally, with the padding, we observe 0 bank conflicts as the offset fixed the bank conflicts in every warp.

Visualization of the indexing:

NO PADDING (shmem[32][32]) column access, warp 0 (threadIdx.y=0):
  row\col   0    1    2  ...  31
    0     [  0][  1][  2]...[ 31]   <- thread 0 reads [0][0] = flat  0, bank  0
    1     [ 32][ 33][ 34]...[ 63]   <- thread 1 reads [1][0] = flat 32, bank  0
    2     [ 64][ 65][ 66]...[ 95]   <- thread 2 reads [2][0] = flat 64, bank  0
    ...
    31    [992][993][994]...[1023]  <- thread31 reads [31][0]= flat 992,bank 0


PADDING (shmem[32][33]) same column access, warp 0:

  row\col   0    1    2  ...  31   32 (33 pad)
    0     [  0][  1][  2]...[ 31][ 32]   <- thread 0 reads [0][0] = flat   0, bank  0
    1     [ 33][ 34][ 35]...[ 64][ 65]   <- thread 1 reads [1][0] = flat  33, bank  1
    2     [ 66][ 67][ 68]...[ 97][ 98]   <- thread 2 reads [2][0] = flat  66, bank  2
    ...
    31    [...][...][...]...[...][...]   <- thread31 reads [31][0]= flat 1023,bank 31

  Row stride is now 33 (not 32). Since 33 mod 32 = 1, each thread shifts by 1 bank.
  All 32 threads land on distinct banks (0,1,2,...,31). Zero conflicts.

  The cost is one extra float per row = 32 wasted floats total in this case.
*/