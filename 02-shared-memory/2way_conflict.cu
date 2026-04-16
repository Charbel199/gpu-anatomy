#include "../common/check.cuh"
#include "../common/timer.cuh"
#include "../common/bandwidth.cuh"
#include "../common/device_info.cuh"
#include <algorithm>
#include <numeric>
#include <random>

#define N (1 << 24)  // 16M (2^24) elements
#define BLOCK_SIZE 32

__global__ void shmem_2way_conflict(const float* __restrict__ data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // shared mem across a block
    __shared__ float shmem_data[64];

    // each thread writes 1 value into shared mem (HBM -> L2 -> L1 -> registers -> SHMEM, TODO: we will look at a more efficient way to do this later)
    if (idx < n) shmem_data[(threadIdx.x*2)%64] = data[idx];
    __syncthreads();

    // read from shmem
    volatile float _ = shmem_data[(threadIdx.x*2)%64];
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

    printf("\nRunning shmem 2 way conflict kernel ...");
    shmem_2way_conflict<<<grid, BLOCK_SIZE>>>(d_data, N);

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_out));
}


/*
In this kernel, 
    thread 0 -> shmem[0]
    thread 1 -> shmem[2]
    thread 2 -> shmem[4]
    ...
    thread 16 -> shmem[32] BANK CONFLICT WITH thread 0
So on average we should have 16 bank conflicts per warp


After running
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum ./bin/02-shared-memory/2way_conflict
==PROF== Profiling "shmem_2way_conflict" - 0: 0%....50%....100% - 1 pass
Running shmem 2 way conflict kernel ...==PROF== Disconnected from process 4155767
[4155767] 2way_conflict@127.0.0.1
  shmem_2way_conflict(const float *, int) (524288, 1, 1)x(32, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                  526,019
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                  527,486
    -------------------------------------------------------- ----------- ------------

To understand this number, we have to really understand the ncu metrics:
    The metric counts extra replay passes per warp, not individual bank collisions. 
    A 2-way conflict means every bank needs 1 extra pass to resolve,
    but all 32 banks serialize within that single extra pass,
    so the whole warp costs exactly 1 extra pass. That gives 1 × 524,288 = 524,288 (with some overhead),
    which matches perfectly. In general, an N-way conflict costs N-1 extra passes per warp.
*/