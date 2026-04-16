#include "../common/check.cuh"
#include "../common/timer.cuh"
#include "../common/bandwidth.cuh"
#include "../common/device_info.cuh"
#include <algorithm>
#include <numeric>
#include <random>

#define N (1 << 24)  // 16M (2^24) elements
#define BLOCK_SIZE 32

__global__ void shmem_32way_conflict(const float* __restrict__ data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // shared mem across a block
    __shared__ float shmem_data[1024];

    // each thread writes 1 value into shared mem (HBM -> L2 -> L1 -> registers -> SHMEM, TODO: we will look at a more efficient way to do this later)
    if (idx < n) shmem_data[threadIdx.x*32] = data[idx];
    __syncthreads();

    // read from shmem
    volatile float _ = shmem_data[threadIdx.x*32];
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

    printf("\nRunning shmem 32 way conflict kernel ...");
    shmem_32way_conflict<<<grid, BLOCK_SIZE>>>(d_data, N);

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_out));
}


/*
Every thread accesses the same shmem bank

After running
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum ./bin/02-shared-memory/32way_conflict
==PROF== Profiling "shmem_32way_conflict" - 0: 0%....50%....100% - 1 pass
Running shmem 32 way conflict kernel ...==PROF== Disconnected from process 4187341
[4187341] 32way_conflict@127.0.0.1
  shmem_32way_conflict(const float *, int) (524288, 1, 1)x(32, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum               16,628,270
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum               16,283,712
    -------------------------------------------------------- ----------- ------------

Let's see if this matches our calculations, a 32-way conflict requires 32-1=31 extra passes.
31*524,288 = 16,252,928, very close to what we're seeing with ncu
*/