// a cache line is 128 bytes split into 4 32-byte sectors
// the hardware fetches secotrs with at least 1 byte accessed
// if 32 threads (a warp) each ready 32bit floats(4 bytes) at consecutive address, 32 threads * 4 bytes = 128 bytes
// we would have used the 4 sectors -> 100% utilization
// on the other hand if we only load 1 byte per sector, we still have to pull the entire sector (32 bytes)
// and we would only have a 1/32 utilization (useful bytes/fetched bytes => very low)

#include "../common/check.cuh"
#include "../common/timer.cuh"
#include "../common/bandwidth.cuh"
#include "../common/device_info.cuh"
#include <algorithm>
#include <numeric>
#include <random>

#define N (1 << 24)  // 16M (2^24) elements
#define BLOCK_SIZE 256

__global__ void read_coalesced(const float* __restrict__ data,
                               float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = data[idx];
}

__global__ void read_strided(const float* __restrict__ data, float* __restrict__ out, int n, int stride){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = data[idx*stride];
}

__global__ void read_random(const float* __restrict__ data,
                            const int* __restrict__ indices,
                            float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = data[indices[idx]];
}

int main() {
    print_device_info();

    size_t N_float_bytes = N * sizeof(float);  // ~ 64 MB (16M * 4 bytes/float)

    int strides[] = {2, 4, 8, 16, 32, 64};
    int n_strides = sizeof(strides) / sizeof(strides[0]);
    int max_N = N * strides[n_strides-1]; // num of elements = max stride * N

    // shuffled indices on host
    int* h_idx = new int[N];
    std::iota(h_idx, h_idx + N, 0);
    std::shuffle(h_idx, h_idx + N, std::mt19937{42});

    // device memory
    float *d_data, *d_out;
    int *d_idx;
    CUDA_CHECK(cudaMalloc(&d_data, N_float_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, N_float_bytes));
    CUDA_CHECK(cudaMalloc(&d_idx, N * sizeof(int)));

    CUDA_CHECK(cudaMemset(d_data, 0, N_float_bytes));
    CUDA_CHECK(cudaMemcpy(d_idx, h_idx, N * sizeof(int), cudaMemcpyHostToDevice));

    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    printf("Runing coalesced ...\n");
    read_coalesced<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);
    printf("Runing strided ...\n");
    for(int i = 0; i<n_strides; i++){
        // our array is max_N size, but we are always moving N foats
        size_t N_float_bytes_strided = max_N  * sizeof(float);

        // device memory
        float *d_data_strided, *d_out_strided;
        CUDA_CHECK(cudaMalloc(&d_data_strided, N_float_bytes_strided));
        CUDA_CHECK(cudaMalloc(&d_out_strided, N_float_bytes_strided));
        CUDA_CHECK(cudaMemset(d_data_strided, 0, N_float_bytes_strided));

        int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        read_strided<<<grid, BLOCK_SIZE>>>(d_data_strided, d_out_strided, max_N, strides[i]);

        CUDA_CHECK(cudaFree(d_data_strided));
        CUDA_CHECK(cudaFree(d_out_strided));
    }
    printf("Runing random ...\n");
    read_random<<<grid, BLOCK_SIZE>>>(d_data, d_idx, d_out, N);

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_idx));
    delete[] h_idx;

}

/*
After running `ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum ./bin/01_global-memory/sector_utilization`
We got the following results :

[3822842] sector_utilization@127.0.0.1
  read_coalesced(const float *, float *, int) (65536, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    ----------------------------------------------- ----------- ------------
    Metric Name                                     Metric Unit Metric Value
    ----------------------------------------------- ----------- ------------
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                  524,288
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum       sector    2,097,152
    ----------------------------------------------- ----------- ------------

  read_strided(const float *, float *, int, int) (65536, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    ----------------------------------------------- ----------- ------------
    Metric Name                                     Metric Unit Metric Value
    ----------------------------------------------- ----------- ------------
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                  524,288
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum       sector    4,194,304
    ----------------------------------------------- ----------- ------------

  read_strided(const float *, float *, int, int) (65536, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    ----------------------------------------------- ----------- ------------
    Metric Name                                     Metric Unit Metric Value
    ----------------------------------------------- ----------- ------------
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                  524,288
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum       sector    8,388,608
    ----------------------------------------------- ----------- ------------

  read_strided(const float *, float *, int, int) (65536, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    ----------------------------------------------- ----------- ------------
    Metric Name                                     Metric Unit Metric Value
    ----------------------------------------------- ----------- ------------
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                  524,288
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum       sector   16,777,216
    ----------------------------------------------- ----------- ------------

  read_strided(const float *, float *, int, int) (65536, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    ----------------------------------------------- ----------- ------------
    Metric Name                                     Metric Unit Metric Value
    ----------------------------------------------- ----------- ------------
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                  524,288
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum       sector   16,777,216
    ----------------------------------------------- ----------- ------------

  read_strided(const float *, float *, int, int) (65536, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    ----------------------------------------------- ----------- ------------
    Metric Name                                     Metric Unit Metric Value
    ----------------------------------------------- ----------- ------------
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                  524,288
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum       sector   16,777,216
    ----------------------------------------------- ----------- ------------

  read_strided(const float *, float *, int, int) (65536, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    ----------------------------------------------- ----------- ------------
    Metric Name                                     Metric Unit Metric Value
    ----------------------------------------------- ----------- ------------
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                  524,288
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum       sector   16,777,216
    ----------------------------------------------- ----------- ------------

  read_random(const float *, const int *, float *, int) (65536, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    ----------------------------------------------- ----------- ------------
    Metric Name                                     Metric Unit Metric Value
    ----------------------------------------------- ----------- ------------
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                1,048,576
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum       sector   18,874,261
    ----------------------------------------------- ----------- ------------

To understand a bit what we're seeing, 
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum -> Number of warp loads (N= 1<<24 ~ 16M) a warp is 32 threads 16M/32 = 524,288
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum -> How many secotrs were fetched

    e.g. for the coalesced case: 
    We have 524,288 warp loads and 2,097,152 sectors fetched -> 2,097,152/524,288 = 4 sectors per warp request -> 100% utilization (since every warp is able to read a full acche line of 32*4=128 bytes = 4 sectors)

    But for a stride of 2,
    We have 524,288 warp loads and 4,194,304 sectors fetched -> 4,194,304/524,288 = 8 sectors per warp request -> 50% utilization 


You might notice something weird, we have double the amount of warp load requests for the random reads
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                1,048,576
After a bit of digging, I discovered that with an address pattern like addr + stride, the LD/ST unit can generate all 32 addresses with simple arithmetic,
and the L1 tag stage can handle everything in one go -> 1 request per warp. The tag stage is the part of the L1 cache that checks, for each address,
whether its cache line is present by comparing address tags against the tags stored in the cache sets.
With random indices, the 32 addresses scatter across many cache lines, so the tag stage has to check too many different lines at once and exceeds what it can resolve in a single pass. 
The load gets split into 2 passes, which shows up as 2 requests per warp instead of 1
*/