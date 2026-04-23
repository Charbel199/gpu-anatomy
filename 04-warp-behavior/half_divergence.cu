#include "../common/timer.cuh"
#include "../common/bandwidth.cuh"
#include "../common/check.cuh"
#include "../common/device_info.cuh"


#define BLOCK_SIZE 256
#define N (1 << 24)  // 16M (2^24) elements

// both paths do the same operation (x * const + const = one FFMA), just with different
// constants so the compiler can't merge them. Same cost per call
__device__ __noinline__ float path_a(float x) { return x * 2.0f + 1.0f; }
__device__ __noinline__ float path_b(float x) { return x * 3.0f + 2.0f; }

__global__ void no_divergence(const float* __restrict__ data, float* __restrict__ out, int n){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx<n) out[idx] = path_a(data[idx]);
}

__global__ void half_divergence(const float* __restrict__ data, float* __restrict__ out, int n){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx<n) {
        if (idx%2) out[idx] = path_a(data[idx]);
        else       out[idx] = path_b(data[idx]);
    }
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


    bool ncu = true;

    if (!ncu){
        float ms_no_divergence = benchmark([&]() {
            no_divergence<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);
        });
        float ms_half_divergence = benchmark([&]() {
            half_divergence<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);
        });
        print_speedup("No Divergence", ms_no_divergence, "Half Divergence", ms_half_divergence);
    } else {
        no_divergence<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);
        half_divergence<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);
    }

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_out));

}


/*
We run
ncu --metrics smsp__thread_inst_executed_per_inst_executed.ratio,smsp__sass_average_branch_targets_threads_uniform.pct ./bin/04-warp-behavior/half_divergence

And get the following results:

[1285211] half_divergence@127.0.0.1
  no_divergence(const float *, float *, int) (65537, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    ----------------------------------------------------- ----------- ------------
    Metric Name                                           Metric Unit Metric Value
    ----------------------------------------------------- ----------- ------------
    smsp__sass_average_branch_targets_threads_uniform.pct           %          100
    smsp__thread_inst_executed_per_inst_executed.ratio                       32.00
    ----------------------------------------------------- ----------- ------------

  half_divergence(const float *, float *, int) (65537, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    ----------------------------------------------------- ----------- ------------
    Metric Name                                           Metric Unit Metric Value
    ----------------------------------------------------- ----------- ------------
    smsp__sass_average_branch_targets_threads_uniform.pct           %           80
    smsp__thread_inst_executed_per_inst_executed.ratio                       20.63
    ----------------------------------------------------- ----------- ------------

And for wall-clock timing we get:

    --- Comparison ---
    No Divergence                      0.046 ms
    Half Divergence                    0.051 ms
    Speedup                             0.90x

Both kernels now do the same per-thread work (one FFMA via a __noinline__ function call),
so any difference between them is isolated to divergence behavior.

- no_divergence: all 32 lanes call path_a together. Every executed branch (the CALL to path_a
  and the RET from it) is uniform -> 100% branch uniformity and 32.00 threads per instruction.

- half_divergence: odd lanes call path_a, even lanes call path_b. The idx%2 branch splits the
  warp exactly in half, while the CALL/RET branches around it stay uniform. The metric averages
  these together: 1 non-uniform branch out of ~5 total branch events per warp -> 80% uniform.
  The thread-active ratio drops to ~20.63 because many instructions (inside path_a and path_b)
  only have half the warp active.

The wall-clock cost of divergence here is ~11% (0.90x speedup). Small because this kernel is
memory-bound (just a load + store), so the divergence only really penalizes the tiny compute
section (still significant). In a compute-bound kernel the cost would be much larger.
*/