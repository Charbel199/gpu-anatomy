#include "../common/timer.cuh"
#include "../common/bandwidth.cuh"
#include "../common/check.cuh"
#include "../common/device_info.cuh"


#define BLOCK_SIZE 256
#define N ((1 << 24)+7)  // 16M (2^24) elements

// Baseline: every thread does the SAME amount of work (32 iterations of the loop).
// All 32 lanes of every warp march in lockstep -> no divergence.
__global__ void no_divergence(const float* __restrict__ data, float* __restrict__ out, int n){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= n) return;

    float v = data[idx];
    #pragma unroll 1
    for (int i = 0; i < 32; i++) {
        v = v * 1.001f + 0.001f;
    }
    out[idx] = v;
}

// Full divergence: each thread in the warp does a DIFFERENT number of iterations.
// Thread 0 loops once, thread 1 loops twice, ..., thread 31 loops 32 times.
// After thread 0's iteration finishes, lane 0 is masked off while the others continue.
// After thread 1's 2nd iteration, lane 1 is masked off, etc. The warp serializes through
// the loop tail - at peak divergence, only 1 lane is active per instruction.
__global__ void full_divergence(const float* __restrict__ data, float* __restrict__ out, int n){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= n) return;

    float v = data[idx];
    int iters = (threadIdx.x % 32) + 1;  // 1..32, different per lane
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        v = v * 1.001f + 0.001f;
    }
    out[idx] = v;
}

int main(){
    print_device_info();

    size_t N_float_bytes = N * sizeof(float);

    float *d_data, *d_out;
    CUDA_CHECK(cudaMalloc(&d_data, N_float_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, N_float_bytes));
    CUDA_CHECK(cudaMemset(d_data, 0, N_float_bytes));
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    bool ncu = false;

    if (!ncu){
        float ms_no_divergence = benchmark([&]() {
            no_divergence<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);
        });
        float ms_full_divergence = benchmark([&]() {
            full_divergence<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);
        });
        print_speedup("No Divergence", ms_no_divergence, "Full Divergence", ms_full_divergence);
    } else {
        no_divergence<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);
        full_divergence<<<grid, BLOCK_SIZE>>>(d_data, d_out, N);
    }

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_out));
}


/*
We run
ncu --metrics smsp__thread_inst_executed_per_inst_executed.ratio,smsp__sass_average_branch_targets_threads_uniform.pct ./bin/04-warp-behavior/full_divergence

And get the following results:

[1315949] full_divergence@127.0.0.1
  no_divergence(const float *, float *, int) (65537, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    ----------------------------------------------------- ----------- ------------
    Metric Name                                           Metric Unit Metric Value
    ----------------------------------------------------- ----------- ------------
    smsp__sass_average_branch_targets_threads_uniform.pct           %          100
    smsp__thread_inst_executed_per_inst_executed.ratio                       32.00
    ----------------------------------------------------- ----------- ------------

  full_divergence(const float *, float *, int) (65537, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    ----------------------------------------------------- ----------- ------------
    Metric Name                                           Metric Unit Metric Value
    ----------------------------------------------------- ----------- ------------
    smsp__sass_average_branch_targets_threads_uniform.pct           %         3.13
    smsp__thread_inst_executed_per_inst_executed.ratio                       15.47
    ----------------------------------------------------- ----------- ------------

And for wall-clock timing we get:

--- Comparison ---
No Divergence                      0.054 ms
Full Divergence                    0.074 ms
Speedup                             0.73x

Both kernels run a loop of v = v * const + const; the only difference is how many
iterations each thread does:

- no_divergence: every thread loops exactly 32 times. All 32 lanes of the warp march
  in lockstep -> 32.00 threads per instruction, 100% branch uniformity.

- full_divergence: thread i loops (i+1) times. After iteration 1, lane 0 drops out.
  After iteration 2, lane 1 drops out. Etc. By iteration 32, only lane 31 is still
  running. Active lanes per iteration: 32, 31, 30, ..., 1.
  Average = (32+31+...+1)/32 = 16.5, which matches the measured 15.47 closely
  (small gap from overhead instructions outside the loop).
  Branch uniformity drops to ~3% because the loop-exit branch splits the warp on
  almost every iteration.

Wall-clock cost: 37% slower (0.73x speedup). It's not 2x slower (despite the average
active lanes dropping from 32 to ~15) because the warp still issues the same number
of instructions, the loop runs until the LAST thread finishes, 32 iterations at the
warp level either way. The divergence just wastes lanes inside each of those
iterations.

Progression across this folder so far:

Kernel                                            threads/inst   branch uniform
----------------------------------------------    ------------   --------------
no_divergence (simple copy)                              32.00               0%  (no branches)
no_divergence (with path_a call)                         32.00             100%
half_divergence (real branch via __noinline__)           20.63              80%
full_divergence (variable loop count)                    15.47            3.13%
*/
