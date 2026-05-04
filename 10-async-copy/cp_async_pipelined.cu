#include "../common/timer.cuh"
#include "../common/bandwidth.cuh"
#include "../common/check.cuh"
#include "../common/device_info.cuh"

#include <cuda/pipeline>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

#define BLOCK_SIZE 128
#define TILES_PER_BLOCK 8192    // many tiles -> data exceeds L2, real DRAM latency
#define ACC_SIZE 32             // per-thread accumulator array (creates register pressure)
#define INNER_ITERS 1           // number of times heavy_compute repeats its inner loop
#define STAGES 2                // double buffering

// heavy compute that forces all ACC_SIZE accumulators to stay live in registers.
// the compiler can't reuse registers across iterations. This pushes register count up naturally.
__device__ __forceinline__ void heavy_compute(float v, float* acc) {
    #pragma unroll
    for (int iter = 0; iter < INNER_ITERS; iter++) {
        #pragma unroll
        for (int i = 0; i < ACC_SIZE; i++) {
            acc[i] = acc[i] * 1.0001f + v * 0.001f + acc[(i + 7) % ACC_SIZE] * 0.0001f;
        }
    }
}

__global__ void sync_kernel(const float* __restrict__ data, float* __restrict__ out) {
    __shared__ float tile[BLOCK_SIZE];
    int tid = threadIdx.x;
    int block_start = blockIdx.x * TILES_PER_BLOCK * BLOCK_SIZE;

    float acc[ACC_SIZE];
    #pragma unroll
    for (int i = 0; i < ACC_SIZE; i++) acc[i] = (float)i * 0.01f;

    for (int t = 0; t < TILES_PER_BLOCK; t++) {
        tile[tid] = data[block_start + t * BLOCK_SIZE + tid]; // sync load
        __syncthreads();

        heavy_compute(tile[tid], acc);

        __syncthreads();
    }

    // write accumulators out so the compiler doesn't optimize them out
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < ACC_SIZE; i++) sum += acc[i];
    out[blockIdx.x * BLOCK_SIZE + tid] = sum;
}

__global__ void cp_async_kernel(const float* __restrict__ data, float* __restrict__ out) {
    __shared__ float tile[STAGES][BLOCK_SIZE]; // double buffered shared mem
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, STAGES> pipeline_state;

    int tid = threadIdx.x;
    int block_start = blockIdx.x * TILES_PER_BLOCK * BLOCK_SIZE;

    auto block = cg::this_thread_block();
    auto pipe  = cuda::make_pipeline(block, &pipeline_state);

    float acc[ACC_SIZE];
    #pragma unroll
    for (int i = 0; i < ACC_SIZE; i++) acc[i] = (float)i * 0.01f;

    // launch the first tile's load before the loop, otherwise the loop
    // would be waiting on a load that was never issued.
    pipe.producer_acquire();
    cuda::memcpy_async(block, tile[0], data + block_start,
                       sizeof(float) * BLOCK_SIZE, pipe);
    pipe.producer_commit();

    for (int t = 0; t < TILES_PER_BLOCK; t++) {
        // issue NEXT tile's load if any.
        // cp.async returns immediately, the copy runs in hardware while we go on
        // to wait + compute on the current tile. That's where the overlap happens.
        if (t + 1 < TILES_PER_BLOCK) {
            pipe.producer_acquire();
            cuda::memcpy_async(block, tile[(t + 1) % STAGES],
                               data + block_start + (t + 1) * BLOCK_SIZE,
                               sizeof(float) * BLOCK_SIZE, pipe);
            pipe.producer_commit();
        }

        // wait for the CURRENT tile's load to actually finish.
        pipe.consumer_wait();
        __syncthreads();

        heavy_compute(tile[t % STAGES][tid], acc);

        // release the buffer so the producer can reuse it for tile t+2.
        pipe.consumer_release();
    }

    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < ACC_SIZE; i++) sum += acc[i];
    out[blockIdx.x * BLOCK_SIZE + tid] = sum;
}

int main() {
    print_device_info();


    size_t total_in_elements = (size_t)1 << 28;  // 256M floats = 1 GB
    size_t elements_per_block = (size_t)TILES_PER_BLOCK * BLOCK_SIZE;
    int num_blocks = (int)((total_in_elements + elements_per_block - 1) / elements_per_block);

    total_in_elements = (size_t)num_blocks * elements_per_block;  // round up to block multiple
    size_t total_out_elements = (size_t)num_blocks * BLOCK_SIZE;
    size_t in_bytes  = total_in_elements * sizeof(float);
    size_t out_bytes = total_out_elements * sizeof(float);

    printf("Input: %.1f MB, Output: %.3f MB\n", in_bytes / 1e6, out_bytes / 1e6);

    float *d_data, *d_out;
    CUDA_CHECK(cudaMalloc(&d_data, in_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK(cudaMemset(d_data, 0, in_bytes));

    bool ncu = false;

    if (!ncu){
        printf("\nRunning sync_kernel ...");
        float ms_sync = benchmark([&]() {
            sync_kernel<<<num_blocks, BLOCK_SIZE>>>(d_data, d_out);
        });

            
        float ms_pipe = benchmark([&]() {
            cp_async_kernel<<<num_blocks, BLOCK_SIZE>>>(d_data, d_out);
        });

        print_speedup("Sync (no overlap)", ms_sync, "Pipelined (cp.async)", ms_pipe);
    } else {
        printf("\nRunning sync_kernel ...");
        sync_kernel<<<num_blocks, BLOCK_SIZE>>>(d_data, d_out);
        printf("\nRunning cp_async_kernel ...");
        cp_async_kernel<<<num_blocks, BLOCK_SIZE>>>(d_data, d_out);
    }


    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_out));
}


/*
Both kernels have identical
    - launch params (188 blocks, 128 threads each)
    - per-tile work (load, heavy_compute, store accumulators)
    - total work and total memory traffic
    - per-element compute (cross-referenced acc updates)

The ONLY difference is how the load is sequenced relative to compute:
    - sync_kernel:     load -> wait -> compute -> store, sequentially
    - cp_async_kernel: load t+1 IN PARALLEL with compute t, double buffered

Result:
    Input: 1073.7 MB, Output: 0.131 MB

    Running sync_kernel ...
    Running cp_async_kernel ...
    --- Comparison ---
    Sync (no overlap)                  3.441 ms
    Pipelined (cp.async)               2.263 ms
    Speedup                             1.52x


How does the pipeline know which tile to wait on? It maintains a FIFO queue of
submitted loads. Each producer_commit() pushes to the back, each consumer_wait()
pops from the front. Trace:

  t=0 prologue: producer_commit()  -> queue: [tile_0]

  t=0:
    producer_commit()              -> queue: [tile_0, tile_1]
    consumer_wait()                -> pops tile_0, blocks until done
    compute on tile[0]
    consumer_release()             -> tile_0's buffer is free

  t=1:
    producer_commit()              -> queue: [tile_1, tile_2]
                                         (tile_0 already popped)
    consumer_wait()                -> pops tile_1, blocks until done
    compute on tile[1 % 2]
    consumer_release()             -> tile_1's buffer is free

  t=2:
    producer_commit()              -> queue: [tile_2, tile_3]
    consumer_wait()                -> pops tile_2
    ...

That's why the prologue is needed, it makes commit #0 happen before wait #0,
so the FIFO pairs them up correctly.

Three conditions had to line up to actually see a speedup:

  1. LOW OCCUPANCY (11%, organically from algorithmic register pressure)
     Each thread holds 32 accumulators alive in registers via cross-references.
     The compiler can't reuse register slots, so per-thread register count goes
     up to 40, which limits how many warps fit per SM. Without this, the hardware
     scheduler hides memory latency by warp switching and pipelining adds zero value.

  2. WORKING SET EXCEEDS L2 (1073.7 MB >> 128 MB L2)
     Forces real DRAM latency (~400 cycles) instead of L2 latency (~200 cycles).

  3. BALANCED COMPUTE/MEMORY RATIO
     Compute time per tile roughly equal to load time. Too little compute -> memory-bound,
     can't reduce DRAM traffic. Too much -> compute-bound, memory wasn't the limit.
     i tried INNER_ITERS = 16 first and got 1.16x because the kernel was compute-bound.

*/
