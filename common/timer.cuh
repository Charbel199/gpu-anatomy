#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <stdio.h>

struct GpuTimer {
    cudaEvent_t start_event, stop_event;
    cudaStream_t stream;

    GpuTimer(cudaStream_t s = 0) : stream(s) {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }

    ~GpuTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start() {
        cudaEventRecord(start_event, stream);
    }

    void stop() {
        cudaEventRecord(stop_event, stream);
        cudaEventSynchronize(stop_event);
    }

    float ms() {
        float elapsed = 0.0f;
        cudaEventElapsedTime(&elapsed, start_event, stop_event);
        return elapsed;
    }
};

// run a kernel N times and return the median time in ms
template <typename F>
float benchmark(F kernel_launch, int iterations = 100, int warmup = 10) {
    // warmup: fill caches, stabilize clocks
    for (int i = 0; i < warmup; i++) {
        kernel_launch();
    }
    cudaDeviceSynchronize();

    // collect times, max 1024 iterations for simplicity
    float times[1024];
    int n = (iterations < 1024) ? iterations : 1024;

    GpuTimer timer;
    for (int i = 0; i < n; i++) {
        timer.start();
        kernel_launch();
        timer.stop();
        times[i] = timer.ms();
    }

    std::sort(times, times + n);
    return times[n / 2];  // median
}
