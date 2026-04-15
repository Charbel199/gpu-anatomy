#pragma once
#include <cuda_runtime.h>
#include <nvml.h>
#include <stdio.h>

inline void print_device_info(int device = 0) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // get clock rates via NVML
    nvmlInit();
    nvmlDevice_t nvml_dev;
    nvmlDeviceGetHandleByIndex(device, &nvml_dev);

    unsigned int clock_sm = 0, clock_mem = 0;
    nvmlDeviceGetMaxClockInfo(nvml_dev, NVML_CLOCK_SM, &clock_sm);
    nvmlDeviceGetMaxClockInfo(nvml_dev, NVML_CLOCK_MEM, &clock_mem);

    printf("=== Device: %s ===\n", prop.name);
    printf("SM count:              %d\n", prop.multiProcessorCount);
    printf("SM version:            %d.%d\n", prop.major, prop.minor);
    printf("SM clock:              %u MHz\n", clock_sm);
    printf("Memory clock:          %u MHz\n", clock_mem);
    printf("Memory bus width:      %d bit\n", prop.memoryBusWidth);
    printf("L2 cache size:         %d KB\n", prop.l2CacheSize / 1024);
    printf("Shared mem per SM:     %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("Shared mem per block:  %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Registers per SM:      %d\n", prop.regsPerMultiprocessor);
    printf("Registers per block:   %d\n", prop.regsPerBlock);
    printf("Max threads per SM:    %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Warp size:             %d\n", prop.warpSize);
    printf("Global memory:         %zu MB\n", prop.totalGlobalMem / (1024 * 1024));

    // 2.0 = DDR (double data rate, transfers on both clock edges)
    // memoryBusWidth is in bits, / 8 to get bytes
    // / 1e9 to get GB/s
    double bw = 2.0 * clock_mem * 1e6 * (prop.memoryBusWidth / 8.0) / 1e9;
    printf("Theoretical bandwidth: %.1f GB/s\n", bw);
    printf("===\n\n");

    nvmlShutdown();
}
