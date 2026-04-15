#pragma once
#include <stdio.h>

inline void print_bandwidth(const char* label, size_t bytes, float ms) {
    double gb = (double)bytes / (1024.0 * 1024.0 * 1024.0);
    double seconds = (double)ms / 1000.0;
    double gbps = gb / seconds;
    printf("%-40s  %8.3f ms  %8.2f GB/s  (%zu bytes)\n", label, ms, gbps, bytes);
}

inline void print_speedup(const char* baseline, float baseline_ms,
                           const char* optimized, float optimized_ms) {
    printf("\n--- Comparison ---\n");
    printf("%-30s  %8.3f ms\n", baseline, baseline_ms);
    printf("%-30s  %8.3f ms\n", optimized, optimized_ms);
    printf("%-30s  %8.2fx\n", "Speedup", baseline_ms / optimized_ms);
}
