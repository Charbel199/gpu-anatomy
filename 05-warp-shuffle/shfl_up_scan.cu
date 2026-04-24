#include "../common/timer.cuh"
#include "../common/bandwidth.cuh"
#include "../common/check.cuh"
#include "../common/device_info.cuh"


#define BLOCK_SIZE 256

__device__ __forceinline__ float warp_scan(float val, int lane) {
    float n;
    n = __shfl_up_sync(0xFFFFFFFF, val, 1); if (lane >= 1) val+=n;
    n = __shfl_up_sync(0xFFFFFFFF, val, 2); if (lane >= 2) val+=n;
    n = __shfl_up_sync(0xFFFFFFFF, val, 4); if (lane >= 4) val+=n;
    n = __shfl_up_sync(0xFFFFFFFF, val, 8); if (lane >= 8) val+=n;
    n = __shfl_up_sync(0xFFFFFFFF, val, 16); if (lane >= 16) val+=n;
    return val;
}

__global__ void shfl_up_scan(){
    int idx = threadIdx.x + blockDim.x * blockIdx.x; // general idx
    int idx_in_warp = threadIdx.x % 32; // thread warp idx
    
    int my_value = idx_in_warp;
    int received = warp_scan(my_value, idx_in_warp); // we know that there is no warp divergence (no conditional branches) so this will work as expected
    
    printf("idx=%d, idx_in_warp=%d, received=%d\n", idx, idx_in_warp, received);
}


int main(){
    print_device_info();
    printf("\nRunning shfl_up_scan kernel ...");
    shfl_up_scan<<<1, BLOCK_SIZE>>>();
    cudaDeviceSynchronize();
}


/*
This is the signature of __shfl_up_sync
T __shfl_up_sync(unsigned mask, T value, unsigned delta, int width = 32);

- mask: a 32-bit bitmask that represents which lanes are participating in this shuffle
- value: what the calling lane has in its own register, every lane passes its own value
- delta: offset up the warp - each lane reads from (my_lane - delta).
        Lanes near the bottom where (my_lane - delta) < 0 fall out of range and
        just receive their own value back.
- width (optional): splits the warp into sub-groups of size `width`, for a full warp shuffle use the default 32

This code is as simple as the shfl_down_reduce code,

n = __shfl_up_sync(0xFFFFFFFF, val, 1); if (lane >= 1) val+=n;
n = __shfl_up_sync(0xFFFFFFFF, val, 2); if (lane >= 2) val+=n;
n = __shfl_up_sync(0xFFFFFFFF, val, 4); if (lane >= 4) val+=n;
n = __shfl_up_sync(0xFFFFFFFF, val, 8); if (lane >= 8) val+=n;
n = __shfl_up_sync(0xFFFFFFFF, val, 16); if (lane >= 16) val+=n;


Starting with n = __shfl_up_sync(0xFFFFFFFF, val, 1); if (lane >= 1) val+=n;

Every value register first looks at the register to its left.
(lane 0 does not have a left neighbor so it's skipped because of `if (lane >= 1)`)

_These are thread indices in a warp_
0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
+  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +
   0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30

n = __shfl_up_sync(0xFFFFFFFF, val, 2); if (lane >= 2) val+=n;
0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
+  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +
      0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29

n = __shfl_up_sync(0xFFFFFFFF, val, 4); if (lane >= 4) val+=n;
0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
+  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +
            0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27

n = __shfl_up_sync(0xFFFFFFFF, val, 8); if (lane >= 8) val+=n;
0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
+  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +
                        0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23

n = __shfl_up_sync(0xFFFFFFFF, val, 16); if (lane >= 16) val+=n;
0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
+  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +
                                                0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15

At the end of all of this process, we would have partial gradual sum in each thread's register

v[0] = 0;
v[1] = v[0] + 1;
v[2] = v[1] + 2;
...
v[31] = v[30] + 31;

We can take a look at the output:

idx=96, idx_in_warp=0, received=0
idx=97, idx_in_warp=1, received=1
idx=98, idx_in_warp=2, received=3
idx=99, idx_in_warp=3, received=6
idx=100, idx_in_warp=4, received=10
idx=101, idx_in_warp=5, received=15
...
idx=123, idx_in_warp=27, received=378
idx=124, idx_in_warp=28, received=406
idx=125, idx_in_warp=29, received=435
idx=126, idx_in_warp=30, received=465
idx=127, idx_in_warp=31, received=496

At idx_in_warp=31, the value is 496, (31 * (31+1) / 2) = 496
At idx_in_warp=5, the value is 15, 0+1+2+3+4+5 = 15
*/