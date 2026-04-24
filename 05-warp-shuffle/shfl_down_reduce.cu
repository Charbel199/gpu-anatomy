#include "../common/timer.cuh"
#include "../common/bandwidth.cuh"
#include "../common/check.cuh"
#include "../common/device_info.cuh"


#define BLOCK_SIZE 256

__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);
    return val;
}

__global__ void shfl_down_reduce(){
    int idx = threadIdx.x + blockDim.x * blockIdx.x; // general idx
    int idx_in_warp = threadIdx.x % 32; // thread warp idx
    
    int my_value = idx_in_warp;
    int received = warp_reduce_sum(my_value); // we know that there is no warp divergence (no conditional branches) so this will work as expected
    
    printf("idx=%d, idx_in_warp=%d, received=%d\n", idx, idx_in_warp, received);
}


int main(){
    print_device_info();
    printf("\nRunning shfl_down_reduce kernel ...");
    shfl_down_reduce<<<1, BLOCK_SIZE>>>();
    cudaDeviceSynchronize();
}


/*
This is the signature of __shfl_down_sync
T __shfl_down_sync(unsigned mask, T value, unsigned delta, int width = 32);

- mask: a 32-bit bitmask that represents which lanes are participating in this shuffle
- value: what the calling lane has in its own register, every lane passes its own value
- delta: offset down the warp - each lane reads from (my_lane + delta).
        Lanes near the top where (my_lane + delta) >= width fall out of range and
        just receive their own value back.
- width (optional): splits the warp into sub-groups of size `width`, for a full warp shuffle use the default 32

Looking at what we're doing, it's actually pretty simple

    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);

In the first iteration, every value from thread warp idx 0 -> 15, adds the respective values from 16 -> 31

_These are thread indices in a warp_
val += __shfl_down_sync(0xFFFFFFFF, val, 16);
0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
+  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +
16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
                                                (out of range, they get their own value back, result is junk)

0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 ...
+  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  ...
8  9  10 11 12 13 14 15 ...

val += __shfl_down_sync(0xFFFFFFFF, val, 4);
0  1  2  3  4  5  6  7  ...
+  +  +  +  +  +  +  +  ...
4  5  6  7 ...


val += __shfl_down_sync(0xFFFFFFFF, val, 2);
0  1  2  3 ...
+  +  +  + ...
2  3 ...


val += __shfl_down_sync(0xFFFFFFFF, val, 1);
0  1 ...
+  + ...
1 ...

At the end of all of this process, we would have accumulated the final sum of all registers in the register at thread 0.
We can take a look at the output

...
idx=158, idx_in_warp=30, received=976
idx=159, idx_in_warp=31, received=992
idx=160, idx_in_warp=0, received=496
idx=161, idx_in_warp=1, received=512
idx=162, idx_in_warp=2, received=528
idx=163, idx_in_warp=3, received=544
idx=164, idx_in_warp=4, received=560
idx=165, idx_in_warp=5, received=576
idx=166, idx_in_warp=6, received=592
idx=167, idx_in_warp=7, received=608
idx=168, idx_in_warp=8, received=624
idx=169, idx_in_warp=9, received=640
idx=170, idx_in_warp=10, received=656
idx=171, idx_in_warp=11, received=672
idx=172, idx_in_warp=12, received=688
idx=173, idx_in_warp=13, received=704
idx=174, idx_in_warp=14, received=720
idx=175, idx_in_warp=15, received=736
idx=176, idx_in_warp=16, received=752
idx=177, idx_in_warp=17, received=768
idx=178, idx_in_warp=18, received=784
idx=179, idx_in_warp=19, received=800
idx=180, idx_in_warp=20, received=816
idx=181, idx_in_warp=21, received=832
idx=182, idx_in_warp=22, received=848
idx=183, idx_in_warp=23, received=864
idx=184, idx_in_warp=24, received=880
idx=185, idx_in_warp=25, received=896
idx=186, idx_in_warp=26, received=912
idx=187, idx_in_warp=27, received=928
idx=188, idx_in_warp=28, received=944
idx=189, idx_in_warp=29, received=960
idx=190, idx_in_warp=30, received=976
idx=191, idx_in_warp=31, received=992
idx=192, idx_in_warp=0, received=496
idx=193, idx_in_warp=1, received=512
idx=194, idx_in_warp=2, received=528
idx=195, idx_in_warp=3, received=544
...

We get received=496 at idx_in_warp=0, to confirm, every thread starts with
int my_value = idx_in_warp; the register has a value = to its thread index in its warp
So we should get at the end a sum of numbers from 0 -> 31
sum = (31 * (31+1)) / 2 = 496

*/