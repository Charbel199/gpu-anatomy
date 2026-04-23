#include "../common/timer.cuh"
#include "../common/bandwidth.cuh"
#include "../common/check.cuh"
#include "../common/device_info.cuh"


#define BLOCK_SIZE 256

__global__ void broadcast(){
    int idx = threadIdx.x + blockDim.x * blockIdx.x; // general idx
    int idx_in_warp = threadIdx.x % 32; // thread warp idx
    
    int my_value = idx_in_warp;
    int received = __shfl_sync(0xFFFFFFFF, my_value, 2); // all threads in a warp communicate (0xFFFFFFFF) to populate each one's my_value, with the my_value from thread 2 in this warp

    printf("idx=%d, idx_in_warp=%d, received=%d\n", idx, idx_in_warp, received);
}


int main(){
    print_device_info();
    printf("\nRunning broadcast kernel ...");
    broadcast<<<1, BLOCK_SIZE>>>();
    cudaDeviceSynchronize();
}


/*
This is the signature of __shfl_sync
T __shfl_sync(unsigned mask, T value, int src_lane, int width = 32); 

- mask: a 32-bit bitmask that represents which lanes are participating in this shuffle
- value: what the calling lane has in its own register, every lane passes its own value
- src_lane: lane idx whose value we want to read
- width (optional): splits the warp into sub-groups of size `width`, for a full warp shuffle use the default 32 

*/