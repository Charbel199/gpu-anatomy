#include "../common/timer.cuh"
#include "../common/bandwidth.cuh"
#include "../common/check.cuh"
#include "../common/device_info.cuh"

#include <mma.h> // tensor cores api
#include <cuda_fp16.h> // half (fp16)

__global__ void wmma_fp16(const half* __restrict__ a, const half* __restrict__ b, float* c){
    // each fragment is a tile distributed across the warp's registers
    // template params: <use, M, N, K, type, layout>
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag;

    nvcuda::wmma::fill_fragment(c_frag, 0.0f); // zero the accumulator

    // load tiles from HBM (global mem) into fragments
    // (the "16" at the end is the leading dimension (stride in elements between rows))
    nvcuda::wmma::load_matrix_sync(a_frag, a, 16);
    nvcuda::wmma::load_matrix_sync(b_frag, b, 16);

    // A * B + C (tensor core operation) -> output back into c
    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // store the result back to global memory
    nvcuda::wmma::store_matrix_sync(c, c_frag, 16, nvcuda::wmma::mem_row_major);
}


int main() {
    print_device_info();

    size_t N_half_bytes = 16*16 * sizeof(half);
    size_t N_float_bytes = 16*16 * sizeof(float);

    // fill both matrices with 1.0
    // expected result: C[i][j] = sum over k of A[i][k]*B[k][j] = 16 * 1 * 1 = 16
    half h_a[16*16];
    half h_b[16*16];
    float h_c[16*16];
    for (int i = 0; i < 16*16; i++) {
        h_a[i] = __float2half(1.0f);
        h_b[i] = __float2half(1.0f);
    }

    // device memory
    half *d_a, *d_b;
    float *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N_half_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, N_half_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, N_float_bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, N_half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N_half_bytes, cudaMemcpyHostToDevice));

    printf("\nRunning wmma_fp16 kernel ...\n");
    wmma_fp16<<<1, 32>>>(d_a, d_b, d_c);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c, d_c, N_float_bytes, cudaMemcpyDeviceToHost));

    printf("Result matrix (expect 16.0 everywhere):\n"); // should be all 16.0
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            printf("%6.1f ", h_c[i*16 + j]);
        }
        printf("\n");
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}


/*

ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_tensor.sum ./bin/09-tensor-cores/wmma_fp16
==PROF== Profiling "wmma_fp16" - 0: 0%....50%....100% - 1 pass
Result matrix (expect 16.0 everywhere):
  16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0 
  16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0 
  16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0 
  16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0 
  16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0 
  16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0 
  16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0 
  16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0 
  16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0 
  16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0 
  16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0 
  16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0 
  16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0 
  16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0 
  16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0 
  16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0   16.0 
==PROF== Disconnected from process 1322765
[1322765] wmma_fp16@127.0.0.1
  wmma_fp16(const __half *, const __half *, float *) (1, 1, 1)x(32, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    -------------------------------------------------------------- ----------- ------------
    Metric Name                                                    Metric Unit Metric Value
    -------------------------------------------------------------- ----------- ------------
    sm__inst_executed_pipe_tensor.sum                                     inst            2
    sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active           %         0.43
    -------------------------------------------------------------- ----------- ------------

First, the result matches exactly what we expected, every cell is 16.0. Second, we can see that our kernel issues 2 tensor core instructions.
To understand why, let's look at the SASS code:

cuobjdump --dump-sass bin/09-tensor-cores/wmma_fp16 > wmma_fp16.sass


  arch = sm_120                                                                                                                                                                         
  code version = [1,8]                                                                                                                                                                  
  host = linux                                                                                                                                                                          
  compile_size = 64bit                                                                                                                                                                  
  identifier = 09-tensor-cores/wmma_fp16.cu                                                                                                                                             
                                                                                                                                                                                        
      code for sm_120                                                                                                                                                                   
      .target    sm_120                                                                                                                                                                 
                                                                                                                                                                                        
          Function : _Z9wmma_fp16PK6__halfS1_Pf                                                                                                                                         
      .headerflags    @"EF_CUDA_SM120 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM120)"                                                                                                                
                             LDC R1, c[0x0][0x37c] ;
                             S2R R5, SR_LANEID ;
                             LDCU.128 UR8, c[0x0][0x380] ;
                             MOV R3, RZ ;
                             LDCU.64 UR4, c[0x0][0x358] ;
                             LDCU.64 UR6, c[0x0][0x390] ;
                             LOP3.LUT R2, R5, 0x3, RZ, 0xc0, !PT ;
                             SHF.R.U32.HI R5, RZ, 0x2, R5 ;
                             IMAD.WIDE.U32 R2, R5, 0x8, R2 ;
                             LEA R4, P0, R2.reuse, UR8, 0x2 ;
                             LEA R6, P1, R2.reuse, UR10, 0x2 ;
                             LEA.HI.X R5, R2.reuse, UR9, R3.reuse, 0x2, P0 ;
                             LEA.HI.X R7, R2, UR11, R3, 0x2, P1 ;
                             LDG.E R8, desc[UR4][R4.64] ;
                             LDG.E R9, desc[UR4][R4.64+0x100] ;
                             LDG.E R10, desc[UR4][R4.64+0x10] ;
                             LDG.E R11, desc[UR4][R4.64+0x110] ;
                             LDG.E R12, desc[UR4][R6.64] ;
                             LDG.E R13, desc[UR4][R6.64+0x10] ;
                             LDG.E R14, desc[UR4][R6.64+0x100] ;
                             LDG.E R15, desc[UR4][R6.64+0x110] ;
                             >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                             HMMA.16816.F32 R16, R8, R12, RZ ;
                             HMMA.16816.F32 R12, R8, R14, RZ ;
                             <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                             LEA R8, P0, R2, UR6, 0x3 ;
                             LEA.HI.X R9, R2, UR7, R3, 0x3, P0 ;
                             STG.E.64 desc[UR4][R8.64], R16 ;
                             STG.E.64 desc[UR4][R8.64+0x200], R18 ;
                             STG.E.64 desc[UR4][R8.64+0x20], R12 ;
                             STG.E.64 desc[UR4][R8.64+0x220], R14 ;
                             EXIT ;
                             BRA 0x1e0;
                             NOP;
                             NOP;
                             NOP;
                             NOP;
                             NOP;
                             NOP;
                             NOP;
                             NOP;
                             NOP;
                                                                                                                                                                
          ..........                                                                                                                                                                  

Now it's clear why there are 2 tensor core instructions, we see 2 HMMA.16816 instructions in the SASS. The .16816 suffix tells us the native
hardware tile size is M=16, N=8, K=16. So on Blackwell, the native tensor core tile is 16x8x16, but our WMMA call requested 16x16x16.
The compiler splits the work along the 'N' dimension into two 16x8 halves, which translates to 2 HMMA calls.
*/