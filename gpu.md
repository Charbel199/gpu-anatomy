# NVIDIA GPU — Complete Hardware Reference
MENTION IN README GPU I AM USING AND ITS INTERESTING TO COMPARE WITH YOUR CURRENT G PU
## Full Architecture Diagram

```
                              ┌─────────────────────────────────────────────────────────┐
                              │                        GPU CHIP                          │
                              │                                                         │
                              │   ┌───────────────────────────────────────────────────┐  │
                              │   │              GigaThread Engine                     │  │
                              │   │     Global work distributor — assigns thread       │  │
                              │   │     blocks to GPCs/SMs, manages grid launches      │  │
                              │   └──────────────────────┬────────────────────────────┘  │
                              │                          │                               │
          ┌───────────────────┼──────────────────────────┼───────────────────────────────┼───────────────────────┐
          │                   │                          │                               │                       │
          ▼                   │                          ▼                               │                       ▼
  ┌───────────────┐           │              ┌───────────────────┐                       │           ┌───────────────────┐
  │    GPC 0      │           │              │      GPC 1        │          ...          │           │    GPC N          │
  │               │           │              │                   │                       │           │                   │
  │  ┌─────────┐  │           │              │  ┌─────────┐      │                       │           │  ┌─────────┐      │
  │  │Raster   │  │           │              │  │Raster   │      │                       │           │  │Raster   │      │
  │  │Engine   │  │           │              │  │Engine   │      │                       │           │  │Engine   │      │
  │  └─────────┘  │           │              │  └─────────┘      │                       │           │  └─────────┘      │
  │               │           │              │                   │                       │           │                   │
  │  ┌─────┐┌───┐│           │              │  ┌─────┐┌─────┐  │                       │           │  ┌─────┐┌─────┐  │
  │  │TPC 0││...││           │              │  │TPC 0││TPC N│  │                       │           │  │TPC 0││TPC N│  │
  │  │     ││   ││           │              │  │     ││     │  │                       │           │  │     ││     │  │
  │  │ SM  ││   ││           │              │  │ SM  ││ SM  │  │                       │           │  │ SM  ││ SM  │  │
  │  │ SM  ││   ││           │              │  │ SM  ││ SM  │  │                       │           │  │ SM  ││ SM  │  │
  │  └─────┘└───┘│           │              │  └─────┘└─────┘  │                       │           │  └─────┘└─────┘  │
  └───────────────┘           │              └───────────────────┘                       │           └───────────────────┘
          │                   │                          │                               │                       │
          └───────────────────┼──────────────────────────┼───────────────────────────────┼───────────────────────┘
                              │                          │                               │
                              │                          ▼                               │
                              │   ┌───────────────────────────────────────────────────┐  │
                              │   │                  L2 Cache                          │  │
                              │   │    Shared across all SMs — typically 4-96 MB       │  │
                              │   │    Contains: L2 atomic units, cache partitions     │  │
                              │   └──────────────────────┬────────────────────────────┘  │
                              │                          │                               │
                              │   ┌──────────────────────▼────────────────────────────┐  │
                              │   │             Memory Controllers                     │  │
                              │   │    Multiple channels to HBM/GDDR                   │  │
                              │   │    Handle: scheduling, refresh, ECC                │  │
                              │   └──────────────────────┬────────────────────────────┘  │
                              │                          │                               │
                              │   ┌──────────────────────▼────────────────────────────┐  │
                              │   │              HBM / GDDR VRAM                       │  │
                              │   │    The big slow memory — 8GB to 192GB              │  │
                              │   │    Bandwidth: 400 GB/s (GDDR) to 3+ TB/s (HBM3e)  │  │
                              │   └───────────────────────────────────────────────────┘  │
                              │                                                         │
                              │   ┌───────────────────────────────────────────────────┐  │
                              │   │              Fixed Function Engines                 │  │
                              │   │                                                     │  │
                              │   │  ┌──────────┐ ┌──────────┐ ┌──────────────────┐    │  │
                              │   │  │ Copy     │ │ NVENC    │ │ NVDEC            │    │  │
                              │   │  │ Engines  │ │ (video   │ │ (video decode)   │    │  │
                              │   │  │ (DMA)    │ │ encode)  │ │                  │    │  │
                              │   │  └──────────┘ └──────────┘ └──────────────────┘    │  │
                              │   │                                                     │  │
                              │   │  ┌──────────┐ ┌──────────┐ ┌──────────────────┐    │  │
                              │   │  │ NVJPG    │ │ NVOF     │ │ Display Engine   │    │  │
                              │   │  │ (JPEG    │ │ (optical │ │ (video output)   │    │  │
                              │   │  │ decode)  │ │ flow)    │ │                  │    │  │
                              │   │  └──────────┘ └──────────┘ └──────────────────┘    │  │
                              │   └───────────────────────────────────────────────────┘  │
                              │                                                         │
                              │   ┌───────────────────────────────────────────────────┐  │
                              │   │              Interconnects                          │  │
                              │   │                                                     │  │
                              │   │  ┌──────────────┐  ┌────────────────────────────┐  │  │
                              │   │  │ PCIe Gen4/5  │  │ NVLink (GPU-to-GPU)       │  │  │
                              │   │  │ (to host CPU)│  │ 600-900 GB/s per GPU      │  │  │
                              │   │  │ 32-64 GB/s   │  │ NVSwitch for >2 GPU       │  │  │
                              │   │  └──────────────┘  └────────────────────────────┘  │  │
                              │   └───────────────────────────────────────────────────┘  │
                              │                                                         │
                              └─────────────────────────────────────────────────────────┘
```

---

## SM (Streaming Multiprocessor) — Detailed Internals

```
┌─────────────────────────────────── SM ──────────────────────────────────────┐
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                      Instruction Cache                                 │  │
│  │           Caches decoded instructions for active warps                 │  │
│  └────────────────────────────┬───────────────────────────────────────────┘  │
│                               │                                             │
│  ┌────────────────────────────▼───────────────────────────────────────────┐  │
│  │                      Warp Schedulers (4 per SM)                        │  │
│  │                                                                        │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐              │  │
│  │  │Scheduler │  │Scheduler │  │Scheduler │  │Scheduler │              │  │
│  │  │   0      │  │   1      │  │   2      │  │   3      │              │  │
│  │  │          │  │          │  │          │  │          │              │  │
│  │  │Dispatch  │  │Dispatch  │  │Dispatch  │  │Dispatch  │              │  │
│  │  │Unit      │  │Unit      │  │Unit      │  │Unit      │              │  │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘              │  │
│  │       │              │              │              │                    │  │
│  │  Each scheduler manages a pool of warps and picks                     │  │
│  │  one ready warp per cycle to issue an instruction                     │  │
│  │  (Scoreboard tracks register dependencies)                            │  │
│  └───────┼──────────────┼──────────────┼──────────────┼──────────────────┘  │
│          │              │              │              │                      │
│  ┌───────▼──────────────▼──────────────▼──────────────▼──────────────────┐  │
│  │                     Register File                                     │  │
│  │              65,536 x 32-bit registers per SM                         │  │
│  │     Fastest storage — 0 cycle latency (pipelined), ~8 TB/s           │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Cross-Lane Data Path (Warp Shuffle)                            │  │  │
│  │  │  __shfl_sync, __shfl_up_sync, __shfl_down_sync, __shfl_xor_sync│  │  │
│  │  │  Register-to-register across 32 lanes, no memory touched       │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────┬──────────────────────────────────────────────────────────────┘  │
│          │                                                                 │
│  ┌───────▼──────────────────────────────────────────────────────────────┐  │
│  │                    Execution Units (per SM partition)                  │  │
│  │                                                                       │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │  │
│  │  │  FP32 Cores  │  │  INT32 Cores │  │   FP64 Cores │                │  │
│  │  │  (CUDA cores)│  │              │  │   (double)   │                │  │
│  │  │              │  │  Can dual-   │  │              │                │  │
│  │  │  Main ALU    │  │  issue with  │  │  FP64:FP32   │                │  │
│  │  │  for FADD,   │  │  FP32 on     │  │  ratio varies│                │  │
│  │  │  FMUL, FFMA  │  │  Volta+      │  │  1:2 (data)  │                │  │
│  │  │              │  │              │  │  1:32 (game) │                │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                │  │
│  │                                                                       │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │  │
│  │  │  SFU         │  │ Tensor Cores │  │  RT Cores    │                │  │
│  │  │  (Special    │  │              │  │  (Ray Trace) │                │  │
│  │  │  Function)   │  │  Matrix      │  │              │                │  │
│  │  │              │  │  multiply    │  │  BVH traversal│               │  │
│  │  │  sin, cos,   │  │  D = A*B+C   │  │  ray-box     │                │  │
│  │  │  exp, rsqrt, │  │              │  │  ray-triangle│                │  │
│  │  │  rcp, log2   │  │  FP16,BF16,  │  │  intersection│                │  │
│  │  │              │  │  TF32,FP8,   │  │              │                │  │
│  │  │  Lower       │  │  INT8,INT4   │  │  RTX only    │                │  │
│  │  │  throughput  │  │              │  │              │                │  │
│  │  │  than ALU    │  │  Volta+      │  │  Turing+     │                │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                │  │
│  │                                                                       │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │  │
│  │  │  Load/Store Units (LD/ST)                                        │ │  │
│  │  │  Handle all memory operations: global, shared, local, constant   │ │  │
│  │  │  Coalescing logic lives here                                     │ │  │
│  │  └──────────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│          │                                                                 │
│  ┌───────▼──────────────────────────────────────────────────────────────┐  │
│  │               Shared Memory / L1 Cache (Unified)                      │  │
│  │                                                                       │  │
│  │  Configurable split: e.g., 48KB shared + 80KB L1                     │  │
│  │                     or 128KB shared + 0KB L1 (Hopper)                │  │
│  │                                                                       │  │
│  │  ┌─────────────────────┐  ┌────────────────────────────────┐         │  │
│  │  │  Shared Memory      │  │  L1 Data Cache                 │         │  │
│  │  │                     │  │                                 │         │  │
│  │  │  32 banks           │  │  Caches global memory reads    │         │  │
│  │  │  4 bytes per bank   │  │  Hardware-managed              │         │  │
│  │  │  Programmer managed │  │                                 │         │  │
│  │  │  ~5 cycle latency   │  │  Also serves:                  │         │  │
│  │  │                     │  │   - Constant cache              │         │  │
│  │  │  Has atomic unit    │  │   - Texture cache               │         │  │
│  │  │  (shared atomics)   │  │                                 │         │  │
│  │  └─────────────────────┘  └────────────────────────────────┘         │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Tex / Constant Cache Path                                      │  │  │
│  │  │                                                                   │  │  │
│  │  │  Texture Unit: hardware filtering, interpolation,                │  │  │
│  │  │                2D/3D spatial locality optimization                │  │  │
│  │  │  __ldg(): read-only global load through texture cache path       │  │  │
│  │  │                                                                   │  │  │
│  │  │  Constant Cache: broadcast-optimized for uniform reads           │  │  │
│  │  │                  serialized if threads read different addresses   │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Async Copy Engine / TMA (Tensor Memory Accelerator, Hopper+)   │  │  │
│  │  │  cp.async: global → shared without going through registers       │  │  │
│  │  │  TMA: hardware-accelerated tensor tile copies                    │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Parts List

### 1. Chip-Level Organization

| # | Part | What It Does | Key Numbers |
|---|------|-------------|-------------|
| 1 | **GigaThread Engine** | Top-level scheduler. Distributes thread blocks to GPCs/SMs | 1 per GPU |
| 2 | **GPC (Graphics Processing Cluster)** | Groups of TPCs. Intermediate hierarchy level | 4-16 per GPU |
| 3 | **TPC (Texture Processing Cluster)** | Groups of SMs + shared texture units | 2-6 per GPC |
| 4 | **SM (Streaming Multiprocessor)** | The main compute unit. Everything below lives inside an SM | 16-144 per GPU |

### 2. Inside the SM — Scheduling

| # | Part | What It Does | Key Numbers |
|---|------|-------------|-------------|
| 5 | **Warp Scheduler** | Picks a ready warp each cycle and issues its next instruction | 4 per SM |
| 6 | **Dispatch Unit** | Sends the decoded instruction to the correct execution unit | 1-2 per scheduler |
| 7 | **Scoreboard** | Tracks register dependencies to know when a warp is ready | Per scheduler |
| 8 | **Instruction Cache** | Caches decoded instructions to avoid refetching from L2/DRAM | Per SM, small (KB) |
| 9 | **SIMT Stack / Convergence Barrier** | Manages thread divergence — tracks which threads are active | Per warp |

### 3. Inside the SM — Register File

| # | Part | What It Does | Key Numbers |
|---|------|-------------|-------------|
| 10 | **Register File** | Per-thread private storage. Fastest memory on the chip | 65,536 x 32-bit per SM |
| 11 | **Cross-Lane Data Path (Warp Shuffle)** | Lets threads within a warp read each other's registers directly | 32 lanes, ~1 cycle |
| 12 | **Uniform Register File** (Hopper+) | Shared registers for values identical across a warp | Saves register pressure |

### 4. Inside the SM — Compute Units

| # | Part | What It Does | Key Numbers |
|---|------|-------------|-------------|
| 13 | **FP32 CUDA Cores** | Single-precision float: FADD, FMUL, FFMA | 64-128 per SM |
| 14 | **INT32 Cores** | Integer arithmetic. Can dual-issue with FP32 on Volta+ | 64 per SM |
| 15 | **FP64 Cores** | Double-precision float. Far fewer than FP32 | 1:2 ratio (data center), 1:32 (consumer) |
| 16 | **Special Function Units (SFU)** | Transcendentals: sin, cos, exp, rsqrt, rcp, log2 | 4-16 per SM, lower throughput |
| 17 | **Tensor Cores** | Matrix multiply-accumulate: D = A*B + C | 4 per SM (Volta+), supports FP16/BF16/TF32/FP8/INT8/INT4 |
| 18 | **RT Cores** | Hardware ray-BVH intersection: ray-box + ray-triangle tests | 1 per SM (Turing+), RTX only |
| 19 | **Load/Store Units (LD/ST)** | All memory operations. Contains coalescing logic | 32 per SM |

### 5. Inside the SM — Memory

| # | Part | What It Does | Key Numbers |
|---|------|-------------|-------------|
| 20 | **Shared Memory (SMEM)** | Programmer-managed scratchpad. 32 banks, 4B stride | 48-228 KB per SM, ~5 cycle latency |
| 21 | **L1 Data Cache** | Hardware-managed cache for global memory reads | Shares SRAM with SMEM (configurable split) |
| 22 | **Constant Cache** | Broadcast-optimized. Fast if all threads read same address | 64 KB constant memory, cached per SM |
| 23 | **Texture Cache / Texture Units** | 2D/3D spatial locality, hardware interpolation/filtering | Shares L1 SRAM, separate access path |
| 24 | **Local Memory** | Per-thread spill space when registers run out. NOT local — lives in DRAM | Same latency as global memory |
| 25 | **Shared Memory Atomic Unit** | Hardware atomic operations on shared memory addresses | Lower latency than global atomics |
| 26 | **Async Copy Engine** | cp.async: global → shared without register staging | Ampere+ |
| 27 | **TMA (Tensor Memory Accelerator)** | Hardware tensor tile copy engine | Hopper+ only |

### 6. Chip-Level Memory Hierarchy

| # | Part | What It Does | Key Numbers |
|---|------|-------------|-------------|
| 28 | **L2 Cache** | Shared across all SMs. Last stop before DRAM | 4-96 MB, ~200 cycle latency |
| 29 | **L2 Atomic Unit** | Hardware atomics on L2-resident data | Faster than DRAM atomics |
| 30 | **L2 Residency Control** (Ampere+) | Hint which data should stay in L2 | `cudaAccessPolicyWindow` |
| 31 | **Memory Controllers** | Interface between L2 and DRAM. Scheduling, ECC, refresh | 8-16 channels |
| 32 | **HBM / GDDR VRAM** | Main GPU memory. Big, slow relative to caches | 8-192 GB, 400 GB/s to 3+ TB/s |

### 7. Fixed-Function & Accelerator Engines

| # | Part | What It Does | Key Numbers |
|---|------|-------------|-------------|
| 33 | **Copy Engines (CE / DMA)** | Async memcpy: H2D, D2H, D2D, P2P | 2-3 per GPU, run in parallel with compute |
| 34 | **NVENC** | Hardware video encoder (H.264, H.265, AV1) | 1-3 per GPU |
| 35 | **NVDEC** | Hardware video decoder | 1-5 per GPU |
| 36 | **NVJPG** | Hardware JPEG decoder | Orin/Hopper |
| 37 | **NVOF** | Optical flow accelerator | Turing+ |
| 38 | **Raster Engine** | Rasterization for graphics pipeline | 1 per GPC |
| 39 | **ROP (Render Output Units)** | Pixel blending, depth test, antialiasing | Per memory partition |
| 40 | **Display Engine** | Drives video output (HDMI, DP) | Data center GPUs often lack this |
| 41 | **Page Migration Engine** | Handles unified memory page faults + migration | Pascal+ |

### 8. Interconnects

| # | Part | What It Does | Key Numbers |
|---|------|-------------|-------------|
| 42 | **PCIe Interface** | GPU ↔ Host CPU communication | Gen4: 32 GB/s, Gen5: 64 GB/s |
| 43 | **NVLink** | GPU ↔ GPU high-bandwidth link | 600-900 GB/s bidirectional |
| 44 | **NVSwitch** | All-to-all NVLink fabric for multi-GPU | Up to 8 GPUs fully connected |
| 45 | **C2C (Chip-to-Chip)** | Grace-Hopper CPU-GPU coherent link | 900 GB/s, cache-coherent |

### 9. Hopper+ / Blackwell Features

| # | Part | What It Does | Key Numbers |
|---|------|-------------|-------------|
| 46 | **Thread Block Clusters** | Cooperative launch across multiple SMs | SM 9.0+ |
| 47 | **Distributed Shared Memory (DSMEM)** | SM can read another SM's shared memory within a cluster | Hopper+ |
| 48 | **TMA (Tensor Memory Accelerator)** | Hardware-driven tensor tile loads/stores | Hopper+ |
| 49 | **Transformer Engine** | FP8 ↔ FP16 dynamic scaling for transformer layers | Hopper+ |
| 50 | **DPX Instructions** | Hardware dynamic programming (Smith-Waterman, Viterbi) | Hopper+ |
| 51 | **Confidential Computing** | Hardware encryption of GPU memory | Hopper+ |
| 52 | **5th Gen Tensor Cores** | FP4 support, 2x throughput over Hopper | Blackwell |
| 53 | **Multi-Instance GPU (MIG)** | Partition one GPU into isolated instances | A100+, up to 7 instances |

---

## Memory Hierarchy — Speed and Size

```
Speed                    Storage                      Size          Latency      Bandwidth
────────────────────     ───────────────────────       ──────────    ──────────   ──────────────
FASTEST ──►              Registers                     256 KB/SM     ~1 cycle     ~8 TB/s
    │                    Warp Shuffle (cross-lane)     N/A           ~1 cycle     ~4 TB/s
    │                    Shared Memory                 48-228 KB/SM  ~5 cycles    ~1.5 TB/s
    │                    L1 Cache                      (shared SRAM) ~30 cycles   ~1.5 TB/s
    │                    Constant Cache                 8 KB/SM       ~5 cycles    broadcast
    │                    Texture Cache                 (shared SRAM) ~30 cycles   filtered
    │                    L2 Cache                      4-96 MB       ~200 cycles  ~3 TB/s
    │                    Local Memory (spill)          in DRAM       ~400 cycles  (same as global)
    ▼                    Global Memory (HBM/GDDR)      8-192 GB      ~400 cycles  400 GB/s - 3 TB/s
SLOWEST ──►              Host Memory (via PCIe)        system RAM    ~10K cycles  32-64 GB/s
```

---

## GPU Hardware Explorer — Full Project Mapping

Every folder targets specific hardware from the list above. Each `.cu` file is one experiment.

```
01-global-memory/              #32 HBM/GDDR, #31 Memory Controllers, #19 LD/ST
├── coalesced.cu                — adjacent threads, adjacent addresses (best case)
├── strided.cu                  — stride 2, 4, 8, 16, 32 — watch bandwidth drop
├── random.cu                   — random access — worst case
├── read_vs_write.cu            — asymmetric bandwidth on some GPUs
├── vectorized_load.cu          — float4 loads vs float loads
└── sector_utilization.cu       — 32B sectors, measure how much you waste

02-shared-memory/              #20 SMEM, #25 SMEM Atomics (32 banks)
├── no_conflicts.cu             — 32 threads hit 32 different banks
├── 2way_conflict.cu            — stride of 2, two threads per bank
├── 32way_conflict.cu           — all threads hit same bank (broadcast case)
├── padding_fix.cu              — +1 padding eliminates conflicts
├── bank_layout.cu              — visualize which thread hits which bank
├── dynamic_vs_static.cu        — extern __shared__ vs fixed size
└── shared_atomics.cu           — atomicAdd in shared vs global

03-registers/                  #10 Register File, #24 Local Memory (spill)
├── low_register.cu             — few registers, high occupancy
├── high_register.cu            — many registers, low occupancy, faster
├── forced_spill.cu             — force compiler to spill, measure the cost
├── array_constant_idx.cu       — small array with constant index = stays in registers
├── array_dynamic_idx.cu        — dynamic index = spills to local memory
├── launch_bounds.cu            — __launch_bounds__ to control register allocation
└── inspect_spill.cu            — ptxas --verbose to count spills

04-warp-behavior/              #5 Warp Scheduler, #9 SIMT Stack, #6 Dispatch
├── no_divergence.cu            — all 32 threads take same branch
├── half_divergence.cu          — 16/16 split
├── full_divergence.cu          — every thread different branch
├── nested_divergence.cu        — divergence inside divergence
├── divergence_cost.cu          — measure actual cycle cost of divergence
├── predication_vs_branch.cu    — short vs long divergent code
└── warp_vote.cu                — __ballot_sync, __any_sync, __all_sync

05-warp-shuffle/               #11 Cross-Lane Data Path
├── broadcast.cu                — one lane broadcasts to all 31 others
├── shfl_down_reduce.cu         — warp-level sum via __shfl_down_sync
├── shfl_up_scan.cu             — warp-level prefix sum via __shfl_up_sync
├── shfl_xor_butterfly.cu       — butterfly exchange pattern (bitonic merge)
├── shfl_sync_rotate.cu         — rotate values across lanes
├── match_any.cu                — __match_any_sync for deduplication
└── shfl_vs_shared.cu           — same operation, shuffle vs shared memory, compare speed

06-L1-L2-cache/                #21 L1 Cache, #28 L2 Cache, #30 L2 Residency
├── fits_in_L1.cu               — working set < L1 size
├── fits_in_L2.cu               — working set < L2 but > L1
├── exceeds_L2.cu               — working set > L2, hits DRAM
├── cache_bypass_ldcg.cu        — __ldcg() to skip L1
├── cache_streaming_ldcs.cu     — __ldcs() streaming hint
├── l2_residency_control.cu     — cudaAccessPolicyWindow to pin data in L2
└── cache_line_size.cu          — measure 32B sector / 128B line behavior

07-atomics/                    #25 SMEM Atomic Unit, #29 L2 Atomic Unit
├── global_atomic_add.cu        — atomicAdd to global memory
├── shared_atomic_add.cu        — atomicAdd to shared memory
├── warp_reduce_then_atomic.cu  — reduce in warp first, one atomic per warp
├── atomic_cas.cu               — atomicCAS for custom operations
├── atomic_contention.cu        — all threads hit same address vs spread
├── atomic_float.cu             — atomicAdd for floats (Volta+) vs int
└── red_async.cu                — async reduction (Hopper+)

08-occupancy/                  #5 Warp Scheduler, #10 Register File (resource limits)
├── max_occupancy.cu            — minimal registers, max warps per SM
├── half_occupancy.cu           — moderate registers, fewer warps
├── low_occupancy_ilp.cu        — heavy registers, exploit ILP instead
├── shared_mem_limiter.cu       — shared memory as the occupancy bottleneck
├── block_size_sweep.cu         — 32, 64, 128, 256, 512, 1024 — profile each
├── occupancy_api.cu            — cudaOccupancyMaxActiveBlocksPerMultiprocessor
└── occupancy_vs_perf.cu        — prove that max occupancy != max performance

09-tensor-cores/               #17 Tensor Cores
├── wmma_fp16.cu                — basic WMMA matmul with FP16
├── wmma_tf32.cu                — TF32 (Ampere+) — FP32 range, less precision
├── wmma_int8.cu                — INT8 matmul for inference
├── wmma_bf16.cu                — BF16 for training
├── wmma_fp8.cu                 — FP8 (Hopper+)
├── tensor_vs_cuda_core.cu      — same matmul on tensor cores vs FP32 cores
└── mma_ptx.cu                  — raw PTX mma instruction for full control

10-async-copy/                 #26 Async Copy Engine, #27 TMA
├── sync_load.cu                — traditional global → register → shared
├── cp_async.cu                 — cp.async global → shared (skip registers)
├── cp_async_pipelined.cu       — double/triple buffer with cp.async
├── tma_1d.cu                   — TMA 1D tile load (Hopper+)
├── tma_2d.cu                   — TMA 2D tile load (Hopper+)
└── async_vs_sync.cu            — measure overlap benefit

11-multi-stream/               #33 Copy Engines, #42 PCIe
├── single_stream.cu            — everything sequential on default stream
├── multi_stream_kernels.cu     — overlapped kernel launches
├── copy_compute_overlap.cu     — H2D + kernel + D2H pipelined
├── multi_copy_engine.cu        — H2D and D2H simultaneously (2 copy engines)
├── stream_priorities.cu        — cudaStreamCreateWithPriority
├── stream_callbacks.cu         — cudaStreamAddCallback for CPU-GPU sync
└── events_timing.cu            — cudaEvent for precise timing

12-memory-bandwidth/           #32 HBM, #31 Memory Controllers (end-to-end)
├── peak_read.cu                — measure max read bandwidth
├── peak_write.cu               — measure max write bandwidth
├── peak_copy.cu                — read + write (copy kernel)
├── read_only_ldg.cu            — __ldg() through texture path
├── write_streaming.cu          — streaming stores, bypass cache
├── effective_vs_peak.cu        — how close can you get to theoretical max
└── bandwidth_vs_blocksize.cu   — bandwidth as a function of parallelism

13-constant-memory/            #22 Constant Cache
├── constant_uniform.cu         — all threads read same address (broadcast, fast)
├── constant_divergent.cu       — threads read different addresses (serialized)
├── constant_vs_global.cu       — when is __constant__ faster than global
├── constant_vs_define.cu       — __constant__ vs #define vs constexpr
└── constant_size_limit.cu      — 64KB limit, what happens when you exceed

14-texture-memory/             #23 Texture Cache / Texture Units
├── tex1d_linear.cu             — hardware linear interpolation
├── tex2d_spatial.cu            — 2D spatial locality, neighbor access
├── tex_vs_global.cu            — when texture path beats global loads
├── ldg_readonly.cu             — __ldg() uses texture cache without texture API
├── tex_normalized.cu           — normalized coordinates, free clamping/wrapping
└── surface_readwrite.cu        — surface objects for read+write through texture path

15-local-memory/               #24 Local Memory (register spill)
├── no_spill.cu                 — everything in registers, baseline
├── forced_spill_vars.cu        — too many variables, compiler spills
├── forced_spill_array.cu       — large local array with dynamic index
├── spill_detection.cu          — ptxas --verbose, count lmem usage
├── spill_cost.cu               — measure latency of spilled vs register access
└── compiler_hints.cu           — __launch_bounds__, #pragma unroll to reduce spill

16-special-function-units/     #16 SFU
├── sfu_sincos.cu               — __sinf(), __cosf() throughput
├── sfu_rsqrt.cu                — __frsqrt_rn() fast reciprocal sqrt
├── sfu_exp_log.cu              — __expf(), __log2f()
├── sfu_rcp.cu                  — __frcp_rn() reciprocal
├── full_vs_fast_math.cu        — sin() vs __sinf() — accuracy vs speed
├── sfu_throughput.cu           — SFU ops/cycle vs ALU ops/cycle
└── compiler_fast_math.cu       — --use_fast_math flag effect

17-fp64-int32-dual-issue/      #15 FP64 Cores, #14 INT32 Cores, #13 FP32 Cores
├── fp32_throughput.cu          — pure FP32 FFMA, measure peak FLOPS
├── fp64_throughput.cu          — pure FP64 DFMA, measure FP64:FP32 ratio
├── int32_throughput.cu         — pure INT32, measure peak IOPS
├── fp32_int32_dual.cu          — mix FP32+INT32 in same kernel (Volta+ dual issue)
├── fp32_fp64_mix.cu            — mixing precisions, see how scheduler handles it
└── consumer_vs_datacenter.cu   — same FP64 kernel on RTX vs A100

18-rt-cores/                   #18 RT Cores
├── bvh_build.cu                — build acceleration structure (OptiX)
├── bvh_traversal.cu            — hardware ray-BVH intersection
├── rt_vs_software.cu           — same traversal in CUDA cores vs RT cores
├── ray_triangle.cu             — ray-triangle intersection throughput
└── rt_occupancy.cu             — RT core utilization alongside CUDA cores

19-nvlink-gpu-to-gpu/          #43 NVLink, #44 NVSwitch
├── peer_enable.cu              — cudaDeviceEnablePeerAccess setup
├── peer_memcpy.cu              — cudaMemcpyPeer bandwidth
├── peer_direct_access.cu       — GPU 0 kernel reads GPU 1 memory directly
├── nvlink_vs_pcie.cu           — same transfer, NVLink vs PCIe path
├── nvlink_bidirectional.cu     — simultaneous both directions
└── multi_gpu_reduce.cu         — allreduce across GPUs

20-unified-memory/             #41 Page Migration Engine
├── managed_basic.cu            — cudaMallocManaged, let driver handle placement
├── prefetch_hint.cu            — cudaMemPrefetchAsync to target GPU
├── oversubscribe.cu            — allocate more than GPU VRAM, trigger page faults
├── access_counters.cu          — cudaMemAdvise hints (ReadMostly, PreferredLocation)
├── thrashing.cu                — CPU and GPU both touch same pages, measure cost
└── managed_vs_explicit.cu      — managed memory vs cudaMemcpy, when is each faster

21-thread-block-clusters/      #46 Clusters, #47 DSMEM
├── cluster_launch.cu           — cudaLaunchKernelEx with cluster dimension
├── dsmem_read.cu               — read another SM's shared memory in cluster
├── dsmem_write.cu              — write to another SM's shared memory
├── cluster_sync.cu             — cluster.sync() barrier
├── cluster_vs_global.cu        — DSMEM access vs going through L2/global
└── cluster_histogram.cu        — distributed histogram across cluster SMs

22-instruction-cache/          #8 Instruction Cache
├── small_kernel.cu             — kernel fits in icache, baseline
├── medium_kernel.cu            — kernel barely fits, measure transition
├── huge_kernel.cu              — massive unrolled kernel, icache thrashing
├── loop_vs_unroll.cu           — compact loop vs unrolled: icache pressure tradeoff
└── function_calls.cu           — __noinline__ vs inline, icache impact

23-transformer-engine/         #49 Transformer Engine (FP8 dynamic scaling)
├── fp8_matmul.cu               — basic FP8 GEMM via Transformer Engine API
├── fp8_vs_fp16.cu              — same matmul, FP8 vs FP16, throughput + accuracy
├── dynamic_scaling.cu          — amax tracking, scale factor computation
├── fp8_linear.cu               — FP8 linear layer forward + backward
└── delayed_scaling.cu          — delayed vs just-in-time scaling strategies

24-mig/                        #53 Multi-Instance GPU
├── mig_create.cu               — create GPU instances via NVML
├── mig_isolate.cu              — run kernels on specific instance
├── mig_bandwidth.cu            — memory bandwidth per instance
├── mig_compute.cu              — compute throughput per instance
└── mig_vs_full.cu              — same workload on MIG instance vs full GPU
```

---

## Architecture Generations Quick Reference

| Generation | Year | SM Version | Key Addition |
|-----------|------|-----------|-------------|
| Fermi | 2010 | SM 2.0 | L1/L2 cache, ECC |
| Kepler | 2012 | SM 3.x | Warp shuffle, dynamic parallelism |
| Maxwell | 2014 | SM 5.x | Unified shared/L1, improved power |
| Pascal | 2016 | SM 6.x | HBM2, NVLink, unified memory |
| Volta | 2017 | SM 7.0 | Tensor cores, independent thread scheduling |
| Turing | 2018 | SM 7.5 | RT cores, INT32/FP32 dual issue |
| Ampere | 2020 | SM 8.x | Async copy, L2 residency, TF32, sparsity |
| Hopper | 2022 | SM 9.0 | TMA, DPX, FP8, thread block clusters, DSMEM |
| Blackwell | 2024 | SM 10.0 | FP4 tensor cores, 5th gen, mega-GPU (2 dies) |