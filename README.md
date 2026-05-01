# gpu-anatomy

Hands-on experiments touching every part of an NVIDIA GPU. One folder per hardware unit, one `.cu` file per experiment.

A companion **UI walkthrough** is available to follow these experiments in order with explanations: [link TBD].

## What's in here

Folders covered so far:

- `01-global-memory/`: coalesced vs strided vs random access, vectorized loads, sector utilization
- `02-shared-memory/`: bank conflicts (2-way, 32-way), padding, swizzling, shared atomics
- `03-registers/`: register count vs occupancy, spilling to local memory
- `04-warp-behavior/`: divergence, predication, branch uniformity
- `05-warp-shuffle/`: broadcast, reduce, prefix scan
- `09-tensor-cores/`: WMMA FP16 matmul
- `10-async-copy/`: `cp.async`, double-buffered pipelining
- `18-rt-cores/`: OptiX, BVH traversal, ray-triangle (renders a real image)

## Building

```bash
make
```

Builds every `.cu` into `bin/<folder>/<name>`. OptiX experiments
(`18-rt-cores/`) only build if `OPTIX_HOME` is set: see that folder's
README for setup.

Build a single experiment:
```bash
make bin/03-registers/forced_spill
```

## Running

```bash
./bin/03-registers/forced_spill
```

Most files have an ncu command in their bottom comment block to reproduce
the measurements.

## Dependencies

- CUDA toolkit (tested with 13.x)
- NVIDIA driver new enough for your GPU
- `nvidia-smi`, `ncu`, `cuobjdump` (ship with CUDA)
- For `18-rt-cores/` only: NVIDIA OptiX SDK
