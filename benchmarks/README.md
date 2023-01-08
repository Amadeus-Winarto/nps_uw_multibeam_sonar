# Benchmarking Column Sum

This folder provides several implementation of column-wise sum for benchmarking purposes.

## Result Summary

| Approach           | Runtime (ms) | Runtime (ms) | Runtime (ms) |
|--------------------|--------------|--------------|--------------|
| CUDA (Current Impl.)               | 0.02         | 0.02         | 0.03         |
| Sequential (Eigen) | 58.32        | 55.525       | 56.885       |
| Parallel (OMP)     | 63.1         | 61.095       | 61.675       |

It'll be interesting to see if we can get anywhere near the CUDA implementation (~10ms would be a nice target).

## Background

This repository is an implementation of this [paper](https://www.frontiersin.org/articles/10.3389/frobt.2021.706646), which implements a sonar plugin in CUDA.

The following functions are written in CUDA:

- column_wise_sum
- matrix_multiplication
- diagonal_matmul

Therefore, we need to benchmark different CPU-based implementations with the CUDA implementations in order to be able to port to CPU.

## Run Benchmark

Modify the given `Makefile` such that `NVCC` points to your nvcc compiler

```bash
make
./benchmark_sum
```

The given Makefile uses NVCC at O2, which was faster than O3 on my computer (11th Gen Intel i7-11800H, RTX 3060 Laptop, CUDA Driver 527.92, WSL Ubuntu 20.04)
