# Benchmarking Column Sum

This folder provides several implementation of the following sum for benchmarking purposes.

- column-wise sum
- matmul

## Result Summary

Results are generated with the following setup:

- 11th Gen Intel i7-11800H
- RTX 3060 Laptop
- CUDA 11.4 release v11.4.48
- CUDA Driver 527.92
- WSL Ubuntu 20.04

### Column-Wise Sum

| Approach           | Runtime (ms) | Runtime (ms) | Runtime (ms) |
|--------------------|--------------|--------------|--------------|
| CUDA (Current Impl.)               | 0.02         | 0.02         | 0.03         |
| Sequential (Eigen) | 58.32        | 55.525       | 56.885       |
| Parallel (OMP)     | 63.1         | 61.095       | 61.675       |

It'll be interesting to see if we can get anywhere near the CUDA implementation (~10ms would be a nice target).

### Matrix Multiplication

| Approach           | Runtime (ms) | Runtime (ms) | Runtime (ms) |
|--------------------|--------------|--------------|--------------|
| CUDA (Curr. Impl.) | 2.9923       | 3.08952      | 3.06342      |
| cuBLAS             | 0.374161     | 0.38427      | 0.38554      |
| Eigen              | 118.18       | 115.722      | 118.083      |

The above is generated with random 1000 x 1000 square matrices, repeated 100 times.

## Background

This repository is an implementation of this [paper](https://www.frontiersin.org/articles/10.3389/frobt.2021.706646), which implements a sonar plugin in CUDA.

The following functions are written in CUDA:

- column_wise_sum
- matrix_multiplication
- diagonal_matmul

There are two things we can do:

- Benchmark different CPU-based implementations with the CUDA implementations to be able to port to CPU.
- Benchmark different CUDA implementations to achieve speed up

## Run Benchmark

### Column-Wise Sum

Modify the given `Makefile` such that `NVCC` points to your nvcc compiler

```bash
make
./benchmark_sum
```

The given Makefile uses NVCC at O2, which was faster than O3 on my computer (11th Gen Intel i7-11800H, RTX 3060 Laptop, CUDA Driver 527.92, WSL Ubuntu 20.04)

### Matmul

```bash
cd matmul
nvcc cublas.cu -lcublas -lcurand -O3
```

## Conclusion

- There is almost no way CPU implementation can be onpar with the current CUDA implementation, given how optimised the current implementation is
- There is room for improvement on the current CUDA implementation, although not much left. Any further improvement will require large changes to the code structure
