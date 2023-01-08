#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>
#include "chrono"
#include "eigen3/Eigen/Dense"

#include "benchmark.h"
#include "iostream"

#include "cuda_runtime.h"
#define BLOCK_SIZE 32

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if (err != cudaSuccess) {
		fprintf(
			stderr, "%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n", msg, file_name, line_number,
			cudaGetErrorString(err)
		);
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

template<typename T, typename S>
constexpr void SAFE_CALL(T call, S msg)
{
	_safe_cuda_call((call), (msg), __FILE__, __LINE__);
}

template<typename T>
__global__ void column_sums_reduce(const T* __restrict__ in, T* __restrict__ out, size_t width, size_t height)
{
	__shared__ T sdata[BLOCK_SIZE][BLOCK_SIZE + 1];
	size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	size_t width_stride = gridDim.x * blockDim.x;
	size_t full_width = (width & (~((unsigned long long) (BLOCK_SIZE - 1)))) +
	                    ((width & (BLOCK_SIZE - 1)) ? BLOCK_SIZE : 0); // round up to next block
	for (size_t w = idx; w < full_width; w += width_stride) {          // grid-stride loop across matrix width
		sdata[threadIdx.y][threadIdx.x] = 0;
		size_t in_ptr = w + threadIdx.y * width;
		for (size_t h = threadIdx.y; h < height; h += BLOCK_SIZE) { // block-stride loop across matrix height
			sdata[threadIdx.y][threadIdx.x] += (w < width) ? in[in_ptr] : 0;
			in_ptr += width * BLOCK_SIZE;
		}
		__syncthreads();
		T my_val = sdata[threadIdx.x][threadIdx.y];
		for (int i = warpSize >> 1; i > 0; i >>= 1) // warp-wise parallel sum reduction
			my_val += __shfl_xor_sync(0xFFFFFFFFU, my_val, i);
		__syncthreads();
		if (threadIdx.x == 0)
			sdata[0][threadIdx.y] = my_val;
		__syncthreads();
		if ((threadIdx.y == 0) && ((w) < width))
			out[w] = sdata[0][threadIdx.x];
	}
}

auto benchmark_cuda_sum(int nFreq, int nRays, int raySkips, int nRepeats) -> void
{
	// Prepare Inputs
	const size_t width = nFreq;
	const size_t height = (int) (nRays / raySkips);

	const size_t P_Ray_N = width * height;
	const size_t P_Ray_Bytes = sizeof(float) * P_Ray_N;

	const int P_Ray_F_N = nFreq;
	const auto P_Ray_F_Bytes = sizeof(float) * P_Ray_F_N;

	auto P_Ray_real = std::vector<float>(P_Ray_N);
	std::fill(P_Ray_real.begin(), P_Ray_real.end(), 1.0f);

	// Setup
	float *d_P_Ray_real = nullptr, *d_P_Ray_F_real = nullptr;
	const dim3 dimGrid_Ray((nFreq + BLOCK_SIZE - 1) / BLOCK_SIZE);
	const dim3 dimBlock((BLOCK_SIZE, BLOCK_SIZE));

	SAFE_CALL(cudaMalloc(&d_P_Ray_real, P_Ray_Bytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMemcpy(d_P_Ray_real, P_Ray_real.data(), P_Ray_Bytes, cudaMemcpyHostToDevice), "CUDA Memcpy Failed");
	SAFE_CALL(cudaMalloc(&d_P_Ray_F_real, P_Ray_F_Bytes), "CUDA Malloc Failed");

	// Run Kernel
	std::vector<float> durations(nRepeats);
	auto shape = std::pair<float, float>(0, 0);

	for (int i = 0; i < nRepeats; i++) {
		cudaDeviceSynchronize();

		const auto start = std::chrono::high_resolution_clock::now();
		column_sums_reduce<<<dimGrid_Ray, dimBlock>>>(d_P_Ray_real, d_P_Ray_F_real, width, height);
		std::vector<float> P_Ray_F_real(P_Ray_F_N);
		SAFE_CALL(
			cudaMemcpy(P_Ray_F_real.data(), d_P_Ray_F_real, P_Ray_F_Bytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Failed"
		);
		SAFE_CALL(cudaDeviceSynchronize(), "CUDA Device Synchronize Failed");
		const auto results = Eigen::Map<Eigen::MatrixXf>(P_Ray_F_real.data(), 1, height);
		const auto stop = std::chrono::high_resolution_clock::now();

		durations.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
		shape.first = results.rows();
		shape.second = results.cols();
	}

	cudaFree(d_P_Ray_real);
	cudaFree(d_P_Ray_F_real);

	auto start = 0.;
	for (auto d : durations) {
		start += d;
	}

	std::cout << "CUDA Time (Avg.): " << start / durations.size() << "ms" << std::endl;
	std::cout << "CUDA Shape: " << shape.first << "x" << shape.second << std::endl;
}