#include <iostream>
#include <vector>

constexpr auto BLOCK_SIZE = 32;

template<typename T>
__global__ void kernelSum(
	const T* __restrict__ input, T* __restrict__ per_block_results,
	const size_t lda, // pitch of input in words of sizeof(T),
	const size_t n
)
{
	extern __shared__ T sdata[];

	T x = 0.0;
	const T* p = &input[blockIdx.x * lda];
	// Accumulate per thread partial sum
	for (int i = threadIdx.x; i < n; i += blockDim.x) {
		x += p[i];
	}

	// load thread partial sum into shared memory
	sdata[threadIdx.x] = x;
	__syncthreads();

	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
		if (threadIdx.x < offset) {
			sdata[threadIdx.x] += sdata[threadIdx.x + offset];
		}
		__syncthreads();
	}

	// thread 0 writes the final result
	if (threadIdx.x == 0) {
		per_block_results[blockIdx.x] = sdata[0];
	}
}

template<typename T>
__global__ void kernelSum2(const T* __restrict__ in, T* __restrict__ out, size_t width, size_t height)
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

auto main() -> int
{
	constexpr auto num_cols = 9e5;
	constexpr auto num_rows = 114;

	auto a = std::vector<float>();
	a.reserve(num_rows * num_cols);

	for (auto i = 0; i < (num_rows * num_cols); i++) {
		a.push_back(i % 10);
	}

	float* a_;
	const auto size_a = num_rows * num_cols * sizeof(float);
	cudaMalloc(&a_, size_a);
	cudaMemcpy(a_, a.data(), size_a, cudaMemcpyHostToDevice);

	float* b_;
	size_t size_b = num_cols * sizeof(float);
	cudaMalloc(&b_, size_b);

	// select number of warps per block according to size of the
	// colum and launch one block per column. Probably makes sense
	// to have at least 4:1 column size to block size

	float milliseconds = 0;

	{
		const dim3 blocksize(32, 32);
		const dim3 gridsize(num_cols);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);
		kernelSum2<float><<<gridsize, blocksize>>>(a_, b_, num_cols, num_rows);
		if (cudaPeekAtLastError() != cudaSuccess) {
			std::cout << "Kernel Error: " << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
		}
		cudaEventRecord(stop);

		std::vector<float> b(num_cols);
		cudaMemcpy(b.data(), b_, size_b, cudaMemcpyDeviceToHost);

		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&milliseconds, start, stop);
	}
	std::cout << "Kernel 2 Time: " << milliseconds << " ms" << std::endl;

	{
		const dim3 blocksize(32 * 32);
		const dim3 gridsize(num_cols);
		size_t shmsize = sizeof(float) * (size_t) blocksize.x;

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);
		kernelSum<float><<<gridsize, blocksize, shmsize>>>(a_, b_, num_rows, num_cols);
		cudaEventRecord(stop);

		std::vector<float> b(num_cols);
		cudaMemcpy(b.data(), b_, size_b, cudaMemcpyDeviceToHost);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&milliseconds, start, stop);
	}
	std::cout << "Kernel 1 Time: " << milliseconds << " ms" << std::endl;

	cudaDeviceReset();

	return 0;
}