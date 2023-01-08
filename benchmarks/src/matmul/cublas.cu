#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>

#include <eigen3/Eigen/Dense>

constexpr auto M = 1000; // number of rows in matrix A
constexpr auto N = 1000; // number of columns in matrix A
constexpr auto O = 1000;  // number of columns in matrix B

constexpr auto num_repeats = 100;

template<typename T>
void print_matrix(const T A, int nr_rows_A, int nr_cols_A)
{
	for (int i = 0; i < nr_rows_A; ++i) {
		for (int j = 0; j < nr_cols_A; ++j) {
			std::cout << A[j * nr_rows_A + i] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill_rand(float* A, int nr_rows_A, int nr_cols_A)
{
	// Create a pseudo-random number generator
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
	// Set the seed for the random number generator using the system clock

	curandSetPseudoRandomGeneratorSeed(
		prng, 0 //(unsigned long long) std::chrono::system_clock::now().time_since_epoch().count()
	);
	// Fill the array with random numbers on the device
	curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

void gpu_blas_mmul(
	cublasHandle_t& handle, const float* A, const float* B, float* C, const int m, const int k, const int n
)
{
	int lda = m, ldb = k, ldc = m;
	const float alf = 1;
	const float bet = 0;
	const float* alpha = &alf;
	const float* beta = &bet;
	cublasSgemm( // C = alpha * A * B + beta * C
		handle,
		CUBLAS_OP_N, // Mode of A: One of CUBLAS_OP_N, CUBLAS_OP_T, or CUBLAS_OP_C
		CUBLAS_OP_N, // Mode of B: Normal, Tranpose, or Conjugate Tranpose (See above for enum)
		m,           // Number of rows of matrix op(A) and C
		n,           // Number of columns of matrix op(B) and C
		k,           // Number of columns of matrix op(A) and rows of matrix op(B)
		alpha,       // Scalar alpha
		A,           // Pointer to the first element of matrix A
		lda,         // Leading dimension of matrix A
		B,           // Pointer to the first element of matrix B
		ldb,         // Leading dimension of matrix B
		beta,        // Scalar beta
		C,           // Pointer to the first element of matrix C
		ldc          // Leading dimension of matrix C
	);
}

__global__ void gpu_custom_mmul(float* a, float* b, float* c, int m, int n, int k)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0;
	if (col < k && row < m) {
		for (int i = 0; i < n; i++) {
			sum += a[row * n + i] * b[i * k + col];
		}
		c[row * k + col] = sum;
	}
}

auto main() -> int
{
	// Allocate 3 arrays on CPU
	constexpr auto nr_rows_A = M;
	constexpr auto nr_cols_A = N;
	constexpr auto nr_rows_B = N;
	constexpr auto nr_cols_B = O;

	// True by definition of matrix multiplication
	constexpr auto nr_rows_C = M;
	constexpr auto nr_cols_C = O;

	auto h_A = std::vector<float>(nr_rows_A * nr_cols_A * sizeof(float));
	auto h_B = std::vector<float>(nr_rows_B * nr_cols_B * sizeof(float));
	auto h_C = std::vector<float>(nr_rows_C * nr_cols_C * sizeof(float));

	// Allocate 3 arrays on GPU
	float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
	cudaMalloc(&d_A, nr_rows_A * nr_cols_A * sizeof(float));
	cudaMalloc(&d_B, nr_rows_B * nr_cols_B * sizeof(float));
	cudaMalloc(&d_C, h_C.size() * sizeof(float));

	// Fill the arrays A and B on GPU with random numbers
	GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
	GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);
	cudaDeviceSynchronize();

	// Optionally we can copy the data back on CPU and print the arrays
	cudaMemcpy(h_A.data(), d_A, nr_rows_A * nr_cols_A * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_B.data(), d_B, nr_rows_B * nr_cols_B * sizeof(float), cudaMemcpyDeviceToHost);

	// Use custom kernel to multiply A and B on GPU
	{
		std::vector<float> durations(num_repeats);
		for (int i = 0; i < num_repeats; i++) {
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);

			cudaDeviceSynchronize();

			cudaEventRecord(start);
			gpu_custom_mmul<<<dim3(ceil(nr_cols_C / 32.0), ceil(nr_rows_C / 32.0)), dim3(32, 32)>>>(
				d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B
			);
			cudaEventRecord(stop);

			// Copy the result on host memory
			cudaMemcpy(h_C.data(), d_C, nr_rows_C * nr_cols_C * sizeof(float), cudaMemcpyDeviceToHost);

			// Display time
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			durations[i] = milliseconds;

			// Destroy the handle
			cudaEventDestroy(start);
			cudaEventDestroy(stop);
		}
		std::cout
			<< "Time (Custom): " << std::accumulate(durations.begin(), durations.end(), 0.) / (durations.size() * 1.0)
			<< " ms" << std::endl;
	}

	// Multiply A and B on GPU with CUBLAS
	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);
	{
		std::vector<float> durations(num_repeats);

		auto h_C2 = std::vector<float>(nr_rows_C * nr_cols_C * sizeof(float));
		for (int i = 0; i < num_repeats; i++) {
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);

			cudaDeviceSynchronize();

			cudaEventRecord(start);
			gpu_blas_mmul(handle, d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
			cudaEventRecord(stop);

			// Copy the result on host memory
			cudaMemcpy(h_C2.data(), d_C, nr_rows_C * nr_cols_C * sizeof(float), cudaMemcpyDeviceToHost);

			// Display Time
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			durations[i] = milliseconds;

			cudaEventDestroy(start);
			cudaEventDestroy(stop);
		}
		std::cout
			<< "Time (CUBLAS): " << std::accumulate(durations.begin(), durations.end(), 0.) / (durations.size() * 1.0)
			<< " ms" << std::endl;

		// Check if the result is correct
		for (int i = 0; i < nr_rows_C; i++) {
			for (int j = 0; j < nr_cols_C; j++) {
				if (std::abs(h_C[i * nr_cols_C + j] - h_C2[i * nr_cols_C + j]) > 1e-3) {
					std::cout << "CUBLAS result differ from custom kernel result!" << std::endl;
					return 1;
				}
			}
		}
	}
	// Destroy the handle
	cublasDestroy(handle);

	// Free GPU memory
	cudaDeviceReset();

	// Multiply A and B on CPU with Eigen
	{
		std::vector<float> durations(num_repeats);

		auto h_C2 = std::vector<float>(nr_rows_C * nr_cols_C * sizeof(float));
		for (int i = 0; i < num_repeats; i++) {
			auto start = std::chrono::high_resolution_clock::now();
			Eigen::MatrixXf A = Eigen::Map<Eigen::MatrixXf>(h_A.data(), nr_rows_A, nr_cols_A);
			Eigen::MatrixXf B = Eigen::Map<Eigen::MatrixXf>(h_B.data(), nr_rows_B, nr_cols_B);

			Eigen::MatrixXf C = A * B;
			auto stop = std::chrono::high_resolution_clock::now();
			durations.push_back(std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0);
			std::copy(C.data(), C.data() + C.size(), h_C2.data());
		}
		std::cout
			<< "Time (Eigen): " << std::accumulate(durations.begin(), durations.end(), 0.) / (durations.size() * 1.0)
			<< " ms" << std::endl;

		// Check if the result is correct
		for (int i = 0; i < nr_rows_C; i++) {
			for (int j = 0; j < nr_cols_C; j++) {
				if (std::abs(h_C[i * nr_cols_C + j] - h_C2[i * nr_cols_C + j]) > 1e-3) {
					std::cout << "CPU result differ from custom kernel result!" << std::endl;
					return 1;
				}
			}
		}
	}
	return 0;
}