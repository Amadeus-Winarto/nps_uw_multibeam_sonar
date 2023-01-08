#include <vector>
#include "benchmark.h"
#include "chrono"
#include "eigen3/Eigen/Dense"
#include "iostream"
#include "omp.h"
static constexpr auto num_cores = 8;

static auto column_sums_reduce(const Eigen::MatrixXf& in)
{
	// Approach 1: Transpose and parallelize over rows (58.655ms)
	Eigen::MatrixXf out(in.cols(), 1);
	const auto transposed = in.transpose();
#pragma omp parallel for num_threads(num_cores)
	for (int i = 0; i < transposed.rows(); ++i) {
		out(i, 0) = transposed.row(i).sum();
	}
	return out.transpose();

	// Approach 2: Parallelize over columns (79.39ms)
	// 	Eigen::MatrixXf out(1, in.cols());
	// #pragma omp parallel for num_threads(num_cores)
	// 	for (int i = 0; i < in.cols(); ++i) {
	// 		out(0, i) = in.col(i).sum();
	// 	}
	// 	return out;

	// Approach 3 : Parallelize over rows (64.77ms)
	// 	Eigen::MatrixXf out(1, in.cols());
	// 	for (int j = 0; j < in.cols(); ++j) {
	// 		double colSum {};
	// #pragma omp parallel for reduction(+ : colSum) num_threads(num_cores)
	// 		for (int i = 0; i < in.rows(); ++i) {
	// 			colSum += in(i, j);
	// 		}
	// 		out(0, j) = colSum;
	// 	}
	// 	return out;
}

auto benchmark_omp_sum(int nFreq, int nRays, int raySkips, int nRepeats) -> void
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

	// Run Kernel
	std::vector<float> durations(nRepeats);
	auto shape = std::pair<float, float>(0, 0);

	for (int i = 0; i < nRepeats; i++) {
		const auto start = std::chrono::high_resolution_clock::now();
		const auto results = column_sums_reduce(Eigen::Map<Eigen::MatrixXf>(P_Ray_real.data(), width, height));
		const auto stop = std::chrono::high_resolution_clock::now();

		durations.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
		shape.first = results.rows();
		shape.second = results.cols();
	}

	auto start = 0.;
	for (const auto d : durations) {
		start += d;
	}

	std::cout << "CPU Time (Avg.): " << start / durations.size() << "ms" << std::endl;
	std::cout << "CPU Shape: " << shape.first << "x" << shape.second << std::endl;
}