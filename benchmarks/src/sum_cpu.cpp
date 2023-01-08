#include "benchmark.h"
#include "chrono"
#include "eigen3/Eigen/Dense"
#include "iostream"
#include "vector"

static auto column_sums_reduce(const Eigen::MatrixXf& in)
{
	return in.colwise().sum();
}

auto benchmark_cpu_sum(int nFreq, int nRays, int raySkips, int nRepeats) -> void
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