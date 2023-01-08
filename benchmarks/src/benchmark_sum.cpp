#include "benchmark.h"
#include "iostream"

auto main() -> int
{
	benchmark_cuda_sum();
	std::cout << std::endl;
	benchmark_cpu_sum();
	std::cout << std::endl;
	benchmark_omp_sum();
}
