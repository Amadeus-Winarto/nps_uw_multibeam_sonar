#pragma once

constexpr auto frequency = 9e5;
constexpr auto num_rays = 114;
constexpr auto ray_skips = 1;

constexpr auto num_repeats = 100;

auto benchmark_cuda_sum(
	int nFreq = frequency, int nRays = num_rays, int raySkips = ray_skips, int nRepeats = num_repeats
) -> void;
auto benchmark_cpu_sum(int nFreq = frequency, int nRays = num_rays, int raySkips = ray_skips, int nRepeats = num_repeats)
	-> void;
auto benchmark_omp_sum(int nFreq = frequency, int nRays = num_rays, int raySkips = ray_skips, int nRepeats = num_repeats)
	-> void;