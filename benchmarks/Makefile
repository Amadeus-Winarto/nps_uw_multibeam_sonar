# Makefile

NVCC        := /usr/local/cuda-11/bin/nvcc
COMMON_SRCS := src/benchmark_sum.cpp  src/sum_cpu.cpp src/sum_omp.cpp
CUDA_SRCS   := src/sum_cuda.cu
INCLUDES 	:= include/ 

OUTPUT      := benchmark_sum

.DEFAULT_TARGET: all

all: $(OUTPUT)

$(OUTPUT): $(CUDA_SRCS) $(COMMON_SRCS)
	$(NVCC) -I$(INCLUDES) -std c++17 -O2 -lineinfo -dlto -arch=sm_86 -Xcompiler -fopenmp -lgomp -o $@ $^

clean:
	rm -f $(OUTPUT)